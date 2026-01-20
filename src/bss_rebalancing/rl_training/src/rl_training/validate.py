import os
import pickle
import argparse
import warnings
import torch
import gymnasium_env
import gymnasium as gym
import numpy as np

from tqdm import tqdm
from rl_training.agents import DQNAgent
from rl_training.utils import convert_graph_to_data, convert_seconds_to_hours_minutes, Actions
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import initialize_cells_subgraph

# ----------------------------------------------------------------------------------------------------------------------

# Default paths
data_path = "../data/"
results_path = "results/"
model_path = None  # Will be set via command line

# Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device available: {device.type}")

# Seed for reproducibility
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Validation parameters
params = {
    "total_timeslots": 56,  # Total number of time slots (1 week = 7 days * 8 timeslots)
    "maximum_number_of_bikes": 375,  # Maximum number of bikes in the system
    "gamma": 0.95,  # Discount factor (needed for environment)
}

reward_params = {
    'W_ZERO_BIKES': 1.0,
    'W_CRITICAL_ZONES': 1.0,
    'W_DROP_PICKUP': 0.9,
    'W_MOVEMENT': 0.7,
    'W_CHARGE_BIKE': 0.9,
    'W_STAY': 0.7,
}

enable_logging = False
args = None  # Will be set by argparse


# ----------------------------------------------------------------------------------------------------------------------

def validate_agent(env: gym, agent: DQNAgent, validation_results_path: str, epsilon: float = 0.05) -> dict:
    """
    Run a validation episode with a trained agent.

    Args:
        env: Gymnasium environment
        agent: Trained DQN agent
        validation_results_path: Path to save validation results
        epsilon: Epsilon value for epsilon-greedy policy (default: 0.05 for near-greedy)

    Returns:
        Dictionary containing validation metrics
    """
    # Initialize episode metrics
    timeslot = 0
    timeslots_completed = 0
    rewards_per_timeslot = []
    total_reward = 0
    failures_per_timeslot = []
    total_failures = 0
    action_per_step = []
    global_criticals = []
    reward_tracking = [[] for _ in range(len(Actions))]
    deployed_bikes = []

    # Reset environment
    options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_logging,
        'depot_id': 18,
        'initial_cell': 18,
        'reward_params': reward_params,
    }

    node_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'bikes',
    ]

    agent_state, info = env.reset(options=options)

    state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
    state.agent_state = agent_state
    state.steps = info['steps']
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_matrix = info['distance_matrix']

    custom_features = {
        'visits': 0,
        'operations': 0,
        'rebalanced': 0,
        'failures': 0,
        'critic_score': 0.0,
        'num_bikes': 0.0,
    }
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    not_done = True
    iterations = 0

    # Save original epsilon and set validation epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = epsilon

    # Progress bar
    tbar = tqdm(
        total=params["total_timeslots"],
        desc="Validation starting",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    while not_done:
        # Prepare the state for the agent
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # Select action using epsilon-greedy policy
        action = agent.select_action(single_state, epsilon_greedy=True)

        # Step the environment
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Update state
        env_cells_subgraph = info['cells_subgraph']
        next_state = convert_graph_to_data(env_cells_subgraph, node_features=node_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        state = next_state

        # Update metrics
        action_per_step.append(action)
        global_criticals.append(info['global_critic_score'])
        reward_tracking[action].append(reward)

        total_reward += reward
        total_failures += sum(info['failures'])

        # Check if episode is complete
        not_done = not done

        # Update cell graph
        for cell_id, cell in cell_dict.items():
            center_node = cell.get_center_node()
            if center_node in cell_graph:
                cell_graph.nodes[center_node]['critic_score'] += env_cells_subgraph.nodes[center_node]['critic_score']
                cell_graph.nodes[center_node]['num_bikes'] += env_cells_subgraph.nodes[center_node]['bikes']
            else:
                raise ValueError(f"Node {center_node} not found in the subgraph.")

        if timeslot_terminated:
            timeslots_completed += 1

            # Record metrics for the current time slot
            rewards_per_timeslot.append(total_reward / 360)
            failures_per_timeslot.append(total_failures)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset time slot metrics
            total_reward = 0
            total_failures = 0
            timeslot = 0 if timeslot == 7 else timeslot + 1

            # Update progress bar
            tbar.set_description(f"Validating: Week {info['week'] % 52}, {info['day'].capitalize()} "
                                 f"at {convert_seconds_to_hours_minutes(info['time'])}")
            tbar.set_postfix({'eps': agent.epsilon, 'failures': sum(failures_per_timeslot)})
            tbar.update(1)

        iterations += 1

        if done:
            # Finalize cell graph statistics
            for cell_id, cell in cell_dict.items():
                center_node = cell.get_center_node()
                if center_node in cell_graph:
                    cell_graph.nodes[center_node]['visits'] = env_cells_subgraph.nodes[center_node]['visits']
                    cell_graph.nodes[center_node]['operations'] = env_cells_subgraph.nodes[center_node]['operations']
                    cell_graph.nodes[center_node]['rebalanced'] = env_cells_subgraph.nodes[center_node]['rebalanced']
                    cell_graph.nodes[center_node]['failures'] = env_cells_subgraph.nodes[center_node]['failures']
                    cell_graph.nodes[center_node]['critic_score'] = cell_graph.nodes[center_node][
                                                                        'critic_score'] / iterations
                    cell_graph.nodes[center_node]['num_bikes'] = cell_graph.nodes[center_node]['num_bikes'] / iterations
                else:
                    raise ValueError(f"Node {center_node} not found in the subgraph.")

        # Clean up
        del single_state

    tbar.close()
    env.close()
    torch.cuda.empty_cache()

    # Restore original epsilon
    agent.epsilon = original_epsilon

    # Compile results
    results = {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "total_trips": info["total_trips"],
        "total_invalid": info["total_invalid"],
        "action_per_step": action_per_step,
        "global_criticals": global_criticals,
        "reward_tracking": reward_tracking,
        "deployed_bikes": deployed_bikes,
        "cell_subgraph": cell_graph,
    }

    return results


# ----------------------------------------------------------------------------------------------------------------------

def main():
    global model_path, data_path, results_path, enable_logging, device

    warnings.filterwarnings("ignore")
    print(f"Device in use: {device}\n")
    print(f"Validation parameters: {params}\n")
    print(f"Reward parameters: {reward_params}\n")

    # Create validation results path
    validation_results_path = os.path.join(results_path, 'validation_standalone/')
    if not os.path.exists(validation_results_path):
        os.makedirs(validation_results_path)
        print(f"Created validation results directory: {validation_results_path}")

    # Create the environment
    env = gym.make('gymnasium_env/FullyDynamicEnv-v0', data_path=data_path, results_path=validation_results_path)
    env.unwrapped.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Initialize the DQN agent (with dummy replay buffer)
    from rl_training.memory import ReplayBuffer
    dummy_replay_buffer = ReplayBuffer(1000)  # Small buffer, won't be used

    agent = DQNAgent(
        replay_buffer=dummy_replay_buffer,
        num_actions=env.action_space.n,
        gamma=params["gamma"],
        epsilon_start=0.05,
        epsilon_end=0.05,
        epsilon_decay=1.0,
        lr=1e-4,
        device=device,
        tau=0.005,
        soft_update=True,
    )

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading trained model from: {model_path}")
    agent.load_model(model_path)
    print("Model loaded successfully!\n")

    # Run validation
    try:
        print("Starting validation episode...")
        validation_results = validate_agent(env, agent, validation_results_path, epsilon=args.epsilon)

        # Calculate statistics
        total_trips = validation_results['total_trips']
        total_invalid = validation_results['total_invalid']
        total_failures = sum(validation_results['failures_per_timeslot'])
        mean_failures = total_failures / params["total_timeslots"]
        mean_reward = np.mean(validation_results['rewards_per_timeslot'])

        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"Total trips: {total_trips}")
        print(f"Total failures: {total_failures}")
        print(f"Mean failures per timeslot: {mean_failures:.2f}")
        print(f"Invalid actions: {total_invalid}")
        print(f"Mean reward per timeslot: {mean_reward:.4f}")
        print(f"Failure rate: {(total_failures / total_trips) * 100:.2f}%")
        print("=" * 80)

        # Save validation results
        results_data_path = os.path.join(validation_results_path, 'data/')
        if not os.path.exists(results_data_path):
            os.makedirs(results_data_path)

        for key, value in validation_results.items():
            with open(os.path.join(results_data_path, f'{key}.pkl'), 'wb') as f:
                pickle.dump(value, f)

        print(f"\nValidation results saved to: {results_data_path}")

        # Save summary statistics
        summary = {
            'total_trips': total_trips,
            'total_failures': total_failures,
            'mean_failures': mean_failures,
            'total_invalid': total_invalid,
            'mean_reward': mean_reward,
            'failure_rate': (total_failures / total_trips) * 100,
            'failures_per_timeslot': validation_results['failures_per_timeslot'],
            'rewards_per_timeslot': validation_results['rewards_per_timeslot'],
        }

        with open(os.path.join(validation_results_path, 'summary.pkl'), 'wb') as f:
            pickle.dump(summary, f)

        print("Validation completed successfully!")

    except Exception as e:
        print(f"An error occurred during validation: {e}")
        raise e
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a trained Dueling DQN agent.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (trained_agent.pt)')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='Path to the data folder.')
    parser.add_argument('--results_path', type=str, default=results_path,
                        help='Path to save validation results.')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device to use.')
    parser.add_argument('--enable_logging', action='store_true',
                        help='Enable logging.')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Epsilon value for epsilon-greedy policy (default: 0.05)')
    parser.add_argument('--total_timeslots', type=int, default=params['total_timeslots'],
                        help='Total number of timeslots for validation (default: 56 = 1 week)')

    args = parser.parse_args()

    # Assign variables
    model_path = args.model_path
    data_path = args.data_path
    results_path = args.results_path
    enable_logging = args.enable_logging
    params["total_timeslots"] = args.total_timeslots

    # Validate paths
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Set up device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main()
