"""
Validation script for trained DQN agent.
Matches the structure and improvements from train.py.
"""

import os
import argparse
import warnings
import logging
import torch
import gymnasium
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from rl_training.agents import DQNAgent
from rl_training.utils import (convert_graph_to_data, convert_seconds_to_hours_minutes,
                               set_seed, setup_logger, setup_device)
from rl_training.memory import ReplayBuffer
from rl_training.results import ResultsManager, EpisodeResults
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions, initialize_cells_subgraph

# ----------------------------------------------------------------------------------------------------------------------
# Default paths and parameters
# ----------------------------------------------------------------------------------------------------------------------

# Device configuration
devices = ["cpu"]
if torch.cuda.is_available():
    num_cuda = torch.cuda.device_count()
    for i in range(num_cuda):
        devices.append(f"cuda:{i}")
if torch.backends.mps.is_available():
    devices.append("mps")

print(f"Devices available: {devices}")

# Validation parameters
params = {
    'seed': 42,                                     # Random seed for reproducibility
    'total_timeslots': 56,                          # Total number of time slots (1 week = 7 days * 8 timeslots)
    'gamma': 0.95,                                  # Discount factor (needed for environment)
    'maximum_number_of_bikes': 500,                 # Max number of bikes in the system
    'epsilon': 0.05,                                # Validation epsilon (near-greedy)
    'depot_position_id': 103,                       # ID (cell) of the depot position
    'initial_cell_id': 103,                         # Initial cell where the truck starts
}

reward_params = {
    'W_ZERO_BIKES': 1.0,
    'W_CRITICAL_ZONES': 1.0,
    'W_DROP_PICKUP': 0.9,
    'W_MOVEMENT': 0.7,
    'W_CHARGE_BIKE': 0.9,
    'W_STAY': 0.7,
}

# Logging formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def validate_dqn(
        env: gymnasium.Env,
        agent: DQNAgent,
        episode: int,
        device: torch.device,
        enable_logging: bool,
        tbar: tqdm = None
) -> dict:
    """
    Run a validation episode with a trained agent.

    Args:
        env: Gymnasium environment
        agent: Trained DQN agent
        episode: Episode number
        device: Torch device
        tbar: Progress bar (optional)

    Returns:
        Dictionary containing validation metrics
    """
    # Per-timeslot metrics
    rewards_per_timeslot = []
    failures_per_timeslot = []
    deployed_bikes = []

    # Per-step metrics
    action_per_step = []
    global_critic_scores = []
    reward_tracking = {idx: [] for idx in range(len(Actions))}

    # Accumulators (reset each timeslot)
    total_reward = 0.0
    total_failures = 0
    timeslots_completed = 0
    iterations = 0

    # Reset environment
    reset_options = {
        'total_timeslots': params['total_timeslots'],
        'maximum_number_of_bikes': params['maximum_number_of_bikes'],
        'discount_factor': params['gamma'],
        'logging': enable_logging,
        'depot_id': params['depot_position_id'],
        'initial_cell': params['initial_cell_id'],
        'reward_params': reward_params,
        'node_features': ['truck_cell', 'critic_score', 'eligibility_score', 'bikes'],
    }

    agent_state, info = env.reset(options=reset_options)

    # Initialize state
    node_features = ['truck_cell', 'critic_score', 'eligibility_score', 'bikes']
    state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_matrix = info['distance_matrix']

    # Initialize cell graph for spatial statistics
    custom_features = {
        'visits': 0,
        'operations': 0,
        'rebalanced': 0,
        'failures': 0,
        'failure_rates': 0.0,
        'critic_score': 0.0,
        'num_bikes': 0.0,
    }
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    # Save original epsilon and set validation epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = params['epsilon']

    done = False

    while not done:
        # Prepare state for agent (S)
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor([state.agent_state], dtype=torch.float32).unsqueeze(0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # Select action (A)
        action = agent.select_action(single_state, epsilon_greedy=True)

        # Step environment (R)
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Create next state (S')
        next_state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # Record step metrics
        action_per_step.append(action)
        reward_tracking[action].append(reward)
        global_critic_scores.append(info['global_critic_score'])
        total_reward += reward
        total_failures += sum(info['failures'])
        iterations += 1

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            center_node = cell.get_center_node()
            if center_node not in cell_graph:
                raise ValueError(f"Node {center_node} not found in cell_graph")

            cell_graph.nodes[center_node]['critic_score'] = info['cells_subgraph'].nodes[center_node]['critic_score']
            cell_graph.nodes[center_node]['num_bikes'] = info['cells_subgraph'].nodes[center_node]['bikes']

        # Handle timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1

            # Record timeslot metrics
            rewards_per_timeslot.append(total_reward / 360)
            failures_per_timeslot.append(total_failures)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset accumulators
            total_reward = 0.0
            total_failures = 0

        # Update progress bar
        if tbar is not None:
            tbar.set_description(
                f"Validation Ep. {episode}, Week {info['week']}/{52}, {info['day'].capitalize()} "
                f"at {convert_seconds_to_hours_minutes(info['time'])}"
            )
            tbar.set_postfix(eps=agent.epsilon, failures=sum(failures_per_timeslot))
            tbar.update(1)

        # Move to next state
        state = next_state
        del single_state

    # Copy final episode values and compute averages
    for cell_id, cell in cell_dict.items():
        center_node = cell.get_center_node()
        if center_node not in cell_graph:
            raise ValueError(f"Node {center_node} not found in cell_graph")

        cell_graph.nodes[center_node]['visits'] = info['cells_subgraph'].nodes[center_node]['visits']
        cell_graph.nodes[center_node]['operations'] = info['cells_subgraph'].nodes[center_node]['operations']
        cell_graph.nodes[center_node]['rebalanced'] = info['cells_subgraph'].nodes[center_node]['rebalanced']
        cell_graph.nodes[center_node]['failures'] = info['cells_subgraph'].nodes[center_node]['failures']
        cell_graph.nodes[center_node]['failure_rates'] = info['cells_subgraph'].nodes[center_node]['failure_rates']

        # Average cumulative values
        cell_graph.nodes[center_node]['critic_score'] /= iterations
        cell_graph.nodes[center_node]['num_bikes'] /= iterations

    # Cleanup
    env.close()
    torch.cuda.empty_cache()

    # Restore original epsilon
    agent.epsilon = original_epsilon

    return {
        'rewards_per_timeslot': rewards_per_timeslot,
        'failures_per_timeslot': failures_per_timeslot,
        'total_trips': info['total_trips'],
        'total_invalid': info['total_invalid'],
        'action_per_step': action_per_step,
        'reward_tracking': reward_tracking,
        'deployed_bikes': deployed_bikes,
        'global_critic_scores': global_critic_scores,
        'cell_subgraph': cell_graph,
    }


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for validation CLI."""
    parser = argparse.ArgumentParser(
        description="BSS Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate trained model
  bss-validate --model-path results/run_000/models/best_model.pt --data-path data

  # Specify validation run ID and results path
  bss-validate --model-path results/run_000/models/best_model.pt --run-id 1 --results-path validation_results

  # Use GPU device
  bss-validate --model-path results/run_000/models/best_model.pt --device cuda:0

  # Custom epsilon and timeslots
  bss-validate --model-path results/run_000/models/best_model.pt --epsilon 0.1 --total-timeslots 112
        """
    )

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model file (e.g., trained_agent.pt)')
    parser.add_argument('--run-id', type=int, default=0,
                        help='Run ID for the validation experiment.')
    parser.add_argument('--data-path', type=str, default='data/',
                        help='Path to the data folder.')
    parser.add_argument('--results-path', type=str, default='results/',
                        help='Path to the results folder.')
    parser.add_argument('--device', type=str, default='cpu',
                        help=f'Hardware device to use. Available options: {devices}.')
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help='Random seed for reproducibility.')
    parser.add_argument('--num-bikes', type=int, default=params['maximum_number_of_bikes'],
                        help='Number of bikes in the system.')
    parser.add_argument('--epsilon', type=float, default=params['epsilon'],
                        help='Epsilon value for epsilon-greedy policy (default: 0.05)')
    parser.add_argument('--total-timeslots', type=int, default=params['total_timeslots'],
                        help='Total number of timeslots for validation (default: 56 = 1 week)')
    parser.add_argument('--enable-logging', action='store_true',
                        help='Enable logging.')

    return parser


def main():
    """Main validation function."""

    warnings.filterwarnings("ignore")

    # Parse arguments
    args = create_parser().parse_args()

    # Save parsed arguments to global variables
    run_id = args.run_id
    data_path = args.data_path
    results_path = args.results_path
    enable_logging = args.enable_logging

    # Update params dict
    params['seed'] = args.seed
    params['total_timeslots'] = args.total_timeslots
    params['maximum_number_of_bikes'] = args.num_bikes
    params['epsilon'] = args.epsilon
    params['results_path'] = results_path

    # Set random seed
    set_seed(params['seed'])

    # Validate paths
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Set up device
    device = setup_device(args.device.lower(), devices)

    # Create results manager
    results_manager = ResultsManager(results_path, run_id)

    # Set up logger
    logger = setup_logger(
        'validation_logger',
        str(results_manager.validation_path) + '/validation.log',
        level=logging.INFO
    )

    logger.info(f"Validation started with the following parameters: {params}")
    logger.info(f"Reward parameters: {reward_params}")
    logger.info(f"Loading model from: {args.model_path}")

    print("=" * 80)
    print(f"Device in use: {device}")
    print(f"Validation parameters: {params}")
    print(f"Reward parameters: {reward_params}")
    print("=" * 80)

    # Create environment
    env = gym.make(
        'gymnasium_env:FullyDynamicEnv-v0',
        data_path=data_path,
        results_path=f"{str(results_manager.validation_path)}/"
    )
    env.unwrapped.seed(params['seed'])
    env.action_space.seed(params['seed'])
    env.observation_space.seed(params['seed'])

    # Initialize agent (with dummy replay buffer)
    dummy_replay_buffer = ReplayBuffer(1000)  # Small buffer, won't be used
    agent = DQNAgent(
        replay_buffer=dummy_replay_buffer,
        num_actions=env.action_space.n,
        gamma=params['gamma'],
        epsilon_start=params['epsilon'],
        epsilon_end=params['epsilon'],
        epsilon_decay=1.0,  # No decay during validation
        lr=1e-4,
        device=device,
        tau=0.005,
        soft_update=True,
    )

    # Load trained model
    print(f"Loading trained model from {args.model_path}")
    agent.load_model(args.model_path)
    print("Model loaded successfully!")

    try:
        # Run validation
        print("\nStarting validation episode...")

        tbar = tqdm(
            range(params['total_timeslots']),
            desc="Validation starting...",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        validation_dict = validate_dqn(env, agent, episode=0, device=device, enable_logging=enable_logging, tbar=tbar)

        tbar.close()

        # Convert to EpisodeResults
        validation_results = EpisodeResults(
            episode=0,
            mode='validation',
            total_reward=sum(validation_dict['rewards_per_timeslot']),
            mean_failures=sum(validation_dict['failures_per_timeslot']) / params['total_timeslots'],
            total_failures=sum(validation_dict['failures_per_timeslot']),
            total_trips=validation_dict['total_trips'],
            total_invalid=validation_dict['total_invalid'],
            epsilon=agent.epsilon,
            rewards_per_timeslot=validation_dict['rewards_per_timeslot'],
            failures_per_timeslot=validation_dict['failures_per_timeslot'],
            epsilon_per_timeslot=[],  # Not tracked in validation
            deployed_bikes=validation_dict['deployed_bikes'],
            action_per_step=validation_dict['action_per_step'],
            reward_tracking=validation_dict['reward_tracking'],
            losses=[],  # Not tracked in validation
            global_critic_scores=validation_dict['global_critic_scores'],
            q_values_per_timeslot=[],  # Not tracked in validation
            cell_subgraph=validation_dict['cell_subgraph'],
        )

        # Save results using ResultsManager
        results_manager.save_episode(validation_results)

        # Print summary
        print("=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"Total trips: {validation_results.total_trips}")
        print(f"Total failures: {validation_results.total_failures}")
        print(f"Mean failures per timeslot: {validation_results.mean_failures:.2f}")
        print(f"Invalid actions: {validation_results.total_invalid}")
        print(f"Mean reward per timeslot: {np.mean(validation_results.rewards_per_timeslot):.4f}")
        print(f"Failure rate: {(validation_results.total_failures / validation_results.total_trips * 100):.2f}%")
        print("=" * 80)

        logger.info(f"Validation completed - Mean Failures: {validation_results.mean_failures:.2f}, "
                    f"Total Failures: {validation_results.total_failures}/{validation_results.total_trips}")

        print(f"\n✓ Validation results saved to {results_manager.validation_path}")
        print("Validation completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠ Validation interrupted by user.")
        return
    except Exception as e:
        print(f"\n✗ An error occurred during validation: {e}")
        logger.error(f"Validation failed: {e}")
        raise e


if __name__ == "__main__":
    main()
