import os
import gymnasium
import torch
import argparse
import gc
import warnings
import logging

import gymnasium_env

import gymnasium as gym
import numpy as np

from tqdm import tqdm
from rl_training.agents import DQNAgent
from rl_training.utils import convert_graph_to_data, convert_seconds_to_hours_minutes, set_seed
from rl_training.memory import ReplayBuffer
from rl_training.results import ResultsManager, EpisodeResults
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions, initialize_cells_subgraph

# ----------------------------------------------------------------------------------------------------------------------

data_path = "data/"
results_path = "results/"
run_id = 0
seed = 42

# if GPU is to be used
devices = ["cpu"]

if torch.cuda.is_available():
    num_cuda = torch.cuda.device_count()
    for i in range(num_cuda):
        devices.append(f"cuda:{i}")

if torch.backends.mps.is_available():
    devices.append("mps")

print(f"Devices available: {devices}\n")

params = {
    "num_episodes": 140,                            # Total number of training episodes
    "batch_size": 64,                               # Batch size for replay buffer sampling
    "replay_buffer_capacity": int(1e5),             # Capacity of replay buffer: 0.1 million transitions
    # "input_dimentions": 72,
    "gamma": 0.95,                                  # Discount factor
    "epsilon_start": 1.0,                           # Starting exploration rate
    "epsilon_delta": 0.05,                          # Epsilon decay rate
    "epsilon_end": 0.01,                            # Minimum exploration rate
    "epsilon_decay": 1e-5,                          # Epsilon decay constant
    "exploration_time": 0.6,                        # Fraction of total training time for exploration
    "lr": 1e-4,                                     # Learning rate
    "total_timeslots": 56,                          # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 500,                 # Maximum number of bikes in the system
    "results_path": results_path,                   # Path to save results
    "soft_update": True,                            # Use soft update for target network
    "tau": 0.005,                                   # Tau parameter for soft update
    "depot_position_id": 103,                       # ID (cell) of the depot position
    "initial_cell_id": 103                          # Initial cell where the truck starts
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
one_validation = False

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# ----------------------------------------------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="BSS Train Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Run full preprocessing pipeline
            bss-train --data-path data/

            # Specify run ID and results path
            bss-train --run-id 1 --data-path data/ --results-path results/
            
            # Use GPU device
            bss-train --data-path data/ --device cuda:0
            
            # Set random seed and number of episodes
            bss-train --data-path data/ --seed 123 --num-episodes 150
            
            # Enable logging
            bss-train --data-path data/ --enable-logging 
            
            # Perform only one validation at the end of training
            bss-train --data-path data/ --one-validation  
            
            # Perform training with number of bikes and exploration time
            bss-train --data-path data/ --num-bikes 300 --exploration-time 0.8
        """,
    )

    parser.add_argument(
        '--run-id',
        type=int,
        default=run_id,
        help='Run ID for the experiment.'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=data_path,
        help='Path to the data folder.'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default=results_path,
        help='Path to the results folder.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help=f'Hardware device to use. Available options: {devices}.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=seed,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=params['num_episodes'],
        help='Number of episodes to train.'
    )
    parser.add_argument(
        '--num-bikes',
        type=int,
        default=params['maximum_number_of_bikes'],
        help='Number of bikes of the system.'
    )
    parser.add_argument(
        '--exploration-time',
        type=float,
        default=params['exploration_time'],
        help='Number of episodes to explore.'
    )
    parser.add_argument(
        '--enable-logging',
        action='store_true',
        help='Enable logging.'
    )  # TODO: fix this feature
    parser.add_argument(
        '--one-validation',
        action='store_true',
        help='Performs only one validation at the end of the training.'
    )  # TODO: fix this feature

    return parser

# ----------------------------------------------------------------------------------------------------------------------

def train_dqn(env: gymnasium.Env, agent: DQNAgent, batch_size: int, episode: int,
              device: torch.device, tbar = None) -> dict:
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    rewards_per_timeslot = []
    failures_per_timeslot = []
    epsilon_per_timeslot = []
    deployed_bikes = []
    q_values_per_timeslot = []

    # Per-step metrics
    action_per_step = []
    losses = []
    global_critic_scores = []
    reward_tracking = {idx: [] for idx in range(len(Actions))}

    # Accumulators (reset each timeslot)
    total_reward = 0.0
    total_failures = 0
    timeslots_completed = 0
    iterations = 0

    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_logging,
        'depot_id': params['depot_position_id'],
        'initial_cell': params['initial_cell_id'],
        'reward_params': reward_params,
    }

    node_features = ['truck_cell', 'critic_score', 'eligibility_score', 'bikes']
    agent_state, info = env.reset(options=reset_options)

    # Initialize state
    state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_matrix = info['distance_matrix']

    # Initialize cell graph for spatial statistics
    custom_features = {
        'visits': 0, 'operations': 0, 'rebalanced': 0,
        'failures': 0, 'failure_rates': 0.0,
        'critic_score': 0.0, 'num_bikes': 0.0,
    }
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    # ============================================================================
    # Main training loop
    # ============================================================================
    done = False
    while not done:
        # Prepare state for agent (S)
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # Select action and step environment (A)
        action = agent.select_action(single_state, epsilon_greedy=True)

        # Step into: get reward and observation (R)
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Create next state (S')
        next_state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # Store transition and train
        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step(batch_size)

        # Record step metrics
        action_per_step.append(action)
        reward_tracking[action].append(reward)
        losses.append(loss if loss is not None else 0.0)
        global_critic_scores.append(info['global_critic_score'])
        total_reward += reward
        total_failures += sum(info['failures'])
        iterations += 1

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            center_node = cell.get_center_node()
            if center_node not in cell_graph:
                raise ValueError(f"Node {center_node} not found in cell_graph")

            cell_graph.nodes[center_node]['critic_score'] += info['cells_subgraph'].nodes[center_node]['critic_score']
            cell_graph.nodes[center_node]['num_bikes'] += info['cells_subgraph'].nodes[center_node]['bikes']

        # Handle timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1

            # Update networks
            if timeslots_completed % 8 == 0:
                agent.update_target_network()
            agent.update_epsilon()

            # Record Q-values (expensive - only once per timeslot)
            with torch.no_grad():
                q_values = agent.get_q_values(single_state)
                q_values_per_timeslot.append(q_values[0].squeeze().cpu().numpy())

            # Record timeslot metrics
            rewards_per_timeslot.append(total_reward / 360)
            failures_per_timeslot.append(total_failures)
            epsilon_per_timeslot.append(agent.epsilon)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset accumulators
            total_reward = 0.0
            total_failures = 0

            # Update progress bar
            if tbar is not None:
                tbar.set_description(
                    f"Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.set_postfix({'eps': agent.epsilon})
                tbar.update(1)

        # Move to next state
        state = next_state
        del single_state

    # ============================================================================
    # Finalize episode statistics
    # ============================================================================
    for cell_id, cell in cell_dict.items():
        center_node = cell.get_center_node()
        if center_node not in cell_graph:
            raise ValueError(f"Node {center_node} not found in cell_graph")

        # Copy final episode values
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

    # ============================================================================
    # Return results
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "total_trips": info["total_trips"],
        "total_invalid": info["total_invalid"],
        "q_values_per_timeslot": q_values_per_timeslot,
        "action_per_step": action_per_step,
        "global_critic_scores": global_critic_scores,
        "losses": losses,
        "reward_tracking": reward_tracking,
        "epsilon_per_timeslot": epsilon_per_timeslot,
        "deployed_bikes": deployed_bikes,
        "cell_subgraph": cell_graph,
    }


def validate_dqn(env: gymnasium.Env, agent: DQNAgent, episode: int, device: torch.device, tbar: tqdm, enable_val_logging: bool) -> dict:
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    rewards_per_timeslot = []
    failures_per_timeslot = []
    deployed_bikes = []

    # Per-step metrics
    action_per_step = []
    reward_tracking = {idx: [] for idx in range(len(Actions))}

    # Accumulators (reset each timeslot)
    total_reward = 0.0
    total_failures = 0
    timeslots_completed = 0
    iterations = 0

    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_val_logging,
        'depot_id': params['depot_position_id'],
        'initial_cell': params['initial_cell_id'],
        'reward_params': reward_params,
    }

    node_features = ['truck_cell', 'critic_score', 'eligibility_score', 'bikes']
    agent_state, info = env.reset(options=reset_options)

    # Initialize state
    state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_matrix = info['distance_matrix']

    # Initialize cell graph for spatial statistics
    custom_features = {
        'visits': 0, 'operations': 0, 'rebalanced': 0,
        'failures': 0, 'failure_rates': 0.0,
        'critic_score': 0.0, 'num_bikes': 0.0,
    }
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    # ============================================================================
    # Temporarily set epsilon for validation (mostly greedy policy)
    # ============================================================================
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.05  # Low exploration for validation

    # ============================================================================
    # Main validation loop
    # ============================================================================
    done = False
    while not done:
        # Prepare state for agent
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # Select action (epsilon=0.05, mostly greedy)
        action = agent.select_action(single_state, epsilon_greedy=True)
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Create next state
        next_state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # Record step metrics (no training in validation)
        action_per_step.append(action)
        reward_tracking[action].append(reward)
        total_reward += reward
        total_failures += sum(info['failures'])
        iterations += 1

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            center_node = cell.get_center_node()
            if center_node not in cell_graph:
                raise ValueError(f"Node {center_node} not found in cell_graph")

            cell_graph.nodes[center_node]['critic_score'] += info['cells_subgraph'].nodes[center_node]['critic_score']
            cell_graph.nodes[center_node]['num_bikes'] += info['cells_subgraph'].nodes[center_node]['bikes']

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
            tbar.set_description(
                f"Validating Episode {episode}, Week {info['week'] % 52}, "
                f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
            )
            tbar.set_postfix({'eps': agent.epsilon})

        # Move to next state
        state = next_state
        del single_state  # Free GPU memory

    # ============================================================================
    # Finalize episode statistics
    # ============================================================================
    for cell_id, cell in cell_dict.items():
        center_node = cell.get_center_node()
        if center_node not in cell_graph:
            raise ValueError(f"Node {center_node} not found in cell_graph")

        # Copy final episode values
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

    # Restore original epsilon
    agent.epsilon = previous_epsilon

    # ============================================================================
    # Return results (no losses, q_values, epsilon tracking in validation)
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "total_trips": info["total_trips"],
        "total_invalid": info["total_invalid"],
        "action_per_step": action_per_step,
        "reward_tracking": reward_tracking,
        "deployed_bikes": deployed_bikes,
        "cell_subgraph": cell_graph,
        # Validation doesn't track these (return empty for consistency)
        "q_values_per_timeslot": [],
        "epsilon_per_timeslot": [],
        "losses": [],
        "global_critic_scores": [],
    }


# ----------------------------------------------------------------------------------------------------------------------

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def setup_device(device_str: str) -> torch.device:
    if device_str not in devices:
        raise ValueError(
            f"Invalid device '{device_str}'. Available options: {devices}"
        )

    device = torch.device(device_str)
    if device.type == "cuda":
        gpu_id = device.index
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"Using CUDA device {gpu_id}: {gpu_name}")
    else:
        print(f"Using device: {device.type}")

    return device

# ----------------------------------------------------------------------------------------------------------------------

def main():
    global run_id, data_path, results_path, enable_logging, seed, one_validation

    warnings.filterwarnings("ignore")
    args = create_parser().parse_args()

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    device = setup_device(args.device.lower())

    # Save parsed arguments to global variables
    run_id = args.run_id
    data_path = args.data_path
    results_path = args.results_path
    seed = args.seed
    enable_logging = args.enable_logging
    one_validation = args.one_validation

    # Update params dict with parsed values
    params['num_episodes'] = args.num_episodes
    params['maximum_number_of_bikes'] = args.num_bikes
    params['exploration_time'] = args.exploration_time
    params['results_path'] = results_path

    set_seed(seed)

    # At 60% of the total timeslots (60% of the training) the epsilon should be 0.1
    params["epsilon_decay"] = ((params["exploration_time"] * params["num_episodes"] * params["total_timeslots"])**2) / np.log(10)
    print(f"\nParams in use: {params}\n")
    print(f"Reward params in use: {reward_params}\n")
    if one_validation:
        print("one_validation = True")

    results_manager = ResultsManager(results_path, run_id)
    results_manager.save_hyperparameters(params, reward_params)

    # Create the environment
    env = gym.make('gymnasium_env/FullyDynamicEnv-v0', data_path=data_path, results_path=str(results_manager.training_path))
    env.unwrapped.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])

    # Initialize the DQN agent
    agent = DQNAgent(
        replay_buffer=replay_buffer,
        num_actions=env.action_space.n,
        gamma=params["gamma"],
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay=params["epsilon_decay"],
        lr=params["lr"],
        device=device,
        tau=params["tau"],
        soft_update=params["soft_update"],
    )

    # Train the agent using the training loop
    starting_episode = 0
    last_validation_score = None
    try:
        tbar = tqdm(
            range(params["total_timeslots"]*params["num_episodes"]),
            desc="Training computation is starting ... ... ... ...",
            initial=starting_episode*params["total_timeslots"],
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        logger = setup_logger('training_logger', str(results_manager.training_path) + 'training.log', level=logging.INFO)

        logger.info(f"Training started with the following parameters: {params}")

        # Train and validation loop
        best_validation_score = 1e4
        for episode in range(starting_episode, params["num_episodes"]):

            training_dict = train_dqn(env, agent, params["batch_size"], episode, device, tbar)

            # Convert to EpisodeResults
            training_results = EpisodeResults(
                episode=episode,
                mode='train',
                total_reward=sum(training_dict['rewards_per_timeslot']),
                mean_failures=sum(training_dict['failures_per_timeslot']) / params["total_timeslots"],
                total_failures=sum(training_dict['failures_per_timeslot']),
                total_trips=training_dict['total_trips'],
                total_invalid=training_dict['total_invalid'],
                epsilon=agent.epsilon,
                rewards_per_timeslot=training_dict['rewards_per_timeslot'],
                failures_per_timeslot=training_dict['failures_per_timeslot'],
                epsilon_per_timeslot=training_dict['epsilon_per_timeslot'],
                deployed_bikes=training_dict['deployed_bikes'],
                action_per_step=training_dict['action_per_step'],
                reward_tracking=training_dict['reward_tracking'],
                losses=training_dict['losses'],
                global_critic_scores=training_dict['global_critic_scores'],
                q_values_per_timeslot=training_dict['q_values_per_timeslot'],
                cell_subgraph=training_dict['cell_subgraph'],
            )

            # Save using ResultsManager
            results_manager.save_episode(training_results)

            logger.info(
                f"Episode {episode}: Mean Failures = {training_results.mean_failures:.2f}, "
                f"Total Failures = {training_results.total_failures}/{training_results.total_trips}, "
                f"Invalid Actions = {training_results.total_invalid}"
            )

            # Save model if the training and validation score is better
            should_validate = (episode%10 == 0 and (agent.epsilon < 0.15 or training_results.mean_failures < 10)
                               and not one_validation) or episode == (params["num_episodes"]-1)
            if should_validate:
                enable_val_logging = enable_logging 
                if episode > params["num_episodes"]*0.9: 
                    enable_val_logging = True
                # validate the training with a greedy validation, VALIDATE_DQN             
                validation_dict = validate_dqn(env, agent, episode, device, tbar, enable_val_logging)

                validation_results = EpisodeResults(
                    episode=episode,
                    mode='validation',
                    total_reward=sum(validation_dict['rewards_per_timeslot']),
                    mean_failures=sum(validation_dict['failures_per_timeslot']) / params["total_timeslots"],
                    total_failures=sum(validation_dict['failures_per_timeslot']),
                    total_trips=validation_dict['total_trips'],
                    total_invalid=validation_dict['total_invalid'],
                    epsilon=agent.epsilon,
                    rewards_per_timeslot=validation_dict['rewards_per_timeslot'],
                    failures_per_timeslot=validation_dict['failures_per_timeslot'],
                    epsilon_per_timeslot=validation_dict.get('epsilon_per_timeslot', []),
                    deployed_bikes=validation_dict['deployed_bikes'],
                    action_per_step=validation_dict['action_per_step'],
                    reward_tracking=validation_dict['reward_tracking'],
                    losses=validation_dict.get('losses', []),
                    global_critic_scores=validation_dict.get('global_critic_scores', []),
                    q_values_per_timeslot=validation_dict.get('q_values_per_timeslot', []),
                    cell_subgraph=validation_dict['cell_subgraph'],
                )

                # Save using ResultsManager
                results_manager.save_episode(validation_results)

                last_validation_score = validation_results.total_failures

                logger.info(
                    f"Episode {episode}: Mean Validation Failures = {validation_results.mean_failures:.2f}, "
                    f"Total Validation Failures = {validation_results.total_failures}/{validation_results.total_trips},"
                    f" Invalid Actions = {validation_results.total_invalid}, "
                    f"Best Validation Failures = {best_validation_score}"
                )

                if validation_results.total_failures < best_validation_score or episode > 125:
                    if validation_results.total_failures < best_validation_score:
                        best_validation_score = validation_results.total_failures
                        is_best = True
                    else:
                        is_best = False

                    # Save the trained model using ResultsManager
                    model_path = results_manager.save_model(
                        agent=agent,
                        episode=episode,
                        score=validation_results.total_failures,
                        model_type='best' if is_best else 'checkpoint',
                        save_best=is_best
                    )

                    logger.info(f"Model saved: {model_path}")

            gc.collect()

        final_score = last_validation_score if last_validation_score is not None else float('inf')
        results_manager.save_model(
            agent=agent,
            episode=params["num_episodes"] - 1,
            score=final_score,
            model_type='final'
        )
        logger.info(f"Final model saved with score: {final_score}")

        # Save aggregated summaries
        results_manager.save_run_summary()
        logger.info("Training completed successfully")
        tbar.close()
    except Exception as e:
        raise e
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return

    # Print the rewards after training
    print(f"\nTraining {run_id} completed.")
