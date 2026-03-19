import os
import gymnasium
import torch
import argparse
import gc
import warnings
import logging
import gymnasium_env

import multiprocessing as mp
import gymnasium as gym
import numpy as np

from tqdm import tqdm
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions

from rl_training.agents import DQNAgent
from rl_training.memory import ReplayBuffer
from rl_training.results import ResultsManager, EpisodeResults
from rl_training.logging_config import init_logging, LoggingConfig, get_logger
from rl_training.utils import (
    convert_graph_to_data,
    convert_seconds_to_hours_minutes,
    set_seed,
    setup_device,
    build_cell_graph_from_cells,
    update_cell_graph_features
)

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

print(f"Devices available: {devices}\n")

params = {
    "seed": 42,                                     # Random seed for reproducibility
    "num_episodes": 140,                            # Total number of training episodes
    "batch_size": 64,                               # Batch size for replay buffer sampling
    "replay_buffer_capacity": int(1e5),             # Capacity of replay buffer: 0.1 million transitions
    "gamma": 0.95,                                  # Discount factor
    "epsilon_start": 1.0,                           # Starting exploration rate
    "epsilon_delta": 0.05,                          # Epsilon decay rate
    "epsilon_end": 0.01,                            # Minimum exploration rate
    "epsilon_decay": 1e-5,                          # Epsilon decay constant
    "exploration_time": 0.6,                        # Fraction of total training time for exploration
    "lr": 1e-4,                                     # Learning rate
    "soft_update": True,                            # Use soft update for target network
    "tau": 0.005,                                   # Tau parameter for soft update

    "total_timeslots": 56,                          # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 500,                 # Maximum number of bikes in the system
    "minimum_number_of_bikes": 1,                   # Minimum number of bikes per cell
    "enable_repositioning": False,                  # Use base repositioning strategy at the start of each episode
    "use_net_flow": False,                          # Use net flow repositioning strategy at the start of each episode
    "depot_position_id": 18,                        # ID (cell) of the depot position
    "initial_cell_id": 18                           # Initial cell where the truck starts
}

reward_params = {
    'W_ZERO_BIKES': 1.0,
    'W_CRITICAL_ZONES': 1.0,
    'W_DROP_PICKUP': 0.9,
    'W_MOVEMENT': 0.7,
    'W_CHARGE_BIKE': 0.9,
    'W_STAY': 0.7,
}

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
        default=0,
        help='Run ID for the experiment.'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/',
        help='Path to the data folder.'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default='results/',
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
        default=params["seed"],
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=params['num_episodes'],
        help='Number of episodes to train.'
    )
    parser.add_argument(
        '--max-num-bikes',
        type=int,
        default=params['maximum_number_of_bikes'],
        help='Number of bikes of the system.'
    )
    parser.add_argument(
        '--min-num-bikes',
        type=int,
        default=params['minimum_number_of_bikes'],
        help='Minimum number of bikes per cell.'
    )
    parser.add_argument(
        '--enable-repositioning',
        action='store_true',
        help='Enable repositioning beyond minimum bikes per cell at the start of each episode.'
    )
    parser.add_argument(
        '--use-net-flow',
        action='store_true',
        help='Use net-flow-based repositioning instead of random at the start of each episode.'
    )
    parser.add_argument(
        '--exploration-time',
        type=float,
        default=params['exploration_time'],
        help='Number of episodes to explore.'
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable logging.'
    )
    parser.add_argument(
        '--one-validation',
        action='store_true',
        help='Performs only one validation at the end of the training.'
    )  # TODO: fix this feature

    return parser

# ----------------------------------------------------------------------------------------------------------------------

def train_dqn(
    env: gymnasium.Env,
    agent: DQNAgent,
    batch_size: int,
    episode: int,
    device: torch.device,
    run_id: int,
    logging_enabled: bool,
    tbar=None,
    episode_results_path: str | None = None,
) -> dict:
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    rewards = []
    failures = []
    epsilons = []
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    traveling_bikes = []
    q_values = []

    # Per-step metrics
    action_per_step = []
    global_critic_scores = []
    reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}

    # Accumulators (reset each timeslot)
    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    iterations = 0

    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': params["minimum_number_of_bikes"],
        'enable_repositioning': params["enable_repositioning"],
        'use_net_flow': params["use_net_flow"],
        'discount_factor': params["gamma"],
        'depot_id': params['depot_position_id'],
        # 'initial_cell': params['initial_cell_id'],
        'reward_params': reward_params,
    }
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path

    agent_state, info = env.reset(options=reset_options)

    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_lookup = info['distance_lookup']

    # Build initial graph from cells (reads metrics automatically)
    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict,
        nodes_dict=nodes_dict,
        distance_lookup=distance_lookup
    )

    # Define which metrics to use as GNN features: THEY SHOULD MATCH THE FEATURES USED IN THE CELLS
    gnn_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'total_bikes',
    ]

    # Initialize state
    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # ============================================================================
    # Main training loop
    # ============================================================================
    episode_cell_stats = {
        cell_id: {
            'critic_sum': 0.0,
            'eligibility_sum': 0.0,
            'bikes_sum': 0.0,
        }
        for cell_id in cell_dict.keys()  # from initial info after reset
    }

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

        # Only update node attributes
        cell_dict = info['cell_dict'] # <- VERY BIG BUG, IT WAS PREVIOUSLY MISSING
        update_cell_graph_features(cell_graph, cell_dict)

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['eligibility_sum'] += cell.get_eligibility_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] = cell.get_dead_bikes()

        # Create next state (S')
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # Store transition and train
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train_step(batch_size)

        # Record step metrics
        action_per_step.append(action)
        reward_tracking_per_action[action].append(reward)
        global_critic_scores.append(info['global_critic_score'])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info['failures'])
        iterations += 1

        # Handle timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1

            # Update networks
            if timeslots_completed % 8 == 0:
                agent.update_target_network()
            agent.update_epsilon()

            # Record Q-values (expensive - only once per timeslot)
            with torch.no_grad():
                q_val_tensor = agent.get_q_values(single_state)
                q_values.append(q_val_tensor[0].squeeze().cpu().numpy())

            # Record timeslot metrics
            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            epsilons.append(agent.epsilon)
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])

            # Reset accumulators
            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0

            # Update progress bar
            if tbar is not None:
                tbar.set_description(
                    f"[TRAIN] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.set_postfix({'eps': agent.epsilon})
                tbar.update(1)

        # Move to next state
        state = next_state
        del single_state

    # Cleanup
    torch.cuda.empty_cache()

    steps_in_episode = iterations
    for cell_id, stats in episode_cell_stats.items():
        center_node = cell_dict[cell_id].get_center_node()
        if center_node not in cell_graph.nodes:
            continue

        if steps_in_episode > 0:
            critic_mean = stats.get('critic_sum', 0.0) / steps_in_episode
            eligibility_mean = stats.get('eligibility_sum', 0.0) / steps_in_episode
            bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
            dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
        else:
            critic_mean = eligibility_mean = bikes_mean = dead_bikes_mean = 0.0

        nx_attrs = cell_graph.nodes[center_node]

        nx_attrs['critic_mean'] = critic_mean
        nx_attrs['eligibility_mean'] = eligibility_mean
        nx_attrs['failure_sum'] = cell_dict[cell_id].get_failures()
        nx_attrs['failure_rate'] = cell_dict[cell_id].get_failure_rate()
        nx_attrs['visits_sum'] = cell_dict[cell_id].get_visits()
        nx_attrs['ops_sum'] = cell_dict[cell_id].get_ops()
        nx_attrs['bikes_mean'] = bikes_mean
        nx_attrs['dead_bikes_mean'] = dead_bikes_mean

    # ============================================================================
    # Return results
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards,
        "failures_per_timeslot": failures,
        "total_invalid_actions": info["total_invalid_actions"],
        "q_values_per_timeslot": q_values,
        "action_per_step": action_per_step,
        "global_critic_scores": global_critic_scores,
        "reward_tracking_per_action": reward_tracking_per_action,
        "epsilon_per_timeslot": epsilons,
        "deployed_bikes": system_bikes,
        "truck_load": truck_load,
        "depot_load": depot_load,
        "outside_system_bikes": outside_system_bikes,
        'traveling_bikes': traveling_bikes,
        "cell_subgraph": cell_graph,
    }

# ----------------------------------------------------------------------------------------------------------------------

def validate_dqn(
        env: gymnasium.Env,
        agent: DQNAgent,
        episode: int,
        device: torch.device,
        run_id: int,
        logging_enabled: bool,
        tbar=None,
        episode_results_path: str | None = None,
        params_snapshot: dict | None = None,
        reward_params_snapshot: dict | None = None,
) -> dict:
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    rewards = []
    failures = []
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    traveling_bikes = []

    # Per-step metrics
    action_per_step = []
    global_critic_scores = []
    reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}

    # Accumulators (reset each timeslot)
    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    iterations = 0

    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    _params = params_snapshot if params_snapshot is not None else params
    _reward_params = reward_params_snapshot if reward_params_snapshot is not None else reward_params

    reset_options = {
        'total_timeslots': _params["total_timeslots"],
        'maximum_number_of_bikes': _params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': _params["minimum_number_of_bikes"],
        'enable_repositioning': _params["enable_repositioning"],
        'use_net_flow': _params["use_net_flow"],
        'discount_factor': _params["gamma"],
        'depot_id': _params['depot_position_id'],
        'reward_params': _reward_params,
    }
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path

    agent_state, info = env.reset(options=reset_options)

    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_lookup = info['distance_lookup']

    # Build initial graph from cells (reads metrics automatically)
    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict,
        nodes_dict=nodes_dict,
        distance_lookup=distance_lookup
    )

    # Define which metrics to use as GNN features
    gnn_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'total_bikes',
    ]

    # Initialize state
    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # ============================================================================
    # Temporarily set epsilon for validation (mostly greedy policy)
    # ============================================================================
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.05  # Low exploration for validation

    # ============================================================================
    # Main validation loop
    # ============================================================================
    episode_cell_stats = {
        cell_id: {"critic_sum": 0.0, "eligibility_sum": 0.0, "bikes_sum": 0.0}
        for cell_id in cell_dict.keys()
    }

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

        # Select action and step environment (A) (epsilon=0.05, mostly greedy)
        action = agent.select_action(single_state, epsilon_greedy=True)

        # Step into: get reward and observation (R)
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Only update node attributes
        cell_dict = info['cell_dict']
        update_cell_graph_features(cell_graph, cell_dict)

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['eligibility_sum'] += cell.get_eligibility_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] = cell.get_dead_bikes()

        # Create next state (S')
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # Record step metrics (no training in validation)
        action_per_step.append(action)
        reward_tracking_per_action[action].append(reward)
        global_critic_scores.append(info['global_critic_score'])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info['failures'])
        iterations += 1

        # Handle timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1

            # Record timeslot metrics
            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])

            # Reset accumulators
            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0

            # Update progress bar
            if tbar is not None:
                tbar.set_description(
                    f"[VALIDATION] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.set_postfix({'eps': agent.epsilon})
                tbar.update(1)

        # Move to next state
        state = next_state
        del single_state  # Free GPU memory

    # Cleanup
    torch.cuda.empty_cache()

    # Restore original epsilon
    agent.epsilon = previous_epsilon

    steps_in_episode = iterations
    for cell_id, stats in episode_cell_stats.items():
        center_node = cell_dict[cell_id].get_center_node()
        if center_node not in cell_graph.nodes:
            continue

        if steps_in_episode > 0:
            critic_mean = stats.get('critic_sum', 0.0) / steps_in_episode
            eligibility_mean = stats.get('eligibility_sum', 0.0) / steps_in_episode
            bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
            dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
        else:
            critic_mean = eligibility_mean = bikes_mean = dead_bikes_mean = 0.0

        nx_attrs = cell_graph.nodes[center_node]

        nx_attrs['critic_mean'] = critic_mean
        nx_attrs['eligibility_mean'] = eligibility_mean
        nx_attrs['failure_sum'] = cell_dict[cell_id].get_failures()
        nx_attrs['failure_rate'] = cell_dict[cell_id].get_failure_rate()
        nx_attrs['visits_sum'] = cell_dict[cell_id].get_visits()
        nx_attrs['ops_sum'] = cell_dict[cell_id].get_ops()
        nx_attrs['bikes_mean'] = bikes_mean
        nx_attrs['dead_bikes_mean'] = dead_bikes_mean

    # ============================================================================
    # Return results (no losses, q_values, epsilon tracking in validation)
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards,
        "failures_per_timeslot": failures,
        "total_invalid_actions": info["total_invalid_actions"],
        "action_per_step": action_per_step,
        "global_critic_scores": global_critic_scores,
        "reward_tracking_per_action": reward_tracking_per_action,
        "deployed_bikes": system_bikes,
        "truck_load": truck_load,
        "depot_load": depot_load,
        "outside_system_bikes": outside_system_bikes,
        'traveling_bikes': traveling_bikes,
        "cell_subgraph": cell_graph,
        # Validation doesn't track these (return empty for consistency)
        "q_values_per_timeslot": [],
        "epsilon_per_timeslot": [],
    }

# ----------------------------------------------------------------------------------------------------------------------
# Parallel validation worker
# ----------------------------------------------------------------------------------------------------------------------

def _validation_worker(
        state_dict: dict,
        num_actions: int,
        observation_space_len: int,
        data_path: str,
        episode_results_path: str,
        episode: int,
        run_id: int,
        logging_enabled: bool,
        params_snapshot: dict,
        reward_params_snapshot: dict,
        device_str: str,
        result_queue: mp.Queue,
) -> None:
    """
    Fully isolated validation process. Reconstructs its own agent (frozen, no
    replay buffer) and its own gym environment on the specified device.
    Sends ('success', result_dict) or ('error', error_str) back via result_queue.
    """
    val_env = None
    val_tbar = None

    try:
        val_device = torch.device(device_str)

        # ------------------------------------------------------------------
        # Reconstruct frozen agent from the weight snapshot — no replay buffer
        # ------------------------------------------------------------------
        val_agent = DQNAgent(
            num_actions=num_actions,
            observation_space_len=observation_space_len,
            gamma=params_snapshot["gamma"],
            epsilon_start=0.01,
            epsilon_end=0.01,
            epsilon_decay=1,
            lr=params_snapshot["lr"],
            device=val_device,
            tau=params_snapshot["tau"],
            soft_update=False,
            replay_buffer=None,
        )
        val_agent.train_model.load_state_dict(
            {k: v.to(val_device) for k, v in state_dict.items()}
        )
        val_agent.train_model.eval()

        # ------------------------------------------------------------------
        # Own environment instance
        # ------------------------------------------------------------------
        val_env = gym.make(
            'gymnasium_env/FullyDynamicEnv-v0',
            data_path=data_path,
            results_path=f"{episode_results_path}/",
            seed=params_snapshot['seed'],
            logging_enabled=logging_enabled,
        )

        # ------------------------------------------------------------------
        # Own tqdm bar at position=1, disappears when done (leave=False)
        # ------------------------------------------------------------------
        val_tbar = tqdm(
            total=params_snapshot["total_timeslots"],
            desc=f"[VAL] Epis {episode}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        result = validate_dqn(
            env=val_env,
            agent=val_agent,
            episode=episode,
            device=val_device,
            run_id=run_id,
            logging_enabled=logging_enabled,
            tbar=val_tbar,
            episode_results_path=episode_results_path,
            params_snapshot=params_snapshot,
            reward_params_snapshot=reward_params_snapshot,
        )

        # cell_subgraph is a networkx graph — it's picklable, no issues
        result_queue.put(('success', result))

    except Exception:
        import traceback
        result_queue.put(('error', traceback.format_exc()))
    finally:
        if val_tbar is not None:
            val_tbar.close()
        if val_env is not None:
            val_env.close()
        torch.cuda.empty_cache()


def _collect_pending_validation(
        validation_process: mp.Process | None,
        result_queue: mp.Queue,
        pending_episode: int | None,
        results_manager: ResultsManager,
        best_validation_score: float,
        num_days: int,
        agent: DQNAgent,
        logger,
        block: bool = False,
) -> tuple[float, float | None]:
    """
    Checks whether the running validation process has finished and, if so,
    collects and processes its result.

    Args:
        block: If True, wait for the process to finish (used at end of training).

    Returns:
        Updated best_validation_score, and last_validation_score (or None if
        no result was collected this call).
    """
    if validation_process is None:
        return best_validation_score, None

    if block:
        validation_process.join()
    elif validation_process.is_alive():
        return best_validation_score, None

    # Process has finished — drain the queue
    if result_queue.empty():
        logger.error(f"Validation process for episode {pending_episode} exited with no result.")
        return best_validation_score, None

    status, payload = result_queue.get_nowait()

    if status == 'error':
        logger.error(f"Validation process for episode {pending_episode} failed:\n{payload}")
        return best_validation_score, None

    validation_dict = payload

    validation_results = EpisodeResults(
        episode=pending_episode,
        mode='validation',
        epsilon=0.05,
        epsilon_per_timeslot=validation_dict.get('epsilon_per_timeslot', []),
        rewards_per_timeslot=validation_dict['rewards_per_timeslot'],
        total_reward=sum(validation_dict['rewards_per_timeslot']),
        failures_per_timeslot=validation_dict['failures_per_timeslot'],
        total_failures=sum(validation_dict['failures_per_timeslot']),
        mean_daily_failures=sum(validation_dict['failures_per_timeslot']) / num_days,
        action_per_step=validation_dict['action_per_step'],
        total_invalid_actions=validation_dict['total_invalid_actions'],
        reward_tracking_per_action=validation_dict['reward_tracking_per_action'],
        q_values_per_timeslot=validation_dict.get('q_values_per_timeslot', []),
        mean_q_values=float(np.mean(validation_dict['q_values_per_timeslot'])) if validation_dict[
            'q_values_per_timeslot'] else 0.0,
        deployed_bikes=validation_dict['deployed_bikes'],
        global_critic_scores=validation_dict.get('global_critic_scores', []),
        cell_subgraph=validation_dict['cell_subgraph'],
    )

    results_manager.save_episode(validation_results)
    last_validation_score = validation_results.total_failures

    is_best = validation_results.total_failures < best_validation_score
    if is_best:
        best_validation_score = validation_results.total_failures

    model_path = results_manager.save_model(
        agent=agent,
        episode=pending_episode,
        score=validation_results.total_failures,
        model_type='best' if is_best else 'checkpoint',
        save_best=is_best,
    )

    logger.info(
        f"Episode {pending_episode}: Validation failures = {validation_results.mean_daily_failures:.2f} mean / "
        f"{validation_results.total_failures} total | "
        f"Best = {best_validation_score} | "
        f"Invalid actions = {validation_results.total_invalid_actions} | "
        f"Model saved: {model_path} ({'best' if is_best else 'checkpoint'})"
    )

    return best_validation_score, last_validation_score

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # spawn is required before any CUDA context is created
    mp.set_start_method('spawn', force=True)

    warnings.filterwarnings("ignore")
    args = create_parser().parse_args()

    device = setup_device(args.device.lower(), devices)

    # Save parsed arguments to global variables
    run_id = args.run_id
    data_path = args.data_path
    results_path = args.results_path
    logging_enabled = args.log
    one_validation = args.one_validation

    # Update params dict with parsed values
    params['seed'] = args.seed
    params['num_episodes'] = args.num_episodes
    params['maximum_number_of_bikes'] = args.max_num_bikes
    params['minimum_number_of_bikes'] = args.min_num_bikes
    params['enable_repositioning'] = args.enable_repositioning
    params['use_net_flow'] = args.use_net_flow
    params['exploration_time'] = args.exploration_time

    set_seed(params['seed'])

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # At 60% of the total timeslots (60% of the training) the epsilon should be 0.1
    params["epsilon_decay"] = ((params["exploration_time"] * params["num_episodes"] * params["total_timeslots"])**2) / np.log(10)
    print(f"\nParams in use: {params}\n")
    print(f"Reward params in use: {reward_params}\n")
    if one_validation:
        print("one_validation = True")

    results_manager = ResultsManager(results_path, run_id)
    results_manager.save_hyperparameters(params, reward_params)

    # Init logging
    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(results_manager.training_path, "logs"),
        run_id=run_id,
        console=False,
        logger_name="train",
    ))

    logger = get_logger("train", logger_name="train")
    logger.info("Starting training loop")

    # Create the environment
    env = gym.make(
        'gymnasium_env/FullyDynamicEnv-v0',
        data_path=data_path,
        results_path=f"{str(results_manager.training_path)}/",
        seed=params['seed'],
        logging_enabled=logging_enabled
    )
    env.unwrapped.seed(params['seed'])
    env.action_space.seed(params['seed'])
    env.observation_space.seed(params['seed'])

    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])

    # Initialize the DQN agent
    agent = DQNAgent(
        replay_buffer=replay_buffer,
        num_actions=env.action_space.n,
        observation_space_len=env.observation_space.shape[0],
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
    num_days = params["total_timeslots"] // 8

    # Parallel validation state
    validation_process: mp.Process | None = None
    result_queue: mp.Queue = mp.Queue()
    pending_validation_episode: int | None = None

    try:
        tbar = tqdm(
            range(params["total_timeslots"]*params["num_episodes"]),
            desc="Training computation is starting ",
            initial=starting_episode*params["total_timeslots"],
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        logger.info(f"Training started with the following parameters: {params}")

        # Train and validation loop
        best_training_score = 1e4
        best_validation_score = 1e4
        for episode in range(starting_episode, params["num_episodes"]):
            # ------------------------------------------------------------------
            # Opportunistically collect any finished validation result before
            # starting a new training episode — non-blocking
            # ------------------------------------------------------------------

            if validation_process is not None and not validation_process.is_alive():
                best_validation_score, collected_score = _collect_pending_validation(
                    validation_process=validation_process,
                    result_queue=result_queue,
                    pending_episode=pending_validation_episode,
                    results_manager=results_manager,
                    best_validation_score=best_validation_score,
                    num_days=num_days,
                    agent=agent,
                    logger=logger,
                )
                if collected_score is not None:
                    last_validation_score = collected_score
                validation_process = None
                pending_validation_episode = None

            training_dict = train_dqn(
                env=env,
                agent=agent,
                batch_size=params["batch_size"],
                episode=episode,
                device=device,
                run_id=run_id,
                logging_enabled=logging_enabled,
                tbar=tbar,
                episode_results_path=os.path.join(f"{str(results_manager.training_path)}", f"episode_{episode:03d}")
            )

            # Convert to EpisodeResults
            training_results = EpisodeResults(
                episode=episode,
                mode='train',
                epsilon=agent.epsilon,
                epsilon_per_timeslot=training_dict['epsilon_per_timeslot'],
                rewards_per_timeslot=training_dict['rewards_per_timeslot'],
                total_reward=sum(training_dict['rewards_per_timeslot']),
                failures_per_timeslot=training_dict['failures_per_timeslot'],
                total_failures=sum(training_dict['failures_per_timeslot']),
                mean_daily_failures=sum(training_dict['failures_per_timeslot']) / num_days,
                q_values_per_timeslot=training_dict['q_values_per_timeslot'],
                mean_q_values=float(np.mean(training_dict['q_values_per_timeslot'])) if training_dict['q_values_per_timeslot'] else 0.0,
                deployed_bikes=training_dict['deployed_bikes'],
                truck_load=training_dict['truck_load'],
                depot_load=training_dict['depot_load'],
                outside_system_bikes=training_dict['outside_system_bikes'],
                action_per_step=training_dict['action_per_step'],
                total_invalid_actions=training_dict['total_invalid_actions'],
                reward_tracking_per_action=training_dict['reward_tracking_per_action'],
                global_critic_scores=training_dict['global_critic_scores'],
                cell_subgraph=training_dict['cell_subgraph'],
            )

            # Save using ResultsManager
            results_manager.save_episode(training_results)  # TODO: parallelize savings

            logger.info(
                f"Episode {episode}: Mean Failures = {training_results.mean_daily_failures:.2f}, "
                f"Total Failures = {training_results.total_failures}, "
                f"Invalid Actions = {training_results.total_invalid_actions}"
            )

            # Decide whether to validate
            if training_results.mean_daily_failures < best_training_score:
                best_training_score = training_results.mean_daily_failures
                should_validate = agent.epsilon < 0.15 and not one_validation
            else:
                should_validate = False

            # Always validate on the last episode
            if episode == params["num_episodes"] - 1:
                should_validate = True

            if should_validate:
                results_manager.save_model(
                    agent=agent,
                    episode=episode,
                    score=training_results.mean_daily_failures,
                    model_type='best',
                    save_best=True,
                )
                logger.info(
                    f"Episode {episode}: Saved best model "
                    f"(mean_daily_failures={training_results.mean_daily_failures:.2f})"
                )
                # if validation_process is not None and validation_process.is_alive():
                #     # A validation is already running — skip this one to avoid
                #     # hammering the GPU and stacking up processes
                #     logger.info(
                #         f"Episode {episode}: Skipping validation, previous one (episode "
                #         f"{pending_validation_episode}) still running."
                #     )
                # else:
                #     # Snapshot weights to CPU — fast, avoids deep-copying the
                #     # entire agent (which would drag the replay buffer along)
                #     state_dict_cpu = {
                #         k: v.cpu().clone()
                #         for k, v in agent.train_model.state_dict().items()
                #     }
                #
                #     val_episode_path = os.path.join(
                #         str(results_manager.validation_path), f"episode_{episode:03d}"
                #     )
                #
                #     validation_process = mp.Process(
                #         target=_validation_worker,
                #         args=(
                #             state_dict_cpu,
                #             env.action_space.n,
                #             env.observation_space.shape[0],
                #             data_path,
                #             val_episode_path,
                #             episode,
                #             run_id,
                #             logging_enabled,
                #             dict(params),
                #             dict(reward_params),
                #             str(device),
                #             result_queue,
                #         ),
                #         daemon=True,
                #     )
                #     pending_validation_episode = episode
                #     validation_process.start()
                #     logger.info(
                #         f"Episode {episode}: Launched validation process "
                #         f"(PID {validation_process.pid}) on {device}."
                #     )

            gc.collect()

        # ------------------------------------------------------------------
        # Training done — block until any outstanding validation finishes
        # ------------------------------------------------------------------
        # if validation_process is not None:
        #     logger.info(
        #         f"Training complete. Waiting for validation of episode "
        #         f"{pending_validation_episode} to finish..."
        #     )
        #     best_validation_score, collected_score = _collect_pending_validation(
        #         validation_process=validation_process,
        #         result_queue=result_queue,
        #         pending_episode=pending_validation_episode,
        #         results_manager=results_manager,
        #         best_validation_score=best_validation_score,
        #         num_days=num_days,
        #         agent=agent,
        #         logger=logger,
        #         block=True,
        #     )
        #     if collected_score is not None:
        #         last_validation_score = collected_score
        #     validation_process = None

        final_score = last_validation_score if last_validation_score is not None else best_training_score
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
        env.close()
    except Exception as e:
        # Make sure we don't leave orphan processes on crash
        if validation_process is not None and validation_process.is_alive():
            validation_process.terminate()
            validation_process.join()
        raise e
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if validation_process is not None and validation_process.is_alive():
            validation_process.terminate()
            validation_process.join()
        return

    # Print the rewards after training
    print(f"\nTraining {run_id} completed.")
