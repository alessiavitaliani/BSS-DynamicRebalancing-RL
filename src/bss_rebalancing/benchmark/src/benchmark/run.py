"""
Benchmark runner for static bike-sharing environment.

This module provides the main execution logic for running benchmarks
on the static environment.
"""

import os
import argparse
import warnings
import logging
from typing import Union

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdm_telegram

import gymnasium_env
from .utils import convert_seconds_to_hours_minutes, build_cell_graph_from_cells, update_cell_graph_features
from .logging_config import LoggingConfig, init_logging, get_logger
from .results_manager import EpisodeResults, ResultsManager


# =============================================================================
# Configuration Constants
# =============================================================================

params = {
    "seed": 42,                                     # Random seed for reproducibility
    "num_episodes": 1,                              # Total number of episodes
    "total_timeslots": 56,                          # Total number of time slots in one episode

    "num_rebalancing_events": 2,                    # Number of rebalancing events per episode
    "enable_repositioning": False,                  # Use base repositioning strategy at the start of each episode
    "use_net_flow": False,                          # Use net flow repositioning strategy at the start of each episode

    "maximum_number_of_bikes": 500,                 # Maximum number of bikes in the system
    "minimum_number_of_bikes": 1,                   # Minimum number of bikes per cell
    "depot_position_id": 18,                        # ID (cell) of the depot position
    "initial_cell_id": 18                           # Initial cell where the truck starts
}


# =============================================================================
# Simulation Functions
# =============================================================================

def run_simulation(
    env: gym.Env,
    progress_bar: Union[tqdm, tqdm_telegram],
    episode_results_path: str | None = None,
) -> dict:
    """
    Run a single simulation episode.

    Args:
        env: Gymnasium environment instance.
        progress_bar: Progress bar for tracking simulation progress.
        episode_results_path:

    Returns:
        Dictionary containing:
            - failures: List of failure counts per timeslot
            - rebalance_time: List of rebalancing operation durations
    """
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    timeslot = 0
    failures = []
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    traveling_bikes = []
    global_critic_scores = []
    rebalance_times = []

    # Accumulators (reset each timeslot)
    timeslots_completed = 0
    iterations = 0

    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    reset_options = {
        'total_timeslots': params['total_timeslots'],
        'maximum_number_of_bikes': params['maximum_number_of_bikes'],
        'minimum_number_of_bikes': params['minimum_number_of_bikes'],
        'depot_id': params['depot_position_id'],
        'initial_cell': params['initial_cell_id'],
        'num_rebalancing_events': params['num_rebalancing_events'],
        'enable_repositioning': params['enable_repositioning'],
        'use_net_flow': params['use_net_flow']

    }
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path

    # Reset environment
    _, info = env.reset(options=reset_options)

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

    # ============================================================================
    # Main bench loop
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
        # Step environment (action=0 for static environment)
        _, _, done, timeslot_terminated, info = env.step(0)

        # Only update node attributes
        cell_dict = info['cell_dict']
        update_cell_graph_features(cell_graph, cell_dict)

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['eligibility_sum'] += cell.get_eligibility_score()
            stats['bikes_sum'] += cell.get_total_bikes()

        # Record step metrics
        iterations += 1

        # Process timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1

            # Record timeslot metrics
            failures.append(info['failures'])
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])
            rebalance_times.extend(info['rebalance_time'])
            global_critic_scores.append(info['global_critic_score'])

            # Update progress bar with current status
            week_display = info['week']
            day_display = info['day'].capitalize()
            time_display = convert_seconds_to_hours_minutes(info['time'])

            progress_bar.set_description(
                f"[BENCHMARK] Week {week_display}, "
                f"{day_display} at {time_display}"
            )

            # Update timeslot counter
            timeslot = (timeslot + 1) % 8

            # Update progress bar
            progress_bar.update(1)

        steps_in_episode = iterations
        for cell_id, stats in episode_cell_stats.items():
            center_node = cell_dict[cell_id].get_center_node()
            if center_node not in cell_graph.nodes:
                continue

            if steps_in_episode > 0:
                bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
                dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
            else:
                bikes_mean = dead_bikes_mean = 0.0

            nx_attrs = cell_graph.nodes[center_node]
            nx_attrs['failure_sum'] = cell_dict[cell_id].get_failures()
            nx_attrs['failure_rate'] = cell_dict[cell_id].get_failure_rate()
            nx_attrs['bikes_mean'] = bikes_mean

    env.close()

    return {
        "failures_per_timeslot": failures,
        "global_critic_scores": global_critic_scores,
        "deployed_bikes": system_bikes,
        "truck_load": truck_load,
        "depot_load": depot_load,
        "outside_system_bikes": outside_system_bikes,
        'traveling_bikes': traveling_bikes,
        "cell_subgraph": cell_graph,
        'rebalance_time': rebalance_times
    }


# =============================================================================
# Main Execution
# =============================================================================

def run_benchmark(config: dict):
    """
    Execute benchmark simulation with given configuration.

    Args:
        config: Configuration dictionary containing all parameters.
    """
    warnings.filterwarnings("ignore")

    # Save parsed arguments to global variables
    run_id = config['run_id']
    data_path = config['data_path']
    results_path = config['results_path']
    logging_enabled = config['log']

    params['seed'] = config['seed']
    params['num_episodes'] = config['num_episodes']
    params['maximum_number_of_bikes'] = config['maximum_number_of_bikes']
    params['minimum_number_of_bikes'] = config['minimum_number_of_bikes']
    params['enable_repositioning'] = config['enable_repositioning']
    params['use_net_flow'] = config['use_net_flow']

    # Set random seeds for reproducibility
    np.random.seed(params['seed'])

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    results_manager = ResultsManager(results_path, run_id)
    results_manager.save_hyperparameters(params)

    # Init logging
    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(results_manager.bench_path, "logs"),
        run_id=run_id,
        console=False,
        logger_name="benchmark",
    ))

    logger = get_logger("run", logger_name="benchmark")
    logger.info("Starting benchmark")

    # Create environment
    env = gym.make(
        'gymnasium_env/StaticEnv-v0',
        data_path=config['data_path'],
        results_path=f"{str(results_manager.bench_path)}/",
        seed=params['seed'],
        logging_enabled=logging_enabled
    )
    env.unwrapped.seed(params['seed'])

    # Calculate total simulation steps
    total_steps = params['total_timeslots'] * params['num_episodes']

    # Initialize progress bar
    progress_bar = tqdm(
        range(total_steps),
        desc="Episode 0, Week 0, Monday at 01:00:00",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    # Collect results across all episodes
    total_failures = []

    episode_results = run_simulation(
        env=env,
        progress_bar=progress_bar,
        episode_results_path=f"{str(results_manager.bench_path)}/"
    )
    total_failures.extend(episode_results['failures_per_timeslot'])

    progress_bar.close()

    num_days = params["total_timeslots"] // 8

    ep_results = EpisodeResults(
        episode=0,
        mode='benchmark',
        mean_daily_failures=sum(total_failures) / num_days if num_days > 0 else 0.0,
        total_failures=sum(total_failures),
        failures_per_timeslot=total_failures,
        deployed_bikes=episode_results['deployed_bikes'],
        truck_load=episode_results['truck_load'],
        depot_load=episode_results['depot_load'],
        outside_system_bikes=episode_results['outside_system_bikes'],
        traveling_bikes=episode_results['traveling_bikes'],
        cell_subgraph=episode_results['cell_subgraph'],
    )
    results_manager.save_episode(ep_results)
    results_manager.save_run_summary()

    print(f"\nTotal Failures: {ep_results.total_failures}")
    print("\nBenchmark completed successfully.")


def parse_arguments() -> dict:
    """
    Parse command-line arguments.

    Returns:
        Dictionary containing configuration parameters.
    """
    parser = argparse.ArgumentParser(
        description="BSS Benchmark Runner for Static Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            
                # Run benchmark with default settings
                python -m benchmark.run
                
                # Specify custom data and results paths
                python -m benchmark.run --data-path data/ --results-path results/
                
                # Customize simulation parameters
                python -m benchmark.run --num-episodes 5
                
                # Adjust fleet parameters
                python -m benchmark.run --maximum-number-of-bikes 400 --fixed-rebal-bikes-per-cell 10
                
                # Set a unique run identifier
                python -m benchmark.run --run-id 42
                
                # Combine multiple options
                python -m benchmark.run --data-path data/ --results-path results/ --num-episodes 3 --maximum-number-of-bikes 350 --run-id 7     
            """,
    )

    parser.add_argument(
        '--run-id',
        type=int,
        default=0,
        help='Unique identifier for this benchmark run'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/',
        help='Path to the data folder'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default='results/',
        help='Path where results should be saved (benchmark subfolder will be created)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=params['seed'],
        help='Random seed for reproducibility'
    )
    # Simulation parameters
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=params['num_episodes'],
        help='Number of episodes to simulate'
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
        '--log',
        action='store_true',
        help='Enable logging.'
    )

    args = parser.parse_args()

    # Convert to configuration dictionary
    config = {
        'run_id': args.run_id,
        'data_path': args.data_path,
        'results_path': args.results_path,
        'seed': args.seed,
        'num_episodes': args.num_episodes,
        'maximum_number_of_bikes': args.max_num_bikes,
        'minimum_number_of_bikes': args.min_num_bikes,
        'enable_repositioning': args.enable_repositioning,
        'use_net_flow': args.use_net_flow,
        'log': args.log,
    }

    return config


def main():
    """Main entry point for benchmark execution."""
    config = parse_arguments()
    run_benchmark(config)


if __name__ == '__main__':
    main()
