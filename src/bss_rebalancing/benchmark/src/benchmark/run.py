"""
Benchmark runner for static bike-sharing environment.

This module provides the main execution logic for running benchmarks
on the static environment.
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Union

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdm_telegram

import gymnasium_env
from .utils import convert_seconds_to_hours_minutes


# =============================================================================
# Configuration Constants
# =============================================================================

class BenchmarkDefaults:
    """Default configuration values for benchmark runs."""

    # Simulation parameters
    NUM_EPISODES = 1
    TOTAL_TIMESLOTS = 56  # 1 week of simulation (8 timeslots/day * 7 days)

    # Fleet parameters
    MAXIMUM_BIKES = 300
    BIKES_PER_CELL = 5

    # Environment setup
    DEPOT_ID = 103
    INITIAL_CELL_ID = 103
    NUM_REBALANCING_EVENTS = 8

    # Reproducibility
    RANDOM_SEED = 32

    # Paths
    DEFAULT_DATA_PATH = "data/"
    DEFAULT_RESULTS_PATH = "results/"
    BENCHMARK_SUBFOLDER = "benchmark"

    # Output files
    FAILURES_FILE = "total_failures.pkl"
    REBALANCE_TIME_FILE = "rebalance_time.pkl"


# =============================================================================
# Simulation Functions
# =============================================================================

def run_simulation(
    env: gym.Env,
    episode: int,
    config: dict,
    progress_bar: Union[tqdm, tqdm_telegram]
) -> dict:
    """
    Run a single simulation episode.

    Args:
        env: Gymnasium environment instance.
        episode: Episode number (for logging).
        config: Configuration dictionary with simulation parameters.
        progress_bar: Progress bar for tracking simulation progress.

    Returns:
        Dictionary containing:
            - failures: List of failure counts per timeslot
            - rebalance_time: List of rebalancing operation durations
    """
    # Initialize metrics tracking
    timeslot = 0
    failures_per_timeslot = []
    rebalance_time = []

    # Configure environment reset options
    reset_options = {
        'total_timeslots': config['total_timeslots'],
        'maximum_number_of_bikes': config['maximum_number_of_bikes'],
        'fixed_rebal_bikes_per_cell': config['fixed_rebal_bikes_per_cell'],
        'depot_id': BenchmarkDefaults.DEPOT_ID,
        'initial_cell': BenchmarkDefaults.INITIAL_CELL_ID,
        'num_rebalancing_events': BenchmarkDefaults.NUM_REBALANCING_EVENTS
    }

    # Reset environment
    env.reset(options=reset_options)

    # Run episode until completion
    done = False
    while not done:
        # Step environment (action=0 for static environment)
        _, _, done, terminated, info = env.step(0)

        # Process timeslot completion
        if terminated:
            # Update progress bar with current status
            week_display = info['week'] % 52
            day_display = info['day'].capitalize()
            time_display = convert_seconds_to_hours_minutes(info['time'])

            progress_bar.set_description(
                f"Episode {episode}, Week {week_display}, "
                f"{day_display} at {time_display}"
            )

            # Update timeslot counter
            timeslot = (timeslot + 1) % 8

            # Collect metrics
            failures_per_timeslot.extend(info['failures'])
            rebalance_time.extend(info['rebalance_time'])

            # Update progress bar
            progress_bar.update(1)

    env.close()

    return {
        'failures': failures_per_timeslot,
        'rebalance_time': rebalance_time
    }


def save_results(results: dict, run_id: int, results_path: str):
    """
    Save benchmark results to disk.

    Args:
        results: Dictionary containing failures and rebalance_time lists.
        run_id: Unique identifier for this benchmark run.
        results_path: Base path where results should be saved.
    """
    # Create benchmark subfolder in the specified results path
    results_path = Path(results_path)
    benchmark_dir = results_path / BenchmarkDefaults.BENCHMARK_SUBFOLDER
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Create run-specific subdirectory
    run_dir = benchmark_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save failures data
    failures_path = run_dir / BenchmarkDefaults.FAILURES_FILE
    with open(failures_path, 'wb') as f:
        pickle.dump(results['failures'], f)

    # Save rebalancing time data
    rebalance_path = run_dir / BenchmarkDefaults.REBALANCE_TIME_FILE
    with open(rebalance_path, 'wb') as f:
        pickle.dump(results['rebalance_time'], f)

    print(f"\nResults saved to: {run_dir}")


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

    # Set random seeds for reproducibility
    np.random.seed(BenchmarkDefaults.RANDOM_SEED)
    torch.manual_seed(BenchmarkDefaults.RANDOM_SEED)

    # Create environment
    env = gym.make(
        'gymnasium_env/StaticEnv-v0',
        data_path=config['data_path']
    )
    env.unwrapped.seed(BenchmarkDefaults.RANDOM_SEED)

    # Calculate total simulation steps
    total_steps = config['total_timeslots'] * config['num_episodes']

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
    total_rebalance_time = []

    # Run episodes
    for episode in range(config['num_episodes']):
        episode_results = run_simulation(env, episode, config, progress_bar)
        total_failures.extend(episode_results['failures'])
        total_rebalance_time.extend(episode_results['rebalance_time'])

    # Close progress bar
    progress_bar.close()

    # Display summary
    print(f"\nTotal Failures: {sum(total_failures)}")

    # Save results # TODO: save resul
    results = {
        'failures': total_failures,
        'rebalance_time': total_rebalance_time
    }
    save_results(results, config['run_id'], config['results_path'])

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
                python -m benchmark.run --num-episodes 5 --total-timeslots 56
                
                # Adjust fleet parameters
                python -m benchmark.run --maximum-number-of-bikes 400 --fixed-rebal-bikes-per-cell 10
                
                # Set a unique run identifier
                python -m benchmark.run --run-id 42
                
                # Combine multiple options
                python -m benchmark.run --data-path data/ --results-path results/ --num-episodes 3 --maximum-number-of-bikes 350 --run-id 7     
            """,
    )

    # Path configuration
    parser.add_argument(
        '--data-path',
        type=str,
        default=BenchmarkDefaults.DEFAULT_DATA_PATH,
        help='Path to the data folder'
    )

    parser.add_argument(
        '--results-path',
        type=str,
        default=BenchmarkDefaults.DEFAULT_RESULTS_PATH,
        help='Path where results should be saved (benchmark subfolder will be created)'
    )

    # Run identification
    parser.add_argument(
        '--run-id',
        type=int,
        default=0,
        help='Unique identifier for this benchmark run'
    )

    # Simulation parameters
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=BenchmarkDefaults.NUM_EPISODES,
        help='Number of episodes to simulate'
    )

    parser.add_argument(
        '--total-timeslots',
        type=int,
        default=BenchmarkDefaults.TOTAL_TIMESLOTS,
        help='Total timeslots per episode (8 per day)'
    )

    # Fleet parameters
    parser.add_argument(
        '--maximum-number-of-bikes',
        type=int,
        default=BenchmarkDefaults.MAXIMUM_BIKES,
        help='Maximum number of bikes in the system'
    )

    parser.add_argument(
        '--fixed-rebal-bikes-per-cell',
        type=int,
        default=BenchmarkDefaults.BIKES_PER_CELL,
        help='Minimum bikes per cell after rebalancing'
    )

    args = parser.parse_args()

    # Convert to configuration dictionary
    config = {
        'data_path': args.data_path,
        'results_path': args.results_path,
        'run_id': args.run_id,
        'num_episodes': args.num_episodes,
        'total_timeslots': args.total_timeslots,
        'maximum_number_of_bikes': args.maximum_number_of_bikes,
        'fixed_rebal_bikes_per_cell': args.fixed_rebal_bikes_per_cell
    }

    return config


def main():
    """Main entry point for benchmark execution."""
    config = parse_arguments()
    run_benchmark(config)


if __name__ == '__main__':
    main()
