import os
import torch
import argparse
import pickle
import warnings

import gymnasium as gym
import numpy as np

import gymnasium_env.register_env

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from utils import convert_seconds_to_hours_minutes

# ----------------------------------------------------------------------------------------------------------------------

data_path = "../data/"

# set seed
seed = 31
np.random.seed(seed)
torch.manual_seed(seed)

params = {
    "num_episodes": 40,  # Total number of training episodes
    "total_timeslots": 56,  # Total number of time slots in one episode (1 month)
    "fixed_rebal_bikes_per_cell": 5,  # Min bikes per cell after static rebalancing (deactivated on options!!!)
    "maximum_number_of_bikes": 250  # Maximum number of bikes in the system

}


def simulate_env(env: gym, episode: int, tbar: tqdm | tqdm_telegram) -> dict:
    # Initialize episode metrics
    timeslot = 0
    timeslots_completed = 0
    failures_per_timeslot = []
    rebalance_time = []

    # Reset environment and agent state
    options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'total_timeslots': params["total_timeslots"],
        # 'fixed_rebal_bikes_per_cell': params["fixed_rebal_bikes_per_cell"],
        'depot_id': 18,  # 491 back
        'initial_cell': 18,  # 185 back
        'num_rebalancing_events': 2
    }

    env.reset(options=options)

    not_done = True

    while not_done:
        # Step the environment with the chosen action
        *_, done, terminated, info = env.step(0)  # this is the step

        # Check if the episode is complete
        not_done = not done

        if terminated:
            timeslots_completed += 1

            # Log progress
            tbar.set_description(f"Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                 f"at {convert_seconds_to_hours_minutes(info['time'])}")
            timeslot = 0 if timeslot == 7 else timeslot + 1

            failures_per_timeslot.extend(info['failures'])
            rebalance_time.extend(info['rebalance_time'])

            # Update progress bar
            tbar.update(1)

    env.close()

    return {'failures': failures_per_timeslot, 'rebalance_time': rebalance_time}


# ----------------------------------------------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    # Create the environment
    env = gym.make('gymnasium_env/StaticEnv-v0', data_path=data_path)
    env.unwrapped.seed(seed)

    tbar = tqdm(
        range(int(params["total_timeslots"] / 8) * params["num_episodes"]),
        desc="Training Episode 0, Week 0, Monday at 01:00:00",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    total_failures = []
    rebalance_time = []

    for episode in range(0, params["num_episodes"]):
        results = simulate_env(env, episode, tbar)  # This is the sauce
        total_failures.extend(results['failures'])
        rebalance_time.extend(results['rebalance_time'])

    tbar.close()

    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/total_failures.pkl', 'wb') as f:
        pickle.dump(total_failures, f)
    with open('results/rebalance_time.pkl', 'wb') as f:
        pickle.dump(rebalance_time, f)

    # Print the rewards after training
    print("\nSimulation completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark file')
    parser.add_argument('--data_path', type=str, default="data/", help='Path to the data folder')

    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path

    main()