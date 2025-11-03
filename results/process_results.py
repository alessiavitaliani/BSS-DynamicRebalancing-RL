import pickle
import os
import numpy as np
import pandas as pd

from utils import plot_data_online, plot_confusion_matrix
from collections import defaultdict
from tqdm import tqdm

# CHANGE THIS VARIABLE TO 'train', 'validate' or 'benchmark'
MODE = 'benchmark'

action_bin_labels = ['STAY', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']
epsilon_threshold = 1.1

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
num2days = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}


# ----------------------------------------------------------------------------------------------------------------------

def find_index(lst, eps_thr):
    for i, epsilon in enumerate(lst):
        if epsilon <= eps_thr:
            return i


def load_data(base_path: str):
    with open(os.path.join(base_path, 'rewards_per_timeslot.pkl'), 'rb') as f:
        rewards_per_timeslot = pickle.load(f)
    with open(os.path.join(base_path, 'failures_per_timeslot.pkl'), 'rb') as f:
        failures_per_timeslot = pickle.load(f)
    with open(os.path.join(base_path, 'action_per_step.pkl'), 'rb') as f:
        action_per_step = pickle.load(f)
    with open(os.path.join(base_path, 'q_values_per_timeslot.pkl'), 'rb') as f:
        q_values_per_timeslot = pickle.load(f)
    with open(os.path.join(base_path, 'epsilon_per_timeslot.pkl'), 'rb') as f:
        epsilon_per_timeslot = pickle.load(f)
    return rewards_per_timeslot, failures_per_timeslot, action_per_step, q_values_per_timeslot, epsilon_per_timeslot


# ----------------------------------------------------------------------------------------------------------------------

def process_training_results(base_path: str):
    data_path = os.path.join(base_path, "data")

    # Get all subfolders in the 'data' directory
    timeslot_folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
    timeslot_folders.sort()

    q_values_tuple = []
    total_failures = []

    tbar = tqdm(total=len(timeslot_folders), desc="Processing data per time slot")

    for timeslot_folder in timeslot_folders:
        timeslot_path = os.path.join(data_path, timeslot_folder)

        # Load data
        # rewards_per_timeslot, failures_per_timeslot, action_per_step = load_data(timeslot_path)
        # with open(os.path.join(timeslot_path, 'q_values_per_timeslot.pkl'), 'rb') as f:
        #     q_values_per_timeslot = pickle.load(f)
        #     q_values_tuple.extend(q_values_per_timeslot)

        # epsilons = [e for _, e in rewards_per_timeslot]
        # rewards = [r for r, _ in rewards_per_timeslot]
        # failures = [f for f, _ in failures_per_timeslot]
        # total_failures.extend(failures)
        rewards, failures, actions, q_values, epsilons = load_data(timeslot_path)

        q_values_tuple.extend(q_values)
        total_failures.extend(failures)

        # Action bins
        action_bins = [0] * len(action_bin_labels)
        for a in actions:
            action_bins[a] += 1

        # Create plots directory if it doesn't exist
        plots_path = os.path.join(base_path, 'plots')
        plots_timeslot_path = os.path.join(plots_path, timeslot_folder)
        if not os.path.exists(plots_timeslot_path):
            os.makedirs(plots_timeslot_path, exist_ok=True)
            print(f"Directory '{plots_timeslot_path}' created.")
        os.makedirs(plots_path, exist_ok=True)

        # Generate plots
        plot_data_online(rewards, idx=1, xlabel='Time Slot', ylabel='Reward',
                         save_path=os.path.join(plots_timeslot_path, 'rewards_per_timeslot.png'))
        plot_data_online(failures, idx=2, xlabel='Time Slot', ylabel='Failures',
                         save_path=os.path.join(plots_timeslot_path, 'failures_per_timeslot.png'))
        plot_data_online(action_bins, idx=4, xlabel='Action', ylabel='Frequency', show_histogram=True,
                         bin_labels=action_bin_labels, save_path=os.path.join(plots_timeslot_path, 'action_bins.png'))
        plot_data_online(epsilons, idx=5, xlabel='Time Slot', ylabel='Epsilon',
                         save_path=os.path.join(plots_timeslot_path, 'epsilon.png'), mean=False)

        tbar.update(1)

    q_values_index = find_index(epsilons, epsilon_threshold)
    q_values = [q for q in q_values_tuple[q_values_index:]]
    mean_q_values = [np.mean(q) for q in q_values]
    plots_path = os.path.join(base_path, 'plots')
    plot_data_online(mean_q_values, idx=3, xlabel='Time Slot', ylabel='Q-Value',
                     save_path=os.path.join(plots_path, 'q_values_per_timeslot.png'))
    plot_data_online(total_failures, idx=6, xlabel='Time Slot', ylabel='Total Failures',
                     save_path=os.path.join(plots_path, 'total_failures.png'))


def process_validation_results(base_path: str):
    rewards_per_timeslot, failures_per_timeslot, action_per_step, epsilon_per_timeslot = load_data(base_path)

    failures_df = pd.DataFrame()

    for day, day_num in days2num.items():
        rewards = []
        failures = []
        for timeslot in range(0, 8):
            k = 0
            rews = 0
            fails = 0

            while k * 7 * 8 + day_num * 8 + timeslot < len(
                    rewards_per_timeslot) and k * 7 * 8 + day_num * 8 + timeslot < len(failures_per_timeslot):
                rews += rewards_per_timeslot[k * 7 * 8 + day_num * 8 + timeslot][0]
                fails += failures_per_timeslot[k * 7 * 8 + day_num * 8 + timeslot][0]
                k += 1

            if k > 0:
                rews /= k
                fails /= k

            rewards.append(rews)
            failures.append(fails)

            # Assign to DataFrame
            failures_df.loc[timeslot, day.capitalize()] = fails

        if not os.path.exists(base_path + 'plots/rewards'):
            os.makedirs(base_path + 'plots/rewards')
            print(f"Directory '{base_path}plots/rewards' created.")
        plot_data_online(rewards, idx=1, xlabel='Time Slot', ylabel='Reward',
                         save_path=base_path + 'plots/rewards/rewards_per_timeslot_' + day + '.png')

        if not os.path.exists(base_path + 'plots/failures'):
            os.makedirs(base_path + 'plots/failures')
            print(f"Directory '{base_path}plots/failures' created.")
        plot_data_online(failures, idx=2, xlabel='Time Slot', ylabel='Failures',
                         save_path=base_path + 'plots/failures/failures_per_timeslot_' + day + '.png')

        plot_confusion_matrix(failures_df, title="Failures per Time Slot", x_label="Day", y_label="Time Slot",
                              cbar_label="Failures", save_path=base_path + 'plots/failures/failures_heatmap.png')

    action_bins = [0] * len(action_bin_labels)
    for action, _ in action_per_step:
        action_bins[action] += 1

    plot_data_online(action_bins, idx=4, xlabel='Action', ylabel='Frequency', show_histogram=True,
                     bin_labels=action_bin_labels, save_path=base_path + 'plots/action_bins.png')


def process_benchmark_results(base_path: str):
    """
    Loads all .pkl files in a folder and generates corresponding plots.

    Parameters
    ----------
    base_path : str
        Path to the folder containing .pkl files.
    idx : int, optional
        Index passed to plot_data_online (default is 1).
    xlabel : str, optional
        Label for the x-axis (default is 'Time Slot').
    ylabel : str, optional
        Label for the y-axis (default is 'Data').
    """
    idx = 1
    xlabel = 'Time Slot'
    ylabels = {
        'rebalance_time.pkl': 'Rebalance Time (seconds)',
        'total_failures.pkl': 'Failures',
    }
    default_ylabel = 'Data'

    # Ensure the path exists
    if not os.path.exists(base_path):
        print(f"Folder not found: {base_path}")
        return

    # Loop through all .pkl files in the directory
    for filename in os.listdir(base_path):
        if filename.endswith('.pkl'):
            filepath = os.path.join(base_path, filename)

            # Load pickle data
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            # Determine y-axis label
            ylabel = ylabels.get(filename, default_ylabel) if ylabels else default_ylabel

            # Create output path for plot
            plot_name = os.path.splitext(filename)[0] + '.png'
            plot_path = os.path.join(base_path, plot_name)

            # Call your plotting function
            try:
                plot_data_online(
                    data,
                    idx=idx,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    save_path=plot_path
                )
                print(f"Saved plot: {plot_path}")
            except Exception as e:
                print(f"Error plotting {filename}: {e}")


# ----------------------------------------------------------------------------------------------------------------------

def main():
    if MODE == 'train':
        base_path = 'results/training_1'
        process_training_results(base_path)
    elif MODE == 'validate':
        base_path = 'results/validation_1'
        process_validation_results(base_path)
    elif MODE == 'benchmark':
        base_path = 'benchmarks/results'
        process_benchmark_results(base_path)
        # with open('benchmarks/results/rebalance_time.pkl', 'rb') as f:
        #     data = pickle.load(f)
        #     plot_data_online(data, idx=1, xlabel='Time Slot', ylabel='Data', save_path=os.path.join(base_path, 'rebalance_times.png'))
    else:
        raise ValueError("Invalid mode. Choose 'train', 'validate' or 'benchmark'.")

    if not os.path.exists(os.path.join(base_path, 'plots')):
        os.makedirs(os.path.join(base_path, 'plots'))

    total_failures = []
    with open(os.path.join('results/total_failures_baseline.pkl'), 'rb') as f:
        total_failures = pickle.load(f)

    failures = [f for f, _ in total_failures]

    plot_data_online(failures, idx=6, xlabel='Time Slot', ylabel='Total Failures',
                     save_path=os.path.join('total_failures_baseline.png'))

    # with open('benchmarks/results/rebalance_time.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     plot_data_online(data, idx=1, xlabel='Time Slot', ylabel='Data')


if __name__ == '__main__':
    main()