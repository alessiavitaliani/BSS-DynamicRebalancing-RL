import os
import pickle
import numpy as np

def concatenate_data(training_dir='training'):
    """
    Reads subfolders named numerically from 'training',
    loads the list from 'data_per_timeslot.pkl' in each,
    concatenates them, and saves them in 'data.pkl' under 'training'.
    """
    # List all items in training_dir
    items = os.listdir(training_dir)
    # Filter only folders that look numeric (e.g. "00", "01", "02", ...)
    numeric_subdirs = [d for d in items if os.path.isdir(os.path.join(training_dir, d)) and d.isdigit()]

    # Sort them numerically (so 1 comes before 10, etc.)
    numeric_subdirs.sort(key=lambda x: int(x))

    # Prepare a list to hold the concatenated data
    all_data = []

    # Read each 'data_per_timeslot.pkl' from the sorted subdirectories
    for subdir in numeric_subdirs:
        pkl_path = os.path.join(training_dir, subdir, 'rewards_per_timeslot.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                # q_values_per_timeslot = pickle.load(f)
                # data = []
                # for q_value in q_values_per_timeslot:
                #     data.append(np.mean(q_value))
                all_data.extend(data)
        else:
            print(f"Warning: '{pkl_path}' not found. Skipping...")

    # Save the combined list to 'data.pkl' in the training directory
    output_path = os.path.join(training_dir, 'rewards.pkl')
    with open(output_path, 'wb') as out_file:
        pickle.dump(all_data, out_file)
    print(f"Combined data saved to: {output_path}")


if __name__ == '__main__':
    concatenate_data('results/training_0/data/')