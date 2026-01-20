"""
Preprocess global rates.

This module computes the global (total) request rates for each day/timeslot
combination from the rate matrices.
"""

import argparse
import os
import pickle

import pandas as pd
from tqdm import tqdm

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.utils import kahan_sum


def run(config: PreprocessingConfig) -> None:
    """
    Run the global rates preprocessing step.

    Parameters:
        config: The preprocessing configuration.
    """
    os.makedirs(config.utils_path, exist_ok=True)

    global_rates = {}
    tbar = tqdm(total=len(config.days_of_week) * config.num_time_slots, desc="Processing global rates")

    for day in config.days_of_week:
        for timeslot in range(config.num_time_slots):
            matrix_path = os.path.join(
                config.data_path,
                "matrices",
                config.month_str,
                str(timeslot).zfill(2),
            )

            rate_matrix_file = os.path.join(matrix_path, f"{day.lower()}-rate-matrix.csv")
            if not os.path.exists(rate_matrix_file):
                print(f"Rate matrix not found: {rate_matrix_file}, skipping...")
                tbar.update(1)
                continue

            rate_matrix = pd.read_csv(rate_matrix_file, index_col="osmid")
            global_rate = kahan_sum(rate_matrix.to_numpy().flatten())
            global_rates[(day.lower(), timeslot)] = global_rate

            tbar.update(1)

    global_rates_path = os.path.join(config.data_path, config.global_rates_path)
    print(f"\nSaving global rates to {global_rates_path}...")

    with open(global_rates_path, "wb") as f:
        pickle.dump(global_rates, f)

    # print("Global rates:")
    # for key, value in global_rates.items():
    #     print(f"  {key}: {value:.6f}")


def main():
    """CLI entry point for preprocess_global_rates."""
    parser = argparse.ArgumentParser(description="Preprocess global rates.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--months", type=str, default="9,10", help="Comma-separated months to process.")

    args = parser.parse_args()
    months = [int(m.strip()) for m in args.months.split(",")]

    config = PreprocessingConfig(data_path=args.data_path, months=months)
    run(config)


if __name__ == "__main__":
    main()
