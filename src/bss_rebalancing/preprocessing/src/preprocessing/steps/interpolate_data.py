"""
Interpolate rate data to build PMF matrices.

This module interpolates the sparse rate matrices to create probability
mass function (PMF) matrices for trip distribution.
"""

import argparse
import os
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.graph import load_graph
from preprocessing.core.utils import compute_distance, kahan_sum, nodes_within_radius


def build_pmf_matrix(
    rate_matrix: pd.DataFrame,
    nearby_nodes_dict: Dict[int, Dict[int, Tuple[float, float]]],
    nodes_dict: Dict[int, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Build the PMF matrix from the rate matrix using spatial interpolation.

    Parameters:
        rate_matrix: The rate matrix (sparse).
        nearby_nodes_dict: Dictionary of nearby nodes for each node.
        nodes_dict: Dictionary of node coordinates.

    Returns:
        Interpolated PMF matrix.
    """
    pmf_df = rate_matrix.copy(deep=True)

    non_zero_rows = pmf_df[pmf_df.sum(axis=1) != 0].index.astype(int)

    for row in non_zero_rows:
        rates = pmf_df.loc[row, :]
        non_zero_nodes = rates[~rates.eq(0)].index.astype(int)
        zero_nodes = rates[rates.eq(0)].index.astype(int)

        for node_id in zero_nodes:
            nearby_nodes = nearby_nodes_dict.get(node_id, {})
            nearby_non_zero_nodes = {nz_id: nearby_nodes[nz_id] for nz_id in non_zero_nodes if nz_id in nearby_nodes}

            if nearby_non_zero_nodes:
                coords = nodes_dict[node_id]
                distances = np.array(
                    [compute_distance(coords, nn_coords) for nn_coords in nearby_non_zero_nodes.values()]
                )
                rts = np.array([rates.loc[nn_id] for nn_id in nearby_non_zero_nodes])

                num, den = 0.0, 0.0
                for distance, rate in zip(distances, rts):
                    num += rate / distance
                    den += 1 / distance
                pmf_df.loc[row, node_id] = num / den

    # Interpolate zero rows
    zero_rows = pmf_df[pmf_df.sum(axis=1) == 0].index.astype(int)

    for idx in zero_rows:
        nearby_nodes = nearby_nodes_dict.get(idx, {})
        nearby_non_zero_nodes = {nz_id: nearby_nodes[nz_id] for nz_id in non_zero_rows if nz_id in nearby_nodes}

        if nearby_non_zero_nodes:
            coords = nodes_dict[idx]
            distances = np.array(
                [compute_distance(coords, nn_coords) for nn_coords in nearby_non_zero_nodes.values()]
            )

            num = pd.Series(0.0, index=pmf_df.columns)
            den = 0.0
            for distance, nn_id in zip(distances, nearby_non_zero_nodes):
                num += pmf_df.loc[nn_id, :]
                den += 1 / distance
            pmf_df.loc[idx, :] = num / den

    return pmf_df


def build_pmf_matrix_external_trips(
    df: pd.Series,
    nearby_nodes_dict: Dict[int, Dict[int, Tuple[float, float]]],
    nodes_dict: Dict[int, Tuple[float, float]],
) -> pd.Series:
    """
    Build PMF for external trips (trips entering/leaving the network).

    Parameters:
        df: Series of rates for external trips.
        nearby_nodes_dict: Dictionary of nearby nodes for each node.
        nodes_dict: Dictionary of node coordinates.

    Returns:
        Interpolated series.
    """
    non_zero_nodes = df[~df.eq(0)].index.astype(int)
    zero_nodes = df[df.eq(0)].index.astype(int)

    # Exclude the external marker node
    if 10000 in non_zero_nodes:
        non_zero_nodes = non_zero_nodes[non_zero_nodes != 10000]
    if 10000 in zero_nodes:
        zero_nodes = zero_nodes[zero_nodes != 10000]

    for node_id in zero_nodes:
        nearby_nodes = nearby_nodes_dict.get(node_id, {})
        nearby_non_zero_nodes = {nz_id: nearby_nodes[nz_id] for nz_id in non_zero_nodes if nz_id in nearby_nodes}

        if nearby_non_zero_nodes:
            coords = nodes_dict[node_id]
            distances = np.array(
                [compute_distance(coords, nn_coords) for nn_coords in nearby_non_zero_nodes.values()]
            )
            rts = np.array([df.loc[nn_id] for nn_id in nearby_non_zero_nodes])

            num, den = 0.0, 0.0
            for distance, rate in zip(distances, rts):
                num += rate / distance
                den += 1 / distance
            df.loc[node_id] = num / den

    return df


def run(config: PreprocessingConfig) -> None:
    """
    Run the interpolate data step.

    Parameters:
        config: The preprocessing configuration.
    """
    print("Initializing the graph...")
    graph = load_graph(config.graph_path)

    # Build nodes dictionary
    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row["y"], row["x"]) for node_id, row in nodes_gdf.iterrows()}

    # Build nearby nodes dictionary
    print("Building nearby nodes dictionary...")
    nearby_nodes_dict = {
        node_id: nodes_within_radius(node_id, nodes_dict, config.interpolation_radius)
        for node_id in tqdm(nodes_dict, desc="Building Nearby Nodes", dynamic_ncols=True)
    }

    print("\nBuilding the PMF matrices...")

    tbar = tqdm(
        total=len(config.days_of_week) * config.num_time_slots,
        desc="Processing Data",
        position=0,
        dynamic_ncols=True,
    )

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
            rate_matrix.index = rate_matrix.index.astype(int)
            rate_matrix.columns = rate_matrix.columns.astype(int)

            # Save and remove external trips row/column
            saved_row = rate_matrix.loc[10000, :].copy()
            saved_col = rate_matrix.loc[:, 10000].copy()
            rate_matrix = rate_matrix.drop(index=10000, columns=10000)

            # Build PMF matrix
            pmf_matrix = build_pmf_matrix(rate_matrix, nearby_nodes_dict, nodes_dict)
            saved_row = build_pmf_matrix_external_trips(saved_row, nearby_nodes_dict, nodes_dict)
            saved_col = build_pmf_matrix_external_trips(saved_col, nearby_nodes_dict, nodes_dict)

            # Restore external trips
            pmf_matrix.loc[10000, :] = saved_row
            pmf_matrix.loc[:, 10000] = saved_col

            # Normalize to PMF
            total_sum = kahan_sum(pmf_matrix.to_numpy().flatten())
            if total_sum > 0:
                pmf_matrix = pmf_matrix / total_sum

            # Save PMF matrix
            pmf_matrix.to_csv(os.path.join(matrix_path, f"{day.lower()}-pmf-matrix.csv"), index=True)

            tbar.update(1)


def main():
    """CLI entry point for interpolate_data."""
    parser = argparse.ArgumentParser(description="Interpolate rate matrices to build PMF matrices.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--months", type=str, default="9,10", help="Comma-separated months to process.")
    parser.add_argument("--radius", type=int, default=500, help="Interpolation radius in meters.")

    args = parser.parse_args()
    months = [int(m.strip()) for m in args.months.split(",")]

    config = PreprocessingConfig(data_path=args.data_path, months=months, interpolation_radius=args.radius)
    run(config)


if __name__ == "__main__":
    main()
