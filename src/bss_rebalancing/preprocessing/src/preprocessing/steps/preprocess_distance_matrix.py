"""
Preprocess distance matrix for routing.

This module computes the shortest path distances between all pairs of nodes
in the graph.
"""

import argparse
import os

import networkx as nx
import osmnx as ox
import pandas as pd

from preprocessing import plot_graph
from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.graph import load_graph


def initialize_distance_matrix(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Initialize a distance matrix based on shortest paths in the graph.

    Parameters:
        G: The graph representing the road network.

    Returns:
        DataFrame containing shortest path distances between all node pairs.
    """
    print("Calculating all shortest paths...")

    node_ids = ox.graph_to_gdfs(G, edges=False).index
    df = pd.DataFrame(index=node_ids, columns=node_ids, dtype="int")
    df = df.fillna(0)

    # Calculate shortest paths on undirected graph
    G_undirected = G.to_undirected()
    distances = dict(nx.all_pairs_dijkstra_path_length(G_undirected, weight="length"))

    for i in node_ids:
        for j in node_ids:
            df.at[i, j] = int(distances[i][j])

    return df


def run(config: PreprocessingConfig) -> None:
    """
    Run the distance matrix preprocessing step.

    Parameters:
        config: The preprocessing configuration.
    """
    os.makedirs(config.utils_path, exist_ok=True)

    print("Initializing the graph...")
    graph = load_graph(config.graph_path)

    print("Initializing the distance matrix...")
    distance_matrix = initialize_distance_matrix(graph)

    distance_matrix_path = os.path.join(config.data_path, config.distance_matrix_path)
    print(f"Saving distance matrix to {distance_matrix_path}...")
    distance_matrix.to_csv(distance_matrix_path, index=True)

    print(f"Distance matrix shape: {distance_matrix.shape}")


def main():
    """CLI entry point for preprocess_distance_matrix."""
    parser = argparse.ArgumentParser(description="Preprocess the distance matrix.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")

    args = parser.parse_args()

    config = PreprocessingConfig(data_path=args.data_path)
    run(config)


if __name__ == "__main__":
    main()
