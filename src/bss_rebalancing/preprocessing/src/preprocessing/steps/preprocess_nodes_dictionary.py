"""
Preprocess nodes dictionary.

This module creates a dictionary of nearby nodes for each node in the graph,
used for user-related operations.
"""

import argparse
import os
import pickle

import osmnx as ox
from tqdm import tqdm

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.graph import load_graph
from preprocessing.core.utils import nodes_within_radius


def run(config: PreprocessingConfig) -> None:
    """
    Run the nodes dictionary preprocessing step.

    Parameters:
        config: The preprocessing configuration.
    """
    os.makedirs(config.utils_path, exist_ok=True)

    print("Initializing the graph...")
    graph = load_graph(config.graph_path)

    # Build nodes dictionary
    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row["y"], row["x"]) for node_id, row in nodes_gdf.iterrows()}

    print(f"Creating nearby nodes dictionary (radius={config.user_radius}m)...")
    nearby_nodes_dict = {
        node_id: nodes_within_radius(node_id, nodes_dict, config.user_radius)
        for node_id in tqdm(nodes_dict, desc="Nodes")
    }

    nearby_nodes_path = os.path.join(config.data_path, config.nearby_nodes_path)
    print(f"Saving nearby nodes dictionary to {nearby_nodes_path}...")

    with open(nearby_nodes_path, "wb") as file:
        pickle.dump(nearby_nodes_dict, file)

    print(f"Processed {len(nearby_nodes_dict)} nodes.")


def main():
    """CLI entry point for preprocess_nodes_dictionary."""
    parser = argparse.ArgumentParser(description="Preprocess the nodes dictionary.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--radius", type=int, default=250, help="Radius in meters for nearby nodes.")

    args = parser.parse_args()

    config = PreprocessingConfig(data_path=args.data_path, user_radius=args.radius)
    run(config)


if __name__ == "__main__":
    main()
