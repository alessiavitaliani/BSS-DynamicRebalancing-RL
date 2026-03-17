import torch
import requests
import networkx as nx
import numpy as np
import psutil, os
import random
import logging

from torch_geometric.utils import from_networkx
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from gymnasium_env.simulator.cell import Cell

plt.ion()

def set_seed(seed):
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.geometric.seed(seed)


def setup_device(device_str: str, devices: list) -> torch.device:
    """Set up the computation device."""
    if device_str not in devices:
        raise ValueError(f"Invalid device '{device_str}'. Available options: {devices}")

    device = torch.device(device_str)

    if device.type == "cuda":
        gpu_id = device.index
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"Using CUDA device {gpu_id}: {gpu_name}")
    else:
        print(f"Using device: {device.type}")

    return device


def convert_seconds_to_hours_minutes(seconds) -> str:
    """
    Converts seconds to a formatted string of hours, minutes, and seconds.

    Parameters:
        - seconds: Time duration in seconds.

    Returns:
        - A string formatted as "HH:MM:SS".
    """
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def build_cell_graph_from_cells(
        cells: dict[int, Cell],
        nodes_dict: dict[int, tuple[float, float]],
        distance_lookup: dict[int, dict],
) -> nx.MultiDiGraph:
    """
    Build a NetworkX graph from Cell objects for GNN processing.

    Reads all node features directly from Cell.metrics instead of passing them as parameters.

    Parameters:
        cells: Dictionary of Cell objects keyed by cell_id
        nodes_dict: Dictionary mapping node_id to (lat, lon) coordinates
        distance_lookup: Dictionary for fast distance lookups between nodes

    Returns:
        NetworkX MultiDiGraph with nodes representing cells and edges representing adjacency
    """
    graph = nx.MultiDiGraph()
    graph.graph['crs'] = "EPSG:4326"

    max_distance = 0.0

    # First pass: Add nodes and find max distance
    for cell_id, cell in cells.items():
        center_node = cell.get_center_node()
        coords = nodes_dict[center_node]

        # Get ALL metrics from the cell
        node_attrs = {
            "cell_id": cell_id,
            "x": coords[1],  # longitude
            "y": coords[0],  # latitude
            "boundary": cell.get_boundary(),
        }

        # Add all cell metrics as node attributes
        node_attrs.update(cell.get_all_metrics())

        graph.add_node(center_node, **node_attrs)

        # Calculate max distance for normalization
        for adj_cell_id in cell.get_adjacent_cells().values():
            if adj_cell_id and adj_cell_id in cells:
                adj_center = cells[adj_cell_id].get_center_node()
                distance = distance_lookup[center_node][str(adj_center)]
                max_distance = max(max_distance, distance)

    # Second pass: Add edges with normalized distances
    for cell_id, cell in cells.items():
        center_node = cell.get_center_node()

        for adj_cell_id in cell.get_adjacent_cells().values():
            if adj_cell_id and adj_cell_id in cells:
                adj_center = cells[adj_cell_id].get_center_node()
                distance = distance_lookup[center_node][str(adj_center)]

                # Normalize distance
                normalized_distance = distance / max_distance if max_distance > 0 else 0.0

                graph.add_edge(
                    center_node,
                    adj_center,
                    distance=normalized_distance,
                    raw_distance=distance  # Keep raw distance too
                )

    return graph


def update_cell_graph_features(graph: nx.MultiDiGraph, cells: dict) -> nx.MultiDiGraph:
    """Update only node feature attributes in-place. Edges/structure unchanged."""
    for cell_id, cell in cells.items():
        center_node = cell.get_center_node()
        if center_node in graph.nodes:
            graph.nodes[center_node].update(cell.get_all_metrics())
    return graph


def convert_graph_to_data(
        graph: nx.MultiDiGraph,
        node_features: list[str] = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Parameters:
        graph: NetworkX MultiDiGraph with node attributes from Cell.metrics
        node_features: List of metric names to use as features. If None, uses defaults.

    Returns:
        PyTorch Geometric Data object ready for GNN
    """
    # Default features if none specified
    if node_features is None:
        # throw an error if node_features is not provided, since we rely on Cell metrics
        raise ValueError("node_features must be provided to specify which Cell metrics to use as features")

    # Convert to PyG Data
    data = from_networkx(graph)

    # Extract node features
    node_feature_tensors = []
    for attr in node_features:
        nodes = sorted(graph.nodes())
        feature_values = [
            graph.nodes[n].get(attr, 0.0) or 0.0
            for n in nodes
        ]
        node_feature_tensors.append(
            torch.tensor(feature_values, dtype=torch.float32).unsqueeze(-1)
        )

    data.x = torch.cat(node_feature_tensors, dim=-1)

    # Extract edge attributes
    edge_distances = []
    for u, v, k, attr in graph.edges(data=True, keys=True):
        edge_distances.append(attr.get('distance', 0.0))

    data.edge_attr = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(-1)

    # Store feature names for reference
    data.feature_names = node_features

    return data

