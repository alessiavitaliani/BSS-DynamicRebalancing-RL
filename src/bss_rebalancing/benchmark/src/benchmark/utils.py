"""
Benchmark Utility Functions.

This module provides utility functions for the benchmark system including
time conversion, plotting, visualization, and monitoring capabilities.

Author: Edoardo Scarpel
"""

import random

import networkx as nx
import numpy as np

from gymnasium_env.simulator.cell import Cell


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# =============================================================================
# Time Conversion
# =============================================================================

def convert_seconds_to_hours_minutes(seconds: int) -> str:
    """
    Convert seconds to formatted time string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted string as "HH:MM:SS".

    Example:
        >>> convert_seconds_to_hours_minutes(3661)
        '01:01:01'
    """
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24  # Wrap to 24-hour format
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =============================================================================
# Graph Conversion
# =============================================================================

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
