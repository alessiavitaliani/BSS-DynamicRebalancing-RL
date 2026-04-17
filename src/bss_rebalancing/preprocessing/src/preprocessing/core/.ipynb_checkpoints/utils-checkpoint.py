"""
Shared utility functions for the preprocessing pipeline.
"""

import polars as pl
from typing import Dict, Tuple, Hashable
from haversine import haversine, Unit


def reorder_df(df: pl.DataFrame, col: str) -> pl.DataFrame:
    # Sort rows by node_id
    df = df.sort(col)

    # Extract numeric column names (excluding node_id)
    data_cols = df.select(pl.exclude(col)).columns

    # Sort columns numerically (important: cast to int!)
    sorted_cols = sorted(data_cols, key=lambda x: int(x))

    # Reorder dataframe
    df = df.select([col] + sorted_cols)

    return df


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def nodes_within_radius(
    target_node: int | Hashable,
    nodes_dict: Dict[int, Tuple[float, float]],
    radius: int,
) -> Dict[int, Tuple[float, float]]:
    """
    Find all nodes within a specified radius of a target node.

    Parameters:
        target_node: The ID of the target node.
        nodes_dict: Dictionary mapping node IDs to (lat, lon) coordinates.
        radius: The radius in meters.

    Returns:
        Dictionary of nodes within the radius, mapping node IDs to coordinates.

    Raises:
        ValueError: If the target node is not found in nodes_dict.
    """
    target_coords = nodes_dict.get(target_node)
    if not target_coords:
        raise ValueError("Target node not found in nodes dictionary")

    nearby_nodes = {
        node_id: coords
        for node_id, coords in nodes_dict.items()
        if node_id != target_node and haversine(target_coords, coords, unit=Unit.METERS) <= radius
    }
    return nearby_nodes