"""
Shared utility functions for the preprocessing pipeline.
"""

import calendar
from typing import Dict, Tuple, Hashable
from haversine import haversine, Unit

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
