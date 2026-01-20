"""
Shared utility functions for the preprocessing pipeline.
"""

import calendar
from typing import Dict, Tuple

import numpy as np
from geopy.distance import distance


def kahan_sum(arr) -> float:
    """
    Compute the sum of an array using Kahan summation for improved numerical accuracy.

    Parameters:
        arr: Array-like of values to sum.

    Returns:
        The sum of the array values with reduced floating-point error.
    """
    total = 0.0
    c = 0.0  # A running compensation for lost low-order bits.
    for value in arr:
        y = value - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def compute_distance(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """
    Calculate the distance between two pairs of coordinates in meters.

    Parameters:
        coords1: A tuple (lat, lon) for the first coordinate.
        coords2: A tuple (lat, lon) for the second coordinate.

    Returns:
        The distance in meters between the two coordinates.
    """
    return distance(coords1, coords2).meters


def haversine_distance(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """
    Calculate the haversine distance between two pairs of coordinates in meters.

    This is faster than geodesic distance but slightly less accurate.

    Parameters:
        coords1: A tuple (lat, lon) for the first coordinate.
        coords2: A tuple (lat, lon) for the second coordinate.

    Returns:
        The haversine distance in meters between the two coordinates.
    """
    # Convert degrees to radians
    lat1, lon1 = np.radians(coords1)
    lat2, lon2 = np.radians(coords2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in meters
    earth_radius = 6371000
    return earth_radius * c


def count_specific_day(year: int, month: int, day_name: str) -> int:
    """
    Count the number of occurrences of a specific day in a given month.

    Parameters:
        year: The year.
        month: The month (1-12).
        day_name: Name of the day (e.g., 'Monday', 'Tuesday').

    Returns:
        The number of times the specified day occurs in that month.

    Raises:
        ValueError: If an invalid day name is provided.
    """
    day_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    if day_name.lower() not in day_map:
        raise ValueError(
            "Invalid day name. Choose from: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday."
        )

    day_number = day_map[day_name.lower()]
    num_days = calendar.monthrange(year, month)[1]

    count = 0
    for day in range(1, num_days + 1):
        if calendar.weekday(year, month, day) == day_number:
            count += 1

    return count


def nodes_within_radius(
    target_node: int,
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
        if node_id != target_node and haversine_distance(target_coords, coords) <= radius
    }

    return nearby_nodes
