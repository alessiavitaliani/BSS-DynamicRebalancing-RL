"""
Grid and cell-related utilities for truck routing.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from haversine import haversine, Unit
from gymnasium_env.simulator.cell import Cell


def divide_graph_into_cells(graph: nx.MultiDiGraph, cell_size: int) -> Dict[int, Cell]:
    """
    Divide the graph area into diamond-shaped cells.

    Parameters:
        graph: The graph object representing the street network.
        cell_size: The size of the cell in meters.

    Returns:
        Dictionary mapping cell IDs to Cell objects.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)
    min_x, min_y, max_x, max_y = nodes.total_bounds

    # Conversion factors
    lat_deg_to_meter = 111320  # 1 degree latitude = 111320 meters
    lat_lon_to_meter = 111320 * abs(np.cos(np.radians(np.mean([min_y, max_y]))))

    x_diagonal_deg = cell_size * np.sqrt(2) / lat_lon_to_meter
    y_diagonal_deg = cell_size * np.sqrt(2) / lat_deg_to_meter

    # Create grid centers for two offset grids
    x_centers_od = np.arange(min_x, max_x + x_diagonal_deg, x_diagonal_deg)
    x_centers_even = np.arange(min_x + x_diagonal_deg / 2, max_x + x_diagonal_deg, x_diagonal_deg)
    y_centers_od = np.arange(min_y, max_y + y_diagonal_deg, y_diagonal_deg)
    y_centers_even = np.arange(min_y + y_diagonal_deg / 2, max_y + y_diagonal_deg, y_diagonal_deg)

    cell_dict = {}
    cell_id = 0

    for x_centers, y_centers in zip([x_centers_od, x_centers_even], [y_centers_od, y_centers_even]):
        for x in x_centers:
            for y in y_centers:
                # Create diamond-shaped cell
                vertices = [
                    (x + x_diagonal_deg / 2, y),
                    (x, y + y_diagonal_deg / 2),
                    (x - x_diagonal_deg / 2, y),
                    (x, y - y_diagonal_deg / 2),
                ]
                cell_boundary = Polygon(vertices)
                cell = Cell(
                    cell_id=cell_id,
                    boundary=cell_boundary,
                    cell_size=cell_size
                )
                cell_dict[cell_id] = cell
                cell_id += 1

    return cell_dict


def assign_nodes_to_cells(graph: nx.MultiDiGraph, cell_dict: Dict[int, Cell]) -> List[Tuple[int, int]]:
    """
    Assign graph nodes to their containing cells.

    Parameters:
        graph: The graph object representing the street network.
        cell_dict: Dictionary of Cell objects.

    Returns:
        List of (node_id, cell_id) tuples.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)

    tbar = tqdm(total=len(nodes), desc="Assigning nodes to cells", dynamic_ncols=True)
    node_assignments = []

    for node_id, row in nodes.iterrows():
        node_point = Point(row["x"], row["y"])

        for cell in cell_dict.values():
            if cell.get_boundary().contains(node_point):
                cell.get_nodes().append(node_id)
                node_assignments.append((node_id, cell.get_id()))
                break

        tbar.update(1)

    return node_assignments


def set_adjacent_cells(cell_dict: Dict[int, Cell]) -> None:
    """
    Set the adjacent cells for each cell in the dictionary.

    Parameters:
        cell_dict: Dictionary of Cell objects (modified in place).
    """
    tbar = tqdm(total=len(cell_dict), desc="Setting adjacent cells", dynamic_ncols=True)

    for cell in cell_dict.values():
        center_coords = cell.get_boundary().centroid.coords[0]

        for adj_cell in cell_dict.values():
            if adj_cell.get_id() != cell.get_id():
                adj_center_coords = adj_cell.get_boundary().centroid.coords[0]

                if haversine(center_coords, adj_center_coords, unit=Unit.METERS) < 300:
                    lon_diff = center_coords[0] - adj_center_coords[0]
                    lat_diff = center_coords[1] - adj_center_coords[1]

                    cell_adj_cells = cell.get_adjacent_cells()
                    adj_cell_adj_cells = adj_cell.get_adjacent_cells()

                    if lon_diff > 0 and lat_diff > 0:
                        cell_adj_cells["left"] = adj_cell.get_id()
                        adj_cell_adj_cells["right"] = cell.get_id()

                    if lon_diff < 0 and lat_diff < 0:
                        cell_adj_cells["right"] = adj_cell.get_id()
                        adj_cell_adj_cells["left"] = cell.get_id()

                    if lon_diff > 0 > lat_diff:
                        cell_adj_cells["up"] = adj_cell.get_id()
                        adj_cell_adj_cells["down"] = cell.get_id()

                    if lon_diff < 0 < lat_diff:
                        cell_adj_cells["down"] = adj_cell.get_id()
                        adj_cell_adj_cells["up"] = cell.get_id()

        tbar.update(1)


def remove_empty_cells(cell_dict: Dict[int, Cell]) -> Dict[int, Cell]:
    """
    Remove cells that contain no nodes.

    Parameters:
        cell_dict: Dictionary of Cell objects.

    Returns:
        New dictionary with empty cells removed.
    """
    return {cell_id: cell for cell_id, cell in cell_dict.items() if len(cell.get_nodes()) > 0}