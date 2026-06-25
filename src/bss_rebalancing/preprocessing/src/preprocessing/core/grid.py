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
def set_adjacent_cells(cell_dict: Dict[int, Cell]) -> None:
    """
    Set the adjacent cells for each cell in the dictionary using coordinate-based direction.
    """
    tbar = tqdm(total=len(cell_dict), desc="Setting adjacent cells", dynamic_ncols=True)
    
    # 1. AUMENTO SOGLIA: Se hai poche celle (es. 22), 
    # la distanza tra i centri è di diversi km. 5000m assicura la connessione.
    cell_size = 300
    threshold = cell_size*15 

    cells = list(cell_dict.values())

    for cell in cells:
        center = cell.get_boundary().centroid.coords[0]
        cell_adj = cell.get_adjacent_cells()
        
        # Initialize the minimum distances to infinity for all 4 directions
        closest_dist = {'up': float('inf'), 'down': float('inf'), 'left': float('inf'), 'right': float('inf')}
        closest_cells = {'up': None, 'down': None, 'left': None, 'right': None}

        for adj_cell in cells:
            if adj_cell.get_id() == cell.get_id():
                continue
            
            adj_center = adj_cell.get_boundary().centroid.coords[0]
            
            # center[0] is the longitude (X), center[1] is the latitude (Y)
            lon_diff = adj_center[0] - center[0]
            lat_diff = adj_center[1] - center[1]
            
            # Calculate the distance in meters
            dist = haversine((center[1], center[0]), (adj_center[1], adj_center[0]), unit=Unit.METERS)
            
            # If the cell is too far away, ignore it
            if dist > threshold:
                continue
            
            # Rotate the vectors to correct the grid's perspective (45°)
            x_rot = lon_diff + lat_diff
            y_rot = lat_diff - lon_diff
            
            # Determine the direction of the neighbor using the rotated axes
            if abs(y_rot) > abs(x_rot):
                direction = "up" if y_rot > 0 else "down"
            else:
                direction = "right" if x_rot > 0 else "left"
            # ----------------------------------------------------
            
            # If this is the closest neighbor we've found so far in this direction, let's save it
            if dist < closest_dist[direction]:
                closest_dist[direction] = dist
                closest_cells[direction] = adj_cell.get_id()

        # Finally, assign the found neighbors to the cell
        for direction in ['up', 'down', 'left', 'right']:
            cell_adj[direction] = closest_cells[direction]

        tbar.update(1)
        
    tbar.close()

    # Enforce absolute symmetry
    for cell_id, cell in cell_dict.items():
        adj = cell.get_adjacent_cells()
        
        if adj['right'] is not None:
            cell_dict[adj['right']].get_adjacent_cells()['left'] = cell_id
        if adj['left'] is not None:
            cell_dict[adj['left']].get_adjacent_cells()['right'] = cell_id
        if adj['up'] is not None:
            cell_dict[adj['up']].get_adjacent_cells()['down'] = cell_id
        if adj['down'] is not None:
            cell_dict[adj['down']].get_adjacent_cells()['up'] = cell_id


def remove_empty_cells(cell_dict: Dict[int, Cell]) -> Dict[int, Cell]:
    """
    Remove cells that contain no nodes.

    Parameters:
        cell_dict: Dictionary of Cell objects.

    Returns:
        New dictionary with empty cells removed.
    """
    return {cell_id: cell for cell_id, cell in cell_dict.items() if len(cell.get_nodes()) > 0}