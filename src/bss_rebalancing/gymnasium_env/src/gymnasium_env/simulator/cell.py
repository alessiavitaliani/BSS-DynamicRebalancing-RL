import osmnx as ox
import networkx as nx
import ast
import math

from geopy.distance import geodesic
from shapely.geometry import Polygon, Point
from shapely import wkt

class Cell:
    def __init__(self, cell_id, boundary: Polygon):
        """
        Initialize a Cell object.

        Parameters:
        cell_id (int): Unique identifier for the cell.
        boundary (Polygon): Values of the boundaries of the cell.
        nodes (array): This is a value extracted or computed from the graph (nx.MultiDiGraph) file.
        center_node (int): ID of the station designated as the center station of the cell.
        diagonal (int): This is a value extracted or computed from the graph (nx.MultiDiGraph) file.
        adjacent_cell (dict): Dictionary of the 4 adjacent cell of a cell with directions associated with the cell IDs of the adjacent sells. Default is None
        is_critical (boolean): True if critic_score is greater of 0.0, False otherwise.
        metrics (dict): Dictionary of the metrics of the cell.
        """
        self.id = cell_id
        self.boundary = boundary
        self.nodes = []
        self.center_node = 0
        self.diagonal = 0
        self.adjacent_cells = {'up': None, 'down': None, 'left': None, 'right': None}
        self.is_critical = False
        self.metrics = {
            "total_bikes": 0,
            "request_rate": 0.0,
            "visits": 0,
            "operations": 0,
            "total_departures": 0,
            "failures": 0,
            "total_rebalanced": 0,
            "critic_score": 0.0,
            "surplus_bikes": 0,
            "eligibility_score": 0.0
        }

    def __str__(self):
        return (
            f"Cell {self.id}: "
            f"Bikes: {self.metrics['total_bikes']}, "
            f"Critic Score: {self.metrics['critic_score']}, "
            f"Ops: {self.metrics['operations']}"
        )

    def set_center_node(self, graph: nx.MultiDiGraph):
        center_coords = self.boundary.centroid.coords[0]
        nearest_node = ox.distance.nearest_nodes(graph, center_coords[0], center_coords[1])
        if nearest_node in self.nodes:
            self.center_node = nearest_node
        else:
            raise ValueError("Center node not found in cell nodes")

    def update_metric(self, metric_name: str, value: float | int):
        """Update a specific metric."""
        self.metrics[metric_name] = value

    def get_metric(self, metric_name: str) -> float:
        """Get a specific metric value."""
        return self.metrics.get(metric_name, 0.0)

    def get_all_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        return self.metrics.copy()
    
    def reset(self):
        for key in self.metrics:
            self.metrics[key] = 0 if isinstance(self.metrics[key], int) else 0.0
        self.is_critical = False

    def set_diagonal(self):
        """
        This is a torch geometric function used by the precomputing algorithms to compute the distance between cells
        """
        coords = list(self.boundary.exterior.coords)[:-1]
        side_length_meters = geodesic(coords[0], coords[1]).meters
        self.diagonal = int(math.sqrt(2) * side_length_meters)

    def set_total_bikes(self, total_bikes: int):
        self.metrics["total_bikes"] = total_bikes

    def set_request_rate(self, request_rate: float):
        self.metrics["request_rate"] = request_rate

    def set_visits(self, visits: int):
        self.metrics["visits"] = visits

    def set_ops(self, ops: int):
        self.metrics["operations"] = ops

    def set_eligibility_score(self, eligibility_score: float):
        self.metrics["eligibility_score"] = eligibility_score

    def set_critic_score(self, critic_score: float):
        """
        This function sets the critic_score to the passed value and updates the respective flags of the cell
        Parameters:
        critic_score (float): Value of the critic score to set
        """
        self.metrics["critic_score"] = critic_score
        if critic_score > 0.05:
            self.is_critical = True
            self.metrics["surplus_bikes"] = 0
        else:
            self.is_critical = False

    def set_surplus_bikes(self, surplus_threshold: float = 0.67):
        """
        This function sets the surplus score base on the surplus_threshold of the critic_score.
        If the cell is critic or the negative critic_score is not inferior to the treashold, then the cell is not in surplus.
        self.surplus_bikes is the number of bikes in surplus.

        Parameters:
        surplus_threshold (float): Value of the critic score under which the cell is considered "in surplus"
        """
        if surplus_threshold <= 0.0 or surplus_threshold >= 1.0:
            raise ValueError("Invalid surplus_threshold selected. Must be between 0 and 1 .")
        critic_score = self.metrics["critic_score"]
        total_bikes = self.metrics["total_bikes"]
        if critic_score > - surplus_threshold:
            self.metrics["surplus_bikes"] = 0
        else:
            self.metrics["surplus_bikes"] = total_bikes - math.floor((total_bikes*((1 + critic_score)/(1 - critic_score)))/((1-surplus_threshold)/(1+surplus_threshold)))

    def get_id(self) -> int:
        return self.id

    def get_boundary(self) -> Polygon:
        return self.boundary

    def get_nodes(self) -> list[int]:
        return self.nodes

    def get_center_node(self) -> int:
        return self.center_node

    def get_adjacent_cells(self) -> dict:
        return self.adjacent_cells

    def get_diagonal(self) -> int:
        return self.diagonal

    def get_total_bikes(self) -> int:
        return self.metrics["total_bikes"]

    def get_request_rate(self) -> float:
        return self.metrics["request_rate"]

    def get_visits(self) -> int:
        return self.metrics["visits"]

    def get_ops(self) -> int:
        return self.metrics["operations"]
    
    def get_total_departures(self) -> int:
        return self.metrics["total_departures"]

    def get_failures(self) -> int:
        return self.metrics["failures"]

    def get_total_rebalanced(self) -> int:
        return self.metrics["total_rebalanced"]

    def get_critic_score(self) -> float:
        return self.metrics["critic_score"]

    def get_surplus_bikes(self) -> float:
        return self.metrics["surplus_bikes"]

    def get_eligibility_score(self) -> float:
        return self.metrics["eligibility_score"]
    
    def add_departure(self, d: int = 1):
        """
        This function adds a number of departures to the departure counter of the cell
        Parameters:
        d (int): Number of departures to add (default 1)
        """
        self.metrics["total_departures"] = self.metrics["total_departures"] + d

    def add_failure(self, f: int = 1):
        """
        This function adds a number of failures to the failure counter of the cell
        Parameters:
        f (int): Number of failures to add (default 1)
        """
        self.metrics["failures"] = self.metrics["failures"] + f

    def update_rebalanced_times(self):
        """
        This functions adds one to the counter of the times the cell is rebalanced
        """
        self.metrics["total_rebalanced"] = self.metrics["total_rebalanced"] + 1

    def update_eligibility_score(self, eligibility_decay: float):
        """
        This function updates the eligibility decay of the cell by one step.
        Parameters:
        eligibility_decay (float): Rate of the decay
        """
        if  self.metrics["eligibility_score"] < 0.001:
            self.metrics["eligibility_score"] = 0.0
        else:
            self.metrics["eligibility_score"] *= eligibility_decay
