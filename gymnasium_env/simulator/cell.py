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
        total_bikes (int): Number of Bike objects in the cell.
        request_rate (float): Rate of bike requests from the cell. Default is 0.0.
        visits (int): Number of time the truck has visited the cell (specific increment function is not defined here)
        operations (int): Number of time the truck has perform operations in the cell (specific increment function is not defined here)
        failures (int): Number of total failures occurred in this cell during the episode.
        critic_score (float): Score representing the rate between bike requested and available bikes (specific function is not defined here)
        is_critical (boolean): True if critic_score is greater of 0.0, False otherwise.
        surplus_bikes (int): Number of bikes in eccess of what is needed (specific function is not defined here)
        eligibility_score (float): Eligibility value decaing by time from the last visit of the truck.
        """
        self.id = cell_id
        self.boundary = boundary
        self.nodes = []
        self.center_node = 0
        self.diagonal = 0
        self.adjacent_cells = {'up': None, 'down': None, 'left': None, 'right': None}
        self.total_bikes = 0
        self.request_rate = 0
        self.visits = 0
        self.operations = 0
        self.failures = 0
        self.total_rebalanced = 0
        self.critic_score = 0
        self.is_critical = False
        self.surplus_bikes = 0
        self.eligibility_score = 0

    def __str__(self):
        return f"Cell {self.id}: Bikes: {self.total_bikes}, Critic Score: {self.critic_score}, Ops: {self.operations}"

    def set_center_node(self, graph: nx.MultiDiGraph):
        center_coords = self.boundary.centroid.coords[0]
        nearest_node = ox.distance.nearest_nodes(graph, center_coords[0], center_coords[1])
        if nearest_node in self.nodes:
            self.center_node = nearest_node
        else:
            raise ValueError("Center node not found in cell nodes")

    def contain_nodes (self, point: Point) -> bool:
        return self.boundary.contains(point)

    def to_dict(self):
        # Serialize boundary as WKT (Well-Known Text) and other attributes
        return {
            'id': self.id,
            'boundary': self.boundary.wkt,  # Convert boundary to WKT string for saving
            'nodes': ','.join(map(str, self.nodes)),  # Store nodes as a comma-separated string
            'center_node': self.center_node,
            'diagonal': self.diagonal,
            'adjacent_cells': self.adjacent_cells,
        }

    @classmethod
    def from_dict(cls, data):
        # Convert WKT boundary back to a Polygon, and nodes from a comma-separated string
        boundary = wkt.loads(data['boundary'])
        cell = cls(cell_id=data['id'], boundary=boundary)
        cell.nodes = list(map(int, data['nodes'].split(',')))
        cell.center_node = data['center_node']
        cell.diagonal = data['diagonal']
        cell.adjacent_cells = ast.literal_eval(data['adjacent_cells'])
        return cell
    
    def reset(self):
        self.total_bikes = 0
        self.request_rate = 0
        self.visits = 0
        self.operations = 0
        self.failures = 0
        self.total_rebalanced = 0
        self.critic_score = 0
        self.is_critical = False
        self.surplus_bikes = 0

    def reset_failures(self):
        self.failures = 0

    def reset_total_rebalanced(self):
        self.total_rebalanced = 0

    def set_diagonal(self):
        """
        This is a torch geometric function used by the precomputing algorithms to compute the distance between cells
        """
        coords = list(self.boundary.exterior.coords)[:-1]
        side_length_meters = geodesic(coords[0], coords[1]).meters
        self.diagonal = int(math.sqrt(2) * side_length_meters)

    def set_total_bikes(self, total_bikes: int):
        self.total_bikes = total_bikes

    def set_request_rate(self, request_rate: float):
        self.request_rate = request_rate

    def set_visits(self, visits: int):
        self.visits = visits

    def set_ops(self, ops: int):
        self.operations = ops

    def set_critic_score(self, critic_score: float):
        """
        This function sets the critic_score to the passed value and updates the respective flags of the cell
        Parameters:
        critic_score (float): Value of the critic score to set
        """
        self.critic_score = critic_score
        if critic_score > 0.05:
            self.is_critical = True
            self.surplus_bikes = 0
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
        if self.critic_score > -surplus_threshold:
            self.surplus_bikes = 0
        else:
            self.surplus_bikes = self.total_bikes - math.floor((self.total_bikes*((1 + self.critic_score)/(1 - self.critic_score)))/((1-surplus_threshold)/(1+surplus_threshold)))

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
        return self.total_bikes

    def get_request_rate(self) -> float:
        return self.request_rate

    def get_visits(self) -> int:
        return self.visits

    def get_ops(self) -> int:
        return self.operations

    def get_failures(self) -> int:
        return self.failures

    def get_total_rebalanced(self) -> int:
        return self.total_rebalanced

    def get_critic_score(self) -> float:
        return self.critic_score

    def get_surplus_bikes(self) -> float:
        return self.surplus_bikes

    def add_failure(self, f: int = 1):
        """
        This function adds a number of failures to the failure counter of the cell
        Parameters:
        f (int): Number of failures to add (default 1)
        """
        self.failures += f

    def update_rebalanced_times(self):
        """
        This functions adds one to the counter of the times the cell is rebalanced
        """
        self.total_rebalanced += 1

    def update_eligibility_score(self, eligibility_decay: float):
        """
        This function updates the eligibility decay of the cell by one step.
        Parameters:
        eligibility_decay (float): Rate of the decay
        """
        if  self.eligibility_score < 0.001:
            self.eligibility_score = 0.0
        else:
            self.eligibility_score *= eligibility_decay
