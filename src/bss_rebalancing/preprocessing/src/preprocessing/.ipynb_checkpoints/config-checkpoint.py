"""
Shared configuration for the preprocessing pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    # Location settings
    #place: List[str] = field(default_factory=lambda: ["Cambridge, Massachusetts, USA"])
    place: List[str] = field(default_factory=lambda: ["Manhattan, New York City, New York, USA"])
    network_type: str = "bike"

    # Path settings
    #data_path: str = "data_cambridge/"
    data_path: str = "data_manhattan/"
    #graph_file: str = "utils/cambridge_network.graphml"
    graph_file: str = "utils/manhattan_network.graphml"
    cell_data_path: str = "utils/cell_data.pkl"
    global_rates_path: str = "utils/global_rates.pkl"
    distance_matrix_path: str = "utils/distance_matrix.csv"
    nearby_nodes_path: str = "utils/nearby_nodes.pkl"

    # Time settings
    #year: int = 2022 # Cambridge
    year: int = 2024 # Manhattan
    months: List[int] = field(default_factory=lambda: [9, 10])

    # Nodes to remove from graph
    # nodes_to_remove: List[Tuple[float, float]] = field(default_factory=lambda: [(42.365455, -71.14254)])
    # nodes_to_remove: List[int] = field(default_factory=lambda: [330,482,54,256,36,324])
    nodes_to_remove : List[str] = field(
        default_factory=lambda: []
    )

    # Days of week to process
    days_of_week: List[str] = field(
        default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    # Bounding box (north, south, east, west) for Cambridge
    #bbox: Optional[Tuple[float, float, float, float]] = field(
    #    default_factory=lambda: (42.370, 42.353, -71.070, -71.117)
    #)
    # Bounding box (north, south, east, west) for Manhattan
    bbox: Optional[Tuple[float, float, float, float]] = field(
        default_factory=lambda: (40.8822, 40.6970, -73.9067, -74.0205)
    )

    # Grid settings
    #cell_size: int = 300  # meters (Cambridge)
    cell_size: int = 500  # meters (Manhattan)

    # Radius settings
    interpolation_radius: int = 500  # meters for PMF interpolation
    user_radius: int = 250  # meters for nearby nodes

    # Number of time slots per day
    num_time_slots: int = 8

    @property
    def month_str(self) -> str:
        """Return month string in format '09-10'."""
        return f"{str(self.months[0]).zfill(2)}-{str(self.months[-1]).zfill(2)}"

    @property
    def graph_path(self) -> str:
        """Return full path to graph file."""
        return os.path.join(self.data_path, self.graph_file)

    @property
    def utils_path(self) -> str:
        """Return path to utils directory."""
        return os.path.join(self.data_path, "utils")

    @property
    def trips_path(self) -> str:
        """Return path to trips directory."""
        return os.path.join(self.data_path, "trips")


# Default configuration instance
DEFAULT_CONFIG = PreprocessingConfig()