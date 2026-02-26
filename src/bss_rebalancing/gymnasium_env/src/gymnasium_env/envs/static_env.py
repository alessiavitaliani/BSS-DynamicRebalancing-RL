"""
Static Bike-Sharing System Environment for Reinforcement Learning.

This module implements a Gymnasium environment for simulating static bike
rebalancing in a bike-sharing system. The environment simulates bike trips
and periodic system-wide rebalancing operations.

Author: Edoardo Scarpel
"""

import bisect
import pickle
import random
from typing import Optional

import gymnasium as gym
import numpy as np
import osmnx as ox
import pandas as pd
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import event_handler, simulate_environment
from gymnasium_env.simulator.utils import (
    initialize_bikes,
    initialize_graph,
    initialize_stations,
    truncated_gaussian,
)
from gymnasium_env.simulator.truck_simulator import tsp_rebalancing


# =============================================================================
# Module-Level Constants
# =============================================================================

class FilePaths:
    """File path constants for data loading."""

    GRAPH_FILE = 'utils/cambridge_network.graphml'
    CELL_FILE = 'utils/cell_data.pkl'
    DISTANCE_MATRIX_FILE = 'utils/distance_matrix.csv'
    MATRICES_FOLDER = 'matrices/09-10'
    RATES_FOLDER = 'rates/09-10'
    TRIPS_FOLDER = 'trips/'
    NEARBY_NODES_FILE = 'utils/nearby_nodes.pkl'
    VELOCITY_MATRIX_FILE = 'utils/ev_velocity_matrix.csv'
    CONSUMPTION_MATRIX_FILE = 'utils/ev_consumption_matrix.csv'
    GLOBAL_RATES_FILE = 'utils/global_rates.pkl'


# Day mappings
DAYS_TO_NUM = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6
}

NUM_TO_DAYS = {
    0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday',
    4: 'friday', 5: 'saturday', 6: 'sunday'
}


# =============================================================================
# Environment Constants
# =============================================================================

class EnvDefaults:
    """Default configuration values for the static environment."""

    # Time parameters
    TIMESLOT_DURATION_HOURS = 3
    TIMESLOT_DURATION_SECONDS = TIMESLOT_DURATION_HOURS * 3600
    STEP_DURATION_SECONDS = 30
    TIMESLOTS_PER_DAY = 8

    # Default starting conditions
    DEFAULT_DAY = 'monday'
    DEFAULT_TIMESLOT = 0
    DEFAULT_TOTAL_TIMESLOTS = 56
    DEFAULT_DEPOT_ID = 491

    # Bike fleet parameters
    DEFAULT_BIKES_PER_CELL = 5

    # Rebalancing parameters
    DEFAULT_NUM_REBALANCING_EVENTS = 1


# =============================================================================
# Main Environment Class
# =============================================================================

class StaticEnv(gym.Env):
    """
    A Gymnasium environment for bike-sharing system static rebalancing.

    This environment simulates a bike-sharing network where periodic
    system-wide rebalancing occurs at fixed intervals. Unlike the dynamic
    environment, there is no continuous truck control - instead, the system
    is rebalanced using a TSP-based approach at scheduled times.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, data_path: str):
        """
        Initialize the static bike-sharing environment.

        Args:
            data_path: Path to directory containing network data, PMF matrices, etc.
        """
        super().__init__()

        # Store path
        self._data_path = data_path

        # Load and initialize network components
        self._initialize_network()

        # Load precomputed data
        self._load_precomputed_data()

        # Initialize state variables to None/default values
        self._initialize_state_variables()

    def _initialize_network(self):
        """Load and initialize the street network and geographic data."""
        # Load the OSMnx graph
        self._graph = initialize_graph(self._data_path + FilePaths.GRAPH_FILE)

    def _load_precomputed_data(self):
        """Load precomputed data files from disk."""
        # Load nearby nodes dictionary
        with open(self._data_path + FilePaths.NEARBY_NODES_FILE, 'rb') as file:
            self._nearby_nodes_dict = pickle.load(file)

        # Load cell definitions
        with open(self._data_path + FilePaths.CELL_FILE, 'rb') as file:
            self._cells = pickle.load(file)

        # Load distance matrix
        self._distance_matrix = pd.read_csv(
            self._data_path + FilePaths.DISTANCE_MATRIX_FILE,
            index_col='osmid'
        )
        self._distance_matrix.index = self._distance_matrix.index.astype(int)
        self._distance_matrix.columns = self._distance_matrix.columns.astype(int)

        # Load velocity and consumption matrices
        self._velocity_matrix = pd.read_csv(
            self._data_path + FilePaths.VELOCITY_MATRIX_FILE,
            index_col='hour'
        )
        self._consumption_matrix = pd.read_csv(
            self._data_path + FilePaths.CONSUMPTION_MATRIX_FILE,
            index_col='hour'
        )

    def _initialize_state_variables(self):
        """Initialize all simulation state variables to default/None values."""
        # PMF and demand data
        self._global_rate_dict = None

        # Bike storage
        self._system_bikes = None
        self._outside_system_bikes = None
        self._depot = None
        self._depot_node = None

        # Fleet parameters
        self._maximum_number_of_bikes = 0
        self._fixed_rebal_bikes_per_cell = EnvDefaults.DEFAULT_BIKES_PER_CELL

        # Station objects
        self._stations = None

        # Event buffer for simulation
        self._event_buffer = []

        # Time tracking
        self._env_time = 0
        self._day = EnvDefaults.DEFAULT_DAY
        self._timeslot = EnvDefaults.DEFAULT_TIMESLOT
        self._total_timeslots = EnvDefaults.DEFAULT_TOTAL_TIMESLOTS

        # Rebalancing configuration
        self._num_rebalancing_events = EnvDefaults.DEFAULT_NUM_REBALANCING_EVENTS
        self._rebalancing_hours = []

        # Counters
        self._timeslots_completed = 0
        self._days_completed = 0

        # Bike ID generator
        self._next_bike_id = 0

    # -------------------------------------------------------------------------
    # Environment Reset
    # -------------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Dictionary of reset options:
                - day (str): Starting day of week
                - timeslot (int): Starting timeslot (0-7)
                - total_timeslots (int): Total timeslots to simulate
                - maximum_number_of_bikes (int): Fleet size
                - fixed_rebal_bikes_per_cell (int): Base bikes per cell during rebalancing
                - num_rebalancing_events (int): Number of rebalancing operations per day
                - depot_id (int): Depot cell ID

        Returns:
            Tuple of (empty_observation_dict, empty_info_dict)
        """
        super().reset(seed=seed)

        # Parse options with defaults
        options = options or {}
        self._apply_reset_options(options)

        # Reset all cells
        for cell in self._cells.values():
            cell.reset()

        # Initialize depot and bike fleet
        depot_id = options.get('depot_id', EnvDefaults.DEFAULT_DEPOT_ID)
        self._initialize_depot(depot_id)

        # Reset counters
        self._reset_counters()

        # Clear event buffer
        self._event_buffer = []

        # Create stations
        self._create_stations()

        # Assign cells to stations
        self._assign_cells_to_stations()

        # Initialize day/timeslot and generate events
        self._initialize_day()

        # Distribute bikes based on predicted net flow
        self._distribute_bikes_to_stations()

        return {}, {}

    def _apply_reset_options(self, options: dict):
        """Apply reset options to environment state."""
        # Time configuration
        self._day = options.get('day', EnvDefaults.DEFAULT_DAY)
        self._timeslot = options.get('timeslot', EnvDefaults.DEFAULT_TIMESLOT)
        self._total_timeslots = options.get(
            'total_timeslots',
            EnvDefaults.DEFAULT_TOTAL_TIMESLOTS
        )

        # Fleet configuration
        self._maximum_number_of_bikes = options.get(
            'maximum_number_of_bikes',
            self._maximum_number_of_bikes
        )
        self._fixed_rebal_bikes_per_cell = options.get(
            'fixed_rebal_bikes_per_cell',
            EnvDefaults.DEFAULT_BIKES_PER_CELL
        )

        # Rebalancing configuration
        self._num_rebalancing_events = options.get(
            'num_rebalancing_events',
            EnvDefaults.DEFAULT_NUM_REBALANCING_EVENTS
        )

        # Set rebalancing hours
        if self._num_rebalancing_events > 0:
            interval = 24 // self._num_rebalancing_events
            self._rebalancing_hours = [
                (i + 3) % 24 for i in range(0, 24, interval)
            ]
            self._rebalancing_hours = sorted(self._rebalancing_hours)

    def _initialize_depot(self, depot_id: int):
        """Initialize the depot with bikes."""
        self._next_bike_id = 0
        self._depot_node = self._cells.get(depot_id).get_center_node()
        self._depot, self._next_bike_id = initialize_bikes(
            n=self._maximum_number_of_bikes,
            next_bike_id=self._next_bike_id
        )

    def _reset_counters(self):
        """Reset all simulation counters to initial values."""
        self._timeslot = 0
        self._timeslots_completed = 0
        self._days_completed = 0

    def _create_stations(self):
        """Create Station objects for all nodes in the network."""
        from gymnasium_env.simulator.station import Station

        gdf_nodes = ox.graph_to_gdfs(self._graph, edges=False)
        stations = {}

        for index, row in gdf_nodes.iterrows():
            station = Station(index, row['y'], row['x'])    # type: ignore
            stations[index] = station

        # Add the "out of system" virtual station
        stations[10000] = Station(10000, 0, 0)

        self._stations = stations

    def _assign_cells_to_stations(self):
        """Assign each station to its containing cell."""
        # Set cell for each station based on cell node membership
        for cell in self._cells.values():
            for node in cell.nodes:
                self._stations[node].set_cell(cell)

        # Validate all stations have cells assigned
        for station in self._stations.values():
            station_id = station.get_station_id()
            if station.get_cell() is None and station_id != 10000:
                raise ValueError(
                    f"Station {station} is not assigned to a cell."
                )

    def _distribute_bikes_to_stations(self):
        """Distribute bikes to stations based on predicted net flow."""
        # Compute net flow per cell
        net_flow_per_cell = self._compute_net_flow()

        # Base allocation: fixed bikes per cell
        bikes_per_cell = {
            cell_id: self._fixed_rebal_bikes_per_cell
            for cell_id in self._cells.keys()
        }

        # Calculate remaining bikes to distribute
        remaining_bikes = (
            self._maximum_number_of_bikes
            - self._fixed_rebal_bikes_per_cell * len(self._cells)
        )

        # Distribute remaining bikes based on negative flow (high demand areas)
        total_negative_flow = sum(
            flow for flow in net_flow_per_cell.values() if flow < 0
        )

        bikes_positioned = 0
        for cell_id, flow in net_flow_per_cell.items():
            if flow < 0:
                proportional_bikes = int(
                    (flow / total_negative_flow) * remaining_bikes
                )
                bikes_per_cell[cell_id] += proportional_bikes
                bikes_positioned += proportional_bikes

        # Distribute any remaining bikes randomly to high-demand cells
        if bikes_positioned < remaining_bikes:
            high_demand_cells = [
                cell_id for cell_id, flow in net_flow_per_cell.items()
                if flow < 0
            ]
            random.shuffle(high_demand_cells)

            for cell_id in high_demand_cells:
                bikes_per_cell[cell_id] += 1
                bikes_positioned += 1
                if bikes_positioned == remaining_bikes:
                    break

        # Convert cell-based distribution to station-based
        bikes_per_station = {stn_id: 0 for stn_id in self._stations.keys()}
        for cell_id, num_of_bikes in bikes_per_cell.items():
            center_node = self._cells[cell_id].get_center_node()
            bikes_per_station[center_node] = num_of_bikes

        # Initialize system bikes at stations
        self._system_bikes, self._outside_system_bikes, self._next_bike_id = (
            initialize_stations(
                stations=self._stations,
                depot=self._depot,
                bikes_per_station=bikes_per_station,
                next_bike_id=self._next_bike_id,
            )
        )

    # -------------------------------------------------------------------------
    # Environment Step
    # -------------------------------------------------------------------------

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step in the environment (simulate one full timeslot).

        Args:
            action: Unused (static environment has no actions).

        Returns:
            Tuple of (observation, reward, done, terminated, info)
        """
        terminated = False
        total_failures = 0
        failures_per_timeslot = []
        rebalance_time = []

        # Process events until timeslot ends
        while not terminated:
            self._env_time += EnvDefaults.STEP_DURATION_SECONDS

            # Process all events that occurred before current time
            while self._event_buffer and self._event_buffer[0].time < self._env_time:
                event = self._event_buffer.pop(0)
                failure, self._next_bike_id = event_handler(
                    event=event,
                    station_dict=self._stations,
                    nearby_nodes_dict=self._nearby_nodes_dict,
                    distance_matrix=self._distance_matrix,
                    system_bikes=self._system_bikes,
                    outside_system_bikes=self._outside_system_bikes,
                    next_bike_id=self._next_bike_id
                )
                total_failures += failure

            # Check for rebalancing events at hour boundaries
            if self._env_time % 3600 == 0:
                hour = ((self._env_time // 3600) + 1) % 24
                if hour in self._rebalancing_hours:
                    time_to_rebalance = self._perform_rebalancing()
                    rebalance_time.append(time_to_rebalance)

            # Check for timeslot termination
            timeslot_duration = EnvDefaults.TIMESLOT_DURATION_SECONDS
            if self._env_time >= timeslot_duration * (self._timeslot + 1):
                self._timeslot = (self._timeslot + 1) % EnvDefaults.TIMESLOTS_PER_DAY
                failures_per_timeslot.append(total_failures)
                total_failures = 0

                # Check for day rollover
                if self._timeslot == 0:
                    self._day = NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7]
                    self._days_completed += 1
                    self._initialize_day()

                terminated = True
                self._timeslots_completed += 1

        # Build info dictionary
        info = {
            'failures': failures_per_timeslot,
            'time': self._env_time + (self._timeslot * 3 + 1) * 3600,
            'day': self._day,
            'week': int(self._days_completed // 7),
            'rebalance_time': rebalance_time
        }

        # Check if episode is complete
        done = self._timeslots_completed == self._total_timeslots

        return {}, 0, done, terminated, info

    # -------------------------------------------------------------------------
    # Rebalancing Operations
    # -------------------------------------------------------------------------

    def _perform_rebalancing(self) -> int:
        """
        Perform system-wide rebalancing operation.

        Returns:
            Time required for rebalancing in seconds.
        """
        # Try rebalancing with decreasing bike count if necessary
        num_bikes_per_cell = self._fixed_rebal_bikes_per_cell
        time_to_rebalance = -1

        while time_to_rebalance == -1:
            try:
                time_to_rebalance = self._rebalance_system(num_bikes_per_cell)
            except ValueError:
                num_bikes_per_cell -= 1
                if num_bikes_per_cell < 0:
                    raise ValueError(
                        "Unable to perform rebalancing: insufficient bikes"
                    )

        return time_to_rebalance

    def _rebalance_system(self, num_bikes_per_cell: int) -> int:
        """
        Rebalance the system to target distribution.

        Args:
            num_bikes_per_cell: Base number of bikes per cell.

        Returns:
            Time required for rebalancing in seconds.

        Raises:
            ValueError: If insufficient bikes are available.
        """
        # Add bikes back to the system from outside pool
        while len(self._system_bikes) < self._maximum_number_of_bikes:
            bike_id = next(iter(self._outside_system_bikes))
            bike = self._outside_system_bikes.pop(bike_id)
            self._system_bikes[bike.get_bike_id()] = bike

        # Compute net flow and target distribution
        net_flow_per_cell = self._compute_net_flow()
        bikes_per_cell = {
            cell_id: num_bikes_per_cell
            for cell_id in self._cells.keys()
        }

        # Calculate available bikes
        available_bikes = sum(
            1 for bike in self._system_bikes.values() if bike.available
        )
        remaining_bikes = available_bikes - num_bikes_per_cell * len(self._cells)

        # Check if there are sufficient bikes
        if remaining_bikes < 0:
            raise ValueError(
                "Low on bikes! The maximum number of bikes is too low to "
                "fulfill minimum bike rebalancing."
            )

        # Distribute remaining bikes based on negative flow
        total_negative_flow = sum(
            flow for flow in net_flow_per_cell.values() if flow < 0
        )

        used_bikes = 0
        for cell_id, flow in net_flow_per_cell.items():
            if flow < 0:
                proportional_bikes = int(
                    (flow / total_negative_flow) * remaining_bikes
                )
                bikes_per_cell[cell_id] += proportional_bikes
                used_bikes += proportional_bikes

        # Assign remaining bikes to high-demand cells randomly
        if used_bikes < remaining_bikes:
            high_demand_cells = [
                cell_id for cell_id, flow in net_flow_per_cell.items()
                if flow < 0
            ]
            random.shuffle(high_demand_cells)

            for cell_id in high_demand_cells:
                bikes_per_cell[cell_id] += 1
                used_bikes += 1
                if used_bikes == remaining_bikes:
                    break

        # Compute rebalancing time using TSP
        time_to_rebalance = self._compute_rebalancing_time(bikes_per_cell)

        # Execute rebalancing: empty all stations
        for station in self._stations.values():
            if station.get_station_id() != 10000:
                while station.get_number_of_bikes() > 0:
                    bike = station.unlock_bike()
                    bike.set_availability(True)

        # Redistribute bikes according to plan
        available_bikes_dict = {
            bike_id: bike
            for bike_id, bike in self._system_bikes.items()
            if bike.available
        }

        for cell_id, num_of_bikes in bikes_per_cell.items():
            center_node_id = self._cells[cell_id].get_center_node()
            for _ in range(num_of_bikes):
                bike_id = next(iter(available_bikes_dict))
                bike = available_bikes_dict.pop(bike_id)
                self._stations[center_node_id].lock_bike(bike)

        # Charge all bikes
        for bike in self._system_bikes.values():
            bike.set_battery(bike.get_max_battery())

        return time_to_rebalance

    def _compute_rebalancing_time(self, bikes_per_cell: dict) -> int:
        """
        Compute time required for rebalancing using TSP.

        Args:
            bikes_per_cell: Target bike distribution.

        Returns:
            Rebalancing time in seconds.
        """
        # Identify surplus and deficit nodes
        surplus_nodes = {}
        deficit_nodes = {}

        for cell_id, cell in self._cells.items():
            cell_bikes = cell.get_total_bikes()
            target_bikes = bikes_per_cell[cell_id]

            if cell_bikes > target_bikes:
                surplus_nodes[cell.get_center_node()] = cell_bikes - target_bikes
            elif cell_bikes < target_bikes:
                deficit_nodes[cell.get_center_node()] = target_bikes - cell_bikes

        # Compute TSP distance
        distance, _ = tsp_rebalancing(
            surplus_nodes,
            deficit_nodes,
            self._depot_node,
            self._distance_matrix
        )

        # Compute time based on distance and velocity
        hour = ((self._env_time // 3600) + 1) % 24
        mean_truck_velocity = self._velocity_matrix.loc[hour, self._day]
        velocity_kmh = truncated_gaussian(10, 70, mean_truck_velocity, 5)
        time_seconds = int(distance * 3.6 / velocity_kmh)

        return time_seconds

    # -------------------------------------------------------------------------
    # Event Simulation
    # -------------------------------------------------------------------------

    def _initialize_day(self):
        """Initialize simulation for the current day (all 8 timeslots)."""
        total_events = []

        # Generate events for all timeslots of the day
        for timeslot in range(EnvDefaults.TIMESLOTS_PER_DAY):
            # Load PMF matrix and global rate
            pmf_matrix, global_rate = self._load_pmf_matrix_global_rate(
                self._day,
                timeslot
            )

            # Flatten PMF matrix for event simulation
            flattened_pmf = self._flatten_pmf_matrix(pmf_matrix)

            # Generate events for this timeslot
            events = simulate_environment(
                duration=EnvDefaults.TIMESLOT_DURATION_SECONDS,
                timeslot=timeslot,
                global_rate=global_rate,
                pmf=flattened_pmf,
                stations=self._stations,
                distance_matrix=self._distance_matrix,
            )

            # Shift event times to correct timeslot
            for event in events:
                event.time += EnvDefaults.TIMESLOT_DURATION_SECONDS * timeslot

            total_events.extend(events)

        # Insert all events into event buffer (maintains sorted order)
        for event in total_events:
            bisect.insort(self._event_buffer, event, key=lambda x: x.time)

        # Reset environment time
        self._env_time = 0

    @staticmethod
    def _flatten_pmf_matrix(pmf_matrix: pd.DataFrame) -> pd.DataFrame:
        """Flatten a PMF matrix for event simulation."""
        values = pmf_matrix.values.flatten()
        ids = [
            (row, col)
            for row in pmf_matrix.index
            for col in pmf_matrix.columns
        ]

        flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
        flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)

        return flattened_pmf

    def _load_pmf_matrix_global_rate(
        self,
        day: str,
        timeslot: int
    ) -> tuple[pd.DataFrame, float]:
        """
        Load the PMF matrix and global rate for a given day and timeslot.

        Args:
            day: Day of week (lowercase).
            timeslot: Timeslot index (0-7).

        Returns:
            Tuple of (pmf_matrix DataFrame, global_rate float).
        """
        # Construct file path
        matrix_path = (
            self._data_path +
            FilePaths.MATRICES_FOLDER +
            '/' +
            str(timeslot).zfill(2) +
            '/'
        )

        pmf_matrix = pd.read_csv(
            matrix_path + day.lower() + '-pmf-matrix.csv',
            index_col='osmid'
        )

        # Convert index and columns to integers
        pmf_matrix.index = pmf_matrix.index.astype(int)
        pmf_matrix.columns = pmf_matrix.columns.astype(int)

        # Add out-of-system entry
        pmf_matrix.loc[10000, 10000] = 0.0

        # Load global rates if not cached
        if self._global_rate_dict is None:
            with open(
                self._data_path + FilePaths.GLOBAL_RATES_FILE, 'rb'
            ) as f:
                self._global_rate_dict = pickle.load(f)

        global_rate = self._global_rate_dict[(day.lower(), timeslot)]

        return pmf_matrix, global_rate

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _compute_net_flow(self, upper_bound: Optional[int] = None) -> dict:
        """
        Compute net flow per cell from the event buffer.

        Args:
            upper_bound: Optional time limit for event consideration.

        Returns:
            Dictionary mapping cell_id to net flow (arrivals - departures).
        """
        net_flow_per_cell = {cell_id: 0 for cell_id in self._cells.keys()}

        for event in self._event_buffer:
            if upper_bound is not None and event.time > upper_bound:
                break

            if event.is_departure():
                station_id = event.trip.get_start_location().get_station_id()
                if station_id != 10000:
                    cell = self._stations[station_id].get_cell()
                    net_flow_per_cell[cell.get_id()] -= 1

            elif event.is_arrival():
                station_id = event.trip.get_end_location().get_station_id()
                if station_id != 10000:
                    cell = self._stations[station_id].get_cell()
                    net_flow_per_cell[cell.get_id()] += 1

        return net_flow_per_cell

    # -------------------------------------------------------------------------
    # Gymnasium Interface Methods
    # -------------------------------------------------------------------------

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
