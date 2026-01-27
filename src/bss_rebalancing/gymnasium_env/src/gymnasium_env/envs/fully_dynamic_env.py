"""
Fully Dynamic Bike-Sharing System Environment for Reinforcement Learning.

This module implements a Gymnasium environment for training RL agents to perform
dynamic bike rebalancing in a bike-sharing system. The environment simulates
bike trips, truck movements, and the state of the entire network.

Author: Edoardo Scarpel
"""

import bisect
import math
import pickle
import threading
from typing import Optional

import gymnasium as gym
import numpy as np
import osmnx as ox
import pandas as pd
import torch.nn as nn
from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import event_handler, simulate_environment
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.truck_simulator import (
    charge_bike,
    drop_bike,
    move_down,
    move_left,
    move_right,
    move_up,
    pick_up_bike,
    stay,
)
from gymnasium_env.simulator.utils import (
    ACTION_TO_STR,
    DAYS_TO_NUM,
    DEFAULT_PATHS,
    NUM_TO_DAYS,
    Actions,
    Logger,
    convert_seconds_to_hours_minutes,
    detect_self_loops,
    initialize_bikes,
    initialize_cells_subgraph,
    initialize_graph,
    initialize_stations,
    logistic_penalty_function,
)

# =============================================================================
# Module-Level Configuration
# =============================================================================

# Enable detailed logging of state and trips (typically disabled for training)
logging_state_and_trips = False


# =============================================================================
# Environment Constants
# =============================================================================

class EnvDefaults:
    """Default configuration values for the environment."""

    # Bike fleet parameters
    MAX_BIKES = 500
    MIN_BIKES_PER_CELL = 1

    # Truck parameters
    MAX_TRUCK_LOAD = 30
    INITIAL_TRUCK_BIKES = 15

    # Time parameters
    TIMESLOT_DURATION_HOURS = 3
    TIMESLOT_DURATION_SECONDS = TIMESLOT_DURATION_HOURS * 3600
    STEP_DURATION_SECONDS = 30
    TIMESLOTS_PER_DAY = 8

    # RL parameters
    DISCOUNT_FACTOR = 0.99
    ELIGIBILITY_DECAY = 0.9968
    BORDER_ELIGIBILITY_DECAY = 0.99

    # Default starting conditions
    DEFAULT_DAY = 'monday'
    DEFAULT_TIMESLOT = 0
    DEFAULT_TOTAL_TIMESLOTS = 56
    DEFAULT_DEPOT_ID = 1


class BorderType:
    """Cell border classification based on number of missing adjacent cells."""
    NORMAL = 0          # All 4 adjacent cells exist
    EDGE = 1            # 1 adjacent cell missing
    CORNER = 2          # 2 adjacent cells missing
    DEAD_END = 3        # 3 adjacent cells missing


class RewardComponents:
    """Reward function component values."""

    # Base step cost
    BASE_COST = -0.1

    # Invalid action penalty
    INVALID_ACTION = -1.0

    # Loop detection penalty
    LOOP_PENALTY = -0.6

    # Drop bike rewards
    DROP_BASE = 0.01
    DROP_REBALANCED_CRITICAL = 2.0
    DROP_IN_CRITICAL = 1.0
    DROP_IN_SURPLUS = -0.5

    # Pick-up rewards
    PICKUP_FROM_CRITICAL = -0.5
    PICKUP_DEBALANCED_CELL = -2.0
    PICKUP_FROM_SURPLUS = 0.2

    # Charge bike rewards
    CHARGE_USELESS_CRITICAL = -0.1
    CHARGE_USELESS_NORMAL = -0.3
    CHARGE_USEFUL_CRITICAL = 0.5
    CHARGE_USEFUL_NORMAL = 0.3
    CHARGE_USEFUL_THRESHOLD = 0.8

    # Eligibility penalties
    ELIGIBILITY_HIGH_THRESHOLD = 0.7
    ELIGIBILITY_LOW_THRESHOLD = 0.2
    ELIGIBILITY_REVISIT_PENALTY = -0.2
    ELIGIBILITY_EXPLORATION_BONUS = 0.2
    ELIGIBILITY_EMPTY_TRUCK_PENALTY = -0.1

    # Stay penalties
    STAY_BASE = -0.1
    STAY_IN_CRITICAL = -1.0
    STAY_NO_CRITIC = 0.3


# =============================================================================
# Main Environment Class
# =============================================================================

class FullyDynamicEnv(gym.Env):
    """
    A Gymnasium environment for bike-sharing system dynamic rebalancing.

    This environment simulates a bike-sharing network where a rebalancing truck
    must distribute bikes across stations to minimize service failures. The agent
    controls the truck's movements and bike pickup/drop operations.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, data_path: str, results_path: str = None):
        """
        Initialize the bike-sharing environment.

        Args:
            data_path: Path to directory containing network data, PMF matrices, etc.
            results_path: Path to directory for storing logs and results.
        """
        super().__init__()

        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Space()

        # Store paths and initialize logger
        self._data_path = data_path
        self._logger = Logger(results_path + 'env_output.log') # TODO: FIX LOGGER FOR GYM ENV

        # Load and initialize network components
        self._initialize_network()

        # Load precomputed data
        self._load_precomputed_data()

        # Define action and observation spaces
        self._define_spaces()

        # Initialize state variables to None/default values
        self._initialize_state_variables()
        self._event_buffer = None
        self._next_event_buffer = None

        # Initialize cell embedding for neural network
        self._encoder = nn.Embedding(len(self._cells), 27) # TODO: CHECK WHY NOT USED
        self._zone_id_to_index = { # TODO: CHECK WHY NOT USED
            zone_id: idx
            for idx, zone_id in enumerate(sorted(self._cells.keys()))
        }

        # Configure logging
        self._logging = False
        self._logger.set_logging(self._logging)

        # Initialize simulation counters
        self._reset_counters()

    def _initialize_network(self):
        """Load and initialize the street network and geographic data."""
        # Load the OSMnx graph
        self._graph = initialize_graph(self._data_path + DEFAULT_PATHS.graph_file)

        # Compute bounding box for coordinate normalization
        nodes, _ = ox.graph_to_gdfs(self._graph)
        self._min_lat, self._max_lat = nodes['y'].min(), nodes['y'].max()
        self._min_lon, self._max_lon = nodes['x'].min(), nodes['x'].max()

        # Create node coordinate lookup dictionary
        nodes_gdf = ox.graph_to_gdfs(self._graph, edges=False)
        self._nodes_dict = {
            node_id: (row['y'], row['x'])
            for node_id, row in nodes_gdf.iterrows()
        }

    def _load_precomputed_data(self):
        """Load precomputed data files from disk."""
        # Load nearby nodes dictionary (for trip assignment to the nearest node with available bike)
        with open(self._data_path + DEFAULT_PATHS.nearby_nodes_file, 'rb') as file:
            self._nearby_nodes_dict = pickle.load(file)

        # Load cell definitions
        with open(self._data_path + DEFAULT_PATHS.cell_file, 'rb') as file:
            self._cells = pickle.load(file)

        # Reset each cell
        for cell in self._cells.values(): cell.reset()

        # Load distance matrix
        self._distance_matrix = pd.read_csv(
            self._data_path + DEFAULT_PATHS.distance_matrix_file,
            index_col='osmid'
        )
        self._distance_matrix.index = self._distance_matrix.index.astype(int)
        self._distance_matrix.columns = self._distance_matrix.columns.astype(int)

        # Load velocity and consumption matrices
        self._velocity_matrix = pd.read_csv(
            self._data_path + DEFAULT_PATHS.velocity_matrix_file,
            index_col='hour'
        )
        self._consumption_matrix = pd.read_csv(
            self._data_path + DEFAULT_PATHS.consumption_matrix_file,
            index_col='hour'
        )

    def _define_spaces(self):
        """Define action and observation spaces."""
        self._action_space = spaces.Discrete(len(Actions))

        # Observation space: time encoding (1 + 7 + 24 = 32 components)
        # Note: actual observation is larger due to cell encodings
        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + 7 + 24,),
            dtype=np.float32
        )

    def _initialize_state_variables(self):
        """Initialize all simulation state variables to default/None values."""
        # PMF and demand data
        self._pmf_matrix = None
        self._global_rate = None
        self._global_rate_dict = None

        # Bike storage
        self._system_bikes = None
        self._outside_system_bikes = None
        self._depot = None
        self._depot_node = None

        # Fleet parameters
        self._maximum_number_of_bikes = EnvDefaults.MAX_BIKES
        self._min_bikes_per_cell = EnvDefaults.MIN_BIKES_PER_CELL

        # Station and truck objects
        self._stations = None
        self._truck = None

        # Event buffers for simulation
        self._next_timeslot_event_buffer = None # TODO: CHECK WHY NOT USED

        # Time tracking
        self._env_time = 0
        self._day = EnvDefaults.DEFAULT_DAY
        self._timeslot = 0
        self._total_timeslots = 0

        # Graph representation for GNN
        self._cell_subgraph = None

        # Bike ID generator
        self._next_bike_id = 0

        # Background computation thread
        self._background_thread = None

        # RL parameters
        self._discount_factor = EnvDefaults.DISCOUNT_FACTOR
        self._eligibility_decay = EnvDefaults.ELIGIBILITY_DECAY
        self._borders_eligibility_decay = EnvDefaults.BORDER_ELIGIBILITY_DECAY
        self._global_critic_score = 0.0
        self._reward_params = None

        # Action tracking
        self._last_move_action = None
        self._invalid_action = False
        self._last_cell_border_type = BorderType.NORMAL

    def _reset_counters(self):
        """Reset all simulation counters to initial values."""
        self._timeslot = 0
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_visits = 1
        self._total_failures = 0
        self._total_trips = 0
        self._total_invalid_actions = 0
        self._last_depleted_bikes = 0
        self._total_low_battery_bikes = 0

    # -------------------------------------------------------------------------
    # Environment Reset
    # -------------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]: # type: ignore
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Dictionary of reset options:
                - logging (bool): Enable detailed logging
                - day (str): Starting day of week
                - timeslot (int): Starting timeslot (0-7)
                - total_timeslots (int): Total timeslots to simulate
                - maximum_number_of_bikes (int): Fleet size
                - min_bikes_per_cell (int): Minimum bikes per cell
                - depot_id (int): Depot cell ID
                - initial_cell (int): Truck starting cell
                - max_truck_load (int): Truck capacity
                - discount_factor (float): RL discount factor
                - reward_params (dict): Custom reward parameters

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        # Parse options with defaults
        options = options or {}
        self._apply_reset_options(options)

        # Reset all cells
        for cell in self._cells.values(): cell.reset()

        # Extract cell and truck configuration from options
        cell_id_list = list(self._cells.keys())
        truck_cell_id = options.get('initial_cell', np.random.choice(cell_id_list))
        max_truck_load = options.get('max_truck_load', EnvDefaults.MAX_TRUCK_LOAD)
        depot_id = options.get('depot_id', EnvDefaults.DEFAULT_DEPOT_ID)

        # Mark initial truck cell as visited
        self._cells[truck_cell_id].set_visits(1)

        # Initialize depot and bike fleet
        self._initialize_depot(depot_id)

        # Reset counters
        self._reset_counters()

        # Clear event buffers
        self._event_buffer = None
        self._next_event_buffer = None

        # Create stations
        self._create_stations()

        # Assign cells to stations
        self._assign_cells_to_stations()

        # Initialize truck
        self._initialize_truck(truck_cell_id, max_truck_load)

        # Initialize day/timeslot and generate events
        self._initialize_day_timeslot()

        # Distribute bikes based on predicted net flow
        self._distribute_bikes_to_stations()

        # Initialize cell subgraph for GNN
        self._initialize_cell_subgraph(options=options)

        # Update graph with initial metrics
        self._update_graph()

        # Build initial observation and info
        observation = self._get_obs()
        info = self._build_reset_info()

        return observation, info

    def _apply_reset_options(self, options: dict):
        """Apply reset options to environment state."""
        # Logging configuration
        self._logging = options.get('logging', False)
        self._logger.set_logging(self._logging)

        # Time configuration
        self._day = options.get('day', EnvDefaults.DEFAULT_DAY)
        self._timeslot = options.get('timeslot', EnvDefaults.DEFAULT_TIMESLOT)
        self._total_timeslots = options.get('total_timeslots', EnvDefaults.DEFAULT_TOTAL_TIMESLOTS)

        # Fleet configuration
        self._maximum_number_of_bikes = options.get(
            'maximum_number_of_bikes',
            self._maximum_number_of_bikes
        )
        self._min_bikes_per_cell = options.get(
            'min_bikes_per_cell',
            self._min_bikes_per_cell
        )

        # RL configuration
        self._discount_factor = options.get('discount_factor', EnvDefaults.DISCOUNT_FACTOR)
        self._reward_params = options.get('reward_params', None)

        # Reset action state
        self._invalid_action = False
        self._global_critic_score = 0.0

    def _initialize_depot(self, depot_id: int):
        """Initialize the depot with bikes."""
        self._next_bike_id = 0
        if depot_id not in self._cells:
            raise ValueError(f"Depot cell ID {depot_id} not found in cells.")
        self._depot_node = self._cells.get(depot_id).get_center_node()
        self._depot, self._next_bike_id = initialize_bikes(
            n=self._maximum_number_of_bikes,
            next_bike_id=self._next_bike_id
        )

    def _create_stations(self):
        """Create Station objects for all nodes in the network."""
        from gymnasium_env.simulator.station import Station

        gdf_nodes = ox.graph_to_gdfs(self._graph, edges=False)
        stations = {}

        for index, row in gdf_nodes.iterrows():
            station = Station(index, row['y'], row['x']) # type: ignore
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
                raise ValueError(f"Station {station} is not assigned to a cell.")

    def _initialize_truck(self, truck_cell_id: int, max_truck_load: int):
        """Initialize the rebalancing truck."""
        cell = self._cells[truck_cell_id]

        # Load initial bikes onto truck from depot
        initial_bike_keys = list(self._depot.keys())[:EnvDefaults.INITIAL_TRUCK_BIKES]
        bikes = {key: self._depot.pop(key) for key in initial_bike_keys}

        self._truck = Truck(
            cell.center_node,
            cell,
            bikes=bikes,
            max_load=max_truck_load
        )

    def _distribute_bikes_to_stations(self):
        """Distribute bikes to stations based on predicted net flow."""
        bikes_per_cell = self._net_flow_based_repositioning()

        # Convert cell-based distribution to station-based
        bikes_per_station = {stn_id: 0 for stn_id in self._stations.keys()}
        for cell_id, num_of_bikes in bikes_per_cell.items():
            center_node = self._cells[cell_id].get_center_node()
            bikes_per_station[center_node] = num_of_bikes

        # Initialize system bikes at stations
        self._system_bikes, self._outside_system_bikes, self._next_bike_id = initialize_stations(
            stations=self._stations,
            depot=self._depot,
            bikes_per_station=bikes_per_station,
            next_bike_id=self._next_bike_id,
        )

    def _initialize_cell_subgraph(self, options: dict):
        """Initialize the cell-level subgraph for GNN processing."""
        self._cell_subgraph = initialize_cells_subgraph(
            self._cells,
            self._nodes_dict, # type: ignore
            self._distance_matrix,
            options.get('cell_subgraph_params', None)
        )

    def _build_reset_info(self) -> dict:
        """Build the info dictionary returned by reset()."""
        return {
            'cells_subgraph': self._cell_subgraph,
            'cell_dict': self._cells,
            'nodes_dict': self._nodes_dict,
            'steps': 0,
            'failures': [],
            'number_of_system_bikes': len(self._system_bikes),
            'distance_matrix': self._distance_matrix,
        }

    # -------------------------------------------------------------------------
    # Environment Step
    # -------------------------------------------------------------------------

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]: # type: ignore
        """
        Execute one step in the environment.

        Args:
            action: Integer action from the action space.

        Returns:
            Tuple of (observation, reward, done, terminated, info)
        """
        # Start new log line
        self._logger.new_log_line(timeslot=(8 * DAYS_TO_NUM[self._day] + self._timeslot))

        # Store pre-action state
        action_cell = self._truck.get_cell()

        # Ensure bike fleet consistency
        self._adjust_depot_system_discrepancy()

        # Execute the action
        elapsed_time, distance = self._execute_action(action)

        # Log action details
        self._logger.log_starting_action(
            ACTION_TO_STR[action],
            elapsed_time,
            action_cell.get_id(),
            self._invalid_action
        )

        # Store pre-update state for reward calculation
        truck_cell = self._truck.get_cell()
        # CHECK ME: should this be action_cell instead of truck_cell?
        old_eligibility_score = truck_cell.eligibility_score
        old_critic_score = truck_cell.get_critic_score()
        old_global_critic_score = self._global_critic_score

        self._logger.log(
            message=f"Cell {truck_cell.get_id()} before update has "
                    f"Eligibility: {old_eligibility_score} Critic: {old_critic_score}"
        )

        if logging_state_and_trips:
            self._log_current_state()

        # Advance simulation
        # Process events during elapsed time
        steps = math.ceil(elapsed_time / EnvDefaults.STEP_DURATION_SECONDS)
        failures = self._jump_to_next_state(steps)

        # Handle bike placement after movement
        self._handle_post_action_bike_placement(action)

        # Log action completion
        self._logger.log_ending_action(
            self._invalid_action,
            time=convert_seconds_to_hours_minutes(
                (self._timeslot * 3 + 1) * 3600 + self._env_time
            )
        )

        # Final state update
        failures.extend(self._jump_to_next_state(steps=1))
        steps += 1
        self._total_failures += sum(failures)

        # Update cell visit statistics
        self._update_visit_statistics(action, truck_cell)

        # Update critic scores across all cells
        self._update_critic_scores()

        # Log truck state
        self._logger.log_truck(self._truck, len(self._depot))

        # Compute outputs
        reward = self._get_reward(
            action,
            old_eligibility_score,
            old_critic_score,
            old_global_critic_score
        )
        observation = self._get_obs(action)
        self._update_graph()

        # Log step results
        self._logger.log(
            message=f"--> Reward = {reward}, System bikes = {len(self._system_bikes)}, "
                    f"Invalid_actions = {self._total_invalid_actions}, "
                    f"Total failures = {self._total_failures}\n"
        )

        # Build info dictionary
        info = self._build_step_info(steps, failures, truck_cell)

        # Update action tracking
        self._update_action_tracking(action)

        # Check for timeslot/episode termination
        terminated, done = self._check_termination()

        # Apply final penalty if episode complete
        if done:
            reward -= (self._total_failures / self._total_timeslots) / 10.0
            self._logger.log_done(
                time=convert_seconds_to_hours_minutes(
                    (self._timeslot * 3 + 1) * 3600 + self._env_time
                )
            )

        return observation, reward, done, terminated, info

    def _execute_action(self, action: int) -> tuple[int, float]:
        """
        Execute the given action and return elapsed time and distance.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (elapsed_time_seconds, distance_traveled)
        """
        self._invalid_action = False

        # Calculate current mean truck velocity
        hours = divmod((self._timeslot * 3 + 1) * 3600 + self._env_time, 3600)[0] % 24
        mean_truck_velocity = self._velocity_matrix.loc[hours, self._day]

        elapsed_time = 0
        distance = 0

        if action == Actions.STAY.value:
            elapsed_time = stay(self._truck)

        elif action == Actions.RIGHT.value:
            elapsed_time, distance, self._invalid_action = move_right(
                self._truck, self._distance_matrix, self._cells, mean_truck_velocity
            )

        elif action == Actions.UP.value:
            elapsed_time, distance, self._invalid_action = move_up(
                self._truck, self._distance_matrix, self._cells, mean_truck_velocity
            )

        elif action == Actions.LEFT.value:
            elapsed_time, distance, self._invalid_action = move_left(
                self._truck, self._distance_matrix, self._cells, mean_truck_velocity
            )

        elif action == Actions.DOWN.value:
            elapsed_time, distance, self._invalid_action = move_down(
                self._truck, self._distance_matrix, self._cells, mean_truck_velocity
            )

        elif action == Actions.DROP_BIKE.value:
            if len(self._system_bikes) < self._maximum_number_of_bikes:
                elapsed_time, distance = drop_bike(
                    self._truck,
                    self._distance_matrix,
                    mean_truck_velocity,
                    self._depot_node,
                    self._depot
                )
            else:
                elapsed_time = 0
                self._invalid_action = True

        elif action == Actions.PICK_UP_BIKE.value:
            elapsed_time, distance, self._invalid_action = pick_up_bike(
                self._truck,
                self._stations,
                self._distance_matrix,
                mean_truck_velocity,
                self._depot_node,
                self._depot,
                self._system_bikes
            )

        elif action == Actions.CHARGE_BIKE.value:
            elapsed_time, distance, self._invalid_action = charge_bike(
                self._truck,
                self._stations,
                self._distance_matrix,
                mean_truck_velocity,
                self._depot_node,
                self._depot,
                self._system_bikes
            )

        return elapsed_time, distance

    def _handle_post_action_bike_placement(self, action: int):
        """Handle bike placement after charge or drop actions."""
        is_placement_action = action in {Actions.CHARGE_BIKE.value, Actions.DROP_BIKE.value}

        if is_placement_action and not self._invalid_action:
            station = self._stations.get(self._truck.get_position())
            bike = self._truck.unload_bike()
            station.lock_bike(bike)
            self._system_bikes[bike.get_bike_id()] = bike

    def _update_visit_statistics(self, action: int, truck_cell):
        """Update cell visit and operation statistics."""
        if not self._invalid_action:
            if action == Actions.PICK_UP_BIKE.value:
                truck_cell.set_ops(truck_cell.get_ops() + 1)
            elif action == Actions.DROP_BIKE.value:
                truck_cell.set_ops(truck_cell.get_ops() - 1)

        truck_cell.set_visits(truck_cell.get_visits() + 1)
        self._total_visits += 1

    def _update_action_tracking(self, action: int):
        """Update action tracking variables after step."""
        if not self._invalid_action:
            self._last_move_action = action
        else:
            self._total_invalid_actions += 1

        self._last_cell_border_type = sum(self._get_borders())

    def _check_termination(self) -> tuple[bool, bool]:
        """
        Check if timeslot or episode has ended.

        Returns:
            Tuple of (terminated, done) flags.
        """
        terminated = False

        # Check if current timeslot is complete
        if self._env_time >= EnvDefaults.TIMESLOT_DURATION_SECONDS:
            # Advance to next timeslot
            self._timeslot = (self._timeslot + 1) % EnvDefaults.TIMESLOTS_PER_DAY

            # Advance day if needed
            if self._timeslot == 0:
                self._day = NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7]
                self._days_completed += 1

            # Adjust event times for new timeslot
            for event in self._event_buffer:
                event.time -= EnvDefaults.TIMESLOT_DURATION_SECONDS

            # Preserve remaining time into next timeslot
            env_time_diff = self._env_time - EnvDefaults.TIMESLOT_DURATION_SECONDS
            self._initialize_day_timeslot()
            self._env_time = env_time_diff

            self._timeslots_completed += 1
            terminated = True

            # Log termination
            self._logger.log_terminated(
                time=convert_seconds_to_hours_minutes(
                    (self._timeslot * 3 + 1) * 3600 + self._env_time
                )
            )

        # Check if episode is complete
        done = self._timeslots_completed == self._total_timeslots

        return terminated, done

    def _build_step_info(self, steps: int, failures: list, truck_cell) -> dict:
        """Build the info dictionary returned by step()."""
        return {
            'cells_subgraph': self._cell_subgraph,
            'time': self._env_time + (self._timeslot * 3 + 1) * 3600,
            'day': self._day,
            'week': int(self._days_completed // 7),
            'year': int(self._days_completed // 365),
            'failures': failures,
            'number_of_system_bikes': len(self._system_bikes),
            'steps': steps,
            'global_critic_score':self._global_critic_score,
            'total_trips': self._total_trips,
            'total_invalid':self._total_invalid_actions,
        }

    def _log_current_state(self):
        """Log the current state for debugging."""
        self._logger.log_state(
            step=int(self._env_time / EnvDefaults.STEP_DURATION_SECONDS),
            time=convert_seconds_to_hours_minutes(
                (self._timeslot * 3 + 1) * 3600 + self._env_time
            )
        )

    # -------------------------------------------------------------------------
    # Gymnasium Interface Methods
    # -------------------------------------------------------------------------

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Clean up resources."""
        if self._background_thread is not None:
            self._background_thread.join()

    # -------------------------------------------------------------------------
    # Event Simulation
    # -------------------------------------------------------------------------

    def _precompute_poisson_events(self):
        """Background thread for precomputing Poisson events for future timeslot."""
        # Calculate target timeslot (2 slots ahead)
        timeslot = (self._timeslot + 2) % EnvDefaults.TIMESLOTS_PER_DAY
        day = NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7] if timeslot == 0 else self._day

        # Load demand data
        pmf_matrix, global_rate = self._load_pmf_matrix_global_rate(day, timeslot)

        # Flatten PMF matrix for event simulation
        flattened_pmf = self._flatten_pmf_matrix(pmf_matrix)

        # Generate events
        self._next_event_buffer = simulate_environment(
            duration=EnvDefaults.TIMESLOT_DURATION_SECONDS,
            timeslot=timeslot,
            global_rate=global_rate,
            pmf=flattened_pmf,
            stations=self._stations,
            distance_matrix=self._distance_matrix,
        )

        # Shift event times to correct timeslot
        for event in self._next_event_buffer:
            event.time += EnvDefaults.TIMESLOT_DURATION_SECONDS

    def _initialize_day_timeslot(self):
        """Initialize simulation for the current day and timeslot."""
        # Load PMF matrix and global rate
        self._pmf_matrix, self._global_rate = self._load_pmf_matrix_global_rate(
            self._day,
            self._timeslot
        )

        # Update station demand/arrival rates
        for stn_id, stn in self._stations.items():
            stn.set_request_rate(self._pmf_matrix.loc[stn_id, :].sum() * self._global_rate)
            stn.set_arrival_rate(self._pmf_matrix.loc[:, stn_id].sum() * self._global_rate)

        # Update cell request rates
        for cell in self._cells.values():
            total_request_rate = sum(
                self._stations[node].get_request_rate()
                for node in cell.get_nodes()
            )
            cell.set_request_rate(total_request_rate)

        # Generate events for current timeslot if needed
        if self._event_buffer is None:
            flattened_pmf = self._flatten_pmf_matrix(self._pmf_matrix)
            self._event_buffer = simulate_environment(
                duration=EnvDefaults.TIMESLOT_DURATION_SECONDS,
                timeslot=self._timeslot,
                global_rate=self._global_rate,
                pmf=flattened_pmf,
                stations=self._stations,
                distance_matrix=self._distance_matrix,
            )

        # Generate events for next timeslot if needed
        if self._next_event_buffer is None:
            next_timeslot = (self._timeslot + 1) % EnvDefaults.TIMESLOTS_PER_DAY
            next_day = NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7] if next_timeslot == 0 else self._day
            next_pmf_matrix, next_global_rate = self._load_pmf_matrix_global_rate(
                next_day,
                next_timeslot
            )

            flattened_pmf = self._flatten_pmf_matrix(next_pmf_matrix)
            self._next_event_buffer = simulate_environment(
                duration=EnvDefaults.TIMESLOT_DURATION_SECONDS,
                timeslot=next_timeslot,
                global_rate=next_global_rate,
                pmf=flattened_pmf,
                stations=self._stations,
                distance_matrix=self._distance_matrix,
            )

            # Shift times to next slot
            for event in self._next_event_buffer:
                event.time += EnvDefaults.TIMESLOT_DURATION_SECONDS

        # Merge _next_event_buffer into current buffer
        for event in self._next_event_buffer:
            bisect.insort(self._event_buffer, event, key=lambda x: x.time)
        self._next_event_buffer = None

        # Start background thread for precomputing future events
        self._background_thread = threading.Thread(target=self._precompute_poisson_events)
        self._background_thread.start()

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
                DEFAULT_PATHS.matrices_folder +
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
            with open(self._data_path + 'utils/global_rates.pkl', 'rb') as f:
                self._global_rate_dict = pickle.load(f)

        global_rate = self._global_rate_dict[(day.lower(), timeslot)]

        return pmf_matrix, global_rate

    def _jump_to_next_state(self, steps: int = 0) -> list:
        """
        Advance the simulation by the given number of steps.

        Args:
            steps: Number of 30-second steps to advance.

        Returns:
            List of failure counts per step.
        """
        failures = []

        for step in range(steps):
            # Increment environment time
            self._env_time += EnvDefaults.STEP_DURATION_SECONDS

            # Update eligibility scores for all cells
            self._update_eligibility_scores()

            # Process events that occurred during this step
            step_failures = self._process_events()
            failures.append(step_failures)

            # Log state if detailed logging enabled
            if logging_state_and_trips:
                self._log_current_state()

        return failures

    def _update_eligibility_scores(self):
        """Update eligibility scores for all cells based on decay."""
        for cell_id, cell in self._cells.items():
            # Use different decay rate for border cells
            adjacent_cells = cell.get_adjacent_cells()
            has_all_neighbors = all(adj is not None for adj in adjacent_cells.values())

            if has_all_neighbors:
                cell.update_eligibility_score(self._eligibility_decay)
            else:
                cell.update_eligibility_score(self._borders_eligibility_decay)

        # Current truck cell always has maximum eligibility
        self._truck.get_cell().eligibility_score = 1.0

    def _process_events(self) -> int:
        """
        Process all events that occurred before current env_time.

        Returns:
            Number of failures during this step.
        """
        total_step_failures = 0

        while self._event_buffer and self._event_buffer[0].time < self._env_time:
            event = self._event_buffer.pop(0)

            failure, self._next_bike_id = event_handler(
                event=event,
                station_dict=self._stations,
                nearby_nodes_dict=self._nearby_nodes_dict,
                distance_matrix=self._distance_matrix,
                system_bikes=self._system_bikes,
                outside_system_bikes=self._outside_system_bikes,
                logger=self._logger,
                logging_state_and_trips=logging_state_and_trips,
                next_bike_id=self._next_bike_id
            )

            self._total_trips += 1
            total_step_failures += failure

        return total_step_failures

    # -------------------------------------------------------------------------
    # Observation Construction
    # -------------------------------------------------------------------------

    def _get_obs(self, action: int = None) -> np.array: # type: ignore
        """
        Construct the observation vector.

        Args:
            action: Previous action taken (None on reset).

        Returns:
            Numpy array observation.
        """
        # Previous action encoding
        ohe_previous_action = self._encode_previous_action(action)

        # Truck cell position encoding
        ohe_cell_position = self._encode_truck_position()

        # Critical cells encoding
        ohe_cell_critic = self._encode_critical_cells()

        # Border flags
        ohe_borders = self._get_borders()

        # Scalar features
        scalar_features = self._get_scalar_features()

        # Concatenate all features
        observation = np.concatenate([
            scalar_features,
            ohe_previous_action,
            ohe_cell_position,
            ohe_cell_critic,
            ohe_borders,
        ])

        return observation.astype(np.float16)

    def _encode_previous_action(self, action: Optional[int]) -> np.array:   # type: ignore
        """One-hot encode the previous action."""
        if action is None:
            return np.zeros(self._action_space.n, dtype=np.float16)

        return np.array(
            [1.0 if action == a.value else 0.0 for a in Actions],
            dtype=np.float16
        )

    def _encode_truck_position(self) -> np.array:   # type: ignore
        """One-hot encode the truck's current cell position."""
        truck_cell_id = self._truck.get_cell().get_id()
        sorted_cells_keys = sorted(self._cells.keys())

        return np.array(
            [1.0 if truck_cell_id == cell_id else 0.0 for cell_id in sorted_cells_keys],
            dtype=np.float16
        )

    def _encode_critical_cells(self) -> np.array:   # type: ignore
        """Encode which cells are in critical state."""
        return np.array(
            [1.0 if cell.is_critical else 0.0 for cell in self._cells.values()],
            dtype=np.float16
        )

    def _get_scalar_features(self) -> np.array: # type: ignore
        """Get scalar observation features."""
        truck_cell = self._truck.get_cell()

        return np.array([
            self._truck.get_load() + len(self._depot),  # Combined truck + depot load
            1.0 if truck_cell.get_surplus_bikes() > 0 else 0.0,  # Surplus flag
            1.0 if truck_cell.get_total_bikes() == 0 else 0.0,  # Empty cell flag
            self._global_critic_score,  # Global critic score
        ], dtype=np.float16)

    def _get_borders(self) -> np.array: # type: ignore
        """
        Get border flags indicating which adjacent cells are missing.

        Returns:
            Array of 4 flags (one per direction).
        """
        adjacent_cells = self._truck.get_cell().get_adjacent_cells()
        return np.array(
            [1 if adj_cell is None else 0 for adj_cell in adjacent_cells.values()],
            dtype=np.float32
        )

    def _get_truck_position(self) -> tuple[float, float]:
        """
        Get the truck's normalized position coordinates.

        Returns:
            Tuple of (normalized_lat, normalized_lon).
        """
        truck_coords = self._nodes_dict.get(self._truck.get_position())

        normalized_lat = (truck_coords[0] - self._min_lat) / (self._max_lat - self._min_lat)
        normalized_lon = (truck_coords[1] - self._min_lon) / (self._max_lon - self._min_lon)

        return normalized_lat, normalized_lon

    # -------------------------------------------------------------------------
    # Critic Score Computation
    # -------------------------------------------------------------------------

    def _update_critic_scores(self):
        """Update critic scores for all cells based on expected demand."""
        # Compute expected departures per cell
        expected_departures_per_cell = {}
        expected_max_departure = {}
        expected_after_arrival = {}

        # Scan events to compute expected flow
        for event in self._event_buffer:
            if event.time > self._env_time + EnvDefaults.TIMESLOT_DURATION_SECONDS:
                break

            if event.is_departure():
                self._process_departure_for_critic(
                    event,
                    expected_departures_per_cell,
                    expected_max_departure,
                    expected_after_arrival
                )
            elif event.is_arrival():
                self._process_arrival_for_critic(
                    event,
                    expected_departures_per_cell,
                    expected_max_departure,
                    expected_after_arrival
                )

        # Update critic scores for each cell
        self._apply_critic_scores(
            expected_max_departure,
            expected_after_arrival
        )

    @staticmethod
    def _process_departure_for_critic(
            event,
            expected_departures: dict,
            expected_max: dict,
            expected_after: dict
    ):
        """Process a departure event for critic score calculation."""
        start_location = event.get_trip().get_start_location()

        if start_location.get_station_id() != 10000:
            cell = start_location.get_cell()
            cell_id = cell.get_id()
            expected_departures[cell_id] = expected_departures.get(cell_id, 0) + 1

            # Update max departure tracking
            previous_max = expected_max.get(cell_id, 0)
            expected_max[cell_id] = max(previous_max, expected_departures[cell_id])
            expected_after[cell_id] = expected_max[cell_id] - expected_departures[cell_id]

    @staticmethod
    def _process_arrival_for_critic(
            event,
            expected_departures: dict,
            expected_max: dict,
            expected_after: dict
    ):
        """Process an arrival event for critic score calculation."""
        end_location = event.get_trip().get_end_location()

        if end_location.get_station_id() != 10000:
            cell = end_location.get_cell()
            cell_id = cell.get_id()
            expected_departures[cell_id] = expected_departures.get(cell_id, 0) - 1

            # Update max departure tracking
            previous_max = expected_max.get(cell_id, 0)
            expected_max[cell_id] = max(previous_max, expected_departures[cell_id])
            expected_after[cell_id] = expected_max[cell_id] - expected_departures[cell_id]

    def _apply_critic_scores(self, expected_max_departure: dict, expected_after_arrival: dict):
        """Apply computed critic scores to all cells."""
        self._global_critic_score = 0.0

        for cell_id, cell in self._cells.items():
            available = cell.get_total_bikes()
            expected = expected_max_departure.get(cell_id, 0) + self._min_bikes_per_cell
            aft_arrivals = expected_after_arrival.get(cell_id, 0)

            # Compute critic score
            if expected > 0:
                # Formula: -1 + e^[(1 + 0.2*arrivals) * (1 - available/expected)]
                critic_score = -1 + np.exp(
                    (1 + 0.2 * aft_arrivals) * (1 - (available / expected))
                )
            else:
                critic_score = -available

            cell.set_critic_score(critic_score)
            cell.set_surplus_bikes(surplus_threshold=0.67)

            # Apply binary critic score threshold
            if critic_score > 0:
                cell.set_critic_score(1.0)

            # Accumulate global critic score
            if critic_score > 0.0:
                self._global_critic_score += 1

    # -------------------------------------------------------------------------
    # Reward Function
    # -------------------------------------------------------------------------

    def _get_reward(
            self,
            action: int,
            old_eligibility_score: float,
            old_critic_score: float,
            old_global_critic_score: float
    ) -> float:
        """
        Compute the reward for the current step.

        Args:
            action: Action taken this step.
            old_eligibility_score: Eligibility score before action.
            old_critic_score: Critic score before action.
            old_global_critic_score: Global critic score before action.

        Returns:
            Float reward value.
        """
        # Handle invalid actions immediately
        if self._invalid_action:
            return RewardComponents.INVALID_ACTION

        # Compute common state indicators
        loop_detected = detect_self_loops((action, self._last_move_action))
        was_critical = old_critic_score > 0.0
        was_surplus = old_critic_score <= -0.67
        is_critical = self._truck.get_cell().get_critic_score() > 0.0

        # Initialize reward components
        drop_reward = 0.0
        pick_up_reward = 0.0
        bike_charge_reward = 0.0
        eligibility_penalty = 0.0
        stay_penalty = 0.0
        loop_penalty = 0.0

        # Apply loop penalty
        if loop_detected:
            loop_penalty = RewardComponents.LOOP_PENALTY

        # Compute action-specific rewards
        if action == Actions.DROP_BIKE.value:
            drop_reward = self._compute_drop_reward(was_critical, is_critical, was_surplus)

        elif action == Actions.PICK_UP_BIKE.value:
            pick_up_reward = self._compute_pickup_reward(
                was_critical, is_critical, was_surplus, loop_detected
            )

        elif action == Actions.CHARGE_BIKE.value:
            bike_charge_reward = self._compute_charge_reward(was_critical)

        elif action in {Actions.UP.value, Actions.DOWN.value,
                        Actions.LEFT.value, Actions.RIGHT.value}:
            eligibility_penalty = self._compute_movement_penalty(
                old_eligibility_score, was_critical
            )

        elif action == Actions.STAY.value:
            stay_penalty, loop_penalty = self._compute_stay_penalty(
                was_critical, old_global_critic_score, loop_penalty
            )

        # Combine all reward components
        reward = (
                RewardComponents.BASE_COST
                + drop_reward
                + pick_up_reward
                + bike_charge_reward
                + eligibility_penalty
                + stay_penalty
                + loop_penalty
        )

        self._logger.log(
            message=f"split reward {drop_reward} {pick_up_reward} {bike_charge_reward} "
                    f"{eligibility_penalty} {stay_penalty} {loop_penalty}"
        )

        return reward

    def _compute_drop_reward(
            self,
            was_critical: bool,
            is_critical: bool,
            was_surplus: bool
    ) -> float:
        """Compute reward for drop bike action."""
        drop_reward = RewardComponents.DROP_BASE

        if was_critical and not is_critical:
            # Successfully rebalanced a critical cell
            drop_reward = RewardComponents.DROP_REBALANCED_CRITICAL
            self._truck.get_cell().update_rebalanced_times()
            self._reset_eligibility_scores()
        elif was_critical:
            # Dropped in critical cell but still critical
            drop_reward = RewardComponents.DROP_IN_CRITICAL
        elif was_surplus:
            # Dropped in surplus cell (bad)
            drop_reward = RewardComponents.DROP_IN_SURPLUS

        return drop_reward

    @staticmethod
    def _compute_pickup_reward(
            was_critical: bool,
            is_critical: bool,
            was_surplus: bool,
            loop_detected: bool
    ) -> float:
        """Compute reward for pickup bike action."""
        pick_up_reward = 0.0

        if was_critical:
            pick_up_reward = RewardComponents.PICKUP_FROM_CRITICAL
        elif is_critical and not was_critical:
            # Debalanced a cell by picking up
            pick_up_reward = RewardComponents.PICKUP_DEBALANCED_CELL
        elif was_surplus and not loop_detected:
            pick_up_reward = RewardComponents.PICKUP_FROM_SURPLUS

        return pick_up_reward

    def _compute_charge_reward(self, was_critical: bool) -> float:
        """Compute reward for charge bike action."""
        # Check if charge was useful (bike was sufficiently discharged)
        if self._truck.last_charge < RewardComponents.CHARGE_USEFUL_THRESHOLD:
            # Useless charge
            if was_critical:
                return RewardComponents.CHARGE_USELESS_CRITICAL
            else:
                return RewardComponents.CHARGE_USELESS_NORMAL
        else:
            # Useful charge
            if was_critical:
                return RewardComponents.CHARGE_USEFUL_CRITICAL
            else:
                return RewardComponents.CHARGE_USEFUL_NORMAL

    def _compute_movement_penalty(
            self,
            old_eligibility_score: float,
            was_critical: bool
    ) -> float:
        """Compute penalty for movement actions based on eligibility."""
        eligibility_penalty = 0.0

        if old_eligibility_score > RewardComponents.ELIGIBILITY_HIGH_THRESHOLD and not was_critical:
            # Revisiting a recently visited cell
            eligibility_penalty = RewardComponents.ELIGIBILITY_REVISIT_PENALTY
            # Bonus for exiting border cells
            eligibility_penalty += 0.1 * self._last_cell_border_type
        elif old_eligibility_score < RewardComponents.ELIGIBILITY_LOW_THRESHOLD:
            # Exploring unvisited areas
            eligibility_penalty += RewardComponents.ELIGIBILITY_EXPLORATION_BONUS

        # Penalty when truck is empty
        if self._truck.get_load() == 0:
            eligibility_penalty += RewardComponents.ELIGIBILITY_EMPTY_TRUCK_PENALTY

        return eligibility_penalty

    @staticmethod
    def _compute_stay_penalty(
            was_critical: bool,
            old_global_critic_score: float,
            loop_penalty: float
    ) -> tuple[float, float]:
        """Compute penalty for stay action."""
        stay_penalty = RewardComponents.STAY_BASE

        if was_critical:
            stay_penalty = RewardComponents.STAY_IN_CRITICAL
        elif old_global_critic_score <= 0.0:
            stay_penalty = RewardComponents.STAY_NO_CRITIC
            loop_penalty = 0.0  # Override loop penalty

        return stay_penalty, loop_penalty

    def _reset_eligibility_scores(self):
        """Reset eligibility scores after successful rebalancing."""
        for cell in self._cells.values():
            if cell.eligibility_score > 0.3:
                cell.eligibility_score -= 0.2

        self._logger.warning(
            message=f" ---------> {self._truck.get_cell().get_id()} has been rebalanced. "
                    f"All eligibility scores adjusted ###################################"
        )
        self._truck.get_cell().eligibility_score = 1.0

    # -------------------------------------------------------------------------
    # Graph Update
    # -------------------------------------------------------------------------

    def _update_graph(self):
        """Update the cell subgraph with current regional metrics."""
        depleted_bikes = 0

        for cell_id, cell in self._cells.items():
            center_node = cell.get_center_node()

            # Aggregate battery levels and rates
            cell_bikes = cell.get_total_bikes()
            demand_rate = 0.0
            arrival_rate = 0.0
            battery_levels = []

            for node in cell.nodes:
                station = self._stations[node]
                bikes = station.get_bikes()

                # Collect battery levels
                battery_levels.extend(
                    bike.get_battery() / bike.get_max_battery()
                    for bike in bikes.values()
                )

                # Aggregate rates
                demand_rate += station.get_request_rate()
                arrival_rate += station.get_arrival_rate()

            # Normalize rates
            demand_rate /= self._global_rate
            arrival_rate /= self._global_rate

            # Count low battery bikes
            low_battery_bikes = sum(1 for battery in battery_levels if battery <= 0.2)
            if low_battery_bikes > 0:
                depleted_bikes += low_battery_bikes

            # Normalize low battery count
            num_bikes = len(battery_levels)
            low_battery_ratio = low_battery_bikes / num_bikes if num_bikes > 0 else 0

            # Update subgraph node attributes
            if center_node in self._cell_subgraph:
                node_attrs = self._cell_subgraph.nodes[center_node]
                node_attrs['truck_cell'] = 1.0 if cell_id == self._truck.get_cell().get_id() else 0.0
                node_attrs['low_battery_bikes'] = low_battery_ratio
                node_attrs['rebalanced'] = cell.get_total_rebalanced()
                node_attrs['failure_rates'] = (
                    cell.get_failures() / cell.get_total_departures()
                    if cell.get_total_departures() != 0 else 0
                )
                node_attrs['failures'] = cell.get_failures()
                node_attrs['bikes'] = cell_bikes
                node_attrs['critic_score'] = cell.get_critic_score()
                node_attrs['visits'] = cell.get_visits()
                node_attrs['operations'] = cell.get_ops()
                node_attrs['eligibility_score'] = cell.eligibility_score
            else:
                raise ValueError(f"Node {center_node} not found in the subgraph.")

        # Track battery depletion events
        if depleted_bikes > self._last_depleted_bikes:
            self._total_low_battery_bikes += 1
        self._last_depleted_bikes = depleted_bikes

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _compute_bike_per_region_cost(self) -> float:
        """Compute the logistic penalty cost for bike distribution."""
        total_cost = 0.0

        for cell_id, cell in self._cells.items():
            n_bikes = cell.get_total_bikes()
            cost = logistic_penalty_function(M=1, k=1, b=100, x=n_bikes)
            total_cost += cost

        return total_cost

    def _adjust_depot_system_discrepancy(self):
        """
        Ensure the total bike count matches the maximum fleet size.

        Bikes that exit the system are replaced from the outside_system_bikes pool.
        """
        depot_load = len(self._depot)
        system_load = len(self._system_bikes)
        truck_load = self._truck.get_load()
        total_bikes = depot_load + system_load + truck_load

        if total_bikes < self._maximum_number_of_bikes:
            bikes_to_add = self._maximum_number_of_bikes - total_bikes

            for _ in range(bikes_to_add):
                bike_id = next(iter(self._outside_system_bikes))
                bike = self._outside_system_bikes.pop(bike_id)
                bike.reset()
                self._depot[bike.get_bike_id()] = bike

            self._logger.warning(
                message=f" ---------> System has been adjusted to max_number_of_bikes "
                        f"adding {bikes_to_add} bikes"
            )

    def _net_flow_based_repositioning(self, upper_bound: int = None) -> dict:
        """
        Compute initial bike distribution based on predicted net flow.

        Args:
            upper_bound: Optional time limit for event consideration.

        Returns:
            Dictionary mapping cell_id to number of bikes.
        """
        # Initialize net flow tracking
        net_flow_per_cell = {cell_id: 0 for cell_id in self._cells.keys()}

        # Compute net flow from events
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

        # Base allocation: 5 bikes per cell
        base_bikes_per_cell = EnvDefaults.MIN_BIKES_PER_CELL
        bikes_per_cell = {cell_id: base_bikes_per_cell for cell_id in self._cells.keys()}

        # Distribute remaining bikes based on negative flow (high demand areas)
        remaining_bikes = (
                self._maximum_number_of_bikes
                - self._truck.get_load()
                - base_bikes_per_cell * len(self._cells)
        )

        total_negative_flow = sum(
            flow for flow in net_flow_per_cell.values() if flow < 0
        )

        bikes_positioned = 0
        for cell_id, flow in net_flow_per_cell.items():
            if flow < 0:
                proportional_bikes = int((flow / total_negative_flow) * remaining_bikes)
                bikes_per_cell[cell_id] += proportional_bikes
                bikes_positioned += proportional_bikes

        # Distribute any remaining bikes randomly to high-demand cells
        if bikes_positioned < remaining_bikes:
            high_demand_cells = [
                cell_id for cell_id, flow in net_flow_per_cell.items() if flow < 0
            ]
            np.random.shuffle(high_demand_cells)

            for cell_id in high_demand_cells:
                bikes_per_cell[cell_id] += 1
                bikes_positioned += 1
                if bikes_positioned == remaining_bikes:
                    break

        return bikes_per_cell