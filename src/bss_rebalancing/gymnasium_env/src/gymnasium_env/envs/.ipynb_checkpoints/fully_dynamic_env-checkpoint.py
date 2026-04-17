"""
Fully Dynamic Bike-Sharing System Environment for Reinforcement Learning.

This module implements a Gymnasium environment for training RL agents to perform
dynamic bike rebalancing in a bike-sharing system. The environment simulates
bike trips, truck movements, and the state of the entire network.

Author: Edoardo Scarpel
"""

import math
import os.path
import heapq
import logging

import gymnasium as gym
import numpy as np
import osmnx as ox
import polars as pl

from gymnasium import spaces
from gymnasium.utils import seeding
from dataclasses import dataclass, field
from collections import deque
from tqdm import tqdm

from gymnasium_env.simulator import Bike
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import TripSample
from gymnasium_env.simulator.bike_simulator import event_handler, simulate_events, build_events
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.truck_simulator import (
    charge_bike,
    drop_bike,
    move,
    pick_up_bike,
    stay,
    ACTION_TO_DIRECTION
)
from gymnasium_env.simulator.utils import (
    DAYS_TO_NUM,
    DEFAULT_PATHS,
    NUM_TO_DAYS,
    Actions,
    convert_seconds_to_hours_minutes_day,
    detect_self_loops,
    initialize_bikes,
    initialize_graph,
    initialize_stations,
    load_preprocessed_data,
    flatten_pmf_matrix,
    cache_precomputed_buffers,
    load_cached_buffers
)
from gymnasium_env.simulator.env_logger import EnvLogger

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
    MAX_BIKES = 1000
    MIN_BIKES_PER_CELL = 5
    BASE_REPOSITIONING = True
    NET_FLOW_BASED_REPOSITIONING = False

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

    # Precomputation parameters
    PRECOMPUTED_EPISODE_TIMESLOTS = 56  # 7 days × 8 slots

    # Other parameters
    USE_BINARY_CRITIC = True


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
    CHARGE_LOW_BATTERY_THRESHOLD = 0.8

    # Eligibility penalties
    ELIGIBILITY_HIGH_THRESHOLD = 0.7
    ELIGIBILITY_LOW_THRESHOLD = 0.2
    ELIGIBILITY_REVISIT_PENALTY = -0.2
    ELIGIBILITY_EXPLORATION_BONUS = 0.3
    ELIGIBILITY_EMPTY_TRUCK_PENALTY = -0.05

    # Stay penalties
    STAY_BASE = -0.1
    STAY_IN_CRITICAL = -1.0
    STAY_NO_CRITIC = 0.0        # FIX A: REDUCED TO 0.0 FROM 0.3

    # Other
    SURPLUS_THRESHOLD = -0.67
    DEPLOY_WEIGHT = 0.02
    DEPOT_WEIGHT = 0.02


@dataclass
class Depot:
    id: int | None = None
    bikes: dict[int, Bike] = field(default_factory=dict)

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

    def __init__(self, data_path: str, results_path: str = None, seed: int = None, logging_enabled: bool = False):
        """
        Initialize the bike-sharing environment.

        Args:
            data_path: Path to directory containing network data, PMF matrices, etc.
            results_path: Path to directory for storing logs and results.
        """
        super().__init__()

        # Store paths and initialize logger
        self._data_path = data_path
        self._results_path = results_path

        # Env logger: lazy init
        self._env_logger = EnvLogger(name="env")
        self._env_logger.init(
            log_dir=os.path.join(results_path, "logs"),
            filename="env.log",
            level=logging.INFO,
            enabled=logging_enabled,
        )

        # -------------------------------------------------------------------------
        # Load and initialize network components
        # -------------------------------------------------------------------------

        # Load the OSMnx graph
        self._graph = initialize_graph(os.path.join(self._data_path, DEFAULT_PATHS.graph_file))

        # Compute bounding box for coordinate normalization
        nodes_gdf = ox.graph_to_gdfs(self._graph, edges=False)
        self._min_lat, self._max_lat = nodes_gdf['y'].min(), nodes_gdf['y'].max()
        self._min_lon, self._max_lon = nodes_gdf['x'].min(), nodes_gdf['x'].max()
        self._nodes_dict = {nid: (row['y'], row['x']) for nid, row in nodes_gdf.iterrows()}

        # -------------------------------------------------------------------------
        # Load precomputed data
        # -------------------------------------------------------------------------

        (
            self._distance_matrix,
            self._velocity_matrix,
            self._consumption_matrix,
            self._nearby_nodes_dict,
            self._global_rate_dict,
            self._cells,
        ) = load_preprocessed_data(
            DEFAULT_PATHS,
            self._data_path,
        )

        self._distance_lookup = {
            row["node_id"]: row
            for row in self._distance_matrix.iter_rows(named=True)
        }

        self._velocity_lookup = {
            row["hour"]: row
            for row in self._velocity_matrix.iter_rows(named=True)
        }

        self._consumption_lookup = {
            row["hour"]: row
            for row in self._consumption_matrix.iter_rows(named=True)
        }

        self._sorted_cell_ids = sorted(self._cells.keys())

        # -------------------------------------------------------------------------
        # Initialize stations, bikes and assign cells
        # -------------------------------------------------------------------------

        # Build stations once
        stations: dict[int, Station] = {}
        for node_id, (lat, lon) in self._nodes_dict.items():
            stations[node_id] = Station(node_id, lat, lon)      # type: ignore

        # Assign cells to stations once
        for cell in self._cells.values():
            for node in cell.get_nodes():
                stations[node].set_cell(cell)

        all_stations_have_a_cell = all(stn.get_cell() is not None for stn in stations.values())
        assert all_stations_have_a_cell, \
            "Not all stations were assigned a cell. Check cell definitions and station locations."

        # Virtual station 10000
        stations[10000] = Station(10000, 0.0, 0.0)

        self._stations = stations

        # -------------------------------------------------------------------------
        # Define action and observation spaces
        # -------------------------------------------------------------------------

        # Action space
        self.action_space = spaces.Discrete(len(Actions))

        # Observation space
        obs_dim = 17 + 2 * len(self._cells)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float16
        )

        # -------------------------------------------------------------------------
        # Initialize state variables to None/default values
        # -------------------------------------------------------------------------

        # Bike storage
        self._system_bikes = None
        self._outside_system_bikes = None
        self._travelling_bikes = None
        self._depot = Depot()

        # Fleet parameters
        self._maximum_number_of_bikes = EnvDefaults.MAX_BIKES
        self._min_bikes_per_cell = EnvDefaults.MIN_BIKES_PER_CELL
        self._enable_repositioning = EnvDefaults.BASE_REPOSITIONING
        self._use_net_flow = EnvDefaults.NET_FLOW_BASED_REPOSITIONING

        # Station and truck objects
        self._truck = None

        # Time tracking
        self._env_time = 0
        self._day = EnvDefaults.DEFAULT_DAY
        self._timeslot = EnvDefaults.DEFAULT_TIMESLOT
        self._total_timeslots = EnvDefaults.DEFAULT_TOTAL_TIMESLOTS

        # RL parameters
        self._discount_factor = EnvDefaults.DISCOUNT_FACTOR
        self._eligibility_decay = EnvDefaults.ELIGIBILITY_DECAY
        self._borders_eligibility_decay = EnvDefaults.BORDER_ELIGIBILITY_DECAY
        self._global_critic_score = 0.0
        self._use_binary_critic = EnvDefaults.USE_BINARY_CRITIC
        self._reward_params = None

        # Action tracking
        self._last_move_action = None
        self._invalid_action = False
        self._last_cell_border_type = BorderType.NORMAL

        # Event buffers
        self._event_buffer = None

        # -------------------------------------------------------------------------
        # Initialize simulation counters
        # -------------------------------------------------------------------------
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_visits = 1
        self._total_failures = 0
        self._total_invalid_actions = 0

        # -------------------------------------------------------------------------
        # Precompute events for a full episode at initialization for reproducibility and performance
        # -------------------------------------------------------------------------
        self._precomputed_buffers: dict[int, list[TripSample]] = {}
        self._event_fingerprint: str = ""
        self._precompute_full_episode(seed=seed)

    def _precompute_full_episode(self, seed: int = 42) -> None:
        """
        Precompute all events for a full episode (56 timeslots) at init time.
        Stores one list per timeslot in self._precomputed_buffers.
        """
        cache_dir = os.path.join(self._data_path, '.cache')
        cache_file = os.path.join(cache_dir, 'precomputed_buffers.pkl')

        precomputed_buffers: dict[int, list[TripSample]] = load_cached_buffers(cache_file)

        if len(precomputed_buffers) == EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS:
            self._precomputed_buffers = precomputed_buffers
            print("Loaded precomputed buffers from cache.")
            return

        np.random.seed(seed)
        day = EnvDefaults.DEFAULT_DAY
        self._precomputed_buffers = {}

        with tqdm(total=EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS,
                  desc="Precomputing episode", unit="slot",
                  leave=False, dynamic_ncols=True) as pbar:

            for slot_index in range(EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS):
                timeslot = slot_index % EnvDefaults.TIMESLOTS_PER_DAY
                pmf_matrix = self._load_pmf_matrix(day, timeslot)
                global_rate = self._global_rate_dict[(day.lower(), timeslot)]
                flattened_pmf = flatten_pmf_matrix(pmf_matrix)

                self._precomputed_buffers[slot_index] = simulate_events(
                    duration=EnvDefaults.TIMESLOT_DURATION_SECONDS,
                    timeslot=timeslot,
                    global_rate=global_rate,
                    pmf=flattened_pmf,
                    distance_lookup=self._distance_lookup,
                )

                if timeslot == EnvDefaults.TIMESLOTS_PER_DAY - 1:
                    day = NUM_TO_DAYS[(DAYS_TO_NUM[day] + 1) % 7]

                pbar.set_postfix(day=day, slot=timeslot)
                pbar.update(1)

        cache_precomputed_buffers(self._precomputed_buffers, cache_file)

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

        # Optional: per-episode log directory
        episode_results_path = options.get("results_path")
        if episode_results_path is not None:
            self._update_logging_path(episode_results_path)
        else:
            self._env_logger.set_enabled(False)

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
            'minimum_number_of_bikes',
            self._min_bikes_per_cell
        )

        # RL configuration
        self._discount_factor = options.get('discount_factor', EnvDefaults.DISCOUNT_FACTOR)
        self._reward_params = options.get('reward_params', None)

        # Reset action state
        self._invalid_action = False
        self._global_critic_score = 0.0
        self._use_binary_critic = options.get(
            'use_binary_critic', EnvDefaults.USE_BINARY_CRITIC
        )

        # Reset all cells
        for cell in self._cells.values(): cell.reset()

        # Reset all stations
        for stn in self._stations.values(): stn.reset()

        # Extract cell and truck configuration from options
        cell_id_list = list(self._cells.keys())
        truck_cell_id = options.get('initial_cell', np.random.choice(cell_id_list))
        max_truck_load = options.get('max_truck_load', EnvDefaults.MAX_TRUCK_LOAD)
        depot_id = options.get('depot_id', EnvDefaults.DEFAULT_DEPOT_ID)
        self._enable_repositioning = options.get('enable_repositioning', EnvDefaults.BASE_REPOSITIONING)
        self._use_net_flow = options.get('use_net_flow', EnvDefaults.NET_FLOW_BASED_REPOSITIONING)

        # Mark initial truck cell as visited
        self._cells[truck_cell_id].set_visits(1)

        # Initialize depot and bike fleet
        if depot_id not in self._cells:
            raise ValueError(f"Depot cell ID {depot_id} not found in cells.")
        self._depot.id= self._cells.get(depot_id).get_center_node()
        self._depot.bikes = initialize_bikes(n=self._maximum_number_of_bikes)
        self._travelling_bikes = {}

        # -------------------------------------------------------------------------
        # Reset counters
        # -------------------------------------------------------------------------
        self._timeslots_completed = 0
        self._days_completed = 0
        self._total_visits = 1
        self._total_failures = 0
        self._total_invalid_actions = 0

        # Clear event buffers
        self._event_buffer = None

        # -------------------------------------------------------------------------
        # Initialize truck
        # -------------------------------------------------------------------------
        # Load initial bikes onto truck from depot
        initial_bike_keys = list(self._depot.bikes.keys())[:EnvDefaults.INITIAL_TRUCK_BIKES]
        bikes = {key: self._depot.bikes.pop(key) for key in initial_bike_keys}

        truck_cell = self._cells[truck_cell_id]
        self._truck = Truck(
            position=truck_cell.get_center_node(),
            cell=truck_cell,
            bikes=bikes,
            max_load=max_truck_load
        )

        # -------------------------------------------------------------------------
        # Initialize day/timeslot and generate events
        # -------------------------------------------------------------------------
        self._initialize_day_timeslot()

        # -------------------------------------------------------------------------
        # Distribute bikes based on predicted net flow if enabled, otherwise use base repositioning
        # -------------------------------------------------------------------------
        bikes_per_cell = self._bike_repositioning(
            enable_repositioning=self._enable_repositioning,
            use_net_flow=self._use_net_flow,
        )

        # Convert cell-based distribution to station-based
        bikes_per_station = {stn_id: 0 for stn_id in self._stations.keys()}
        for cell_id, num_of_bikes in bikes_per_cell.items():
            center_node = self._cells[cell_id].get_center_node()
            bikes_per_station[center_node] = num_of_bikes

        # Initialize system bikes at stations
        self._system_bikes, self._outside_system_bikes = initialize_stations(
            stations=self._stations,
            depot_bikes=self._depot.bikes,
            bikes_per_station=bikes_per_station
        )

        # -------------------------------------------------------------------------
        # Update graph with initial metrics
        # -------------------------------------------------------------------------
        self._update_cells_metrics()

        # -------------------------------------------------------------------------
        # Build initial observation and info
        # -------------------------------------------------------------------------
        observation = self._get_obs()
        info = {
            'cell_dict': self._cells,
            'nodes_dict': self._nodes_dict,
            'steps': 0,
            'failures': [],
            'depot_bikes': len(self._depot.bikes),
            'truck_bikes': self._truck.get_load(),
            'number_of_system_bikes': len(self._system_bikes),
            'number_of_outside_bikes': len(self._outside_system_bikes),
            'distance_lookup': self._distance_lookup,
        }

        # Log number of bikes in the system, in the truck, in the depot and in the outside system
        self._env_logger.info(
            f"Initial bikes -> "
            f"system={len(self._system_bikes)}, "
            f"truck={self._truck.get_load()}, "
            f"depot={len(self._depot.bikes)}, "
            f"outside={len(self._outside_system_bikes)}"
        )

        return observation, info

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
        # Execute the action
        self._invalid_action = False

        # Calculate current mean truck velocity
        hours = divmod((self._timeslot * 3 + 1) * 3600 + self._env_time, 3600)[0] % 24
        mean_truck_velocity = self._velocity_lookup[hours][self._day]

        elapsed_time = 0
        distance = 0

        if action == Actions.STAY.value:
            elapsed_time = stay(truck=self._truck)

        elif action in ACTION_TO_DIRECTION:
            elapsed_time, distance, self._invalid_action = move(
                action=action,
                truck=self._truck,
                distance_lookup=self._distance_lookup,
                cell_dict=self._cells,
                mean_velocity=mean_truck_velocity
            )

        elif action == Actions.DROP_BIKE.value:
            elapsed_time, distance, self._invalid_action = drop_bike(
                truck=self._truck,
                distance_lookup=self._distance_lookup,
                mean_velocity=mean_truck_velocity,
                depot=self._depot,
                system_bikes=self._system_bikes,
                maximum_number_of_bikes=self._maximum_number_of_bikes
            )

        elif action == Actions.PICK_UP_BIKE.value:
            elapsed_time, distance, self._invalid_action = pick_up_bike(
                truck=self._truck,
                station_dict=self._stations,
                distance_lookup=self._distance_lookup,
                mean_velocity=mean_truck_velocity,
                depot=self._depot,
                system_bikes=self._system_bikes
            )

        elif action == Actions.CHARGE_BIKE.value:
            elapsed_time, distance, self._invalid_action = charge_bike(
                truck=self._truck,
                station_dict=self._stations,
                distance_lookup=self._distance_lookup,
                mean_velocity=mean_truck_velocity,
                depot=self._depot,
                system_bikes=self._system_bikes
            )

        # Start new log line
        timeslot_index = 8 * DAYS_TO_NUM[self._day] + self._timeslot
        self._env_logger.new_log_line(timeslot=timeslot_index)
        self._env_logger.log_starting_action(
            action=str(Actions(action).name),
            t=elapsed_time,
            cell_id=self._truck.get_cell().get_id(),
            invalid=False,
        )

        # Store pre-update state for reward calculation
        truck_cell = self._truck.get_cell()
        old_eligibility_score = truck_cell.get_eligibility_score()
        old_critic_score = truck_cell.get_critic_score()
        old_global_critic_score = self._global_critic_score
        old_surplus_bikes = truck_cell.get_surplus_bikes()  # FIX A: added direct computation of old_surplus_bikes

        self._env_logger.info(
            f"Cell {truck_cell.get_id()} before update has "
            f"Eligibility: {old_eligibility_score} Critic: {old_critic_score}"
        )

        # -------------------------------------------------------------------------
        # Advance simulation
        # -------------------------------------------------------------------------
        # Process events during elapsed time
        steps = math.ceil(elapsed_time / EnvDefaults.STEP_DURATION_SECONDS)
        failures = self._jump_to_next_state(steps)

        # Handle bike placement after movement
        is_placement_action = action in {Actions.CHARGE_BIKE.value, Actions.DROP_BIKE.value}

        if is_placement_action and not self._invalid_action:
            station = self._stations.get(self._truck.get_position())
            bike = self._truck.unload_bike()
            station.lock_bike(bike)
            self._system_bikes[bike.get_bike_id()] = bike

        # Final state update -> S'
        failures.extend(self._jump_to_next_state(steps=1))
        steps += 1
        self._total_failures += sum(failures)

        # Update cell statistics
        self._update_cells_metrics()
        truck_cell.set_eligibility_score(1.0)

        self._env_logger.log_state(
            step=steps,
            time=convert_seconds_to_hours_minutes_day(
                day=self._day.upper(),
                seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time
            ),
        )
        self._env_logger.log_truck(
            truck=self._truck,
            depot_bikes=len(self._depot.bikes),
        )

        if not self._invalid_action:
            if action == Actions.PICK_UP_BIKE.value:
                truck_cell.set_ops(truck_cell.get_ops() + 1)
            elif action == Actions.DROP_BIKE.value:
                truck_cell.set_ops(truck_cell.get_ops() - 1)

        if action in {Actions.UP.value, Actions.DOWN.value, Actions.LEFT.value, Actions.RIGHT.value}:
            truck_cell.set_visits(truck_cell.get_visits() + 1)
            self._total_visits += 1

        # Update action tracking
        if not self._invalid_action:
            self._last_move_action = action
        else:
            self._total_invalid_actions += 1

        self._last_cell_border_type = sum(self._get_borders())

        self._env_logger.log_ending_action(
            invalid=self._invalid_action,
            time=convert_seconds_to_hours_minutes_day(
                day=self._day.upper(),
                seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time
            ),
        )

        # Compute outputs
        reward = self._get_reward(
            action,
            old_eligibility_score,
            old_critic_score,
            old_global_critic_score,
            old_surplus_bikes
        )
        observation = self._get_obs(action)

        # Build info dictionary
        info = {
            'time': self._env_time + (self._timeslot * 3 + 1) * 3600,
            'day': self._day if self._timeslot < EnvDefaults.TIMESLOTS_PER_DAY - 1 else NUM_TO_DAYS[(DAYS_TO_NUM[self._day] + 1) % 7],
            'week': self._days_completed // 7,
            'year': self._days_completed // 365,
            'failures': failures,
            'steps': steps,
            'global_critic_score':self._global_critic_score,
            'total_invalid_actions':self._total_invalid_actions,
            'cell_dict': self._cells,
            'distance': distance,
            'depot_bikes': len(self._depot.bikes),
            'truck_bikes': self._truck.get_load(),
            'number_of_system_bikes': len(self._system_bikes),
            'number_of_outside_bikes': len(self._outside_system_bikes),
            'number_of_traveling_bikes': len(self._travelling_bikes),
        }

        # Check for timeslot/episode termination
        terminated, done = self._check_termination()

        # Apply final penalty if episode complete
        if done:
            reward -= (self._total_failures / self._total_timeslots) # / 10.0     # RUN: ID-1 REMOVING / 10.0 TO INCREASE PENALTY
            self._env_logger.log_done(
                time=convert_seconds_to_hours_minutes_day(
                    day=self._day.upper(),
                    seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time
                )
            )

        return observation, reward, done, terminated, info

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

            # Preserve remaining time into next timeslot
            env_time_diff = self._env_time - EnvDefaults.TIMESLOT_DURATION_SECONDS
            self._initialize_day_timeslot()
            self._env_time = env_time_diff

            self._timeslots_completed += 1
            terminated = True

        # Check if episode is complete
        done = self._timeslots_completed == self._total_timeslots

        return terminated, done

    # -------------------------------------------------------------------------
    # Gymnasium Interface Methods
    # -------------------------------------------------------------------------

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Clean up resources."""
        return

    # -------------------------------------------------------------------------
    # Event Simulation
    # -------------------------------------------------------------------------

    def _initialize_day_timeslot(self) -> None:
        current_slot_index = (
                DAYS_TO_NUM[self._day] * EnvDefaults.TIMESLOTS_PER_DAY + self._timeslot
        )
        next_slot_index = (current_slot_index + 1) % EnvDefaults.PRECOMPUTED_EPISODE_TIMESLOTS

        if self._event_buffer is None:
            # Cold start: build full 6-hour window
            current_events = build_events(
                self._precomputed_buffers[current_slot_index],
                self._stations,
                time_offset=0,
            )
            next_events = build_events(
                self._precomputed_buffers[next_slot_index],
                self._stations,
                time_offset=EnvDefaults.TIMESLOT_DURATION_SECONDS,
            )
            self._event_buffer = deque(
                heapq.merge(current_events, next_events, key=lambda e: e.time)
            )
        else:
            # Advance: shift surviving events, append next slot fresh
            for event in self._event_buffer:
                event.time -= EnvDefaults.TIMESLOT_DURATION_SECONDS

            next_events = build_events(
                self._precomputed_buffers[next_slot_index],
                self._stations,
                time_offset=EnvDefaults.TIMESLOT_DURATION_SECONDS,
            )
            self._event_buffer = deque(
                heapq.merge(self._event_buffer, next_events, key=lambda e: e.time)
            )

        # --- PMF / rate update (unchanged from your draft) ---
        pmf_matrix = self._load_pmf_matrix(self._day, self._timeslot)
        global_rate = self._global_rate_dict[(self._day.lower(), self._timeslot)]

        pmf_lookup = {
            row["node_id"]: row
            for row in pmf_matrix.iter_rows(named=True)
        }

        for stn_id, stn in self._stations.items():
            row_data = pmf_lookup[stn_id]
            row_sum = sum(v for k, v in row_data.items() if k != "node_id")
            stn.set_request_rate(row_sum * global_rate)
            col_sum = sum(
                pmf_lookup[orig][str(stn_id)]
                for orig in pmf_lookup
                if str(stn_id) in pmf_lookup[orig]
            )
            stn.set_arrival_rate(col_sum * global_rate)

        for cell in self._cells.values():
            nodes = cell.get_nodes()
            cell.set_demand_rate(sum(self._stations[n].get_demand_rate() for n in nodes))
            cell.set_arrival_rate(sum(self._stations[n].get_arrival_rate() for n in nodes))

        # Reset environment clock
        self._env_time = 0

    def _load_pmf_matrix(self, day: str, timeslot: int) -> pl.DataFrame:
        """Load the PMF matrix and global rate for a given day and timeslot."""
        # Construct file path
        matrix_path = os.path.join(self._data_path, DEFAULT_PATHS.matrices_folder, day.lower())
        pmf_matrix = pl.read_csv(os.path.join(matrix_path, str(timeslot).zfill(2) + '-pmf-matrix.csv'))

        return pmf_matrix

    def _jump_to_next_state(self, steps: int = 0) -> list:
        """ Advance the simulation by the given number of steps."""
        failures = []

        for step in range(steps):
            # Increment environment time
            self._env_time += EnvDefaults.STEP_DURATION_SECONDS
            self._env_logger.set_env_time(
                convert_seconds_to_hours_minutes_day(
                    day=self._day.upper(),
                    seconds=(self._timeslot * 3 + 1) * 3600 + self._env_time
                )
            )

            # Update eligibility scores for all cells
            for _, cell in self._cells.items():
                if cell.has_all_neighbors():
                    cell.update_eligibility_score(self._eligibility_decay)
                else:
                    cell.update_eligibility_score(self._borders_eligibility_decay)

            # Process events that occurred during this step
            step_failures = self._process_events()
            failures.append(step_failures)

        return failures

    def _process_events(self) -> int:
        """
        Process all events that occurred before current env_time.

        Returns:
            Number of failures during this step.
        """
        total_step_failures = 0

        while self._event_buffer and self._event_buffer[0].time < self._env_time:
            event = self._event_buffer.popleft()

            # Process the event and get any resulting failure
            failure = event_handler(
                event=event,
                station_dict=self._stations,
                nearby_nodes_dict=self._nearby_nodes_dict,
                distance_lookup=self._distance_lookup,
                system_bikes=self._system_bikes,
                outside_system_bikes=self._outside_system_bikes,
                traveling_bikes=self._travelling_bikes,
                logger=self._env_logger,
                logging_state_and_trips=logging_state_and_trips,
                depot=self._depot,
                maximum_number_of_bikes=self._maximum_number_of_bikes,
                truck_load=self._truck.get_load()
            )
            total_step_failures += failure

        return total_step_failures

    # -------------------------------------------------------------------------
    # Observation Construction
    # -------------------------------------------------------------------------

    def _get_obs(self, action: int = None) -> np.array: # type: ignore
        """Construct the observation vector."""
        # Previous action encoding
        ohe_previous_action = np.array(
            [1.0 if action == a.value else 0.0 for a in Actions],
            dtype=np.float16
        )

        # Truck cell position encoding
        truck_cell_id = self._truck.get_cell().get_id()
        sorted_cells_keys = self._sorted_cell_ids

        ohe_cell_position = np.array(
            [1.0 if truck_cell_id == cell_id else 0.0 for cell_id in sorted_cells_keys],
            dtype=np.float16
        )

        # Critical cells encoding
        ohe_cell_critic = np.array(
            [1.0 if cell.is_critical() else 0.0 for cell in self._cells.values()],
            dtype=np.float16
        )

        # Border flags
        ohe_borders = self._get_borders()

        # Scalar features
        truck_cell = self._truck.get_cell()

        depot_load_ratio = len(self._depot.bikes) / self._maximum_number_of_bikes
        system_ratio = (len(self._system_bikes) + self._truck.get_load()) / self._maximum_number_of_bikes

        scalar_features = np.array([
            system_ratio,
            depot_load_ratio,   # RUN: ID-6
            1.0 if truck_cell.get_surplus_bikes() > 0 else 0.0,     # Surplus flag
            1.0 if truck_cell.get_total_bikes() == 0 else 0.0,      # Empty cell flag
            self._global_critic_score / len(self._cells),           # Global critic score
        ], dtype=np.float16)

        # Concatenate all features
        observation = np.concatenate([
            scalar_features,
            ohe_previous_action,
            ohe_cell_position,
            ohe_cell_critic,
            ohe_borders,
        ])

        return observation


    def _get_borders(self) -> np.array: # type: ignore
        """
        Get border flags indicating which adjacent cells are missing.

        Returns:
            Array of 4 flags (one per direction).
        """
        adjacent_cells = self._truck.get_cell().get_adjacent_cells()
        return np.array(
            [1 if adj_cell is None else 0 for adj_cell in adjacent_cells.values()],
            dtype=np.float16
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
    # Reward Function
    # -------------------------------------------------------------------------

    def _get_reward(
        self,
        action: int,
        old_eligibility_score: float,
        old_critic_score: float,
        old_global_critic_score: float,
        old_surplus_bikes: float
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
        was_surplus = old_surplus_bikes > 0     # FIX A: replaced the was_surplus with this from the one below
        # was_surplus = old_critic_score <= RewardComponents.SURPLUS_THRESHOLD
        is_critical = self._truck.get_cell().is_critical()

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
            # bike_charge_reward = 0.0  # Placeholder for future implementation

        elif action in {Actions.UP.value, Actions.DOWN.value,
                        Actions.LEFT.value, Actions.RIGHT.value}:
            eligibility_penalty = self._compute_movement_penalty(
                old_eligibility_score, was_critical
            )

        elif action == Actions.STAY.value:
            stay_penalty, loop_penalty = self._compute_stay_penalty(
                was_critical, old_global_critic_score, loop_penalty
            )

        delta_global = self._global_critic_score - old_global_critic_score

        deploy_ratio = len(self._system_bikes) / self._maximum_number_of_bikes
        r_deploy = RewardComponents.DEPLOY_WEIGHT * deploy_ratio

        idle_ratio = len(self._depot.bikes) / self._maximum_number_of_bikes
        r_depot = -RewardComponents.DEPOT_WEIGHT * idle_ratio

        # Combine all reward components
        reward = (
                RewardComponents.BASE_COST
                + drop_reward
                + pick_up_reward
                + bike_charge_reward
                + eligibility_penalty
                + stay_penalty
                + loop_penalty
                + r_deploy
                + r_depot
                - 0.5 * delta_global    # RUN: ID-2
        )

        self._env_logger.info(
            f"split reward {drop_reward} {pick_up_reward} {bike_charge_reward} "
            f"{eligibility_penalty} {stay_penalty} {loop_penalty}"
        )

        return reward

    # -------------------------------------------------------------------------
    # Reward Computation Utility Methods
    # -------------------------------------------------------------------------

    def _compute_drop_reward(
            self,
            was_critical: bool,
            is_critical: bool,
            was_surplus: bool
    ) -> float:
        """Compute reward for drop bike action."""
        drop_reward = RewardComponents.DROP_BASE

        if was_critical and not is_critical:    # If the truck manages to rebalance the cell -> Big reward and (optional) reset all eligibility to 0.0
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
        if self._truck.last_charge < RewardComponents.CHARGE_LOW_BATTERY_THRESHOLD:
            # Bike was already nearly full → useless charge
            if was_critical:
                return RewardComponents.CHARGE_USELESS_CRITICAL  # -0.1
            else:
                return RewardComponents.CHARGE_USELESS_NORMAL  # -0.3
        else:
            # Bike was sufficiently discharged → useful charge
            if was_critical:
                return RewardComponents.CHARGE_USEFUL_CRITICAL  # +0.5
            else:
                return RewardComponents.CHARGE_USEFUL_NORMAL  # +0.3

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
            if cell.get_eligibility_score() > 0.3:
                cell.set_eligibility_score(cell.get_eligibility_score() - 0.2)

        self._env_logger.info(
            message=f" ---------> {self._truck.get_cell().get_id()} has been rebalanced. "
                    f"All eligibility scores adjusted ###################################"
        )
        self._truck.get_cell().set_eligibility_score(1.0)

    # -------------------------------------------------------------------------
    # Graph Update
    # -------------------------------------------------------------------------

    def _update_cells_metrics(self):
        """Update the cell subgraph with current regional metrics."""

        # ── Recompute expected departures from remaining buffer (old-env style) ──
        expected_departures: dict[int, int] = {}
        expected_max_departure: dict[int, int] = {}
        expected_after_arrival: dict[int, int] = {}

        for event in self._event_buffer:
            if event.time > self._env_time + EnvDefaults.TIMESLOT_DURATION_SECONDS:
                break
            cell = None
            cell_id = None
            if event.is_departure():
                loc = event.get_trip().get_start_location()
                if loc.get_station_id() != 10000:
                    cell = loc.get_cell()
                    cell_id = cell.get_id()
                    expected_departures[cell_id] = expected_departures.get(cell_id, 0) + 1
            elif event.is_arrival():
                loc = event.get_trip().get_end_location()
                if loc.get_station_id() != 10000:
                    cell = loc.get_cell()
                    cell_id = cell.get_id()
                    expected_departures[cell_id] = expected_departures.get(cell_id, 0) - 1
            if cell is not None:
                previous_max = expected_max_departure.get(cell_id, 0)
                expected_max_departure[cell_id] = max(previous_max, expected_departures[cell_id])
                expected_after_arrival[cell_id] = expected_max_departure[cell_id] - expected_departures[cell_id]

        # ── Update each cell ──────────────────────────────────────────────────────
        self._global_critic_score = 0
        for cell_id, cell in self._cells.items():
            expected = expected_max_departure.get(cell_id, 0) + self._min_bikes_per_cell
            aft_arrivals = expected_after_arrival.get(cell_id, 0)
            cell.update_metrics(
                stations=self._stations,
                expected=expected,
                aft_arrivals=aft_arrivals
            )
            cell.set_metric(
                'truck_cell',
                1.0 if cell.get_id() == self._truck.get_cell().get_id() else 0.0
            )

            # FOR TEST
            raw_critic = cell.get_critic_score()

            if self._use_binary_critic:
                # Binary classification: critic = 1 if “bad”, else 0
                is_critical = raw_critic > 0.0
                cell.set_critic_score(1.0 if is_critical else raw_critic)   # FIX A: changed from <1.0 if is_critical else 0.0>
                if is_critical:
                    self._global_critic_score += 1.0
            else:
                self._global_critic_score += max(raw_critic, 0.0)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _bike_repositioning(
            self,
            enable_repositioning: bool = False,
            use_net_flow: bool = False,
    ) -> dict:
        """
        Compute initial bike distribution.

        Args:
            enable_repositioning: If False, allocate minimum bikes per cell
                (plus uniform extras) and stop. If True, perform additional
                repositioning using either net flow or random allocation.
            use_net_flow: If True and enable_repositioning is True, bias the
                extra bikes according to predicted net outflow; otherwise
                distribute them randomly.

        Returns:
            Dictionary mapping cell_id to number of bikes.
        """
        # -------------------------------------------------------------------
        # Step 1: Uniform base allocation, adjusted if fleet is too small
        # -------------------------------------------------------------------
        available = self._maximum_number_of_bikes - self._truck.get_load()
        base_bikes_per_cell = self._min_bikes_per_cell
        if base_bikes_per_cell * len(self._cells) > available:
            base_bikes_per_cell = available // len(self._cells)
            self._env_logger.warning(
                f"Base bikes per cell adjusted to {base_bikes_per_cell} "
                f"due to fleet size / truck load constraints."
            )

        bikes_per_cell = {
            cell_id: base_bikes_per_cell for cell_id in self._cells.keys()
        }
        remaining = available - base_bikes_per_cell * len(self._cells)

        self._env_logger.info(
            f"Base bikes per cell adjusted to {base_bikes_per_cell} "
            f"with {len(self._cells)} cells"
        )

        if remaining == 0:
            return bikes_per_cell

        if remaining < 0:
            raise ValueError(
                "Fleet size minus truck load is too small to allocate "
                "the base number of bikes per cell."
            )

        # If repositioning is disabled, just keep uniform allocation
        if not enable_repositioning:
            return bikes_per_cell

        # -------------------------------------------------------------------
        # Step 2: Net flow biased distribution (only current timeslot events)
        # -------------------------------------------------------------------
        bikes_positioned = 0

        if use_net_flow:
            net_flow_per_cell = {cell_id: 0 for cell_id in self._cells.keys()}

            for event in self._event_buffer:
                if event.time > EnvDefaults.TIMESLOT_DURATION_SECONDS:
                    break  # buffer is sorted — stop at next-timeslot events
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

            total_negative_flow = sum(
                f for f in net_flow_per_cell.values() if f < 0
            )

            if total_negative_flow < 0:  # guard: at least one cell has net outflow
                for cell_id, flow in net_flow_per_cell.items():
                    if flow < 0:
                        proportional_bikes = int(
                            (flow / total_negative_flow) * remaining
                        )
                        bikes_per_cell[cell_id] += proportional_bikes
                        bikes_positioned += proportional_bikes

        # -------------------------------------------------------------------
        # Step 3: Distribute any remaining bikes randomly
        # -------------------------------------------------------------------
        leftover = remaining - bikes_positioned
        if leftover > 0:
            cell_ids = list(self._cells.keys())
            chosen = np.random.choice(cell_ids, size=leftover, replace=True)
            for cell_id in chosen:
                bikes_per_cell[cell_id] += 1

        return bikes_per_cell

    def _update_logging_path(self, results_path: str | None) -> None:
        """
        Update the env logging directory at runtime (e.g. per episode).
        """
        self._results_path = results_path
        if results_path is None:
            return

        episode_log_dir = os.path.join(results_path, "logs")
        self._env_logger.reconfigure(
            log_dir=episode_log_dir,
            filename="env.log",
        )

    # def _adjust_depot_system_discrepancy(self):
    #     """
    #     Ensure the total bike count matches the maximum fleet size.
    #
    #     Bikes that exit the system are replaced from the outside_system_bikes pool.
    #     """
    #     depot_load = len(self._depot.bikes)
    #     system_load = len(self._system_bikes)
    #     truck_load = self._truck.get_load()
    #     total_bikes = depot_load + system_load + truck_load
    #
    #     if total_bikes < self._maximum_number_of_bikes:
    #         bikes_to_add = self._maximum_number_of_bikes - total_bikes
    #
    #         for _ in range(bikes_to_add):
    #             bike_id = next(iter(self._outside_system_bikes))
    #             bike = self._outside_system_bikes.pop(bike_id)
    #             bike.reset()
    #             self._depot.bikes[bike.get_bike_id()] = bike
    #
    #         self._logger.warning(
    #             message=f" ---------> System has been adjusted to max_number_of_bikes "
    #                     f"adding {bikes_to_add} bikes"
    #         )