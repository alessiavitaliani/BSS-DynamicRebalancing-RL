import math
import pickle
import threading
import random
import bisect
import torch

import gymnasium as gym
import numpy as np
import pandas as pd
import osmnx as ox
import torch.nn as nn

from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import simulate_environment, event_handler
from gymnasium_env.simulator.truck_simulator import (move_up, move_down, move_left, move_right, drop_bike, pick_up_bike,
                                                     charge_bike, stay)
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import (initialize_graph, initialize_stations, Logger, Actions, initialize_bikes,
                                           convert_seconds_to_hours_minutes, initialize_cells_subgraph,
                                           logistic_penalty_function)

params = {
    'graph_file': 'utils/cambridge_network.graphml',
    'cell_file': 'utils/cell_data.pkl',
    'distance_matrix_file': 'utils/distance_matrix.csv',
    'filtered_stations_file': 'utils/filtered_stations.csv',
    'matrices_folder': 'matrices/09-10',
    'rates_folder': 'rates/09-10',
    'trips_folder': 'trips/',
    'nearby_nodes': 'utils/nearby_nodes.pkl',
    'velocity_matrix': 'utils/ev_velocity_matrix.csv',
    'consumption_matrix': 'utils/ev_consumption_matrix.csv'
}

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
num2days = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}

# ----------------------------------------------------------------------------------------------------------------------

def _detect_self_loops(actions: tuple) -> bool:
    # Define valid 2-step back-and-forth patterns as tuples
    self_loop_patterns_2 = [
        (Actions.UP.value, Actions.DOWN.value),
        (Actions.DOWN.value, Actions.UP.value),
        (Actions.LEFT.value, Actions.RIGHT.value),
        (Actions.RIGHT.value, Actions.LEFT.value),
        (Actions.PICK_UP_BIKE.value, Actions.DROP_BIKE.value),
        (Actions.DROP_BIKE.value, Actions.PICK_UP_BIKE.value)
    ]

    return actions in self_loop_patterns_2


class FullyDynamicEnv(gym.Env):

    # Initialize the environment
    def __init__(self, data_path: str):
        super().__init__()

        # Initialize paths and logger
        self.data_path = data_path
        self.logger = Logger(data_path + 'env_output.log')

        # Initialize the graph
        self.graph = initialize_graph(data_path + params['graph_file'])

        # Compute the bounding box
        nodes, _ = ox.graph_to_gdfs(self.graph)
        self.min_lat, self.max_lat = nodes['y'].min(), nodes['y'].max()
        self.min_lon, self.max_lon = nodes['x'].min(), nodes['x'].max()

        # Initialize the nodes dictionary
        nodes_gdf = ox.graph_to_gdfs(self.graph, edges=False)
        self.nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

        # Load nearby nodes dictionary
        with open(data_path + params['nearby_nodes'], 'rb') as file:
            self.nearby_nodes_dict = pickle.load(file)

        # Initialize the cells
        with open(data_path + params['cell_file'], 'rb') as file:
            self.cells = pickle.load(file)

        # Ensure eligibility_score is present in each cell
        for cell in self.cells.values():
            cell.eligibility_score = 0.0

        # Initialize the distance matrix
        self.distance_matrix = pd.read_csv(data_path + params['distance_matrix_file'], index_col='osmid')
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)

        # Initialize the velocity matrix
        self.velocity_matrix = pd.read_csv(data_path + params['velocity_matrix'], index_col='hour')

        # Initialize the consumption matrix
        self.consumption_matrix = pd.read_csv(data_path + params['consumption_matrix'], index_col='hour')

        # Define action space
        self.action_space = spaces.Discrete(len(Actions))

        # Define observation space
        # Features: [1 agent features + 7 days + 24 hours]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + 7 + 24,),
            dtype=np.float32
        )

        # Initialize simulation state variables
        self.pmf_matrix, self.global_rate, self.global_rate_dict = None, None, None
        self.system_bikes, self.outside_system_bikes = None, None
        self.depot, self.depot_node = None, None
        self.maximum_number_of_bikes = 3500
        self.stations = None
        self.truck = None
        self.event_buffer = None
        self.next_event_buffer = None
        self.next_timeslot_event_buffer = None
        self.env_time = 0
        self.timeslot = 0
        self.day = 'monday'
        self.cell_subgraph = None
        self.timeslots_completed = 0
        self.days_completed = 0
        self.next_bike_id = 0
        self.total_timeslots = 0
        self.background_thread = None
        self.discount_factor = 0.99
        self.eligibility_decay = 0.9968
        self.zero_bikes_penalty = []
        self.reward_params = None
        self.total_visits = 1
        self.total_failures = 0
        # self.history_4 = deque(maxlen=4)
        self.last_move_action = None
        self.invalid_drop_action = False

        self.encoder = nn.Embedding(len(self.cells), 27)
        self.zone_id_to_index = {zone_id: idx for idx, zone_id in enumerate(sorted(self.cells.keys()))}

        # Set logging options
        self.logging = False
        self.logger.set_logging(self.logging)


    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # Call parent class reset
        super().reset(seed=seed)

        #Enabling the logging
        self.logging = options.get('logging', False) if options else False
        self.logger.set_logging(self.logging)

        # Day and time slot options
        self.day = options.get('day', 'monday') if options else 'monday'
        self.timeslot = options.get('timeslot', 0) if options else 0
        self.total_timeslots = options.get('total_timeslots', 56) if options else 56

        # Bike options
        self.maximum_number_of_bikes = options.get('maximum_number_of_bikes', self.maximum_number_of_bikes) if options else self.maximum_number_of_bikes
        depot_id = options.get('depot_id', 491) if options else 491

        # Truck options
        cell_id_list = list(self.cells.keys())
        truck_cell_id = np.random.choice(cell_id_list)
        max_truck_load = options.get('max_truck_load', 30) if options else 30

        # Discount factor option
        self.discount_factor = options.get('discount_factor', 0.99) if options else 0.99

        # Reward parameters
        self.reward_params = options.get('reward_params', None) if options else None
        self.invalid_drop_action = False

        # Reset the cells
        for cell in self.cells.values():
            cell.reset()
        self.total_visits = 1
        self.cells[truck_cell_id].set_visits(1)

        # Reset reward items
        self.zero_bikes_penalty = []

        # Initialize the depot
        self.next_bike_id = 0
        self.depot_node = self.cells.get(depot_id).get_center_node()
        self.depot, self.next_bike_id = initialize_bikes(n=self.maximum_number_of_bikes, next_bike_id=self.next_bike_id)

        self.timeslots_completed = 0
        self.days_completed = 0

        self.event_buffer = None
        self.next_event_buffer = None
        self.total_failures = 0
        # self.history_4 = deque(maxlen=4)

        # Create stations dictionary
        from gymnasium_env.simulator.station import Station
        gdf_nodes = ox.graph_to_gdfs(self.graph, edges=False)
        stns = {}
        for index, row in gdf_nodes.iterrows():
            station = Station(index, row['y'], row['x'])
            stns[index] = station
        stns[10000] = Station(10000, 0, 0)
        self.stations = stns

        # Set the cell for each station
        for cell in self.cells.values():
            for node in cell.nodes:
                self.stations[node].set_cell(cell)

        for station in self.stations.values():
            if station.get_cell() is None and station.get_station_id() != 10000:
                raise ValueError(f"Station {station} is not assigned to a cell.")

        # Initialize the truck
        cell = self.cells[truck_cell_id]
        bikes = {key: self.depot.pop(key) for key in list(self.depot.keys())[:15]}
        self.truck = Truck(cell.center_node, cell, bikes=bikes, max_load=max_truck_load)

        # Initialize the day and time slot
        self._initialize_day_timeslot()

        # Compute bikes per cell based on net flow
        bikes_per_cell = self._net_flow_based_repositioning()

        # Initialize stations and system bikes
        bikes_per_station = {stn_id: 0 for stn_id in self.stations.keys()}
        for cell_id, num_of_bikes in bikes_per_cell.items():
            stn_id = self.cells[cell_id].get_center_node()
            bikes_per_station[stn_id] = num_of_bikes

        # Initialize the system bikes
        self.system_bikes, self.outside_system_bikes, self.next_bike_id = initialize_stations(
            stations=self.stations,
            depot=self.depot,
            bikes_per_station=bikes_per_station,
            next_bike_id=self.next_bike_id,
        )

        # Initialize the cell subgraph
        custom_features = {
            'truck_cell': 0.0,
            'total_bikes': 0.0,
            'critic_score': 0.0,
            'visits': 0,
            'eligibility_score': 0.0,
            # TURN OFF THIS TO DISABLE BATTERY CHARGE
            # 'low_battery_bikes': 0.0,
        }
        self.cell_subgraph = initialize_cells_subgraph(self.cells, self.nodes_dict, self.distance_matrix, custom_features)

        # Update the graph with regional metrics
        self._update_graph()

        # Return the initial observation and an optional info dictionary
        observation = self._get_obs()
        info = {
            'cells_subgraph': self.cell_subgraph,
            'network_graph': self.graph,
            'cell_dict': self.cells,
            'nodes_dict': self.nodes_dict,
            'agent_position': self._get_truck_position(),
            'steps': 0,
            'number_of_system_bikes': len(self.system_bikes),
            'truck_neighbor_cells': self.truck.get_cell().get_adjacent_cells(),
            'distance_matrix': self.distance_matrix,
            'truck_load': self.truck.get_load(),
        }

        return observation, info


    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        # Log the start of the step
        self.logger.new_log_line()

        # Check for discrepancies in the depot + system bikes between total bikes
        self._adjust_depot_system_discrepancy()

        # Initialize the variables
        t = 0
        distance = 0

        hours = divmod((self.timeslot * 3 + 1) * 3600 + self.env_time, 3600)[0] % 24
        mean_truck_velocity = self.velocity_matrix.loc[hours, self.day]
        bike_picked_up = False

        # Perform the action
        if action == Actions.STAY.value:
            t = stay(self.truck)
            self.logger.log_starting_action('STAY', t)
        elif action == Actions.RIGHT.value:
            t, distance = move_right(self.truck, self.distance_matrix, self.cells, mean_truck_velocity)
            self.logger.log_starting_action('RIGHT', t)
            # Append last action to history
            # self.history_4.append(action)
        elif action == Actions.UP.value:
            t, distance = move_up(self.truck, self.distance_matrix, self.cells, mean_truck_velocity)
            self.logger.log_starting_action('UP', t)
            # Append last action to history
            # self.history_4.append(action)
        elif action == Actions.LEFT.value:
            t, distance = move_left(self.truck, self.distance_matrix, self.cells, mean_truck_velocity)
            self.logger.log_starting_action('LEFT', t)
            # Append last action to history
            # self.history_4.append(action)
        elif action == Actions.DOWN.value:
            t, distance = move_down(self.truck, self.distance_matrix, self.cells, mean_truck_velocity)
            self.logger.log_starting_action('DOWN', t)
            # Append last action to history
            # self.history_4.append(action)
        elif action == Actions.DROP_BIKE.value:
            if len(self.system_bikes) < (self.maximum_number_of_bikes - 5) or self.truck.get_load() > 0:
                t, distance = drop_bike(self.truck, self.distance_matrix, mean_truck_velocity, self.depot_node, self.depot)
            else:
                t = stay(self.truck)
                self.invalid_drop_action = True
            self.logger.log_starting_action('DROP_BIKE', t)
        elif action == Actions.PICK_UP_BIKE.value:
            t, distance, _ = pick_up_bike(self.truck, self.stations, self.distance_matrix, mean_truck_velocity,
                                          self.depot_node, self.depot, self.system_bikes)
            self.logger.log_starting_action('PICK_UP_BIKE', t)

        # TURN OFF THIS TO DISABLE BATTERY CHARGE
        # elif action == Actions.CHARGE_BIKE.value:
        #     t, distance, bike_picked_up = charge_bike(self.truck, self.stations, self.distance_matrix, mean_truck_velocity,
        #                                               self.depot_node, self.depot, self.system_bikes)
        #     self.logger.log_starting_action('CHARGE_BIKE', t)

        # Calculate steps and log the state
        steps = math.ceil(t / 30)
        self.logger.log_state(
            step=int(self.env_time / 30),
            time=convert_seconds_to_hours_minutes((self.timeslot * 3 + 1) * 3600 + self.env_time)
        )

        self.zero_bikes_penalty = []

        # Update the environment state
        failures = self._jump_to_next_state(steps)

        # Handle specific actions post-environment update
        # TURN OFF THIS TO DISABLE BATTERY CHARGE
        # if action in {Actions.DROP_BIKE.value, Actions.CHARGE_BIKE.value}:
        if bike_picked_up or (action == Actions.DROP_BIKE.value and not self.invalid_drop_action):
            station = self.stations.get(self.truck.get_position())
            bike = self.truck.unload_bike()
            station.lock_bike(bike)
            self.system_bikes[bike.get_bike_id()] = bike

        # Log the ending action
        self.logger.log_ending_action(
            time=convert_seconds_to_hours_minutes((self.timeslot * 3 + 1) * 3600 + self.env_time)
        )

        # Perform a final state update after the last action
        # print("Entering in last state update")
        failures.extend(self._jump_to_next_state(steps=1))
        steps += 1
        self.total_failures += sum(failures)

        # Update the last visited cells
        if action in {Actions.UP.value, Actions.DOWN.value, Actions.LEFT.value, Actions.RIGHT.value}:
            truck_cell = self.truck.get_cell()
            truck_cell.set_visits(truck_cell.get_visits() + 1)
            self.total_visits += 1

        # Log truck state
        self.logger.log_truck(self.truck)

        # Compute the outputs
        reward = self._get_reward(action)
        observation = self._get_obs(action)
        self._update_graph()

        info = {
            'cells_subgraph': self.cell_subgraph,
            'agent_position': self._get_truck_position(),
            'time': self.env_time + (self.timeslot * 3 + 1) * 3600,
            'day': self.day,
            'week': int(self.days_completed // 7),
            'year': int(self.days_completed // 365),
            'failures': failures,
            'number_of_system_bikes': len(self.system_bikes),
            'steps': steps,
            'truck_neighbor_cells': self.truck.get_cell().get_adjacent_cells(),
            'truck_load': self.truck.get_load(),
        }

        self.invalid_drop_action = False

        terminated = False
        if self.env_time >= 3600*3:
            self.timeslot = (self.timeslot + 1) % 8
            self.day = num2days[(days2num[self.day] + 1) % 7] if self.timeslot == 0 else self.day
            self.days_completed += 1 if self.timeslot == 0 else 0
            for event in self.event_buffer:
                event.time -= 3600*3
            env_time_diff = self.env_time - 3600*3
            self._initialize_day_timeslot()
            self.env_time = env_time_diff
            self.timeslots_completed += 1
            terminated = True


        done = True if self.timeslots_completed == self.total_timeslots else False

        if done:
            reward -= (self.total_failures / self.total_timeslots) / 10.0

        # Return the step results
        return observation, reward, done, terminated, info


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close(self):
        """Clean up resources."""
        self.background_thread.join()


    def _precompute_poisson_events(self):
        """Background thread for precomputing Poisson events."""
        timeslot = (self.timeslot + 2) % 8
        day = num2days[(days2num[self.day] + 1) % 7] if timeslot == 0 else self.day

        # Flatten the PMF matrix for event simulation
        pmf_matrix, global_rate = self._load_pmf_matrix_global_rate(day, timeslot)

        # Flatten the PMF matrix for event simulation
        values = pmf_matrix.values.flatten()
        ids = [(row, col) for row in pmf_matrix.index for col in pmf_matrix.columns]
        flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
        flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)

        self.next_event_buffer = simulate_environment(
            duration=3600 * 3,  # 3 hours
            timeslot=timeslot,
            global_rate=global_rate,
            pmf=flattened_pmf,
            stations=self.stations,
            distance_matrix=self.distance_matrix,
        )

        # Update time of each event
        for event in self.next_event_buffer:
            event.time += 3600 * 3


    def _initialize_day_timeslot(self):
        # Load PMF matrix and global rate for the current day and time slot
        self.pmf_matrix, self.global_rate = self._load_pmf_matrix_global_rate(self.day, self.timeslot)

        for stn_id, stn in self.stations.items():
            stn.set_request_rate(self.pmf_matrix.loc[stn_id, :].sum()*self.global_rate)
            stn.set_arrival_rate(self.pmf_matrix.loc[:, stn_id].sum()*self.global_rate)

        # Update the request rate for each cell
        for cell in self.cells.values():
            total_request_rate = 0
            for node in cell.get_nodes():
                total_request_rate += self.stations[node].get_request_rate()
            cell.set_request_rate(total_request_rate)

        if self.event_buffer is None:
            # Flatten the PMF matrix for event simulation
            values = self.pmf_matrix.values.flatten()
            ids = [(row, col) for row in self.pmf_matrix.index for col in self.pmf_matrix.columns]
            flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
            flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)
            self.event_buffer = simulate_environment(
                duration=3600 * 3,  # 3 hours
                timeslot=self.timeslot,
                global_rate=self.global_rate,
                pmf=flattened_pmf,
                stations=self.stations,
                distance_matrix=self.distance_matrix,
            )

        # Simulate the environment for the time slot
        if self.next_event_buffer is None:
            next_timeslot = (self.timeslot + 1) % 8
            next_day = num2days[(days2num[self.day] + 1) % 7] if next_timeslot == 0 else self.day
            next_pmf_matrix, next_global_rate = self._load_pmf_matrix_global_rate(next_day, next_timeslot)

            # Flatten the PMF matrix for event simulation
            values = next_pmf_matrix.values.flatten()
            ids = [(row, col) for row in next_pmf_matrix.index for col in next_pmf_matrix.columns]
            flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
            flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)
            self.next_event_buffer = simulate_environment(
                duration=3600 * 3,  # 3 hours
                timeslot=next_timeslot,
                global_rate=next_global_rate,
                pmf=flattened_pmf,
                stations=self.stations,
                distance_matrix=self.distance_matrix,
            )

            # Update time of each event
            for event in self.next_event_buffer:
                event.time += 3600 * 3

        for event in self.next_event_buffer:
            bisect.insort(self.event_buffer, event, key=lambda x: x.time)
        self.next_event_buffer = None

        self.background_thread = threading.Thread(target=self._precompute_poisson_events)
        self.background_thread.start()

        # Initialize environment time
        self.env_time = 0


    def _load_pmf_matrix_global_rate(self, day: str, timeslot: int) -> tuple[pd.DataFrame, float]:
        # Load the PMF matrix and global rate for a given day and time slot
        matrix_path = self.data_path + params['matrices_folder'] + '/' + str(timeslot).zfill(2) + '/'
        pmf_matrix = pd.read_csv(matrix_path + day.lower() + '-pmf-matrix.csv', index_col='osmid')

        # Convert index and columns to integers
        pmf_matrix.index = pmf_matrix.index.astype(int)
        pmf_matrix.columns = pmf_matrix.columns.astype(int)
        pmf_matrix.loc[10000, 10000] = 0.0

        if self.global_rate_dict is None:
            with open(self.data_path + 'utils/global_rates.pkl', 'rb') as f:
                self.global_rate_dict = pickle.load(f)

        global_rate = self.global_rate_dict[(day.lower(), timeslot)]

        return pmf_matrix, global_rate


    def _jump_to_next_state(self, steps: int = 0) -> list:
        failures = []

        # Iterate through each step
        for step in range(0, steps):
            # Increment the environment time by 30 seconds
            self.env_time += 30

            for cell_id, cell in self.cells.items():
                cell.update_eligibility_score(self.eligibility_decay)
                if cell_id == self.truck.get_cell().get_id():
                    cell.eligibility_score = 1.0

            total_failures = 0
            # Process all events that occurred before the updated environment time
            while self.event_buffer and self.event_buffer[0].time < self.env_time:
                # print("Entering in while")
                event = self.event_buffer.pop(0)
                failure, self.next_bike_id = event_handler(
                    event=event,
                    station_dict=self.stations,
                    nearby_nodes_dict=self.nearby_nodes_dict,
                    distance_matrix=self.distance_matrix,
                    system_bikes=self.system_bikes,
                    outside_system_bikes=self.outside_system_bikes,
                    logger=self.logger,
                    next_bike_id=self.next_bike_id
                )
                total_failures += failure
            failures.append(total_failures)

            # Log the updated state after processing events
            self.logger.log_state(
                step=int(self.env_time / 30),
                time=convert_seconds_to_hours_minutes((self.timeslot * 3 + 1) * 3600 + self.env_time)
            )

        # Return the total number of failures encountered
        return failures


    def _get_obs(self, action: int = None) -> np.array:
        # Encode time slot and day
        hour = divmod((self.timeslot * 3 + 1) * 3600 + self.env_time, 3600)[0]
        ohe_hour = np.array([1 if hour == i else 0 for i in range(24)], dtype=np.float32)
        ohe_day = np.array([1 if self.day == d else 0 for d in days2num.keys()], dtype=np.float32)

        if action is None:
            ohe_previous_move_action = np.zeros(self.action_space.n, dtype=np.float32)
        else:
            ohe_previous_move_action = np.array(
                [1 if action == actn.value else 0 for actn in Actions],
                dtype=np.float32
            )

        truck_cell_id = self.truck.get_cell().get_id()
        sorted_cells_keys = sorted(self.cells.keys())
        ohe_cell_position = np.array(
            [1 if truck_cell_id == cell_id else 0 for cell_id in sorted_cells_keys],
            dtype=np.float32
        )
        # cell_embedding = self.encoder(torch.tensor(self.zone_id_to_index[truck_cell_id])).detach().numpy()

        # Combine all features into a single observation array
        observation = np.concatenate([
            np.array([self.truck.get_load() / self.truck.max_load], dtype=np.float32),
            np.array([1 if len(self.system_bikes) > self.maximum_number_of_bikes - 5 else 0], dtype=np.float32),
            ohe_day,
            ohe_hour,
            ohe_previous_move_action,
            ohe_cell_position,
            # cell_embedding,
        ])

        return observation.astype(np.float32)


    def _get_reward(self, action: int) -> float:
        # ----------------------------
        # Compute expected departures per cell
        # ----------------------------
        expected_departures_per_cell = {}
        for event in self.event_buffer:
            if event.time > self.env_time + 3600 * 3:
                break
            if event.is_departure():
                start_location = event.get_trip().get_start_location()
                if start_location.get_station_id() != 10000:
                    cell = start_location.get_cell()
                    cell_id = cell.get_id()
                    expected_departures_per_cell[cell_id] = expected_departures_per_cell.get(cell_id, 0) + 1
            elif event.is_arrival():
                end_location = event.get_trip().get_end_location()
                if end_location.get_station_id() != 10000:
                    cell = end_location.get_cell()
                    cell_id = cell.get_id()
                    expected_departures_per_cell[cell_id] = expected_departures_per_cell.get(cell_id, 0) - 1

        # ----------------------------
        # Update critic scores and compute a penalty for critical zones
        # ----------------------------
        # global_critic_penalty = 0.0
        truck_cell_previous_critic_score = self.truck.get_cell().get_critic_score()
        for cell_id, cell in self.cells.items():
            critic_score = 0.0
            expected = 0
            if cell_id in expected_departures_per_cell:
                expected = expected_departures_per_cell[cell_id]
            available = cell.get_total_bikes()
            if expected > 0:
                # critic_score = max(0.0, 1.0 - (available / expected))
                critic_score = (expected - available) / (expected + available)
            cell.surplus_score = available - expected
            cell.set_critic_score(critic_score)
            # global_critic_penalty += critic_score

        if truck_cell_previous_critic_score > 0.0 >= self.truck.get_cell().get_critic_score():
            for cell in self.cells.values():
                cell.eligibility_score = 0.0

        # ----------------------------
        # Drop / Pick Up penalty
        # ----------------------------
        truck_cell = self.truck.get_cell()
        drop_bonus = 0.0
        if action == Actions.DROP_BIKE.value:
            if self.invalid_drop_action:
                drop_bonus = -1.0
                # drop_bonus = -0.4
            elif truck_cell.get_critic_score() > 0.0:
                drop_bonus = 1.0
                # drop_bonus = 0.3
                # drop_bonus = 0.1

        # if action == Actions.DROP_BIKE.value:
        #     if truck_cell.get_critic_score() > 0.0:
        #         if self.invalid_drop_action:
        #             # drop_bonus += -1.0
        #             drop_bonus += -0.4
        #         else:
        #             # drop_bonus += 1.0
        #             drop_bonus += 0.3

        pick_up_penalty = 0.0
        if action == Actions.PICK_UP_BIKE.value:
            if truck_cell.get_critic_score() > 0.0:
                pick_up_penalty += -0.1
                # pick_up_penalty += -0.05
                # pick_up_penalty += -0.02
            else:
                pick_up_penalty += 0.1
                # pick_up_penalty += 0.05
                # pick_up_penalty += 0.02

        # ----------------------------
        # Move penalty (e.g. discourage unnecessary movements)
        # ----------------------------
        action_penalty = 0.0
        if action in {Actions.UP.value, Actions.DOWN.value, Actions.LEFT.value, Actions.RIGHT.value}:
            is_2_step_loop = _detect_self_loops((action, self.last_move_action))
            self.last_move_action = action
            if is_2_step_loop:
                action_penalty += -0.1
                # action_penalty += -0.15

        if action in {Actions.PICK_UP_BIKE.value, Actions.DROP_BIKE.value}:
            is_2_step_loop = _detect_self_loops((action, self.last_move_action))
            self.last_move_action = action
            if is_2_step_loop:
                action_penalty += -0.25
                # action_penalty += -0.3
                # action_penalty += -0.5

        # ----------------------------
        # Bike charging penalty (e.g. discourage charging a bike that isn’t sufficiently discharged)
        # ----------------------------
        # TURN OFF THIS TO DISABLE BATTERY CHARGE
        bike_charge_penalty = 0.0
        if action == Actions.CHARGE_BIKE.value:
            is_critical = truck_cell.get_critic_score() > 0.0
            if self.truck.last_charge < 0.8:
                if is_critical:
                    bike_charge_penalty -= 0.1
                    # bike_charge_penalty -= 0.05
                else:
                    bike_charge_penalty -= 0.3
                    # bike_charge_penalty -= 0.2
            else:
                if is_critical:
                    bike_charge_penalty += 0.5
                    # bike_charge_penalty += 0.2

        # ----------------------------
        # Stay penalty
        # ----------------------------
        stay_penalty = 0.0
        if action == Actions.STAY.value:
            if truck_cell.get_critic_score() > 0.0:
                stay_penalty += -1.0
                # stay_penalty += -0.4

        # ----------------------------
        # Position penalty
        # ----------------------------
        position_penalty = 0.0
        # TRY WITH THIS
        if action in {Actions.UP.value, Actions.DOWN.value, Actions.LEFT.value, Actions.RIGHT.value, Actions.STAY.value}:
            position_penalty += -0.2 if truck_cell.eligibility_score > 0.7 and truck_cell.get_critic_score() <= 0.0 else 0.0
            # position_penalty += -0.2 if truck_cell.eligibility_score > 0.8 and truck_cell.get_critic_score() <= 0.0 else 0.0

        # ----------------------------
        # Combine all reward components with their weights
        # ----------------------------
        reward = (
            1.0
            + stay_penalty
            + position_penalty
            + action_penalty
            + drop_bonus
            + pick_up_penalty
            + bike_charge_penalty
        )

        return reward


    def _compute_bike_per_region_cost(self) -> float:
        total_cost = 0.0
        for cell_id, cell in self.cells.items():
            n_bikes = cell.get_total_bikes()
            cost = logistic_penalty_function(M=1, k=1, b=100, x=n_bikes)
            total_cost += cost
        return total_cost


    def _update_graph(self):
        """
        Update the attributes of the subgraph with regional metrics.

        Parameters:
            - subgraph (nx.Graph): The subgraph to update with regional metrics.
        """
        for cell_id, cell in self.cells.items():
            center_node = cell.get_center_node()

            # Initialize regional metrics
            bike_load = cell.get_total_bikes() / self.maximum_number_of_bikes
            demand_rate, arrival_rate = 0.0, 0.0
            battery_levels = []

            # Single loop to aggregate metrics
            for node in cell.nodes:
                station = self.stations[node]
                bikes = station.get_bikes()
                battery_levels.extend(bike.get_battery() / bike.get_max_battery() for bike in bikes.values())

                # Update regional metrics
                demand_rate += station.get_request_rate()
                arrival_rate += station.get_arrival_rate()

            # Normalize demand and arrival rates
            demand_rate /= self.global_rate
            arrival_rate /= self.global_rate

            # Low battery bikes
            low_battery_bikes = sum(1 for battery in battery_levels if battery <= 0.2)
            low_battery_bikes /= len(battery_levels) if len(battery_levels) > 0 else 1

            # Update attributes in the subgraph
            if center_node in self.cell_subgraph:
                self.cell_subgraph.nodes[center_node]['truck_cell'] = 1.0 if cell_id == self.truck.get_cell().get_id() else 0.0
                # self.cell_subgraph.nodes[center_node]['surplus_score'] = cell.get_mismatch_score() / self.maximum_number_of_bikes
                # TURN OFF THIS TO DISABLE BATTERY CHARGE
                # self.cell_subgraph.nodes[center_node]['low_battery_bikes'] = low_battery_bikes
                self.cell_subgraph.nodes[center_node]['total_bikes'] = bike_load
                self.cell_subgraph.nodes[center_node]['critic_score'] = cell.get_critic_score()
                self.cell_subgraph.nodes[center_node]['visits'] = cell.get_visits() / self.total_visits
                self.cell_subgraph.nodes[center_node]['eligibility_score'] = cell.eligibility_score
            else:
                raise ValueError(f"Node {center_node} not found in the subgraph.")


    def _get_truck_position(self) -> tuple[float, float]:
        truck_coords = self.nodes_dict.get(self.truck.get_position())
        normalized_coords = ((truck_coords[0] - self.min_lat) / (self.max_lat - self.min_lat),
                             (truck_coords[1] - self.min_lon) / (self.max_lon - self.min_lon))
        return normalized_coords


    def _adjust_depot_system_discrepancy(self):
        """
        Adjust the discrepancy of the depot + system bikes to the maximum number of bikes.
        This is done to prevent the truck from failing to load bikes from the depot.
        """
        depot_load = len(self.depot)
        system_load = len(self.system_bikes)

        if depot_load + system_load < self.maximum_number_of_bikes:
            n_bikes = self.maximum_number_of_bikes - depot_load - system_load
            for _ in range(n_bikes):
                bike = self.outside_system_bikes.pop(next(iter(self.outside_system_bikes)))
                bike.reset()
                self.depot[bike.get_bike_id()] = bike

    def _net_flow_based_repositioning(self, upper_bound: int = None) -> dict:
        # Compute net flow per cell
        net_flow_per_cell = {cell_id: 0 for cell_id in self.cells.keys()}
        for event in self.event_buffer:
            if upper_bound is not None and event.time > upper_bound:
                break
            if event.is_departure():
                station_id = event.trip.get_start_location().get_station_id()
                if station_id != 10000:
                    cell = self.stations[station_id].get_cell()
                    net_flow_per_cell[cell.get_id()] -= 1
            elif event.is_arrival():
                station_id = event.trip.get_end_location().get_station_id()
                if station_id != 10000:
                    cell = self.stations[station_id].get_cell()
                    net_flow_per_cell[cell.get_id()] += 1

        # Assign bikes to cells based on net flow
        bikes_per_cell = {cell_id: 5 for cell_id in self.cells.keys()}
        left_bikes = self.maximum_number_of_bikes - 5 * len(self.cells)
        total_negative_flow = sum(flow for flow in net_flow_per_cell.values() if flow < 0)
        bike_positioned = 0
        for cell_id, flow in net_flow_per_cell.items():
            if flow < 0:
                num_of_bikes = int((flow / total_negative_flow) * left_bikes)
                bikes_per_cell[cell_id] += num_of_bikes
                bike_positioned += num_of_bikes

        # Assign the remaining bikes to cells with negative flow randomly
        if bike_positioned < left_bikes:
            cell_ids = [cell_key for cell_key, flow in net_flow_per_cell.items() if flow < 0]
            random.shuffle(cell_ids)
            for cell_id in cell_ids:
                bikes_per_cell[cell_id] += 1
                bike_positioned += 1
                if bike_positioned == left_bikes:
                    break

        return bikes_per_cell