import networkx as nx

from networkx.algorithms.approximation import traveling_salesman_problem

from gymnasium_env.simulator.cell import Cell
from gymnasium_env.simulator.utils import truncated_gaussian, Actions
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.station import Station

# TODO: Move all the code into a class "TruckSimulator" to avoid the use of global variables and to make the code more modular and testable.

# ----------------------------------------------------------------------------------------------------------------------
"""
    This description is valid for the methods: move_up, move_down, move_left, move right
    Moves the truck in the respective direction and calculates the time and distance costs of this actions.

    Parameters:
        - truck (Truck): The truck to move
        - distance_lookup (dict[int, dict]): Dictionary of the distance between two stations, with station IDs as keys
        - cell_dict (dict): Dictionary of the Cells of the map
        - mean_velocity (int): Mean velocity of the truck moving to the next cell

    Returns:
        - time (int): Value of the time to move the destination cell
        - distance (int): Value of the total distance covered by the truck
        - border_hit (flag) : Flag to indicate if the truck tried to exit the map
    """


# Maps directional Actions to the corresponding adjacent cell key
ACTION_TO_DIRECTION: dict[int, str] = {
    Actions.UP.value: 'up',
    Actions.DOWN.value: 'down',
    Actions.LEFT.value: 'left',
    Actions.RIGHT.value: 'right',
}


def move(
    action: int,
    truck: Truck,
    distance_lookup: dict[int, dict],
    cell_dict: dict[int, Cell],
    mean_velocity: int
) -> tuple[int, int, bool]:
    """
    Move the truck in the direction specified by the action and compute the time and distance costs.

    Parameters:
        - action (int): One of Actions.UP, DOWN, LEFT, RIGHT (as .value)
        - truck (Truck): The truck to move
        - distance_lookup (dict[int, dict]): Dictionary of the distance between two stations, with station IDs as keys
        - cell_dict (dict): Dictionary of the Cells of the map
        - mean_velocity (int): Mean velocity of the truck moving to the next cell

    Returns:
        - time (int): Time cost to reach the destination cell
        - distance (int): Distance covered by the truck
        - border_hit (bool): True if the truck tried to exit the map, False otherwise
    """
    direction = ACTION_TO_DIRECTION.get(action)
    if direction is None:
        raise ValueError(f"Action {action} is not a valid move action. Expected one of {list(ACTION_TO_DIRECTION.keys())}")

    cell = truck.get_cell()
    target_cell = cell_dict.get(cell.get_adjacent_cells().get(direction))

    if target_cell is None:
        return 0, 0, True

    distance = distance_lookup[truck.get_position()][str(target_cell.get_center_node())]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(target_cell.get_center_node())
    truck.set_cell(target_cell)

    return time, distance, False


def drop_bike(
        truck: Truck,
        distance_lookup: dict[int, dict],
        mean_velocity: int,
        depot,
        system_bikes: dict,
        maximum_number_of_bikes: int,
        node: int = None
) -> tuple[int, int, bool]:
    """
    Unloads a bike to the center_node of the cell where the truck is located or a specified station "node".
    If the truck is empty, go get more bikes at the depot.
    WARNING: It is assumed that the depot + truck load is never zero when performing this method.
    WARNING: This function doesn't actually drop the bike, it just computes the time and distance to reach the dropping station.
             This is done to avoid the event handler to assign this bike before the truck has reach the station,
             i.e. the bike cannot be ready before it's dropped.
    --> Drop the bike manually in the simulator after time 't' is served and the simulation has advanced.

    Parameters:
        - truck (Truck): The truck to move
        - distance_lookup (dict[int, dict]): Dictionary of the distance between two stations, with station IDs as keys
        - mean_velocity (int): Mean velocity of the truck moving to the next cell
        - depot (Depot): class instance of the depot
        - node (int): Station ID for the bike drop. 

    Returns:
        - int: Value of the time round trip to the depot if more bikes are needed, else = 0 
        - int: Value of the total distance round trip to the depot if more bikes are needed, else = 0
    """
    time = 0
    distance = 0

    has_bike_to_drop = truck.get_load() > 0 or len(depot.bikes) > 0
    system_has_room = len(system_bikes) < maximum_number_of_bikes

    if not has_bike_to_drop or not system_has_room:
        return time, distance, True

    position = truck.get_position()
    target_node = truck.get_cell().get_center_node() if node is None else node

    # check if the truck is empty. If so, go get more bikes.
    if truck.get_load() == 0:
        distance = distance_lookup[truck.get_position()][str(depot.id)]
        velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
        time += int(distance * 3.6 / velocity_kmh)
        position = depot.id

        bikes_to_load = min(truck.max_load // 2, len(depot.bikes))
        bikes = {key: depot.bikes.pop(key) for key in list(depot.bikes.keys())[:bikes_to_load]}
        truck.set_load(bikes)

    if position != target_node:
        distance += distance_lookup[position][str(target_node)]
        velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
        time += int(distance * 3.6 / velocity_kmh)
        truck.set_position(target_node)

    truck.leaving_cell = truck.get_cell()

    # If you are asking where the "lock_bike" action is,
    # this can be found in the simulator after the advancing of the simulation time.
    # This is trivial since a bike cannot be ready at a station before the truck has reach this station.
    # This is done to avoid the event handler to assign a bike before is dropped down.

    return time, distance, False


def pick_up_bike(
    truck: Truck,
    station_dict: dict[int, Station],
    distance_lookup: dict[int, dict],
    mean_velocity: int,
    depot,
    system_bikes: dict
) -> tuple[int, int, bool]:
    """
    Picks up a bike from a station based on the lowest "bike_metric".
    The "bikes_metric" is a dictionary where the value of each bike is a normalized value proportional to distance and battery value.

    Parameters:
        - truck (Truck): The truck to move
        - station_dict (dict): Dictionary of system's stations.
        - distance_lookup (dict[int, dict]): Dictionary of the distance between two stations, with station IDs as keys
        - mean_velocity (int): Mean velocity of the truck moving to the next cell
        - depot (Depot): class instance of the depot
        - system_bikes (dict): Dictionary of bikes inside the system.

    Returns:
        - int: Value of the time round trip to the depot if more bikes are needed, else = 0 
        - int: Value of the total distance round trip to the depot if more bikes are needed, else = 0
        - bool = True
    """
    cell = truck.get_cell()
    # Flag no bikes in the cell, no bike picked up
    if cell.get_total_bikes() == 0:
        return 0, 0, True

    bike_dict = {}
    for station_id in cell.get_nodes():
        bike_dict.update(station_dict[station_id].get_bikes())

    occupied_stations = [
        station_id for station_id in cell.get_nodes()
        if station_dict[station_id].get_number_of_bikes() > 0
    ]
    if not occupied_stations:
        return 0, 0, True

    max_distance = max(
        distance_lookup[truck.get_position()][str(station_id)]
        for station_id in occupied_stations
    )

    # Compute the metric for each bike
    bikes_metric = {}
    for station_id in occupied_stations:
        station = station_dict[station_id]
        if station.get_number_of_bikes() == 0:
            continue
        distance = distance_lookup[truck.get_position()][str(station_id)]
        norm_distance = distance / max_distance if max_distance > 0 else 0.0

        for bike_id, bike in station.get_bikes().items():
            norm_batt = bike.get_battery() / bike.get_max_battery()
            # Low battery  → small norm_batt  → small metric → preferred ✅
            # Far distance → large norm_dist  → large metric → avoided  ✅
            # Weight battery more than distance since charging is the goal
            bikes_metric[bike_id] = 0.6 * norm_batt + 0.4 * norm_distance

    # Find the lowest metric bike
    bike_id = min(bikes_metric, key=bikes_metric.get)

    # Go get the chosen bike
    time, distance = _collect_bike(truck, distance_lookup, mean_velocity,
                                   depot, system_bikes, bike_id, bike_dict)

    return time, distance, False


def charge_bike(
    truck: Truck,
    station_dict: dict[int, Station],
    distance_lookup: dict[int, dict],
    mean_velocity: int,
    depot,
    system_bikes: dict
) -> tuple[int, int, bool]:
    """
    Picks up a bike with the lowest battery.
    WARNING: This function doesn't drop the bike afterward to avoid the event handler to assign this bike
                before the time of the charging is over, i.e. the bike cannot be ready before it's dropped.
    --> Drop the bike manually from the simulator after time 't' is served and the simulation has advanced.

    Parameters:
        - truck (Truck): The truck to move
        - station_dict (dict): Dictionary of system's stations.
        - distance_lookup (dict[int, dict]): Dictionary of the distance between two stations, with station IDs as keys
        - mean_velocity (int): Mean velocity of the truck moving to the next cell
        - depot (Depot): class instance of the depot
        - system_bikes (dict): Dictionary of bikes inside the system.

    Returns:
        - int: Value of the time round trip to the depot if more bikes are needed, else = 0 
        - int: Value of the total distance round trip to the depot if more bikes are needed, else = 0
        - bool = True
    """

    cell = truck.get_cell()
    
    # Flag no bike picked up
    if cell.get_total_bikes() == 0:
        return 0, 0, True

    bike_dict = {}
    for station_id in cell.get_nodes():
        bike_dict.update(station_dict[station_id].get_bikes())

    # Find the lowest metric bike
    bike_id = min(bike_dict, key=lambda bid: bike_dict[bid].get_battery())

    # Record battery ratio BEFORE the bike is collected
    truck.last_charge = bike_dict[bike_id].get_battery() / bike_dict[bike_id].get_max_battery()

    # Go get the chosen bike
    time, distance = _collect_bike(truck, distance_lookup, mean_velocity,
                                   depot, system_bikes, bike_id, bike_dict)

    # If you are asking where the "lock_bike" action is,
    # this can be found in the simulator after the advancing of the simulation time.
    # This is trivial since a bike cannot be ready at a station before the truck has reach this station.
    # This is done to avoid the event handler to assign a bike before is dropped down.

    return time, distance, False


def stay(truck: Truck) -> int:
    truck.leaving_cell = truck.get_cell()
    return 60

# ----------------------------------------------------------------------------------------------------------------------

def _collect_bike(
    truck: Truck,
    distance_lookup: dict[int, dict],
    mean_velocity: int,
    depot,
    system_bikes: dict,
    bike_id: int,        # already selected by the caller
    bike_dict: dict,     # flat {bike_id: bike} for the whole cell
) -> tuple[int, int]:
    """
    Shared logic: travel to bike, unlock it, load it (dumping to depot if full).
    Returns (time, distance). Caller is responsible for bike selection.
    """
    station = bike_dict[bike_id].get_station()
    distance = distance_lookup[truck.get_position()][str(station.get_station_id())]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    bike = station.unlock_bike(bike_id)
    if bike_id not in system_bikes:
        raise ValueError(f"Bike {bike_id} not in system")
    system_bikes.pop(bike_id)

    try:
        truck.load_bike(bike)
    except ValueError:  # truck full — dump to depot first
        distance_to_depot = distance_lookup[truck.get_position()][str(depot.id)]
        velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
        t_to_depot = int(distance_to_depot * 3.6 / velocity_kmh)
        distance += 2 * distance_to_depot
        time += 2 * t_to_depot
        while truck.get_load() > truck.max_load // 2:
            bk = truck.unload_bike()
            bk.reset()
            depot.bikes[bk.get_bike_id()] = bk
        truck.load_bike(bike)

    truck.set_position(station.get_station_id())
    truck.leaving_cell = truck.get_cell()
    return time, distance

# ----------------------------------------------------------------------------------------------------------------------

def tsp_rebalancing(surplus_nodes: dict, deficit_nodes: dict, starting_node, distance_lookup: dict[int, dict]):
    """
    Computes the system rebalancing using the traveling_salesman_problem algorithm to calculate the total time and the total path of the truck.

    Parameters:
        - surplus_nodes (dict): Dictionary of the nodes with excess bikes
        - deficit_nodes (dict): Dictionary of the nodes with deficit bikes
        - starting_node (int): Station ID to be the first station visited
        - distance_lookup (dict[int, dict]): Dictionary of the distance between two stations, with station IDs as keys

    Returns:
        - total_distance (int): Value of the total distance covered by the truck for the rebalancing
        - final_route (dict): Ordered dictionary of the stations to visit
    """

    all_nodes = list(surplus_nodes.keys()) + list(deficit_nodes.keys())
    tsp_graph = nx.Graph()

    # Check if there are nodes to process
    if not all_nodes:
        raise ValueError("No valid surplus or deficit nodes to rebalance.")

    # Distance calculation and all edges connections(between nodes in all_nodes)
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            node_i, node_j = all_nodes[i], all_nodes[j]
            distance = distance_lookup[node_i][str(node_j)]
            tsp_graph.add_edge(node_i, node_j, weight=distance)

    # Ensure starting node is included
    for node in all_nodes:
        distance = distance_lookup[starting_node][str(node)]
        tsp_graph.add_edge(starting_node, node, weight=distance)

    # Solve TSP to get the initial path
    tsp_path = traveling_salesman_problem(tsp_graph, cycle=False)

    # Variables to track progress
    total_distance = 0
    truck_bikes = 0
    final_route = []
    skipped_deficit_nodes = {}
    total_missing_bikes = 0

    # Process the TSP path dynamically
    current_node = starting_node

    for node in tsp_path:
        if node not in surplus_nodes and node not in deficit_nodes:
            continue

        distance = distance_lookup[current_node][str(node)]
        total_distance += distance
        final_route.append(node)

        # If it's a surplus node, pick up bikes, otherwise drop them
        if node in surplus_nodes:
            truck_bikes += surplus_nodes.pop(node, 0)
        elif node in deficit_nodes:
            deficit_demand = -deficit_nodes[node]
            if truck_bikes >= deficit_demand:
                # Enough bikes: drop them
                truck_bikes -= deficit_demand
                deficit_nodes.pop(node)
            else:
                # Not enough bikes: track it and move on
                skipped_deficit_nodes[node] = deficit_demand
                total_missing_bikes += deficit_demand

        # Move to the next node
        current_node = node

        # Once the truck has enough bikes to satisfy all skipped deficits, go back in an efficient order
        if 0 < total_missing_bikes <= truck_bikes:
            # Solve a new TSP for skipped deficit nodes
            backtrack_graph = nx.Graph()
            skipped_list = list(skipped_deficit_nodes.keys())

            for i in range(len(skipped_list)):
                for j in range(i + 1, len(skipped_list)):
                    node_i, node_j = skipped_list[i], skipped_list[j]
                    distance = distance_lookup[node_i][str(node_j)]
                    backtrack_graph.add_edge(node_i, node_j, weight=distance)

            # Ensure we start from the last node we visited
            for n in skipped_list:
                distance = distance_lookup[current_node][str(n)]
                backtrack_graph.add_edge(current_node, n, weight=distance)

            # Solve TSP for revisiting skipped nodes
            backtrack_path = traveling_salesman_problem(backtrack_graph, cycle=False)

            # Visit skipped nodes
            for n in backtrack_path:
                if n in skipped_deficit_nodes:
                    distance = distance_lookup[current_node][str(n)]
                    total_distance += distance
                    final_route.append(n)

                    # Drop bikes
                    bikes_needed = skipped_deficit_nodes[n]
                    truck_bikes -= bikes_needed
                    total_missing_bikes -= bikes_needed
                    skipped_deficit_nodes.pop(n)

                    # Move to the next node
                    current_node = n

    return total_distance, final_route