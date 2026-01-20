# gymnasium_env/simulator/__init__.py
from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.cell import Cell
from gymnasium_env.simulator.event import Event, EventType
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import Trip
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import (
    Actions,
    ActionHistoryEncoder,
    Logger,
    kahan_sum,
    logistic_penalty_function,
    compute_distance,
    generate_poisson_events,
    convert_seconds_to_hours_minutes,
    truncated_gaussian,
    nodes_within_radius,
    load_cells_from_csv,
    initialize_graph,
    initialize_bikes,
    initialize_stations,
    initialize_cells_subgraph,
    plot_graph,
    plot_graph_with_grid,
    DEFAULT_PATHS,
    DAYS_TO_NUM,
    NUM_TO_DAYS,
    ACTION_TO_STR,
    STR_TO_ACTION,
    detect_self_loops,
)
from gymnasium_env.simulator.bike_simulator import simulate_environment, event_handler
from gymnasium_env.simulator.truck_simulator import (
    move_up,
    move_down,
    move_left,
    move_right,
    drop_bike,
    pick_up_bike,
    charge_bike,
    stay,
    tsp_rebalancing,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Edoardo Scarpel"

__all__ = [
    "Bike",
    "Cell",
    "Event",
    "EventType",
    "Station",
    "Trip",
    "Truck",
    "Actions",
    "ActionHistoryEncoder",
    "Logger",
    "kahan_sum",
    "logistic_penalty_function",
    "compute_distance",
    "generate_poisson_events",
    "convert_seconds_to_hours_minutes",
    "truncated_gaussian",
    "nodes_within_radius",
    "load_cells_from_csv",
    "initialize_graph",
    "initialize_bikes",
    "initialize_stations",
    "initialize_cells_subgraph",
    "plot_graph",
    "plot_graph_with_grid",
    "simulate_environment",
    "event_handler",
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "drop_bike",
    "pick_up_bike",
    "charge_bike",
    "stay",
    "tsp_rebalancing",
    "DEFAULT_PATHS",
    "DAYS_TO_NUM",
    "NUM_TO_DAYS",
    "ACTION_TO_STR",
    "STR_TO_ACTION",
    "detect_self_loops",
]
