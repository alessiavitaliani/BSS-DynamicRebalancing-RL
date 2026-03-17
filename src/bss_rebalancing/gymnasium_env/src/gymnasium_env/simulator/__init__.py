# gymnasium_env/simulator/__init__.py
from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.cell import Cell
from gymnasium_env.simulator.event import Event, EventType
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.trip import Trip, TripSample
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import (
    Actions,
    logistic_penalty_function,
    generate_poisson_events,
    truncated_gaussian,
    convert_seconds_to_hours_minutes_day,
    initialize_graph,
    initialize_bikes,
    initialize_stations,
    DEFAULT_PATHS,
    DAYS_TO_NUM,
    NUM_TO_DAYS,
    ACTION_TO_STR,
    STR_TO_ACTION,
    detect_self_loops,
    load_preprocessed_data,
    flatten_pmf_matrix,
    cache_precomputed_buffers,
    load_cached_buffers
)
from gymnasium_env.simulator.bike_simulator import simulate_events, build_events, event_handler
from gymnasium_env.simulator.truck_simulator import (
    move,
    drop_bike,
    pick_up_bike,
    charge_bike,
    stay,
    tsp_rebalancing,
)
from gymnasium_env.simulator.env_logger import EnvLogger

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
    "TripSample",
    "Truck",
    "Actions",
    "logistic_penalty_function",
    "generate_poisson_events",
    "truncated_gaussian",
    "convert_seconds_to_hours_minutes_day",
    "initialize_graph",
    "initialize_bikes",
    "initialize_stations",
    "simulate_events",
    "build_events",
    "event_handler",
    "move",
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
    "EnvLogger",
    "load_preprocessed_data",
    "flatten_pmf_matrix",
    "cache_precomputed_buffers",
    "load_cached_buffers"
]
