import polars as pl
import numpy as np

from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.trip import Trip, TripSample
from gymnasium_env.simulator.event import EventType, Event
from gymnasium_env.simulator.utils import generate_poisson_events, truncated_gaussian
from gymnasium_env.simulator.env_logger import EnvLogger

# ----------------------------------------------------------------------------------------------------------------------

def event_handler(
        event: Event,
        station_dict: dict[int, Station],
        nearby_nodes_dict: dict[int, list[int]],
        distance_lookup: dict[int, dict],
        system_bikes: dict[int, Bike],
        outside_system_bikes: dict[int, Bike],
        traveling_bikes: dict[int, Bike],
        depot,
        maximum_number_of_bikes: int,
        truck_load: int,
        logger: EnvLogger = None,
        logging_state_and_trips: bool = False
) -> int:
    """
    Handle the event based on its type.

    Parameters:
        - event (Event): The event object to be processed.
        - station_dict (dict): A dictionary containing the stations in the network.
        - nearby_nodes_dict (dict): A dictionary containing the nearby nodes for each station.
        - distance_lookup (dict): A dictionary containing the distance lookup between stations.
        - system_bikes (dict): A dictionary containing the bikes in the system.
        - outside_system_bikes (dict): A dictionary containing the bikes outside the system.
        - logger (Logger): To log the event

    Returns:
        - bool: A boolean indicating whether the event failed or not.
    """
    failure = 0
    if event.is_departure():
        trip = departure_handler(
            trip=event.get_trip(),
            station_dict=station_dict,
            nearby_nodes_dict=nearby_nodes_dict,
            distance_lookup=distance_lookup,
            outside_system_bikes=outside_system_bikes,
            traveling_bikes=traveling_bikes,
            depot=depot
        )
        if logging_state_and_trips:
            logger.log_trip(trip)
        if trip.is_failed():
            failure = 1
            if logger is not None:
                logger.log_no_available_bikes(trip.get_start_location(), trip.get_end_location())
    else:
        arrival_handler(
            trip=event.get_trip(),
            system_bikes=system_bikes,
            outside_system_bikes=outside_system_bikes,
            depot=depot,
            maximum_number_of_bikes=maximum_number_of_bikes,
            traveling_bikes=traveling_bikes,
            truck_load=truck_load
        )

    return failure


def departure_handler(
        trip: Trip,
        station_dict: dict,
        nearby_nodes_dict: dict[int, list[int]],
        distance_lookup: dict[int, dict],
        outside_system_bikes: dict[int, Bike],
        traveling_bikes: dict[int, Bike],
        depot
) -> Trip:
    """
    Handle the departure of a trip from the starting station.

    Parameters:
        - trip (Trip): The trip object to be processed.
        - station_dict (dict): A dictionary containing the stations in the network.
        - nearby_nodes_dict (dict): A dictionary containing the nearby nodes for each station.
        - distance_lookup (dict): A dictionary containing the distance lookup between stations.
        - outside_system_bikes (dict): A dictionary containing the bikes outside the system.

    Returns:
        - Trip: The trip object after processing.
    """
    start_station = trip.get_start_location()
    start_station_id = start_station.get_station_id()

    # Check if the starting station is outside the system
    if start_station_id == 10000:
        # Draw from outside_system_bikes first, then depot, never create new ones
        if len(outside_system_bikes) > 0:
            bike = outside_system_bikes.pop(next(iter(outside_system_bikes)))
        elif len(depot.bikes) > 0:
            bike = depot.bikes.pop(next(iter(depot.bikes)))
            bike.set_battery(bike.get_max_battery())
        else:
            # Fleet is fully deployed, external trip just fails silently
            trip.set_failed(True)
            return trip
        trip.set_bike(bike)
        trip.set_failed(False)
        traveling_bikes[bike.get_bike_id()] = bike
        return trip

    # Here starting station is inside the system
    start_station.get_cell().add_departure()
    # Check if there are any bikes available at the starting station
    if start_station.get_number_of_bikes() > 0:
        bike = start_station.unlock_bike()
        if bike.get_battery() > trip.get_distance()/1000:
            trip.set_bike(bike)
            trip.set_failed(False)
            traveling_bikes[bike.get_bike_id()] = bike
            return trip
        else:
            start_station.lock_bike(bike)

    # Check if there are any bikes available at nearby stations
    nodes_dist_dict = {
        node_id: distance_lookup[start_station_id][str(node_id)]
        for node_id in nearby_nodes_dict[start_station_id]
    }
    for node_id, _ in sorted(nodes_dist_dict.items(), key=lambda item: item[1]):
        if station_dict[node_id].get_number_of_bikes() > 0:
            bike = station_dict[node_id].unlock_bike()
            if bike.get_battery() > trip.get_distance()/1000:
                trip.set_deviated_location(station_dict[node_id])
                trip.set_bike(bike)
                trip.set_failed(False)
                trip.set_deviated(True)
                traveling_bikes[bike.get_bike_id()] = bike
                return trip
            else:
                station_dict[node_id].lock_bike(bike)

    # Here the trip departure was inside, the station was empty and all nearby stations were also empty.
    trip.set_failed(True)
    start_station.get_cell().add_failure()
    return trip


def arrival_handler(
        trip: Trip,
        system_bikes: dict[int, Bike],
        outside_system_bikes: dict[int, Bike],
        depot,
        maximum_number_of_bikes: int,
        traveling_bikes: dict[int, Bike],
        truck_load: int
):
    """
    Handle the arrival of a trip at the destination station.

    Parameters:
        - trip (Trip): The trip object to be processed.
    """
    if trip.is_failed():
        return

    start_station = trip.get_start_location()
    end_station = trip.get_end_location()

    bike = trip.get_bike()
    # TURN OFF THIS TO DISABLE BATTERY CHARGE
    bike.set_battery(bike.get_battery() - trip.get_distance()/1000)
    traveling_bikes.pop(bike.get_bike_id())

    # Move the bike to the outside system if the destination station is outside the system
    if end_station.get_station_id() == 10000:
        bike.reset()
        system_bikes.pop(bike.get_bike_id())
        # Only return to depot if fleet isn't already at cap
        total = len(system_bikes) + len(depot.bikes) + truck_load
        if total < maximum_number_of_bikes:
            depot.bikes[bike.get_bike_id()] = bike
        else:
            outside_system_bikes[bike.get_bike_id()] = bike  # truly excess, let it go
        return

    # Move the bike back to the system if the starting station is outside the system
    if start_station.get_station_id() == 10000:
        bike.set_battery(bike.get_max_battery())
        system_bikes[bike.get_bike_id()] = bike

    end_station.lock_bike(bike)

# ----------------------------------------------------------------------------------------------------------------------

def simulate_events(
        duration: int,
        timeslot: int,
        global_rate: float,
        pmf: pl.DataFrame,
        distance_lookup: dict[int, dict],
) -> list[TripSample]:
    """
    Sample trip requests for a timeslot via Poisson process.
    Returns lightweight TripSample records — no Trip or Event objects created.
    """
    # Convert pmf for faster access
    pmf_data = pmf.to_dicts()
    cumsum_values = pmf.select("cumsum").to_series().to_numpy()

    # Simulate requests
    event_times = generate_poisson_events(global_rate, duration)
    samples = []

    for event_time in event_times:
        idx = np.searchsorted(cumsum_values, np.random.rand())
        row = pmf_data[idx]['id']
        origin_id = row['node_id']
        dest_id = row['col_id']

        if origin_id == 10000 or dest_id == 10000:
            distance = 0
            travel_time = 1
        else:
            velocity_kmh = truncated_gaussian(5, 25, 15, 5)
            distance = int(distance_lookup[origin_id][str(dest_id)])
            travel_time = int(distance * 3.6 / velocity_kmh)

        abs_start = event_time + 3600 * (3 * timeslot + 1)

        samples.append(TripSample(
            dep_time=event_time,
            travel_time=travel_time,
            start_station_id=origin_id,
            end_station_id=dest_id,
            distance=distance,
            abs_start_time=abs_start,
        ))

    samples.sort(key=lambda s: s.dep_time)
    return samples


def build_events(
    samples: list[TripSample],
    stations: dict,
    time_offset: int = 0,
) -> list[Event]:
    """
    Build fresh Trip + Event pairs from TripSample records.
    Each Trip is shared between its departure and arrival Event.
    bike=None on all trips — departure_handler assigns it at runtime.
    """
    events = []

    for s in samples:
        trip = Trip(
            start_time=s.abs_start_time,
            end_time=s.abs_start_time + s.travel_time,
            start_location=stations[s.start_station_id],
            end_location=stations[s.end_station_id],
            bike=None,
            distance=s.distance,
        )
        events.append(Event(
            time=s.dep_time + time_offset,
            event_type=EventType.DEPARTURE,
            trip=trip,
        ))
        events.append(Event(
            time=s.dep_time + s.travel_time + time_offset,
            event_type=EventType.ARRIVAL,
            trip=trip,
        ))

    events.sort(key=lambda e: e.time)
    return events