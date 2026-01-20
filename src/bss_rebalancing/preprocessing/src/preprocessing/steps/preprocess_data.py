"""
Preprocess trip data to compute Poisson rates.

This module processes raw trip data and computes Poisson request rates
for each station pair, organized by day and time slot.
"""

import argparse
import os
from typing import List

import networkx as nx
import osmnx as ox
import pandas as pd
from tqdm import tqdm

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.graph import initialize_graph
from preprocessing.core.utils import count_specific_day


def compute_poisson_rates(
    df: pd.DataFrame,
    year: int,
    months: List[int],
    day_of_week: str,
    time_slot: int,
) -> pd.DataFrame:
    """
    Compute the Poisson request rates for each station pair on a specific day and time slot.

    Parameters:
        df: DataFrame containing the trip data.
        year: The year of the data.
        months: List of months to process.
        day_of_week: Day of the week to filter by (e.g., "Monday", "Tuesday").
        time_slot: Time slot (0-7), each representing a 3-hour interval starting from 1:00 am.

    Returns:
        DataFrame containing the Poisson request rates for each station pair.
    """
    df = df.copy()
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["day_of_week"] = df["starttime"].dt.day_name()
    df["hour"] = df["starttime"].dt.hour

    # Filter by day of week
    df_filtered = df[df["day_of_week"] == day_of_week]

    # Define time slot hours
    start_hour = 1 + time_slot * 3
    end_hour = start_hour + 3

    # Filter by time slot
    df_filtered = df_filtered[(df_filtered["hour"] >= start_hour) & (df_filtered["hour"] < end_hour)]

    # Calculate number of days in the time period
    num_days = sum(count_specific_day(year, month, day_of_week) for month in months)

    # Total time in seconds for this time slot across all days
    total_time_seconds = num_days * 3 * 3600

    # Group by station pairs
    grouped_df = (
        df_filtered.groupby(
            [
                "start station id",
                "start station name",
                "start station latitude",
                "start station longitude",
                "end station id",
                "end station name",
                "end station latitude",
                "end station longitude",
            ]
        )
        .size()
        .reset_index(name="trip_count")
    )

    # Compute Poisson rate
    rate_df = grouped_df.copy()
    rate_df["lambda"] = rate_df["trip_count"] / total_time_seconds
    rate_df["day_of_week"] = day_of_week
    rate_df["time_slot"] = time_slot

    return rate_df


def map_trip_to_graph_node(
    G: nx.MultiDiGraph,
    trip_df: pd.DataFrame,
    filtered_stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map the start and end stations of the trips to the nearest nodes in the graph.

    Parameters:
        G: The graph representing the road network.
        trip_df: The DataFrame containing the trip data.
        filtered_stations: DataFrame of stations within the graph area.

    Returns:
        DataFrame with stations mapped to graph nodes.
    """
    valid_station_ids = set(filtered_stations["start station id"])

    tbar = tqdm(
        total=trip_df.shape[0],
        desc="Mapping Trips to Graph Nodes",
        leave=False,
        position=1,
        dynamic_ncols=True,
    )

    rows_to_drop = []

    for index, row in trip_df.iterrows():
        start_station_id = row["start station id"]
        end_station_id = row["end station id"]

        start_inside = start_station_id in valid_station_ids
        end_inside = end_station_id in valid_station_ids

        if start_inside and end_inside:
            start_node = ox.distance.nearest_nodes(G, Y=row["start station latitude"], X=row["start station longitude"])
            end_node = ox.distance.nearest_nodes(G, Y=row["end station latitude"], X=row["end station longitude"])

            trip_df.at[index, "start station id"] = start_node
            trip_df.at[index, "end station id"] = end_node
            trip_df.at[index, "start station latitude"] = G.nodes[start_node]["y"]
            trip_df.at[index, "start station longitude"] = G.nodes[start_node]["x"]
            trip_df.at[index, "end station latitude"] = G.nodes[end_node]["y"]
            trip_df.at[index, "end station longitude"] = G.nodes[end_node]["x"]

        elif start_inside:
            start_node = ox.distance.nearest_nodes(G, Y=row["start station latitude"], X=row["start station longitude"])
            trip_df.at[index, "start station id"] = start_node
            trip_df.at[index, "end station id"] = 10000  # External node marker
            trip_df.at[index, "start station latitude"] = G.nodes[start_node]["y"]
            trip_df.at[index, "start station longitude"] = G.nodes[start_node]["x"]
            trip_df.at[index, "end station latitude"] = 0.0
            trip_df.at[index, "end station longitude"] = 0.0

        elif end_inside:
            end_node = ox.distance.nearest_nodes(G, Y=row["end station latitude"], X=row["end station longitude"])
            trip_df.at[index, "start station id"] = 10000  # External node marker
            trip_df.at[index, "end station id"] = end_node
            trip_df.at[index, "start station latitude"] = 0.0
            trip_df.at[index, "start station longitude"] = 0.0
            trip_df.at[index, "end station latitude"] = G.nodes[end_node]["y"]
            trip_df.at[index, "end station longitude"] = G.nodes[end_node]["x"]

        else:
            rows_to_drop.append(index)

        tbar.update(1)

    trip_df.drop(rows_to_drop, inplace=True)
    trip_df.reset_index(drop=True, inplace=True)

    return trip_df


def initialize_rate_matrix(G: nx.MultiDiGraph, rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize a rate matrix based on the Poisson rates.

    Parameters:
        G: The graph representing the road network.
        rate_df: DataFrame containing the Poisson rates for station pairs.

    Returns:
        Square rate matrix indexed by node IDs.
    """
    node_ids = ox.graph_to_gdfs(G, edges=False).index
    df = pd.DataFrame(index=node_ids, columns=node_ids, dtype="float64")

    # Add external node column/row
    df.loc[10000] = 0.0
    df[10000] = 0.0
    df = df.fillna(0.0)

    for _, row in rate_df.iterrows():
        i = row["start station id"]
        j = row["end station id"]
        rate = row["lambda"]
        df.at[i, j] = rate

    return df


def run(config: PreprocessingConfig) -> None:
    """
    Run the preprocess data step.

    Parameters:
        config: The preprocessing configuration.
    """
    # Ensure directories exist
    os.makedirs(config.utils_path, exist_ok=True)

    print("Initializing the graph...")
    graph = initialize_graph(
        places=config.place,
        network_type=config.network_type,
        graph_path=config.graph_path,
        remove_isolated_nodes=True,
        simplify_network=True,
        nodes_to_remove=config.nodes_to_remove,
        bbox=config.bbox,
    )

    print(f"\nProcessing data for year {config.year} and months {config.months}...")

    # Load trip data
    trip_df = pd.DataFrame()
    for month in config.months:
        path = os.path.join(config.trips_path, f"{config.year}{str(month).zfill(2)}-bluebikes-tripdata.csv")
        if os.path.isfile(path):
            trip_df = pd.concat([trip_df, pd.read_csv(path)], ignore_index=True)
        else:
            print(f"Trip data file for month {month} does not exist. Skipping...")

    if trip_df.empty:
        print("No trip data found. Exiting.")
        return

    # Load filtered stations
    filtered_stations_path = os.path.join(config.utils_path, "filtered_stations.csv")
    if not os.path.exists(filtered_stations_path):
        print(f"Warning: {filtered_stations_path} not found. Creating from trip data...")
        # Create filtered stations from trip data
        stations = trip_df[["start station id", "start station name", "start station latitude", "start station longitude"]].drop_duplicates()
        stations.to_csv(filtered_stations_path, index=False)

    filtered_stations = pd.read_csv(filtered_stations_path)

    tbar = tqdm(
        total=len(config.days_of_week) * config.num_time_slots,
        desc="Processing Data",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )

    for day in config.days_of_week:
        for timeslot in range(config.num_time_slots):
            # Compute Poisson rates
            poisson_rates_df = compute_poisson_rates(trip_df, config.year, config.months, day, timeslot)

            # Map trips to graph nodes
            poisson_rates_df = map_trip_to_graph_node(graph, poisson_rates_df, filtered_stations)

            # Save Poisson rates
            rates_path = os.path.join(
                config.data_path,
                "rates",
                config.month_str,
                str(timeslot).zfill(2),
            )
            os.makedirs(rates_path, exist_ok=True)
            poisson_rates_df.to_csv(os.path.join(rates_path, f"{day.lower()}-poisson-rates.csv"), index=False)

            # Initialize and save rate matrix
            rate_matrix = initialize_rate_matrix(graph, poisson_rates_df)
            matrix_path = os.path.join(
                config.data_path,
                "matrices",
                config.month_str,
                str(timeslot).zfill(2),
            )
            os.makedirs(matrix_path, exist_ok=True)
            rate_matrix.to_csv(os.path.join(matrix_path, f"{day.lower()}-rate-matrix.csv"), index=True)

            tbar.update(1)


def main():
    """CLI entry point for preprocess_data."""
    parser = argparse.ArgumentParser(description="Preprocess trip data to compute Poisson rates.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to data directory.")
    parser.add_argument("--year", type=int, default=DEFAULT_CONFIG.year, help="Year of data to process.")
    parser.add_argument("--months", type=str, default="9,10", help="Comma-separated months to process.")

    args = parser.parse_args()
    months = [int(m.strip()) for m in args.months.split(",")]

    config = PreprocessingConfig(data_path=args.data_path, year=args.year, months=months)
    run(config)


if __name__ == "__main__":
    main()
