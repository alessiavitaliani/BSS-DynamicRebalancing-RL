"""
Preprocess trip data to compute Poisson rates using Polars.

This module processes raw trip data and computes Poisson request rates
for each station pair, organized by day and time slot.
"""

import argparse
import os
import math
import pickle

import networkx as nx
import numpy as np
import osmnx as ox
import polars as pl

from haversine import haversine, Unit
from tqdm import tqdm
from typing import List

from preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.core.graph import initialize_graph


def filter_and_map_stations(
    trip_df: pl.DataFrame,
    graph: nx.MultiDiGraph,
    bbox: tuple = None,
    max_distance_meters: float = 100.0
) -> pl.DataFrame:
    """
    Extract unique stations from trips, optionally filter by bbox, and map to graph nodes.

    Parameters:
        trip_df: Polars DataFrame containing trip data
        graph: NetworkX graph of the road network
        bbox: Optional (north, south, east, west) bounding box
        max_distance_meters: Maximum distance for station-to-node mapping

    Returns:
        DataFrame with mapped stations (id, name, lat, lon, nearest_node, node_lat, node_lon)
    """
    # Extract and merge starting and ending stations
    starting_stations = (
        trip_df
        .select(['start station id', 'start station name',
                'start station latitude', 'start station longitude'])
        .unique()
        .rename({
            'start station id': 'id',
            'start station name': 'name',
            'start station latitude': 'latitude',
            'start station longitude': 'longitude',
        })
    )

    ending_stations = (
        trip_df
        .select(['end station id', 'end station name',
                'end station latitude', 'end station longitude'])
        .unique()
        .rename({
            'end station id': 'id',
            'end station name': 'name',
            'end station latitude': 'latitude',
            'end station longitude': 'longitude',
        })
    )

    stations = starting_stations.join(
        ending_stations,
        on=['id', 'name', 'latitude', 'longitude'],
        how="full",
        coalesce=True
    )

    # Apply bbox filter if provided
    if bbox:
        north, south, east, west = bbox
        stations = stations.filter(
            (pl.col('latitude') >= south) &
            (pl.col('latitude') <= north) &
            (pl.col('longitude') >= west) &
            (pl.col('longitude') <= east)
        )

    # Map stations to nearest graph nodes
    lons = stations['longitude'].to_list()
    lats = stations['latitude'].to_list()
    nearest_node_ids = ox.distance.nearest_nodes(graph, lons, lats)

    # Get node coordinates and calculate distances
    node_coords = [(graph.nodes[nid]['y'], graph.nodes[nid]['x'])
                   for nid in nearest_node_ids]

    distances = [
        haversine((lats[i], lons[i]), node_coords[i], unit=Unit.METERS)
        for i in range(len(lats))
    ]

    mapped_stations = stations.with_columns([
        pl.Series('nearest_node', nearest_node_ids),
        pl.Series('node_latitude', [nc[0] for nc in node_coords]),
        pl.Series('node_longitude', [nc[1] for nc in node_coords]),
        pl.Series('distance_meters', distances)
    ])

    # Filter by maximum distance
    mapped_stations = mapped_stations.filter(
        pl.col('distance_meters') <= max_distance_meters
    )

    return mapped_stations


def filter_trips_and_compute_timeslots(
    trip_df: pl.DataFrame,
    valid_station_ids: List[int]
) -> pl.DataFrame:
    """
    Filter trips by valid stations and assign weekday/timeslot labels.

    Timeslot logic: Each day starts at 01:00 AM (not midnight).
    - Timeslots 0-6: 3-hour periods starting at 01:00, 04:00, ..., 19:00
    - Timeslot 7: 22:00-23:59 (2 hours)
    - Timeslot 8: 00:00-00:59 (1 hour, assigned to previous day)
    - Timeslot 8 is later merged into timeslot 7

    Parameters:
        trip_df: Raw trip DataFrame
        valid_station_ids: List of valid station IDs

    Returns:
        Filtered DataFrame with weekday and timeslot columns
    """
    datetime_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.2f",
        "%Y-%m-%d %H:%M:%S%.3f",
        "%Y-%m-%d %H:%M:%S%.4f",
    ]

    filtered_trips = (
        trip_df
        .filter(
            pl.col("start station id").is_in(valid_station_ids) |
            pl.col("end station id").is_in(valid_station_ids)
        )
        .with_columns([
            # Replace invalid stations with external node marker (10000)
            pl.when(pl.col("start station id").is_in(valid_station_ids))
              .then(pl.col("start station id"))
              .otherwise(10000)
              .alias("start station id"),
            pl.when(pl.col("end station id").is_in(valid_station_ids))
              .then(pl.col("end station id"))
              .otherwise(10000)
              .alias("end station id"),
            # Parse datetime columns
            pl.coalesce([
                pl.col("starttime").str.to_datetime(format=fmt, strict=False)
                for fmt in datetime_formats
            ]).alias("starttime"),
            pl.coalesce([
                pl.col("starttime").str.to_date(format=fmt, strict=False)
                for fmt in datetime_formats
            ]).alias("startday"),
        ])
        .with_columns([
            # Shift hour 0 to previous day
            pl.when(pl.col("starttime").dt.hour() == 0)
              .then(pl.col("startday") - pl.duration(days=1))
              .otherwise(pl.col("startday"))
              .alias("startday"),
        ])
        .with_columns([
            # Assign weekday (ISO: Monday=1, Sunday=7)
            pl.col("startday").dt.weekday().alias("weekday"),
            # Assign timeslot
            pl.when(pl.col("starttime").dt.hour() == 0)
              .then(8)  # Midnight hour gets timeslot 8 (merged to 7 later)
              .otherwise((pl.col("starttime").dt.hour() - 1) // 3)
              .alias("timeslot")
        ])
        .with_columns([
            # Ensure weekday is 1-7 (wrap Sunday=0 to 7)
            pl.when(pl.col("weekday") == 0)
              .then(7)
              .otherwise(pl.col("weekday"))
              .alias("weekday")
        ])
    )

    return filtered_trips


def compute_timeslot_counts(trip_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute total hours available for each weekday-timeslot combination.

    Parameters:
        trip_df: Trip DataFrame with starttime column

    Returns:
        DataFrame with columns: weekday, timeslot, total_seconds
    """
    timeslot_counts = (
        trip_df
        .group_by(['weekday', 'timeslot'])
        .agg(
            pl.col('startday').n_unique().alias('num_occurrences')
        )
        .with_columns(
            hours_per_timeslot = pl.when(pl.col('timeslot') == 7)
                .then(2)  # 22:00-23:59 is 2 hours
                .when(pl.col('timeslot') == 8)
                .then(1)  # 00:00-00:59 is 1 hour
                .otherwise(3)  # All others are 3 hours
        )
        .with_columns(
            total_hours = pl.col('num_occurrences') * pl.col('hours_per_timeslot')
        )
        .with_columns(
            # Map timeslot 8 to 7
            timeslot = pl.when(pl.col('timeslot') == 8)
                .then(7)
                .otherwise(pl.col('timeslot'))
        )
        .group_by(['weekday', 'timeslot'])
        .agg(
            pl.col('total_hours').sum().alias('total_hours')
        )
        .sort(['weekday', 'timeslot'], descending=False)
    )

    return timeslot_counts


def compute_poisson_rates(
    filtered_trips: pl.DataFrame,
    timeslot_counts: pl.DataFrame
) -> pl.DataFrame:
    """
    Compute Poisson rates (rate = trips / seconds) for each station pair.

    Parameters:
        filtered_trips: Filtered trips with weekday/timeslot
        timeslot_counts: Total hours per weekday-timeslot

    Returns:
        DataFrame with columns: start/end station id, weekday, timeslot, rate
    """
    rates = (
        filtered_trips
        .with_columns(
            # Map timeslot 8 to 7
            timeslot = pl.when(pl.col('timeslot') == 8)
                .then(7)
                .otherwise(pl.col('timeslot'))
        )
        .group_by(['start station id', 'end station id', 'weekday', 'timeslot'])
        .agg(
            pl.len().alias("total_trips")
        )
        .join(timeslot_counts, on=['weekday', 'timeslot'], how='left')
        .with_columns(
            rate = pl.col('total_trips') / (pl.col('total_hours') * 3600)
        )
    )

    return rates


def compute_rate_matrix(
    graph: nx.MultiDiGraph,
    rate_df: pl.DataFrame,
    weekday: int,
    timeslot: int
) -> np.ndarray:
    """
    Create a square rate matrix for a specific weekday-timeslot.

    Parameters:
        graph: NetworkX graph
        rate_df: DataFrame with Poisson rates
        weekday: Day of week (1-7)
        timeslot: Time slot (0-7)

    Returns:
        NumPy array indexed by graph node IDs (plus external node 10000)
    """
    node_ids = list(ox.graph_to_gdfs(graph, edges=False).index) + [10000]
    n_nodes = len(node_ids)
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Initialize zero matrix
    rate_matrix = np.zeros((n_nodes, n_nodes))

    # Fill with rates for this specific weekday-timeslot
    filtered_rates = rate_df.filter(
        (pl.col('weekday') == weekday) &
        (pl.col('timeslot') == timeslot)
    )

    for row in filtered_rates.iter_rows(named=True):
        i = node_to_idx[row['start station id']]
        j = node_to_idx[row['end station id']]
        rate_matrix[i, j] = row['rate']

    return rate_matrix


def run(config: PreprocessingConfig) -> None:
    """
    Run the preprocessing pipeline: filter stations, compute rates, save matrices.

    Parameters:
        config: Preprocessing configuration object
    """
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

    print(f"\nLoading trip data for year {config.year}, months {config.months}...")

    # Load trip data
    trip_df = pl.DataFrame()
    for month in config.months:
        path = os.path.join(
            config.trips_path,
            f"{config.year}{str(month).zfill(2)}-bluebikes-tripdata.csv"
        )
        if os.path.isfile(path):
            monthly_data = pl.read_csv(path)
            trip_df = pl.concat([trip_df, monthly_data]) if trip_df.height > 0 else monthly_data
        else:
            print(f"Warning: Trip data for month {month} not found. Skipping...")

    if trip_df.height == 0:
        print("No trip data found. Exiting.")
        return

    # Map stations to graph nodes
    print("Mapping stations to graph nodes...")
    mapped_stations = filter_and_map_stations(
        trip_df,
        graph,
        bbox=config.bbox,
        max_distance_meters=100.0
    )

    # Save filtered stations
    filtered_stations_path = os.path.join(config.utils_path, "filtered_stations.csv")
    mapped_stations.select(['id', 'name', 'latitude', 'longitude']).write_csv(filtered_stations_path)
    print(f"Saved {mapped_stations.height} filtered stations to {filtered_stations_path}")

    # Filter trips and assign timeslots
    print("Filtering trips and computing timeslots...")
    valid_station_ids = mapped_stations.select("id").to_series().to_list()
    filtered_trips = filter_trips_and_compute_timeslots(trip_df, valid_station_ids)

    # Compute total seconds per weekday-timeslot
    print("Computing time coverage for each weekday-timeslot...")
    timeslot_counts = compute_timeslot_counts(filtered_trips)

    # Compute Poisson rates
    print("Computing Poisson rates...")
    rates = compute_poisson_rates(filtered_trips, timeslot_counts)

    global_rates = {}

    print("\nGenerating rate matrices...")
    pbar = tqdm(
        total=7 * 8,
        desc="Saving matrices",
        dynamic_ncols=True
    )

    for day_idx, day in enumerate(config.days_of_week, start=1):
        for timeslot in range(config.num_time_slots):
            # Update pbar description with current weekday and timeslot
            pbar.set_description(f"Generating {day} timeslot {timeslot}")

            day_name = day.lower()

            # Create rate matrix
            rate_matrix = compute_rate_matrix(graph, rates, day_idx, timeslot)

            # Compute global rate
            global_rate = math.fsum(rate_matrix.flatten())
            global_rates[(day_name, timeslot)] = global_rate

            # Convert to Polars DataFrame for saving
            node_ids = list(ox.graph_to_gdfs(graph, edges=False).index) + [10000]
            matrix_df = pl.DataFrame(
                rate_matrix,
                schema=[str(nid) for nid in node_ids]
            ).with_columns(
                pl.Series('node_id', node_ids)
            ).select(
                ['node_id'] + [str(nid) for nid in node_ids]
            )

            # Save matrix
            matrix_path = os.path.join(
                config.data_path,
                "matrices",
                config.month_str,
                str(timeslot).zfill(2)
            )
            os.makedirs(matrix_path, exist_ok=True)
            matrix_df.write_csv(
                os.path.join(matrix_path, f"{day_name}-rate-matrix.csv")
            )

            # Also save detailed rates CSV (compatible with old format)
            rate_detail = rates.filter(
                (pl.col('weekday') == day_idx) &
                (pl.col('timeslot') == timeslot)
            )

            rates_path = os.path.join(
                config.data_path,
                "rates",
                config.month_str,
                str(timeslot).zfill(2)
            )
            os.makedirs(rates_path, exist_ok=True)
            rate_detail.write_csv(
                os.path.join(rates_path, f"{day_name}-poisson-rates.csv")
            )

            pbar.update(1)

    pbar.close()

    # Save global rates at the end
    global_rates_path = os.path.join(config.data_path, config.global_rates_path)
    print(f"\nSaving global rates to {global_rates_path}...")
    with open(global_rates_path, "wb") as f:
        pickle.dump(global_rates, f)

    print("\nPreprocessing complete!")


def main():
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess trip data to compute Poisson rates using Polars."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_CONFIG.data_path,
        help="Path to data directory"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_CONFIG.year,
        help="Year of data to process"
    )
    parser.add_argument(
        "--months",
        type=str,
        default="9",
        help="Comma-separated months to process"
    )

    args = parser.parse_args()
    months = [int(m.strip()) for m in args.months.split(",")]

    config = PreprocessingConfig(
        data_path=args.data_path,
        year=args.year,
        months=months
    )
    
    run(config)


if __name__ == "__main__":
    main()
