"""
Seleziona la cella migliore da cui far partire il truck (depot_position_id /
initial_cell_id), pesando per la domanda reale osservata nei dati di trip,
invece di usare il semplice centro geometrico dell'area.

Basato sulla struttura reale confermata:
    cell_data.pkl -> dict {cell_id: oggetto Cell}
        Cell.get_boundary()  -> poligono (shapely), con .centroid.x/.y (lon/lat)
        Cell.get_id()        -> cell_id

Non serve NetworkX: cell_data.pkl e' gia' un dizionario piatto.
"""

import pickle
from pathlib import Path

import pandas as pd
from shapely.geometry import Point


def load_cell_data(area_data_path: Path) -> dict:
    """Carica cell_data.pkl: {cell_id: oggetto Cell}."""
    with open(area_data_path / "utils" / "cell_data.pkl", "rb") as f:
        return pickle.load(f)


def load_filtered_stations(area_data_path: Path) -> pd.DataFrame:
    """
    Carica filtered_stations.csv, salvato da preprocess_data.py
    (colonne: id, name, latitude, longitude).
    """
    return pd.read_csv(area_data_path / "utils" / "filtered_stations.csv")


def assign_stations_to_cells(stations_df: pd.DataFrame, cell_data: dict) -> dict:
    """
    Point-in-polygon: per ogni stazione (lat/lon), trova la cella il cui
    boundary la contiene.
    """
    cell_boundaries = [
        (cell_id, cell.get_boundary())
        for cell_id, cell in cell_data.items()
    ]

    station_to_cell = {}
    for row in stations_df.itertuples():
        p = Point(row.longitude, row.latitude)  # shapely: x=lon, y=lat
        for cell_id, boundary in cell_boundaries:
            if boundary.contains(p):
                station_to_cell[row.id] = cell_id
                break
        # se nessuna cella la contiene (stazione al bordo esatto dell'area),
        # resta fuori dal conteggio — trascurabile per il baricentro
    return station_to_cell


def load_trip_counts_per_station(trip_csv_path: Path) -> pd.Series:
    """
    Conta partenze + arrivi per stazione dal CSV di trip gia' unito dal
    download step (formato: '{source.id}-{month_str}-tripdata.csv').
    """
    df = pd.read_csv(trip_csv_path, usecols=["start station id", "end station id"])
    all_stations = pd.concat([df["start station id"], df["end station id"]])
    return all_stations.value_counts()


def find_best_depot_cell(cell_data: dict, station_to_cell: dict,
                          trip_counts_per_station: pd.Series) -> tuple[int, dict]:
    """
    Restituisce (best_cell_id, debug_info) con la cella piu' vicina al
    baricentro della domanda, e per confronto anche il centro geometrico.
    """
    # 1. aggrega i conteggi trip per cella
    trips_per_cell: dict[int, int] = {}
    for station_id, count in trip_counts_per_station.items():
        cell_id = station_to_cell.get(station_id)
        if cell_id is None:
            continue  # stazione fuori dall'area / non mappata
        trips_per_cell[cell_id] = trips_per_cell.get(cell_id, 0) + int(count)

    if not trips_per_cell:
        raise ValueError("Nessun trip mappato a nessuna cella — controlla station_to_cell.")

    # 2. baricentro pesato per la domanda
    total_weight = sum(trips_per_cell.values())
    weighted_lat = sum(
        cell_data[cid].get_boundary().centroid.y * w
        for cid, w in trips_per_cell.items()
    ) / total_weight
    weighted_lon = sum(
        cell_data[cid].get_boundary().centroid.x * w
        for cid, w in trips_per_cell.items()
    ) / total_weight

    # 3. anche il centro geometrico puro, per confronto
    all_centroids = [
        (cell.get_boundary().centroid.y, cell.get_boundary().centroid.x)
        for cell in cell_data.values()
    ]
    geo_lat = sum(c[0] for c in all_centroids) / len(all_centroids)
    geo_lon = sum(c[1] for c in all_centroids) / len(all_centroids)

    # 4. cella piu' vicina al baricentro pesato (e al centro geometrico)
    def nearest_cell(target_lat, target_lon):
        best_id, best_dist = None, float("inf")
        for cell_id, cell in cell_data.items():
            c = cell.get_boundary().centroid
            dist = (c.y - target_lat) ** 2 + (c.x - target_lon) ** 2
            if dist < best_dist:
                best_dist, best_id = dist, cell_id
        return best_id

    weighted_best = nearest_cell(weighted_lat, weighted_lon)
    geo_best = nearest_cell(geo_lat, geo_lon)

    # 5. top 5 celle per domanda, utile per sanity-check manuale
    top5 = sorted(trips_per_cell.items(), key=lambda kv: -kv[1])[:5]

    debug_info = {
        "weighted_centroid": (weighted_lat, weighted_lon),
        "geo_centroid": (geo_lat, geo_lon),
        "geo_center_cell": geo_best,
        "top5_cells_by_demand": top5,
        "n_cells_with_data": len(trips_per_cell),
        "n_cells_total": len(cell_data),
    }
    return weighted_best, debug_info


if __name__ == "__main__":
    AREA_PATH = Path("data_boston/")   # <-- adatta
    TRIP_CSV = Path("data_boston/trips/bluebikes-09-10-tripdata.csv")  # <-- adatta

    cell_data = load_cell_data(AREA_PATH)
    stations_df = load_filtered_stations(AREA_PATH)
    station_to_cell = assign_stations_to_cells(stations_df, cell_data)
    trip_counts = load_trip_counts_per_station(TRIP_CSV)

    best_cell, info = find_best_depot_cell(cell_data, station_to_cell, trip_counts)

    print(f"Cella depot consigliata (pesata sulla domanda): {best_cell}")
    print(f"Cella centro geometrico puro (per confronto):    {info['geo_center_cell']}")
    print(f"Stazioni mappate a una cella: {len(station_to_cell)}/{len(stations_df)}")
    print(f"Celle con dati di domanda: {info['n_cells_with_data']}/{info['n_cells_total']}")
    print(f"Top 5 celle per volume di trip: {info['top5_cells_by_demand']}")