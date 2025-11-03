import torch
import matplotlib
import requests
import networkx as nx
import osmnx as ox
import numpy as np
import geopandas as gpd
import psutil, os

from torch_geometric.utils import from_networkx
from matplotlib import pyplot as plt
from enum import Enum

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class Actions(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    CHARGE_BIKE = 7

def convert_graph_to_data(graph: nx.MultiDiGraph):
    """
    Converts a NetworkX MultiDiGraph to a PyTorch Geometric Data object.

    Parameters:
        - graph: The input MultiDiGraph representing the graph.

    Returns:
        - A PyTorch Geometric Data object with node features, edge attributes, and edge types.
    """
    data = from_networkx(graph)

    # Extract node attributes
    node_attrs = [
        'demand_rate',
        'arrival_rate',
        'average_battery_level',
        'low_battery_ratio',
        'variance_battery_level',
        'total_bikes',
        'bike_load',
        'visits',
        'critic_score',
    ]
    data.x = torch.cat([
        torch.tensor(
            [graph.nodes[n].get(attr, 0) for n in graph.nodes()],
            dtype=torch.float
        ).unsqueeze(dim=-1) for attr in node_attrs
    ], dim=-1)

    # Extract edge types and attributes
    edge_types = []
    edge_attrs = ['distance']
    edge_attr_list = {attr: [] for attr in edge_attrs}
    for u, v, k, attr in graph.edges(data=True, keys=True):
        edge_types.append(k)
        for edge_attr in edge_attrs:
            edge_attr_list[edge_attr].append(attr[edge_attr])

    # Map edge types to integers
    edge_type_mapping = {e_type: i for i, e_type in enumerate(set(edge_types))}
    edge_type_indices = torch.tensor(
        [edge_type_mapping[e_type] for e_type in edge_types],
        dtype=torch.long
    )

    # Add edge types and attributes to the Data object
    data.edge_type = edge_type_indices
    data.edge_attr = torch.cat([
        torch.tensor(edge_attr_list[attr], dtype=torch.float).unsqueeze(dim=-1)
        for attr in edge_attrs
    ], dim=-1)
    data.edge_index = data.edge_index

    return data


def convert_seconds_to_hours_minutes(seconds) -> str:
    """
    Converts seconds to a formatted string of hours, minutes, and seconds.

    Parameters:
        - seconds: Time duration in seconds.

    Returns:
        - A string formatted as "HH:MM:SS".
    """
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def plot_data_online(data, show_result=False, idx=1, xlabel='Step', ylabel='Reward', show_histogram=False,
                     bin_labels=None, save_path=None):
    """
    Plots rewards data online during training or displays final results.

    Parameters:
        - data: List or NumPy array of rewards data.
        - show_result: If True, displays the final results (default=False).
        - idx: Index of the plot figure (default=1).
        - xlabel: Label for the x-axis (default='Step').
        - ylabel: Label for the y-axis (default='Reward').
        - show_histogram: If True, displays a histogram of the data (default=False).
    """
    # Convert input data to a PyTorch tensor
    data = np.array(data)
    data_t = torch.tensor(data, dtype=torch.float)

    plt.figure(idx)
    plt.clf()

    if show_histogram:
        plt.title('Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        bins = len(data)
        plt.bar(range(bins), data, alpha=0.75, edgecolor='black')

        # Set custom labels for the x-axis if provided
        if bin_labels is not None:
            if len(bin_labels) != bins:
                raise ValueError("The length of bin_labels must match the number of bins.")
            plt.xticks(ticks=range(bins), labels=bin_labels, rotation=45, ha='right')
    else:
        if show_result:
            plt.title('Result')
        else:
            plt.title('Training...')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data_t.numpy())

        # Compute and plot 100-step moving averages
        if len(data_t) >= 100:
            means = data_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    plt.close()


def plot_graph_with_truck_path(graph: nx.MultiDiGraph, cell_dict: dict, nodes_dict: dict, path: list[tuple[int, int]],
                               show_result: bool, idx=1, x_lim=None, y_lim=None, save_path=None):
    plt.figure(idx)
    plt.clf()

    # Extract nodes and edges in WGS84 coordinates (lon, lat)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Convert cell_dict into a GeoDataFrame in WGS84 for easy plotting
    grid_geoms = [cell.boundary for cell in cell_dict.values()]
    cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")  # WGS84 CRS

    # Plot setup
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.5)
    # Plot the graph nodes
    nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.5)

    # Overlay the grid cells
    cell_gdf.plot(ax=ax, linewidth=0.8, edgecolor="red", facecolor="blue", alpha=0.2)

    for cell in cell_dict.values():
        center_node = cell.center_node
        if center_node != 0:
            node_coords = graph.nodes[center_node]['x'], graph.nodes[center_node]['y']
            ax.plot(node_coords[0], node_coords[1], marker='o', color='yellow', markersize=4,
                    label=f"Center Node {cell.id}")

    # Plot the truck's path
    for source, target in path:
        if source in nodes_dict and target in nodes_dict:
            source_coords = nodes_dict[source]
            target_coords = nodes_dict[target]
            ax.plot([source_coords[1], target_coords[1]],
                    [source_coords[0], target_coords[0]],
                    color='yellow', linewidth=2, alpha=0.8)


    truck_coords = nodes_dict[path[-1][1]]
    ax.plot(truck_coords[1], truck_coords[0], marker='o', color='red', markersize=10, label="Truck position")

    if x_lim is not None and y_lim is not None:
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')

    # Configure plot appearance
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    plt.close(fig)


# Function to send a Telegram message
def send_telegram_message(message: str, BOT_TOKEN: str, CHAT_ID: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Message sent successfully.")
        else:
            print(f"Failed to send message. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

