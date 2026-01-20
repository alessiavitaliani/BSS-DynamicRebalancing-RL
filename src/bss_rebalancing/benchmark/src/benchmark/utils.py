"""
Benchmark Utility Functions.

This module provides utility functions for the benchmark system including
time conversion, plotting, visualization, and monitoring capabilities.

Author: Edoardo Scarpel
"""

import os
from enum import Enum
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import psutil
import requests
import torch
from torch_geometric.utils import from_networkx


# =============================================================================
# Matplotlib Configuration
# =============================================================================

# Detect IPython environment for interactive plotting
_is_ipython = 'inline' in matplotlib.get_backend()
if _is_ipython:
    from IPython import display
    plt.ion()


# =============================================================================
# Enumerations
# =============================================================================

class Actions(Enum):
    """Agent action space for dynamic rebalancing."""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    CHARGE_BIKE = 7


# =============================================================================
# Time Conversion
# =============================================================================

def convert_seconds_to_hours_minutes(seconds: int) -> str:
    """
    Convert seconds to formatted time string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted string as "HH:MM:SS".

    Example:
        >>> convert_seconds_to_hours_minutes(3661)
        '01:01:01'
    """
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24  # Wrap to 24-hour format
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =============================================================================
# Graph Conversion
# =============================================================================

def convert_graph_to_data(graph: nx.MultiDiGraph):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        graph: Input MultiDiGraph representing the spatial network.

    Returns:
        PyTorch Geometric Data object with node features, edge attributes,
        and edge types suitable for GNN processing.
    """
    # Convert to base PyG data structure
    data = from_networkx(graph)

    # Define node attributes to extract
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

    # Extract and concatenate node features
    data.x = torch.cat([
        torch.tensor(
            [graph.nodes[n].get(attr, 0) for n in graph.nodes()],
            dtype=torch.float
        ).unsqueeze(dim=-1)
        for attr in node_attrs
    ], dim=-1)

    # Extract edge information
    edge_types = []
    edge_attrs = ['distance']
    edge_attr_list = {attr: [] for attr in edge_attrs}

    for u, v, k, attr in graph.edges(data=True, keys=True):
        edge_types.append(k)
        for edge_attr in edge_attrs:
            edge_attr_list[edge_attr].append(attr[edge_attr])

    # Map edge types to integer indices
    edge_type_mapping = {
        e_type: i for i, e_type in enumerate(set(edge_types))
    }
    edge_type_indices = torch.tensor(
        [edge_type_mapping[e_type] for e_type in edge_types],
        dtype=torch.long
    )

    # Add edge information to data object
    data.edge_type = edge_type_indices
    data.edge_attr = torch.cat([
        torch.tensor(
            edge_attr_list[attr],
            dtype=torch.float
        ).unsqueeze(dim=-1)
        for attr in edge_attrs
    ], dim=-1)

    return data


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_data_online(
    data: np.ndarray,
    show_result: bool = False,
    idx: int = 1,
    xlabel: str = 'Step',
    ylabel: str = 'Reward',
    show_histogram: bool = False,
    bin_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training data with optional moving average.

    Args:
        data: Array of data points to plot.
        show_result: If True, displays final results; otherwise shows training.
        idx: Figure index for matplotlib.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        show_histogram: If True, displays histogram instead of line plot.
        bin_labels: Optional custom labels for histogram bins.
        save_path: Optional path to save the figure.
    """
    # Convert to tensor for processing
    data = np.array(data)
    data_t = torch.tensor(data, dtype=torch.float)

    # Setup figure
    plt.figure(idx)
    plt.clf()

    if show_histogram:
        # Histogram mode
        plt.title('Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        bins = len(data)
        plt.bar(range(bins), data, alpha=0.75, edgecolor='black')

        # Apply custom bin labels if provided
        if bin_labels is not None:
            if len(bin_labels) != bins:
                raise ValueError(
                    "bin_labels length must match number of bins"
                )
            plt.xticks(
                ticks=range(bins),
                labels=bin_labels,
                rotation=45,
                ha='right'
            )
    else:
        # Line plot mode
        title = 'Result' if show_result else 'Training...'
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data_t.numpy())

        # Add 100-step moving average if sufficient data
        if len(data_t) >= 100:
            means = data_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.pause(0.001)

        if _is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
            plt.close()


def plot_graph_with_truck_path(
    graph: nx.MultiDiGraph,
    cell_dict: dict,
    nodes_dict: dict,
    path: List[Tuple[int, int]],
    show_result: bool,
    idx: int = 1,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize street network with truck path overlay.

    Args:
        graph: OSMnx MultiDiGraph of street network.
        cell_dict: Dictionary mapping cell IDs to Cell objects.
        nodes_dict: Dictionary mapping node IDs to (lat, lon) coordinates.
        path: List of (source, target) node ID pairs representing truck path.
        show_result: If True, displays final results; otherwise training.
        idx: Figure index for matplotlib.
        x_lim: Optional x-axis limits (longitude).
        y_lim: Optional y-axis limits (latitude).
        save_path: Optional path to save the figure.
    """
    plt.figure(idx)
    plt.clf()

    # Extract network components
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Create cell boundaries GeoDataFrame
    grid_geoms = [cell.boundary for cell in cell_dict.values()]
    cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")

    # Setup plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot network elements
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.5)
    nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.5)
    cell_gdf.plot(
        ax=ax,
        linewidth=0.8,
        edgecolor="red",
        facecolor="blue",
        alpha=0.2
    )

    # Plot cell centers
    for cell in cell_dict.values():
        center_node = cell.center_node
        if center_node != 0:
            node_coords = (
                graph.nodes[center_node]['x'],
                graph.nodes[center_node]['y']
            )
            ax.plot(
                node_coords[0],
                node_coords[1],
                marker='o',
                color='yellow',
                markersize=4,
                label=f"Center Node {cell.id}"
            )

    # Plot truck path
    for source, target in path:
        if source in nodes_dict and target in nodes_dict:
            source_coords = nodes_dict[source]
            target_coords = nodes_dict[target]
            ax.plot(
                [source_coords[1], target_coords[1]],
                [source_coords[0], target_coords[0]],
                color='yellow',
                linewidth=2,
                alpha=0.8
            )

    # Plot truck position
    if path:
        truck_coords = nodes_dict[path[-1][1]]
        ax.plot(
            truck_coords[1],
            truck_coords[0],
            marker='o',
            color='red',
            markersize=10,
            label="Truck position"
        )

    # Apply axis limits if provided
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    # Set title
    title = 'Result' if show_result else 'Training...'
    plt.title(title)

    # Configure appearance
    plt.axis('off')
    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.pause(0.001)

        if _is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
            plt.close(fig)


# =============================================================================
# Communication and Monitoring
# =============================================================================

def send_telegram_message(message: str, bot_token: str, chat_id: str) -> bool:
    """
    Send a message via Telegram bot.

    Args:
        message: Text message to send.
        bot_token: Telegram bot API token.
        chat_id: Telegram chat ID to send message to.

    Returns:
        True if message sent successfully, False otherwise.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram message sent successfully.")
            return True
        else:
            print(
                f"Failed to send Telegram message. "
                f"Status code: {response.status_code}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")
        return False


def get_memory_usage() -> float:
    """
    Get current process memory usage.

    Returns:
        Memory usage in megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
