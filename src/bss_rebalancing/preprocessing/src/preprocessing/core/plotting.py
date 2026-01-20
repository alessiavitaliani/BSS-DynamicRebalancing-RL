"""
Visualization functions for graphs and grids.
"""

import os
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.colors import Normalize

from preprocessing.core.utils import kahan_sum


def plot_graph(graph: nx.MultiDiGraph, save_path: str = "") -> None:
    """
    Plot the OSMnx graph.

    Parameters:
        graph: The OSMnx graph.
        save_path: Directory path to save the plot. If empty, plot is displayed.
    """
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    fig, ax = plt.subplots(figsize=(15, 12), facecolor="white")
    plt.subplots_adjust(left=0, top=1.02, right=1.2, bottom=0, wspace=0, hspace=0)

    edges.plot(ax=ax, linewidth=0.5, edgecolor="#DC143C", alpha=1, zorder=1)
    nodes.plot(ax=ax, markersize=15, color="#4169E1", alpha=1, zorder=2)

    # print node id 145 on the plot
    for node_id, node_data in graph.nodes(data=True):
        if node_id == 145:
            ax.text(
                node_data["x"],
                node_data["y"],
                str(node_id),
                fontsize=8,
                color="#4169E1",
                ha="center",
                va="center",
                zorder=3,
            )

    plt.axis("off")
    if save_path:
        output_file = os.path.join(save_path) if os.path.isdir(save_path) else save_path
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()


def plot_graph_with_colored_nodes(
    graph: nx.MultiDiGraph,
    rate_matrix: pd.DataFrame,
    axis: int = 0,
    colormap: Optional[str] = None,
    save_path: str = "",
) -> None:
    """
    Plot the OSMnx graph with nodes colored based on total request rates.

    Parameters:
        graph: The OSMnx graph.
        rate_matrix: A matrix containing request rates for each node.
        axis: Axis along which to sum rates (0 for rows, 1 for columns).
        colormap: Name of the matplotlib colormap to use.
        save_path: Path to save the plot (optional).
    """
    rate_matrix_indices = rate_matrix.index
    max_index = rate_matrix_indices.max()

    if axis == 0:
        sum_array = np.zeros(max_index + 1)
        for idx in rate_matrix.index:
            sum_array[idx] = kahan_sum(rate_matrix.loc[idx].values)
    else:
        sum_array = np.zeros(max_index + 1)
        for idx in rate_matrix.columns:
            sum_array[int(idx)] = kahan_sum(rate_matrix[idx].values)

    min_rate = 0
    max_rate = sum_array.max()

    print(f"Rate range: {min_rate} - {max_rate}")

    # Normalize total rates to [0, 1] for colormap
    norm = Normalize(vmin=min_rate, vmax=max_rate)

    if colormap is not None:
        cmap = plt.get_cmap(colormap)
        node_colors = {node: cmap(norm(rate_matrix.loc[node].sum())) for node in graph.nodes}
    else:
        if axis == 0:
            node_colors = {
                node: (0, 0.5, 0, 1) if kahan_sum(rate_matrix.loc[node].values) != 0 else (0.7, 0.7, 0.7, 1)
                for node in graph.nodes
            }
        else:
            node_colors = {
                node: (0, 0.5, 0.5, 1) if rate_matrix.loc[node].sum() != 0 else (0.7, 0.7, 0.7, 1)
                for node in graph.nodes
            }

    # Plot the graph
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    fig, ax = plt.subplots(figsize=(15, 12), facecolor="white")
    plt.subplots_adjust(left=0, top=1.02, right=1.2, bottom=0, wspace=0, hspace=0)

    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=1, zorder=1)
    nodes["color"] = nodes.index.map(lambda node_id: node_colors.get(node_id))
    nodes.plot(ax=ax, markersize=20, color=nodes["color"], alpha=1, zorder=2)

    if colormap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.1, pad=0.01)
        cbar.set_ticks([min_rate, max_rate / 2, max_rate])
        cbar.set_ticklabels(
            [
                f"Min: {(min_rate * 1e4):.1f}",
                f"50%: {((max_rate / 2) * 1e4):.3f} x 10^-4",
                f"Max: {(max_rate * 1e4):.3f} x 10^-4",
            ]
        )
        cbar.ax.tick_params(axis="y", colors="black", labelsize=16)
        cbar.ax.yaxis.set_tick_params(labelcolor="black")
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.set_position([1.0 - 0.19, 1.0 - 0.23, 0.15, 0.2])

    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_graph_with_grid(
    graph: nx.MultiDiGraph,
    cell_dict: Dict,
    plot_center_nodes: bool = False,
    plot_number_cells: bool = False,
    save_path: str = "",
) -> None:
    """
    Plot the graph with the grid overlay.

    Parameters:
        graph: The OSMnx graph.
        cell_dict: Dictionary of Cell objects.
        plot_center_nodes: Whether to plot the center nodes of each cell.
        plot_number_cells: Whether to plot the cell IDs.
        save_path: Directory path to save the plot. If empty, plot is displayed.
    """
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Convert cell_dict into a GeoDataFrame
    grid_geoms = [cell.boundary for cell in cell_dict.values()]
    cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot edges and nodes
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=1.0, zorder=1)
    nodes.plot(ax=ax, markersize=2, color="black", alpha=1.0, zorder=1)

    # Overlay grid cells
    cell_gdf.plot(ax=ax, linewidth=1.5, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0.1), zorder=2)

    for cell in cell_dict.values():
        center_node = cell.center_node
        if center_node != 0:
            node_coords = graph.nodes[center_node]["x"], graph.nodes[center_node]["y"]

            # Connect to adjacent cells' center nodes
            for direction, adjacent_cell in cell.adjacent_cells.items():
                if adjacent_cell is not None and adjacent_cell in cell_dict:
                    adjacent_center_node = cell_dict[adjacent_cell].center_node
                    if adjacent_center_node != 0:
                        adj_coords = (
                            graph.nodes[adjacent_center_node]["x"],
                            graph.nodes[adjacent_center_node]["y"],
                        )
                        ax.plot(
                            [node_coords[0], adj_coords[0]],
                            [node_coords[1], adj_coords[1]],
                            color="#FFA500",
                            linewidth=2,
                            alpha=1.0,
                            zorder=3,
                        )

            if plot_center_nodes:
                ax.plot(node_coords[0], node_coords[1], marker="o", color="#000000", markersize=7, zorder=4)
                ax.plot(
                    node_coords[0],
                    node_coords[1],
                    marker="o",
                    color="#FFA500",
                    markersize=4,
                    zorder=4,
                )

        if plot_number_cells:
            center_coords = cell.boundary.centroid.coords[0]
            ax.text(
                center_coords[0]-0.0005,
                center_coords[1]+0.0005,
                str(cell.id),
                fontsize=15,
                color="#000000",
                ha="center",
                va="center",
                weight="bold",
                zorder=5,
            )

    plt.axis("off")
    if save_path:
        output_file = os.path.join(save_path) if os.path.isdir(save_path) else save_path
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()