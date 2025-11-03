import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from enum import Enum
import networkx as nx
import osmnx as ox
import io
import base64
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from matplotlib import pyplot as plt


class Actions(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    CHARGE_BIKE = 7


# Initialize usefull vectors
num2days = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
action_bin_labels = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']

# plot fonts sizes
title_size = 50
label_size = 40
tick_size = 29
legend_size = 35

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def initialize_graph(graph_path: str = None) -> nx.MultiDiGraph:
    if os.path.isfile(graph_path):
        graph = ox.load_graphml(graph_path)
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError("Network file does not exist. Please check the file path.")

    return graph


def plot_data_online(data, show_result=False, idx=1, xlabel='Step', ylabel='Reward', show_histogram=False,
                     bin_labels=None, title="Plot", save_path=None, mean=True):
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
    new_data = [0] * 8
    if isinstance(data, dict):
        for timeslot, value in data.items():
            print(timeslot, value)
            new_data[timeslot] = value
        data = new_data

    # Ensure input data is a NumPy array
    data = np.array(data, dtype=np.float32)

    plt.figure(idx)
    plt.clf()

    if show_histogram:
        plt.title(title)
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
            plt.title(title)
        else:
            plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)

        if mean:
            cumulative_mean = np.cumsum(data) / np.arange(1, len(data) + 1, dtype=np.float32)
            plt.plot(cumulative_mean)

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
        else:
            plt.show(block=True)


def plot_confusion_matrix(data: pd.DataFrame, title="Heatmap", x_label="", y_label="", cbar_label="", cmap="YlGnBu",
                          save_path=None):
    """
    Plots a heatmap for failures with days on the x-axis and time slots on the y-axis.

    Parameters:
        - failures: 2D array or DataFrame where rows correspond to time slots and columns to days.
        - days: List of strings for the days of the week (x-axis labels).
        - time_slots: List of strings for the time slots (y-axis labels).
        - title: Title of the heatmap (default: "Failure Heatmap").
        - save_path: Path to save the plot (default: None, which means display it).
    """
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        data=data,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        xticklabels=data.columns,
        yticklabels=data.index
    )

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=15)

    # Set x-axis ticks on top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')  # Move x-axis label to the top

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------

def get_episode_options(training_path):
    if not os.path.exists(training_path):
        return []

    episode_folders = sorted(
        [folder for folder in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, folder))],
        key=lambda x: int(x)
    )

    options = [{"label": folder, "value": folder} for folder in episode_folders]
    return options


def load_results_old(training_path, episode_folder="all"):
    if episode_folder == "all":
        rewards, failures, q_values, loss, epsilon, deployed_bikes = [], [], [], [], [], []
        action_per_step, reward_tracking = [], [[] for _ in action_bin_labels]
        episode_folders = get_episode_options(training_path)[1:]  # Exclude "All Timeslots" option
        for folder in [opt['value'] for opt in episode_folders]:
            r, f, q, l, e, a, rt, b = load_results_old(training_path, folder)
            rewards.extend(r)
            failures.extend(f)
            q_values.extend(q)
            loss.extend(l)
            epsilon.extend(e)
            action_per_step.extend(a)
            deployed_bikes.extend(b)
            for i, inner_rt in enumerate(rt):
                reward_tracking[i].extend(inner_rt)
        return rewards, failures, q_values, loss, epsilon, action_per_step, reward_tracking, deployed_bikes

    timeslot_path = os.path.join(training_path, episode_folder)
    if not os.path.exists(timeslot_path):
        return [], [], [], [], [], [], [], []

    with open(os.path.join(timeslot_path, 'rewards_per_timeslot.pkl'), 'rb') as f:
        rewards = pickle.load(f)
    with open(os.path.join(timeslot_path, 'failures_per_timeslot.pkl'), 'rb') as f:
        failures = pickle.load(f)
    with open(os.path.join(timeslot_path, 'q_values_per_timeslot.pkl'), 'rb') as f:
        q_values = pickle.load(f)
    with open(os.path.join(timeslot_path, 'losses.pkl'), 'rb') as f:
        loss = pickle.load(f)
    with open(os.path.join(timeslot_path, 'reward_tracking.pkl'), 'rb') as f:
        reward_tracking = pickle.load(f)
    with open(os.path.join(timeslot_path, 'epsilon_per_timeslot.pkl'), 'rb') as f:
        epsilon = pickle.load(f)
    with open(os.path.join(timeslot_path, 'action_per_step.pkl'), 'rb') as f:
        action_per_step = pickle.load(f)
    with open(os.path.join(timeslot_path, 'deployed_bikes.pkl'), 'rb') as f:
        deployed_bikes = pickle.load(f)

    return rewards, failures, q_values, loss, epsilon, action_per_step, reward_tracking, deployed_bikes


def load_results(training_path, episode_folder="all", metric="rewards_per_timeslot", sum_episode_data=False):
    if episode_folder == "all":
        results = []
        if metric == "reward_tracking":
            results = [[] for _ in action_bin_labels]
        episode_folders = get_episode_options(training_path)[1:]  # Exclude "All Timeslots" option
        for folder in [opt['value'] for opt in episode_folders]:
            r = load_results(training_path, folder, metric)
            if metric != "reward_tracking":
                # Handle non-iterable metrics safely
                if sum_episode_data:  # and metric == "failures_per_timeslot":
                    results.append(sum(r))
                elif isinstance(r, list):
                    results.extend(r)
                else:
                    results.append(r)
            else:
                for i, inner_rt in enumerate(r):
                    results[i].extend(inner_rt)
        return results
    elif episode_folder == "last":
        ef = get_episode_options(training_path)[-1]
        episode_folder = ef['value']

    timeslot_path = os.path.join(training_path, episode_folder)
    if not os.path.exists(timeslot_path):
        return []

    with open(os.path.join(timeslot_path, metric + '.pkl'), 'rb') as f:
        results = pickle.load(f)

    return results


# def create_plot(data, title, y_axis_label, x_axis_label, cumulative=False, action_plot=False,
#                 failures_plot=False, episode_size=56):
#     if not data:
#         return go.Figure().update_layout(title=title, yaxis_title=y_axis_label)

#     if action_plot:
#         fig = go.Figure()
#         fig.add_trace(go.Bar(x=action_bin_labels, y=data))
#         fig.update_layout(
#             title=title,
#             yaxis_title=y_axis_label,
#             legend=dict(
#                 x=0.9,  # Horizontal position (0 = left, 1 = right)
#                 y=0.9,  # Vertical position (0 = bottom, 1 = top)
#                 bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
#                 bordercolor='black',
#                 borderwidth=1
#             )
#         )
#     else:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(y=data, mode='lines', name="Values"))
#         if cumulative:
#             if failures_plot:
#                 y_cumulative = []
#                 for i in range(0, len(data), episode_size):
#                     segment = data[i:i + episode_size]
#                     if len(segment) > 0:
#                         y_cumulative.extend(np.cumsum(segment) / np.arange(1, len(segment) + 1))
#             else:
#                 y_cumulative = np.cumsum(data) / np.arange(1, len(data) + 1)
#             fig.add_trace(go.Scatter(y=y_cumulative, mode='lines', name="Cumulative Mean", line=dict(color='red')))

#         fig.update_layout(
#             title=title,
#             yaxis_title=y_axis_label,
#             xaxis_title=x_axis_label,
#             legend=dict(
#                 x=0.84,  # Horizontal position (0 = left, 1 = right)
#                 y=0.97,  # Vertical position (0 = bottom, 1 = top)
#                 bgcolor='rgba(255, 255, 255, 1)',
#                 bordercolor='black',
#                 borderwidth=1
#             )
#         )
#     return fig

def create_plot(data, title, y_axis_label, x_axis_label,
                cumulative=False, action_plot=False,
                failures_plot=False, episode_size=56):
    # If there is no data, return an empty figure with a title and labels
    if data is None or len(data) == 0:
        return go.Figure().update_layout(
            title=dict(text=title, font=dict(size=title_size)),
            yaxis=dict(title=dict(text=y_axis_label, font=dict(size=label_size))),
            xaxis=dict(title=dict(text=x_axis_label, font=dict(size=label_size))),
            font=dict(size=tick_size)
        )

    # If it's an action plot, create a bar chart
    if action_plot:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=action_bin_labels, y=data))
        fig.update_layout(
            title=dict(text=title, font=dict(size=title_size)),
            yaxis=dict(title=dict(text=y_axis_label, font=dict(size=label_size)),
                       tickfont=dict(size=tick_size)),
            xaxis=dict(title=dict(text=x_axis_label, font=dict(size=label_size)),
                       tickfont=dict(size=tick_size)),
            font=dict(size=tick_size),
            legend=dict(
                x=0.9,  # Horizontal position (0 = left, 1 = right)
                y=0.9,  # Vertical position (0 = bottom, 1 = top)
                bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                bordercolor='black',
                borderwidth=1,
                font=dict(size=legend_size)
            )
        )
    else:
        # Otherwise, create a line plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data, mode='lines', name="Values"))

        # If cumulative is enabled, compute and plot cumulative mean
        if cumulative:
            if failures_plot:
                y_cumulative = []
                for i in range(0, len(data), episode_size):
                    segment = data[i:i + episode_size]
                    if len(segment) > 0:
                        y_cumulative.extend(np.cumsum(segment) / np.arange(1, len(segment) + 1))
            else:
                y_cumulative = np.cumsum(data) / np.arange(1, len(data) + 1)

            fig.add_trace(go.Scatter(
                y=y_cumulative,
                mode='lines',
                name="Cumulative Mean",
                line=dict(color='red')
            ))

        # Update layout with fonts and legend configuration
        fig.update_layout(
            title=dict(text=title, font=dict(size=title_size)),
            yaxis=dict(title=dict(text=y_axis_label, font=dict(size=label_size)),
                       tickfont=dict(size=tick_size)),
            xaxis=dict(title=dict(text=x_axis_label, font=dict(size=label_size)),
                       tickfont=dict(size=tick_size)),
            font=dict(size=tick_size),
            legend=dict(
                x=0.84,  # Horizontal position (0 = left, 1 = right)
                y=0.97,  # Vertical position (0 = bottom, 1 = top)
                bgcolor='rgba(255, 255, 255, 1)',
                bordercolor='black',
                borderwidth=1
            )
        )

    return fig


def generate_osmnx_graph(graph: nx.MultiDiGraph, cell_dict: dict, cell_values: dict, percentage: bool = False):
    # Extract nodes and edges in WGS84 coordinates (lon, lat)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Convert cell_dict into a GeoDataFrame in WGS84 for easy plotting
    grid_geoms = [cell.boundary for cell in cell_dict.values()]
    cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")

    # Define colormap (use 'hot' for a heatmap effect)
    min_value = min(0, *cell_values.values()) if cell_values else 0
    max_value = max(0.1, *cell_values.values()) if cell_values else 1
    # Choose colormap based on sign of values
    if min_value < 0:
        cmap = cm.get_cmap("coolwarm")  # diverging colormap
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=min_value,
                                    vmax=max_value)  # Normalize colors based on frequency range
    else:
        cmap = cm.get_cmap("viridis")  # Choose colormap (e.g., 'hot', 'viridis', 'coolwarm')
        norm = mcolors.Normalize(vmin=min_value, vmax=max_value)  # Normalize colors based on frequency range

    # Plot setup
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)
    # Plot the graph nodes
    nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.7)

    # Plot each cell with the heatmap color
    for cell_id, cell in cell_dict.items():
        color = cmap(norm(cell_values[cell_id]))
        cell_gdf[cell_gdf.geometry == cell.boundary].plot(ax=ax, linewidth=0.8, edgecolor="red",
                                                          facecolor=mcolors.to_hex(color), alpha=0.8)

    for cell_id, cell in cell_dict.items():
        center_node = cell.center_node
        if center_node != 0:
            node_coords = graph.nodes[center_node]['x'], graph.nodes[center_node]['y']
            ax.plot(node_coords[0], node_coords[1], marker='o', color='yellow', markersize=4,
                    label=f"Center Node {cell.id}")
        center_coords = cell.boundary.centroid.coords[0]
        if percentage:
            ax.text(center_coords[0], center_coords[1], f"{cell_values[cell_id] * 100:.2f}%", fontsize=14,
                    color='black', ha='center', va='center', weight='bold')
        else:
            ax.text(center_coords[0], center_coords[1], f"{cell_values[cell_id]:.2f}", fontsize=15,
                    color='black', ha='center', va='center', weight='bold')

    # Hide x-ticks, y-ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)  # Optional: removes the border/frame

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)

    # Encode image to base64 to display in Dash
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded_image}"


def compare_failures_across_trainings(training_paths, training_labels=None):
    """
    Compare total failures across multiple training runs.

    Parameters
    ----------
    training_paths : list of str
        Paths to each training folder (e.g., ["results/training_51/data", "results/training_52/data"])
    training_labels : list of str, optional
        Labels for the legend. If None, folder names will be used.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A line plot comparing total failures per episode across trainings.
    """
    fig = go.Figure()

    if training_labels is None:
        training_labels = [os.path.basename(os.path.dirname(p)) for p in training_paths]

    for path, label in zip(training_paths, training_labels):
        failures = load_results(path, episode_folder="all", metric="failures_per_timeslot", sum_episode_data=True)

        # In your modified load_results, this already sums per episode,
        # so `failures` should be a list: [sum_ep1, sum_ep2, sum_ep3, ...]
        if not failures:
            continue

        fig.add_trace(go.Scatter(
            y=failures,
            mode='lines',
            name=label
        ))

    # # --- Change x-axis ticks from timeslots → episodes ---
    # timeslots_per_day = 56*10
    # num_timeslots = len(failures)

    # # Choose tick positions at day boundaries
    # tick_positions = list(range(0, num_timeslots, timeslots_per_day))
    # tick_labels = [str(int(i / timeslots_per_day)*10) for i in tick_positions]

    # fig.update_layout(
    #     xaxis=dict(
    #         tickmode='array',
    #         tickvals=tick_positions,
    #         ticktext=tick_labels,
    #         title="Episode"
    #     )
    # )

    fig.update_layout(
        title=dict(
            text="Total Failures per Episode Across Trainings",
            font=dict(size=title_size, family="Poppins, sans-serif")
        ),
        xaxis=dict(
            title=dict(text="Episode", font=dict(size=label_size)),
            tickfont=dict(size=tick_size)
        ),
        yaxis=dict(
            title=dict(text="Total Failures", font=dict(size=label_size)),
            tickfont=dict(size=tick_size)
        ),
        legend=dict(
            x=0.7,
            y=0.9,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=legend_size)
        )
    )

    return fig


def compute_mean_failures(training_paths, last_n=20, bench=False):
    """
    Compute the mean failure-per-timeslot vector across the last N episodes
    from one or more training/data folders.

    Parameters
    ----------
    training_paths : list of str
        List of full paths to 'data' folders (e.g. ['results/training_54/data', ...]).
    last_n : int
        Number of last episodes to include (default = 20).

    Returns
    -------
    np.ndarray
        Mean failure-per-timeslot vector (same length as individual episode vectors).
    """

    all_vectors = []

    for training_path in training_paths:
        if not os.path.isdir(training_path):
            print(f"⚠️ Skipping {training_path} — not found.")
            continue

        if bench:
            # training_path = os.path.join(training_paths)

            ep_path = os.path.join(training_path, "total_failures.pkl")
            with open(ep_path, "rb") as f:
                vec = np.array(pickle.load(f), dtype=float)

            blocks = [vec[i:i + 56] for i in range(0, len(vec), 56)]
            all_vectors.extend(blocks)
        else:
            # List and sort subfolders numerically
            subfolders = sorted(
                [f for f in os.listdir(training_path) if f.isdigit()],
                key=lambda x: int(x)
            )

            if not subfolders:
                print(f"⚠️ No numeric subfolders in {training_path}.")
                continue

            # Select the last N
            last_episodes = subfolders[-last_n:]

            for ep in last_episodes:
                ep_path = os.path.join(training_path, ep, "failures_per_timeslot.pkl")
                if not os.path.exists(ep_path):
                    print(f"⚠️ Missing file: {ep_path}")
                    continue

                with open(ep_path, "rb") as f:
                    vec = pickle.load(f)
                    all_vectors.append(np.array(vec, dtype=float))

    if not all_vectors:
        print("⚠️ No valid data found.")
        return np.array([])

    # Stack and average elementwise
    min_len = min(len(v) for v in all_vectors)  # ensure consistent length
    trimmed = np.array([v[:min_len] for v in all_vectors])
    mean_vector = np.mean(trimmed, axis=0)

    return mean_vector


def compute_mean_invalid(training_paths):
    # mean_vector = []
    episode_dict = {}

    for training_path in training_paths:
        if not os.path.isdir(training_path):
            print(f"⚠️ Skipping {training_path} — not found.")
            continue

        # List and sort subfolders numerically
        subfolders = sorted(
            [f for f in os.listdir(training_path) if f.isdigit()],
            key=lambda x: int(x)
        )

        if not subfolders:
            print(f"⚠️ No numeric subfolders in {training_path}.")
            continue

        for ep in subfolders:
            ep_path = os.path.join(training_path, ep, "total_invalid.pkl")
            if not os.path.exists(ep_path):
                print(f"⚠️ Missing file: {ep_path}")
                continue

            with open(ep_path, "rb") as f:
                val = pickle.load(f)
                if isinstance(val, (int, float)):
                    ep_index = int(ep)
                    episode_dict.setdefault(ep_index, []).append(val)
                else:
                    print(f"⚠️ Unexpected type in {ep_path}: {type(val)}")

    if not episode_dict:
        print("⚠️ No valid data found.")
        return np.array([])

    # Assicuriamoci che gli episodi siano ordinati per numero
    sorted_episodes = sorted(episode_dict.keys())

    # Calcola la media per ogni episodio
    mean_vector = np.array([np.mean(episode_dict[i]) for i in sorted_episodes])

    return mean_vector


def easy_3_line_plotter(line1, line2, line3=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=line1,
        mode='lines',
        name="Dynamic appr."
    ))
    fig.add_trace(go.Scatter(
        y=line2,
        mode='lines',
        name="Static appr."
    ))
    if line3 is not None:
        fig.add_trace(go.Scatter(
            y=line3,
            mode='lines',
            name="No rebalance"
        ))

    # --- Map timeslots to weekdays using your dictionary ---
    timeslots_per_day = 8
    num_timeslots = len(line1)

    # Positions for each full day
    tick_positions = list(range(0, num_timeslots, timeslots_per_day))

    # Choose starting weekday (0 = monday)
    start_day_index = 0

    # Create repeating weekday labels
    tick_labels = [
        num2days[(start_day_index + (i // timeslots_per_day)) % 7]
        for i in tick_positions
    ]

    # Update layout
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            title=dict(text="Day", font=dict(size=label_size))
        )
    )

    fig.update_layout(
        title=dict(
            text="Comparison between aproaches",
            font=dict(size=title_size, family="Poppins, sans-serif")
        ),
        xaxis=dict(
            title=dict(text="Day", font=dict(size=label_size)),
            tickfont=dict(size=tick_size)
        ),
        yaxis=dict(
            title=dict(text="Average # of Failures", font=dict(size=label_size)),
            tickfont=dict(size=tick_size)
        ),
        legend=dict(
            x=0.7,
            y=0.9,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=legend_size)
        )
    )

    return fig