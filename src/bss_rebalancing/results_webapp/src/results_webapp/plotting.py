"""Plotting utilities for the results webapp."""

from typing import Dict, List, Optional

import io
import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import osmnx as ox
import geopandas as gpd

matplotlib.use('Agg')

# Color scheme
COLORS = {
    'primary': '#4A90E2',
    'secondary': '#50C878',
    'warning': '#FFB84D',
    'danger': '#E74C3C',
    'dark': '#2C3E50',
    'light': '#ECF0F1'
}


def create_metric_plot(
        data: pd.Series,
        title: str,
        yaxis_label: str,
        show_cumulative: bool = False,
        color: str = COLORS['primary']
) -> go.Figure:
    """
    Create a line plot for a metric.

    Args:
        data: Pandas Series with metric values
        title: Plot title
        yaxis_label: Y-axis label
        show_cumulative: Whether to show cumulative mean
        color: Line color

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name='Value',
        line=dict(color=color, width=2),
        hovertemplate='Episode: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))

    # Cumulative mean
    if show_cumulative and len(data) > 1:
        cumulative_mean = data.expanding().mean()
        fig.add_trace(go.Scatter(
            x=cumulative_mean.index,
            y=cumulative_mean.values,
            mode='lines',
            name='Cumulative Mean',
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
            hovertemplate='Episode: %{x}<br>Mean: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis=dict(title='Episode', gridcolor='lightgray'),
        yaxis=dict(title=yaxis_label, gridcolor='lightgray'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


def create_timeslot_plot(
        timeslot_df: pd.DataFrame,
        metric: str,
        title: str,
        yaxis_label: str,
        scale: float = 1.0
) -> go.Figure:
    """
    Create a plot for per-timeslot metrics.

    Args:
        timeslot_df: DataFrame with timeslot metrics
        metric: Column name to plot
        title: Plot title
        yaxis_label: Y-axis label
        scale:

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timeslot_df['timeslot'],
        y=timeslot_df[metric]*scale,
        mode='lines+markers',
        name=metric,
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title='Timeslot', gridcolor='lightgray'),
        yaxis=dict(title=yaxis_label, gridcolor='lightgray'),
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def create_action_distribution_plot(actions: List[int]) -> go.Figure:
    """
    Create bar plot of action distribution.

    Args:
        actions: List of action indices

    Returns:
        Plotly Figure object
    """
    action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'DROP', 'PICKUP', 'CHARGE']
    action_counts = [actions.count(i) for i in range(8)]

    fig = go.Figure(data=[
        go.Bar(
            x=action_names,
            y=action_counts,
            marker=dict(color=COLORS['primary']),
            hovertemplate='%{x}: %{y} times<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text='Action Distribution', font=dict(size=18)),
        xaxis=dict(title='Action'),
        yaxis=dict(title='Count'),
        template='plotly_white'
    )

    return fig


def create_comparison_plot(
        train_summary: pd.DataFrame,
        val_summary: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Create comparison plot between training and validation.

    Args:
        train_summary: Training summary DataFrame
        val_summary: Validation summary DataFrame (optional)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Training failures
    fig.add_trace(go.Scatter(
        x=train_summary['episode'],
        y=train_summary['total_failures'],
        mode='lines',
        name='Training',
        line=dict(color=COLORS['primary'], width=2)
    ))

    # Validation failures (if available)
    if val_summary is not None and not val_summary.empty:
        fig.add_trace(go.Scatter(
            x=val_summary['episode'],
            y=val_summary['total_failures'],
            mode='lines+markers',
            name='Validation',
            line=dict(color=COLORS['secondary'], width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title=dict(text='Training vs Validation Failures', font=dict(size=20)),
        xaxis=dict(title='Episode', gridcolor='lightgray'),
        yaxis=dict(title='Total Failures', gridcolor='lightgray'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


def create_reward_tracking_plot(reward_tracking: Dict[int, List[float]]) -> go.Figure:
    """
    Create plot showing rewards per action type.

    Args:
        reward_tracking: Dictionary mapping action indices to reward lists

    Returns:
        Plotly Figure object
    """
    action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'DROP', 'PICKUP', 'CHARGE']

    mean_rewards = [np.mean(reward_tracking.get(i, [0])) for i in range(8)]

    fig = go.Figure(data=[
        go.Bar(
            x=action_names,
            y=mean_rewards,
            marker=dict(
                color=mean_rewards,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Mean Reward')
            ),
            hovertemplate='%{x}: %{y:.3f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text='Mean Reward per Action', font=dict(size=18)),
        xaxis=dict(title='Action'),
        yaxis=dict(title='Mean Reward'),
        template='plotly_white'
    )

    return fig




# ── Per-metric colormap config ────────────────────────────────────────────────
GRAPH_METRIC_CONFIG = {
    'critic_mean':      {'label': 'Critic Score (Mean)', 'fmt': '.3f'},
    'eligibility_mean': {'label': 'Eligibility (Mean)',  'fmt': '.3f'},
    'failure_sum':      {'label': 'Failure Sum',         'fmt': 'd'},
    'visits_sum':       {'label': 'Visits Sum',          'fmt': 'd'},
    'ops_sum':          {'label': 'Operations Sum',      'fmt': 'd'},
    'bikes_mean':       {'label': 'Bikes (Mean)',        'fmt': '.1f'},
}

_CMAP = 'viridis'

def create_graph_heatmap_plot(
    base_graph,
    cell_subgraph,
    metric: str,
    percentage: bool = False,
    out_path=None,
) -> str:

    # --- Layer 1: OSMnx base graph ---
    fig, ax = ox.plot_graph(
        base_graph,
        show=False,
        close=False,
        bgcolor="white",
        edge_color="#1a1a1a",
        edge_linewidth=0.5,
        node_size=0
    )

    # --- Layer 2: Cells grid + labels in single loop ---
    cell_geoms = []
    cell_values = []
    vmin, vmax = float("inf"), float("-inf")

    total_visits = None
    if metric == 'visits_sum':
        # Special case: convert visits_sum to percentage of total visits
        total_visits = sum(data.get('visits_sum', 0) for _, data in cell_subgraph.nodes(data=True))
        percentage = True

    for node, data in cell_subgraph.nodes(data=True):
        b = data.get('boundary')
        if b is None:
            continue
        v = data.get(metric)
        val = float(v) if v is not None else 0.0
        cell_geoms.append(b)
        cell_values.append(val)
        if val < vmin: vmin = val
        if val > vmax: vmax = val

        cx, cy = b.centroid.x, b.centroid.y
        fmt = GRAPH_METRIC_CONFIG.get(metric, {}).get('fmt', '.2f')
        if metric == 'visits_sum' and total_visits:
            val = val / total_visits if total_visits > 0 else 0.0
        label = (f"{val * 100:.2f}%" if percentage
                 else f"{int(val)}" if fmt == 'd'
                 else f"{val:{fmt}}")
        ax.text(cx, cy, label,
                ha="center", va="center",
                fontsize=8, color="black", fontweight='bold',
                zorder=3)

    if not cell_geoms:
        plt.close(fig)
        return ""

    cell_gdf = gpd.GeoDataFrame({"value": cell_values, "geometry": cell_geoms}, crs="EPSG:4326")

    norm = mcolors.Normalize(vmin=min(vmin, 0.0), vmax=max(vmax, 0.1))
    cmap = plt.colormaps["viridis"]

    cell_gdf.plot(
        ax=ax,
        column="value",
        cmap="viridis",
        norm=norm,
        alpha=0.6,
        linewidth=1.5,
        edgecolor=(0, 0, 0, 1.0),
        zorder=2,
        legend=False
    )

    # Re-apply zorder on labels after gdf.plot() draws over them
    for txt in ax.texts:
        txt.set_zorder(4)

    # Extend view to fit all cells
    total_bounds = cell_gdf.total_bounds
    pad = 0.0005
    ax.set_xlim(total_bounds[0] - pad, total_bounds[2] + pad)
    ax.set_ylim(total_bounds[1] - pad, total_bounds[3] + pad)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, aspect=30)
    cbar.set_label(
        GRAPH_METRIC_CONFIG.get(metric, {}).get('label', metric),
        fontsize=10, rotation=270, labelpad=14
    )
    cbar.ax.tick_params(labelsize=8)

    if out_path is not None:
        plt.savefig(out_path, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        return ""
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return f"image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
