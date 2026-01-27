"""Plotting utilities for the results webapp."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
        yaxis_label: str
) -> go.Figure:
    """
    Create a plot for per-timeslot metrics.

    Args:
        timeslot_df: DataFrame with timeslot metrics
        metric: Column name to plot
        title: Plot title
        yaxis_label: Y-axis label

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timeslot_df['timeslot'],
        y=timeslot_df[metric],
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
