"""Dash callbacks for the results webapp."""

from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import Input, Output, State, html

from results_webapp.data_loader import (
    discover_runs,
    get_available_episodes,
    load_episode_data,
    load_run_config,
    load_summary_data,
    load_concatenated_timeslot_data,
    load_concatenated_step_data
)
from results_webapp.plotting import (
    COLORS,
    create_action_distribution_plot,
    create_metric_plot,
    create_reward_tracking_plot,
    create_timeslot_plot,
)

def register_callbacks(app):
    """Register all callbacks for the Dash app."""

    # ============================================================================
    # Callback: Update run options
    # ============================================================================
    @app.callback(
        Output('run-selector', 'options'),
        Output('run-selector', 'value'),
        Input('interval-component', 'n_intervals'),
        State('run-selector', 'value')  # ← Remember current selection
    )
    def update_run_options(n, current_run):
        """Discover available runs and preserve selection."""
        runs = discover_runs(app.results_path)
        options = [{'label': label, 'value': str(path)} for label, path in runs.items()]

        if not options:
            return [], None

        # Check if current selection is still valid
        run_values = [opt['value'] for opt in options]

        if current_run and current_run in run_values:
            # Keep the current selection
            return options, current_run

        # If no current selection or it's invalid, default to first
        return options, options[0]['value']

    # ============================================================================
    # Callback: Update episode options
    # ============================================================================
    @app.callback(
        Output('episode-selector', 'options'),
        Output('episode-selector', 'value'),
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
        Input('interval-component', 'n_intervals'),
        State('episode-selector', 'value')
    )
    def update_episode_options(run_path, mode, n, current_episode):
        """Update available episodes based on selected run and mode."""
        if not run_path:
            return [], None

        run_dir = Path(run_path)
        episodes = get_available_episodes(run_dir, mode)

        options = [{'label': f'Episode {ep}', 'value': ep} for ep in episodes]

        if not options:
            return [], None

        # If current episode is still valid, keep it
        if current_episode is not None and current_episode in episodes:
            return options, current_episode

        # Otherwise, default to last episode
        return options, options[-1]['value']

    # ============================================================================
    # Callback: Display configuration
    # ============================================================================
    @app.callback(
        Output('config-display', 'children'),
        Input('run-selector', 'value')
    )
    def display_config(run_path):
        """Display run configuration."""
        if not run_path:
            return html.P('No run selected', className='text-muted')

        config = load_run_config(Path(run_path))
        if not config:
            return html.P('Configuration not available', className='text-muted')

        params = config.get('hyperparameters', {})

        lr = params.get('lr')
        if lr is not None:
            lr_str = f"1e-{abs(int(np.log10(lr)))}" if lr < 1e-2 else f"{lr}"
        else:
            lr_str = 'N/A'

        return dbc.Row([
            dbc.Col([
                html.Strong('Episodes: '), f"{params.get('num_episodes', 'N/A')}"
            ], width=3),
            dbc.Col([
                html.Strong('Batch Size: '), f"{params.get('batch_size', 'N/A')}"
            ], width=3),
            dbc.Col([
                html.Strong('Learning Rate: '), f"{lr_str}"
            ], width=3),
            dbc.Col([
                html.Strong('Gamma: '), f"{params.get('gamma', 'N/A')}"
            ], width=3),
        ])

    # ============================================================================
    # Callback: Update overview plots (OPTIMIZED - USES SUMMARY DATA ONLY)
    # ============================================================================
    @app.callback(
        Output('mean-failures-plot', 'figure'),
        Output('rewards-plot', 'figure'),
        Output('epsilon-plot', 'figure'),
        Output('total-failures-plot', 'figure'),
        Output('qvalues-overview-plot', 'figure'),
        Output('global-critic-overview-plot', 'figure'),
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('episode-selector', 'value')
    )
    def update_overview_plots(run_path, mode, n, current_episode):
        """Update overview plots using ONLY summary data (fast!)."""

        empty_fig = go.Figure()
        empty_fig.update_layout(title='No data available')

        if not run_path:
            empty_fig.update_layout(title='No run selected')
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        run_dir = Path(run_path)

        # Load summary data (FAST - single CSV or build from scalars.json only)
        summary = load_summary_data(run_dir, mode)

        # Check if we have any data
        if summary is None or summary.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f'No {mode} data available yet')
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        # ============================================
        # Plot 1: Mean Failures per Episode
        # ============================================
        if 'mean_failures' in summary.columns:
            mean_failures_series = summary.set_index('episode')['mean_failures']
            mean_failures_fig = create_metric_plot(
                mean_failures_series,
                f'{mode.capitalize()} Mean Failures per Episode',
                'Mean Failures',
                show_cumulative=False,
                color=COLORS['primary']
            )
        else:
            mean_failures_fig = go.Figure()
            mean_failures_fig.update_layout(title='No mean failures data')

        # ============================================
        # Plot 2: Total Rewards per Episode
        # ============================================
        if 'total_reward' in summary.columns:
            rewards_series = summary.set_index('episode')['total_reward']
            rewards_fig = create_metric_plot(
                rewards_series,
                f'{mode.capitalize()} Total Reward per Episode',
                'Total Reward',
                show_cumulative=True,
                color=COLORS['secondary']
            )
        else:
            rewards_fig = go.Figure()
            rewards_fig.update_layout(title='No reward data')

        # ============================================
        # Plot 3: Epsilon Decay
        # ============================================
        if 'epsilon' in summary.columns:
            epsilon_series = summary.set_index('episode')['epsilon']
            epsilon_fig = create_metric_plot(
                epsilon_series,
                f'{mode.capitalize()} Epsilon Decay',
                'Epsilon',
                show_cumulative=False,
                color=COLORS['warning']
            )
        else:
            epsilon_fig = go.Figure()
            epsilon_fig.update_layout(title='No epsilon data')

        # ============================================
        # Plot 4: Total Failures per Episode
        # ============================================
        if 'total_failures' in summary.columns:
            failures_series = summary.set_index('episode')['total_failures']
            failures_fig = create_metric_plot(
                failures_series,
                f'{mode.capitalize()} Total Failures per Episode',
                'Total Failures',
                show_cumulative=True,
                color=COLORS['danger']
            )
        else:
            failures_fig = go.Figure()
            failures_fig.update_layout(title='No failure data')

        # ============================================
        # Plots 5-7: Training-only metrics
        # ============================================
        if mode == 'training':
            # These require loading individual episode data
            # We'll show placeholders for now unless we need them

            qval_fig = go.Figure()
            qval_fig.update_layout(
                title='Q-values: Select specific episode in "Episode Details" tab',
                annotations=[{
                    'text': 'Viewing Q-values requires loading individual episodes.<br>Use the "Episode Details" tab for detailed analysis.',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 12, 'color': 'gray'}
                }]
            )

            critic_fig = go.Figure()
            critic_fig.update_layout(
                title='Global Critic: Select specific episode in "Episode Details" tab',
                annotations=[{
                    'text': 'Viewing global critic scores requires loading individual episodes.<br>Use the "Episode Details" tab for detailed analysis.',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 12, 'color': 'gray'}
                }]
            )

        else:  # validation mode
            # Show message for training-only plots
            qval_fig = go.Figure()
            qval_fig.update_layout(
                title='Q-values not available in validation mode',
                annotations=[{
                    'text': 'Q-values are only tracked during training',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }]
            )

            critic_fig = go.Figure()
            critic_fig.update_layout(
                title='Global Critic not available in validation mode',
                annotations=[{
                    'text': 'Global Critic is only tracked during training',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }]
            )

        return mean_failures_fig, rewards_fig, epsilon_fig, failures_fig, qval_fig, critic_fig

    # ============================================================================
    # Callback: Update episode detail plots
    # ============================================================================
    @app.callback(
        Output('episode-stats-cards', 'children'),
        Output('timeslot-rewards-plot', 'figure'),
        Output('timeslot-failures-plot', 'figure'),
        Output('action-distribution-plot', 'figure'),
        Output('reward-tracking-plot', 'figure'),
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
        Input('episode-selector', 'value')
    )
    def update_episode_details(run_path, mode, episode):
        """Update episode detail plots."""
        empty_fig = go.Figure()
        empty_fig.update_layout(title='No data available')

        if not run_path or episode is None:
            return html.P('Select an episode'), empty_fig, empty_fig, empty_fig, empty_fig

        run_dir = Path(run_path)
        episode_data = load_episode_data(run_dir, mode, episode)

        if not episode_data:
            return html.P('Episode data not available'), empty_fig, empty_fig, empty_fig, empty_fig

        scalars = episode_data.get('scalars', {})
        timeslot_df = episode_data.get('timeslot_metrics')
        step_data = episode_data.get('step_data', {})

        # Stats cards
        stats_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scalars.get('total_failures', 'N/A')}", className='text-danger'),
                        html.P('Total Failures', className='text-muted mb-0')
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scalars.get('total_reward', 0):.2f}", className='text-success'),
                        html.P('Total Reward', className='text-muted mb-0')
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scalars.get('mean_failures', 0):.2f}", className='text-warning'),
                        html.P('Mean Failures', className='text-muted mb-0')
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{scalars.get('epsilon', 0):.3f}", className='text-info'),
                        html.P('Epsilon', className='text-muted mb-0')
                    ])
                ])
            ], width=3),
        ])

        # Timeslot plots
        if timeslot_df is not None:
            rewards_plot = create_timeslot_plot(
                timeslot_df, 'reward', 'Rewards per Timeslot', 'Reward'
            )
            failures_plot = create_timeslot_plot(
                timeslot_df, 'failures', 'Failures per Timeslot', 'Failures'
            )
        else:
            rewards_plot = empty_fig
            failures_plot = empty_fig

        # Action distribution
        actions = step_data.get('actions', [])
        if actions:
            action_plot = create_action_distribution_plot(actions)
        else:
            action_plot = empty_fig

        # Reward tracking
        reward_tracking = step_data.get('reward_tracking', {})
        if reward_tracking:
            reward_track_plot = create_reward_tracking_plot(reward_tracking)
        else:
            reward_track_plot = empty_fig

        return stats_cards, rewards_plot, failures_plot, action_plot, reward_track_plot
