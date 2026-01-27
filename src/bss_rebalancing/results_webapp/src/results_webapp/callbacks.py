"""Dash callbacks for the results webapp."""

from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        Input('interval-component', 'n_intervals')
    )
    def update_run_options(n):
        """Discover available runs."""
        runs = discover_runs(app.results_path)
        options = [{'label': label, 'value': str(path)} for label, path in runs.items()]

        if options:
            return options, options[0]['value']
        return [], None

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

        print(f"[DEBUG] Found {len(episodes)} episodes in {run_dir.name}/{mode}")

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

        return dbc.Row([
            dbc.Col([
                html.Strong('Episodes: '), f"{params.get('num_episodes', 'N/A')}"
            ], width=3),
            dbc.Col([
                html.Strong('Batch Size: '), f"{params.get('batch_size', 'N/A')}"
            ], width=3),
            dbc.Col([
                html.Strong('Learning Rate: '), f"{params.get('lr', 'N/A')}"
            ], width=3),
            dbc.Col([
                html.Strong('Gamma: '), f"{params.get('gamma', 'N/A')}"
            ], width=3),
        ])

    # ============================================================================
    # Callback: Update overview plots (UPDATED - concatenated timeslot data)
    # ============================================================================
    @app.callback(
        Output('failures-plot', 'figure'),
        Output('rewards-plot', 'figure'),
        Output('epsilon-plot', 'figure'),
        Output('comparison-plot', 'figure'),  # Now shows loss instead of comparison
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
        Input('interval-component', 'n_intervals')
    )
    def update_overview_plots(run_path, mode, n):
        """Update overview plots with concatenated timeslot data."""

        if not run_path:
            empty_fig = go.Figure()
            empty_fig.update_layout(title='No run selected')
            return empty_fig, empty_fig, empty_fig, empty_fig

        run_dir = Path(run_path)

        # Load concatenated timeslot data
        rewards = load_concatenated_timeslot_data(run_dir, mode, 'reward')
        failures = load_concatenated_timeslot_data(run_dir, mode, 'failures')
        deployed_bikes = load_concatenated_timeslot_data(run_dir, mode, 'deployed_bikes')

        # Load epsilon from summary (per episode)
        summary = load_summary_data(run_dir, mode)

        # Initialize losses variable
        losses = pd.Series()

        # For training mode, also load losses and q-values
        if mode == 'training':
            losses = load_concatenated_step_data(run_dir, mode, 'losses')
            q_values = load_concatenated_step_data(run_dir, mode, 'q_values')

        if len(rewards) == 0 and len(failures) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f'No {mode} data available yet')
            return empty_fig, empty_fig, empty_fig, empty_fig

        # Rewards plot (concatenated timeslots)
        if len(rewards) > 0:
            rewards_fig = create_metric_plot(
                rewards,
                f'{mode.capitalize()} Rewards per Timeslot (All Episodes)',
                'Reward',
                show_cumulative=True,
                color=COLORS['secondary']
            )
            rewards_fig.update_layout(xaxis_title='Timeslot (Cumulative)')
        else:
            rewards_fig = go.Figure()
            rewards_fig.update_layout(title='No reward data')

        # Failures plot (concatenated timeslots)
        if len(failures) > 0:
            failures_fig = create_metric_plot(
                failures,
                f'{mode.capitalize()} Failures per Timeslot (All Episodes)',
                'Failures',
                show_cumulative=True,
                color=COLORS['danger']
            )
            failures_fig.update_layout(xaxis_title='Timeslot (Cumulative)')
        else:
            failures_fig = go.Figure()
            failures_fig.update_layout(title='No failure data')

        # Epsilon plot (per episode)
        if summary is not None and not summary.empty and 'epsilon' in summary.columns:
            epsilon_fig = create_metric_plot(
                summary.set_index('episode')['epsilon'],
                'Epsilon Decay',
                'Epsilon',
                show_cumulative=False,
                color=COLORS['warning']
            )
        else:
            epsilon_fig = go.Figure()
            epsilon_fig.update_layout(title='No epsilon data')

        # Fourth plot: Training Loss (for training mode) or Deployed Bikes (for validation)
        if mode == 'training' and len(losses) > 0:
            # Training loss with smoothing
            window_size = min(200, len(losses) // 20) if len(losses) > 20 else 1
            fourth_fig = go.Figure()
            fourth_fig.add_trace(go.Scatter(
                y=losses,
                mode='lines',
                name='Loss',
                line=dict(color=COLORS['primary'], width=1),
                opacity=0.3
            ))
            if window_size > 1:
                fourth_fig.add_trace(go.Scatter(
                    y=losses.rolling(window=window_size).mean(),
                    mode='lines',
                    name=f'Moving Avg ({window_size})',
                    line=dict(color=COLORS['danger'], width=2)
                ))
            fourth_fig.update_layout(
                title='Training Loss (All Episodes)',
                xaxis_title='Step (Cumulative)',
                yaxis_title='Loss',
                hovermode='x unified'
            )
        elif len(deployed_bikes) > 0:
            # Deployed bikes for validation or if no loss data
            fourth_fig = create_metric_plot(
                deployed_bikes,
                'Deployed Bikes per Timeslot (All Episodes)',
                'Number of Bikes',
                show_cumulative=False,
                color=COLORS['primary']
            )
            fourth_fig.update_layout(xaxis_title='Timeslot (Cumulative)')
        else:
            fourth_fig = go.Figure()
            fourth_fig.update_layout(title='No additional data')

        return failures_fig, rewards_fig, epsilon_fig, fourth_fig

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

    # ============================================================================
    # Callback: Update training dynamics plots
    # ============================================================================
    @app.callback(
        Output('loss-plot', 'figure'),
        Output('qvalues-plot', 'figure'),
        Output('global-critic-scores-plot', 'figure'),
        Output('deployed-bikes-plot', 'figure'),
        Input('run-selector', 'value'),
        Input('episode-selector', 'value')
    )
    def update_training_dynamics(run_path, episode):
        """Update training dynamics plots."""
        print("[DEBUG] Updating training dynamics plots")
        empty_fig = go.Figure()
        empty_fig.update_layout(title='No data available')

        if not run_path or episode is None:
            return empty_fig, empty_fig, empty_fig, empty_fig

        run_dir = Path(run_path)
        episode_data = load_episode_data(run_dir, 'training', episode)

        if not episode_data:
            return empty_fig, empty_fig, empty_fig, empty_fig

        step_data = episode_data.get('step_data', {})
        timeslot_df = episode_data.get('timeslot_metrics')

        # Loss plot
        losses = step_data.get('losses', [])
        if losses:
            # Filter out None values and create moving average
            valid_losses = [l for l in losses if l is not None]
            if valid_losses:
                loss_series = pd.Series(valid_losses)
                # Add moving average for smoothing
                window_size = min(100, len(loss_series) // 10)
                if window_size > 1:
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(
                        y=loss_series,
                        mode='lines',
                        name='Loss',
                        line=dict(color=COLORS['primary'], width=1),
                        opacity=0.5
                    ))
                    loss_fig.add_trace(go.Scatter(
                        y=loss_series.rolling(window=window_size).mean(),
                        mode='lines',
                        name=f'Moving Avg ({window_size})',
                        line=dict(color=COLORS['danger'], width=2)
                    ))
                    loss_fig.update_layout(
                        title=dict(text='Training Loss', font=dict(size=18)),
                        xaxis=dict(title='Step', gridcolor='lightgray'),
                        yaxis=dict(title='Loss', gridcolor='lightgray'),
                        template='plotly_white',
                        hovermode='x unified'
                    )
                else:
                    loss_fig = create_metric_plot(
                        loss_series, 'Training Loss', 'Loss', show_cumulative=False
                    )
            else:
                loss_fig = empty_fig
        else:
            loss_fig = empty_fig

        # Q-values plot (mean per timeslot)
        q_values = step_data.get('q_values', [])
        if q_values:
            mean_qvals = [np.mean(q) if len(q) > 0 else 0 for q in q_values]
            qval_series = pd.Series(mean_qvals)
            qval_fig = create_metric_plot(
                qval_series, 'Mean Q-Values per Timeslot', 'Q-Value', show_cumulative=False
            )
        else:
            qval_fig = empty_fig

        # Global critics plot
        global_critic_scores = step_data.get('global_critic_scores', [])
        if global_critic_scores:
            critic_series = pd.Series(global_critic_scores)
            # Add moving average for smoothing
            window_size = min(100, len(critic_series) // 10)
            if window_size > 1:
                critic_fig = go.Figure()
                critic_fig.add_trace(go.Scatter(
                    y=critic_series,
                    mode='lines',
                    name='Global Critic',
                    line=dict(color=COLORS['primary'], width=1),
                    opacity=0.5
                ))
                critic_fig.add_trace(go.Scatter(
                    y=critic_series.rolling(window=window_size).mean(),
                    mode='lines',
                    name=f'Moving Avg ({window_size})',
                    line=dict(color=COLORS['secondary'], width=2)
                ))
                critic_fig.update_layout(
                    title=dict(text='Global Critic Score', font=dict(size=18)),
                    xaxis=dict(title='Step', gridcolor='lightgray'),
                    yaxis=dict(title='Score', gridcolor='lightgray'),
                    template='plotly_white',
                    hovermode='x unified'
                )
            else:
                critic_fig = create_metric_plot(
                    critic_series, 'Global Critic Score', 'Score', show_cumulative=False
                )
        else:
            critic_fig = empty_fig

        # Deployed bikes plot
        if timeslot_df is not None and 'deployed_bikes' in timeslot_df.columns:
            bikes_fig = create_timeslot_plot(
                timeslot_df, 'deployed_bikes', 'Deployed Bikes per Timeslot', 'Bikes'
            )
        else:
            bikes_fig = empty_fig

        return loss_fig, qval_fig, critic_fig, bikes_fig
