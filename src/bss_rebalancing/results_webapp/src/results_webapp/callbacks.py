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
    # Callback: Update overview plots (FIXED)
    # ============================================================================
    @app.callback(
        Output('failures-plot', 'figure'),
        Output('rewards-plot', 'figure'),
        Output('epsilon-plot', 'figure'),
        Output('deployed-bikes-overview-plot', 'figure'),
        Output('qvalues-overview-plot', 'figure'),
        Output('global-critic-overview-plot', 'figure'),
        Output('loss-overview-plot', 'figure'),
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('episode-selector', 'value')
    )
    def update_overview_plots(run_path, mode, n, current_episode):
        """Update overview plots with concatenated timeslot data."""

        empty_fig = go.Figure()
        empty_fig.update_layout(title='No data available')

        if not run_path:
            empty_fig.update_layout(title='No run selected')
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        run_dir = Path(run_path)

        # Load concatenated timeslot data (available for both training and validation)
        rewards = load_concatenated_timeslot_data(run_dir, mode, 'reward')
        failures = load_concatenated_timeslot_data(run_dir, mode, 'failures')
        deployed_bikes = load_concatenated_timeslot_data(run_dir, mode, 'deployed_bikes')

        # Load epsilon from summary (per episode)
        summary = load_summary_data(run_dir, mode)

        # Check if we have any data at all
        if len(rewards) == 0 and len(failures) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f'No {mode} data available yet')
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        # ============================================
        # Plot 1: Failures (both modes)
        # ============================================
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

        # ============================================
        # Plot 2: Rewards (both modes)
        # ============================================
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

        # ============================================
        # Plot 3: Epsilon decay (both modes)
        # ============================================
        if summary is not None and not summary.empty and 'epsilon' in summary.columns:
            epsilon_fig = create_metric_plot(
                summary.set_index('episode')['epsilon'],
                f'{mode.capitalize()} Epsilon Decay',
                'Epsilon',
                show_cumulative=False,
                color=COLORS['warning']
            )
        else:
            epsilon_fig = go.Figure()
            epsilon_fig.update_layout(title='No epsilon data')

        # ============================================
        # Plot 4: Deployed bikes (both modes)
        # ============================================
        if len(deployed_bikes) > 0:
            bikes_fig = create_metric_plot(
                deployed_bikes,
                f'{mode.capitalize()} Deployed Bikes per Timeslot (All Episodes)',
                'Number of Bikes',
                show_cumulative=False,
                color=COLORS['primary']
            )
            bikes_fig.update_layout(xaxis_title='Timeslot (Cumulative)')
        else:
            bikes_fig = go.Figure()
            bikes_fig.update_layout(title='No deployed bikes data')

        # ============================================
        # Plots 5-7: Training-only metrics
        # ============================================
        if mode == 'training':
            # Load training-specific data
            losses = load_concatenated_step_data(run_dir, mode, 'losses')
            q_values = load_concatenated_step_data(run_dir, mode, 'q_values')
            global_critic_scores = load_concatenated_step_data(run_dir, mode, 'global_critic_scores')

            # Plot 5: Q-values
            if len(q_values) > 0:
                qval_fig = create_metric_plot(
                    q_values,
                    'Mean Q-Values per Timeslot (All Episodes)',
                    'Q-Value',
                    show_cumulative=False,
                    color=COLORS['primary']
                )
                qval_fig.update_layout(xaxis_title='Timeslot (Cumulative)')
            else:
                qval_fig = go.Figure()
                qval_fig.update_layout(title='No Q-value data')

            # Plot 6: Global Critic Scores
            if len(global_critic_scores) > 0:
                window_size = min(200, len(global_critic_scores) // 20) if len(global_critic_scores) > 20 else 1
                critic_fig = go.Figure()

                if window_size > 1:
                    critic_fig.add_trace(go.Scatter(
                        y=global_critic_scores,
                        mode='lines',
                        name='Global Critic',
                        line=dict(color=COLORS['primary'], width=1),
                        opacity=0.3
                    ))
                    critic_fig.add_trace(go.Scatter(
                        y=global_critic_scores.rolling(window=window_size).mean(),
                        mode='lines',
                        name=f'Moving Avg ({window_size})',
                        line=dict(color=COLORS['secondary'], width=2)
                    ))
                else:
                    critic_fig.add_trace(go.Scatter(
                        y=global_critic_scores,
                        mode='lines',
                        name='Global Critic',
                        line=dict(color=COLORS['secondary'], width=2)
                    ))

                critic_fig.update_layout(
                    title='Global Critic Score (All Episodes)',
                    xaxis_title='Step (Cumulative)',
                    yaxis_title='Score',
                    hovermode='x unified'
                )
            else:
                critic_fig = go.Figure()
                critic_fig.update_layout(title='No global critic data')

            # Plot 7: Training Loss
            if current_episode is not None:
                episode_data = load_episode_data(run_dir, mode, current_episode)

                if episode_data and 'step_data' in episode_data:
                    losses = episode_data['step_data'].get('losses', [])
                    # Filter out None values
                    losses = [l for l in losses if l is not None]

                    if len(losses) > 0:
                        losses_series = pd.Series(losses)
                        window_size = min(100, len(losses_series) // 10) if len(losses_series) > 10 else 1
                        loss_fig = go.Figure()

                        if window_size > 1:
                            loss_fig.add_trace(go.Scatter(
                                y=losses_series,
                                mode='lines',
                                name='Loss',
                                line=dict(color=COLORS['primary'], width=1),
                                opacity=0.3
                            ))
                            loss_fig.add_trace(go.Scatter(
                                y=losses_series.rolling(window=window_size).mean(),
                                mode='lines',
                                name=f'Moving Avg ({window_size})',
                                line=dict(color=COLORS['danger'], width=2)
                            ))
                        else:
                            loss_fig.add_trace(go.Scatter(
                                y=losses_series,
                                mode='lines',
                                name='Loss',
                                line=dict(color=COLORS['danger'], width=2)
                            ))

                        loss_fig.update_layout(
                            title=f'Training Loss (Episode {current_episode})',
                            xaxis_title='Step',
                            yaxis_title='Loss',
                            hovermode='x unified'
                        )
                    else:
                        loss_fig = go.Figure()
                        loss_fig.update_layout(title=f'No loss data for episode {current_episode}')
                else:
                    loss_fig = go.Figure()
                    loss_fig.update_layout(title='Select an episode to view loss')
            else:
                loss_fig = go.Figure()
                loss_fig.update_layout(title='Select an episode to view training loss')


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

            loss_fig = go.Figure()
            loss_fig.update_layout(
                title='Training Loss not available in validation mode',
                annotations=[{
                    'text': 'Loss is only computed during training',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }]
            )

        return failures_fig, rewards_fig, epsilon_fig, bikes_fig, qval_fig, critic_fig, loss_fig

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
