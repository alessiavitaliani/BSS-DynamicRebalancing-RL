"""Dash callbacks for the results webapp."""

from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from dash import Input, Output, State, html

from results_webapp.data_loader import (
    discover_runs,
    get_available_episodes,
    load_episode_data,
    load_run_config,
    load_summary_data,
    load_base_graph,
    load_best_model_metadata,
)
from results_webapp.plotting import (
    COLORS,
    create_action_distribution_plot,
    create_graph_heatmap_plot,
    create_metric_plot,
    create_reward_tracking_plot,
    create_timeslot_plot,
)

# ---------------------------------------------------------------------------
# Helpers shared across callbacks
# ---------------------------------------------------------------------------

def _empty_fig(title: str = 'No data available') -> go.Figure:
    f = go.Figure()
    f.update_layout(title=title, template='plotly_white')
    return f


def _build_episode_detail_outputs(
    run_dir: Path,
    mode: str,
    episode: int,
    prefix: str,
    is_benchmark: bool = False,
):
    """
    Core logic for episode-detail callbacks.

    Returns a tuple matching:
        stats_cards,
        rewards_plot, failures_plot,
        action_plot, reward_track_plot,
        inside_system_bikes_plot, depot_load_plot, truck_load_plot,
        on_road_bikes_plot, outside_system_bikes_plot, demand_plot

    For benchmark, action_plot and reward_track_plot are always empty figures.
    For benchmark, epsilon card is hidden from stats.
    """
    ef = _empty_fig()

    if not run_dir or episode is None:
        return (html.P('Select an episode'),) + (ef,) * 10

    episode_data = load_episode_data(run_dir, mode, episode)
    if not episode_data:
        return (html.P('Episode data not available'),) + (ef,) * 10

    scalars     = episode_data.get('scalars', {})
    timeslot_df = episode_data.get('timeslot_metrics')
    step_data   = episode_data.get('step_data', {}) or {}

    total_failures = scalars.get('total_failures', 'N/A')
    total_demand = (
        int(timeslot_df['demand'].sum())
        if timeslot_df is not None and 'demand' in timeslot_df.columns
        else 0
    )
    failure_rate = (
        total_failures / total_demand
        if isinstance(total_failures, (int, float)) and total_demand
        else 0
    )

    # ── Stats cards ──────────────────────────────────────────────────────────
    base_cards = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{total_failures}", className='text-danger'),
            html.P('Total Failures', className='text-muted mb-0')
        ]))),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{scalars.get('total_reward', 0):.2f}", className='text-success'),
            html.P('Total Reward', className='text-muted mb-0')
        ]))),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{scalars.get('mean_daily_failures', 0):.2f}", className='text-warning'),
            html.P('Mean Daily Failures', className='text-muted mb-0')
        ]))),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(f"{failure_rate:.2%}", className='text-primary'),
            html.P('Failure Rate', className='text-muted mb-0')
        ]))),
    ]
    if not is_benchmark:
        base_cards.append(
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{scalars.get('epsilon', 0):.3f}", className='text-info'),
                html.P('Epsilon', className='text-muted mb-0')
            ])))
        )
    stats_cards = dbc.Row(base_cards, className='g-3')

    # ── Timeslot plots ────────────────────────────────────────────────────────
    if timeslot_df is not None:
        rewards_plot = create_timeslot_plot(
            timeslot_df, 'reward', 'Rewards per Timeslot', 'Reward'
        ) if 'reward' in timeslot_df.columns else ef

        failures_plot = create_timeslot_plot(
            timeslot_df, 'failures',
            f'Failures per Timeslot — Total: {total_failures}', 'Failures'
        ) if 'failures' in timeslot_df.columns else ef

        inside_plot = create_timeslot_plot(
            timeslot_df, 'inside_system_bikes',
            'Total Bikes Inside System per Timeslot', '# Bikes'
        ) if 'inside_system_bikes' in timeslot_df.columns else ef

        depot_plot = create_timeslot_plot(
            timeslot_df, 'depot_load', 'Depot Load per Timeslot', '# Bikes'
        ) if 'depot_load' in timeslot_df.columns else ef

        truck_plot = create_timeslot_plot(
            timeslot_df, 'truck_load', 'Truck Load per Timeslot', '# Bikes'
        ) if 'truck_load' in timeslot_df.columns else ef

        on_road_plot = create_timeslot_plot(
            timeslot_df, 'deployed_bikes', 'On-Road Bikes per Timeslot', '# Bikes'
        ) if 'deployed_bikes' in timeslot_df.columns else ef

        outside_plot = create_timeslot_plot(
            timeslot_df, 'outside_system_bikes',
            'Bikes Outside System per Timeslot', '# Bikes'
        ) if 'outside_system_bikes' in timeslot_df.columns else ef

        demand_plot = create_timeslot_plot(
            timeslot_df, 'demand',
            f'Demand per Timeslot — Total: {total_demand}', 'Bike Demand'
        ) if 'demand' in timeslot_df.columns else ef
    else:
        rewards_plot = failures_plot = inside_plot = depot_plot = \
            truck_plot = on_road_plot = outside_plot = demand_plot = ef

    # ── Step-level plots (training/validation only) ───────────────────────────
    if is_benchmark:
        action_plot = ef
        reward_track_plot = ef
    else:
        actions = step_data.get('actions', [])
        action_plot = create_action_distribution_plot(actions) if actions else ef

        reward_tracking = step_data.get('reward_tracking_per_action', {})
        reward_track_plot = create_reward_tracking_plot(reward_tracking) if reward_tracking else ef

    return (
        stats_cards,
        rewards_plot, failures_plot,
        action_plot, reward_track_plot,
        inside_plot, depot_plot, truck_plot,
        on_road_plot, outside_plot, demand_plot,
    )


# ---------------------------------------------------------------------------
# Main registration
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register all callbacks for the Dash app."""

    # ========================================================================
    # Callback: Render the correct tab structure based on mode
    # ========================================================================
    @app.callback(
        Output('mode-tabs-container', 'children'),
        Input('mode-selector', 'value'),
    )
    def render_mode_tabs(mode):
        """Swap the entire tab tree when the mode changes."""
        from results_webapp.app import _training_tabs, _validation_tabs, _benchmark_tabs
        if mode == 'training':
            return _training_tabs()
        elif mode == 'validation':
            return _validation_tabs()
        elif mode == 'benchmark':
            return _benchmark_tabs()
        return html.P('Unknown mode', className='text-danger')

    # ========================================================================
    # Callback: Update run options
    # ========================================================================
    @app.callback(
        Output('run-selector', 'options'),
        Output('run-selector', 'value'),
        Input('interval-component', 'n_intervals'),
        State('run-selector', 'value'),
    )
    def update_run_options(n, current_run):
        runs = discover_runs(app.results_path)
        options = [{'label': label, 'value': str(path)} for label, path in runs.items()]

        if not options:
            return [], None

        run_values = [opt['value'] for opt in options]
        if current_run and current_run in run_values:
            return options, current_run
        return options, options[0]['value']

    # ========================================================================
    # Callback: Update episode options
    # ========================================================================
    @app.callback(
        Output('episode-selector', 'options'),
        Output('episode-selector', 'value'),
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
        Input('interval-component', 'n_intervals'),
        State('episode-selector', 'value'),
    )
    def update_episode_options(run_path, mode, n, current_episode):
        if not run_path:
            return [], None

        run_dir = Path(run_path)
        episodes = get_available_episodes(run_dir, mode)

        if mode == 'benchmark':
            options = [{'label': f'Episode {ep}', 'value': ep} for ep in episodes]
            if not options:
                return [], None
            if current_episode is not None and current_episode in episodes:
                return options, current_episode
            return options, options[-1]['value']

        options = [{'label': f'Episode {ep}', 'value': ep} for ep in episodes]
        if not options:
            return [], None

        if current_episode is not None and current_episode in episodes:
            return options, current_episode
        return options, options[-1]['value']

    # ========================================================================
    # Callback: Display configuration
    # ========================================================================
    @app.callback(
        Output('config-display', 'children'),
        Input('run-selector', 'value'),
        Input('mode-selector', 'value'),
    )
    def display_config(run_path, mode):
        if not run_path:
            return html.P('No run selected', className='text-muted')

        config = load_run_config(Path(run_path), mode)
        if not config:
            return html.P(
                f'Configuration not available for mode "{mode}"',
                className='text-muted'
            )

        params = config.get('hyperparameters', {})

        lr = params.get('lr')
        if lr is not None:
            lr_str = f"1e-{abs(int(np.log10(lr)))}" if lr < 1e-2 else f"{lr}"
        else:
            lr_str = 'N/A'

        return dbc.Tab([
            dbc.Row([
                dbc.Col([html.Strong('Seed: '),               f"{params.get('seed', 'N/A')}"],             width=2),
                dbc.Col([html.Strong('Episodes: '),           f"{params.get('num_episodes', 'N/A')}"],     width=2),
                dbc.Col([html.Strong('Batch Size: '),         f"{params.get('batch_size', 'N/A')}"],       width=2),
                dbc.Col([html.Strong('Learning Rate: '),      lr_str],                                     width=2),
                dbc.Col([html.Strong('Gamma: '),              f"{params.get('gamma', 'N/A')}"],            width=2),
                dbc.Col([html.Strong('Exploration Time: '),   f"{params.get('exploration_time', 'N/A')}"], width=2),
            ]),
            dbc.Row([
                dbc.Col([html.Strong('Soft update: '),        f"{params.get('soft_update', 'N/A')}"],          width=2),
                dbc.Col([html.Strong('Tau: '),                f"{params.get('tau', 'N/A')}"],                  width=2),
                dbc.Col([html.Strong('Initial Repositioning: '), f"{params.get('enable_repositioning', 'N/A')}"], width=2),
                dbc.Col([html.Strong('Net flow IR: '),        f"{params.get('use_net_flow', 'N/A')}"],       width=2),
                dbc.Col([html.Strong('Max bikes: '),          f"{params.get('maximum_number_of_bikes', 'N/A')}"], width=2),
                dbc.Col([html.Strong('Min bikes: '),          f"{params.get('minimum_number_of_bikes', 'N/A')}"], width=2),
            ]),
        ])

    # ========================================================================
    # Callback: Training — Overview plots
    # ========================================================================
    @app.callback(
        Output('mean-failures-plot', 'figure'),
        Output('rewards-plot',       'figure'),
        Output('epsilon-plot',       'figure'),
        Output('total-failures-plot','figure'),
        Input('run-selector',        'value'),
        Input('mode-selector',       'value'),
        Input('interval-component',  'n_intervals'),
        Input('episode-selector',    'value'),
    )
    def update_overview_plots(run_path, mode, n, current_episode):
        """Overview tab — training only.  Returns empty figs for other modes."""
        ef = _empty_fig()

        if mode != 'training' or not run_path:
            return ef, ef, ef, ef

        run_dir = Path(run_path)
        summary = load_summary_data(run_dir, 'training')

        if summary is None or summary.empty:
            nf = _empty_fig('No training data available yet')
            return nf, nf, nf, nf

        def _metric(col, title, ylabel, cumulative=False, color=COLORS['primary']):
            if col in summary.columns:
                return create_metric_plot(
                    summary.set_index('episode')[col],
                    title, ylabel,
                    show_cumulative=cumulative,
                    color=color,
                )
            return _empty_fig(f'No {col} data')

        mean_fail_fig = _metric('mean_daily_failures',
                                'Training Mean Daily Failures per Episode',
                                'Mean Daily Failures', color=COLORS['primary'])
        rewards_fig   = _metric('total_reward',
                                'Training Total Reward per Episode',
                                'Total Reward', cumulative=True, color=COLORS['secondary'])
        epsilon_fig   = _metric('epsilon',
                                'Training Epsilon Decay',
                                'Epsilon', color=COLORS['warning'])
        failures_fig  = _metric('total_failures',
                                'Training Total Failures per Episode',
                                'Total Failures', cumulative=True, color=COLORS['danger'])

        return mean_fail_fig, rewards_fig, epsilon_fig, failures_fig

    # ========================================================================
    # Callback: Training — Episode Details
    # ========================================================================
    @app.callback(
        Output('train-episode-stats-cards',       'children'),
        Output('train-timeslot-rewards-plot',     'figure'),
        Output('train-timeslot-failures-plot',    'figure'),
        Output('train-action-distribution-plot',  'figure'),
        Output('train-reward-tracking-plot',      'figure'),
        Output('train-inside-system-bikes-plot',  'figure'),
        Output('train-depot-load-plot',           'figure'),
        Output('train-truck-load-plot',           'figure'),
        Output('train-on-road-bikes-plot',        'figure'),
        Output('train-outside-system-bikes-plot', 'figure'),
        Output('train-demand',                    'figure'),
        Input('run-selector',    'value'),
        Input('mode-selector',   'value'),
        Input('episode-selector','value'),
    )
    def update_training_episode_details(run_path, mode, episode):
        ef = _empty_fig()
        if mode != 'training':
            return (html.P(''),) + (ef,) * 10
        return _build_episode_detail_outputs(
            Path(run_path) if run_path else None,
            'training', episode, 'train',
        )

    # ========================================================================
    # Callback: Validation — Best tab
    # ========================================================================
    @app.callback(
        Output('best-episode-banner',             'children'),
        Output('best-episode-stats-cards',        'children'),
        Output('best-timeslot-rewards-plot',      'figure'),
        Output('best-timeslot-failures-plot',     'figure'),
        Output('best-action-distribution-plot',   'figure'),
        Output('best-reward-tracking-plot',       'figure'),
        Output('best-inside-system-bikes-plot',   'figure'),
        Output('best-depot-load-plot',            'figure'),
        Output('best-truck-load-plot',            'figure'),
        Output('best-on-road-bikes-plot',         'figure'),
        Output('best-outside-system-bikes-plot',  'figure'),
        Output('best-demand',                     'figure'),
        Input('run-selector',   'value'),
        Input('mode-selector',  'value'),
        Input('interval-component', 'n_intervals'),
    )
    def update_best_tab(run_path, mode, n):
        ef = _empty_fig()
        empty_11 = (ef,) * 11

        if mode != 'validation' or not run_path:
            return (html.P(''),) + (html.P(''),) + (html.P(''),) + empty_11

        run_dir  = Path(run_path)
        metadata = load_best_model_metadata(run_dir)

        if metadata is None:
            banner = dbc.Alert(
                '⚠ No best model metadata found (models/best/metadata.json missing).',
                color='warning'
            )
            return (banner,) + (html.P(''),) + (html.P(''),) + empty_11

        best_ep   = metadata.get('episode')
        best_score= metadata.get('score', 'N/A')

        banner = dbc.Alert(
            [
                html.Strong(f'🏆 Best model: episode {best_ep}'),
                f'  —  score: {best_score:.4f}' if isinstance(best_score, float) else f'  —  score: {best_score}',
            ],
            color='success', className='mb-2'
        )

        # The best-model validation data lives at validation/episode_{best_ep}/episode_000/
        detail_outputs = _build_episode_detail_outputs(
            run_dir, 'validation', best_ep, 'best',
        )
        rest = detail_outputs                  # full tuple including stats_cards

        # Output order: banner, best-stats-cards, *rest (stats_cards + 10 figs)
        return (banner,) + rest

    # ========================================================================
    # Callback: Validation — Episode Details
    # ========================================================================
    @app.callback(
        Output('val-episode-stats-cards',       'children'),
        Output('val-timeslot-rewards-plot',     'figure'),
        Output('val-timeslot-failures-plot',    'figure'),
        Output('val-action-distribution-plot',  'figure'),
        Output('val-reward-tracking-plot',      'figure'),
        Output('val-inside-system-bikes-plot',  'figure'),
        Output('val-depot-load-plot',           'figure'),
        Output('val-truck-load-plot',           'figure'),
        Output('val-on-road-bikes-plot',        'figure'),
        Output('val-outside-system-bikes-plot', 'figure'),
        Output('val-demand',                    'figure'),
        Input('run-selector',    'value'),
        Input('mode-selector',   'value'),
        Input('episode-selector','value'),
    )
    def update_validation_episode_details(run_path, mode, episode):
        ef = _empty_fig()
        if mode != 'validation':
            return (html.P(''),) + (ef,) * 10
        return _build_episode_detail_outputs(
            Path(run_path) if run_path else None,
            'validation', episode, 'val',
        )

    # ========================================================================
    # Callback: Benchmark tab
    # ========================================================================
    @app.callback(
        Output('bench-stats-cards',        'children'),
        Output('bench-failures-plot',      'figure'),
        Output('bench-rebalance-times-plot','figure'),
        Output('bench-demand-plot',        'figure'),
        Input('run-selector',       'value'),
        Input('mode-selector',      'value'),
        Input('episode-selector',   'value'),
        Input('interval-component', 'n_intervals'),
    )
    def update_benchmark_tab(run_path, mode, episode, n):
        ef = _empty_fig()

        if mode != 'benchmark' or not run_path:
            return html.P(''), ef, ef, ef

        if episode is None:
            return dbc.Alert('Select a benchmark episode.', color='secondary'), ef, ef, ef

        run_dir      = Path(run_path)
        episode_data = load_episode_data(run_dir, 'benchmark', episode)

        if episode_data is None:
            return (
                dbc.Alert('No benchmark data found for this episode.', color='warning'),
                ef, ef, ef,
            )

        scalars         = episode_data.get('scalars', {})
        timeslot_df     = episode_data.get('timeslot_metrics')
        rebalance_events = episode_data.get('rebalance_events', [])  # event durations, not timeslot

        total_failures = scalars.get('total_failures', 0)
        if timeslot_df is not None and 'failures' in timeslot_df.columns and total_failures == 0:
            total_failures = int(timeslot_df['failures'].sum())

        num_days   = len(timeslot_df) // 8 if timeslot_df is not None else 0
        mean_daily = total_failures / num_days if num_days > 0 else 0.0

        total_demand = (
            int(timeslot_df['demand'].sum())
            if timeslot_df is not None and 'demand' in timeslot_df.columns
            else 0
        )

        stats_cards = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{total_failures}", className='text-danger'),
                html.P('Total Failures', className='text-muted mb-0'),
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{mean_daily:.1f}", className='text-warning'),
                html.P('Mean Daily Failures', className='text-muted mb-0'),
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{num_days}", className='text-info'),
                html.P('Days Simulated', className='text-muted mb-0'),
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(f"{total_demand}", className='text-secondary'),
                html.P('Total Demand', className='text-muted mb-0'),
            ])), width=3),
        ])

        # Failures — timeslot line plot
        if timeslot_df is not None and 'failures' in timeslot_df.columns:
            failures_fig = create_timeslot_plot(
                timeslot_df, 'failures',
                f'Benchmark Failures per Timeslot — Total: {total_failures}',
                'Failures'
            )
        else:
            failures_fig = _empty_fig('No failures data')

        # Rebalance times — histogram of event durations (seconds → minutes)
        if timeslot_df is not None and 'rebalance_times' in timeslot_df.columns:
            rebal_fig = create_timeslot_plot(
                timeslot_df=timeslot_df,
                metric='rebalance_times',
                title=f'Benchmark Rebalance Time per Timeslot',
                yaxis_label='Rebalance Duration (min)',
                scale=1/60
            )
        else:
            rebal_fig = _empty_fig('No rebalance event data')

        # Demand — timeslot line plot
        if timeslot_df is not None and 'demand' in timeslot_df.columns:
            demand_fig = create_timeslot_plot(
                timeslot_df, 'demand',
                f'Benchmark Demand per Timeslot — Total: {total_demand}',
                'Bike Demand'
            )
        else:
            demand_fig = _empty_fig('No demand data')

        return stats_cards, failures_fig, rebal_fig, demand_fig

    # ========================================================================
    # Callbacks: Spatial heatmaps (training, validation, best)
    # ========================================================================

    def _heatmap_callback(prefix, mode_value, episode_input):
        """Factory that wires up a heatmap callback for a given prefix."""

        @app.callback(
            Output(f'{prefix}-graph-heatmap-plot',   'src'),
            Output(f'{prefix}-graph-heatmap-status', 'children'),
            Input('run-selector',                    'value'),
            Input('mode-selector',                   'value'),
            Input(episode_input,                     'value'),
            Input(f'{prefix}-graph-metric-selector', 'value'),
            prevent_initial_call=True,
        )
        def _cb(run_path, mode, episode, metric):
            import time
            NO_IMG = ('', '')

            if mode != mode_value or not run_path or not metric:
                return NO_IMG
            if prefix != 'best' and episode is None:
                return NO_IMG

            try:
                run_dir      = Path(run_path)
                ep_to_load   = episode

                # For 'best', ignore episode selector — load from metadata
                if prefix == 'best':
                    meta = load_best_model_metadata(run_dir)
                    if meta is None:
                        return '', '⚠ No best model metadata found'
                    ep_to_load = meta.get('episode')

                episode_data  = load_episode_data(run_dir, mode_value, ep_to_load)
                cell_subgraph = episode_data.get('cell_subgraph') if episode_data else None
                base_graph    = _get_base_graph(app)

                if cell_subgraph is None:
                    return '', f'⚠ No cell_subgraph for episode {ep_to_load}'

                n_boundary = sum(
                    1 for _, d in cell_subgraph.nodes(data=True)
                    if d.get('boundary') is not None
                )
                if n_boundary == 0:
                    return '', '⚠ No "boundary" attr on nodes'

                assets_dir = Path(__file__).parent / 'assets'
                assets_dir.mkdir(exist_ok=True)
                out_path = assets_dir / f'{prefix}_heatmap.png'

                create_graph_heatmap_plot(
                    base_graph=base_graph,
                    cell_subgraph=cell_subgraph,
                    metric=metric,
                    out_path=out_path,
                )

                ts = int(time.time())
                return f'/assets/{prefix}_heatmap.png?t={ts}', f'✓ {metric}, episode {ep_to_load}'

            except Exception as exc:
                import traceback; traceback.print_exc()
                return '', f'✗ {type(exc).__name__}: {exc}'

        return _cb   # keep reference so Dash doesn't GC it

    # Register heatmap callbacks for each prefix/mode combination.
    # 'best' doesn't use the episode-selector (it auto-loads from metadata).
    _train_heatmap_cb = _heatmap_callback('train', 'training',   'episode-selector')
    _val_heatmap_cb   = _heatmap_callback('val',   'validation', 'episode-selector')
    _best_heatmap_cb  = _heatmap_callback('best',  'validation', 'episode-selector')


# ---------------------------------------------------------------------------
# Graph cache helper
# ---------------------------------------------------------------------------

def _get_base_graph(app):
    """Load and cache the base road graph on the app object."""
    if not hasattr(app, 'data_path') or app.data_path is None:
        return None
    if not hasattr(app, '_base_graph_cache'):
        app._base_graph_cache = {}
    key = str(app.data_path)
    if key not in app._base_graph_cache:
        app._base_graph_cache[key] = load_base_graph(app.data_path)
    return app._base_graph_cache[key]