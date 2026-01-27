"""Main Dash application."""

import argparse
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from results_webapp.callbacks import register_callbacks
from results_webapp.layouts import create_header, create_controls, PLOT_CONFIG


def create_app(results_path: str, port: int = 8050, update_interval_ms: int = 5000):
    """
    Create and configure the Dash app.

    Args:
        results_path: Path to results directory
        port: Port to run server on
        update_interval_ms: Auto-refresh interval in milliseconds

    Returns:
        Configured Dash app
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )

    app.title = "BSS Training Dashboard"

    # Store config in app
    app.results_path = Path(results_path)
    app.port = port
    app.update_interval_ms = update_interval_ms

    # Layout
    app.layout = dbc.Container([
        dcc.Store(id='run-data-store'),

        create_header(),

        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=update_interval_ms,
            n_intervals=0
        ),

        create_controls(),

        # Config display
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Run Configuration'),
                    dbc.CardBody(id='config-display')
                ])
            ])
        ], className='mb-4'),

        # Tabs for different views
        dbc.Tabs([
            # Overview Tab
            dbc.Tab(label='📊 Overview', children=[
                # Row 1: Failures and Rewards
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='failures-plot', config=PLOT_CONFIG)
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='rewards-plot', config=PLOT_CONFIG)
                    ], width=6),
                ], className='mt-3'),

                # Row 2: Epsilon and Deployed Bikes
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='epsilon-plot', config=PLOT_CONFIG)
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='deployed-bikes-overview-plot', config=PLOT_CONFIG)
                    ], width=6),
                ], className='mt-3'),

                # Row 3: Q-values and Global Critic (only for training)
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='qvalues-overview-plot', config=PLOT_CONFIG)
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='global-critic-overview-plot', config=PLOT_CONFIG)
                    ], width=6),
                ], className='mt-3'),

                # Row 4: Training Loss (only for training mode)
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='loss-overview-plot', config=PLOT_CONFIG)
                    ], width=12),
                ], className='mt-3'),
            ]),

            # Episode Details Tab
            dbc.Tab(label='🔍 Episode Details', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div(id='episode-stats-cards')
                    ])
                ], className='mt-3'),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='timeslot-rewards-plot', config=PLOT_CONFIG)
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='timeslot-failures-plot', config=PLOT_CONFIG)
                    ], width=6),
                ], className='mt-3'),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='action-distribution-plot', config=PLOT_CONFIG)
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='reward-tracking-plot', config=PLOT_CONFIG)
                    ], width=6),
                ], className='mt-3'),
            ]),
        ])

    ], fluid=True, style={'padding': '20px'})

    # Register callbacks
    register_callbacks(app)

    return app


def main():
    """Main entry point for the webapp."""
    parser = argparse.ArgumentParser(description='BSS Results WebApp')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Path to results directory')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run server on')
    parser.add_argument('--update-interval', type=int, default=5000,
                        help='Auto-refresh interval in milliseconds')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    args = parser.parse_args()

    app = create_app(
        results_path=args.results_path,
        port=args.port,
        update_interval_ms=args.update_interval
    )

    print(f"Starting BSS Results WebApp on http://0.0.0.0:{args.port}")
    print(f"Results path: {args.results_path}")

    app.run(debug=args.debug, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
