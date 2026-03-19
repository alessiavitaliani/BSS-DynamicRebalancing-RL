"""UI layouts for the results webapp."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_header():
    """Create app header."""
    return dbc.Row([
        dbc.Col([
            html.H1('Train-Val Dashboard', className='text-center mb-4',
                    style={'color': '#4A90E2'})
        ])
    ])


def create_controls():
    """Create control panel."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label('Select Run:', className='fw-bold'),
                    dcc.Dropdown(
                        id='run-selector',
                        clearable=False,
                        placeholder='Select a run...'
                    )
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label('Mode:', className='fw-bold'),
                    dcc.RadioItems(
                        id='mode-selector',
                        options=[
                            {'label': ' Training', 'value': 'training'},
                            {'label': ' Validation', 'value': 'validation'}
                        ],
                        value='training',
                        inline=True,
                        inputStyle={'margin-right': '8px'},  # Space between radio and label
                        labelStyle={'margin-right': '20px'}
                    )
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label('Episode:', className='fw-bold'),
                    dcc.Dropdown(
                        id='episode-selector',
                        clearable=False,
                        placeholder='Select episode...'
                    )
                ])
            ])
        ], width=4),
    ], className='mb-4')


PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'plot',
        'height': 450,
        'width': 800,
        'scale': 3
    }
}
