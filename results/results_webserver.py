import os
import pickle
import numpy as np
import dash
import osmnx as ox
from dash import dcc, html
from dash.dependencies import Input, Output
from utils import load_results, get_episode_options, create_plot, action_bin_labels, generate_osmnx_graph, \
    initialize_graph, compare_failures_across_trainings, compute_mean_failures, easy_3_line_plotter, \
    compute_mean_invalid

# insert here the training runs to evaluate
# phases = [55,541,542,543,544,545,546,557]
phases = [54, 540, 53, 530, 531, 52, 51, 509, 508, 507, 506, 502, 501, 50]
validation_paths = True
tests = [0, 10, 11]
port = 8050
image_height = 900  # Image height in pixels
image_width = 1200
image_scale = 3

# show the total failures during each episode instead of each timeslot
sum_episode_data = True

# Base paths for two training phases
BASE_PATH = "results/"
TRAINING_PATHS = {}
for n in phases:
    label = f"Phase {n} Training"
    folder = f"training_{n}"
    TRAINING_PATHS.update({label: os.path.join(BASE_PATH, folder, "data")})
if validation_paths:
    for n in phases:
        label = f"Phase {n} Validation"
        folder = f"validation_{n}"
        TRAINING_PATHS.update({label: os.path.join(BASE_PATH, folder, "data")})
for n in tests:
    label = f"Test {n}"
    folder = f"test_{n}"
    TRAINING_PATHS.update({label: os.path.join(BASE_PATH, folder, "data")})

# "Training Phase 3": os.path.join(BASE_PATH, "training_3", "data"),
# "Training Phase 3": os.path.join(BASE_PATH, "training_3", "data"),
# "Training Phase 4": os.path.join(BASE_PATH, "training_4", "data"),
# "Validation Phase 2": os.path.join(BASE_PATH, "validation_2", "data"),
# "Validation Phase 3": os.path.join(BASE_PATH, "validation_3", "data"),
# "Validation Phase 4": os.path.join(BASE_PATH, "validation_4", "data"),


# Import external stylesheets (Google Fonts)
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap']

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Initialize usefull vectors
num2days = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}

# Layout with dropdown selector, auto-update, and plots
app.layout = html.Div([
    html.H1("Training Results Dashboard", style={'text-align': 'center', 'color': '#4A90E2'}),

    dcc.Interval(
        id='interval-component',
        interval=5 * 60 * 1000,  # 5 minutes in milliseconds
        n_intervals=0
    ),

    html.Div([

        html.Div([
            # Comparison plot
            html.Div([
                html.Div([
                    html.Button("Update Failures Comparison", id="update-btn-failures-comparison", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),

                dcc.Graph(
                    id="failures-comparison-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'failures_comparison_plot',
                            'height': image_height,
                            'width': image_width,
                            'scale': image_scale
                        }
                    })
            ], style={'width': '100%', 'padding': '10px', 'background-color': 'white',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)'}),

            # Average failure plot
            html.Div([
                html.Div([
                    html.Button("Update Week Failures Average", id="update-btn-failures-average", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),

                dcc.Graph(
                    id="failures-week-average-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'failures_week_average_plot',
                            'height': image_height,
                            'width': image_width,
                            'scale': image_scale
                        }
                    })
            ], style={'width': '100%', 'padding': '10px', 'background-color': 'white',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)'})
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'}),

        # Dropdown for training phase selection
        html.Div([
            html.Label("Select Training Phase:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="training-selector",
                options=[{"label": name, "value": path} for name, path in TRAINING_PATHS.items()],
                value=list(TRAINING_PATHS.values())[0],  # Default to first training phase
                clearable=False,
                style={'width': '100%'}
            ),
        ], style={'margin-bottom': '20px'}),

        html.Div([
            # Reward plot section
            html.Div([
                html.Div([
                    html.Button("Update Reward Plot", id="update-btn-reward", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="reward-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'rewards_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Rewards:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-reward", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Reward tracking plot section
            html.Div([
                html.Div([
                    html.Button("Update Reward Tracking Plot", id="update-btn-reward-tracking", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="reward-tracking-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'reward_tracking_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Action:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id="action-selector-reward-tracking",
                    options=[{'label': label, 'value': label} for label in action_bin_labels],
                    value=action_bin_labels[0],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Failure plot section
            html.Div([
                html.Div([
                    html.Button("Update Failure Plot", id="update-btn-failure", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="failure-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'failures_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Failures:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-failure", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '15px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Q-value plot section
            html.Div([
                html.Div([
                    html.Button("Update Q-Value Plot", id="update-btn-q-value", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="q-value-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'q_values_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Q-Values:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-q-value", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Deployed bikes plot section
            html.Div([
                html.Div([
                    html.Button("Update Deployed Bikes Plot", id="update-btn-bikes", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="deployed-bikes-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'deployed_bikes_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Bikes:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-bikes", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Action plot section
            html.Div([
                html.Div([
                    html.Button("Update Action Plot", id="update-btn-action", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="action-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'actions_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Actions:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-action", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Reward per Action, Epsilon, Loss, invalid actions plots
            html.Div([
                html.Div([
                    html.Button("Update Plots", id="update-btn-plots", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="reward-action-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'reward_per_action_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                dcc.Graph(
                    id="epsilon-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'epsilon_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                dcc.Graph(
                    id="loss-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'loss_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                # dcc.Graph(
                #     id="failures-baseline-plot",
                #     style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                #     config={
                #         'toImageButtonOptions': {
                #             'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                #             'filename': 'failures_baseline_plot',
                #             'height': image_height,   # Image height in pixels
                #             'width': image_width,     # Image width in pixels
                #             'scale': image_scale      # Multiply resolution (e.g., for high-DPI)
                #         }
                #     }),
                dcc.Graph(
                    id="invalid-actions-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'invalid_actions_plot',
                            'height': image_height,  # Image height in pixels
                            'width': image_width,  # Image width in pixels
                            'scale': image_scale  # Multiply resolution (e.g., for high-DPI)
                        }
                    })
            ], style={'width': '100%', 'padding': '10px', 'background-color': 'white',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)'}),

            # Graph section
            html.Div([
                html.Label("Select metric:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-graph", value="last", clearable=False, style={'width': '100%'}),
                dcc.Dropdown(
                    id="metric-selector-graph",
                    options=[{"label": "Visits", "value": "visits"},
                             {"label": "Operations", "value": "operations"},
                             {"label": "Rebalance counter", "value": "rebalanced"},
                             {"label": "Failures", "value": "failures"},
                             {"label": "Critic Score", "value": "critic_score"},
                             {"label": "Num of bikes", "value": "num_bikes"}],
                    value="visits",
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Button("Update OSMnx Graph", id="update-btn-osmnx", n_clicks=0,
                            style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                   'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                html.Img(id="osmnx-graph", style={'width': '60%', 'border': '1px solid #d3d3d3', 'padding': '10px'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',
                      'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'}),
    ], style={'max-width': '1200px', 'margin': '0 auto'}),
], style={'font-family': 'Poppins, sans-serif', 'padding': '20px'})


@app.callback(
    [Output("episode-selector-reward", "options"),
     Output("episode-selector-failure", "options"),
     Output("episode-selector-bikes", "options"),
     Output("episode-selector-q-value", "options"),
     Output("episode-selector-action", "options"),
     Output("episode-selector-graph", "options")],
    [Input("training-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_episode_dropdown(training_path, n_intervals):
    options = get_episode_options(training_path)
    reversed_options = [{"label": "All Episodes", "value": "all"}] + options[0:][::-1]
    reversed_graph_options = [{"label": "Last Episode", "value": "last"}] + options[0:][::-1]
    return [reversed_options] * 5 + [reversed_graph_options]


@app.callback(
    Output("reward-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-reward", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-reward", "value")]
)
def update_reward_plot(n_intervals, n_clicks, training_path, episode_folder):
    rewards = load_results(training_path, episode_folder, metric="rewards_per_timeslot")
    fig = create_plot(rewards, "Rewards per timeslot", "Reward", "Timeslot", cumulative=True)

    # --- Change x-axis ticks from timeslots → days ---
    timeslots_per_day = 56 * 10
    num_timeslots = len(rewards)

    # Choose tick positions at day boundaries
    tick_positions = list(range(0, num_timeslots, timeslots_per_day))
    tick_labels = [str(int(i / timeslots_per_day) * 10) for i in tick_positions]

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            title="Episode"
        )
    )

    return fig


@app.callback(
    Output("failure-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-failure", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-failure", "value")]
)
def update_failure_plot(n_intervals, n_clicks, training_path, episode_folder):
    failures = load_results(training_path, episode_folder, metric="failures_per_timeslot",
                            sum_episode_data=sum_episode_data)
    fig = create_plot(failures, "Failures per timeslot", "Failures", "Timeslot",
                      cumulative=True, failures_plot=True)

    if not sum_episode_data:
        # --- Change x-axis ticks from timeslots → days ---
        timeslots_per_day = 56 * 10
        num_timeslots = len(failures)

        # Choose tick positions at day boundaries
        tick_positions = list(range(0, num_timeslots, timeslots_per_day))
        tick_labels = [str(int(i / timeslots_per_day) * 10) for i in tick_positions]

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=tick_positions,
                ticktext=tick_labels,
                title="Episode"
            )
        )
    else:
        fig.update_layout(
            xaxis=dict(
                title="Episode"
            )
        )

    return fig


@app.callback(
    Output("failures-comparison-plot", "figure"),
    [Input("update-btn-failures-comparison", "n_clicks")]
)
def update_failures_comparison(n_clicks):
    # Manually choose which trainings to compare
    # training_paths = [
    #     "results/training_54/data",
    #     "results/training_551/data",
    #     "results/training_55/data",
    #     "results/training_552/data",
    #     "results/training_553/data"
    # ]
    # labels = ["250 bikes","225 Bikes","200 Bikes","185 Bikes","150 Bikes"]

    training_paths = [
        "results/training_54/data",
        "results/training_530/data",
        "results/training_531/data"
    ]
    labels = ["140 episodes", "100 episodes", "52 episodes"]

    fig = compare_failures_across_trainings(training_paths, labels)

    return fig


@app.callback(
    Output("failures-week-average-plot", "figure"),
    [Input("update-btn-failures-average", "n_clicks")]
)
def update_mean_failures_plot(n_clicks):
    training_paths = [
        # "results/training_553/data",
        "results/training_542/data",
        "results/training_543/data",
        "results/training_544/data",
        "results/training_545/data",
        "results/training_546/data",
        "results/training_557/data",
    ]
    test_paths = [
        "results/test_10/data"
    ]
    bench_path = [
        "results"
    ]
    mean_failures = compute_mean_failures(training_paths)
    mean_failures_notruck = compute_mean_failures(test_paths)
    mean_failures_bench = compute_mean_failures(bench_path, bench=True)

    # fig = create_plot(mean_failures, "Invalid actions", "Average # of Inv. actions", "Episode", cumulative=False, failures_plot=True)
    fig = easy_3_line_plotter(mean_failures, mean_failures_bench)  # , mean_failures_notruck)

    return fig


@app.callback(
    Output("deployed-bikes-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-bikes", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-bikes", "value")]
)
def update_deployed_bikes_plot(n_intervals, n_clicks, training_path, episode_folder):
    deployed_bikes = load_results(training_path, episode_folder, metric="deployed_bikes")
    fig = create_plot(deployed_bikes, "Deployed bikes per timeslot", "Bikes", "Timeslot", cumulative=False)

    # --- Change x-axis ticks from timeslots → episodes ---
    timeslots_per_day = 56 * 10
    num_timeslots = len(deployed_bikes)

    # Choose tick positions at day boundaries
    tick_positions = list(range(0, num_timeslots, timeslots_per_day))
    tick_labels = [str(int(i / timeslots_per_day) * 10) for i in tick_positions]

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            title="Episode"
        )
    )

    return fig


@app.callback(
    Output("q-value-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-q-value", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-q-value", "value")]
)
def update_q_value_plot(n_intervals, n_clicks, training_path, episode_folder):
    q_values_per_timeslot = load_results(training_path, episode_folder, metric="q_values_per_timeslot")

    q_values = []
    for q_value in q_values_per_timeslot:
        q_values.append(np.mean(q_value))

    fig = create_plot(q_values, "Q-Values per timeslot", "Q-Value", "Times", cumulative=True)

    # --- Change x-axis ticks from timeslots → days ---
    timeslots_per_day = 56 * 10
    num_timeslots = len(q_value)

    # Choose tick positions at day boundaries
    tick_positions = list(range(0, num_timeslots, timeslots_per_day))
    tick_labels = [str(int(i / timeslots_per_day) * 10) for i in tick_positions]

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            title="Episode"
        )
    )

    return fig


@app.callback(
    [Output("epsilon-plot", "figure"),
     Output("loss-plot", "figure"),
     Output("reward-action-plot", "figure"),
     Output("invalid-actions-plot", "figure")],
    [Input("interval-component", "n_intervals"),
     Input("training-selector", "value"),
     Input("update-btn-plots", "n_clicks")]
)
def update_epsilon_loss(n_intervals, training_path, n_clicks):
    loss = load_results(training_path, metric="losses")
    try:
        epsilon = load_results(training_path, metric="epsilon_per_timeslot")
        epsilon_plot = create_plot(epsilon, "Epsilon over time", "Epsilon", "Timeslot", cumulative=False)

        # --- Change x-axis ticks from timeslots → days ---
        timeslots_per_day = 56 * 10
        num_timeslots = len(epsilon)

        # Choose tick positions at day boundaries
        tick_positions = list(range(0, num_timeslots, timeslots_per_day))
        tick_labels = [str(int(i / timeslots_per_day) * 10) for i in tick_positions]

        epsilon_plot.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=tick_positions,
                ticktext=tick_labels,
                title="Episode"
            )
        )

    except FileNotFoundError:
        epsilon_plot = None
    total_invalid = load_results(training_path, metric="total_invalid")
    last_episode_folder = get_episode_options(training_path)[-1]
    last_episode_reward_tracking = load_results(training_path, last_episode_folder['value'], metric="reward_tracking")

    actns_len = len(last_episode_reward_tracking)
    reward_per_action = [0] * actns_len
    for action, _ in enumerate(action_bin_labels[:actns_len]):
        reward_per_action[action] = np.mean(last_episode_reward_tracking[action])

    loss_plot = create_plot(loss, "Loss over time", "Loss", "Timeslot", cumulative=False)
    reward_action_plot = create_plot(reward_per_action, "Reward per action", "Reward", "Timeslot", action_plot=True)
    invalid_actions_plot = create_plot(total_invalid, "Invalid actions", "Inv. actions", "Episode", cumulative=False)

    return epsilon_plot, loss_plot, reward_action_plot, invalid_actions_plot


@app.callback(
    Output("action-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-action", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-action", "value")]
)
def update_action_plot(n_intervals, n_clicks, training_path, episode_folder):
    action_per_step = load_results(training_path, episode_folder, metric="action_per_step")

    action_bins = [0] * len(action_bin_labels)
    for action in action_per_step:
        action_bins[action] += 1

    fig = create_plot(action_bins, "Actions per timeslot", "Actions", "Timeslot", action_plot=True)

    # Add an annotation to display the selected timeslot
    fig.add_annotation(
        x=0.99,  # Near the right edge
        y=0.98,  # Slightly above the plot area
        xref="paper",  # Relative to the whole plot area
        yref="paper",
        text=f"Episode: {episode_folder}",  # Display the current timeslot
        showarrow=False,
        font=dict(
            family="Poppins, sans-serif",
            size=12,
            color="black"
        ),
        align="right",
        bgcolor="rgba(255, 255, 255, 1.0)",  # Optional background for better readability
        bordercolor="black",
        borderwidth=1
    )

    return fig


# @app.callback(
#     Output("failures-baseline-plot", "figure"),
#     [Input("interval-component", "n_intervals"),
#      Input("update-btn-plots", "n_clicks")]
# )

# def update_failures_baseline_plot(interval, n_clicks):
#     with open(os.path.join('benchmarks/results/total_failures.pkl'), 'rb') as f:
#         failures = pickle.load(f)
#     fig = create_plot(failures, "Failures", "Failures", "Timeslot", cumulative=True,
#                        failures_plot=True)

#     # --- Map timeslots to weekdays using your dictionary ---
#     timeslots_per_day = 8
#     num_timeslots = len(failures)


#     # Positions for each full day
#     tick_positions = list(range(0, num_timeslots, timeslots_per_day))

#     # Choose starting weekday (0 = monday)
#     start_day_index = 0

#     # Create repeating weekday labels
#     tick_labels = [
#         num2days[(start_day_index + (i // timeslots_per_day)) % 7]
#         for i in tick_positions
#     ]

#     # Update layout
#     fig.update_layout(
#         xaxis=dict(
#             tickmode="array",
#             tickvals=tick_positions,
#             ticktext=tick_labels,
#             title="Day"
#         )
#     )

#     return fig

@app.callback(
    Output("reward-tracking-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-reward-tracking", "n_clicks"),
     Input("training-selector", "value"),
     Input("action-selector-reward-tracking", "value")]
)
def update_reward_tracking_plot(n_intervals, n_clicks, training_path, action):
    reward_tracking = load_results(training_path, "all", metric="reward_tracking")
    action_value = action_bin_labels.index(action)
    rewards = reward_tracking[action_value]
    return create_plot(rewards, "Rewards per Step", "Reward", "Step", cumulative=True)


@app.callback(
    Output("osmnx-graph", "src"),
    [Input("training-selector", "value"),
     Input("metric-selector-graph", "value"),
     Input("episode-selector-graph", "value"),
     Input("update-btn-osmnx", "n_clicks")]
)
def update_osmnx_graph(training_path, metric, episode_folder, n_clicks):
    graph = initialize_graph('data/utils/cambridge_network.graphml')
    # Initialize the cells
    with open('data/utils/cell_data.pkl', 'rb') as file:
        cells = pickle.load(file)

    # episode_folder = get_episode_options(training_path)[-1]
    cell_subgraph = load_results(training_path, episode_folder, metric="cell_subgraph")
    nodes = ox.graph_to_gdfs(cell_subgraph, nodes=True, edges=False)

    cell_values = {}
    total_visits = 0
    total_ops = 0
    for _, node in nodes.iterrows():
        cell_values[node['cell_id']] = node[metric]
        total_visits += node['visits']
        # total_ops += node['operations']

    if metric == "visits":
        for cell_id in cell_values.keys():
            cell_values[cell_id] = cell_values[cell_id] / total_visits
    # elif metric == "operations":
    #     for cell_id in cell_values.keys():
    #         cell_values[cell_id] = cell_values[cell_id] / total_ops

    return generate_osmnx_graph(graph, cells,
                                cell_values)  # , percentage=(metric == "visits" or metric == "critic_score"))


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port)