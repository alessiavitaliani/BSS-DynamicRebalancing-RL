"""
BSS Results WebApp

Web dashboard for visualizing and monitoring BSS Dynamic Rebalancing RL training results.
"""

__version__ = "1.0.0"
__author__ = "Edoardo Scarpel"

from results_webapp.data_loader import (
    discover_runs,
    load_run_config,
    load_summary_data,
    load_episode_data,
    get_available_episodes,
    build_summary_from_episodes,
    load_bench_data,
    load_base_graph
)

from results_webapp.plotting import (
    create_metric_plot,
    create_timeslot_plot,
    create_action_distribution_plot,
    create_comparison_plot,
    create_reward_tracking_plot,
)

__all__ = [
    # Data loading
    "discover_runs",
    "load_run_config",
    "load_summary_data",
    "load_episode_data",
    "get_available_episodes",
    "build_summary_from_episodes",
    "load_bench_data",
    "load_base_graph",
    # Plotting
    "create_metric_plot",
    "create_timeslot_plot",
    "create_action_distribution_plot",
    "create_comparison_plot",
    "create_reward_tracking_plot",
]
