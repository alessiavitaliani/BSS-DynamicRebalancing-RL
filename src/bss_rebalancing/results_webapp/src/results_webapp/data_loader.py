"""Data loading utilities for the results webapp."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import networkx as nx
import osmnx as ox
import pandas as pd
from pandas import DataFrame


def discover_runs(base_path: Path) -> Dict[str, Path]:
    """
    Discover all available runs in the results directory.

    Args:
        base_path: Base results directory path

    Returns:
        Dictionary mapping run labels to run directories
    """
    if not base_path.exists():
        return {}

    runs = {}
    for run_dir in sorted(base_path.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            run_id = run_dir.name.split('_')[1]
            runs[f"Run {run_id}"] = run_dir

    return runs


def load_run_config(run_dir: Path) -> Optional[Dict]:
    """
    Load hyperparameters and config for a run.

    Args:
        run_dir: Path to run directory

    Returns:
        Dictionary containing run configuration or None if not found
    """
    config_path = run_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def build_summary_from_episodes(run_dir: Path, mode: str) -> Optional[pd.DataFrame]:
    """
    Build summary DataFrame by aggregating data from individual episode folders.
    This is used when the summary CSV doesn't exist yet (during training).

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'

    Returns:
        DataFrame with episode-level summary data or None if no episodes found
    """
    mode_dir = run_dir / mode
    if not mode_dir.exists():
        return None

    episodes_data = []

    # Iterate through episode folders
    for ep_dir in sorted(mode_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith('episode_'):
            continue

        try:
            episode_num = int(ep_dir.name.split('_')[1])
        except (IndexError, ValueError):
            continue

        # Load scalars.json for this episode
        scalars_path = ep_dir / 'scalars.json'
        if not scalars_path.exists():
            continue

        with open(scalars_path, 'r') as f:
            scalars = json.load(f)

        # Add to list
        episodes_data.append({
            'episode': episode_num,
            'total_reward': scalars.get('total_reward', 0.0),
            'mean_daily_failures': scalars.get('mean_daily_failures', 0.0),
            'total_failures': scalars.get('total_failures', 0),
            'total_invalid': scalars.get('total_invalid', 0),
            'epsilon': scalars.get('epsilon', 0.0),
        })

    if not episodes_data:
        return None

    # Create DataFrame and sort by episode
    df = pd.DataFrame(episodes_data)
    df = df.sort_values('episode').reset_index(drop=True)

    return df


def load_summary_data(run_dir: Path, mode: str = 'training') -> Optional[pd.DataFrame]:
    """
    Load aggregated summary CSV for training or validation.
    If CSV doesn't exist, builds it from individual episode folders.

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'

    Returns:
        DataFrame with episode-level summary data or None if not found
    """
    summary_path = run_dir / mode / f'{mode}_summary.csv'

    # First, try to load the CSV (fast)
    if summary_path.exists():
        return pd.read_csv(summary_path)

    # If CSV doesn't exist, build from episodes (slower but works during training)
    return build_summary_from_episodes(run_dir, mode)


def load_episode_data(run_dir: Path, mode: str, episode: int) -> Optional[Dict]:
    """
    Load detailed data for a specific episode.

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'
        episode: Episode number

    Returns:
        Dictionary containing episode data or None if not found
    """
    episode_dir = run_dir / mode / f"episode_{episode:03d}"

    if not episode_dir.exists():
        return None

    data = {}

    # Load scalars (JSON)
    scalars_path = episode_dir / 'scalars.json'
    if scalars_path.exists():
        with open(scalars_path, 'r') as f:
            data['scalars'] = json.load(f)

    # Load timeslot metrics (CSV)
    timeslot_path = episode_dir / 'timeslot_metrics.csv'
    if timeslot_path.exists():
        data['timeslot_metrics'] = pd.read_csv(timeslot_path)
        required_cols = ['deployed_bikes', 'truck_load', 'depot_load']
        if all(col in data['timeslot_metrics'].columns for col in required_cols):
            data['timeslot_metrics']['inside_system_bikes'] = (
                    data['timeslot_metrics']['deployed_bikes'] +
                    data['timeslot_metrics']['truck_load'] +
                    data['timeslot_metrics']['depot_load']
            )

    # Load step data (pickle)
    step_data_path = episode_dir / 'step_data.pkl.gz'
    if step_data_path.exists():
        with open(step_data_path, 'rb') as f:
            data['step_data'] = pickle.load(f)

    # Load cell subgraph (if needed)
    subgraph_path = episode_dir / 'cell_subgraph.gpickle'
    if subgraph_path.exists():
        with open(episode_dir / 'cell_subgraph.gpickle', 'rb') as f:
            data['cell_subgraph'] = pickle.load(f)
    else:
        data['cell_subgraph'] = None

    return data


def get_available_episodes(run_dir: Path, mode: str) -> List[int]:
    """
    Get list of available episode numbers for a run.

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'

    Returns:
        Sorted list of available episode numbers
    """
    mode_dir = run_dir / mode
    if not mode_dir.exists():
        return []

    episodes = []
    for ep_dir in mode_dir.iterdir():
        if ep_dir.is_dir() and ep_dir.name.startswith('episode_'):
            try:
                ep_num = int(ep_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            # Only include episodes that have completed — scalars.json is
            # written at save time, so its absence means the episode is
            # still in progress (only the log folder exists so far)
            if (ep_dir / 'scalars.json').exists():
                episodes.append(ep_num)

    return sorted(episodes)


def load_base_graph(data_path: Path) -> Optional[nx.MultiDiGraph]:
    """
    Load the full Cambridge road network graph.

    Args:
        data_path: Path to the data directory (parent of 'utils/')

    Returns:
        NetworkX MultiDiGraph or None if not found
    """
    graph_path = data_path / 'utils' / 'cambridge_network.graphml'
    if not graph_path.exists():
        return None
    return ox.load_graphml(graph_path)


def load_bench_data(
        run_dir: Path,
        mode: str = 'benchmark'
) -> tuple[int | Any, dict[str, DataFrame], Any | None] | None:
    """
    Load benchmark results saved by ResultsManager.save_episode().

    ResultsManager writes:
      bench_path / scalars.json          — total_failures, mean_daily_failures
      bench_path / timeslot_metrics.csv  — per-timeslot failures, truck_load, …
    """
    bench_dir = run_dir / mode

    scalars_path = bench_dir / 'scalars.json'
    timeslot_path = bench_dir / 'timeslot_metrics.csv'

    if not scalars_path.exists() and not timeslot_path.exists():
        return None

    dfs: dict[str, pd.DataFrame] = {}
    total_failures = 0

    if scalars_path.exists():
        with open(scalars_path, 'r') as f:
            scalars = json.load(f)
        total_failures = scalars.get('total_failures', 0)

    if timeslot_path.exists():
        timeslot_df = pd.read_csv(timeslot_path)

        if 'failures' in timeslot_df.columns:
            dfs['failures'] = timeslot_df[['timeslot', 'failures']].copy()
            if total_failures == 0:
                total_failures = int(timeslot_df['failures'].sum())

        for col in ['truck_load', 'depot_load', 'deployed_bikes',
                    'outside_system_bikes', 'traveling_bikes', 'rebalance_times']:
            if col in timeslot_df.columns:
                dfs[col] = timeslot_df[['timeslot', col]].copy()

    subgraph_path = bench_dir / 'cell_subgraph.gpickle'
    subgraph = None
    if subgraph_path.exists():
        with open(subgraph_path, 'rb') as f:
            subgraph = pickle.load(f)

    return total_failures, dfs, subgraph