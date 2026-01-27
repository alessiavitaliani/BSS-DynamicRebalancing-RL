"""Data loading utilities for the results webapp."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


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
            'mean_failures': scalars.get('mean_failures', 0.0),
            'total_failures': scalars.get('total_failures', 0),
            'total_trips': scalars.get('total_trips', 0),
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
                episodes.append(ep_num)
            except (IndexError, ValueError):
                continue

    return sorted(episodes)


def load_concatenated_timeslot_data(run_dir: Path, mode: str, metric: str) -> pd.Series:
    """
    Load and concatenate timeslot data across all episodes.

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'
        metric: Name of metric column to extract

    Returns:
        Series with concatenated timeslot data
    """
    episodes = get_available_episodes(run_dir, mode)
    all_data = []

    for episode in episodes:
        episode_data = load_episode_data(run_dir, mode, episode)
        if episode_data and 'timeslot_metrics' in episode_data:
            timeslot_df = episode_data['timeslot_metrics']
            if metric in timeslot_df.columns:
                all_data.extend(timeslot_df[metric].tolist())

    return pd.Series(all_data)


def load_concatenated_step_data(run_dir: Path, mode: str, metric: str) -> pd.Series:
    """
    Load and concatenate step-level data across all episodes.

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'
        metric: Name of metric in step_data dict

    Returns:
        Series with concatenated step data
    """
    episodes = get_available_episodes(run_dir, mode)
    all_data = []

    for episode in episodes:
        episode_data = load_episode_data(run_dir, mode, episode)
        if episode_data and 'step_data' in episode_data:
            step_data = episode_data['step_data']
            if metric in step_data:
                data = step_data[metric]
                # Handle different data types
                if metric == 'q_values':
                    # Q-values are per timeslot, compute mean
                    all_data.extend([np.mean(q) if len(q) > 0 else 0 for q in data])
                elif metric == 'losses':
                    # Filter out None values
                    all_data.extend([l for l in data if l is not None])
                else:
                    all_data.extend(data)

    return pd.Series(all_data)
