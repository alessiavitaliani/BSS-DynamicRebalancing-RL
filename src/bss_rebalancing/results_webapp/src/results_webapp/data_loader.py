"""Data loading utilities for the results webapp."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import osmnx as ox
import pandas as pd


def _add_inside_system_bikes(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the derived 'inside_system_bikes' column (deployed + truck + depot)
    to a timeslot_metrics DataFrame, if the source columns are present.
    Shared helper so combined AND per-area DataFrames get it consistently —
    previously only the combined one did, so the "Total bikes inside system"
    plot was blank whenever a single area was selected."""
    required_cols = ['deployed_bikes', 'truck_load', 'depot_load']
    if all(col in df.columns for col in required_cols):
        df['inside_system_bikes'] = (
            df['deployed_bikes'] + df['truck_load'] + df['depot_load']
        )
    return df


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


def load_run_config(run_dir: Path, mode: str) -> Optional[Dict]:
    """
    Load hyperparameters and config for a run.

    For validation, each episode folder contains its own config.json — we load
    the first available one.  For training/benchmark the old paths still apply.

    Args:
        run_dir: Path to run directory
        mode: 'training', 'validation', or 'benchmark'

    Returns:
        Dictionary containing run configuration or None if not found
    """
    if mode == 'validation':
        val_dir = run_dir / 'validation'
        if val_dir.exists():
            for ep_dir in sorted(val_dir.iterdir()):
                if ep_dir.is_dir() and ep_dir.name.startswith('episode_'):
                    cfg = ep_dir / 'config.json'
                    if cfg.exists():
                        with open(cfg, 'r') as f:
                            return json.load(f)
        return None

    config_path = run_dir / mode / 'config.json'
    old_path = run_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    elif old_path.exists():
        with open(old_path, 'r') as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _val_inner_dir(val_ep_dir: Path) -> Path:
    """
    Return the inner episode directory for a validation episode.

    Structure: validation/episode_NNN/episode_000/
    We always use episode_000 (the primary seed run).
    """
    return val_ep_dir / 'episode_000'


def get_available_episodes(run_dir: Path, mode: str) -> List[int]:
    """
    Get list of available episode numbers for a run.

    For validation the structure is:
        validation/episode_NNN/episode_000/scalars.json

    Args:
        run_dir: Path to run directory
        mode: 'training', 'validation', or 'benchmark'

    Returns:
        Sorted list of available episode numbers
    """
    if mode == 'benchmark':
        # Structure: benchmark/episode_NNN/scalars.json
        bench_dir = run_dir / 'benchmark'
        if not bench_dir.exists():
            return []
        episodes = []
        for ep_dir in bench_dir.iterdir():
            if not ep_dir.is_dir() or not ep_dir.name.startswith('episode_'):
                continue
            try:
                ep_num = int(ep_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            if (ep_dir / 'scalars.json').exists():
                episodes.append(ep_num)
        return sorted(episodes)

    if mode == 'validation':
        val_dir = run_dir / 'validation'
        if not val_dir.exists():
            return []

        episodes = []
        for ep_dir in val_dir.iterdir():
            if not ep_dir.is_dir() or not ep_dir.name.startswith('episode_'):
                continue
            try:
                ep_num = int(ep_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            # Only include if the inner episode_000/scalars.json is present
            if (_val_inner_dir(ep_dir) / 'scalars.json').exists():
                episodes.append(ep_num)
        return sorted(episodes)

    # --- training ---
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
            if (ep_dir / 'scalars.json').exists():
                episodes.append(ep_num)

    return sorted(episodes)


def build_summary_from_episodes(run_dir: Path, mode: str, area: str | None = None) -> Optional[pd.DataFrame]:
    """
    Build summary DataFrame by aggregating data from individual episode folders.

    Works for training and validation (using the corrected inner paths).

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'
        area: None/'__all__' for the combined run; an area label (e.g.
            'area_0') to build the summary from that area's own
            per_area/<area>/scalars.json files instead (see
            rl_training/results/results_manager.py's save_episode()). Falls
            back silently to the combined scalars.json if an episode has no
            per-area breakdown (e.g. single-area runs, or older episodes).

    Returns:
        DataFrame with episode-level summary data or None if no episodes found
    """
    use_area = bool(area) and area != '__all__'

    if mode == 'validation':
        val_dir = run_dir / 'validation'
        if not val_dir.exists():
            return None

        episodes_data = []
        for ep_dir in sorted(val_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith('episode_'):
                continue
            try:
                episode_num = int(ep_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue

            inner_dir = _val_inner_dir(ep_dir)
            scalars_path = inner_dir / 'scalars.json'
            if use_area:
                area_scalars_path = inner_dir / 'per_area' / area / 'scalars.json'
                if area_scalars_path.exists():
                    scalars_path = area_scalars_path
            if not scalars_path.exists():
                continue

            with open(scalars_path, 'r') as f:
                scalars = json.load(f)

            episodes_data.append({
                'episode': episode_num,
                'total_reward': scalars.get('total_reward', 0.0),
                'mean_daily_failures': scalars.get('mean_daily_failures', 0.0),
                'total_failures': scalars.get('total_failures', 0),
                'total_invalid_actions': scalars.get('total_invalid_actions', 0),
                #'epsilon': scalars.get('epsilon', 0.0),
                'policy_loss': scalars.get('policy_loss', 0.0),   
                'value_loss': scalars.get('value_loss', 0.0),     
                'entropy': scalars.get('entropy', 0.0),
            })

        if not episodes_data:
            return None

        df = pd.DataFrame(episodes_data)
        return df.sort_values('episode').reset_index(drop=True)

    # --- training ---
    mode_dir = run_dir / mode
    if not mode_dir.exists():
        return None

    episodes_data = []
    for ep_dir in sorted(mode_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith('episode_'):
            continue
        try:
            episode_num = int(ep_dir.name.split('_')[1])
        except (IndexError, ValueError):
            continue

        scalars_path = ep_dir / 'scalars.json'
        if use_area:
            area_scalars_path = ep_dir / 'per_area' / area / 'scalars.json'
            if area_scalars_path.exists():
                scalars_path = area_scalars_path
        if not scalars_path.exists():
            continue

        with open(scalars_path, 'r') as f:
            scalars = json.load(f)

        episodes_data.append({
            'episode': episode_num,
            'total_reward': scalars.get('total_reward', 0.0),
            'mean_daily_failures': scalars.get('mean_daily_failures', 0.0),
            'total_failures': scalars.get('total_failures', 0),
            'total_invalid': scalars.get('total_invalid', 0),
            #'epsilon': scalars.get('epsilon', 0.0),
            # policy_loss/value_loss/entropy are properties of the ONE shared
            # PPO update (same buffer across areas), so they're always
            # identical regardless of area — read from the combined file even
            # when `area` is set (per-area scalars.json doesn't have them).
            'policy_loss': scalars.get('policy_loss', 0.0) if not use_area else None,
            'value_loss': scalars.get('value_loss', 0.0) if not use_area else None,
            'entropy': scalars.get('entropy', 0.0) if not use_area else None,
        })

    if not episodes_data:
        return None

    df = pd.DataFrame(episodes_data)

    if use_area:
        # Backfill policy_loss/value_loss/entropy from the combined
        # scalars.json per episode (see comment above).
        combined_map = {}
        for ep_dir in sorted(mode_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith('episode_'):
                continue
            try:
                episode_num = int(ep_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            combined_path = ep_dir / 'scalars.json'
            if combined_path.exists():
                with open(combined_path, 'r') as f:
                    combined_map[episode_num] = json.load(f)
        df['policy_loss'] = df['episode'].map(lambda e: combined_map.get(e, {}).get('policy_loss', 0.0))
        df['value_loss'] = df['episode'].map(lambda e: combined_map.get(e, {}).get('value_loss', 0.0))
        df['entropy'] = df['episode'].map(lambda e: combined_map.get(e, {}).get('entropy', 0.0))

    return df.sort_values('episode').reset_index(drop=True)


def load_summary_data(run_dir: Path, mode: str = 'training', area: str | None = None) -> Optional[pd.DataFrame]:
    """
    Load aggregated summary CSV for training or validation.
    If CSV doesn't exist, builds it from individual episode folders.

    For validation the summary CSV lives at validation/episode_NNN/validation_summary.csv
    but since episodes are independent runs we just build from scalars.

    Args:
        run_dir: Path to run directory
        mode: 'training' or 'validation'
        area: None/'__all__' for the combined run; an area label (e.g.
            'area_0') to build the per-area summary instead. Always rebuilt
            from per-episode scalars.json in this case — the aggregated
            {mode}_summary.csv shortcut only has the combined series.

    Returns:
        DataFrame with episode-level summary data or None if not found
    """
    if mode == 'training' and not (area and area != '__all__'):
        summary_path = run_dir / mode / f'{mode}_summary.csv'
        if summary_path.exists():
            return pd.read_csv(summary_path)

    return build_summary_from_episodes(run_dir, mode, area=area)


def load_episode_data(run_dir: Path, mode: str, episode: int) -> Optional[Dict]:
    """
    Load detailed data for a specific episode.

    For validation: path is validation/episode_NNN/episode_000/
    For training:   path is training/episode_NNN/
    For benchmark:  episode is ignored; reads from benchmark/ directly

    Args:
        run_dir: Path to run directory
        mode: 'training', 'validation', or 'benchmark'
        episode: Episode number (ignored for benchmark)

    Returns:
        Dictionary containing episode data or None if not found
    """
    if mode == 'benchmark':
        return _load_benchmark_episode(run_dir, episode)

    if mode == 'validation':
        episode_dir = _val_inner_dir(run_dir / 'validation' / f'episode_{episode:03d}')
    else:
        episode_dir = run_dir / mode / f'episode_{episode:03d}'

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
        data['timeslot_metrics'] = _add_inside_system_bikes(pd.read_csv(timeslot_path))

    # Load step data (pickle)
    step_data_path = episode_dir / 'step_data.pkl.gz'
    if step_data_path.exists():
        with open(step_data_path, 'rb') as f:
            data['step_data'] = pickle.load(f)

    # Load cell subgraph
    subgraph_path = episode_dir / 'cell_subgraph.gpickle'
    if subgraph_path.exists():
        with open(subgraph_path, 'rb') as f:
            data['cell_subgraph'] = pickle.load(f)
    else:
        data['cell_subgraph'] = None

    # Load per-area breakdown (multi-area training/validation only — see
    # train_ppo_multi_env / validate_ppo_multi_env in rl_training). Always set
    # the key (empty dict if absent) so callers don't need hasattr checks.
    per_area_dir = episode_dir / 'per_area'
    data['per_area'] = {}
    if per_area_dir.exists():
        for area_dir in sorted(per_area_dir.iterdir()):
            if not area_dir.is_dir():
                continue
            area_entry: Dict = {}
            area_ts_path = area_dir / 'timeslot_metrics.csv'
            if area_ts_path.exists():
                area_entry['timeslot_metrics'] = _add_inside_system_bikes(pd.read_csv(area_ts_path))
            area_scalars_path = area_dir / 'scalars.json'
            if area_scalars_path.exists():
                with open(area_scalars_path, 'r') as f:
                    area_entry['scalars'] = json.load(f)
            area_sg_path = area_dir / 'cell_subgraph.gpickle'
            if area_sg_path.exists():
                with open(area_sg_path, 'rb') as f:
                    area_entry['cell_subgraph'] = pickle.load(f)
            else:
                area_entry['cell_subgraph'] = None
            data['per_area'][area_dir.name] = area_entry

    return data


def _load_benchmark_episode(run_dir: Path, episode: int) -> Optional[Dict]:
    """
    Load a single benchmark episode.

    Structure: benchmark/episode_NNN/
        scalars.json
        timeslot_metrics.csv   — per-timeslot columns (failures, deployed_bikes, …)
                                  also contains 'rebalance_times' column BUT that
                                  column stores event durations padded to timeslot
                                  length — for display we keep it as a raw list via
                                  'rebalance_events' so the callback can histogram it.
        cell_subgraph.gpickle  — optional

    Note: benchmark has no step_data (no actions, no reward tracking, no Q-values).
    """
    episode_dir = run_dir / 'benchmark' / f'episode_{episode:03d}'
    if not episode_dir.exists():
        return None

    data: Dict = {}

    scalars_path = episode_dir / 'scalars.json'
    if scalars_path.exists():
        with open(scalars_path, 'r') as f:
            data['scalars'] = json.load(f)

    timeslot_path = episode_dir / 'timeslot_metrics.csv'
    if timeslot_path.exists():
        data['timeslot_metrics'] = _add_inside_system_bikes(pd.read_csv(timeslot_path))

    subgraph_path = episode_dir / 'cell_subgraph.gpickle'
    if subgraph_path.exists():
        with open(subgraph_path, 'rb') as f:
            data['cell_subgraph'] = pickle.load(f)
    else:
        data['cell_subgraph'] = None

    # Load per-area breakdown (multi-area benchmark only — see
    # run_benchmark_multi_env in benchmark/run.py).
    per_area_dir = episode_dir / 'per_area'
    data['per_area'] = {}
    if per_area_dir.exists():
        for area_dir in sorted(per_area_dir.iterdir()):
            if not area_dir.is_dir():
                continue
            area_entry: Dict = {}
            area_ts_path = area_dir / 'timeslot_metrics.csv'
            if area_ts_path.exists():
                area_entry['timeslot_metrics'] = _add_inside_system_bikes(pd.read_csv(area_ts_path))
            area_scalars_path = area_dir / 'scalars.json'
            if area_scalars_path.exists():
                with open(area_scalars_path, 'r') as f:
                    area_entry['scalars'] = json.load(f)
            area_sg_path = area_dir / 'cell_subgraph.gpickle'
            if area_sg_path.exists():
                with open(area_sg_path, 'rb') as f:
                    area_entry['cell_subgraph'] = pickle.load(f)
            else:
                area_entry['cell_subgraph'] = None
            data['per_area'][area_dir.name] = area_entry

    return data if data else None


# ---------------------------------------------------------------------------
# Best model metadata
# ---------------------------------------------------------------------------

def load_best_model_metadata(run_dir: Path) -> Optional[Dict]:
    """
    Load metadata for the best saved model.

    Path: models/best/metadata.json
    Relevant key: 'episode' — the validation episode number that corresponds
    to this best model.

    Returns:
        Dict with at least {'episode': int, 'score': float} or None
    """
    metadata_path = run_dir / 'models' / 'best' / 'metadata.json'
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_base_graph(data_path: Path) -> Optional[nx.MultiDiGraph]:
    """
    Load the full Cambridge road network graph.

    Args:
        data_path: Path to the data directory (parent of 'utils/')

    Returns:
        NetworkX MultiDiGraph or None if not found
    """
    graph_path = data_path / 'utils' / 'network.graphml'
    if not graph_path.exists():
        return None
    return ox.load_graphml(graph_path)


def get_area_data_paths(run_dir: Path) -> Optional[Dict[str, Path]]:
    """
    For a multi-area run (trained with train.py --data-paths, or benchmarked
    with benchmark/run.py --data-paths), returns a mapping from area label
    ('area_0', 'area_1', ...) to that area's data folder Path, read from
    whichever config was saved by save_hyperparameters() (training config
    takes precedence, then validation, then benchmark — a run can have any
    subset of these). Needed to load the correct base road-network map per
    area — the areas are different parts of Manhattan, so overlaying one
    area's cell subgraph on another area's base map would be spatially wrong.

    Returns None for single-area runs (or if the config predates this field).
    """
    for mode in ('training', 'validation', 'benchmark'):
        config = load_run_config(run_dir, mode)
        if not config:
            continue
        data_paths = (config.get('hyperparameters') or {}).get('data_paths')
        if data_paths:
            return {f"area_{i}": Path(p) for i, p in enumerate(data_paths)}
    return None


# ---------------------------------------------------------------------------
# Legacy benchmark loader (kept for any external callers)
# ---------------------------------------------------------------------------

def load_bench_data(
        run_dir: Path,
        mode: str = 'benchmark'
) -> Optional[tuple]:
    """
    Load benchmark results saved by ResultsManager.save_episode().

    Returns (total_failures, dfs, subgraph) or None.
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
                    'outside_system_bikes', 'traveling_bikes', 'rebalance_times',
                    'demand']:
            if col in timeslot_df.columns:
                dfs[col] = timeslot_df[['timeslot', col]].copy()

    subgraph_path = bench_dir / 'cell_subgraph.gpickle'
    subgraph = None
    if subgraph_path.exists():
        with open(subgraph_path, 'rb') as f:
            subgraph = pickle.load(f)

    return total_failures, dfs, subgraph