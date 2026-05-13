import json
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to native Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class EpisodeResults:
    """Container for all metrics from a single benchmark episode."""

    episode: int
    mode: str  # 'benchmark'
    seed: int

    # Episode-level scalars
    total_reward: float = 0.0
    mean_daily_failures: float = 0.0
    total_failures: int = 0

    # Time-series data (per timeslot)
    failures_per_timeslot: List[int] = field(default_factory=list)
    deployed_bikes: List[int] = field(default_factory=list)
    truck_load: List[int] = field(default_factory=list)
    depot_load: List[int] = field(default_factory=list)
    outside_system_bikes: List[int] = field(default_factory=list)
    traveling_bikes: List[int] = field(default_factory=list)
    rebalance_times: List[int] = field(default_factory=list)
    demand_per_timeslot: List[float] = field(default_factory=list)
    global_critic_scores: List[float] = field(default_factory=list)

    # Spatial data (cell subgraph)
    cell_subgraph: Optional[nx.Graph] = None


class ResultsManager:
    """Centralized manager for benchmark results."""

    def __init__(
        self,
        results_path: str,
        run_id: int,
        overwrite: bool = False,
        interactive: bool = True,
    ):
        self.results_path = Path(results_path)
        self.run_id = run_id
        self.run_dir = self.results_path / f"run_{run_id:03d}"
        self.bench_path = self.run_dir / "benchmark"

        self._owned_dirs = [self.bench_path]

        # Check if results already exist — only care about the benchmark subdir,
        # never touch sibling dirs (training/, validation/, models/).
        if self.bench_path.exists() and not overwrite:
            if interactive:
                self._handle_existing_results()
            else:
                raise Exception(
                    f"Benchmark results already exist: {self.bench_path}. "
                    f"Use overwrite=True or change run_id."
                )

        # Create only the benchmark directory
        self.bench_path.mkdir(parents=True, exist_ok=True)

        # Initialize aggregated DataFrame
        self.bench_summary = pd.DataFrame()

    def _handle_existing_results(self):
        """
        Handle case where the benchmark directory already exists.

        Only the 'benchmark' subfolder is removed — training/, validation/,
        and models/ are never touched.
        """
        print(f"⚠️  WARNING: Benchmark results already exist for run {self.run_id}.")
        print(f"   Path: {self.bench_path}")
        print("   Only this directory will be deleted and recreated.")
        print("   Training, validation and model directories are NOT affected.")

        try:
            proceed = input("Are you sure you want to continue? (y/n) ").strip().lower()
        except (ValueError, EOFError):
            proceed = ""
            print("Invalid input! Please enter 'y' or 'n'.")

        if proceed in ["y", "yes"]:
            print(f"Removing {self.bench_path}...")
            shutil.rmtree(self.bench_path)
        else:
            raise Exception(
                f"Aborted: Change the 'run_id' (current: {self.run_id}) "
                f"or the 'results_path' (current: {self.results_path})."
            )

    @classmethod
    def create_with_auto_increment(cls, results_path: str) -> "ResultsManager":
        """Create a ResultsManager with auto-incremented run_id to avoid conflicts."""
        results_path_obj = Path(results_path)
        results_path_obj.mkdir(parents=True, exist_ok=True)

        existing_runs = [
            int(d.name.split("_")[1])
            for d in results_path_obj.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        next_run_id = max(existing_runs, default=-1) + 1

        print(f"Auto-assigned run_id: {next_run_id}")
        return cls(results_path, next_run_id, overwrite=False)

    def save_episode(self, results: EpisodeResults):
        """Save results for a single benchmark episode."""
        if results.mode != "benchmark":
            raise ValueError(
                f"Expected mode='benchmark', got mode='{results.mode}'. "
                "Use the train/val ResultsManager for training and validation episodes."
            )

        results = self._validate_episode_results(results)

        episode_dir = self.bench_path / f"episode_{results.episode:03d}"
        episode_dir.mkdir(exist_ok=True)

        # 1. Save scalars as JSON (human-readable)
        scalars = {
            "episode": results.episode,
            "seed": results.seed,
            "total_reward": results.total_reward,
            "mean_daily_failures": results.mean_daily_failures,
            "total_failures": results.total_failures,
        }
        with open(episode_dir / "scalars.json", "w") as f:
            json.dump(scalars, f, indent=2, cls=_NumpyEncoder)

        # 2. Save time-series as CSV (easy analysis)
        timeslot_df = pd.DataFrame(
            {
                "timeslot": range(len(results.failures_per_timeslot)),
                "failures": results.failures_per_timeslot,
                "deployed_bikes": results.deployed_bikes,
                "truck_load": results.truck_load,
                "depot_load": results.depot_load,
                "outside_system_bikes": results.outside_system_bikes,
                "traveling_bikes": results.traveling_bikes,
                "rebalance_times": results.rebalance_times,
                "demand": results.demand_per_timeslot,
                "global_critic_score": results.global_critic_scores,
            }
        )
        timeslot_df.to_csv(episode_dir / "timeslot_metrics.csv", index=False)

        # 3. Save cell subgraph separately (spatial data)
        if results.cell_subgraph is not None:
            with open(episode_dir / "cell_subgraph.gpickle", "wb") as f:
                pickle.dump(results.cell_subgraph, f)

        # 4. Update aggregated summary
        self._update_summary(results)

    def _update_summary(self, results: EpisodeResults):
        """Append episode summary to aggregated DataFrame."""
        summary_row = {
            "episode": results.episode,
            "total_reward": results.total_reward,
            "mean_daily_failures": results.mean_daily_failures,
            "total_failures": results.total_failures,
        }

        self.bench_summary = pd.concat(
            [self.bench_summary, pd.DataFrame([summary_row])], ignore_index=True
        )

    def save_run_summary(self):
        """Save aggregated summary CSV for the entire benchmark run."""
        if not self.bench_summary.empty:
            self.bench_summary.to_csv(
                self.bench_path / "bench_summary.csv", index=False
            )

    def save_hyperparameters(self, params: dict):
        """Save hyperparameters and configuration."""
        config = {
            "run_id": self.run_id,
            "hyperparameters": params,
        }
        with open(self.bench_path / "config.json", "w") as f:
            json.dump(config, f, indent=2, cls=_NumpyEncoder)

    def load_episode(self, episode: int) -> EpisodeResults:
        """Load results for a specific benchmark episode."""
        episode_dir = self.bench_path / f"episode_{episode:03d}"

        # Load scalars
        with open(episode_dir / "scalars.json", "r") as f:
            scalars = json.load(f)

        # Load time-series
        timeslot_df = pd.read_csv(episode_dir / "timeslot_metrics.csv")

        # Load cell subgraph if it exists
        subgraph_path = episode_dir / "cell_subgraph.gpickle"
        if subgraph_path.exists():
            with open(subgraph_path, "rb") as f:
                cell_subgraph = pickle.load(f)
        else:
            cell_subgraph = None

        return EpisodeResults(
            episode=scalars["episode"],
            mode="benchmark",
            seed=scalars["seed"],
            total_reward=scalars["total_reward"],
            mean_daily_failures=scalars["mean_daily_failures"],
            total_failures=scalars["total_failures"],
            failures_per_timeslot=timeslot_df["failures"].tolist(),
            deployed_bikes=timeslot_df["deployed_bikes"].tolist(),
            truck_load=timeslot_df["truck_load"].tolist(),
            depot_load=timeslot_df["depot_load"].tolist(),
            outside_system_bikes=timeslot_df["outside_system_bikes"].tolist(),
            traveling_bikes=timeslot_df["traveling_bikes"].tolist(),
            rebalance_times=timeslot_df["rebalance_times"].tolist(),
            demand_per_timeslot=timeslot_df["demand"].tolist(),
            global_critic_scores=timeslot_df["global_critic_score"].tolist(),
            cell_subgraph=cell_subgraph,
        )

    @staticmethod
    def _validate_episode_results(results: EpisodeResults) -> EpisodeResults:
        """
        Validate and fix EpisodeResults to ensure data consistency.

        Uses failures_per_timeslot as the reference length (the benchmark has
        no reward timeseries to anchor on).  All other timeslot arrays are
        padded or truncated to match.
        """
        expected_length = len(results.failures_per_timeslot)

        if expected_length == 0:
            raise ValueError("Episode results must have at least one timeslot of data")

        def pad_to_length(data, length, fill_value=0.0):
            if len(data) == length:
                return data
            elif len(data) < length:
                return data + [fill_value] * (length - len(data))
            else:
                return data[:length]

        return EpisodeResults(
            episode=results.episode,
            mode=results.mode,
            seed=results.seed,
            total_reward=results.total_reward,
            mean_daily_failures=results.mean_daily_failures,
            total_failures=results.total_failures,
            # Ensure consistent timeslot-array lengths
            failures_per_timeslot=pad_to_length(
                results.failures_per_timeslot, expected_length, 0
            ),
            deployed_bikes=pad_to_length(results.deployed_bikes, expected_length, 0),
            truck_load=pad_to_length(results.truck_load, expected_length, 0),
            depot_load=pad_to_length(results.depot_load, expected_length, 0),
            outside_system_bikes=pad_to_length(
                results.outside_system_bikes, expected_length, 0
            ),
            traveling_bikes=pad_to_length(results.traveling_bikes, expected_length, 0),
            demand_per_timeslot=pad_to_length(
                results.demand_per_timeslot, expected_length, 0.0
            ),
            global_critic_scores=pad_to_length(
                results.global_critic_scores, expected_length, 0.0
            ),
            # rebalance_times has variable length (events, not timeslots) — keep as-is
            rebalance_times=results.rebalance_times,
            # Spatial data — no length requirement
            cell_subgraph=results.cell_subgraph,
        )
