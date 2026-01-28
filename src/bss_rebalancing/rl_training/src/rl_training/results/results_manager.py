import json
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pickle
import networkx as nx


@dataclass
class EpisodeResults:
    """Container for all metrics from a single episode."""
    episode: int
    mode: str  # 'train' or 'validation'

    # Episode-level scalars
    total_reward: float = 0.0
    mean_failures: float = 0.0
    total_failures: int = 0
    total_trips: int = 0
    total_invalid: int = 0
    epsilon: float = 0.0

    # Time-series data (per timeslot)
    rewards_per_timeslot: List[float] = field(default_factory=list)
    failures_per_timeslot: List[int] = field(default_factory=list)
    epsilon_per_timeslot: List[float] = field(default_factory=list)
    deployed_bikes: List[int] = field(default_factory=list)
    q_values_per_timeslot: List[np.ndarray] = field(default_factory=list)

    # Action-level data (per step)
    action_per_step: List[int] = field(default_factory=list)
    reward_tracking: Dict[int, List[float]] = field(default_factory=dict)
    losses: List[float] = field(default_factory=list)
    global_critic_scores: List[float] = field(default_factory=list)

    # Spatial data (cell subgraph)
    cell_subgraph: Optional[nx.Graph] = None


class ResultsManager:
    """Centralized manager for training and validation results."""

    def __init__(self, results_path: str, run_id: int, overwrite: bool = False, interactive: bool = True):
        self.results_path = Path(results_path)
        self.run_id = run_id
        self.run_dir = self.results_path / f"run_{run_id:03d}"
        self.training_path = self.run_dir / "training"
        self.validation_path = self.run_dir / "validation"
        self.models_path = self.run_dir / "models"

        # Check if results already exist
        if self.run_dir.exists() and not overwrite:
            if interactive:
                self._handle_existing_results()
            else:
                raise Exception(
                    f"Results directory already exists: {self.run_dir}. "
                    f"Use overwrite=True or change run_id."
                )

        # Create directories
        self.training_path.mkdir(parents=True, exist_ok=True)
        self.validation_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Initialize aggregated DataFrames
        self.training_summary = pd.DataFrame()
        self.validation_summary = pd.DataFrame()

        # Track best models
        self.best_models = []

    def _handle_existing_results(self):
        """Handle case where results directory already exists."""
        print(f"⚠️  WARNING: Results folder for run {self.run_id} already exists.")
        print(f"   Path: {self.run_dir}")
        print("   Data will be overwritten with new results.")

        try:
            proceed = input("Are you sure you want to continue? (y/n) ").strip().lower()
        except (ValueError, EOFError):
            proceed = ""
            print("Invalid input! Please enter 'y' or 'n'.")

        if proceed in ['y', 'yes']:
            print(f"Removing existing results at {self.run_dir}...")
            shutil.rmtree(self.run_dir)
        else:
            raise Exception(
                f"Aborted: Change the 'run_id' (current: {self.run_id}) "
                f"or the 'results_path' (current: {self.results_path})."
            )

    @classmethod
    def create_with_auto_increment(cls, results_path: str) -> 'ResultsManager':
        """Create a ResultsManager with auto-incremented run_id to avoid conflicts."""
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)

        # Find next available run_id
        existing_runs = [
            int(d.name.split('_')[1])
            for d in results_path.iterdir()
            if d.is_dir() and d.name.startswith('run_')
        ]
        next_run_id = max(existing_runs, default=-1) + 1

        print(f"Auto-assigned run_id: {next_run_id}")
        return cls(str(results_path), next_run_id, overwrite=False)

    def save_episode(self, results: EpisodeResults):
        """Save results for a single episode."""
        path = self.training_path if results.mode == 'train' else self.validation_path
        episode_dir = path / f"episode_{results.episode:03d}"
        episode_dir.mkdir(exist_ok=True)

        # 1. Save scalars as JSON (human-readable)
        scalars = {
            'episode': results.episode,
            'total_reward': results.total_reward,
            'mean_failures': results.mean_failures,
            'total_failures': results.total_failures,
            'total_trips': results.total_trips,
            'total_invalid': results.total_invalid,
            'epsilon': results.epsilon,
        }
        with open(episode_dir / 'scalars.json', 'w') as f:
            json.dump(scalars, f, indent=2)

        # 2. Save time-series as CSV (easy analysis)
        timeslot_df = pd.DataFrame({
            'timeslot': range(len(results.rewards_per_timeslot)),
            'reward': results.rewards_per_timeslot,
            'failures': results.failures_per_timeslot,
            'epsilon': results.epsilon_per_timeslot,
            'deployed_bikes': results.deployed_bikes,
        })
        timeslot_df.to_csv(episode_dir / 'timeslot_metrics.csv', index=False)

        # 3. Save step-level data as compressed pickle (large data)
        step_data = {
            'actions': results.action_per_step,
            'reward_tracking': results.reward_tracking,
            'losses': results.losses,
            'global_critic_scores': results.global_critic_scores,
            'q_values': results.q_values_per_timeslot,
        }
        with open(episode_dir / 'step_data.pkl.gz', 'wb') as f:
            pickle.dump(step_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. Save cell subgraph separately (spatial data)
        if results.cell_subgraph is not None:
            with open(episode_dir / 'cell_subgraph.gpickle', 'wb') as f:
                pickle.dump(results.cell_subgraph, f)

        # 5. Update aggregated summary
        self._update_summary(results)

    def _update_summary(self, results: EpisodeResults):
        """Append episode summary to aggregated DataFrame."""
        summary_row = {
            'episode': results.episode,
            'total_reward': results.total_reward,
            'mean_failures': results.mean_failures,
            'total_failures': results.total_failures,
            'total_trips': results.total_trips,
            'total_invalid': results.total_invalid,
            'epsilon': results.epsilon,
        }

        if results.mode == 'train':
            self.training_summary = pd.concat([
                self.training_summary,
                pd.DataFrame([summary_row])
            ], ignore_index=True)
        else:
            self.validation_summary = pd.concat([
                self.validation_summary,
                pd.DataFrame([summary_row])
            ], ignore_index=True)

    def save_run_summary(self):
        """Save aggregated summaries for the entire run."""
        if not self.training_summary.empty:
            self.training_summary.to_csv(
                self.training_path / 'training_summary.csv',
                index=False
            )
        if not self.validation_summary.empty:
            self.validation_summary.to_csv(
                self.validation_path / 'validation_summary.csv',
                index=False
            )

    def save_hyperparameters(self, params: dict, reward_params: dict):
        """Save hyperparameters and configuration."""
        config = {
            'run_id': self.run_id,
            'hyperparameters': params,
            'reward_parameters': reward_params,
        }
        with open(self.results_path / f"run_{self.run_id:03d}" / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def load_episode(self, episode: int, mode: str = 'train') -> EpisodeResults:
        """Load results for a specific episode."""
        path = self.training_path if mode == 'train' else self.validation_path
        episode_dir = path / f"episode_{episode:03d}"

        # Load scalars
        with open(episode_dir / 'scalars.json', 'r') as f:
            scalars = json.load(f)

        # Load time-series
        timeslot_df = pd.read_csv(episode_dir / 'timeslot_metrics.csv')

        # Load step data
        with open(episode_dir / 'step_data.pkl.gz', 'rb') as f:
            step_data = pickle.load(f)

        # Load cell subgraph if exists
        subgraph_path = episode_dir / 'cell_subgraph.gpickle'
        if subgraph_path.exists():
            with open(episode_dir / 'cell_subgraph.gpickle', 'rb') as f:
                cell_subgraph = pickle.load(f)
        else:
            cell_subgraph = None

        # Reconstruct EpisodeResults
        return EpisodeResults(
            episode=scalars['episode'],
            mode=mode,
            total_reward=scalars['total_reward'],
            mean_failures=scalars['mean_failures'],
            total_failures=scalars['total_failures'],
            total_trips=scalars['total_trips'],
            total_invalid=scalars['total_invalid'],
            epsilon=scalars['epsilon'],
            rewards_per_timeslot=timeslot_df['reward'].tolist(),
            failures_per_timeslot=timeslot_df['failures'].tolist(),
            epsilon_per_timeslot=timeslot_df['epsilon'].tolist(),
            deployed_bikes=timeslot_df['deployed_bikes'].tolist(),
            action_per_step=step_data['actions'],
            reward_tracking=step_data['reward_tracking'],
            losses=step_data['losses'],
            global_critic_scores=step_data['global_critic_scores'],
            q_values_per_timeslot=step_data['q_values'],
            cell_subgraph=cell_subgraph,
        )

    def save_model(self, agent, episode: int, score: float,
                   model_type: str = 'checkpoint', save_best: bool = False):
        """
        Save trained model.

        Args:
            agent: The agent to save
            episode: Current episode number
            score: Validation score (e.g., total_failures)
            model_type: Type of model save ('checkpoint', 'best', 'final')
            save_best: Whether to update best model tracker
        """
        if model_type == 'checkpoint':
            model_dir = self.models_path / 'checkpoints' / f"episode_{episode:03d}"
        elif model_type == 'best':
            model_dir = self.models_path / 'best' / f"episode_{episode:03d}"
        elif model_type == 'final':
            model_dir = self.models_path / 'final'
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'trained_agent.pt'

        # Save the model
        agent.save_model(str(model_path))

        # Save metadata
        metadata = {
            'episode': episode,
            'score': score,
            'model_type': model_type,
            'saved_at': pd.Timestamp.now().isoformat(),
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update best models tracker
        if save_best:
            self.best_models.append((episode, score))
            self.best_models.sort(key=lambda x: x[1])  # Sort by score (lower is better)

            # Save best models summary
            best_df = pd.DataFrame(self.best_models, columns=['episode', 'score'])
            best_df.to_csv(self.models_path / 'best_models_summary.csv', index=False)

        return model_path

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to the best performing model."""
        if not self.best_models:
            return None
        best_episode, _ = self.best_models[0]  # First element after sorting
        return self.models_path / 'best' / f"episode_{best_episode:03d}" / 'trained_agent.pt'

    def load_model(self, agent, episode: int = None, model_type: str = 'best'):
        """
        Load a saved model.

        Args:
            agent: The agent to load weights into
            episode: Specific episode to load (None for best/final)
            model_type: Type of model to load ('checkpoint', 'best', 'final')
        """
        if model_type == 'best' and episode is None:
            model_path = self.get_best_model_path()
            if model_path is None:
                raise FileNotFoundError("No best model found")
        elif model_type == 'final':
            model_path = self.models_path / 'final' / 'trained_agent.pt'
        elif model_type == 'checkpoint' and episode is not None:
            model_path = self.models_path / 'checkpoints' / f"episode_{episode:03d}" / 'trained_agent.pt'
        else:
            raise ValueError("Invalid combination of model_type and episode")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        agent.load_model(str(model_path))
        return model_path
