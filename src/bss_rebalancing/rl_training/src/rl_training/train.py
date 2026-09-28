import os
import sys
import json
import subprocess
import gymnasium
from dataclasses import dataclass
import torch
import argparse
import gc
import warnings
import logging

import gymnasium_env  # noqa: F401 — registers the gym environment

import gymnasium as gym
import numpy as np
import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions
from gymnasium_env.envs.fully_dynamic_env import EnvDefaults, RewardComponents

from rl_training.agents import PPOAgent
from rl_training.memory import PPOBuffer
from rl_training.results import ResultsManager, EpisodeResults
from rl_training.logging_config import init_logging, LoggingConfig, get_logger
from rl_training.utils import (
    convert_graph_to_data,
    convert_seconds_to_hours_minutes,
    set_seed,
    setup_device,
    build_cell_graph_from_cells,
    update_cell_graph_features
)
from rl_training.networks.ppo import PPO as PPONetwork 

# ------------------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------------------

devices = ["cpu"]
if torch.cuda.is_available():
    num_cuda = torch.cuda.device_count()
    for i in range(num_cuda):
        devices.append(f"cuda:{i}")
if torch.backends.mps.is_available():
    devices.append("mps")

# Only print in the main process, not in spawned children
if mp.current_process().name == "MainProcess":
    print(f"Devices available: {devices}\n")

# ------------------------------------------------------------------------------
# Default params
# ------------------------------------------------------------------------------

params = {
    "seed": int(42),                                # Random seed for reproducibility
    "num_episodes": 250,                            # Total number of training episodes
    "rollout_steps": 4096,                          # Buffer capacity
    "minibatch_size": 512,                          # Dimension of each minibatch
    "gamma": 0.99,                                  # Discount factor
    "exploration_time": 0.7,                        # Fraction of total training time for exploration
    "lr": 5e-5,                                     # Learning rate
    # PPO params
    "clip_coef": 0.2,                               # Clipping coefficient 
    "gae_lambda": 0.95,                             # Generalized Advantage Estimation (GAE) factor 
    "ent_coef": 0.02,                               # Entropy coefficient (starting value — see ent_coef_final for linear decay)
    "ent_coef_final": 0.02,                         # ent_coef decays linearly from `ent_coef` to this value over
                                                     # num_episodes, instead of staying constant. A constant, fairly
                                                     # high ent_coef keeps the policy exploratory/uncertain for a long
                                                     # stretch of training; once the advantage signal finally
                                                     # overcomes it, the policy specializes abruptly (visible as a
                                                     # sudden entropy collapse + reward/failures "step" in the plots).
                                                     # Decaying it gradually smooths that transition. Set equal to
                                                     # `ent_coef` to disable decay and keep the old constant behavior.
    "vf_coef": 0.25,                                # Value coefficient
    "update_epochs": 8,                             # How many times buffer is processed at every update 

    "total_timeslots": 56,                  # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 1000,        # Maximum number of bikes in the system
    "minimum_number_of_bikes": 5,           # Minimum number of bikes per cell
    "enable_repositioning": False,          # Use base repositioning strategy at the start of each episode
    "use_net_flow": False,                  # Use net flow repositioning strategy at the start of each episode
    "depot_position_id": 12,                # ID (cell) of the (shared) central bike depot
    "initial_cell_id": 12,                  # Initial cell for truck #1 (truck #2's cell is
                                            # sampled at random among the remaining cells —
                                            # set "initial_cell_ids" below to pin both explicitly)
    "num_trucks": 1,                        # Number of trucks per env (multi-truck-in-one-map
                                            # support still exists in the env for later — see
                                            # fully_dynamic_env.py — but the current setup uses
                                            # multi-area training instead: see --data-paths)
    "initial_cell_ids": [12, 17],           # Optional: e.g. [12, 40] to fix each truck's
                                            # starting cell explicitly when num_trucks > 1

    "validation_epsilon_threshold": 0.1,
    "validation_timeout": 600,
}

# Define which metrics to use as GNN features
gnn_features = [
    'truck_cell',         # now a count of trucks in the cell (was a 0/1 flag)
    'active_truck_cell',  # 1 for the cell of the truck the shared policy is
                               # currently controlling — lets one set of weights
                               # act for either truck (parameter sharing)
    'critic_score',
    'eligibility_score',
    'total_bikes',
]

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
 
def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="BSS Train Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Run full preprocessing pipeline
            bss-train --data-path data/
 
            # Specify run ID and results path
            bss-train --run-id 1 --data-path data/ --results-path results/
 
            # Use GPU device
            bss-train --data-path data/ --device cuda:0
 
            # Set random seed and number of episodes
            bss-train --data-path data/ --seed 123 --num-episodes 150
 
            # Enable logging
            bss-train --data-path data/ --enable-logging 
 
            # Perform only one validation at the end of training
            bss-train --data-path data/ --one-validation  
 
            # Perform training with number of bikes and exploration time
            bss-train --data-path data/ --num-bikes 300 --exploration-time 0.8
 
            # Use a separate GPU for validation subprocesses
            bss-train --data-path data/ --device cuda:0 --val-device cuda:1
        """,
    )
 
    parser.add_argument(
        '--run-id',
        type=int,
        default=0,
        help='Run ID for the experiment.'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/',
        help='Path to the data folder. Ignored if --data-paths is given.'
    )
    parser.add_argument(
        '--data-paths',
        type=str,
        default=None,
        help=(
            'Comma-separated list of data folder paths for multi-area training '
            '(e.g. "data/manhattan_north,data/manhattan_south"). One truck per '
            'area, all areas share the same policy (parameter sharing). '
            'Overrides --data-path when given.'
        )
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default='results/',
        help='Path to the results folder.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help=f'Hardware device to use. Available options: {devices}.'
    )
    parser.add_argument(
        '--val-device',
        type=str,
        default=None,
        help=f'Hardware device to use for validation subprocesses. Falls back to --device if not specified. Available options: {devices}.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=params["seed"],
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=params['num_episodes'],
        help='Number of episodes to train.'
    )
    parser.add_argument(
        '--max-num-bikes',
        type=int,
        default=params['maximum_number_of_bikes'],
        help='Number of bikes of the system.'
    )
    parser.add_argument(
        '--min-num-bikes',
        type=int,
        default=params['minimum_number_of_bikes'],
        help='Minimum number of bikes per cell.'
    )
    parser.add_argument(
        '--enable-repositioning',
        action='store_true',
        help='Enable repositioning beyond minimum bikes per cell at the start of each episode.'
    )
    parser.add_argument(
        '--use-net-flow',
        action='store_true',
        help='Use net-flow-based repositioning instead of random at the start of each episode.'
    )
    parser.add_argument(
        '--exploration-time',
        type=float,
        default=params['exploration_time'],
        help='Number of episodes to explore.'
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable logging.'
    )
    parser.add_argument(
        '--one-validation',
        action='store_true',
        help='Performs only one validation at the end of the training.'
    )  # TODO: fix this feature
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help="Optimizer for the PPO network. 'sgd' uses SGD with momentum (see --momentum)."
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for the SGD optimizer (ignored if --optimizer=adam).'
    )
    parser.add_argument(
        '--resume-run-id',
        type=int,
        default=None,
        help=(
            'Load network weights from this existing run_id before training starts '
            '(fresh training loop, episode counter starts at 0 — this only warm-starts '
            'the weights, e.g. to switch optimizer/hyperparameters mid-training). '
            'The checkpoint is read from --results-path/run_{resume-run-id:03d}/models/. '
            'Only the network weights are loaded, never the optimizer state — so this '
            'always starts the new optimizer (Adam or SGD) completely fresh.'
        )
    )
    parser.add_argument(
        '--resume-model-type',
        type=str,
        default='best',
        choices=['best', 'final', 'episode'],
        help="Which checkpoint to load from --resume-run-id. Use 'episode' with --resume-episode."
    )
    parser.add_argument(
        '--resume-episode',
        type=int,
        default=None,
        help="Episode number to load when --resume-model-type=episode."
    )
    parser.add_argument(
        '--start-episode',
        type=int,
        default=None,
        help=(
            'Override the episode number training resumes at (only used with '
            '--resume-run-id). Defaults to (checkpoint episode + 1), read from '
            "the checkpoint's own metadata.json, so numbering continues "
            'naturally instead of restarting at 0.'
        )
    )
 
    return parser
 
 
# ------------------------------------------------------------------------------
# Subprocess-based validation helpers
# ------------------------------------------------------------------------------
 
def _get_validate_script_path() -> str:
    """
    Resolve the absolute path to validate.py.
 
    Strategy (in order):
      1. Same directory as this train.py file  ← works for src-layout packages
      2. `bss-validate` console-script on PATH  ← works if installed via pip/setup.py
      3. Raises immediately so you know rather than silently failing.
    """
    candidate = Path(__file__).parent / "validate.py"
    if candidate.exists():
        return str(candidate)
 
    import shutil
    entry = shutil.which("bss-validate")
    if entry:
        return str(entry)
 
    raise FileNotFoundError(
        "Cannot locate validate.py. Expected it next to train.py, "
        "or 'bss-validate' on PATH (installed via pip)."
    )
 
 
def _build_validate_cmd(
        run_id: int,
        data_path: str,
        results_path: str,
        episode: int,
        val_device: str,
        seed: int,
        max_num_bikes: int,
        min_num_bikes: int,
        total_timeslots: int,
        enable_repositioning: bool,
        use_net_flow: bool,
        data_paths: list[str] | None = None,
) -> list[str]:
    """
    Build the argv list to invoke validate.py as a completely independent subprocess
    — exactly as if you typed it in your terminal.
    Uses sys.executable so the subprocess runs in the same venv as training.
 
    If `data_paths` is given (multi-area training), validate.py is invoked with
    --data-paths so it evaluates the shared policy on every area, matching
    validate_ppo_multi_env(). Otherwise falls back to the single --data-path.
    """
    validate_script = str(_get_validate_script_path())
    cmd = [
        sys.executable, validate_script,
        "--run-id", str(run_id),
    ]
    if data_paths:
        cmd += ["--data-paths", ",".join(data_paths)]
    else:
        cmd += ["--data-path", data_path]
    cmd += [
        "--results-path", results_path,
        "--model-type", "episode",
        "--model-episode", str(episode),
        "--device", val_device,
        "--seed", str(seed),
        "--max-num-bikes", str(max_num_bikes),
        "--min-num-bikes", str(min_num_bikes),
        "--total-timeslots", str(total_timeslots),
        "--non-interactive",
    ]
    if enable_repositioning:
        cmd.append("--enable-repositioning")
    if use_net_flow:
        cmd.append("--use-net-flow")
    return cmd
 
 
@dataclass
class _PendingVal:
    """Tracks a validation subprocess running in parallel with training."""
    episode: int
    proc: subprocess.Popen
 
 
def _launch_validation_subprocess(
        cmd: list[str],
        episode: int,
        logger: logging.Logger,
) -> '_PendingVal | None':
    """
    Spawn validate.py as fire-and-forget — training continues immediately.
    Returns a _PendingVal handle to be collected later.
    stdout/stderr are inherited so the validation tqdm bar prints inline.
    """
    logger.info(f"[val] Launching validation subprocess for episode {episode}: {' '.join(cmd)}")
 
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        return _PendingVal(episode=episode, proc=proc)
    except Exception as e:
        logger.error(f"[val] Failed to launch validation subprocess for episode {episode}: {e}")
        print(f"[VAL] Could not launch validation for episode {episode}: {e}")
        return None
 
 
def _collect_pending_val(
        pending: '_PendingVal',
        logger: logging.Logger,
        timeout: int = int(params['validation_timeout']),
) -> bool:
    """
    Wait (up to `timeout` seconds) for a previously launched validation subprocess
    to finish. Returns True on clean exit, False on timeout or non-zero exit.
    Called at the next validation gate, so training is never stalled mid-episode.
    """
    logger.info(f"[val] Waiting for validation of episode {pending.episode} to finish...")
 
    try:
        pending.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pending.proc.kill()
        pending.proc.wait()
        logger.warning(
            f"[val] Validation subprocess for episode {pending.episode} timed out "
            f"after {timeout}s — skipping best-model check."
        )
        print(f"[VAL] Validation timed out for episode {pending.episode}, skipping.")
        return False
 
    if pending.proc.returncode != 0:
        logger.warning(
            f"[val] Validation subprocess for episode {pending.episode} "
            f"exited with code {pending.proc.returncode}."
        )
        print(f"[VAL] Validation exited non-zero ({pending.proc.returncode}) for episode {pending.episode}.")
        return False
 
    logger.info(f"[val] Validation for episode {pending.episode} completed successfully.")
    return True
 
 
def _read_validation_score(
        results_path: str,
        run_id: int,
        episode: int,
        logger: logging.Logger,
) -> float | None:
    """
    After the validation subprocess finishes, read the scalar JSON it wrote and
    return `mean_daily_failures` (lower is better).  Returns None on any I/O error.
    """
    val_tag = ResultsManager.build_val_tag("episode", episode)
    scalars_path = (
            Path(results_path) / f"run_{run_id:03d}" / "validation" / val_tag
            / "episode_000" / "scalars.json"
    )
 
    try:
        with open(scalars_path, "r") as f:
            scalars = json.load(f)
        score = float(scalars["mean_daily_failures"])
        logger.info(f"[val] Episode {episode} validation score (mean_daily_failures): {score:.4f}")
        return score
    except FileNotFoundError:
        logger.warning(f"[val] Scalars file not found at {scalars_path}. Skipping best-model update.")
        return None
    except Exception as e:
        logger.error(f"[val] Failed to read validation score for episode {episode}: {e}")
        return None
 
 
# ========================================
#     train ppo
# ========================================
 
def train_ppo(
    env: gymnasium.Env,
    agent: PPOAgent,
    buffer: PPOBuffer,
    episode: int,
    device: torch.device,
    run_id: int,
    logging_enabled: bool,
    tbar=None,
    episode_results_path: str | None = None,
    seed: int = None,
) -> dict:
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    rewards = []
    failures = []
    state_values = [] # Replaces epsilons and q_values
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    traveling_bikes = []
    demand_per_timeslot = []
 
    # Per-step metrics
    action_per_step = []
    global_critic_scores = []
    # Assumes Actions is accessible in your scope
    # reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}
    reward_tracking_per_action = {} # Initialized dynamically to avoid import issues here
 
    # Accumulators (reset each timeslot)
    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    iterations = 0
    last_cumulative_demand = 0
 
    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    # Ensure params and reward_params are accessible in your real file
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': params["minimum_number_of_bikes"],
        'enable_repositioning': params["enable_repositioning"],
        'use_net_flow': params["use_net_flow"],
        'discount_factor': params["gamma"],
        'depot_id': params['depot_position_id'],
        'num_trucks': params['num_trucks'],
        'initial_cell': params['initial_cell_id'],
    }
    if params.get('initial_cell_ids'):
        # Overrides 'initial_cell' above: fixes every truck's starting cell.
        reset_options['initial_cells'] = params['initial_cell_ids']
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path
 
    agent_state, info = env.reset(options=reset_options)
 
    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_lookup = info['distance_lookup']
 
    # Build initial graph from cells
    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict,
        nodes_dict=nodes_dict,
        distance_lookup=distance_lookup
    )
 
    # Initialize state
    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']
 
    # ============================================================================
    # Main training loop
    # ============================================================================
    episode_cell_stats = {
        cell_id: {
            'critic_sum': 0.0,
            'eligibility_sum': 0.0,
            'bikes_sum': 0.0,
            'bikes_dead_sum': 0.0
        }
        for cell_id in cell_dict.keys()
    }
 
    done = False
    update_freq = params["rollout_steps"]  # Number of steps before update
    step_counter = 0   # Step counter
    pg_loss, v_loss, entropy = 0.0, 0.0, 0.0
    buffer.clear()
    
    while not done:
        # Prepare state for agent (S)
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )
 
        # Retrieve forbidden actions from environment (if provided)
        avoid_actions = info.get("avoid_action", [])
 
        # Select action using PPO Actor (stochastic, no epsilon)
        # We also get the logprob and the Critic's value estimation
        action, logprob, value = agent.select_action(single_state, avoid_action=avoid_actions)
 
        # Step into: get reward and observation (R)
        agent_state, reward, done, timeslot_terminated, info = env.step(action)
        reward = float(np.clip(reward, -2.0, 3.0))
 
        # Update node attributes
        cell_dict = info['cell_dict']
        update_cell_graph_features(cell_graph, cell_dict)
 
        # Update cumulative cell statistics
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['eligibility_sum'] += cell.get_eligibility_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] += cell.get_dead_bikes()
 
        # Create next state (S')
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']
 
        # Store transition in the PPO Rollout Buffer
        # Notice we don't strictly need next_state for standard GAE if we handle dones correctly, 
        # but we save the current interaction data.
        buffer.push(
            state=single_state.cpu(), # Move to CPU to save VRAM during episode
            action=action,
            logprob=logprob.item(),
            reward=reward,
            value=value.item(),
            done=done
        )
 
        # Record step metrics
        step_counter += 1
        action_per_step.append(action)
        #if action not in reward_tracking_per_action:
        #    reward_tracking_per_action[action] = []
        #reward_tracking_per_action[action].append(reward)
        reward_tracking_per_action.setdefault(action, []).append(reward)
        
        global_critic_scores.append(info['global_critic_score'])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info['failures'])
        iterations += 1
        
        # ============================================================================
        # PPO Update after k steps
        # ============================================================================
        if len(buffer) >= update_freq:
            # Bootstrapping
            if done:
                last_value = 0.0
            else:
                next_state_data = Data(
                    x=next_state.x.to(device),
                    edge_index=next_state.edge_index.to(device),
                    edge_attr=next_state.edge_attr.to(device),
                    batch=torch.zeros(next_state.x.size(0), dtype=torch.long).to(device),
                )
                
                with torch.no_grad():
                    # Interroghiamo la rete per avere il valore del next_state
                    _, _, next_value = agent.select_action(next_state_data)
                    last_value = next_value.item()
            
            # Perform the update 
            pg_loss, v_loss, entropy = agent.update(buffer, last_value=last_value)
            
            # Clear the buffer to collect new data
            buffer.clear()
 
        # Handle timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1
 
            # PPO DOES NOT update target networks or epsilons here.
            # We also don't need the expensive get_q_values calculation anymore.
            # We simply record the Critic's value from the last step of the timeslot.
 
            # Record timeslot metrics
            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            state_values.append(value.item()) # Replacing q_values/epsilon tracking
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])
 
            current = sum(cell.get_total_demand() for cell in cell_dict.values())
            demand_per_timeslot.append(current - last_cumulative_demand)
            last_cumulative_demand = current
            
            # Reset accumulators
            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0
 
            # Update progress bar
            if tbar is not None:
                tbar.set_description(
                    f"[TRAIN] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                # Replaced epsilon with the Critic's Value estimation
                tbar.set_postfix({'Val': f"{value.item():.2f}"})
                tbar.update(1)
 
        # Move to next state
        state = next_state
        #del single_state
 
    if len(buffer) > 0:
        pg_loss, v_loss, entropy = agent.update(buffer, last_value=0.0)
        buffer.clear()
        
    # Cleanup
    torch.cuda.empty_cache()
 
    steps_in_episode = iterations
    for cell_id, stats in episode_cell_stats.items():
        center_node = cell_dict[cell_id].get_center_node()
        if center_node not in cell_graph.nodes:
            continue
 
        if steps_in_episode > 0:
            critic_mean = stats.get('critic_sum', 0.0) / steps_in_episode
            eligibility_mean = stats.get('eligibility_sum', 0.0) / steps_in_episode
            bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
            dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
        else:
            critic_mean = eligibility_mean = bikes_mean = dead_bikes_mean = 0.0
 
        nx_attrs = cell_graph.nodes[center_node]
        nx_attrs['critic_mean'] = critic_mean
        nx_attrs['eligibility_mean'] = eligibility_mean
        nx_attrs['failure_sum'] = cell_dict[cell_id].get_failures()
        nx_attrs['failure_rate'] = cell_dict[cell_id].get_failure_rate()
        nx_attrs['visits_sum'] = cell_dict[cell_id].get_visits()
        nx_attrs['ops_sum'] = cell_dict[cell_id].get_ops()
        nx_attrs['bikes_mean'] = bikes_mean
        nx_attrs['dead_bikes_mean'] = dead_bikes_mean
 
    # ============================================================================
    # Return results
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards,
        "failures_per_timeslot": failures,
        "total_invalid_actions": info.get("total_invalid_actions", 0),
        "state_values_per_timeslot": state_values, # Formerly q_values / epsilons
        "action_per_step": action_per_step,
        "global_critic_scores": global_critic_scores,
        "reward_tracking_per_action": reward_tracking_per_action,
        "deployed_bikes": system_bikes,
        "truck_load": truck_load,
        "depot_load": depot_load,
        "outside_system_bikes": outside_system_bikes,
        "traveling_bikes": traveling_bikes,
        "demand_per_timeslot": demand_per_timeslot,
        "cell_subgraph": cell_graph,
        # PPO specific metrics to track training health
        "policy_loss": pg_loss,
        "value_loss": v_loss,
        "entropy": entropy,  # raw (unscaled) policy entropy — see ppo_agent.py
    }
    
 
# ----------------------------------------------------------------------------------------------------------------------
# Multi-area training (2+ separate data_paths, 1 truck each, one shared PPO policy)
# ----------------------------------------------------------------------------------------------------------------------
 
@dataclass
class _EnvContext:
    """Per-environment (per-area) rollout state, kept separate so a single
    shared PPOAgent/PPOBuffer can be fed transitions from several areas whose
    maps (cell graphs) differ in size — the GNN policy doesn't care, since it
    operates per-node/edge regardless of how many cells a given area has."""
    env: gymnasium.Env
    area_label: str
    cell_dict: dict
    cell_graph: object
    state: Data
    info: dict
    episode_cell_stats: dict
    done: bool = False
    timeslots_completed: int = 0
    total_reward_per_timeslot: float = 0.0
    total_failures_per_timeslot: int = 0
    last_cumulative_demand: float = 0.0
    iterations: int = 0
    last_value: float = 0.0
    rewards: list = None
    failures: list = None
    state_values: list = None
    system_bikes: list = None
    truck_load: list = None
    depot_load: list = None
    outside_system_bikes: list = None
    traveling_bikes: list = None
    demand_per_timeslot: list = None
    action_per_step: list = None
    global_critic_scores: list = None
    reward_tracking_per_action: dict = None
 
    def __post_init__(self):
        for name in (
            "rewards", "failures", "state_values", "system_bikes", "truck_load",
            "depot_load", "outside_system_bikes", "traveling_bikes",
            "demand_per_timeslot", "action_per_step", "global_critic_scores",
        ):
            if getattr(self, name) is None:
                setattr(self, name, [])
        if self.reward_tracking_per_action is None:
            self.reward_tracking_per_action = {}
 
 
def _reset_env_context(
    env: gymnasium.Env,
    area_label: str,
    episode_results_path: str | None,
) -> _EnvContext:
    """Resets one area's environment and builds its initial per-area context.
    Mirrors the reset block at the top of train_ppo(), but scoped to one env
    instead of module-level 'state'/'cell_graph' variables."""
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': params["minimum_number_of_bikes"],
        'enable_repositioning': params["enable_repositioning"],
        'use_net_flow': params["use_net_flow"],
        'discount_factor': params["gamma"],
        'depot_id': params['depot_position_id'],
        'num_trucks': 1,  # multi-area mode: 1 truck per area; the "multi-agent"
                           # part is across areas, not within a single map.
    }
    # Cell IDs are area-specific (different maps), so we deliberately do NOT
    # reuse params['initial_cell_id'] here — it would likely be invalid (or
    # mean a different physical place) in a second area. Each env samples a
    # random valid starting cell, unless explicitly pinned per area below.
    per_area_cells = params.get('initial_cell_ids_per_area')
    if per_area_cells and area_label in per_area_cells:
        reset_options['initial_cell'] = per_area_cells[area_label]
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path
 
    agent_state, info = env.reset(options=reset_options)
 
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_lookup = info['distance_lookup']
 
    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict, nodes_dict=nodes_dict, distance_lookup=distance_lookup
    )
 
    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']
 
    episode_cell_stats = {
        cell_id: {
            'critic_sum': 0.0, 'eligibility_sum': 0.0,
            'bikes_sum': 0.0, 'bikes_dead_sum': 0.0,
        }
        for cell_id in cell_dict.keys()
    }
 
    return _EnvContext(
        env=env, area_label=area_label, cell_dict=cell_dict, cell_graph=cell_graph,
        state=state, info=info, episode_cell_stats=episode_cell_stats,
    )
 
 
def train_ppo_multi_env(
    envs: list,
    agent: PPOAgent,
    buffer: PPOBuffer,
    episode: int,
    device: torch.device,
    run_id: int,
    logging_enabled: bool,
    tbar=None,
    episode_results_paths: list | None = None,
    seed: int = None,
) -> dict:
    """
    Multi-area counterpart of train_ppo(): runs one episode across several
    environments (each a different data_path/area, 1 truck each) at once,
    round-robining which area takes the next step, feeding every transition
    into the SAME buffer so agent.update() trains one shared policy on
    experience pooled across all areas (parameter sharing across areas,
    analogous to parameter sharing across trucks in the single-map case).
 
    An area that finishes its episode (reaches total_timeslots) earlier than
    the others simply drops out of the rotation; the loop continues with the
    remaining areas until all are done.
 
    Returns a dict with the exact same keys/shapes as train_ppo(), with
    per-area series concatenated in area order, so callers (main()'s
    EpisodeResults / ResultsManager / validation logic) don't need to change.
    A "per_area" key is added on top with each area's own series, for callers
    that want the breakdown.
 
    Known limitation: "cell_subgraph" in the returned dict is only the first
    area's graph (EpisodeResults has a single graph slot) — extending
    results/plots to show every area's graph is future work.
    """
    if episode_results_paths is None:
        episode_results_paths = [None] * len(envs)
 
    contexts = [
        _reset_env_context(env, f"area_{i}", episode_results_paths[i])
        for i, env in enumerate(envs)
    ]
 
    update_freq = params["rollout_steps"]
    buffer.clear()
    pg_loss, v_loss, entropy = 0.0, 0.0, 0.0
 
    active = list(range(len(contexts)))
    rr_pos = 0
 
    while active:
        idx = active[rr_pos % len(active)]
        ctx = contexts[idx]
 
        # Prepare state for agent (S)
        single_state = Data(
            x=ctx.state.x.to(device),
            edge_index=ctx.state.edge_index.to(device),
            edge_attr=ctx.state.edge_attr.to(device),
            batch=torch.zeros(ctx.state.x.size(0), dtype=torch.long).to(device),
        )
 
        avoid_actions = ctx.info.get("avoid_action", [])
        action, logprob, value = agent.select_action(single_state, avoid_action=avoid_actions)
 
        agent_state, reward, done, timeslot_terminated, info = ctx.env.step(action)
        reward = float(np.clip(reward, -2.0, 3.0))
 
        cell_dict = info['cell_dict']
        update_cell_graph_features(ctx.cell_graph, cell_dict)
        ctx.cell_dict = cell_dict
        ctx.info = info
 
        for cell_id, cell in cell_dict.items():
            stats = ctx.episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['eligibility_sum'] += cell.get_eligibility_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] += cell.get_dead_bikes()
 
        next_state = convert_graph_to_data(ctx.cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']
 
        buffer.push(
            state=single_state.cpu(),
            action=action,
            logprob=logprob.item(),
            reward=reward,
            value=value.item(),
            done=done,
        )
 
        ctx.action_per_step.append(action)
        ctx.reward_tracking_per_action.setdefault(action, []).append(reward)
        ctx.global_critic_scores.append(info['global_critic_score'])
        ctx.total_reward_per_timeslot += reward
        ctx.total_failures_per_timeslot += sum(info['failures'])
        ctx.iterations += 1
        ctx.last_value = value.item()
 
        # ------------------------------------------------------------------
        # PPO update after k steps (k counted across ALL areas combined)
        # ------------------------------------------------------------------
        if len(buffer) >= update_freq:
            if done:
                last_value = 0.0
            else:
                next_state_data = Data(
                    x=next_state.x.to(device),
                    edge_index=next_state.edge_index.to(device),
                    edge_attr=next_state.edge_attr.to(device),
                    batch=torch.zeros(next_state.x.size(0), dtype=torch.long).to(device),
                )
                with torch.no_grad():
                    _, _, next_value = agent.select_action(next_state_data)
                    last_value = next_value.item()
 
            pg_loss, v_loss, entropy = agent.update(buffer, last_value=last_value)
            buffer.clear()
 
        # Handle timeslot completion (per area)
        if timeslot_terminated:
            ctx.timeslots_completed += 1
 
            ctx.rewards.append(ctx.total_reward_per_timeslot)
            ctx.failures.append(ctx.total_failures_per_timeslot)
            ctx.state_values.append(value.item())
            ctx.system_bikes.append(info['number_of_system_bikes'])
            ctx.truck_load.append(info['truck_bikes'])
            ctx.depot_load.append(info['depot_bikes'])
            ctx.outside_system_bikes.append(info['number_of_outside_bikes'])
            ctx.traveling_bikes.append(info['number_of_traveling_bikes'])
 
            current = sum(cell.get_total_demand() for cell in cell_dict.values())
            ctx.demand_per_timeslot.append(current - ctx.last_cumulative_demand)
            ctx.last_cumulative_demand = current
 
            ctx.total_reward_per_timeslot = 0.0
            ctx.total_failures_per_timeslot = 0
 
            if tbar is not None:
                tbar.set_description(
                    f"[TRAIN] Run {run_id}. Epis {episode}, {ctx.area_label}, "
                    f"Week {info['week'] % 52}, {info['day'].capitalize()} "
                    f"at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.set_postfix({'Val': f"{value.item():.2f}"})
                tbar.update(1)
 
        ctx.state = next_state
        ctx.done = done
 
        if done:
            active.remove(idx)
            if active:
                rr_pos = rr_pos % len(active)
        else:
            rr_pos += 1
 
    if len(buffer) > 0:
        pg_loss, v_loss, entropy = agent.update(buffer, last_value=0.0)
        buffer.clear()
 
    torch.cuda.empty_cache()
 
    # Annotate each area's own cell_graph with per-episode stats (same logic
    # as the single-env path, just looped per area)
    for ctx in contexts:
        steps_in_episode = ctx.iterations
        for cell_id, stats in ctx.episode_cell_stats.items():
            center_node = ctx.cell_dict[cell_id].get_center_node()
            if center_node not in ctx.cell_graph.nodes:
                continue
 
            if steps_in_episode > 0:
                critic_mean = stats['critic_sum'] / steps_in_episode
                eligibility_mean = stats['eligibility_sum'] / steps_in_episode
                bikes_mean = stats['bikes_sum'] / steps_in_episode
                dead_bikes_mean = stats['bikes_dead_sum'] / steps_in_episode
            else:
                critic_mean = eligibility_mean = bikes_mean = dead_bikes_mean = 0.0
 
            nx_attrs = ctx.cell_graph.nodes[center_node]
            nx_attrs['critic_mean'] = critic_mean
            nx_attrs['eligibility_mean'] = eligibility_mean
            nx_attrs['failure_sum'] = ctx.cell_dict[cell_id].get_failures()
            nx_attrs['failure_rate'] = ctx.cell_dict[cell_id].get_failure_rate()
            nx_attrs['visits_sum'] = ctx.cell_dict[cell_id].get_visits()
            nx_attrs['ops_sum'] = ctx.cell_dict[cell_id].get_ops()
            nx_attrs['bikes_mean'] = bikes_mean
            nx_attrs['dead_bikes_mean'] = dead_bikes_mean
 
    # ------------------------------------------------------------------
    # Merge into the same shape train_ppo() returns
    # ------------------------------------------------------------------
    merged_reward_tracking: dict = {}
    for ctx in contexts:
        for a, rs in ctx.reward_tracking_per_action.items():
            merged_reward_tracking.setdefault(a, []).extend(rs)
 
    def _concat(field_name: str) -> list:
        out: list = []
        for ctx in contexts:
            out.extend(getattr(ctx, field_name))
        return out
 
    return {
        "rewards_per_timeslot": _concat("rewards"),
        "failures_per_timeslot": _concat("failures"),
        "total_invalid_actions": sum(ctx.info.get("total_invalid_actions", 0) for ctx in contexts),
        "state_values_per_timeslot": _concat("state_values"),
        "action_per_step": _concat("action_per_step"),
        "global_critic_scores": _concat("global_critic_scores"),
        "reward_tracking_per_action": merged_reward_tracking,
        "deployed_bikes": _concat("system_bikes"),
        "truck_load": _concat("truck_load"),
        "depot_load": _concat("depot_load"),
        "outside_system_bikes": _concat("outside_system_bikes"),
        "traveling_bikes": _concat("traveling_bikes"),
        "demand_per_timeslot": _concat("demand_per_timeslot"),
        "cell_subgraph": contexts[0].cell_graph,  # see limitation note in the docstring
        "policy_loss": pg_loss,
        "value_loss": v_loss,
        "entropy": entropy,  # raw (unscaled) policy entropy — see ppo_agent.py
        "per_area": {
            ctx.area_label: {
                "rewards_per_timeslot": ctx.rewards,
                "failures_per_timeslot": ctx.failures,
                "demand_per_timeslot": ctx.demand_per_timeslot,
                "deployed_bikes": ctx.system_bikes,
                "truck_load": ctx.truck_load,
                "depot_load": ctx.depot_load,
                "outside_system_bikes": ctx.outside_system_bikes,
                "traveling_bikes": ctx.traveling_bikes,
                "cell_subgraph": ctx.cell_graph,
            }
            for ctx in contexts
        },
    }
 
 
# ----------------------------------------------------------------------------------------------------------------------
 
def main():
    print("2. Start of the main section")
    # spawn is required before any CUDA context is created
    mp.set_start_method('spawn', force=True)
    warnings.filterwarnings("ignore")
 
    args = create_parser().parse_args()
 
    device = setup_device(args.device.lower(), devices)
    val_device = setup_device(args.val_device.lower(), devices) if args.val_device else device
 
    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------
    validation_process = None
    run_id = args.run_id
    data_path = args.data_path
    data_paths = [p.strip() for p in args.data_paths.split(",")] if args.data_paths else None
    results_path = args.results_path
    logging_enabled = args.log
 
    params['seed'] = args.seed
    params['num_episodes'] = args.num_episodes
    params['optimizer'] = args.optimizer
    params['momentum'] = args.momentum
    params['maximum_number_of_bikes'] = args.max_num_bikes
    params['minimum_number_of_bikes'] = args.min_num_bikes
    params['enable_repositioning'] = args.enable_repositioning
    params['use_net_flow'] = args.use_net_flow
    params['exploration_time'] = args.exploration_time
 
    print(f"Setting seed: {params['seed']}")
    set_seed(params['seed'])
 
    # Ensure the data path(s) exist
    if data_paths:
        for p in data_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"The specified data path does not exist: {p}")
        # Validation subprocess (validate.py) supports --data-paths too, and
        # will be invoked with the same set of areas — see _build_validate_cmd.
        data_path = data_paths[0]
    elif not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")
 
    # At 60% of the total timeslots (60% of the training) the epsilon should be 0.1
 
    results_manager = ResultsManager(
        results_path=results_path,
        run_id=run_id,
        overwrite=False,
        interactive=True
    )
 
    # Init logging
    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(results_manager.training_path, "logs"),
        run_id=run_id,
        console=False,
        logger_name="train",
    ))
    logger = get_logger("train", logger_name="train")
    logger.info("Starting training loop")
 
    print("3. Before gym.make")
    # Create the environment(s). Multi-area mode (data_paths given): one env
    # per area, each with 1 truck, all sharing the same policy — see
    # train_ppo_multi_env(). Single-area mode: unchanged, one env.
    if data_paths:
        envs = [
            gym.make(
                'gymnasium_env/FullyDynamicEnv-v0',
                data_path=p,
                results_path=f"{str(results_manager.training_path)}/area_{i}/",
                seed=params['seed'],
                logging_enabled=logging_enabled
            )
            for i, p in enumerate(data_paths)
        ]
        env = envs[0]  # kept for code below that still refers to a single `env`
        print(f"4. {len(envs)} environments created (multi-area: {data_paths})")
    else:
        env = gym.make(
            'gymnasium_env/FullyDynamicEnv-v0',
            data_path=data_path,
            results_path=f"{str(results_manager.training_path)}/",
            seed=params['seed'],
            logging_enabled=logging_enabled
        )
        envs = [env]
        print("4. Environment created")
 
    # Save hyperparameters
    results_manager.save_hyperparameters(
        params={
            **params,
            **{k: v for k, v in vars(EnvDefaults).items() if not k.startswith('_')},
            # Record the data path(s) so the results webapp can find the right
            # base map per area (see results_webapp app.py --data-paths).
            "data_path": data_path,
            "data_paths": data_paths,
        },
        reward_params={k: v for k, v in vars(RewardComponents).items() if not k.startswith('_')}
    )
 
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Validation device: {val_device}")
    print(f"Params: {params}")
    print("=" * 80)
 
    # ------------------------------------------------------------------
    # Agent with replay buffer
    # ------------------------------------------------------------------
    # Set up replay buffer
    ppo_buffer = PPOBuffer() # PPO
 
    # Initialize the PPO agent
    network = PPONetwork(
        num_actions=env.action_space.n,
        node_features=len(gnn_features),  # was hardcoded 4; now tracks gnn_features above
    ).to(device)
    
    agent = PPOAgent(
        #seed=int(params['seed']),
        network=network,
        lr=params.get("lr", 3e-4),
        gamma=params.get("gamma", 0.99),
        gae_lambda=params.get("gae_lambda"),
        clip_coef=params.get("clip_coef"),
        ent_coef=params.get("ent_coef"),
        vf_coef=params.get("vf_coef"),
        update_epochs=params.get("update_epochs"),
        device=device,
        optimizer=params.get("optimizer", "adam"),
        momentum=params.get("momentum", 0.9),
    )
    print(f"Model initialized successfully (optimizer={agent.optimizer_name}).\n")
 
    # ------------------------------------------------------------------
    # Optional warm start: load network weights from an existing run's
    # checkpoint before training begins. Episode counter still starts at 0,
    # and the optimizer (Adam or SGD, per --optimizer) always starts fresh —
    # only weights are ever saved/loaded, never optimizer state. Typical use:
    # resuming after an interrupted run, or continuing training with a
    # different optimizer/learning-rate/reward configuration.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Optional warm start: load network weights from an existing run's
    # checkpoint before training begins. The optimizer (Adam or SGD, per
    # --optimizer) always starts fresh — only weights are ever saved/loaded,
    # never optimizer state. Typical use: resuming after an interrupted run,
    # or continuing training with a different optimizer/learning-rate/reward
    # configuration.
    #
    # By default the episode counter CONTINUES from the checkpoint's own
    # episode number (read from its metadata.json) instead of restarting at
    # 0, so plots/episode folders read as one continuous arc across the two
    # run directories. Override with --start-episode if you want something
    # else (e.g. restart numbering at 0).
    # ------------------------------------------------------------------
    start_episode = 0
    if args.resume_run_id is not None:
        if args.resume_model_type == 'episode' and args.resume_episode is None:
            raise ValueError("--resume-episode is required when --resume-model-type=episode")
 
        resume_run_dir = Path(args.results_path) / f"run_{args.resume_run_id:03d}"
        resume_models_path = resume_run_dir / "models"
        if args.resume_model_type == 'best':
            checkpoint_dir = resume_models_path / 'best'
        elif args.resume_model_type == 'final':
            checkpoint_dir = resume_models_path / 'final'
        else:  # 'episode'
            checkpoint_dir = resume_models_path / 'episodes' / f'episode_{args.resume_episode:03d}'
        checkpoint_path = checkpoint_dir / 'trained_agent.pt'
 
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
 
        agent.load_model(str(checkpoint_path))
        print(f"[RESUME] Loaded weights from {checkpoint_path}")
 
        if args.start_episode is not None:
            start_episode = args.start_episode
        else:
            metadata_path = checkpoint_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    start_episode = json.load(f)['episode'] + 1
            else:
                print(
                    "[RESUME] No metadata.json found next to the checkpoint — "
                    "episode numbering starts at 0. Pass --start-episode to set "
                    "it explicitly."
                )
        print(f"[RESUME] Episode numbering will start at {start_episode}\n")
 
 
    # Train the agent using the training loop
    # In multi-area mode, per-timeslot series returned by train_ppo_multi_env()
    # are concatenated across areas, so the denominator for mean_daily_failures
    # must scale with the number of areas too, or the metric is inflated.
    num_days = int(params["total_timeslots"] // 8) * max(1, len(envs))
 
    # Best validation score tracker (lower mean_daily_failures = better)
    best_val_score = float("inf")
 
    # Holds the currently running validation subprocess (if any).
    # There is at most one pending validation at a time; we collect it
    # before launching the next one so saving is always serial.
    pending_val: _PendingVal | None = None
 
    try:
        # Each area produces its own tbar.update(1) per completed timeslot
        # (train_ppo_multi_env calls it once per area, not once per
        # "aggregate" timeslot) — so the total must scale by len(envs), or
        # tqdm's total is exhausted partway into episode 0 for multi-area
        # runs, and every episode after that falls back to open-ended
        # "N it [...]" counting instead of a proper progress bar.
        tbar = tqdm(
            range(int(params["total_timeslots"] * params["num_episodes"] * len(envs))),
            desc="Training computation is starting ",
            position=0,
            leave=True,
            dynamic_ncols=True
        )
 
        logger.info(f"Training started with the following parameters: {params}")
 
        # Train loop
        for episode in range(start_episode, start_episode + int(params["num_episodes"])):
            current_seed = int(params['seed'] + episode)
 
            # Linear entropy-coefficient decay across episodes (see
            # ent_coef_final in the params dict above for rationale). With
            # num_episodes == 1 this just uses the starting ent_coef.
            num_episodes = int(params["num_episodes"])
            decay_frac = episode / max(1, num_episodes - 1)
            current_ent_coef = (
                params["ent_coef"]
                + decay_frac * (params["ent_coef_final"] - params["ent_coef"])
            )
            agent.set_ent_coef(current_ent_coef)
 
            # PPO Training
            if data_paths and len(envs) > 1:
                episode_results_paths = [
                    os.path.join(str(results_manager.training_path), f"area_{i}", f"episode_{episode:03d}")
                    for i in range(len(envs))
                ]
                training_dict = train_ppo_multi_env(
                    envs=envs,
                    agent=agent,
                    buffer=ppo_buffer,
                    episode=episode,
                    device=device,
                    run_id=run_id,
                    logging_enabled=logging_enabled,
                    tbar=tbar,
                    episode_results_paths=episode_results_paths,
                    seed=current_seed
                )
            else:
                training_dict = train_ppo(
                    env=env,
                    agent=agent,
                    buffer=ppo_buffer,
                    episode=episode,
                    device=device,
                    run_id=run_id,
                    logging_enabled=logging_enabled,
                    tbar=tbar,
                    episode_results_path=os.path.join(str(results_manager.training_path), f"episode_{episode:03d}"),
                    seed=current_seed
                )
 
            # Build EpisodeResults
            training_results = EpisodeResults(
                episode=episode,
                mode='train',
                seed=current_seed,
                epsilon=0.0, # In PPO, exploration is given by entropy
                rewards_per_timeslot=training_dict['rewards_per_timeslot'],
                demand_per_timeslot=training_dict['demand_per_timeslot'],
                total_reward=sum(training_dict['rewards_per_timeslot']),
                failures_per_timeslot=training_dict['failures_per_timeslot'],
                total_failures=sum(training_dict['failures_per_timeslot']),
                mean_daily_failures=sum(training_dict['failures_per_timeslot']) / num_days,
                state_values_per_timeslot=training_dict.get('state_values_per_timeslot', []),
                mean_state_values=float(np.mean(training_dict.get('state_values_per_timeslot', []))) if training_dict.get('state_values_per_timeslot') else 0.0,
                deployed_bikes=training_dict['deployed_bikes'],
                truck_load=training_dict['truck_load'],
                depot_load=training_dict['depot_load'],
                outside_system_bikes=training_dict['outside_system_bikes'],
                action_per_step=training_dict['action_per_step'],
                total_invalid_actions=training_dict['total_invalid_actions'],
                reward_tracking_per_action=training_dict['reward_tracking_per_action'],
                global_critic_scores=training_dict['global_critic_scores'],
                cell_subgraph=training_dict['cell_subgraph'],
                per_area=training_dict.get('per_area'),
                traveling_bikes=training_dict['traveling_bikes'],
                policy_loss=training_dict['policy_loss'],
                value_loss=training_dict['value_loss'],
                entropy=training_dict['entropy']
            )
 
            # Save training episode results
            results_manager.save_episode(training_results)
 
            current_epsilon = getattr(agent, 'epsilon', 0.0)
            if current_epsilon < params['validation_epsilon_threshold']:
            #if False: # to not have validation for now
                # ── Step A: collect the previous val subprocess (if any) ──────
                # This is the only point where training may briefly wait, and
                # only if val_N-1 hasn't finished by the time train_N is done.
                if pending_val is not None:
                    val_ok = _collect_pending_val(pending_val, logger, int(params['validation_timeout']))
                    if val_ok:
                        val_score = _read_validation_score(
                            results_path=results_path,
                            run_id=run_id,
                            episode=pending_val.episode,
                            logger=logger,
                        )
                        if val_score is not None and val_score < best_val_score:
                            prev_best = best_val_score
                            best_val_score = val_score
                            # Promote the already-saved episode snapshot — do NOT
                            # use the live agent weights (we are now one episode ahead).
                            results_manager.promote_episode_to_best(
                                episode=pending_val.episode,
                                score=val_score,
                            )
                            logger.info(
                                f"[val] Episode {pending_val.episode}: NEW BEST promoted! "
                                f"val_score={val_score:.4f} (prev best={prev_best:.4f})"
                            )
                        elif val_score is not None:
                            logger.info(
                                f"[val] Episode {pending_val.episode}: "
                                f"val_score={val_score:.4f} did not beat best={best_val_score:.4f}"
                            )
                    pending_val = None
 
                # ── Step B: save this episode's model snapshot ────────────────
                # Uses this episode's own training score as metadata.
                # The snapshot is what the validator will load.
                results_manager.save_model(
                    agent=agent,
                    episode=episode,
                    score=training_results.mean_daily_failures,
                    model_type='episode'
                )
                logger.info(f"Episode {episode}: model snapshot saved (epsilon={current_epsilon:.4f})")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
 
                # ── Step C: launch the new val subprocess (non-blocking) ──────
                val_cmd = _build_validate_cmd(
                    run_id=run_id,
                    data_path=data_path,
                    results_path=results_path,
                    episode=episode,
                    val_device=str(val_device),
                    seed=int(params['seed']),
                    max_num_bikes=int(params['maximum_number_of_bikes']),
                    min_num_bikes=int(params['minimum_number_of_bikes']),
                    total_timeslots=int(params['total_timeslots']),
                    enable_repositioning=bool(params['enable_repositioning']),
                    use_net_flow=bool(params['use_net_flow']),
                    data_paths=data_paths,
                )
                pending_val = _launch_validation_subprocess(val_cmd, episode, logger)
 
            logger.info(
                f"Episode {episode}: Seed = {current_seed}, "
                f"Mean Failures = {training_results.mean_daily_failures:.2f}, "
                f"Total Failures = {training_results.total_failures}, "
                f"Invalid Actions = {training_results.total_invalid_actions}, "
                f"Epsilon = {current_epsilon:.4f}"
            )
 
            gc.collect()
 
        # ------------------------------------------------------------------
        # End of training — collect any still-running validation
        # ------------------------------------------------------------------
        if pending_val is not None:
            print(f"\n[VAL] Training finished. Waiting for last validation (episode {pending_val.episode})...")
            val_ok = _collect_pending_val(pending_val, logger, int(params['validation_timeout']))
            if val_ok:
                val_score = _read_validation_score(
                    results_path=results_path,
                    run_id=run_id,
                    episode=pending_val.episode,
                    logger=logger,
                )
                if val_score is not None and val_score < best_val_score:
                    prev_best = best_val_score
                    best_val_score = val_score
                    results_manager.promote_episode_to_best(
                        episode=pending_val.episode,
                        score=val_score,
                    )
                    logger.info(
                        f"[val] Final best: episode {pending_val.episode}, "
                        f"val_score={val_score:.4f} (prev best={prev_best:.4f})"
                    )
                    print(
                        f"\n[VAL] ✓ Final best model: episode {pending_val.episode} "
                        f"— mean_daily_failures={val_score:.4f}"
                    )
 
        # Save aggregated summaries
        results_manager.save_run_summary()
        logger.info("Training completed successfully")
        tbar.close()
        for _e in envs:
            _e.close()
        
        # Save closure to avoid leaked semaphore
        gc.collect()
        try:
            from loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True, kill=True)
        except Exception:
            pass
            
        try:
            for p in mp.active_children():
                p.terminate()
                p.join()
        except Exception:
            pass
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        if pending_val is not None and pending_val.proc.poll() is None:
            print(f"[VAL] Terminating background validation for episode {pending_val.episode}...")
            pending_val.proc.terminate()
        for _e in envs:
            _e.close()
        return
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if pending_val is not None and pending_val.proc.poll() is None:
            pending_val.proc.terminate()
        for _e in envs:
            _e.close()
        raise
 
    print(f"\nTraining {run_id} completed.")
    if best_val_score != float("inf"):
        print(f"Best validation score (mean_daily_failures): {best_val_score:.4f}")
    else:
        print("Best validation score: N/A (Disabled validation in this run)")
 
if __name__ == "__main__":
    print("1. Entered in the main section")
    main()