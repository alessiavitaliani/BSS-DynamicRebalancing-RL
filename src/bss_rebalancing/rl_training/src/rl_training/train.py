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
    "lr": 2.0e-5,                                   # Learning rate
    # PPO params
    "clip_coef": 0.2,                               # Clipping coefficient 
    "gae_lambda": 0.95,                             # Generalized Advantage Estimation (GAE) factor 
    "ent_coef": 0.02,                               # Entropy coefficient
    "vf_coef": 0.25,                                # Value coefficient
    "update_epochs": 6,                             # How many times buffer is processed at every update 

    "total_timeslots": 56,                  # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 1000,        # Maximum number of bikes in the system
    "minimum_number_of_bikes": 5,           # Minimum number of bikes per cell
    "enable_repositioning": False,          # Use base repositioning strategy at the start of each episode
    "use_net_flow": False,                  # Use net flow repositioning strategy at the start of each episode
    "depot_position_id": 12,                # ID (cell) of the depot position
    "initial_cell_id": 12,                  # Initial cell where the truck starts

    "validation_epsilon_threshold": 0.1,
    "validation_timeout": 600,
}

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
        help='Path to the data folder.'
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
) -> list[str]:
    """
    Build the argv list to invoke validate.py as a completely independent subprocess
    — exactly as if you typed it in your terminal.
    Uses sys.executable so the subprocess runs in the same venv as training.
    """
    validate_script = str(_get_validate_script_path())
    cmd = [
        sys.executable, validate_script,
        "--run-id", str(run_id),
        "--data-path", data_path,
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
        'initial_cell': params['initial_cell_id'],
    }
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
    
    # Define which metrics to use as GNN features
    gnn_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'total_bikes',
    ]

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
    pg_loss, v_loss, ent_loss = 0.0, 0.0, 0.0
    buffer.clear()
    
    while not done:
        # Prepare state for agent (S)
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
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
                    agent_state=torch.tensor(next_state.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
                    batch=torch.zeros(next_state.x.size(0), dtype=torch.long).to(device),
                )
                
                with torch.no_grad():
                    # Interroghiamo la rete per avere il valore del next_state
                    _, _, next_value = agent.select_action(next_state_data)
                    last_value = next_value.item()
            
            # Perform the update 
            pg_loss, v_loss, ent_loss = agent.update(buffer, last_value=last_value)
            
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
                    f"{info['day'].capitalize()} at {info.get('time_formatted', 'N/A')}"
                )
                # Replaced epsilon with the Critic's Value estimation
                tbar.set_postfix({'Val': f"{value.item():.2f}"})
                tbar.update(1)

        # Move to next state
        state = next_state
        #del single_state

    if len(buffer) > 0:
        pg_loss, v_loss, ent_loss = agent.update(buffer, last_value=0.0)
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
        "entropy": ent_loss,
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
    results_path = args.results_path
    logging_enabled = args.log

    params['seed'] = args.seed
    params['num_episodes'] = args.num_episodes
    params['maximum_number_of_bikes'] = args.max_num_bikes
    params['minimum_number_of_bikes'] = args.min_num_bikes
    params['enable_repositioning'] = args.enable_repositioning
    params['use_net_flow'] = args.use_net_flow
    params['exploration_time'] = args.exploration_time

    print(f"Setting seed: {params['seed']}")
    set_seed(params['seed'])

    # Ensure the data path exists
    if not os.path.exists(data_path):
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
    # Create the environment
    env = gym.make(
        'gymnasium_env/FullyDynamicEnv-v0',
        data_path=data_path,
        results_path=f"{str(results_manager.training_path)}/",
        seed=params['seed'],
        logging_enabled=logging_enabled
    )
    
    print("4. Environment created")

    # Save hyperparameters
    results_manager.save_hyperparameters(
        params={
            **params,
            **{k: v for k, v in vars(EnvDefaults).items() if not k.startswith('_')},
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
        node_features=4 
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
        device=device
    )
    print("Model initialized successfully.\n")


    # Train the agent using the training loop
    num_days = int(params["total_timeslots"] // 8)

    # Best validation score tracker (lower mean_daily_failures = better)
    best_val_score = float("inf")

    # Holds the currently running validation subprocess (if any).
    # There is at most one pending validation at a time; we collect it
    # before launching the next one so saving is always serial.
    pending_val: _PendingVal | None = None

    try:
        tbar = tqdm(
            range(int(params["total_timeslots"] * params["num_episodes"])),
            desc="Training computation is starting ",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        logger.info(f"Training started with the following parameters: {params}")

        # Train loop
        for episode in range(int(params["num_episodes"])):
            current_seed = int(params['seed'] + episode)
            
            # PPO Training
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
        env.close()
        
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
        env.close()
        return
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if pending_val is not None and pending_val.proc.poll() is None:
            pending_val.proc.terminate()
        env.close()
        raise

    print(f"\nTraining {run_id} completed.")
    if best_val_score != float("inf"):
        print(f"Best validation score (mean_daily_failures): {best_val_score:.4f}")
    else:
        print("Best validation score: N/A (Disabled validation in this run)")

if __name__ == "__main__":
    print("1. Entered in the main section")
    main()