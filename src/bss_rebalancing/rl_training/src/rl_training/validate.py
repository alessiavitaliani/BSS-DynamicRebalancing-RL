"""
Standalone validation script for trained DQN agents.
Mirrors the validate_dqn() / _validation_worker() logic from train.py exactly.
"""

import os
import argparse
import warnings
import logging
import torch
import numpy as np
from dataclasses import dataclass

import gymnasium_env  # noqa: F401 — registers the gym environment
import gymnasium
import gymnasium as gym

from tqdm import tqdm
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions
from gymnasium_env.envs.fully_dynamic_env import EnvDefaults, RewardComponents

from rl_training.agents import PPOAgent
from rl_training.networks.ppo import PPO as PPONetwork
from rl_training.results import ResultsManager, EpisodeResults
from rl_training.logging_config import init_logging, LoggingConfig, get_logger
from rl_training.utils import (
    convert_graph_to_data,
    convert_seconds_to_hours_minutes,
    set_seed,
    setup_device,
    build_cell_graph_from_cells,
    update_cell_graph_features,
)

# ------------------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------------------

devices = ["cpu"]
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")
if torch.backends.mps.is_available():
    devices.append("mps")

# print(f"Devices available: {devices}\n")

# ------------------------------------------------------------------------------
# Default params
# ------------------------------------------------------------------------------

params = {
    "seed": 42,
    "total_timeslots": 56,
    "maximum_number_of_bikes": 1000,
    "minimum_number_of_bikes": 5,
    "gamma": 0.95,
    "lr": 2.0e-5,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.02,
    "vf_coef": 0.25,
    "update_epochs": 6,
    "enable_repositioning": True,
    "use_net_flow": True,
    "depot_position_id": 12,
    "initial_cell_id": 12,
    "num_trucks": 1,
}

reward_params = {
    "W_ZERO_BIKES": 1.0,
    "W_CRITICAL_ZONES": 1.0,
    "W_DROP_PICKUP": 0.9,
    "W_MOVEMENT": 0.7,
    "W_CHARGE_BIKE": 0.9,
    "W_STAY": 0.7,
}

gnn_features = [
    'truck_cell',
    'active_truck_cell',
    'critic_score',
    'eligibility_score',
    'total_bikes',
]

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BSS Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate the best model from run 0
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type best

  # Validate a specific episode
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type episode --model-episode 42

  # Validate an arbitrary .pt file
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-path results/run_000/models/final/trained_agent.pt

  # Override number of bikes and use GPU
  bss-validate --run-id 0 --data-path data/ --max-num-bikes 300 --device cuda:0

  # Non-interactive mode (used by training subprocess — no stdin prompts, fail fast)
  bss-validate --run-id 0 --data-path data/ --model-type episode --model-episode 42 --non-interactive
        """,
    )

    # --- run / paths ---
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help="Run ID whose models/results to use."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to the data folder. Ignored if --data-paths is given."
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        default=None,
        help=(
            'Comma-separated list of data folder paths for multi-area validation '
            '(e.g. "data/manhattan_north,data/manhattan_south"), matching how the '
            'model was trained with --data-paths in train.py. Overrides --data-path.'
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/",
        help="Path to the results folder (same root used during training)."
    )

    # --- model source (mutually exclusive-ish: either --model-type or --model-path) ---
    parser.add_argument(
        "--model-type",
        type=str,
        default="best",
        choices=["best", "episode", "final"],
        help="Which saved model to load from the run directory (default: best).",
    )
    parser.add_argument(
        "--model-episode",
        type=int,
        default=None,
        help="Episode number to load when --model-type=episode.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Direct path to a trained_agent.pt file. Takes priority over --model-type.",
    )

    # --- environment overrides ---
    parser.add_argument(
        "--max-num-bikes",
        type=int,
        default=params["maximum_number_of_bikes"],
        help="Maximum number of bikes in the system."
    )
    parser.add_argument(
        "--min-num-bikes",
        type=int,
        default=params["minimum_number_of_bikes"],
        help="Minimum number of bikes per cell."
    )
    parser.add_argument(
        "--total-timeslots",
        type=int,
        default=params["total_timeslots"],
        help="Total timeslots for the validation episode (default: 56 = 1 week)."
    )
    parser.add_argument(
        "--enable-repositioning",
        action="store_true",
        help="Enable repositioning at the start of the episode."
    )
    parser.add_argument(
        "--use-net-flow",
        action="store_true",
        help="Use net-flow repositioning strategy."
    )

    # --- misc ---
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=f"Hardware device. Available: {devices}."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=params["seed"],
        help="Random seed."
    )
    parser.add_argument(
        "--num-seed-runs",
        type=int,
        default=1,
        help="Number of validation runs with incremented seeds (default: 1)."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable environment logging."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Disable interactive stdin prompts. "
            "Used when validate.py is spawned as a subprocess by train.py. "
            "Causes ResultsManager to raise immediately if the output directory already "
            "exists (instead of asking the user), so the training loop can detect the "
            "failure rather than blocking forever waiting for input."
        ),
    )

    return parser

# ------------------------------------------------------------------------------
# validate_dqn
# ------------------------------------------------------------------------------

def validate_ppo(
        env,
        agent: PPOAgent,
        episode: int,
        device: torch.device,
        run_id: int,
        logging_enabled: bool,
        logger: logging.Logger,
        tbar: tqdm,
        episode_results_path: str | None = None,
        seed: int = None,
) -> dict:
    # -------------------------------------------------------------------------
    # Metrics tracking
    # -------------------------------------------------------------------------
    rewards = []
    failures = []
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    demand_per_timeslot = []
    traveling_bikes = []
    state_values = []

    action_per_step = []
    global_critic_scores = []
    reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}

    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    last_cumulative_demand = 0
    iterations = 0

    # -------------------------------------------------------------------------
    # Environment reset
    # -------------------------------------------------------------------------
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': params["minimum_number_of_bikes"],
        'enable_repositioning': params["enable_repositioning"],
        'use_net_flow': params["use_net_flow"],
        'discount_factor': params["gamma"],
        'depot_id': params['depot_position_id'],
        'num_trucks': params['num_trucks'],
        # 'initial_cell': params['initial_cell_id'],
    }
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path

    agent_state, info = env.reset(seed=seed, options=reset_options)

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

    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # ============================================================================
    # Main validation loop
    # ============================================================================
    episode_cell_stats = {
        cell_id: {
            'critic_sum': 0.0,
            'bikes_sum': 0.0,
            'bikes_dead_sum': 0.0,
        }
        for cell_id in cell_dict.keys()
    }

    done = False
    while not done:
        # ── State (S) → device ──────────────────────────────────────────────────
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # ── Action (A) selection ────────────────────────────────────────────────
        avoid_actions = info.get("avoid_action", [])
        action, _, value = agent.select_action(single_state, avoid_action=avoid_actions)

        # ── Environment step ────────────────────────────────────────────────────
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # ── Graph update ────────────────────────────────────────────────────────
        cell_dict = info['cell_dict']
        update_cell_graph_features(cell_graph, cell_dict)

        # ── Cell stats accumulation ─────────────────────────────────────────────
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] += cell.get_dead_bikes()

        # ── Build next state ────────────────────────────────────────────────────
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # ── Scalar bookkeeping ──────────────────────────────────────────────────
        action_per_step.append(action)
        reward_tracking_per_action[action].append(reward)
        global_critic_scores.append(info['global_critic_score'])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info['failures'])
        iterations += 1

        if timeslot_terminated:
            timeslots_completed += 1

            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            state_values.append(value.item())
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])

            current = sum(cell.get_total_demand() for cell in cell_dict.values())
            demand_per_timeslot.append(current - last_cumulative_demand)
            last_cumulative_demand = current

            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0

            if tbar is not None:
                tbar.set_description(
                    f"[VAL] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.update(1)

        state = next_state
        del single_state

    torch.cuda.empty_cache()

    # ============================================================================
    # Post-episode cell stats
    # ============================================================================
    steps_in_episode = iterations
    for cell_id, stats in episode_cell_stats.items():
        center_node = cell_dict[cell_id].get_center_node()
        if center_node not in cell_graph.nodes:
            continue

        if steps_in_episode > 0:
            critic_mean = stats.get('critic_sum', 0.0) / steps_in_episode
            bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
            dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
        else:
            critic_mean = bikes_mean = dead_bikes_mean = 0.0

        nx_attrs = cell_graph.nodes[center_node]

        nx_attrs['critic_mean'] = critic_mean
        nx_attrs['failure_sum'] = cell_dict[cell_id].get_failures()
        nx_attrs['failure_rate'] = cell_dict[cell_id].get_failure_rate()
        nx_attrs['visits_sum'] = cell_dict[cell_id].get_visits()
        nx_attrs['ops_sum'] = cell_dict[cell_id].get_ops()
        nx_attrs['pick_ups_sum'] = cell_dict[cell_id].get_pick_ups()
        nx_attrs['drops_sum'] = cell_dict[cell_id].get_drops()
        nx_attrs['success_rebalancing'] = cell_dict[cell_id].get_total_rebalanced()
        nx_attrs['bikes_mean'] = bikes_mean
        nx_attrs['dead_bikes_mean'] = dead_bikes_mean

    # ============================================================================
    # Return results
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards,
        "failures_per_timeslot": failures,
        "total_invalid_actions": info["total_invalid_actions"],
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
        "state_values_per_timeslot": state_values,
    }

# ------------------------------------------------------------------------------
# Multi-area validation (mirrors train.py's train_ppo_multi_env, but the agent
# is frozen here — no buffer, no update — we just evaluate the same shared
# policy on each area and report combined + per-area metrics).
# ------------------------------------------------------------------------------

@dataclass
class _ValEnvContext:
    env: gymnasium.Env
    area_label: str
    cell_dict: dict
    cell_graph: object
    state: Data
    info: dict
    episode_cell_stats: dict
    done: bool = False
    total_reward_per_timeslot: float = 0.0
    total_failures_per_timeslot: int = 0
    last_cumulative_demand: float = 0.0
    iterations: int = 0
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
            self.reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}


def _reset_val_env_context(
    env: gymnasium.Env,
    area_label: str,
    episode_results_path: str | None,
    seed: int,
) -> _ValEnvContext:
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': params["minimum_number_of_bikes"],
        'enable_repositioning': params["enable_repositioning"],
        'use_net_flow': params["use_net_flow"],
        'discount_factor': params["gamma"],
        'depot_id': params['depot_position_id'],
        'num_trucks': 1,
    }
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path

    agent_state, info = env.reset(seed=seed, options=reset_options)

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
        cell_id: {'critic_sum': 0.0, 'bikes_sum': 0.0, 'bikes_dead_sum': 0.0}
        for cell_id in cell_dict.keys()
    }

    return _ValEnvContext(
        env=env, area_label=area_label, cell_dict=cell_dict, cell_graph=cell_graph,
        state=state, info=info, episode_cell_stats=episode_cell_stats,
    )


def validate_ppo_multi_env(
        envs: list,
        agent: PPOAgent,
        episode: int,
        device: torch.device,
        run_id: int,
        logging_enabled: bool,
        logger: logging.Logger,
        tbar: tqdm,
        episode_results_paths: list | None = None,
        seed: int = None,
) -> dict:
    """Multi-area counterpart of validate_ppo(): evaluates the SAME frozen
    shared policy on several areas at once (round-robin stepping, no
    training), returning combined + per-area metrics. See
    train_ppo_multi_env() in train.py for the analogous training version."""
    if episode_results_paths is None:
        episode_results_paths = [None] * len(envs)

    contexts = [
        _reset_val_env_context(env, f"area_{i}", episode_results_paths[i], seed)
        for i, env in enumerate(envs)
    ]

    active = list(range(len(contexts)))
    rr_pos = 0
    value = None

    while active:
        idx = active[rr_pos % len(active)]
        ctx = contexts[idx]

        single_state = Data(
            x=ctx.state.x.to(device),
            edge_index=ctx.state.edge_index.to(device),
            edge_attr=ctx.state.edge_attr.to(device),
            batch=torch.zeros(ctx.state.x.size(0), dtype=torch.long).to(device),
        )

        avoid_actions = ctx.info.get("avoid_action", [])
        action, _, value = agent.select_action(single_state, avoid_action=avoid_actions)

        agent_state, reward, done, timeslot_terminated, info = ctx.env.step(action)

        cell_dict = info['cell_dict']
        update_cell_graph_features(ctx.cell_graph, cell_dict)
        ctx.cell_dict = cell_dict
        ctx.info = info

        for cell_id, cell in cell_dict.items():
            stats = ctx.episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] += cell.get_dead_bikes()

        next_state = convert_graph_to_data(ctx.cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        ctx.action_per_step.append(action)
        ctx.reward_tracking_per_action.setdefault(action, []).append(reward)
        ctx.global_critic_scores.append(info['global_critic_score'])
        ctx.total_reward_per_timeslot += reward
        ctx.total_failures_per_timeslot += sum(info['failures'])
        ctx.iterations += 1

        if timeslot_terminated:
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
                    f"[VAL] Run {run_id}. Epis {episode}, {ctx.area_label}, "
                    f"Week {info['week'] % 52}, {info['day'].capitalize()} "
                    f"at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.update(1)

        ctx.state = next_state
        ctx.done = done

        if done:
            active.remove(idx)
            if active:
                rr_pos = rr_pos % len(active)
        else:
            rr_pos += 1

    torch.cuda.empty_cache()

    for ctx in contexts:
        steps_in_episode = ctx.iterations
        for cell_id, stats in ctx.episode_cell_stats.items():
            center_node = ctx.cell_dict[cell_id].get_center_node()
            if center_node not in ctx.cell_graph.nodes:
                continue
            if steps_in_episode > 0:
                critic_mean = stats['critic_sum'] / steps_in_episode
                bikes_mean = stats['bikes_sum'] / steps_in_episode
                dead_bikes_mean = stats['bikes_dead_sum'] / steps_in_episode
            else:
                critic_mean = bikes_mean = dead_bikes_mean = 0.0
            nx_attrs = ctx.cell_graph.nodes[center_node]
            nx_attrs['critic_mean'] = critic_mean
            nx_attrs['failure_sum'] = ctx.cell_dict[cell_id].get_failures()
            nx_attrs['failure_rate'] = ctx.cell_dict[cell_id].get_failure_rate()
            nx_attrs['visits_sum'] = ctx.cell_dict[cell_id].get_visits()
            nx_attrs['ops_sum'] = ctx.cell_dict[cell_id].get_ops()
            nx_attrs['pick_ups_sum'] = ctx.cell_dict[cell_id].get_pick_ups()
            nx_attrs['drops_sum'] = ctx.cell_dict[cell_id].get_drops()
            nx_attrs['success_rebalancing'] = ctx.cell_dict[cell_id].get_total_rebalanced()
            nx_attrs['bikes_mean'] = bikes_mean
            nx_attrs['dead_bikes_mean'] = dead_bikes_mean

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
        "action_per_step": _concat("action_per_step"),
        "global_critic_scores": _concat("global_critic_scores"),
        "reward_tracking_per_action": merged_reward_tracking,
        "deployed_bikes": _concat("system_bikes"),
        "truck_load": _concat("truck_load"),
        "depot_load": _concat("depot_load"),
        "outside_system_bikes": _concat("outside_system_bikes"),
        "traveling_bikes": _concat("traveling_bikes"),
        "demand_per_timeslot": _concat("demand_per_timeslot"),
        "cell_subgraph": contexts[0].cell_graph,  # see limitation note in train_ppo_multi_env
        "state_values_per_timeslot": _concat("state_values"),
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

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    args = create_parser().parse_args()

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------
    run_id = args.run_id
    data_path = args.data_path
    data_paths = [p.strip() for p in args.data_paths.split(",")] if args.data_paths else None
    results_path = args.results_path
    logging_enabled = args.log
    non_interactive = args.non_interactive

    device = setup_device(args.device.lower(), devices, non_interactive=non_interactive)

    params["seed"] = args.seed
    params["num_seed_runs"] = args.num_seed_runs
    params["total_timeslots"] = args.total_timeslots
    params["maximum_number_of_bikes"] = args.max_num_bikes
    params["minimum_number_of_bikes"] = args.min_num_bikes
    params["enable_repositioning"] = args.enable_repositioning
    params["use_net_flow"] = args.use_net_flow

    set_seed(params["seed"])

    if data_paths:
        for p in data_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Data path not found: {p}")
    elif not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # ------------------------------------------------------------------
    # ResultsManager
    # ------------------------------------------------------------------
    if args.model_path is None:
        val_tag = ResultsManager.build_val_tag(args.model_type, args.model_episode)
    else:
        val_tag = "custom"

    results_manager = ResultsManager(
        results_path=results_path,
        run_id=run_id,
        overwrite=non_interactive,
        interactive=not non_interactive,
        val_tag=val_tag,
        mode='validation',
    )

    # Init logging
    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(str(results_manager.validation_path), "logs"),
        run_id=run_id,
        console=False,
        logger_name="validate",
    ))
    logger = get_logger("validate", logger_name="validate")
    logger.info("Starting validation loop")

    # ------------------------------------------------------------------
    # Resolve model_path from CLI args
    # ------------------------------------------------------------------
    if args.model_path is not None:
        # Direct path takes priority
        model_path = args.model_path
        logger.info(f"Using model from direct path: {model_path}")
    else:
        # Derive from run directory via ResultsManager
        if args.model_type == "best":
            model_path = results_manager.get_best_model_path()
            if model_path is None:
                raise FileNotFoundError(f"No best model found in {results_manager.models_path}")
            logger.info(f"Using best model from {model_path}")
        elif args.model_type == "final":
            model_path = results_manager.models_path / "final" / "trained_agent.pt"
            logger.info(f"Using final model from {model_path}")
        elif args.model_type == "episode":
            if args.model_episode is None:
                raise ValueError("--model-episode must be specified when --model-type=episode")
            model_path = (
                    results_manager.models_path
                    / "episodes"
                    / f"episode_{args.model_episode:03d}"
                    / "trained_agent.pt"
            )
            logger.info(f"Using episode model from {model_path}")
        else:
            raise ValueError(f"Unknown --model-type: {args.model_type}")
        model_path = str(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    if data_paths:
        envs = [
            gym.make(
                "gymnasium_env/FullyDynamicEnv-v0",
                data_path=p,
                results_path=f"{str(results_manager.validation_path)}/area_{i}/",
                seed=params["seed"],
                logging_enabled=logging_enabled,
            )
            for i, p in enumerate(data_paths)
        ]
        env = envs[0]
    else:
        env = gym.make(
            "gymnasium_env/FullyDynamicEnv-v0",
            data_path=data_path,
            results_path=f"{str(results_manager.validation_path)}/",
            seed=params["seed"],
            logging_enabled=logging_enabled,
        )
        envs = [env]

    # Save hyperparameters
    results_manager.save_hyperparameters(
        params={
            **params,
            **{k: v for k, v in vars(EnvDefaults).items() if not k.startswith('_')},
            "data_path": data_path,
            "data_paths": data_paths,
        },
        reward_params={k: v for k, v in vars(RewardComponents).items() if not k.startswith('_')}
    )

    if not non_interactive:
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Params: {params}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Agent — frozen, no replay buffer
    # ------------------------------------------------------------------
    network = PPONetwork(num_actions=env.action_space.n, node_features=len(gnn_features)).to(device)
    agent = PPOAgent(
        network=network,
        lr=params.get("lr", 3e-4),
        gamma=params.get("gamma", 0.99),
        gae_lambda=params.get("gae_lambda"),
        clip_coef=params.get("clip_coef"),
        ent_coef=params.get("ent_coef"),
        vf_coef=params.get("vf_coef"),
        update_epochs=params.get("update_epochs"),
        device=device,
    )
    agent.load_model(model_path)
    agent.network.eval()
    if not non_interactive:
        print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    best_score = float("inf")
    # In multi-area mode, per-timeslot series returned by validate_ppo_multi_env()
    # are concatenated across areas, so the denominator for mean_daily_failures
    # must scale with the number of areas too, or the metric is inflated.
    num_days = int(params["total_timeslots"] // 8) * max(1, len(envs))

    try:
        # Same fix as train.py's tbar: each area produces its own
        # tbar.update(1) per completed timeslot, so total must scale by
        # len(envs) or the bar runs out mid-way through episode 0.
        tbar = tqdm(
            range(int(params["total_timeslots"] * params["num_seed_runs"] * len(envs))),
            desc="Validation computation is starting",
            position=1 if non_interactive else 0,
            leave=False if non_interactive else True,
            dynamic_ncols=True,
        )

        logger.info(f"Validation started with the following parameters: {params}")

        for episode in range(int(params["num_seed_runs"])):
            current_seed = int(params["seed"] + episode)
            set_seed(current_seed)

            if data_paths and len(envs) > 1:
                episode_results_paths = [
                    os.path.join(str(results_manager.validation_path), f"area_{i}", f"episode_{episode:03d}")
                    for i in range(len(envs))
                ]
                validation_dict = validate_ppo_multi_env(
                    envs=envs,
                    agent=agent,
                    episode=episode,
                    device=device,
                    run_id=run_id,
                    logging_enabled=logging_enabled,
                    logger=logger,
                    tbar=tbar,
                    episode_results_paths=episode_results_paths,
                    seed=current_seed,
                )
            else:
                validation_dict = validate_ppo(
                    seed=current_seed,
                    env=env,
                    agent=agent,
                    episode=episode,
                    device=device,
                    run_id=run_id,
                    logging_enabled=logging_enabled,
                    logger=logger,
                    tbar=tbar,
                    episode_results_path=os.path.join(str(results_manager.validation_path), f"episode_{episode:03d}"),
                )

            validation_results = EpisodeResults(
                episode=episode,
                mode="validation",
                seed=current_seed,
                epsilon=0.0,   # PPO: nessun epsilon, esplorazione via entropia
                rewards_per_timeslot=validation_dict["rewards_per_timeslot"],
                total_reward=sum(validation_dict["rewards_per_timeslot"]),
                failures_per_timeslot=validation_dict["failures_per_timeslot"],
                total_failures=sum(validation_dict["failures_per_timeslot"]),
                mean_daily_failures=sum(validation_dict["failures_per_timeslot"]) / num_days,
                state_values_per_timeslot=validation_dict.get("state_values_per_timeslot", []),
                mean_state_values=float(np.mean(validation_dict["state_values_per_timeslot"]))
                                if validation_dict.get("state_values_per_timeslot") else 0.0,
                action_per_step=validation_dict["action_per_step"],
                total_invalid_actions=validation_dict["total_invalid_actions"],
                reward_tracking_per_action=validation_dict["reward_tracking_per_action"],
                traveling_bikes=validation_dict['traveling_bikes'],
                deployed_bikes=validation_dict["deployed_bikes"],
                demand_per_timeslot=validation_dict['demand_per_timeslot'],
                truck_load=validation_dict["truck_load"],
                depot_load=validation_dict["depot_load"],
                outside_system_bikes=validation_dict["outside_system_bikes"],
                global_critic_scores=validation_dict.get("global_critic_scores", []),
                cell_subgraph=validation_dict["cell_subgraph"],
                per_area=validation_dict.get("per_area"),
            )

            results_manager.save_episode(validation_results)

            is_best = validation_results.total_failures < best_score
            if is_best:
                best_score = validation_results.total_failures

            logger.info(
                f"Episode {episode} (seed={current_seed}): "
                f"failures={validation_results.total_failures} total / "
                f"{validation_results.mean_daily_failures:.2f} mean daily | "
                f"invalid={validation_results.total_invalid_actions}"
            )

        results_manager.save_run_summary()
        logger.info("Validation completed successfully")
        tbar.close()
        for _e in envs:
            _e.close()
    except KeyboardInterrupt:
        print("\nValidation interrupted.")
        logger.info("Validation interrupted.")
        for _e in envs:
            _e.close()
        return
    except Exception as e:
        logger.error(f"Validation failed: {e}.")
        for _e in envs:
            _e.close()
        raise

    if not non_interactive:
        print(f"\nValidation {run_id} completed.")


if __name__ == "__main__":
    main()