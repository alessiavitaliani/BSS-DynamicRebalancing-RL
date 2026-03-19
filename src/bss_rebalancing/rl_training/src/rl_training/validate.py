"""
Standalone validation script for trained DQN agents.
Mirrors the validate_dqn() / _validation_worker() logic from train.py exactly.
"""

import os
import argparse
import warnings
import logging
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import gymnasium_env  # noqa: F401 — registers the gym environment

from rl_training.agents import DQNAgent
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
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions

# ------------------------------------------------------------------------------
# Device detection (mirrors train.py)
# ------------------------------------------------------------------------------
devices = ["cpu"]
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")
if torch.backends.mps.is_available():
    devices.append("mps")

print(f"Devices available: {devices}\n")

# ------------------------------------------------------------------------------
# Default params — kept in sync with train.py defaults
# ------------------------------------------------------------------------------
params = {
    "seed": 42,
    "total_timeslots": 56,
    "maximum_number_of_bikes": 500,
    "minimum_number_of_bikes": 1,
    "gamma": 0.95,
    "lr": 1e-4,
    "tau": 0.005,
    "enable_repositioning": False,
    "use_net_flow": False,
    "depot_position_id": 18,
    "initial_cell_id": 18,
}

reward_params = {
    "W_ZERO_BIKES": 1.0,
    "W_CRITICAL_ZONES": 1.0,
    "W_DROP_PICKUP": 0.9,
    "W_MOVEMENT": 0.7,
    "W_CHARGE_BIKE": 0.9,
    "W_STAY": 0.7,
}


# ------------------------------------------------------------------------------
# validate_dqn — exact copy of train.py's validate_dqn, nothing removed
# ------------------------------------------------------------------------------
def validate_dqn(
        env,
        agent: DQNAgent,
        episode: int,
        device: torch.device,
        run_id: int,
        epsilon: float,
        logging_enabled: bool,
        tbar=None,
        episode_results_path: str | None = None,
        params_snapshot: dict | None = None,
        reward_params_snapshot: dict | None = None,
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
    traveling_bikes = []

    action_per_step = []
    global_critic_scores = []
    reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}

    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    iterations = 0

    # -------------------------------------------------------------------------
    # Environment reset
    # -------------------------------------------------------------------------
    _params = params_snapshot if params_snapshot is not None else params
    _reward_params = reward_params_snapshot if reward_params_snapshot is not None else reward_params

    reset_options = {
        "total_timeslots": _params["total_timeslots"],
        "maximum_number_of_bikes": _params["maximum_number_of_bikes"],
        "minimum_number_of_bikes": _params["minimum_number_of_bikes"],
        "enable_repositioning": _params["enable_repositioning"],
        "use_net_flow": _params["use_net_flow"],
        "discount_factor": _params["gamma"],
        "depot_id": _params["depot_position_id"],
        "reward_params": _reward_params,
    }
    if episode_results_path is not None:
        reset_options["results_path"] = episode_results_path

    agent_state, info = env.reset(options=reset_options)

    cell_dict = info["cell_dict"]
    nodes_dict = info["nodes_dict"]
    distance_lookup = info["distance_lookup"]

    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict,
        nodes_dict=nodes_dict,
        distance_lookup=distance_lookup,
    )

    gnn_features = [
        "truck_cell",
        "critic_score",
        "eligibility_score",
        "total_bikes",
    ]

    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info["steps"]

    # -------------------------------------------------------------------------
    # Freeze epsilon for validation
    # -------------------------------------------------------------------------
    agent.epsilon = epsilon

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    episode_cell_stats = {
        cell_id: {"critic_sum": 0.0, "eligibility_sum": 0.0, "bikes_sum": 0.0}
        for cell_id in cell_dict.keys()
    }

    done = False
    while not done:
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # Select action and step environment (A) (epsilon=0.05, mostly greedy)
        action = agent.select_action(single_state, epsilon_greedy=True)

        # Step into: get reward and observation (R)
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Only update node attributes
        cell_dict = info['cell_dict']
        update_cell_graph_features(cell_graph, cell_dict)

        # Update cumulative cell statistics (averaged at episode end)
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats["critic_sum"] += cell.get_critic_score()
            stats["eligibility_sum"] += cell.get_eligibility_score()
            stats["bikes_sum"] += cell.get_total_bikes()
            stats["bikes_dead_sum"] = cell.get_dead_bikes()

        # Create next state (S')
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info["steps"]

        # Record step metrics (no training in validation)
        action_per_step.append(action)
        reward_tracking_per_action[action].append(reward)
        global_critic_scores.append(info["global_critic_score"])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info["failures"])
        iterations += 1

        # Handle timeslot completion
        if timeslot_terminated:
            timeslots_completed += 1

            # Record timeslot metrics
            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            system_bikes.append(info["number_of_system_bikes"])
            truck_load.append(info["truck_bikes"])
            depot_load.append(info["depot_bikes"])
            outside_system_bikes.append(info["number_of_outside_bikes"])
            traveling_bikes.append(info["number_of_traveling_bikes"])

            # Reset accumulators
            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0

            # Update progress bar
            if tbar is not None:
                tbar.set_description(
                    f"[VALIDATION] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.set_postfix({"eps": agent.epsilon})
                tbar.update(1)

        # Move to next state
        state = next_state
        del single_state  # Free GPU memory

    # -------------------------------------------------------------------------
    # Post-episode cell stats
    # -------------------------------------------------------------------------
    torch.cuda.empty_cache()

    steps_in_episode = iterations
    for cell_id, stats in episode_cell_stats.items():
        center_node = cell_dict[cell_id].get_center_node()
        if center_node not in cell_graph.nodes:
            continue

        if steps_in_episode > 0:
            critic_mean = stats.get("critic_sum", 0.0) / steps_in_episode
            eligibility_mean = stats.get("eligibility_sum", 0.0) / steps_in_episode
            bikes_mean = stats.get("bikes_sum", 0.0) / steps_in_episode
            dead_bikes_mean = stats.get("bikes_dead_sum", 0.0) / steps_in_episode
        else:
            critic_mean = eligibility_mean = bikes_mean = dead_bikes_mean = 0.0

        nx_attrs = cell_graph.nodes[center_node]
        nx_attrs["critic_mean"] = critic_mean
        nx_attrs["eligibility_mean"] = eligibility_mean
        nx_attrs["failure_sum"] = cell_dict[cell_id].get_failures()
        nx_attrs["failure_rate"] = cell_dict[cell_id].get_failure_rate()
        nx_attrs["visits_sum"] = cell_dict[cell_id].get_visits()
        nx_attrs["ops_sum"] = cell_dict[cell_id].get_ops()
        nx_attrs["bikes_mean"] = bikes_mean
        nx_attrs["dead_bikes_mean"] = dead_bikes_mean

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
        "cell_subgraph": cell_graph,
        # Not tracked during validation — kept for EpisodeResults compatibility
        "q_values_per_timeslot": [],
        "epsilon_per_timeslot": [],
    }


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

  # Validate a specific checkpoint episode
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type checkpoint --model-episode 42

  # Validate an arbitrary .pt file
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-path results/run_000/models/final/trained_agent.pt

  # Override number of bikes and use GPU
  bss-validate --run-id 0 --data-path data/ --max-num-bikes 300 --device cuda:0
        """,
    )

    # --- model source (mutually exclusive-ish: either --model-path or --model-type) ---
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Direct path to a trained_agent.pt file. Takes priority over --model-type.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="best",
        choices=["best", "checkpoint", "final"],
        help="Which saved model to load from the run directory (default: best).",
    )
    parser.add_argument(
        "--model-episode",
        type=int,
        default=None,
        help="Episode number to load when --model-type=checkpoint.",
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
        help="Path to the data folder."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/",
        help="Path to the results folder (same root used during training)."
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
        "--log",
        action="store_true",
        help="Enable environment logging."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of validation episodes to run (default: 1)."
    )

    return parser


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    args = create_parser().parse_args()

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------
    params["seed"] = args.seed
    params["total_timeslots"] = args.total_timeslots
    params["maximum_number_of_bikes"] = args.max_num_bikes
    params["minimum_number_of_bikes"] = args.min_num_bikes
    params["enable_repositioning"] = args.enable_repositioning
    params["use_net_flow"] = args.use_net_flow

    run_id = args.run_id
    data_path = args.data_path
    results_path = args.results_path
    logging_enabled = args.log
    num_episodes = args.num_episodes
    num_days = params["total_timeslots"] // 8

    set_seed(params["seed"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    device = setup_device(args.device.lower(), devices)

    # ------------------------------------------------------------------
    # ResultsManager — reuse the existing run directory so validation
    # results land next to the training results.
    # ------------------------------------------------------------------
    results_manager = ResultsManager(results_path, run_id, overwrite=True, interactive=False)
    results_manager.save_hyperparameters(params, reward_params)

    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(str(results_manager.validation_path), "logs"),
        run_id=run_id,
        console=False,
        logger_name="validate",
    ))
    logger = get_logger("validate", logger_name="validate")
    logger.info(f"Validation started | params={params} | reward_params={reward_params}")

    print("=" * 80)
    print(f"Device: {device}")
    print(f"Params: {params}")
    print(f"Reward params: {reward_params}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Resolve model path
    # ------------------------------------------------------------------
    if args.model_path is not None:
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        # Let ResultsManager resolve it
        if args.model_type == "final":
            model_path = str(results_manager.models_path / "final" / "trained_agent.pt")
        elif args.model_type == "best":
            best = results_manager.get_best_model_path()
            if best is None:
                raise FileNotFoundError(
                    "No best model tracked in this run. "
                    "Use --model-path to point directly at a .pt file."
                )
            model_path = str(best)
        elif args.model_type == "checkpoint":
            if args.model_episode is None:
                raise ValueError("--model-episode is required when --model-type=checkpoint")
            model_path = str(
                results_manager.models_path / "checkpoints"
                / f"episode_{args.model_episode:03d}" / "trained_agent.pt"
            )
        else:
            raise ValueError(f"Unknown model-type: {args.model_type}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Resolved model path not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    print(f"Loading model: {model_path}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = gym.make(
        "gymnasium_env/FullyDynamicEnv-v0",
        data_path=data_path,
        results_path=f"{str(results_manager.validation_path)}/",
        seed=params["seed"],
        logging_enabled=logging_enabled,
    )
    env.unwrapped.seed(params["seed"])
    env.action_space.seed(params["seed"])
    env.observation_space.seed(params["seed"])

    # ------------------------------------------------------------------
    # Agent — frozen, no replay buffer (mirrors _validation_worker)
    # ------------------------------------------------------------------
    agent = DQNAgent(
        num_actions=env.action_space.n,
        observation_space_len=env.observation_space.shape[0],
        gamma=params["gamma"],
        epsilon_start=0.01,
        epsilon_end=0.01,
        epsilon_decay=1,          # no decay
        lr=params["lr"],
        device=device,
        tau=params["tau"],
        soft_update=False,
        replay_buffer=None,
    )
    agent.load_model(model_path)
    agent.train_model.eval()
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    best_score = float("inf")

    try:
        for episode in range(num_episodes):
            val_episode_path = os.path.join(
                str(results_manager.validation_path), f"episode_{episode:03d}"
            )

            tbar = tqdm(
                total=params["total_timeslots"],
                desc=f"[VAL] Episode {episode}",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )

            validation_dict = validate_dqn(
                env=env,
                agent=agent,
                episode=episode,
                device=device,
                run_id=run_id,
                epsilon=agent.epsilon_min,
                logging_enabled=logging_enabled,
                tbar=tbar,
                episode_results_path=val_episode_path,
                params_snapshot=dict(params),
                reward_params_snapshot=dict(reward_params),
            )

            tbar.close()

            # Build EpisodeResults — identical field names to train.py's _collect_pending_validation
            validation_results = EpisodeResults(
                episode=episode,
                mode="validation",
                epsilon=agent.epsilon_min,
                epsilon_per_timeslot=validation_dict.get("epsilon_per_timeslot", []),
                rewards_per_timeslot=validation_dict["rewards_per_timeslot"],
                total_reward=sum(validation_dict["rewards_per_timeslot"]),
                failures_per_timeslot=validation_dict["failures_per_timeslot"],
                total_failures=sum(validation_dict["failures_per_timeslot"]),
                mean_daily_failures=sum(validation_dict["failures_per_timeslot"]) / num_days,
                action_per_step=validation_dict["action_per_step"],
                total_invalid_actions=validation_dict["total_invalid_actions"],
                reward_tracking_per_action=validation_dict["reward_tracking_per_action"],
                q_values_per_timeslot=validation_dict.get("q_values_per_timeslot", []),
                mean_q_values=0.0,
                deployed_bikes=validation_dict["deployed_bikes"],
                truck_load=validation_dict["truck_load"],
                depot_load=validation_dict["depot_load"],
                outside_system_bikes=validation_dict["outside_system_bikes"],
                global_critic_scores=validation_dict.get("global_critic_scores", []),
                cell_subgraph=validation_dict["cell_subgraph"],
            )

            results_manager.save_episode(validation_results)

            is_best = validation_results.total_failures < best_score
            if is_best:
                best_score = validation_results.total_failures

            # Summary
            print("=" * 80)
            print(f"EPISODE {episode} RESULTS")
            print("=" * 80)
            print(f"  Total failures      : {validation_results.total_failures}")
            print(f"  Mean daily failures : {validation_results.mean_daily_failures:.2f}")
            print(f"  Invalid actions     : {validation_results.total_invalid_actions}")
            print(f"  Mean reward/timeslot: {np.mean(validation_results.rewards_per_timeslot):.4f}")
            print(f"  Total reward        : {validation_results.total_reward:.2f}")
            print("=" * 80)

            logger.info(
                f"Episode {episode}: failures={validation_results.total_failures} total / "
                f"{validation_results.mean_daily_failures:.2f} mean daily | "
                f"invalid={validation_results.total_invalid_actions}"
            )

        results_manager.save_run_summary()
        print(f"\nValidation complete. Results saved to {results_manager.validation_path}")

    except KeyboardInterrupt:
        print("\nValidation interrupted.")
        env.close()
        return
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        env.close()
        raise

    env.close()


if __name__ == "__main__":
    main()