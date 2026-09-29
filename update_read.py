#!/usr/bin/env python3
"""
update_readmes.py

Applies the README fixes identified during the code/README consistency
review of BSS-DynamicRebalancing-RL, to reflect the migration from
DQN/DDQN to PPO (and the multi-area / multi-truck training setup).

Files touched (relative to --repo-root):
  - README.md
  - src/bss_rebalancing/rl_training/README.md
  - src/bss_rebalancing/results_webapp/README.md

Usage:
    python update_readmes.py --repo-root /path/to/BSS-DynamicRebalancing-RL
    python update_readmes.py --repo-root . --dry-run   # preview only, no writes

Each replacement is matched on an exact substring. If a substring is not
found (e.g. because the file has already been edited, or line endings
differ), the script reports it and skips that specific edit instead of
failing silently — so re-running after a partial manual edit is safe.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Replacement definitions: (relative_path, old_text, new_text, description)
# `old_text` / `new_text` use plain "\n" — CRLF handling is done on I/O.
# ---------------------------------------------------------------------------

Replacement = tuple[str, str, str, str]

REPLACEMENTS: list[Replacement] = [
    # =======================================================================
    # README.md (root)
    # =======================================================================
    (
        "README.md",
        "It presents a novel framework for dynamically rebalancing bikes in a "
        "dockless **Bike Sharing System (BSS)** using a **Double Deep Q-Network "
        "(DDQN)** trained in a realistic, event-driven simulation environment.",
        "It presents a novel framework for dynamically rebalancing bikes in a "
        "dockless **Bike Sharing System (BSS)** using **Proximal Policy "
        "Optimization (PPO)** trained in a realistic, event-driven simulation "
        "environment.",
        "Intro: DDQN -> PPO",
    ),
    (
        "README.md",
        "- 🧠 A **DDQN agent** learns to make rebalancing decisions under uncertainty",
        "- 🧠 A **PPO agent** (GAT-based actor-critic) learns to make rebalancing "
        "decisions under uncertainty, sharing one policy across multiple "
        "trucks/areas",
        "Highlights bullet: DDQN -> PPO",
    ),
    (
        "README.md",
        "        │       ├── agents/            # DQN agent implementation",
        "        │       ├── agents/            # PPO agent implementation",
        "Folder tree: agents/ comment",
    ),
    (
        "README.md",
        "        │       ├── memory/            # Replay buffer",
        "        │       ├── memory/            # On-policy rollout buffer",
        "Folder tree: memory/ comment",
    ),
    (
        "README.md",
        "Train a DDQN agent with the fully dynamic environment:",
        "Train a PPO agent with the fully dynamic environment:",
        "Section 2 intro sentence",
    ),
    (
        "README.md",
        "### `rl_training`\n"
        "DDQN agent implementation with training and validation pipelines.\n"
        "- **CLI**: `bss-train`, `bss-validate`\n"
        "- **Key modules**: `DQNAgent`, `ReplayBuffer`, `DuelingDQN`",
        "### `rl_training`\n"
        "PPO agent implementation (GAT-based actor-critic) with training and "
        "validation pipelines.\n"
        "- **CLI**: `bss-train`, `bss-validate`\n"
        "- **Key modules**: `PPOAgent`, `PPOBuffer`, `PPO`",
        "Package summary: rl_training",
    ),
    (
        "README.md",
        "The DDQN agent demonstrated:",
        "The PPO agent demonstrated:",
        "Results Summary intro",
    ),
    (
        "README.md",
        "bss-validate \\\n"
        "    --model-path results/run_000/models/best_model.pt \\\n"
        "    --data-path data/ \\\n"
        "    --epsilon 0.05 \\\n"
        "    --total-timeslots 56\n"
        "```\n"
        "\n"
        "**Key arguments**:\n"
        "- `--model-path`: Path to trained model (required)\n"
        "- `--epsilon`: Exploration rate for validation (default: 0.05)\n"
        "- `--total-timeslots`: Episode length (default: 56 = 1 week)\n"
        "- `--run-id`: Validation run identifier (default: 999)",
        "bss-validate \\\n"
        "    --model-path results/run_000/models/best_model.pt \\\n"
        "    --data-path data/ \\\n"
        "    --total-timeslots 56\n"
        "```\n"
        "\n"
        "**Key arguments**:\n"
        "- `--model-path`: Path to trained model (required)\n"
        "- `--total-timeslots`: Episode length (default: 56 = 1 week)\n"
        "- `--run-id`: Validation run identifier (default: 999)\n"
        "\n"
        "(PPO has no exploration-rate flag: unlike the old DQN agent, there is "
        "no `--epsilon`, since PPO's stochastic policy is used directly for "
        "validation.)",
        "Section 3 (Validate) example + key arguments — removes non-existent --epsilon flag",
    ),
    (
        "README.md",
        "- 📊 Real-time metrics (failures, rewards, epsilon decay)",
        "- 📊 Real-time metrics (failures, rewards, policy entropy)",
        "Section 4 features bullet",
    ),

    # =======================================================================
    # src/bss_rebalancing/rl_training/README.md
    # =======================================================================
    (
        "src/bss_rebalancing/rl_training/README.md",
        "Deep Reinforcement Learning training framework for bike-sharing system "
        "dynamic rebalancing using Graph Attention Networks (GAT) and Deep "
        "Q-Networks (DQN).",
        "Deep Reinforcement Learning training framework for bike-sharing system "
        "dynamic rebalancing using Graph Attention Networks (GAT) and Proximal "
        "Policy Optimization (PPO).",
        "Top description: DQN -> PPO",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "- **DQN Agent** with Graph Attention Networks for spatial reasoning\n"
        "- **Experience Replay Buffer** optimized for graph-structured transitions\n"
        "- **Results Management** with structured logging and model checkpointing\n"
        "- **CLI Tools** for training and validation",
        "- **PPO Agent** (GAT-based actor-critic) with Graph Attention Networks "
        "for spatial reasoning\n"
        "- **On-policy rollout buffer** (`PPOBuffer`), cleared after every update\n"
        "- **Results Management** with structured logging and model checkpointing\n"
        "- **CLI Tools** for training and validation, including multi-area "
        "training with a single shared policy",
        "Top bullet list",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "        ├── agents/\n"
        "        │   ├── __init__.py\n"
        "        │   └── dqn_agent.py        # DQN agent implementation\n"
        "        ├── memory/\n"
        "        │   ├── __init__.py\n"
        "        │   └── replay_buffer.py    # Experience replay buffer\n"
        "        ├── networks/\n"
        "        │   ├── __init__.py\n"
        "        │   └── dqn.py              # GAT-based DQN architecture",
        "        ├── agents/\n"
        "        │   ├── __init__.py\n"
        "        │   └── ppo_agent.py        # PPO agent implementation\n"
        "        ├── memory/\n"
        "        │   ├── __init__.py\n"
        "        │   └── ppo_buffer.py       # On-policy rollout buffer\n"
        "        ├── networks/\n"
        "        │   ├── __init__.py\n"
        "        │   └── ppo.py              # GAT-based actor-critic (PPO)",
        "Folder tree",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "### DQN Network\n"
        "\n"
        "Graph Attention Network (GAT) based Q-network with:\n"
        "\n"
        "#### Graph Encoder\n"
        "- **Layer 1**: 4 input features → 64 features (4 heads, concat) → 256\n"
        "- **Layer 2**: 256 → 64 features (4 heads, concat) → 256\n"
        "- **Layer 3**: 256 → 128 features (2 heads, concat) → 256\n"
        "\n"
        "#### Global Pooling\n"
        "- **GlobalAttention**: Attention-based graph-level aggregation → 256\n"
        "\n"
        "#### Graph Embedding\n"
        "- FC layers: 256 → 256 → 128 → 64\n"
        "\n"
        "#### Agent State Encoder\n"
        "- Input: 162-dimensional agent state (truck load, position, action history)\n"
        "- FC layers: 162 → 256 → 256 → 128 → 64\n"
        "\n"
        "#### Fusion and Q-Values\n"
        "- Concatenate: graph embedding (64) + agent embedding (64) → 128\n"
        "- FC layers: 128 → 256 → 128 → 8 (Q-values for 8 actions)",
        "### PPO Network\n"
        "\n"
        "Graph Attention Network (GAT) based actor-critic with a shared encoder:\n"
        "\n"
        "#### Graph Encoder (shared by actor and critic)\n"
        "- **Layer 1** (`GATv2Conv`): `node_features` (5 by default — see "
        "`gnn_features` in `train.py`) → 64 features × 4 heads (concat) → 256\n"
        "- **Layer 2** (`GATv2Conv`): 256 → 64 features × 4 heads (concat) → 256\n"
        "- **Global mean pooling** → 256-d graph embedding\n"
        "\n"
        "#### Critic Head\n"
        "- FC layers: 256 → 128 → 64 → 1 (state value V(s))\n"
        "\n"
        "#### Actor Head\n"
        "- FC layers: 256 → 128 → 64 → `num_actions` (action logits, masked "
        "for invalid moves before sampling)",
        "Architecture section (network)",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "### DQN Agent\n"
        "\n"
        "Features:\n"
        "- **Epsilon-greedy exploration** with exponential decay\n"
        "- **Experience replay** with graph-structured transitions\n"
        "- **Target network** with soft updates (τ=0.005)\n"
        "- **Action masking** to prevent invalid moves\n"
        "- **Gradient clipping** (max_norm=10.0)\n"
        "- **Smooth L1 loss** (Huber loss)\n"
        "\n"
        "### Replay Buffer",
        "### PPO Agent\n"
        "\n"
        "Features:\n"
        "- **Stochastic policy exploration** via the Actor's categorical "
        "distribution entropy (no epsilon-greedy)\n"
        "- **GAE** (Generalized Advantage Estimation) for advantage/return "
        "computation\n"
        "- **Clipped surrogate objective** (`clip_coef`, default 0.2) instead "
        "of a target network\n"
        "- **Action masking** to prevent invalid moves\n"
        "- **Gradient clipping** (max_norm=0.5)\n"
        "- **Combined loss**: clipped policy loss + value loss (MSE) − "
        "entropy bonus\n"
        "- **Optimizer**: Adam (default) or SGD with Nesterov momentum\n"
        "\n"
        "### PPO Buffer",
        "Architecture section (agent)",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "```python\n"
        "import gymnasium as gym\n"
        "from rl_training import DQNAgent, ReplayBuffer, ResultsManager, set_seed\n"
        "\n"
        "# Set seed for reproducibility\n"
        "set_seed(42)\n"
        "\n"
        "# Create environment\n"
        "env = gym.make(\"gymnasium_env/FullyDynamicEnv-v0\", data_path=\"data/\")\n"
        "\n"
        "# Initialize agent\n"
        "replay_buffer = ReplayBuffer(max_size=100000)\n"
        "agent = DQNAgent(\n"
        "    num_actions=8,\n"
        "    replay_buffer=replay_buffer,\n"
        "    gamma=0.95,\n"
        "    lr=1e-4,\n"
        "    device='cuda:0'\n"
        ")\n"
        "\n"
        "# Initialize results manager\n"
        "results_mgr = ResultsManager.create_with_auto_increment(\"results/\")\n"
        "\n"
        "# Training loop\n"
        "for episode in range(140):\n"
        "    state, info = env.reset()\n"
        "    done = False\n"
        "    total_reward = 0\n"
        "\n"
        "    while not done:\n"
        "        action = agent.select_action(state, epsilon_greedy=True)\n"
        "        next_state, reward, done, _, info = env.step(action)\n"
        "\n"
        "        replay_buffer.push(state, action, reward, next_state, done)\n"
        "        loss = agent.train_step(batch_size=64)\n"
        "\n"
        "        state = next_state\n"
        "        total_reward += reward\n"
        "\n"
        "    agent.update_target_network()\n"
        "    print(f\"Episode {episode}: Reward = {total_reward:.2f}\")\n"
        "```",
        "```python\n"
        "import gymnasium as gym\n"
        "from rl_training import PPOAgent, PPOBuffer, PPO, ResultsManager, set_seed\n"
        "\n"
        "# Set seed for reproducibility\n"
        "set_seed(42)\n"
        "\n"
        "# Create environment\n"
        "env = gym.make(\"gymnasium_env/FullyDynamicEnv-v0\", data_path=\"data/\")\n"
        "\n"
        "# Initialize network + PPO agent (on-policy: buffer is per-rollout)\n"
        "network = PPO(num_actions=env.action_space.n, node_features=5)\n"
        "agent = PPOAgent(network, lr=3e-4, gamma=0.99, gae_lambda=0.95,\n"
        "                  clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,\n"
        "                  device='cuda:0')\n"
        "buffer = PPOBuffer()\n"
        "\n"
        "# Initialize results manager\n"
        "results_mgr = ResultsManager.create_with_auto_increment(\"results/\")\n"
        "\n"
        "# Training loop\n"
        "for episode in range(140):\n"
        "    state, info = env.reset()\n"
        "    done = False\n"
        "    total_reward = 0\n"
        "\n"
        "    while not done:\n"
        "        action, logprob, value = agent.select_action(state)\n"
        "        next_state, reward, done, _, info = env.step(action)\n"
        "\n"
        "        buffer.push(state, action, logprob, reward, value, done)\n"
        "        state = next_state\n"
        "        total_reward += reward\n"
        "\n"
        "    # PPO updates on the full rollout (multiple epochs), then clears it\n"
        "    policy_loss, value_loss, entropy = agent.update(buffer)\n"
        "    buffer.clear()\n"
        "    print(f\"Episode {episode}: Reward = {total_reward:.2f}\")\n"
        "```",
        "Custom Training Loop example",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "```python\n"
        "from rl_training import DQNAgent\n"
        "\n"
        "# Load trained model\n"
        "agent = DQNAgent(num_actions=8, device='cuda:0')\n"
        "agent.load_model(\"results/run_001/models/best/episode_095/trained_agent.pt\")\n"
        "\n"
        "# Validate\n"
        "env = gym.make(\"gymnasium_env/FullyDynamicEnv-v0\", data_path=\"data/\")\n"
        "state, _ = env.reset()\n"
        "\n"
        "# Greedy evaluation\n"
        "done = False\n"
        "while not done:\n"
        "    action = agent.select_action(state, greedy=True)\n"
        "    state, reward, done, _, _ = env.step(action)\n"
        "```",
        "```python\n"
        "from rl_training import PPOAgent, PPO\n"
        "\n"
        "# Load trained model\n"
        "network = PPO(num_actions=8, node_features=5)\n"
        "agent = PPOAgent(network, device='cuda:0')\n"
        "agent.load_model(\"results/run_001/models/best/episode_095/trained_agent.pt\")\n"
        "\n"
        "# Validate\n"
        "env = gym.make(\"gymnasium_env/FullyDynamicEnv-v0\", data_path=\"data/\")\n"
        "state, _ = env.reset()\n"
        "\n"
        "# Evaluation (PPO samples from the policy distribution; there is no\n"
        "# separate greedy mode — pass avoid_action=[...] to mask invalid moves)\n"
        "done = False\n"
        "while not done:\n"
        "    action, _, _ = agent.select_action(state)\n"
        "    state, reward, done, _, _ = env.step(action)\n"
        "```",
        "Load and Validate Model example",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "bss-validate \\\n"
        "    --model-path results/run_001/models/best/episode_139/trained_agent.pt \\\n"
        "    --data-path data/ \\\n"
        "    --results-path results/validation/ \\\n"
        "    --min-epsilon 0.05 \\\n"
        "    --num-bikes 300",
        "bss-validate \\\n"
        "    --model-path results/run_001/models/best/episode_139/trained_agent.pt \\\n"
        "    --data-path data/ \\\n"
        "    --results-path results/validation/ \\\n"
        "    --max-num-bikes 300",
        "Validation CLI example — flags renamed/removed (no --min-epsilon, "
        "--num-bikes is now --max-num-bikes/--min-num-bikes)",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "```python\n"
        "params = {\n"
        "    \"num_episodes\": 140,                 # Training episodes\n"
        "    \"batch_size\": 64,                    # Replay buffer batch size\n"
        "    \"replay_buffer_capacity\": 100000,    # Buffer capacity\n"
        "    \"gamma\": 0.95,                       # Discount factor\n"
        "    \"epsilon_start\": 1.0,                # Initial exploration rate\n"
        "    \"epsilon_end\": 0.01,                 # Final exploration rate\n"
        "    \"epsilon_decay\": 1e-5,               # Epsilon decay constant\n"
        "    \"lr\": 1e-4,                          # Learning rate (SGD)\n"
        "    \"total_timeslots\": 56,               # Timeslots per episode (1 week)\n"
        "    \"maximum_number_of_bikes\": 500,      # Fleet size\n"
        "    \"tau\": 0.005,                        # Soft target update rate\n"
        "    \"depot_position_id\": 103,            # Depot cell ID\n"
        "    \"initial_cell_id\": 103               # Starting cell ID\n"
        "}\n"
        "```",
        "```python\n"
        "params = {\n"
        "    \"num_episodes\": 250,                 # Training episodes\n"
        "    \"rollout_steps\": 4096,               # On-policy rollout buffer capacity\n"
        "    \"minibatch_size\": 512,               # Minibatch size per PPO update epoch\n"
        "    \"gamma\": 0.99,                       # Discount factor\n"
        "    \"gae_lambda\": 0.95,                  # GAE factor\n"
        "    \"clip_coef\": 0.2,                    # PPO clipping coefficient\n"
        "    \"ent_coef\": 0.02,                    # Entropy coefficient (can decay to ent_coef_final)\n"
        "    \"vf_coef\": 0.25,                     # Value loss coefficient\n"
        "    \"update_epochs\": 8,                  # Epochs per buffer at every PPO update\n"
        "    \"lr\": 5e-5,                          # Learning rate (Adam by default)\n"
        "    \"total_timeslots\": 56,               # Timeslots per episode (1 week)\n"
        "    \"maximum_number_of_bikes\": 1000,     # Fleet size\n"
        "    \"num_trucks\": 1,                     # Trucks per env (multi-area training is the\n"
        "                                          # preferred way to scale up — see --data-paths)\n"
        "    \"depot_position_id\": 12,             # Depot cell ID\n"
        "    \"initial_cell_id\": 12                # Starting cell ID\n"
        "}\n"
        "```\n"
        "\n"
        "(Values above mirror the defaults in `train.py`; see that file for "
        "the full parameter set, including `initial_cell_ids`, "
        "`enable_repositioning` and `use_net_flow`.)",
        "Training params example dict — was a DQN-era config, now mirrors "
        "the real PPO params in train.py",
    ),
    (
        "src/bss_rebalancing/rl_training/README.md",
        "- Total reward, mean failures, total trips, invalid actions, epsilon",
        "- Total reward, mean failures, total trips, invalid actions, "
        "mean state value",
        "Tracked Metrics (episode-level) list",
    ),

    # =======================================================================
    # src/bss_rebalancing/results_webapp/README.md
    # =======================================================================
    (
        "src/bss_rebalancing/results_webapp/README.md",
        "3. **Epsilon Decay**: Exploration rate evolution over episodes",
        "3. **Policy Entropy**: Exploration signal (policy entropy) evolution "
        "over episodes",
        "Overview tab metric 3",
    ),
    (
        "src/bss_rebalancing/results_webapp/README.md",
        "5. **Q-Values**: Mean Q-values across all actions per timeslot",
        "5. **State Values**: Mean critic value estimates per timeslot",
        "Overview tab metric 5",
    ),
    (
        "src/bss_rebalancing/results_webapp/README.md",
        "7. **Training Loss**: TD error loss with moving average",
        "7. **Training Loss**: PPO policy loss / value loss with moving average",
        "Overview tab metric 7",
    ),
    (
        "src/bss_rebalancing/results_webapp/README.md",
        "- Epsilon Value",
        "- Entropy Value",
        "Episode stats card",
    ),
    (
        "src/bss_rebalancing/results_webapp/README.md",
        "   - Track epsilon decay\n"
        "   - Observe Q-value stabilization\n"
        "   - Check loss convergence",
        "   - Track policy entropy\n"
        "   - Observe critic value stabilization\n"
        "   - Check loss convergence",
        "Monitor Training checklist",
    ),
]


def apply_replacements(repo_root: Path, dry_run: bool) -> int:
    by_file: dict[str, list[Replacement]] = {}
    for rel_path, old, new, desc in REPLACEMENTS:
        by_file.setdefault(rel_path, []).append((rel_path, old, new, desc))

    total_ok, total_fail = 0, 0

    for rel_path, edits in by_file.items():
        file_path = repo_root / rel_path
        if not file_path.exists():
            print(f"[SKIP] {rel_path}: file not found")
            total_fail += len(edits)
            continue

        # Universal-newline read: normalizes CRLF/CR to "\n" for matching.
        original_text = file_path.read_text(encoding="utf-8")
        text = original_text

        for _, old, new, desc in edits:
            count = text.count(old)
            if count == 1:
                text = text.replace(old, new, 1)
                print(f"[OK]   {rel_path}: {desc}")
                total_ok += 1
            elif count == 0:
                print(f"[MISS] {rel_path}: {desc} — old text not found "
                      f"(already edited? line endings differ?)")
                total_fail += 1
            else:
                print(f"[WARN] {rel_path}: {desc} — old text appears "
                      f"{count} times, expected 1; skipped to avoid an "
                      f"ambiguous edit")
                total_fail += 1

        if text != original_text:
            if dry_run:
                diff = difflib.unified_diff(
                    original_text.splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                )
                sys.stdout.writelines(diff)
            else:
                # Preserve the file's original CRLF line endings on write.
                file_path.write_text(text, encoding="utf-8", newline="\r\n")

    print(f"\n{total_ok} edit(s) applied, {total_fail} skipped/failed "
          f"(out of {len(REPLACEMENTS)} total).")
    return 0 if total_fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path("."),
        help="Path to the BSS-DynamicRebalancing-RL repo root "
             "(the folder containing the top-level README.md).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print a unified diff instead of writing changes to disk.",
    )
    args = parser.parse_args()

    return apply_replacements(args.repo_root.resolve(), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())