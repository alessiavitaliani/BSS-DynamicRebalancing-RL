# RL Training

Deep Reinforcement Learning training framework for bike-sharing system dynamic rebalancing using Graph Attention Networks (GAT) and Proximal Policy Optimization (PPO).

This package provides a complete RL training pipeline for learning optimal bike rebalancing policies. It includes:
- **PPO Agent** (GAT-based actor-critic) with Graph Attention Networks for spatial reasoning
- **On-policy rollout buffer** (`PPOBuffer`), cleared after every update
- **Results Management** with structured logging and model checkpointing
- **CLI Tools** for training and validation, including multi-area training with a single shared policy

The framework trains agents to control a rebalancing truck navigating a spatial grid, making decisions about bike pickup, drop-off, and charging to minimize system failures.

---

## Installation

Install in editable mode for development:

```bash
cd rl_training
pip install -e .
```

Or install from the root project directory:

```bash
pip install -e src/bss_rebalancing/rl_training
```

---

## Quick Start

### Training

Basic training with default parameters:

```bash
bss-train --data-path data/ --results-path results/
```

Full training with custom configuration:

```bash
bss-train \
    --data-path data/ \
    --results-path results/ \
    --run-id 1 \
    --num-episodes 150 \
    --num-bikes 300 \
    --device cuda:0 \
    --seed 42 \
    --exploration-time 0.6 \
    --enable-logging
```

### Validation

Validate a trained model:

```bash
bss-validate \
    --model-path results/run_001/models/best/episode_139/trained_agent.pt \
    --data-path data/ \
    --results-path results/validation/ \
    --max-num-bikes 300
```

---

## Package Structure

```
rl_training/
├── README.md
├── pyproject.toml
└── src/
    └── rl_training/
        ├── __init__.py
        ├── train.py                # Training script with CLI
        ├── validate.py             # Validation script with CLI
        ├── utils.py                # Helper functions
        ├── agents/
        │   ├── __init__.py
        │   └── ppo_agent.py        # PPO agent implementation
        ├── memory/
        │   ├── __init__.py
        │   └── ppo_buffer.py       # On-policy rollout buffer
        ├── networks/
        │   ├── __init__.py
        │   └── ppo.py              # GAT-based actor-critic (PPO)
        └── results/
            ├── __init__.py
            └── results_manager.py  # Results logging and management
```

---

## Training Parameters

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run-id` | int | `0` | Experiment run identifier |
| `--data-path` | str | `"data/"` | Path to preprocessed data directory |
| `--results-path` | str | `"results/"` | Path to save results and models |
| `--device` | str | `"cpu"` | Hardware device (`cpu`, `cuda:0`, `mps`) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--num-episodes` | int | `140` | Total training episodes (weeks) |
| `--num-bikes` | int | `500` | System bike fleet size |
| `--exploration-time` | float | `0.6` | Fraction of training for exploration |
| `--enable-logging` | flag | — | Enable detailed environment logging |
| `--one-validation` | flag | — | Validate only at training end |

### Default Hyperparameters

```python
params = {
    "num_episodes": 250,                 # Training episodes
    "rollout_steps": 4096,               # On-policy rollout buffer capacity
    "minibatch_size": 512,               # Minibatch size per PPO update epoch
    "gamma": 0.99,                       # Discount factor
    "gae_lambda": 0.95,                  # GAE factor
    "clip_coef": 0.2,                    # PPO clipping coefficient
    "ent_coef": 0.02,                    # Entropy coefficient (can decay to ent_coef_final)
    "vf_coef": 0.25,                     # Value loss coefficient
    "update_epochs": 8,                  # Epochs per buffer at every PPO update
    "lr": 5e-5,                          # Learning rate (Adam by default)
    "total_timeslots": 56,               # Timeslots per episode (1 week)
    "maximum_number_of_bikes": 1000,     # Fleet size
    "num_trucks": 1,                     # Trucks per env (multi-area training is the
                                          # preferred way to scale up — see --data-paths)
    "depot_position_id": 12,             # Depot cell ID
    "initial_cell_id": 12                # Starting cell ID
}
```

(Values above mirror the defaults in `train.py`; see that file for the full parameter set, including `initial_cell_ids`, `enable_repositioning` and `use_net_flow`.)

### Reward Parameters

Configurable reward weights for shaping agent behavior:

```python
reward_params = {
    'W_ZERO_BIKES': 1.0,          # Weight for empty station penalty
    'W_CRITICAL_ZONES': 1.0,      # Weight for critical zone rewards
    'W_DROP_PICKUP': 0.9,         # Weight for drop/pickup actions
    'W_MOVEMENT': 0.7,            # Weight for movement costs
    'W_CHARGE_BIKE': 0.9,         # Weight for charging actions
    'W_STAY': 0.7,                # Weight for stay penalties
}
```

---

## Architecture

### PPO Network

Graph Attention Network (GAT) based actor-critic with a shared encoder:

#### Graph Encoder (shared by actor and critic)
- **Layer 1** (`GATv2Conv`): `node_features` (5 by default — see `gnn_features` in `train.py`) → 64 features × 4 heads (concat) → 256
- **Layer 2** (`GATv2Conv`): 256 → 64 features × 4 heads (concat) → 256
- **Global mean pooling** → 256-d graph embedding

#### Critic Head
- FC layers: 256 → 128 → 64 → 1 (state value V(s))

#### Actor Head
- FC layers: 256 → 128 → 64 → `num_actions` (action logits, masked for invalid moves before sampling)

### PPO Agent

Features:
- **Stochastic policy exploration** via the Actor's categorical distribution entropy (no epsilon-greedy)
- **GAE** (Generalized Advantage Estimation) for advantage/return computation
- **Clipped surrogate objective** (`clip_coef`, default 0.2) instead of a target network
- **Action masking** to prevent invalid moves
- **Gradient clipping** (max_norm=0.5)
- **Combined loss**: clipped policy loss + value loss (MSE) − entropy bonus
- **Optimizer**: Adam (default) or SGD with Nesterov momentum

### PPO Buffer

Custom `PairData` structure for storing graph transitions:
- **Source state** (S): graph with node features, edges, agent state
- **Target state** (S'): next graph configuration
- **Transition data**: action, reward, done flag, n-steps
- **Batch sampling**: Efficient batching with PyTorch Geometric

---

## Results Management

The `ResultsManager` provides structured experiment tracking:

### Directory Structure

```
results/
└── run_001/
    ├── config.json                     # Hyperparameters
    ├── training/
    │   ├── training_summary.csv        # Aggregated metrics
    │   └── episode_000/
    │       ├── scalars.json            # Total reward, failures, etc.
    │       ├── timeslot_metrics.csv    # Per-timeslot metrics
    │       ├── step_data.pkl.gz        # Actions, Q-values, losses
    │       └── cell_subgraph.gpickle   # Spatial data
    ├── validation/
    │   ├── validation_summary.csv
    │   └── episode_000/
    │       └── ...
    └── models/
        ├── checkpoints/
        │   └── episode_139/
        │       ├── trained_agent.pt
        │       └── metadata.json
        ├── best/
        │   └── episode_095/
        │       └── trained_agent.pt
        └── best_models_summary.csv
```

### Tracked Metrics

**Episode-level**:
- Total reward, mean failures, total trips, invalid actions, mean state value

**Timeslot-level**:
- Reward per timeslot, failures per timeslot, deployed bikes

**Step-level**:
- Actions, reward per action type, Q-values, TD loss, critic scores

**Spatial**:
- Cell-level statistics (bikes, failures, rebalancing operations)

---

## Usage Examples

### Custom Training Loop

```python
import gymnasium as gym
from rl_training import PPOAgent, PPOBuffer, PPO, ResultsManager, set_seed

# Set seed for reproducibility
set_seed(42)

# Create environment
env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path="data/")

# Initialize network + PPO agent (on-policy: buffer is per-rollout)
network = PPO(num_actions=env.action_space.n, node_features=5)
agent = PPOAgent(network, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,
                  device='cuda:0')
buffer = PPOBuffer()

# Initialize results manager
results_mgr = ResultsManager.create_with_auto_increment("results/")

# Training loop
for episode in range(140):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, logprob, value = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        buffer.push(state, action, logprob, reward, value, done)
        state = next_state
        total_reward += reward

    # PPO updates on the full rollout (multiple epochs), then clears it
    policy_loss, value_loss, entropy = agent.update(buffer)
    buffer.clear()
    print(f"Episode {episode}: Reward = {total_reward:.2f}")
```

### Load and Validate Model

```python
from rl_training import PPOAgent, PPO

# Load trained model
network = PPO(num_actions=8, node_features=5)
agent = PPOAgent(network, device='cuda:0')
agent.load_model("results/run_001/models/best/episode_095/trained_agent.pt")

# Validate
env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path="data/")
state, _ = env.reset()

# Evaluation (PPO samples from the policy distribution; there is no
# separate greedy mode — pass avoid_action=[...] to mask invalid moves)
done = False
while not done:
    action, _, _ = agent.select_action(state)
    state, reward, done, _, _ = env.step(action)
```

---

## Troubleshooting

### CUDA Out of Memory

**Issue**: Training crashes with CUDA OOM error

**Solutions**:
- Reduce batch size: `--batch-size 32`
- Reduce replay buffer capacity in code: `replay_buffer_capacity=50000`
- Use CPU: `--device cpu`

### Slow Training

**Issue**: Training is very slow

**Causes**:
- Environment logging enabled
- Large graph size (many cells)
- CPU-only training

**Solutions**:
- Disable logging: remove `--enable-logging`
- Use GPU: `--device cuda:0`
- Increase cell size during preprocessing

### Model Loading Errors

**Issue**: `FileNotFoundError` or checkpoint mismatch

**Solutions**:
- Verify model path exists
- Check model architecture matches (graph/agent state dimensions)
- Ensure PyTorch version compatibility

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** → Prepares data
2. **Gymnasium Env** → Uses preprocessed data for simulation
3. **RL Training** (this package) → Trains agent in simulated environment
4. **Benchmark** → Compares against baselines
5. **Results WebApp** → Visualizes training progress

---

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@mastersthesis{scarpel2025bss,
  title   = {Fully Dynamic Rebalancing of Dockless Bike Sharing Systems 
             using Deep Reinforcement Learning},
  author  = {Scarpel, Edoardo},
  year    = {2025},
  school  = {Università degli Studi di Padova},
  url     = {https://hdl.handle.net/20.500.12608/84368}
}
```

---

## License

See root LICENSE file for details.

---

## Contributing

Part of the BSS Dynamic Rebalancing RL monorepo.  
See main repository for contribution guidelines.

---

## Author

**Edoardo Scarpel**  
Ph.D. Student, University of Padova  
Email: edoardo.scarpel@phd.unipd.it