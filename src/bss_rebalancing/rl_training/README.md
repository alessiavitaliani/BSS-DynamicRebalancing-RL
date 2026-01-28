# RL Training

Deep Reinforcement Learning training framework for bike-sharing system dynamic rebalancing using Graph Attention Networks (GAT) and Deep Q-Networks (DQN).

This package provides a complete RL training pipeline for learning optimal bike rebalancing policies. It includes:
- **DQN Agent** with Graph Attention Networks for spatial reasoning
- **Experience Replay Buffer** optimized for graph-structured transitions
- **Results Management** with structured logging and model checkpointing
- **CLI Tools** for training and validation

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
    --min-epsilon 0.05 \
    --num-bikes 300
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
        │   └── dqn_agent.py        # DQN agent implementation
        ├── memory/
        │   ├── __init__.py
        │   └── replay_buffer.py    # Experience replay buffer
        ├── networks/
        │   ├── __init__.py
        │   └── dqn.py              # GAT-based DQN architecture
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
    "num_episodes": 140,                 # Training episodes
    "batch_size": 64,                    # Replay buffer batch size
    "replay_buffer_capacity": 100000,    # Buffer capacity
    "gamma": 0.95,                       # Discount factor
    "epsilon_start": 1.0,                # Initial exploration rate
    "epsilon_end": 0.01,                 # Final exploration rate
    "epsilon_decay": 1e-5,               # Epsilon decay constant
    "lr": 1e-4,                          # Learning rate (SGD)
    "total_timeslots": 56,               # Timeslots per episode (1 week)
    "maximum_number_of_bikes": 500,      # Fleet size
    "tau": 0.005,                        # Soft target update rate
    "depot_position_id": 103,            # Depot cell ID
    "initial_cell_id": 103               # Starting cell ID
}
```

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

### DQN Network

Graph Attention Network (GAT) based Q-network with:

#### Graph Encoder
- **Layer 1**: 4 input features → 64 features (4 heads, concat) → 256
- **Layer 2**: 256 → 64 features (4 heads, concat) → 256
- **Layer 3**: 256 → 128 features (2 heads, concat) → 256

#### Global Pooling
- **GlobalAttention**: Attention-based graph-level aggregation → 256

#### Graph Embedding
- FC layers: 256 → 256 → 128 → 64

#### Agent State Encoder
- Input: 162-dimensional agent state (truck load, position, action history)
- FC layers: 162 → 256 → 256 → 128 → 64

#### Fusion and Q-Values
- Concatenate: graph embedding (64) + agent embedding (64) → 128
- FC layers: 128 → 256 → 128 → 8 (Q-values for 8 actions)

### DQN Agent

Features:
- **Epsilon-greedy exploration** with exponential decay
- **Experience replay** with graph-structured transitions
- **Target network** with soft updates (τ=0.005)
- **Action masking** to prevent invalid moves
- **Gradient clipping** (max_norm=10.0)
- **Smooth L1 loss** (Huber loss)

### Replay Buffer

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
- Total reward, mean failures, total trips, invalid actions, epsilon

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
from rl_training import DQNAgent, ReplayBuffer, ResultsManager, set_seed

# Set seed for reproducibility
set_seed(42)

# Create environment
env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path="data/")

# Initialize agent
replay_buffer = ReplayBuffer(max_size=100000)
agent = DQNAgent(
    num_actions=8,
    replay_buffer=replay_buffer,
    gamma=0.95,
    lr=1e-4,
    device='cuda:0'
)

# Initialize results manager
results_mgr = ResultsManager.create_with_auto_increment("results/")

# Training loop
for episode in range(140):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon_greedy=True)
        next_state, reward, done, _, info = env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step(batch_size=64)

        state = next_state
        total_reward += reward

    agent.update_target_network()
    print(f"Episode {episode}: Reward = {total_reward:.2f}")
```

### Load and Validate Model

```python
from rl_training import DQNAgent

# Load trained model
agent = DQNAgent(num_actions=8, device='cuda:0')
agent.load_model("results/run_001/models/best/episode_095/trained_agent.pt")

# Validate
env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path="data/")
state, _ = env.reset()

# Greedy evaluation
done = False
while not done:
    action = agent.select_action(state, greedy=True)
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