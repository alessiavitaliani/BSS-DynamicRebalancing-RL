# BSS RL Training

Reinforcement Learning training and validation module for the BSS Dynamic Rebalancing project.

## Installation

Install in editable mode for development:

```bash
cd rl_training
pip install -e .
```

For Telegram progress bar support:
```bash
pip install -e ".[telegram]"
```

## Usage

### Training

```bash
# Basic training
bss-train --data_path ../data/ --results_path results/

# With options
bss-train \
    --data_path ../data/ \
    --results_path results/ \
    --num_episodes 140 \
    --num_bikes 500 \
    --run_id 1 \
    --cuda_device 0 \
    --enable_logging
```

### Validation

```bash
# Validate a trained model
bss-validate \
    --model_path results/validation_1/trained_models/139/trained_agent.pt \
    --data_path ../data/ \
    --results_path results/ \
    --epsilon 0.05
```

## Project Structure

```
rl_training/
├── pyproject.toml
├── README.md
└── src/
    └── rl_training/
        ├── __init__.py
        ├── train.py           # Training script with CLI
        ├── validate.py        # Standalone validation script
        ├── utils.py           # Utility functions
        ├── agents/
        │   ├── __init__.py
        │   └── dqn_agent.py   # DQN Agent implementation
        ├── memory/
        │   ├── __init__.py
        │   └── replay_buffer.py
        └── networks/
            ├── __init__.py
            └── dqn.py         # DQN network architectures
```

## Training Parameters

Default training parameters:

- `num_episodes`: 140 - Total training episodes
- `batch_size`: 64 - Batch size for replay buffer sampling
- `replay_buffer_capacity`: 100,000 - Capacity of replay buffer
- `gamma`: 0.95 - Discount factor
- `epsilon_start`: 1.0 - Starting exploration rate
- `epsilon_end`: 0.01 - Minimum exploration rate
- `lr`: 1e-4 - Learning rate
- `total_timeslots`: 56 - Time slots per episode (1 week)
- `maximum_number_of_bikes`: 500 - System bike capacity
- `tau`: 0.005 - Soft update parameter

## Agents

### DQNAgent

Deep Q-Network agent with:
- Epsilon-greedy action selection
- Soft target network updates
- Experience replay
- Checkpoint save/load support

## Networks

### DQN / DQNv2

Graph Attention Network (GAT) based Q-network with:
- Multi-head attention for graph encoding
- Global attention pooling
- Agent state fusion
- Q-value prediction head
