# BSS Gymnasium Environment

Gymnasium environment for the BSS Dynamic Rebalancing RL project.

## Installation

Install in editable mode for development:

```bash
cd gymnasium_env
pip install -e .
```

## Usage

```python
import gymnasium as gym

# Create the fully dynamic environment
env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path="data/", results_path="../results/")

# Create the static environment
env = gym.make("gymnasium_env/StaticEnv-v0", data_path="data/")

# Reset the environment
obs, info = env.reset(options={
    'day': 'monday',
    'timeslot': 0,
    'total_timeslots': 56,
    'maximum_number_of_bikes': 3500,
})

# Run an episode
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, terminated, info = env.step(action)

env.close()
```

## Project Structure

```
gymnasium_env/
├── pyproject.toml
├── README.md
└── src/
    └── gymnasium_env/
        ├── __init__.py           # Package init with environment registration
        ├── envs/
        │   ├── __init__.py
        │   ├── fully_dynamic_env.py
        │   └── static_env.py
        └── simulator/
            ├── __init__.py
            ├── bike.py
            ├── bike_simulator.py
            ├── cell.py
            ├── event.py
            ├── station.py
            ├── trip.py
            ├── truck.py
            ├── truck_simulator.py
            └── utils.py
```

## Environments

### FullyDynamicEnv

A fully dynamic environment where the agent controls a truck to rebalance bikes across cells.

**Actions:**
- 0: STAY
- 1: UP
- 2: DOWN
- 3: LEFT
- 4: RIGHT
- 5: DROP_BIKE
- 6: PICK_UP_BIKE
- 7: CHARGE_BIKE

### StaticEnv

A static environment for baseline comparison using TSP-based rebalancing.
