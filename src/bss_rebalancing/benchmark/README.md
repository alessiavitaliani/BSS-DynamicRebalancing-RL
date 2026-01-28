# Benchmark

Baseline evaluation framework for bike-sharing system rebalancing strategies using the static TSP-based environment.

This package provides benchmarking capabilities for evaluating static rebalancing strategies in bike-sharing systems. It runs simulations using the `StaticEnv` environment, which performs periodic system-wide rebalancing using Traveling Salesman Problem (TSP) optimization.

The benchmark serves as a baseline for comparing against reinforcement learning approaches by measuring system performance metrics under deterministic rebalancing schedules.

---

## Installation

Install in editable mode for development:

```bash
cd benchmark
pip install -e .
```

Or install from the root project directory:

```bash
pip install -e src/bss_rebalancing/benchmark
```

---

## Quick Start

### Basic Usage

Run a benchmark with default settings:

```bash
bss-benchmark --data-path data/ --results-path results/
```

### Custom Configuration

```bash
bss-benchmark \
    --data-path data/ \
    --results-path results/ \
    --run-id 0 \
    --num-episodes 1 \
    --total-timeslots 56 \
    --maximum-number-of-bikes 300 \
    --fixed-rebal-bikes-per-cell 5
```

### Python API

```python
from benchmark import run_benchmark

config = {
    'data_path': 'data/',
    'results_path': './results',
    'run_id': 0,
    'num_episodes': 1,
    'total_timeslots': 56,
    'maximum_number_of_bikes': 300,
    'fixed_rebal_bikes_per_cell': 5
}

run_benchmark(config)
```

---

## Package Structure

```
benchmark/
├── README.md
├── pyproject.toml
└── src/
    └── benchmark/
        ├── __init__.py       # Package initialization and exports
        ├── run.py            # Main benchmark runner (CLI entry point)
        └── utils.py          # Utility functions
```

---

## CLI Arguments

| Argument | Type | Default      | Description |
|----------|------|--------------|-------------|
| `--data-path` | str | `"data/"`    | Path to preprocessed data directory |
| `--results-path` | str | `"results/"` | Path where results will be saved |
| `--run-id` | int | `0`          | Unique identifier for this benchmark run |
| `--num-episodes` | int | `1`          | Number of simulation episodes |
| `--total-timeslots` | int | `56`         | Total timeslots per episode (8/day × 7 days) |
| `--maximum-number-of-bikes` | int | `300`        | Total bike fleet size |
| `--fixed-rebal-bikes-per-cell` | int | `5`          | Base bikes per cell after rebalancing |

---

## Configuration

### Default Parameters

```python
class BenchmarkDefaults:
    # Simulation parameters
    NUM_EPISODES = 1
    TOTAL_TIMESLOTS = 56              # 1 week (8 timeslots/day × 7 days)

    # Fleet parameters
    MAXIMUM_BIKES = 300
    BIKES_PER_CELL = 5                # Base allocation per cell

    # Environment setup
    DEPOT_ID = 103                    # Depot cell ID
    INITIAL_CELL_ID = 103             # Starting cell ID
    NUM_REBALANCING_EVENTS = 8        # Rebalancing ops per day

    # Reproducibility
    RANDOM_SEED = 32
```

---

## How It Works

### Static Environment

The benchmark uses the `StaticEnv` environment, which:

1. **Distributes bikes** initially based on predicted net flow
2. **Simulates trips** using Poisson-distributed demand
3. **Performs periodic rebalancing** at scheduled intervals (e.g., every 12 hours)
4. **Uses TSP optimization** to determine optimal rebalancing routes
5. **Records metrics** including failures and rebalancing time

### Rebalancing Strategy

The TSP-based rebalancing:
- Identifies **surplus cells** (excess bikes) and **deficit cells** (insufficient bikes)
- Computes an **optimal route** using the Traveling Salesman Problem algorithm
- **Picks up bikes** from surplus locations
- **Drops bikes** at deficit locations
- Returns to depot if truck capacity is reached

### Metrics Collected

**Per Timeslot**:
- Number of trip failures (bike unavailable)
- Rebalancing operation duration (seconds)

**Aggregated**:
- Total failures across all episodes
- Total rebalancing time

---

## Results Output

Results are saved in a structured directory:

```
results/
└── benchmark/
    └── run_0/
        ├── total_failures.pkl       # List of failures per timeslot
        └── rebalance_time.pkl       # List of rebalancing durations
```

### Loading Results

```python
import pickle

# Load failures
with open('results/benchmark/run_0/total_failures.pkl', 'rb') as f:
    failures = pickle.load(f)

# Load rebalancing times
with open('results/benchmark/run_0/rebalance_time.pkl', 'rb') as f:
    rebalance_times = pickle.load(f)

print(f"Total failures: {sum(failures)}")
print(f"Mean rebalancing time: {np.mean(rebalance_times):.2f} seconds")
```

---

## Usage Examples

### Single Episode Benchmark

```bash
bss-benchmark \
    --data-path data/ \
    --results-path results/ \
    --run-id 1 \
    --num-episodes 1 \
    --maximum-number-of-bikes 300
```

### Multi-Episode Benchmark

In this example, we run 5 episodes, each lasting 2 weeks:
```bash
bss-benchmark \
    --data-path data/ \
    --results-path results/ \
    --run-id 2 \
    --num-episodes 5 \
    --total-timeslots 112  # 2 weeks per episode (default is 1 week)
```

### Fleet Size Comparison

Run benchmarks with different fleet sizes:

```bash
# Small fleet
bss-benchmark --run-id 10 --maximum-number-of-bikes 200

# Medium fleet
bss-benchmark --run-id 11 --maximum-number-of-bikes 300

# Large fleet
bss-benchmark --run-id 12 --maximum-number-of-bikes 500
```

### Rebalancing Frequency Study

Vary bikes per cell:

```bash
# Conservative (more bikes per cell)
bss-benchmark --run-id 20 --fixed-rebal-bikes-per-cell 10

# Moderate
bss-benchmark --run-id 21 --fixed-rebal-bikes-per-cell 5

# Aggressive (fewer bikes per cell)
bss-benchmark --run-id 22 --fixed-rebal-bikes-per-cell 2
```

---

## Comparison with RL Training

This benchmark provides a baseline for comparing against RL approaches:

| Metric | Static Benchmark | RL Training |
|--------|------------------|-------------|
| **Decision Making** | Scheduled, deterministic | Learned, adaptive |
| **Rebalancing** | Periodic TSP-based | Continuous agent control |
| **Flexibility** | Fixed schedule | Responds to demand patterns |
| **Computation** | TSP at intervals | Neural network inference per step |

---

## Troubleshooting

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'benchmark'`

**Solution**: Install in editable mode:
```bash
cd benchmark
pip install -e .
```

### Missing Data Files

**Issue**: `FileNotFoundError` when running benchmark

**Solution**: Ensure preprocessing pipeline has been run:
```bash
cd preprocessing
bss-preprocess --data-path /path/to/data --steps download preprocess grid
```

### Environment Not Found

**Issue**: `gymnasium.error.UnregisteredEnv: StaticEnv-v0 not found`

**Solution**: Install gymnasium_env package:
```bash
cd gymnasium_env
pip install -e .
```

### Slow Execution

**Issue**: Benchmark runs very slowly

**Causes**:
- Large fleet size
- Many episodes/timeslots
- Dense spatial grid

**Solutions**:
- Reduce fleet size: `--maximum-number-of-bikes 200`
- Reduce duration: `--total-timeslots 16` (2 days)
- Reduce episodes: `--num-episodes 1`

---

## Workflow Integration

This package is part of the larger BSS rebalancing pipeline:

1. **Preprocessing** → Prepares data
2. **Gymnasium Env** → Uses preprocessed data for simulation
3. **RL Training** → Trains agent in simulated environment
4. **Benchmark** (this package) → Compares against baselines
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
