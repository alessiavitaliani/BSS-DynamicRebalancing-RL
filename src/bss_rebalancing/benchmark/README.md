# Benchmark Module - Library Structure

## Complete Directory Structure

```
benchmark/                          # Root benchmark module directory
├── README.md                       # Package documentation
├── pyproject.toml                  # Package configuration
└── src/                           # Source code directory
    ├── bss_benchmark.egg-info/    # Package metadata (auto-generated)
    │   ├── PKG-INFO
    │   ├── SOURCES.txt
    │   ├── dependency_links.txt
    │   ├── entry_points.txt
    │   ├── requires.txt
    │   └── top_level.txt
    └── benchmark/                  # Main package directory
        ├── __init__.py            # Package initialization and exports
        ├── run.py                 # Main benchmark runner (CLI entry point)
        └── utils.py               # Utility functions
```

## File Mapping

Copy the refactored files to the new structure:

```bash
# From generated files → To library structure

pyproject_final.toml              → benchmark/pyproject.toml
README_benchmark.md               → benchmark/README.md
benchmark_init.py                 → benchmark/src/benchmark/__init__.py
benchmark_run.py                  → benchmark/src/benchmark/run.py
benchmark_utils_refactored.py     → benchmark/src/benchmark/utils.py
```

## Installation Instructions

### 1. Create Directory Structure

```bash
cd src/bss_rebalancing
mkdir -p benchmark/src/benchmark
cd benchmark
```

### 2. Copy Files

```bash
# Copy package configuration and documentation
cp /path/to/pyproject_final.toml pyproject.toml
cp /path/to/README_benchmark.md README.md

# Copy source files
cp /path/to/benchmark_init.py src/benchmark/__init__.py
cp /path/to/benchmark_run.py src/benchmark/run.py
cp /path/to/benchmark_utils_refactored.py src/benchmark/utils.py
```

### 3. Install the Package

```bash
# Development installation (editable mode)
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"

# Or with all optional features
pip install -e ".[all]"
```

## Package Structure Details

### Root Level Files

**pyproject.toml**
- Package metadata and dependencies
- Build system configuration
- Tool configurations (black, isort, pytest, mypy, pylint)
- Entry point: `bss-benchmark = "benchmark.run:main"`

**README.md**
- Package documentation
- Installation instructions
- Usage examples
- API reference

### src/benchmark/ Directory

**__init__.py**
- Package initialization
- Public API exports
- Version information
- Imports from run.py and utils.py

**run.py**
- Main benchmark execution logic
- `run_benchmark()`: Main entry point
- `run_simulation()`: Single episode simulation
- `save_results()`: Results persistence
- `parse_arguments()`: CLI argument parsing
- `main()`: CLI entry point

**utils.py**
- `convert_seconds_to_hours_minutes()`: Time formatting
- `convert_graph_to_data()`: Graph conversion for GNN
- `plot_data_online()`: Training visualization
- `plot_graph_with_truck_path()`: Network visualization
- `send_telegram_message()`: Telegram notifications
- `get_memory_usage()`: Memory monitoring
- `Actions`: Action enumeration

## Usage After Installation

### Command-Line Interface

```bash
# Use the installed CLI command
bss-benchmark --data_path data/ --results_path ./results --run_id 1

# Or run directly as module
python -m benchmark.run --data_path data/
```

### Python API

```python
# Import from installed package
from benchmark import run_benchmark, run_simulation
from benchmark.utils import convert_seconds_to_hours_minutes

# Run benchmark
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

### Import Structure

```python
# Top-level imports (from __init__.py)
from benchmark import (
    main,
    run_benchmark,
    run_simulation,
    Actions,
    convert_seconds_to_hours_minutes,
    get_memory_usage,
)

# Direct module imports
from benchmark.run import BenchmarkDefaults
from benchmark.utils import plot_data_online
```

## Comparison with rl_training Structure

### rl_training/
```
rl_training/
├── README.md
├── pyproject.toml
└── src/
    └── rl_training/
        ├── __init__.py
        ├── agents/              # Agent implementations
        ├── memory/              # Experience replay
        ├── networks/            # Neural networks
        ├── train.py             # Main training script
        ├── utils.py             # Utilities
        └── validate.py          # Validation
```

### benchmark/ (New Structure)
```
benchmark/
├── README.md
├── pyproject.toml
└── src/
    └── benchmark/
        ├── __init__.py
        ├── run.py              # Main benchmark script (like train.py)
        └── utils.py            # Utilities
```

**Key Similarities:**
- Same top-level structure (README, pyproject.toml, src/)
- Source code in `src/{package_name}/`
- Main execution script (train.py vs run.py)
- Utilities module
- Package initialization in __init__.py

**Differences:**
- Benchmark is simpler (no subdirectories for agents/memory/networks)
- Single main execution script vs multiple (train/validate)
- Focused on static environment evaluation

## Development Workflow

### Running Tests

```bash
cd benchmark
pytest
```

### Code Formatting

```bash
# Format code
black src/benchmark/

# Sort imports
isort src/benchmark/
```

### Type Checking

```bash
mypy src/benchmark/
```

### Building Distribution

```bash
# Build package
python -m build

# Install from wheel
pip install dist/bss_benchmark-1.0.0-py3-none-any.whl
```

## Integration with Main Project

Your complete project structure becomes:

```
BSS-DynamicRebalancing-RL-new/
├── data/
├── src/
│   └── bss_rebalancing/
│       ├── gymnasium_env/
│       │   ├── README.md
│       │   ├── pyproject.toml
│       │   └── src/
│       │       └── gymnasium_env/
│       ├── preprocessing/
│       │   ├── README.md
│       │   ├── pyproject.toml
│       │   └── src/
│       │       └── preprocessing/
│       ├── rl_training/
│       │   ├── README.md
│       │   ├── pyproject.toml
│       │   └── src/
│       │       └── rl_training/
│       └── benchmark/              ← NEW
│           ├── README.md
│           ├── pyproject.toml
│           └── src/
│               └── benchmark/
│                   ├── __init__.py
│                   ├── run.py
│                   └── utils.py
└── pyproject.toml
```

## Verification

After installation, verify the structure:

```bash
cd src/bss_rebalancing/benchmark

# Check package is installed
pip list | grep bss-benchmark

# Check CLI is available
which bss-benchmark

# Test import
python -c "from benchmark import run_benchmark; print('Success!')"

# Check version
python -c "import benchmark; print(benchmark.__version__)"
```

## Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
cd src/bss_rebalancing/benchmark
pip uninstall bss-benchmark
pip install -e .
```

### CLI Not Found

```bash
# Reinstall to register entry point
pip install --force-reinstall -e .
```

### Module Not Found

```bash
# Ensure you're in the right directory
cd src/bss_rebalancing/benchmark
python -m benchmark.run --help
```

## Next Steps

1. Create the directory structure
2. Copy all files to their locations
3. Install the package: `pip install -e .`
4. Run tests: `pytest`
5. Try the CLI: `bss-benchmark --help`
6. Run your first benchmark!

## Additional Files to Create

You may want to add:

```
benchmark/
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_run.py
│   └── test_utils.py
├── .gitignore             # Git ignore file
├── LICENSE                # License file
└── CHANGELOG.md           # Version history
```
