# BSS Preprocessing Pipeline

Preprocessing pipeline for the BSS Dynamic Rebalancing RL project.

## Installation

Install in editable mode for development:

```bash
cd preprocessing
pip install -e .
```

## Usage

### Run the full preprocessing pipeline

```bash
bss-preprocess --data-path data/
```

Or using Python module syntax:

```bash
python -m preprocessing --data-path data/
```

### Run individual preprocessing steps

Each step can be run independently:

```bash
# Download trip data
python -m preprocessing.steps.download_trips --data-path data/

# Preprocess trip data (compute Poisson rates)
python -m preprocessing.steps.preprocess_data --data-path data/

# Interpolate data (build PMF matrices)
python -m preprocessing.steps.interpolate_data --data-path data/

# Preprocess truck grid
python -m preprocessing.steps.preprocess_truck_grid --data-path data/

# Preprocess distance matrix
python -m preprocessing.steps.preprocess_distance_matrix --data-path data/

# Preprocess global rates
python -m preprocessing.steps.preprocess_global_rates --data-path data/

# Preprocess nodes dictionary
python -m preprocessing.steps.preprocess_nodes_dictionary --data-path data/

# Create EV matrices
python -m preprocessing.steps.create_ev_matrices --data-path data/
```

### Plot mode (no preprocessing)

Generate plots without running the preprocessing pipeline:

```bash
# Plot base graph only
bss-preprocess --data-path data/ --plot graph

# Plot graph with cell grid
bss-preprocess --data-path data/ --plot grid

# Plot graph with cell grid and cell numbers
bss-preprocess --data-path data/ --plot grid-numbered
```

Plots are saved to `data/plots/` directory.

## Project Structure

```
preprocessing/
├── pyproject.toml
├── README.md
└── src/
    └── preprocessing/
        ├── __init__.py
        ├── __main__.py          # Entry point for: python -m preprocessing
        ├── cli.py               # CLI interface
        ├── config.py            # Shared configuration
        ├── core/
        │   ├── __init__.py
        │   ├── utils.py         # Shared utility functions
        │   ├── graph.py         # Graph initialization and manipulation
        │   ├── plotting.py      # Visualization functions
        │   └── grid.py          # Grid/cell-related utilities
        └── steps/
            ├── __init__.py
            ├── download_trips.py
            ├── preprocess_data.py
            ├── interpolate_data.py
            ├── preprocess_truck_grid.py
            ├── preprocess_distance_matrix.py
            ├── preprocess_global_rates.py
            ├── preprocess_nodes_dictionary.py
            └── create_ev_matrices.py
```