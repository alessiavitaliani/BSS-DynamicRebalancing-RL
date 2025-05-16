# 🚲 Fully Dynamic Rebalancing of Dockless Bike Sharing Systems using Deep Reinforcement Learning
### Last-mile mobility. Real-time simulation. Adaptive rebalancing.

This repository accompanies the thesis project *Fully Dynamic Rebalancing of Dockless Bike Sharing Systems using Deep Reinforcement Learning*. It presents a novel framework for dynamically rebalancing bikes in a dockless **Bike Sharing System (BSS)** using a **Double Deep Q-Network (DDQN)** trained in a realistic, event-driven simulation environment.

---

## 🧠 Project Overview

Cities are increasingly adopting sustainable transport solutions to address congestion, emissions, and urban sprawl. Dockless BSSs offer flexible, green last-mile mobility—but their very flexibility introduces operational complexity. Bikes often cluster in popular zones, leaving others underserved.

This thesis tackles the challenge with a **fully dynamic rebalancing framework** driven by **Reinforcement Learning**, where decisions are made in real time based on real-world demand patterns and traffic conditions.

### 🎯 Highlights
- 🧠 A **DDQN agent** learns to make rebalancing decisions under uncertainty.
- 🧪 **Event-driven simulation** using real **Cambridge, MA** demand data and **TomTom traffic profiles**.
- 🛠️ Baseline comparisons, reward design exploration, and scalability considerations.

---

## 📄 Thesis

You can read the full thesis [(https://hdl.handle.net/20.500.12608/84368)](https://hdl.handle.net/20.500.12608/84368).

---

## 📁 Project Structure

```
├── README.md
├── RL-agent
│   ├── DuelingDQN.py
│   ├── VanillaDQN.py
│   ├── agent.py
│   ├── dummy_file.py
│   ├── replay_memory.py
│   ├── train_model.py
│   ├── utils.py
│   └── validate_model.py
├── benchmarks
│   ├── benchmark.py
│   ├── results
│   │   ├── rebalance_time.pkl
│   │   └── total_failures.pkl
│   └── utils.py
├── data
│   └── utils
│       ├── ev_consumption_matrix.csv
│       ├── ev_velocity_matrix.csv
│       └── filtered_stations.csv
├── gymnasium_env
│   ├── __init__.py
│   ├── envs
│   │   ├── FullyDynamicEnvironment.py
│   │   ├── StaticEnvironment.py
│   │   └── __init__.py
│   ├── register_env.py
│   └── simulator
│       ├── __init__.py
│       ├── bike.py
│       ├── bike_simulator.py
│       ├── cell.py
│       ├── event.py
│       ├── station.py
│       ├── trip.py
│       ├── truck.py
│       ├── truck_simulator.py
│       └── utils.py
├── preprocessing
│   ├── preprocessing.py
│   └── utils
│       ├── __init__.py
│       ├── download_trips_data.py
│       ├── interpolate_data.py
│       ├── preprocess_data.py
│       ├── preprocess_distance_matrix.py
│       ├── preprocess_global_rates.py
│       ├── preprocess_nodes_dictionary.py
│       ├── preprocess_truck_grid.py
│       └── utils.py
├── pyproject.toml
├── requirements.txt
├── results
│   ├── concatenation_results.py
│   ├── process_results.py
│   ├── results_webserver.py
│   ├── total_failures_baseline.pkl
│   ├── total_failures_baseline.png
│   └── utils.py
└── setup.py
```

---

## 🚀 Setup Instructions

1. **Install dependencies**: We recommend using Python 3.11+ and a virtual environment.
    ```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. **Install the custom gymnasium environment**: Use the following to install it in editable mode (required for imports to work correctly).
    ```
    pip install -e .
    ```

---

## 📂 Data Sources
- 🚲 BlueBikes trip data (Cambridge, MA)
- 🛣️ TomTom traffic speed profiles
- 🌐 Street networks from OpenStreetMap via osmnx

*(Note: Some datasets are not included due to licensing. Instructions for downloading and preprocessing them are in `preprocessing/`.)*

Run the preprocessing script to populate the `data` folder with trips and precomputed matrices.

```
python preprocessing/preprocessing.py
```

**Arguments**:
- `--data_path`: The directory where data will be saved (default: `../data/`). It's preferable to leave it as default.

---

## 🧠 Training the DQN Agent

Run:

```
python RL-agent/train_model.py
```

**Arguments**:
- `--data_path`: Path to the data folder.
- `--cuda_device`: CUDA device to use (default: 0).
- `--enable_logging`: Enable logging.
- `--enable_checkpoint`: Enable model checkpointing.
- `--restore_from_checkpoint`: Restore training from the last checkpoint.
- `--num_episodes`: Number of training episodes.
- `--run_id`: ID to identify the experiment run.
- `--exploration_time`: Number of episodes during which to explore.
- `--enable_telegram`: Enable Telegram notifications.
- `--telegram_token`: Telegram bot token.
- `--telegram_chat_id`: Telegram chat ID.

---

## 🧪 Benchmarks and Baselines
Heuristic rebalancing strategies (e.g., static allocation, naive balancing) are implemented under benchmarks/ for comparison with the RL agent.

Run:
```
python benchmarks/benchmark.py
```

**Arguments**:

- `--data_path`: Path to the data folder (default: `../data/`).

---

## 📈 Results Summary

The DDQN agent demonstrated the ability to:
- Adapt to real-time, location-specific demand fluctuations.
- Reduce service failures compared to static and heuristic strategies.
- Operate under fully dynamic, non-simplified simulation conditions.

While not always outperforming all baselines, the agent proved the feasibility of deep RL for real-time bike rebalancing at scale and underscored the importance of careful reward design.

---

## 📚 Citation

If you use this work in your own research, please cite:
```
@thesis{edoardoscarpel2025bss,
  title={Fully Dynamic Rebalancing of Dockless Bike Sharing Systems using Deep Reinforcement Learning},
  author={Edoardo Scarpel},
  year={2025},
  school={Università of Padua},
  url={https://hdl.handle.net/20.500.12608/84368}
}
```

---

## 📬 Contact

For any issues or questions, feel free to reach out or open an issue.
- GitHub: @edos08
