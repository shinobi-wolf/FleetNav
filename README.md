# DroneSim

A simulation framework for studying **drone-assisted Electric Vehicle (EV) routing** in maze environments. The project benchmarks routing planners under range-constrained EV fleets, with and without an aerial **Battery Management System (BMS)** that dispatches UAVs to recharge EVs in the field.

---

## Overview

Ground-level EVs (patrol/delivery vehicles) must visit a set of tasks scattered across a procedurally-generated maze. Each EV starts at a central warehouse and has a finite battery range. When a vehicle's battery is low, UAVs stationed at fixed base stations can fly out to rendezvous with the EV and top up its battery, extending effective range without forcing a detour back to the warehouse.

The framework compares two routing strategies:

| Algorithm | Description |
|-----------|-------------|
| **mTSP** | KMeans clustering of tasks + nearest-neighbour TSP per cluster |
| **EVRP** | Electric VRP solved via RoutingBlocks ALNS with battery-feasibility constraints and makespan minimisation |

Both strategies can be run with or without the BMS (UAV recharging) layer.

---

## Project Structure

```
DroneSim/
├── main.py                  # Entry point
├── console.py               # Pygame visualisation front-end
├── server.py                # (Alternative) server-based front-end
├── baseline_testing.py      # Batch benchmark: mTSP vs EVRP, no BMS
├── bms_testing.py           # Batch benchmark: mTSP vs EVRP, with BMS
├── evrp.py                  # Standalone EVRP utilities
├── requirements.txt         # Python dependencies
├── config/
│   └── world1.yaml          # Simulation configuration
├── data/
│   └── maze.csv             # Pre-generated maze data
├── world_1/
│   ├── world.py             # Core simulation engine (EVs + UAVs)
│   ├── planner.py           # Planning algorithms (mTSP, EVRP, OR-Tools TSP)
│   └── vehicles/
│       ├── EV.py            # Electric vehicle model
│       └── UAV.py           # UAV / drone model
└── runs/                    # Experiment output CSVs
    └── iros_26/             # Results for current experiments
```

---

## Installation

### Prerequisites

- Python 3.9+
- [RoutingBlocks](https://github.com/tumBAIS/RoutingBlocks) (ALNS-based EVRP solver)
- Google OR-Tools

### Install dependencies

```bash
pip install -r requirements.txt
pip install routingblocks ortools
```

`requirements.txt` includes:

```
numpy
pandas
networkx
scikit-learn
pyyaml
pyamaze
pygame
```

---

## Configuration

Edit `config/world1.yaml` to control the simulation parameters:

```yaml
world:
  num_evs: 4                 # Number of ground EVs
  num_tasks: 20              # Number of delivery/patrol tasks
  num_uavs_per_bs: 2         # UAVs per base station
  num_base_stations: 4       # Number of UAV base stations
  maze_loop_percentage: 50   # Maze connectivity (0 = tree, 100 = fully connected)
  maze_size: 35              # Maze grid dimension (N×N)
  bms: true                  # Enable UAV-based battery recharging
  algo: "mtsp"               # Routing algorithm: "mtsp" or "evrp"

ev:
  range: 3000                # EV battery range (in edge-weight units)

uav:
  range: 3000                # UAV battery range
  speed: 3                   # UAV movement speed (cells/step)
```

---

## Usage

### Interactive visualisation

```bash
python main.py
```

This launches the Pygame visualisation showing EVs navigating the maze, UAVs flying from base stations to recharge EVs, and real-time battery indicators.

### Batch benchmarking

**mTSP vs EVRP (no BMS):**
```bash
python baseline_testing.py
```

**mTSP vs EVRP (with BMS):**
```bash
python bms_testing.py
```

Results are written to CSV files under `runs/iros_26/`.

---

## Algorithms

### mTSP Planner
1. Clusters tasks into `num_evs` groups using **KMeans**.
2. Orders each cluster with a **nearest-neighbour TSP** heuristic.
3. Each EV follows its ordered task list, returning to the warehouse to recharge when necessary.

### EVRP Planner
1. Builds an all-pairs shortest-path distance matrix on the maze graph.
2. Models the problem as an **Electric VRP** with the warehouse acting as the sole charging depot.
3. Solves with **RoutingBlocks ALNS** (Adaptive Large Neighbourhood Search), minimising makespan with a penalty for under-utilised vehicles.
4. Optionally uses **Google OR-Tools** for exact TSP ordering within routes.

### Battery Management System (BMS)
- EVs broadcast a recharge request when battery drops below a threshold relative to the cost of returning to the warehouse.
- The nearest idle UAV with sufficient battery is dispatched from its base station.
- The UAV rendezvous with the EV in the field, transfers energy, and returns to its base.
- This allows EVs to complete longer routes without detours.

---

## Output Metrics

Each benchmark run records the following per trial:

| Column | Description |
|--------|-------------|
| `algo` | Routing algorithm used (`mtsp` / `evrp`) |
| `num_evs` | Number of EVs |
| `num_tasks` | Number of tasks |
| `ev_range` | EV battery range |
| `maze_loop_percentage` | Maze connectivity parameter |
| `maze_size` | Maze grid size |
| `bms_enabled` | Whether BMS was active |
| `final_frame` | Simulation steps to completion |
| `num_tasks_completed` | Tasks successfully completed |
| `ev_distance_traveled` | Total EV distance across all vehicles |
| `drone_distance_traveled` | Total UAV distance across all drones |

---

## Results

Pre-computed experiment results are stored in `runs/iros_26/`:

- `results_mtsp_latest.csv` / `results_evrp_latest.csv` — baseline algorithm comparison
- `results_mtsp_bms.csv` / `results_evrp_bms.csv` — algorithm comparison with BMS enabled
- `results_mtsp_v_evrp_reachable.csv` — comparison restricted to reachable task sets

Exploratory data analysis is available in `eda.ipynb` and `results.ipynb`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` / `pandas` | Numerical computation and data handling |
| `networkx` | Maze graph construction and shortest-path queries |
| `scikit-learn` | KMeans clustering for mTSP |
| `pyamaze` | Procedural maze generation |
| `pygame` | Real-time 2-D visualisation |
| `pyyaml` | Configuration file parsing |
| `routingblocks` | ALNS-based EVRP solver |
| `ortools` | OR-Tools TSP solver |
