# DroneSim

A two-level hierarchical routing framework for autonomous last-mile delivery (LMD). The system jointly addresses dynamic urban navigation under traffic uncertainty and limited vehicle endurance due to battery constraints.

## Overview

Autonomous LMD systems face two fundamental bottlenecks: navigating dynamic urban environments and managing limited vehicle battery life. DroneSim tackles both through an integrated framework combining reinforcement learning-based routing with a UAV-assisted Battery Management System.

### Level 1 — Electric Vehicle Routing (EVRP)

The outer planning layer solves the **Green Vehicle Routing Problem (GVRP)**: assign delivery stops to a fleet of ground robots while respecting battery capacity constraints and routing them through Automated Fueling/charging Stations (AFS) as needed.

This layer is augmented with a **UAV-assisted Battery Management System (BMS)**: aerial vehicles deliver replacement batteries to ground robots in the field, eliminating costly depot-return detours. The battery dispatch problem is modeled as a **Segment-VRP**, selecting strategic rendezvous points along robot routes to minimize UAV travel cost.

### Level 2 — Traffic-Responsive Shortest Path (TRSP)

The inner navigation layer handles **node-to-node routing** for each ground robot as it executes its Level 1 plan. Edge traversal times vary due to time-varying traffic (modeled as piecewise-constant speed segments). A reinforcement learning agent learns navigation policies that adapt to these conditions, outperforming static shortest-path approaches.

## Installation

```bash
pip install -r dronesim/trsp/requirements.txt
```

**Additional dependencies:**
- [SUMO](https://sumo.dlr.de/docs/Downloads.php) — traffic simulation (must be installed and on PATH)
- [Gurobi](https://www.gurobi.com/) — MILP solver with valid license (required for exact EVRP solver)
- [RoutingBlocks](https://routingblocks.readthedocs.io/) — heuristic VRP solver

## Usage

### Level 1: EVRP Planning + Simulation

```bash
# Exact MILP solver (requires Gurobi license)
python run_sim.py -c config/evrp/milp.yaml --planner milp

# Heuristic solver (RoutingBlocks ALNS — fast, no license needed)
python run_sim.py -c config/evrp/heuristic.yaml --planner heuristic

# Multi-TSP baseline
python run_sim.py -c config/evrp/mtsp.yaml --planner mtsp

# Plan only, skip simulation
python run_sim.py -c config/evrp/milp.yaml --no-sim
```

### Level 2: TRSP Agent Training + Evaluation

```bash
# Train an RL policy
python dronesim/trsp/train.py

# Evaluate a trained policy
python dronesim/trsp/main.py -p <policy_id>
```

## Configuration

Configs are YAML files under `config/evrp/`. A child config can inherit from a base using `base: path/to/base.yaml` (deep-merge semantics). Key parameters:

| Parameter | Description |
|-----------|-------------|
| `fleet.size` | Number of ground robots |
| `gvrp.customer_count` | Delivery stops to serve |
| `gvrp.afs_count` | Charging stations available |
| `gvrp.fuel_capacity` | Battery capacity |
| `gvrp.consumption_rate` | Energy per unit distance |
| `traffic.speed_limits` | `[min, max]` speed range (km/h) |
| `traffic.seg_duration` | Duration of each traffic segment (minutes) |
| `simulation.render_mode` | `quiet` \| `print` \| `pygame` \| `human` |

## SUMO Networks

Road networks are stored under `data/`. Supported formats: `.sumocfg` (full simulation config) or `.net.xml` (network topology only). Pre-included networks:
- `data/slc/` — Salt Lake City urban network
- `data/rand_grid*/` — procedurally generated grid networks
- `data/grid*/` — simple grid networks

## Project Structure

```
run_sim.py                    # EVRP entry point: plan → simulate → report
dronesim/
├── evrp/
│   ├── evrp_exact.py         # MILP formulation & instance generation
│   ├── planners/             # Pluggable planner implementations
│   │   ├── base.py           # BasePlanner interface (Stop, Route)
│   │   ├── gvrp_milp.py      # Gurobi MILP exact solver
│   │   ├── evrp_heuristic.py # RoutingBlocks ALNS heuristic
│   │   └── mtsp.py           # KMeans + nearest-neighbor baseline
│   └── simulator/
│       └── evrp_sim.py       # SUMO TraCI simulator
└── trsp/
    ├── trsp_sumo.py          # Gymnasium environment (SumoTRSPEnv)
    ├── trainer.py / train.py # RL training harness
    ├── main.py               # Policy evaluation entry point
    └── agents/               # Dijkstra, SARSA, DQN, PPO, A2C, DP
config/evrp/                  # YAML configs with inheritance support
data/                         # SUMO network files
runs/                         # Saved policies and simulation results
```
