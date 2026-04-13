# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Excluded Directories

Do **not** explore, read, or reference any code in `tmp`
<!-- . These include: -->
<!-- - `tmp/` (top-level) -->
<!-- - `dronesim/trsp/tmp.py`
- `dronesim/trsp/tmp.ipynb` -->

## Project Overview

DroneSim is a two-level hierarchical routing framework for autonomous last-mile delivery (LMD). It jointly addresses dynamic urban navigation under traffic uncertainty and limited vehicle endurance due to battery constraints.

### Level 1 — EVRP (Electric Vehicle Routing Problem)
The outer planning layer solves the **Green Vehicle Routing Problem (GVRP)**: assign delivery stops to a fleet of ground robots while respecting battery/fuel capacity and routing them through Automated Fueling Stations (AFS) as needed. This layer is augmented with a **UAV-assisted Battery Management System (BMS)**: aerial vehicles deliver replacement batteries to ground robots in the field, eliminating costly depot-return detours. The battery dispatch problem is modeled as a **Segment-VRP**, selecting strategic rendezvous points along robot routes to minimize UAV travel cost.

### Level 2 — TRSP (Traffic-Responsive Shortest Path)
The inner navigation layer handles **node-to-node routing** for each ground robot as it executes the Level 1 plan. Edge traversal times vary due to piecewise time-varying traffic (pedestrian/vehicular congestion modeled as speed limit segments). A reinforcement learning agent learns policies that adapt to these conditions, outperforming static shortest-path baselines.

## Running the System

### EVRP Planning + Simulation (Level 1)
```bash
# Exact MILP solver (best quality, slowest — requires Gurobi license)
python run_sim.py -c config/evrp/milp.yaml --planner milp

# Fast heuristic (RoutingBlocks ALNS)
python run_sim.py -c config/evrp/heuristic.yaml --planner heuristic

# Multi-TSP baseline (ignores fuel constraints)
python run_sim.py -c config/evrp/mtsp.yaml --planner mtsp

# Plan only, skip SUMO simulation
python run_sim.py -c config/evrp/gvrp.yaml --no-sim

# Suppress route plot
python run_sim.py -c config/evrp/milp.yaml --no-plot
```

### TRSP RL Agent Training + Evaluation (Level 2)
```bash
# Train a policy
python dronesim/trsp/train.py

# Evaluate a trained policy by policy ID
python dronesim/trsp/main.py -p <policy_id>
```

### Smoke Tests / Integration Tests
```bash
python scripts/smoke_test_lmd.py --config config/evrp/base.yaml --steps 8
python dronesim/trsp/test.py
python dronesim/trsp/test_road.py
```

## Architecture

### EVRP Module (`dronesim/evrp/`)

**Workflow** orchestrated by `run_sim.py`:
1. Load YAML config (with inheritance) → build SUMO network graph → sample customers/depot/AFS → compute distance matrix
2. Solve via pluggable planner → returns `list[Route]` (each Route = sequence of `Stop` objects)
3. Replay routes in SUMO via TraCI, model fuel/battery consumption, collect metrics → output `gvrp_routes.png` and `sim_metrics.json`

**Planner registry** (`dronesim/evrp/planners/__init__.py`): All planners implement `BasePlanner.plan(config) → list[Route]`.
- `GVRPMILPPlanner` — Gurobi MILP exact solver with dummy-node expansion for AFS revisits
- `MTSPPlanner` — KMeans clustering + nearest-neighbor TSP (fast baseline, ignores fuel)
- `EVRPHeuristicPlanner` — RoutingBlocks ALNS heuristic

**Core MILP formulation** lives in `dronesim/evrp/evrp_exact.py` — instance generation, dummy-node expansion, and Gurobi model construction.

**Simulator** (`dronesim/evrp/simulator/evrp_sim.py`): Wraps SUMO TraCI to execute routes step-by-step, model battery/fuel consumption, and apply piecewise traffic speed limits (5-minute segments).

### TRSP Module (`dronesim/trsp/`)

A Gymnasium-compliant environment (`SumoTRSPEnv` in `trsp_sumo.py`) wrapping SUMO. The agent observes current position, pending tasks, traffic segment index, and battery level; actions select the next road network junction. Agents in `dronesim/trsp/agents/` share a common `agent.py` interface:
- `DijkstraAgent` — greedy shortest-path baseline
- `SARSAAgent` / `SARSALambdaAgent` — tabular Q-learning
- `DQNAgent` / `PPOAgent` / `A2CAgent` — deep RL via CleanRL
- `DPAgent` — dynamic programming

Training: `train.py` / `trainer.py`. Policies saved as `.pkl` under `runs/trsp/`.

### Configuration System

YAML-based with **inheritance**: child configs declare `base: path/to/base.yaml` and deep-merge on top (dicts merge recursively, scalars override). `config/evrp/base.yaml` holds shared fleet/network/vehicle parameters.

Key config sections:
- `net.net_path` — path to `.sumocfg` or `.net.xml`
- `fleet.size` — number of ground robots
- `gvrp.*` — customer count, AFS count, fuel/battery capacity, consumption rate, distance unit
- `solver.*` — MILP time limit, MIP gap (MILP planner only)
- `traffic.*` — piecewise speed limit range, segment duration (default 5 min)
- `simulation.render_mode` — `quiet` (silent) | `print` (log events to stdout) | `human` (sumo-gui + pygame window; applies `sumo.delay`)

### Key Dependencies

- `gurobipy` — MILP solver (requires commercial license)
- `sumolib` / `traci` — SUMO traffic simulation interface
- `routingblocks` — heuristic VRP solver (ALNS)
- `gymnasium` — RL environment API
- `cleanrl` — DQN/PPO/A2C implementations
- `networkx` — road network graph

### Output Files

| File | Description |
|------|-------------|
| `gvrp_routes.png` | Matplotlib plot of planned routes |
| `sim_metrics.json` | Makespan, distance, energy, recharges per vehicle |
| `generated_gvrp_instance.json` | Problem instance definition |
| `runs/trsp/{agent}/policy.pkl` | Saved RL policy |
