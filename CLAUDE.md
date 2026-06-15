# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Documentation — Read First

At the start of every conversation, read these files to get the latest progress context before doing any work:

- `docs/EVRP.md` — current state of the EVRP (Level 1) module: what is implemented, tested, and pending
- `docs/TRSP.md` — current state of the TRSP (Level 2) module: RL agent progress, experiments, and open issues
- `docs/EVRP_TRSP.md` — integration status between the two levels and cross-cutting concerns

These files are the authoritative source of truth for project progress; they supersede any description in this CLAUDE.md where they conflict.

## Architecture Docs (`arch/`)

High-level architecture diagrams and schema definitions live in `arch/`:

- `arch/dronesim_arch.jpeg` — top-level system diagram (two-level hierarchy: EVRP outer layer + TRSP inner layer)
- `arch/instance_arch.jpeg` — `Instance` object structure and data flow between planner, simulator, and local planner
- `arch/modules.yaml` — machine-readable module dependency map
- `arch/schema.yaml` — config schema reference for TA-EVRP YAML files

## Branch: `dev_trsp` — TRSP Development Only

This branch is exclusively for **Phase 2 (TRSP)** development. When working here:
- Focus only on `dronesim/trsp/`, `train_trsp.py`, `eval_trsp.py`, and `config/trsp/`
- Do **not** explore, read, or suggest changes to `dronesim/evrp/` or EVRP-related code
- The EVRP module is stable, complete, and out of scope for this branch

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

### TA-EVRP Inference (outer plan + inner navigation)
```bash
# Pure EVRP: MILP planner + static Dijkstra (no traffic)
python infer_ta_evrp.py -c config/ta_evrp/milp.yaml

# Pure EVRP: RoutingBlocks ALNS heuristic
python infer_ta_evrp.py -c config/ta_evrp/heuristic.yaml

# Pure EVRP: mTSP baseline
python infer_ta_evrp.py -c config/ta_evrp/mtsp.yaml

# Full TA-EVRP: MILP + SARSA (set trsp.policy_id in config first)
python infer_ta_evrp.py -c config/ta_evrp/milp_sarsa.yaml

# Plan only, skip SUMO simulation
python infer_ta_evrp.py -c config/ta_evrp/milp.yaml --no-sim

# Suppress route plot
python infer_ta_evrp.py -c config/ta_evrp/milp.yaml --no-plot
```

### TRSP RL Agent Training + Evaluation (Level 2)
```bash
# Train a policy
python train_trsp.py -c config/trsp/train.yaml

# Evaluate a trained policy by policy ID
python eval_trsp.py -p <policy_id>

# Run inference episodes
python infer_trsp.py -p <policy_id> -n 10
```

### Smoke Tests / Integration Tests
```bash
python scripts/smoke_test_lmd.py --config config/ta_evrp/base.yaml --steps 8
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

A Gymnasium-compliant environment (`SumoTRSPEnv` in `dronesim/trsp/envs/sumo_trsp_env.py`) registered as `SumoTRSPEnv-v0`. The env wraps SUMO; the agent observes current position, target node, traffic segment index, and steps-to-destination; actions select the next road network junction.

Package layout:
- `dronesim/trsp/__init__.py` — gym env registration
- `dronesim/trsp/envs/sumo_trsp_env.py` — `SumoTRSPEnv(gym.Env)` (under development)
- `dronesim/trsp/fleet/UGVSumo.py` — ground robot state and movement model
- `dronesim/trsp/trainer.py` — `lmd_trainer()` training loop (module-level, called by runner)
- `dronesim/trsp/agents/` — agent implementations sharing the `agent.py` base interface:
  - `DijkstraAgent` — greedy shortest-path baseline
  - `SARSAAgent` / `SARSALambdaAgent` — tabular on-policy TD control
  - `DQNAgent` — deep Q-network (CleanRL-style)
  - `PPOAgent` — proximal policy optimization (stub)

Runner scripts (project root): `train_trsp.py`, `eval_trsp.py`, `infer_trsp.py`. Policies saved under `runs/trsp/<policy_id>/`.

### Configuration System

YAML-based with **inheritance**: child configs declare `base: path/to/base.yaml` and deep-merge on top (dicts merge recursively, scalars override). `config/ta_evrp/base.yaml` holds all shared TA-EVRP parameters.

Key config sections (TA-EVRP schema):
- `evrp.planner` — `milp` | `heuristic` | `mtsp` | `td_milp`
- `trsp.agent` — `dijkstra` | `sarsa` | `dqn` | `ppo` | `a2c`
- `trsp.policy_id` — policy ID under `runs/trsp/`; required for RL agents
- `network.net_path` — path to `.sumocfg` or `.net.xml`
- `simulation.ugv.*` — fleet size, battery range, speed, service/recharge times
- `simulation.tasks.*` — customer count, AFS count
- `traffic.psl.*` — PSL model: K, v_min, v_max, speed_step, seed
- `traffic.simulated.*` — SUMO background vehicular/pedestrian traffic
- `uav_bms.*` — UAV-BMS enabled flag, fleet size, range
- `sumo.render_mode` — `quiet` | `print` | `human`

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
| `runs/ta_evrp/{run_id}/gvrp_routes.png` | Matplotlib plot of planned routes |
| `runs/ta_evrp/{run_id}/sim_metrics.json` | Makespan, distance, energy, recharges per vehicle |
| `runs/ta_evrp/{run_id}/generated_instance.json` | Problem instance definition |
| `runs/trsp/{policy_id}/policy.pkl` | Saved TRSP RL policy |

`run_id` format: `{planner}_{agent}_{YYYYMMDD_HHMMSS}` (e.g. `milp_sarsa_20260428_143200`).
