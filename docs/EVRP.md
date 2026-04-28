# EVRP Module — Current State

The EVRP module solves the **outer planning layer**: assign delivery stops to a fleet of ground robots subject to fuel/battery capacity, route them through Automated Fueling Stations (AFS), and simulate execution in SUMO. It does not concern itself with how vehicles navigate between stops (that is TRSP's job).

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `infer_ta_evrp.py -c <config>` | Plan → simulate → write metrics |
| `infer_ta_evrp.py --no-sim` | Plan only; skip SUMO simulation |
| `infer_ta_evrp.py --no-plot` | Suppress route plot |

Configs live under `config/ta_evrp/`. Pure EVRP runs use configs with `trsp.agent: dijkstra` and `traffic.psl.enabled: false`.

---

## Planners

All planners implement `BasePlanner.plan(config) → list[Route]` (`dronesim/evrp/planners/base.py`). Each `Route` is a sequence of `Stop` objects (depot → customers/AFS → depot).

### `milp` — GVRP Exact Solver ✅ Done

- **File:** `dronesim/evrp/planners/gvrp_milp.py`, core model in `dronesim/evrp/evrp_exact.py`
- **Config:** `config/ta_evrp/milp.yaml`
- **Method:** Gurobi MILP with dummy-node expansion: each AFS node gets `afs_dummy_copies` copies, the depot gets `depot_dummy_copies`. This allows a vehicle to revisit the same AFS/depot multiple times without copy-count constraints.
- **Requires:** Gurobi license
- **Status:** Fully working. Produces optimal or near-optimal routes bounded by solver time limit and MIP gap.

### `heuristic` — RoutingBlocks ALNS ✅ Done

- **File:** `dronesim/evrp/planners/evrp_heuristic.py`
- **Config:** `config/ta_evrp/heuristic.yaml`
- **Method:** Adaptive Large Neighborhood Search via the RoutingBlocks library. Fast; no license needed.
- **Status:** Fully working. Serves as the practical baseline — often matches MILP quality at a fraction of the solve time.

### `mtsp` — Multi-TSP Baseline ✅ Done

- **File:** `dronesim/evrp/planners/mtsp.py`
- **Config:** `config/ta_evrp/mtsp.yaml`
- **Method:** KMeans clustering of customers, then nearest-neighbor TSP per cluster. Ignores fuel constraints entirely.
- **Status:** Fully working. Useful as a lower-bound reference for route length; will strand if fuel runs out.

### `td_milp` — Time-Dependent EVRP Exact Solver ✅ Done (with known limitations)

- **File:** `dronesim/evrp/planners/evrp_td_milp.py`, model in `dronesim/evrp/milp/td_evrp.py`
- **Config:** `config/ta_evrp/milp_td.yaml`
- **Method:** Lu et al. (2020) ILP formulation. Ichoua (2003) FIFO travel-time blending across PSL slot boundaries. Decision variables: route binaries (x), arc-mode binaries (z), departure times (t), service-start times (w), fuel levels (y).
- **Status:** Implemented and producing routes. Key known limitations:
  - LP relaxation bound is always 0 (structural weakness; 100% MIP gap throughout solve)
  - Simplified energy model (`e_per_m × distance`, speed-independent)
  - Ichoua travel-time model in MILP does not perfectly match piecewise-constant simulator

### `ctvtt_ga` — GA-based Planner ❌ Incomplete

- **File:** `dronesim/evrp/planners/evrp_ctvtt_ga.py`
- **Status:** Scaffolding only. Not functional.

---

## Simulators

Two simulators share the same config and metrics interface. Selected via `sumo.simulator` in config (mapped to `simulation.simulator` internally).

### Sequential (`evrp_sim_sequential.py`) ✅ Done

`EVRPSequentialSimulator` — drives each vehicle to completion before starting the next. Blocking `_drive()` loop per leg. Simpler, easier to debug.

### Concurrent (`evrp_sim_concurrent.py`) ✅ Done

`EVRPConcurrentSimulator` — extends `EVRPSequentialSimulator`. All vehicles advance simultaneously within each SUMO step via a per-vehicle state machine (AT_NODE / IN_TRANSIT / DONE). Produces correct makespan (vehicles overlap in time).

### Local Route Resolution

Both simulators expose `_resolve_route(src, dst, sim_time_s) → list[str]` which dispatches to either:
- `_find_edge_route()` — NetworkX Dijkstra on the SUMO net (default)
- `TRSPLocalPlanner.plan()` — RL-policy-guided navigation (when configured; see EVRP_TRSP.md)

Emergency trackbacks to depot always use Dijkstra regardless of `local_planner` setting.

---

## Traffic Model

Both simulators use a **PSL (Piecewise Speed Limit)** model: each edge gets its own independent speed profile over K time slots, matching the TRSP traffic model.

### `PSLTrafficModel` ✅ Done

- **File:** `dronesim/evrp/traffic/segmented.py`
- Per-edge piecewise-constant speed profiles. K time slots, each `delta_t = max_time_s / K` seconds. Each edge independently draws speeds from `[v_min, v_max]` in steps of `speed_step`, seeded via `random.Random(seed)`.
- Config keys: `K`, `v_min`, `v_max`, `speed_step`, `seed`, `profile_type` (under `traffic.psl` in TA-EVRP config)
- Interface: `get_slot(t) → int`, `get_speed(edge_id, slot) → float`
- At each slot transition, `traci.edge.setMaxSpeed(edge_id, speed)` is called for every edge — SUMO vehicles are constrained automatically. No per-vehicle `setMaxSpeed` needed.
- Same model and RNG approach as `dronesim/trsp/traffic.py`; config keys unified.

---

## Configuration

All TA-EVRP configs inherit from `config/ta_evrp/base.yaml` via `base:` key (deep-merge semantics).

### Key config sections

| Section | Purpose |
|---------|---------|
| `network.net_path` | Path to `.sumocfg` or `.net.xml` |
| `evrp.planner` | `milp` \| `heuristic` \| `mtsp` \| `td_milp` |
| `trsp.agent` | `dijkstra` \| `sarsa` \| `dqn` \| `ppo` |
| `trsp.policy_id` | Policy ID under `runs/trsp/`; required for RL agents |
| `simulation.ugv.*` | Fleet size, battery range, speed, service/recharge times |
| `simulation.tasks.*` | Customer count, AFS count |
| `traffic.psl.*` | `enabled`, `K`, `v_min`, `v_max`, `speed_step`, `seed`, `profile_type` |
| `traffic.simulated.*` | SUMO background vehicular/pedestrian traffic |
| `sumo.simulator` | `sequential` or `concurrent` |
| `sumo.render_mode` | `quiet` / `print` / `human` |

### Traffic config example (base.yaml defaults)

```yaml
traffic:
  psl:
    enabled: true
    K: 1              # single slot = constant per-edge speed
    v_min: 40.0       # km/h
    v_max: 100.0      # km/h
    speed_step: 60.0  # gives two levels: {40, 100} km/h
    seed: 42
    profile_type: random
```

---

## Output

| File | Description |
|------|-------------|
| `runs/ta_evrp/<run_id>/gvrp_routes.png` | Matplotlib plot of planned routes |
| `runs/ta_evrp/<run_id>/sim_metrics.json` | Makespan, distance, energy, recharges per vehicle |
| `runs/ta_evrp/<run_id>/generated_instance.json` | Problem instance definition |

`run_id` format: `{planner}_{agent}_{YYYYMMDD_HHMMSS}` (e.g. `milp_dijkstra_20260428_143200`).

---

## Known Issues / Future Work

- `ctvtt_ga` planner is not implemented
- TD-EVRP LP relaxation is permanently 0; energy model is speed-independent
- Heuristic planner sometimes produces infeasible routes on sparse networks
