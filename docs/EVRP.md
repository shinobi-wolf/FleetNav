# EVRP Module — Current State

The EVRP module solves the **outer planning layer**: assign delivery stops to a fleet of ground robots subject to fuel/battery capacity, route them through Automated Fueling Stations (AFS), and simulate execution in SUMO. It does not concern itself with how vehicles navigate between stops (that is TRSP's job).

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `infer_ta_evrp.py` | Plan → simulate → write metrics. Accepts `-c <config>` |
| `infer_ta_evrp.py --no-sim` | Plan only; skip SUMO simulation |
| `infer_ta_evrp.py --no-plot` | Suppress route plot |

Configs live under `config/ta_evrp/`. Pure EVRP runs use configs with `trsp.agent: dijkstra` and `traffic.psl.enabled: false`.

---

## Planners

All planners implement `BasePlanner.plan(config) → list[Route]` (`dronesim/evrp/planners/base.py`). Each `Route` is a sequence of `Stop` objects (depot → customers/AFS → depot).

### `milp` — GVRP Exact Solver ✅ Done

- **File:** `dronesim/evrp/planners/gvrp_milp.py`, core model in `dronesim/evrp/evrp_exact.py`
- **Config:** `config/evrp/milp.yaml`
- **Method:** Gurobi MILP with dummy-node expansion: each AFS node gets `afs_dummy_copies` copies, the depot gets `depot_dummy_copies`. This allows a vehicle to revisit the same AFS/depot multiple times without copy-count constraints.
- **Requires:** Gurobi license
- **Status:** Fully working. Produces optimal or near-optimal routes bounded by solver time limit and MIP gap.

### `heuristic` — RoutingBlocks ALNS ✅ Done

- **File:** `dronesim/evrp/planners/evrp_heuristic.py`
- **Config:** `config/evrp/heuristic.yaml`
- **Method:** Adaptive Large Neighborhood Search via the RoutingBlocks library. Fast; no license needed.
- **Status:** Fully working. Serves as the practical baseline — often matches MILP quality at a fraction of the solve time.

### `mtsp` — Multi-TSP Baseline ✅ Done

- **File:** `dronesim/evrp/planners/mtsp.py`
- **Config:** `config/evrp/mtsp.yaml`
- **Method:** KMeans clustering of customers, then nearest-neighbor TSP per cluster. Ignores fuel constraints entirely.
- **Status:** Fully working. Useful as a lower-bound reference for route length; will strand if fuel runs out.

### `td_evrp_milp` — Time-Dependent EVRP Exact Solver ✅ Done (with known limitations)

- **File:** `dronesim/evrp/planners/evrp_td_milp.py`, model in `dronesim/evrp/milp/td_evrp.py`
- **Config:** `config/evrp/td_evrp_milp.yaml`
- **Method:** Lu et al. (2020) ILP formulation. Three-segment time-of-day speed profile (congestion → free-flow → congestion at T1, T2). Arc travel times computed via Ichoua (2003) FIFO blending across segment boundaries. Decision variables: route binaries (x), arc-mode binaries (z), departure times (t), service-start times (w), fuel levels (y).
- **Status:** Implemented and producing routes. See `docs/td_evrp_milp_formulation_notes.md` and `docs/td_evrp_development_log.md` for full analysis. Key known limitations:
  - LP relaxation bound is always 0 (structural weakness of Lu's formulation; 100% MIP gap throughout solve)
  - Simplified energy model (`e_per_m × distance`, speed-independent) removes the speed-energy tradeoff that motivates TD-EVRP
  - Ichoua travel-time model in the MILP does not match the piecewise-constant simulator → planned vs. simulated makespan diverges near T1/T2 transitions
  - Objective is `w_energy × fuel + w_makespan × T_max`, not Lu's monetary objective

### `ctvtt_ga` — GA-based Planner ❌ Incomplete

- **File:** `dronesim/evrp/planners/evrp_ctvtt_ga.py`
- **Config:** `config/evrp/ctvtt_ga.yaml`
- **Status:** Scaffolding only. Not functional.

---

## Simulators

Two simulators share the same config and metrics interface. Selected via `simulation.simulator`.

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

## Traffic Models

Both simulators use a **global** traffic model (same speed applied to all active vehicles at a given time). This is distinct from TRSP's per-edge PSL model.

### `SegmentedSpeedModel` ✅ Done

- **File:** `dronesim/evrp/traffic/segmented.py`
- Random piecewise-constant speed. K time slots, each `delta_t = T/K` seconds. Speed drawn uniformly from `np.arange(v_min_ms, v_max_ms, step_ms)` at construction.
- Config keys (PSL-style, unified with TRSP): `T`, `K`, `v_min`, `v_max`, `speed_step`, `seed`
- `update(sim_time_s)` returns `True` when the slot changes; simulators broadcast the new speed to all active vehicles.

### `LuTrafficModel` ✅ Done

- **File:** `dronesim/evrp/traffic/lu.py`
- Three-segment profile matching the TD-EVRP MILP: `vc` for `t < T1`, `vf` for `T1 ≤ t < T2`, `vc` for `t ≥ T2`. Parameters from `traffic_lu` config block.

---

## Configuration

All EVRP configs inherit from `config/evrp/base.yaml` via `base:` key (deep-merge semantics).

### Key config sections

| Section | Purpose |
|---------|---------|
| `net.net_path` | Path to `.sumocfg` or `.net.xml` |
| `fleet.size` | Number of ground robots |
| `gvrp.*` | Customer count, AFS count, fuel capacity, consumption rate, speed |
| `traffic.*` | PSL-style keys: `T`, `K`, `v_min`, `v_max`, `speed_step`, `seed`; `model: segmented\|lu` |
| `simulation.simulator` | `sequential` or `concurrent` |
| `simulation.render_mode` | `quiet` / `print` / `human` |
| `simulation.runtime_refuel` | Enable emergency depot trackback when range is insufficient |
| `simulation.local_planner` | `dijkstra` (default) or `trsp` — see EVRP_TRSP.md |
| `simulation.trsp_policy` | Policy ID under `runs/trsp/`; required when `local_planner: trsp` |

### Traffic config (current base.yaml defaults)

```yaml
traffic:
  enabled: true
  model: segmented
  T: 86400.0      # 24-hour horizon
  K: 288          # 5-minute slots
  v_min: 20.0     # km/h
  v_max: 50.0     # km/h
  speed_step: 10.0
  seed: 42
```

---

## Output

| File | Description |
|------|-------------|
| `runs/evrp/<run_id>/gvrp_routes.png` | Matplotlib plot of planned routes |
| `runs/evrp/<run_id>/sim_metrics.json` | Makespan, distance, energy, recharges per vehicle |
| `runs/evrp/<run_id>/generated_gvrp_instance.json` | Problem instance definition |

---

## Known Issues / Future Work

- `ctvtt_ga` planner is not implemented
- TD-EVRP LP relaxation is permanently 0; Ichoua travel-time model diverges from simulator at traffic transition points; energy model is speed-independent
- Heuristic planner sometimes produces infeasible routes on sparse networks (documented in `docs/heuristic_planner_bug_fixes.md`)
