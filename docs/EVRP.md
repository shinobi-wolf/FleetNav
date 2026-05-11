# EVRP Module — Current State

The EVRP module solves the **outer planning layer**: assign delivery stops to a fleet of ground robots subject to fuel/battery capacity, route them through Automated Fueling Stations (AFS), and simulate execution in SUMO. It does not concern itself with how vehicles navigate between stops (that is TRSP's job).

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `infer_ta_evrp.py -c <config>` | Plan → simulate → write metrics |
| `infer_ta_evrp.py --no-sim` | Plan only; skip SUMO simulation |
| `infer_ta_evrp.py --no-plot` | Suppress route plot |

Configs live under `config/ta_evrp/`. "Pure EVRP" typically means `trsp.agent: dijkstra`; traffic can be enabled or disabled independently via `traffic.psl.enabled`.

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

### Time-Dependent MILP (TD-EVRP) ⚠️ Removed from current TA-EVRP configs

Earlier iterations included a TD-EVRP MILP config (`milp_td.yaml`) and a Lu/Ichoua-style time-dependent formulation. This config is not present in the current `config/ta_evrp/` set, and is not part of the default TA-EVRP experiment pipeline.

### `ctvtt_ga` — GA-based Planner ❌ Incomplete

- **File:** `dronesim/evrp/planners/evrp_ctvtt_ga.py`
- **Status:** Scaffolding only. Not functional.

---

## Simulator

### `EVRPSimulator` (`dronesim/evrp/simulator/evrp_sim.py`) ✅ Done

Single time-step-driven simulator: all vehicles advance concurrently within each SUMO step via a per-vehicle state machine (`enroute` / `waiting` / `stranded` / `done`). This produces correct makespan (vehicles overlap in time) while keeping the implementation in one file.

Key per-vehicle state tracked in `_VehicleState`:
- `current_fuel` — remaining range in metres
- `curr_stop_idx` — index into the planned `Route.stops` list
- `status` — `enroute` | `waiting` | `stranded` | `done`
- `curr_traci_veh_id` — active SUMO vehicle ID for the in-transit leg

### `runtime_refuel` ✅ Done

When `simulation.ugv.runtime_refuel: true` in config, the simulator checks at each stop arrival whether the vehicle can reach the next planned stop and still make it back to the depot. If not, it inserts an unplanned emergency detour to the depot for a full recharge before continuing.

- **Check:** `dist(curr→next) + dist(next→depot) > current_fuel` (both distances via Dijkstra / SP)
- **Detour:** spawns an emergency leg to the depot; on arrival refuels to `ugv_range` (full); applies `recharge_time_s` dwell; then spawns a resume leg to the originally planned next stop
- **Metrics:** emergency recharges are counted in `num_recharges`; extra distance driven is recorded

### Local Route Resolution

`EVRPSimulator._resolve_route(src, dst, sim_time_s) → (edge_ids, dist_m)` dispatches to:
- `TRSPLocalPlanner.plan()` — RL-policy-guided navigation (when `trsp.agent != dijkstra`; see EVRP_TRSP.md)
- `_find_edge_route()` — NetworkX Dijkstra on the SUMO net (default / fallback)

When TRSP is active, the **edges driven in SUMO come from the TRSP agent**, but the **distance returned for fuel accounting is always the Dijkstra shortest-path distance** — keeping fuel consumption consistent with the EVRP planner's distance matrix.

---

## Traffic Model

The simulator uses a **PSL (Piecewise Speed Limit)** model: each edge gets its own independent speed profile over K time slots, matching the TRSP traffic model.

### `PSLTrafficModel` ✅ Done

- **File:** `dronesim/evrp/traffic/segmented.py`
- Per-edge piecewise-constant speed profiles. K time slots, each `delta_t = max_time_s / K` seconds. Each edge independently draws speeds from `[v_min, v_max]` in steps of `speed_step`, seeded via `random.Random(seed)`.
- Config keys (under `traffic.psl` in TA-EVRP config): `K`, `v_min_kmh`, `v_max_kmh`, `speed_step_kmh`, `seed`, `profile_type`
- Interface: `get_slot(t) → int`, `get_speed(edge_id, slot) → float` (returns m/s)
- At each slot transition, `traci.edge.setMaxSpeed(edge_id, speed)` is called for every edge — SUMO vehicles are constrained automatically.

---

## Configuration

All TA-EVRP configs inherit from `config/ta_evrp/base.yaml` via `base:` key (deep-merge semantics).

**Unit convention:** config files use human-friendly units (km, km/h, hours, litres). The runner script (`infer_ta_evrp.py`) converts these to SI (m, m/s, s, L/m) before passing to any package function. The `dronesim/evrp/` package itself is unit-free (SI only).

Special exceptions that keep seconds/ms in config: `sumo.step_length` (seconds), `sumo.delay` (ms), `evrp.milp.time_limit_s` (seconds).

### Key config sections

| Section | Purpose |
|---------|---------|
| `network.net_path` | Path to `.sumocfg` or `.net.xml` |
| `evrp.planner` | `milp` \| `heuristic` \| `mtsp` |
| `trsp.agent` | `dijkstra` \| `sarsa` \| `dqn` \| `ppo` |
| `trsp.policy_id` | Policy ID under `runs/trsp/`; required for RL agents |
| `simulation.ugv.battery_capacity_L` | Full-charge range capacity in litres |
| `simulation.ugv.consumption_rate_L_per_km` | Fuel consumption rate in L/km |
| `simulation.ugv.free_flow_speed_kmh` | Vehicle free-flow speed in km/h |
| `simulation.ugv.max_duration_h` | Hard route time limit in hours |
| `simulation.ugv.runtime_refuel` | Enable emergency depot detours (bool) |
| `simulation.tasks.*` | Customer count, AFS count |
| `traffic.psl.*` | `enabled`, `K`, `v_min_kmh`, `v_max_kmh`, `speed_step_kmh`, `seed`, `profile_type` |
| `traffic.simulated.*` | SUMO background vehicular/pedestrian traffic |
| `sumo.render_mode` | `quiet` / `print` / `human` |

### Traffic config example (base.yaml defaults)

```yaml
traffic:
  psl:
    enabled: true
    K: 1               # single slot = constant per-edge speed
    v_min_kmh: 40.0
    v_max_kmh: 100.0
    speed_step_kmh: 60.0   # gives two levels: {40, 100} km/h
    seed: 42
    profile_type: random
```

---

## Output

| File | Description |
|------|-------------|
| `runs/ta_evrp/<run_id>/gvrp_routes.png` | Matplotlib plot of planned routes |
| `runs/ta_evrp/<run_id>/sim_metrics.json` | Makespan, distance, recharges per vehicle |
| `runs/ta_evrp/<run_id>/generated_instance.json` | Problem instance definition |
| `runs/ta_evrp/<run_id>/planned_stops.json` | Planned stop sequence per vehicle (depot/customers/AFS) |
| `runs/ta_evrp/<run_id>/local_routes.json` | Per-leg driven edge lists (what the simulator actually routed between stops) |

`run_id` format: `{planner}_{agent}_{YYYYMMDD_HHMMSS}` (e.g. `milp_dijkstra_20260428_143200`).

---

## Known Issues / Future Work

- `ctvtt_ga` planner is not implemented
- Heuristic planner sometimes produces infeasible routes on sparse networks
- `milp/td_evrp.py` still contains internal unit conversions (skipped during SI cleanup)
