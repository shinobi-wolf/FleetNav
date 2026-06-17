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

All planners implement `EVRPPlanner.plan(instance: Instance) → list[Route]` (`dronesim/evrp_/base.py`). Each `Route` is a sequence of `Stop` objects (depot → customers/AFS → depot). `Stop` carries both `sumo_node_id` (for TraCI) and `node_name` (logical label, e.g. `C1`, `F3_d2`).

Planners live in `dronesim/evrp_/` and are dispatched via `REGISTRY` in `dronesim/evrp_/__init__.py`. The outer orchestrator is `GlobalPlanner` (`dronesim/ta_evrp/planner/evrp.py`), which selects the right planner from `REGISTRY` based on `instance.planner_name`.

### `milp` — GVRP Exact Solver ✅ Done

- **File:** `dronesim/evrp_/milp.py` (Gurobi formulation inline)
- **Config:** `config/ta_evrp/milp.yaml`
- **Method:** Gurobi MILP with dummy-node expansion: each AFS node gets `afs_dummy_copies` copies, the depot gets `depot_dummy_copies`. Allows a vehicle to revisit the same AFS/depot multiple times without copy-count constraints.
- **Requires:** Gurobi license
- **Status:** Fully working. Produces optimal or near-optimal routes bounded by solver time limit and MIP gap.
- **Fuel semantics:** Q = `battery_capacity_L * 1000 / consumption_rate_L_per_km` (range in metres); r = 1.0 (dimensionless). MILP variables y[j] are forward-propagated to show physically meaningful fuel levels, not raw solver slack values.

### `heuristic` — RoutingBlocks ALNS ✅ Done

- **File:** `dronesim/evrp_/heuristic.py`
- **Config:** `config/ta_evrp/heuristic.yaml`
- **Method:** Adaptive Large Neighborhood Search via the RoutingBlocks library. Fast; no license needed.
- **Status:** Fully working. Serves as the practical baseline — often matches MILP quality at a fraction of the solve time.

### `mtsp` — Multi-TSP Baseline ✅ Done

- **File:** `dronesim/evrp_/mtsp.py`
- **Config:** `config/ta_evrp/mtsp.yaml`
- **Method:** KMeans clustering of customers, then nearest-neighbor TSP per cluster. Ignores fuel constraints entirely.
- **Status:** Fully working. Useful as a lower-bound reference for route length; will strand if fuel runs out.

### `td_milp` — Time-Dependent MILP ⚠️ No config yet

- **File:** `dronesim/evrp_/td_milp.py` (Lu et al. 2020, inline)
- **Status:** Implemented and in `REGISTRY`. No `config/ta_evrp/td_milp.yaml` exists yet. Requires `ta_config["traffic_lu"]` and `ta_config["cost_model"]` sections.

### `ctvtt_ga` — GA-based Planner ❌ Not ported

- Exists in `dronesim/evrp/planners/evrp_ctvtt_ga.py` (old module). Not ported to `evrp_/` and not in `REGISTRY`.

---

## Instance

### `Instance` (`dronesim/ta_evrp/instance/instance.py`) ✅ Done

`Instance` is the unified problem container built once per run and shared across `GlobalPlanner`, `LocalPlanner`, and `Simulator`. It bundles everything that used to be passed as separate arguments.

Key attributes:
- `network` — `Network` object: `sumolib.net.Net`, `nx.DiGraph` (with `length`/`id` edge attrs), node positions
- `traffic` — `PSLTrafficModel`: per-edge piecewise speed profiles
- `problem` — `Problem`: distance matrix, depot/customer/AFS node lists, `params` dict (SI units), `generation_info`
- `ta_config` — raw YAML config dict (used by `TDMILPPlanner` and `HeuristicPlanner` for planner-specific settings)
- Convenience properties: `planner_name`, `agent_name`, `policy_id`, `net_path`, `traffic_enabled`

---

## Simulator

### `Simulator` (`dronesim/ta_evrp/simulator/`) ✅ Done

The new `Simulator` is constructed as `Simulator(instance, local_planner)` where `local_planner` is a `LocalPlanner` instance or `None` (for Dijkstra-only runs).

### Local Route Resolution

When TRSP is active, the simulator dispatches to:
- `LocalPlanner.plan(src, dst, sim_time_s)` — RL-policy-guided navigation (see EVRP_TRSP.md); only constructed when `policy_id is not None`
- Dijkstra fallback on the `instance.network.graph` when `local_planner is None`

When TRSP is active, the **edges driven in SUMO come from the TRSP agent**, but the **distance charged for fuel accounting is always the Dijkstra shortest-path distance** — keeping fuel consumption consistent with the EVRP planner's distance matrix.

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

**Unit convention:** config files use human-friendly units (km, km/h, hours, litres, L/km). The runner script (`infer_ta_evrp.py`) converts to SI before passing to any package function. The `dronesim/evrp/` package itself is unit-free (SI only).

Fuel specifically: `battery_capacity_L` and `consumption_rate_L_per_km` are combined into a single **range in metres** (`Q = battery_capacity_L * 1000 / consumption_rate_L_per_km`) with dimensionless consumption rate `r = 1.0`. The MILP fuel constraint then reads: `y[j] ≤ y[i] − d[i,j]`, where all quantities are in metres.

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

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/graph_verification.ipynb` | Visualize the NX routing graph and sampled instance nodes (depot, customers, AFS) from a config file; useful for verifying network loading and instance generation |
| `notebooks/results_analysis.ipynb` | Aggregate and compare TA-EVRP results across planners and agents from `results/ta_evrp_results.csv` |

---

## Known Issues / Future Work

- `ctvtt_ga` planner not ported to `evrp_/`; not in `REGISTRY`
- `td_milp` has no config file (`config/ta_evrp/td_milp.yaml`); can't be selected via CLI yet
- Heuristic planner (ALNS) can return `feasible=False` routes on tight-range configs; the MILP correctly reports infeasibility in the same cases
- `dronesim/evrp/` (old module) still present; can be removed now that `Simulator` in `dronesim/ta_evrp/simulator/` is complete
- SUMO turn restrictions cause `No connection between edge X and Y` errors when the NX Dijkstra path crosses a physically disallowed turn; addressed by rebuilding the net without connection files or handling at planning time
