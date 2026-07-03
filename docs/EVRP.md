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

### `td_milp` — Time-Dependent MILP ✅ Done

- **File:** `dronesim/evrp/td_milp.py` (Lu et al. 2020, inline Gurobi)
- **Config:** `config/ta_evrp/td_milp.yaml`
- **Method:** The `milp` G-VRP formulation with **one** change — arc travel time depends on the **departure time** from the tail node, following the three-period time-of-day congestion profile (morning peak `[0,T1)` at `v_c`, free-flow `[T1,T2)` at `v_f`, evening peak `[T2,u0]` at `v_c`). The min-distance objective, dummy-vertex expansion, and distance-based fuel constraints (`Q` in metres, `r`) are **identical** to `milp`. Cargo/demand is not modelled. Speed is fixed to each period's limit (no per-arc speed optimisation).
- **Formulation:** The constant-time propagation of `milp` (C7–C9) is replaced by per-arc departure-interval variables `z[i,j,m]` / `td[i,j,m]` and service-start times `w[j]`, with the paper's piecewise-linear travel time `T(t)=A_T+B_T·t` (Eq. 1) and constraints Eqs. 16/17/22/23. Helpers `_interval_bounds`, `_travel_coefs` give per-arc `(A_T, B_T)` and bounds.
- **Traffic source:** reads the **same** `traffic.td.*` block the Simulator uses (`v_congestion_kmh`, `v_freeflow_kmh`, `T1_hours`, `T2_hours`); horizon `u0 = Tmax`. Set `traffic.model: td` for end-to-end plan/sim consistency.
- **Objective — makespan first:** minimises `w_makespan · makespan_s + w_distance · distance_m` (defaults `1.0` / `0.001`). Makespan is the metric the congestion profile informs and where `td_milp` beats `milp`/`heuristic`; `milp` already owns distance, and `td_milp`'s stricter time constraints mean it can never undercut `milp` on distance. Minimising makespan drives a *depart-ASAP / no-waiting* schedule, which matches how the Simulator drives — so the planned makespan win carries into the simulated metric. Set `w_makespan: 0` for the pure min-distance, feasibility-only variant. A valid lower bound (farthest customer's free-flow round trip) tightens the otherwise-weak min-max relaxation.
- **Requires:** Gurobi license. The model is larger than `milp` (adds `z`/`td` per arc × valid departure intervals) and the min-max makespan objective is harder — expect to hit `time_limit_s` with a non-trivial gap on bigger instances, though the incumbent is strong. Solver params come from `evrp.td_milp.*`.
- **FIFO caveat:** the travel-time formula (paper Eq. 1) assumes an arc spans at most two adjacent periods. If the longest arc exceeds `min((T2−T1)·v_f, T1·v_c, (u0−T2)·v_c)` the planner prints a `[warn]` and the travel time is approximate for those arcs (the Simulator is always exact, integrating edge-by-edge). Widen the `T1`–`T2` gap or raise `v_freeflow_kmh` to remove it.

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

**Delivery-robot (PED) mode.** With `traffic.model: ped` + `fleet.type: delivery_robot`, the Simulator drives each leg as a SUMO **person** walking on sidewalks (`traci.person.add` + `appendWalkingStage`) instead of a road vehicle, registers an `adr_vtype`, and sets the robot's speed every step from live sidewalk crowding (Weidmann). The routing graph is the pedestrian graph built by `Network(ped_mode=True)`. See `docs/EVRP_TRSP.md` → *Traffic Model Unification → PED* for the full flow.

---

## Traffic Model

The simulator uses a **PSL (Piecewise Speed Limit)** model: each edge gets its own independent speed profile over K time slots, matching the TRSP traffic model.

### `PSLTrafficModel` ✅ Done

- **File:** `dronesim/evrp/traffic/segmented.py`
- Per-edge piecewise-constant speed profiles. K time slots, each `delta_t = max_time_s / K` seconds. Each edge independently draws speeds from `[v_min, v_max]` in steps of `speed_step`, seeded via `random.Random(seed)`.
- Config keys (under `traffic.psl` in TA-EVRP config): `K`, `v_min_kmh`, `v_max_kmh`, `speed_step_kmh`, `seed`, `profile_type`
- Interface: `get_slot(t) → int`, `get_speed(edge_id, slot) → float` (returns m/s)
- At each slot transition, `traci.edge.setMaxSpeed(edge_id, speed)` is called for every edge — SUMO vehicles are constrained automatically.

### `TDTrafficModel` ✅ Done

- **File:** `dronesim/ta_evrp/instance/traffic.py` (subclass of `PSLTrafficModel`)
- Three-period time-of-day congestion model (Lu et al. 2020, Fig. 1), selected via `traffic.model: td`. Global speeds — **every** edge gets the identical profile `[v_c, v_f, v_c]`: slot 0 (`[0,T1)`) and slot 2 (`[T2,…]`) at congestion speed `v_c`, slot 1 (`[T1,T2)`) at free-flow `v_f`. Only `get_slot()` differs from PSL — boundaries at `T1`/`T2` instead of the cyclic `delta_t` grid.
- Config keys (under `traffic.td`): `enabled`, `v_congestion_kmh`, `v_freeflow_kmh`, `T1_hours`, `T2_hours` (default `T1=u0/3`, `T2=2u0/3`).
- Same `get_slot`/`get_speed` interface, so the Simulator applies it through the same `setMaxSpeed`/`setSpeed` code paths as PSL (both are gated by `Simulator._psl_like`). The `td_milp` planner reads the same `traffic.td` block, so plan and simulation use identical speeds.

### PED — live sidewalk speed (delivery robot) ✅ Done

- Selected by `traffic.model: ped` (with `fleet.type: delivery_robot`). Like SVC, the PSL model is **disabled** — `Traffic` returns `PSLTrafficModel(enabled=False, free_flow_ms=<robot walking speed>)`. The Simulator never calls `setMaxSpeed`/`setSpeed` for slots; instead it sets each robot's speed per step from local pedestrian crowding via `dronesim.trsp.ped_speed.weidmann_speed` (params from `traffic.ped_speed_override`). `Network(ped_mode=True)` builds the sidewalk graph (pedestrian-lane edges only, each tagged with `sidewalk_lane`).

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
| `evrp.planner` | `milp` \| `td_milp` \| `heuristic` \| `mtsp` |
| `evrp.td_milp.*` | solver params (`time_limit_s`, `mip_gap`, `output_flag`, dummy copies) + objective weights `w_makespan` (default 1.0) / `w_distance` (default 0.001) |
| `trsp.agent` | `dijkstra` \| `sarsa` \| `dqn` \| `ppo` |
| `trsp.policy_id` | Policy ID under `runs/trsp/`; required for RL agents |
| `simulation.ugv.battery_capacity_L` | Full-charge range capacity in litres |
| `simulation.ugv.consumption_rate_L_per_km` | Fuel consumption rate in L/km |
| `simulation.ugv.free_flow_speed_kmh` | Vehicle free-flow speed in km/h |
| `simulation.ugv.max_duration_h` | Hard route time limit in hours |
| `simulation.ugv.runtime_refuel` | Enable emergency depot detours (bool) |
| `simulation.tasks.*` | Customer count, AFS count |
| `traffic.model` | `psl` \| `td` \| `svc` \| `ped` |
| `fleet.type` | `ev` (default) \| `delivery_robot` (ADR; walks sidewalks, requires `traffic.model: ped`) |
| `traffic.ped_speed_override.*` | `enabled`, `gamma`, `density_jam`, `v_min` — Weidmann sidewalk crowding → robot speed |
| `traffic.psl.*` | `enabled`, `K`, `v_min_kmh`, `v_max_kmh`, `speed_step_kmh`, `seed`, `profile_type` |
| `traffic.td.*` | `enabled`, `v_congestion_kmh`, `v_freeflow_kmh`, `T1_hours`, `T2_hours` (3-period TD model) |
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

- `ctvtt_ga` planner not ported to `evrp/`; not in `REGISTRY`
- Heuristic planner (ALNS) can return `feasible=False` routes on tight-range configs; the MILP correctly reports infeasibility in the same cases
- `dronesim/evrp/` (old module) still present; can be removed now that `Simulator` in `dronesim/ta_evrp/simulator/` is complete
- SUMO turn restrictions cause `No connection between edge X and Y` errors when the NX Dijkstra path crosses a physically disallowed turn; addressed by rebuilding the net without connection files or handling at planning time
