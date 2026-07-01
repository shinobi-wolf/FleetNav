# TA-EVRP Integration — Current State

This document covers the connection between the two modules: how the EVRP outer planner hands legs to the TRSP inner navigator, what fuel accounting applies, and how the traffic models relate.

---

## What Is Integrated

The `Simulator` can use a trained TRSP RL agent to navigate between every pair of consecutive stops, replacing the default Dijkstra shortest-path routing. The change is config-driven: when `trsp.policy_id` is set, a `LocalPlanner` is constructed and injected into the `Simulator`; otherwise only Dijkstra is used.

---

## Config

Set in `config/ta_evrp/` configs under `trsp` and `traffic`:

```yaml
trsp:
  agent: dijkstra     # dijkstra | dsp | sarsa | sarsa_svc | dqn | dqn_svc | a2c | a2c_svc | ppo | ppo_svc
  policy_id: null     # config dir under runs/trsp/; required for RL agents and dsp

traffic:
  model: psl          # psl | td | svc
  psl:
    enabled: true
    K: 2
    ...
  td:
    enabled: true
    v_congestion_kmh: 10.0
    v_freeflow_kmh: 40.0
    T1_hours: 0.0277778
    T2_hours: 0.0555556
  svc:
    route_file: null  # path to rvt_routes.xml; required when model: svc
```

Example — switch to DQN-SVC navigation with SVC background traffic:

```yaml
traffic:
  model: svc
  svc:
    route_file: data/grid_road/traffic_routes/svc_42/rvt_routes.xml

trsp:
  agent: dqn_svc
  policy_id: b3ef6da6   # loads runs/trsp/b3ef6da6/config.yaml + policy.pkl
```

The policy ID must correspond to a directory under `runs/trsp/` containing `config.yaml` and `policy.pkl`.

Run with: `python infer_ta_evrp.py -c config/ta_evrp/milp_policy.yaml` (or `milp_dsp.yaml` for the DSP baseline).

---

## How It Works

### `LocalPlanner` (`dronesim/ta_evrp/planner/trsp.py`)

Constructed in `infer_ta_evrp.py` when `policy_id is not None`, then injected into `Simulator`. Steps:

1. Loads the TRSP config from `runs/trsp/<policy_id>/config.yaml`.
2. Calls `gym.make("SumoTRSPEnv-v0", config=trsp_config)` to build the road graph and PSL traffic model. **SUMO is not started** — `gym.make()` only reads the net file via `sumolib`.
3. Loads the Q-table from `runs/trsp/<policy_id>/policy.pkl`.
4. Exposes `plan(src_node, dst_node, sim_time_s) → list[str]` and `plan_edges_and_nodes(...)` (for route logging).

### `plan()` — Graph-Only Navigation

No SUMO call. Pure Python graph traversal:

```
slot = traffic.get_slot(sim_time_s)
curr_node = src_node
visited = set()
while curr_node != dst_node:
  if curr_node already visited: fall back to full-leg Dijkstra
  compute steps-to-dest for each neighbour (directed SPL)
  best_dir = argmin(steps)
  state = (graph_node_idx, target_node_idx, slot)
  action = argmax(Q[state]) if state in Q else best_dir
  if action invalid: action = best_dir
  append G_dir edge ID to output list
  advance curr_node
```

Cycle detection: if any node is visited twice, the planner falls back to Dijkstra for the full leg. Also falls back on dead-ends, missing edge data, or if max steps (500) are exceeded without arriving.

### DSP — Dynamic Shortest Path (non-learning, traffic-aware)

`dsp` is a baseline that sits between static Dijkstra and the learned RL agents. It is **non-learning** — there is no `policy.pkl`; the `policy_id` only needs to point at a `runs/trsp/<id>/config.yaml` with `agent.algo: dsp` that supplies the road graph, fleet speed, and SVC route file.

Per leg, `plan_edges_and_nodes()` first calls `agent.start_leg(src, dst)` (a generic `hasattr` hook — only DSP defines it), which queries `traci.edge.getLastStepMeanSpeed()` for every edge, re-weights the graph by travel time (`length / speed`), and runs Dijkstra **once** to get the full path. The hop-by-hop rollout then just follows that path. Like `dqn_svc`, DSP reads live TraCI speeds, so it must run inside the SUMO session (`plan()` is called from `Simulator.run()`). The same cycle-detection / max-steps / Dijkstra fallbacks apply.

Ready-to-run example: `python infer_ta_evrp.py -c config/ta_evrp/milp_dsp.yaml` (uses `runs/trsp/dsp_svc_rand_grid/`).

### Dispatcher in `Simulator`

`Simulator._resolve_route(src, dst, sim_time_s)` (`dronesim/ta_evrp/simulator/`):
- If `LocalPlanner` is set → calls `plan()` for the edge list
- Otherwise → calls Dijkstra on `instance.network.graph`

Used for all **planned delivery legs** (depot→customer, customer→customer, customer→AFS, etc.) **and** for emergency `runtime_refuel` detour legs to/from the depot.

The `_needs_emergency_refuel` range check always uses Dijkstra directly — it is never routed through the TRSP planner.

---

## Fuel Accounting

**Default rule (RAPS disabled):** Fuel is debited by the **Dijkstra shortest-path distance** between the two endpoints, regardless of the actual path the TRSP agent takes.

This ensures:
- The fuel gate check (`_needs_emergency_refuel`) uses the same distance that is later consumed — no double-accounting errors.
- If the TRSP agent takes a longer path (due to cycling or traffic-avoidance), it does not strand the vehicle faster than the EVRP planner expected.
- Metrics (`distance_m`) reflect Dijkstra distances, not actual traversed distances.

### Range-Aware Path Switching (RAPS)

Toggle: `trsp.raps_enabled: true` in the TA-EVRP config. Only meaningful when `trsp.agent` is not `dijkstra`.

RAPS is a per-leg decision rule: use the TRSP path only when the vehicle has enough **surplus fuel** to absorb the extra distance beyond the Dijkstra path. When surplus is insufficient, it falls back to Dijkstra and charges the shorter distance. Under RAPS, fuel is charged at the **actual driven distance** (honest accounting).

**Surplus formula** (evaluated before each leg from stop `i` → stop `i+1`):

```
sp_dist    = Dijkstra distance from stop[i] to stop[i+1]
trsp_dist  = sum of non-internal SUMO edge lengths on the TRSP path
extra_dist = trsp_dist - sp_dist

min_after  = stops[i+1].planned_fuel - stops[refuel_idx].planned_fuel
             # planned fuel consumption from stop[i+1] to the next AFS/depot stop
             # refuel_idx = first j ≥ i+1 where node_type ∈ {afs, afs_dummy, depot, depot_dummy}
             # if next stop IS a refuel stop, or none exists: min_after = 0.0

surplus    = vs.current_fuel - sp_dist - min_after
use_trsp   = (extra_dist ≤ surplus)
charge     = trsp_dist if use_trsp else sp_dist
```

`planned_fuel` is set by the MILP planner as remaining fuel (in metres) at each stop. RAPS is fully compatible with the MILP planner; for `heuristic` and `mtsp` planners `planned_fuel=0.0` at intermediate stops, making `min_after=0.0` always, which is conservative.

**Emergency legs** (`_spawn_emg_to_depot_leg`, `_spawn_emg_resume_leg`) are never subject to RAPS — they call `_resolve_route` without RAPS kwargs so the RAPS block is skipped.

### Output artifact

`runs/ta_evrp/<run_id>/raps_log.json` — written only when `raps_enabled=true` and a TRSP agent is present (empty list → no file):

```json
[
  {
    "vehicle_id": 0,
    "leg_idx": 2,
    "from_node": "62259983",
    "to_node": "62261027",
    "dijkstra_dist_m": 1200.0,
    "trsp_dist_m": 1450.0,
    "extra_dist_m": 250.0,
    "surplus_m": 800.0,
    "used_trsp": true
  }
]
```

### Creating a RAPS config

```yaml
# config/ta_evrp/milp_sarsa_raps.yaml
base: config/ta_evrp/milp_sarsa.yaml
trsp:
  raps_enabled: true
```

### Implementation

In `Simulator._resolve_route` (keyword-only RAPS args are `None` unless the caller passes them):
- If `local_planner is None` → Dijkstra immediately.
- TRSP plan is obtained; if empty → fall back to SP.
- If `raps_enabled` and RAPS kwargs are present → compute surplus; log the decision; return TRSP path+dist or SP path+dist.
- Otherwise (legacy) → TRSP path driven, SP distance charged.

The returned distance is what the simulator deducts from `current_fuel` and records in metrics.

---

## Traffic Model Unification

Both TRSP and TA-EVRP support three traffic models, selected via `traffic.model`: `psl`, `td`, and `svc`.

### PSL (Piecewise Speed Limit)

Both TA-EVRP and TRSP share a **PSL model** — per-edge independent speed profiles over K time slots.

| | EVRP `PSLTrafficModel` | TRSP `PSLTrafficModel` |
|---|---|---|
| File | `dronesim/evrp/traffic/segmented.py` | `dronesim/trsp/traffic.py` |
| Speed scope | Per-edge | Per-edge |
| Interface | `get_slot(t)`, `get_speed(edge_id, slot)` | `get_slot(t)`, `get_speed(edge_id, slot)` |
| SUMO integration | `traci.edge.setMaxSpeed()` on slot change | `traci.vehicle.setSpeed()` for the active vehicle |
| Config source | `config/ta_evrp/base.yaml` `traffic.psl:` | `config/trsp/train.yaml` `traffic:` |
| RNG | `random.Random(seed)` | `random.Random(seed)` |

Both models generate a discrete speed level set spanning `[v_min, v_max]` in increments of `speed_step` and draw independently per edge. With identical seed and identical edge set they produce identical profiles — but the TRSP training config and TA-EVRP config can use different seeds and different K values.

Note on config keys: TA-EVRP nests PSL settings under `traffic.psl.*` and uses `_kmh` suffixes (e.g., `v_min_kmh`), while TRSP uses `traffic.*` (e.g., `v_min`). Both store m/s internally. Align both if you want the same profiles end-to-end.

### TD (Time-Dependent congestion — Lu et al. 2020)

A three-period time-of-day model (`traffic.model: td`): a **global** congestion speed `v_c` in the first and last slot and a free-flow speed `v_f` in the middle slot, with boundaries at `T1`/`T2`. `TDTrafficModel` is a `PSLTrafficModel` subclass in both modules (`dronesim/ta_evrp/instance/traffic.py`, `dronesim/trsp/traffic.py`), so it flows through the same `get_slot`/`get_speed` interface, the same `time_slot ∈ {0,1,2}` observation (K=3), and the same Simulator speed-limit code (gated by `Simulator._psl_like`, which covers both `psl` and `td`).

| | EVRP `TDTrafficModel` | TRSP `TDTrafficModel` |
|---|---|---|
| Config block | `traffic.td.*` (`v_congestion_kmh`, `v_freeflow_kmh`, `T1_hours`, `T2_hours`) | flat `traffic.*` (`v_congestion`, `v_freeflow` km/h; `T1`, `T2` seconds) |
| Speed scope | Global (all edges `[v_c, v_f, v_c]`) | Global (all edges `[v_c, v_f, v_c]`) |
| Slot rule | `0` if `t<T1`, `1` if `t<T2`, else `2` | same |

Unlike PSL/TD as a Simulator speed-limit, the **`td_milp` planner** also consumes `traffic.td.*` (EVRP block) to make its arc travel times departure-dependent — so with `traffic.model: td` the plan and the SUMO execution share the same `v_c`/`v_f` and `T1`/`T2`. See `docs/EVRP.md` → `td_milp`.

### SVC (Stochastic Vehicular Congestion)

Real SUMO background vehicles (pre-generated via `randomTrips.py` + `duarouter`) create emergent congestion. The `DQNSVCAgent` observes per-neighbour mean speeds queried live from TraCI (`neighbor_speeds`), replacing the global `time_slot` signal.

**SVC is now fully integrated into TA-EVRP.** Setting `traffic.model: svc` in the TA-EVRP config causes the Simulator to:
1. Append `--route-files <route_file>` to the SUMO command so background vehicles spawn.
2. Skip all PSL slot-based `traci.edge.setMaxSpeed()` and `traci.vehicle.setSpeed()` calls — congestion is emergent, not scripted.

When `trsp.agent: dqn_svc`, `LocalPlanner.plan()` queries `traci.edge.getLastStepMeanSpeed()` at each hop to build the `neighbor_speeds` observation for `DQNSVCAgent.predict()`. TraCI is available at call time because `plan()` is invoked from inside `Simulator.run()` (an active SUMO session).

The `dsp` agent is also SVC-oriented: it reads the same live edge speeds, but once per leg (in `start_leg`) rather than per hop, and re-plans the whole path via travel-time Dijkstra. It needs no trained policy — see the DSP subsection above.

**SUMO warm-up.** `sumo.warm_up_time` (seconds, default `0`) steps the SUMO simulation forward before any delivery vehicle is spawned, letting SVC background traffic build to a steady state first — mirroring `simulation.warm_up_time` in the TRSP env. Time is absolute (like the TRSP env), so a non-zero warm-up offsets `finish_time`/makespan by a constant across all vehicles; relative planner/agent comparisons are unaffected.

---

## Data Flow Diagram

```
infer_ta_evrp.py
  │
  ├─ Instance (dronesim/ta_evrp/instance/instance.py)
  │     ├─ Network: sumolib.net + nx.DiGraph + node positions
  │     ├─ Traffic: PSLTrafficModel (per-edge speed profiles)
  │     └─ Problem: dist matrix, depot/customers/AFS, params (SI)
  │
  ├─ GlobalPlanner (dronesim/ta_evrp/planner/evrp.py)
  │     └─ REGISTRY[planner_name]().plan(instance) → list[Route]
  │           stops carry sumo_node_id + node_name
  │
  ├─ [optional] LocalPlanner(policy_id)   ← dronesim/ta_evrp/planner/trsp.py
  │     ├─ gym.make(SumoTRSPEnv-v0)  ← reads net, no SUMO
  │     └─ load Q-table from runs/trsp/<policy_id>/policy.pkl
  │     (only constructed when policy_id is not None)
  │
  └─ Simulator(instance, local_planner)  ← dronesim/ta_evrp/simulator/
        │
        ├─ build PSLTrafficModel from instance.traffic
        ├─ traci.start(sumo_cmd)
        │
        └─ per leg: _resolve_route(src, dst, sim_time_s)
              ├─ LocalPlanner.plan()   [if local_planner is not None]
              │     └─ Q-table lookup → edge list (drives TRSP path)
              │          SP distance charged for fuel accounting
              └─ dijkstra(instance.network.graph) [if local_planner is None]
                    └─ nx.shortest_path → edge list + SP distance
```

---

## Constraints and Assumptions

- **Same network:** The TRSP policy must be trained on the same SUMO network as the TA-EVRP simulation. `LocalPlanner` builds its graph from the TRSP config's `net.net_path` — mismatched networks cause `KeyError` on node lookup.
- **All RL agents + DSP supported:** `LocalPlanner` dispatches to the correct agent class based on `algo` in the TRSP config. SARSA and SARSA-SVC use neighbour-index actions; DQN/DQN-SVC/A2C/A2C-SVC/PPO/PPO-SVC and DSP use global node-index actions. SVC agents (all `*_svc` and `dsp`) query live TraCI `neighbor_speeds` at each hop and must run inside the SUMO session. DSP is non-learning (no `policy.pkl`). Unknown algos fall back to Dijkstra.
- **No SUMO during plan:** `plan()` is pure Python — no TraCI calls. It runs inside the EVRP SUMO session without conflict.
- **Time slot approximation:** `plan()` receives `sim_time_s` and maps it to a slot via the TRSP PSL model. For `K=1` this is always slot 0 (exact). For `K>1`, the slot reflects the TA-EVRP simulation clock — consistency depends on matching `T` (derived from `max_time_s`) and `K` values between the two configs.

---

## End-to-End Testing Results

End-to-end runs of TA-EVRP with different planners (MILP, mTSP, Heuristic) and TRSP agents are recorded under `runs/ta_evrp/`.

| Results file | Traffic model | Agents covered | RAPS | Sweep config |
|---|---|---|---|---|
| `results/ta_evrp_results.csv` | PSL (K=1–4) | dijkstra, sarsa, dqn, a2c, ppo | off | `config/ta_evrp/sweep_ta_evrp.yaml` |
| `results/ta_evrp_results_svc.csv` | SVC | dijkstra, dsp, sarsa_svc, dqn_svc, a2c_svc, ppo_svc | off | `sweep_dqn_svc.yaml` + `sweep_sarsa_a2c_ppo_svc.yaml` |
| `results/ta_evrp_results_raps.csv` | PSL (K=1–4) + SVC | all PSL + all SVC agents | **on** | `config/ta_evrp/sweep_raps.yaml` |
| `results/ta_evrp_results_td.csv` | TD (3-period) | dijkstra, dsp, sarsa, dqn, a2c, ppo | **on** | `config/ta_evrp/sweep_td.yaml` |

TD sweep: 15 rows (3 baselines × 3 nets + SARSA × 3 nets + DQN/A2C/PPO × 2 nets — no DQN/A2C/PPO policy trained on N1 under TD). Uses `td_milp` planner and `raps_enabled: true`.

Analysis and visualisations:
- `notebooks/ta_evrp_sweep_analysis.ipynb` — PSL, SVC, and **TD** results (RAPS on for TD; off for PSL/SVC sections)
- `notebooks/ta_evrp_sweep_raps_analysis.ipynb` — full RAPS sweep (PSL K=1–4 + SVC, 270 runs)

---

## Current Status

| Feature | Status |
|---------|--------|
| `LocalPlanner` with cycle detection and Dijkstra fallback | ✅ Done |
| `Simulator._resolve_route()` dispatches to `LocalPlanner` | ✅ Done |
| Fuel debited by Dijkstra SP distance (not actual TRSP path length) | ✅ Done |
| `runtime_refuel` — emergency depot detour when range is insufficient | ✅ Done |
| PSL traffic model unified: per-edge speeds in both EVRP and TRSP | ✅ Done |
| `Instance` built once per run, shared by planner and simulator | ✅ Done |
| DQN/A2C/PPO policy support in `LocalPlanner` | ✅ Done |
| End-to-end test: MILP plan + TRSP navigation + metrics | ✅ Done |
| TRSP time slot aligned with EVRP sim clock for K>1 | ✅ Done (PSL traffic end-to-end complete) |
| SVC traffic model trained on N1/N2/N3 (DQN-SVC) | ✅ Done (TRSP standalone) |
| `traffic.model: psl\|svc` selector in TA-EVRP config + simulator | ✅ Done |
| SVC background vehicles in TA-EVRP Simulator via `--route-files` | ✅ Done |
| DQN-SVC `LocalPlanner` support (TraCI `neighbor_speeds` at each hop) | ✅ Done |
| DSP `LocalPlanner` support (live-speed per-leg Dijkstra, non-learning baseline) | ✅ Done |
| SUMO `warm_up_time` in TA-EVRP `sumo:` config (steps traffic before vehicles depart) | ✅ Done |
| SARSA-SVC / A2C-SVC / PPO-SVC trained on N1/N2/N3 (TRSP standalone) | ✅ Done |
| `LocalPlanner` support for sarsa_svc, a2c_svc, ppo_svc (SVC obs + correct action mode) | ✅ Done |
| SVC sweep (all 6 agents × 3 planners × 3 networks) logged to `results/ta_evrp_results_svc.csv` | ✅ Done |
| `ta_evrp_sweep_analysis.ipynb` SVC section updated for all 6 agents | ✅ Done |
| Range-Aware Path Switching (RAPS) — `trsp.raps_enabled` toggle + per-leg surplus check | ✅ Done |
| RAPS sweep (PSL K=1–4 + SVC, all agents × 3 planners × 3 networks = 270 runs) | ✅ Done |
| `ta_evrp_sweep_raps_analysis.ipynb` — analysis notebook for RAPS results | ✅ Done |
| TD (SARSA/DQN/A2C/PPO) policies trained on N1–N3 under 3-period congestion (TRSP standalone) | ✅ Done |
| TD sweep (15 rows: td_milp planner + RAPS, all agents × available nets) → `ta_evrp_results_td.csv` | ✅ Done |
| `ta_evrp_sweep_analysis.ipynb` TD section added (agent bars, % vs Dijkstra, heatmaps) | ✅ Done |
| `trsp_testing_analysis.ipynb` Aggregated Results extended to cover TD traffic condition | ✅ Done |
