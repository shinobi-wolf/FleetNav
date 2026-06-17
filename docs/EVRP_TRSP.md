# TA-EVRP Integration — Current State

This document covers the connection between the two modules: how the EVRP outer planner hands legs to the TRSP inner navigator, what fuel accounting applies, and how the traffic models relate.

---

## What Is Integrated

The `Simulator` can use a trained TRSP RL agent to navigate between every pair of consecutive stops, replacing the default Dijkstra shortest-path routing. The change is config-driven: when `trsp.policy_id` is set, a `LocalPlanner` is constructed and injected into the `Simulator`; otherwise only Dijkstra is used.

---

## Config

Set in `config/ta_evrp/` configs under `trsp`:

```yaml
trsp:
  agent: dijkstra     # dijkstra | dynamic_dijkstra | sarsa | dqn | a2c | ppo
  policy_id: null     # policy ID under runs/trsp/; required for RL agents
```

Example — switch to TRSP navigation with the SARSA policy:

```yaml
trsp:
  agent: sarsa
  policy_id: 77a4169c   # loads runs/trsp/77a4169c/config.yaml + policy.pkl
```

The policy ID must correspond to a directory under `runs/trsp/` containing `config.yaml` and `policy.pkl`.

Run with: `python infer_ta_evrp.py -c config/ta_evrp/milp_sarsa.yaml`

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

### Dispatcher in `Simulator`

`Simulator._resolve_route(src, dst, sim_time_s)` (`dronesim/ta_evrp/simulator/`):
- If `LocalPlanner` is set → calls `plan()` for the edge list
- Otherwise → calls Dijkstra on `instance.network.graph`

Used for all **planned delivery legs** (depot→customer, customer→customer, customer→AFS, etc.) **and** for emergency `runtime_refuel` detour legs to/from the depot.

The `_needs_emergency_refuel` range check always uses Dijkstra directly — it is never routed through the TRSP planner.

---

## Fuel Accounting

**Rule:** Fuel is always debited by the **Dijkstra shortest-path distance** between the two endpoints, regardless of the actual path the TRSP agent takes.

This ensures:
- The fuel gate check (`_needs_emergency_refuel`) uses the same distance that is later consumed — no double-accounting errors.
- If the TRSP agent takes a longer path (due to cycling or traffic-avoidance), it does not strand the vehicle faster than the EVRP planner expected.
- Metrics (`distance_m`) reflect Dijkstra distances, not actual traversed distances.

### Implementation

In `Simulator._resolve_route`:
```python
if self.local_planner is not None:
    edge_ids = self.local_planner.plan(src_node, dst_node, sim_time_s)
    if edge_ids:
        sp_dist = dijkstra_dist(instance.network.graph, src_node, dst_node)
        return edge_ids, sp_dist   # TRSP path driven; SP distance charged
    # empty edge list → fall back to Dijkstra
return dijkstra_route(instance.network.graph, src_node, dst_node)
```

The returned distance is what the simulator deducts from `current_fuel` and records in metrics.

---

## Traffic Model Unification

Both TA-EVRP and TRSP use the **same PSL (Piecewise Speed Limit) model** — per-edge independent speed profiles over K time slots.

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
- **All RL agents supported:** `LocalPlanner` dispatches to the correct agent class based on `algo` in the TRSP config. SARSA uses neighbour-index actions; DQN/A2C/PPO use global node-index actions. Unknown algos fall back to Dijkstra.
- **No SUMO during plan:** `plan()` is pure Python — no TraCI calls. It runs inside the EVRP SUMO session without conflict.
- **Time slot approximation:** `plan()` receives `sim_time_s` and maps it to a slot via the TRSP PSL model. For `K=1` this is always slot 0 (exact). For `K>1`, the slot reflects the TA-EVRP simulation clock — consistency depends on matching `T` (derived from `max_time_s`) and `K` values between the two configs.

---

## End-to-End Testing Results

End-to-end runs of TA-EVRP with different planners (MILP, mTSP, Heuristic) and TRSP agents are recorded under `runs/ta_evrp/`, with aggregated tables in `results/ta_evrp_results.csv`.

In the currently logged experiments, using the SARSA local planner often improves makespan versus the shortest-path baseline, but it is not guaranteed for every planner/network combination.

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
