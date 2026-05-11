# TA-EVRP Integration — Current State

This document covers the connection between the two modules: how the EVRP outer planner hands legs to the TRSP inner navigator, what fuel accounting applies, and how the traffic models relate.

---

## What Is Integrated

`EVRPSimulator` can use a trained TRSP RL agent to navigate between every pair of consecutive stops, replacing the default Dijkstra shortest-path routing. The change is config-driven and fully backward-compatible: the default behavior (`trsp.agent: dijkstra`) is unchanged.

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

### `TRSPLocalPlanner` (`dronesim/ta_evrp/local_planner.py`)

Constructed in `infer_ta_evrp.py` when `trsp.agent != dijkstra`, then injected into `EVRPSimulator`. Steps:

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

### Dispatcher in `EVRPSimulator`

`EVRPSimulator._resolve_route(src, dst, sim_time_s)` (`dronesim/evrp/simulator/evrp_sim.py`):
- If `TRSPLocalPlanner` is set → calls `plan()` for the edge list
- Otherwise → calls `_find_edge_route()` (NetworkX Dijkstra)

Used for all **planned delivery legs** (depot→customer, customer→customer, customer→AFS, etc.) **and** for emergency `runtime_refuel` detour legs to/from the depot.

The `_needs_emergency_refuel` range check always uses `_find_edge_route()` (Dijkstra) directly — it is never routed through the TRSP planner.

---

## Fuel Accounting

**Rule:** Fuel is always debited by the **Dijkstra shortest-path distance** between the two endpoints, regardless of the actual path the TRSP agent takes.

This ensures:
- The fuel gate check (`_needs_emergency_refuel`) uses the same distance that is later consumed — no double-accounting errors.
- If the TRSP agent takes a longer path (due to cycling or traffic-avoidance), it does not strand the vehicle faster than the EVRP planner expected.
- Metrics (`distance_m`) reflect Dijkstra distances, not actual traversed distances.

### Implementation

In `_resolve_route` (`evrp_sim.py`):
```python
if self.local_planner is not None:
    edge_ids = self.local_planner.plan(src_node, dst_node, sim_time_s)
    if edge_ids:
        sp_edges, sp_dist = self._find_edge_route(src_node, dst_node)
        if sp_edges is not None:
            return edge_ids, sp_dist   # TRSP path driven; SP distance charged
        # No SP found (rare); fall back to actual route length
        actual_dist = sum(net.getEdge(e).getLength() for e in edge_ids ...)
        return edge_ids, actual_dist
return self._find_edge_route(src_node, dst_node)
```

The returned distance is what `_spawn_next_leg` deducts from `veh_state.current_fuel` and adds to `sim_stats["distance_covered"]`.

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
  ├─ Planner (milp / heuristic / mtsp)
  │     └─ list[Route] (stops with sumo_node_ids)
  │
  ├─ [optional] TRSPLocalPlanner(policy_id)      ← dronesim/ta_evrp/local_planner.py
  │     ├─ gym.make(SumoTRSPEnv-v0)  ← reads net, no SUMO
  │     └─ load Q-table from policy.pkl
  │
  └─ EVRPSimulator (dronesim/evrp/simulator/evrp_sim.py)
        │
        ├─ build PSLTrafficModel from edge_ids + config
        ├─ traci.start(sumo_cmd)
        │
        └─ per leg: _resolve_route(src, dst, sim_time_s)
              ├─ TRSPLocalPlanner.plan()   [if trsp.agent != dijkstra]
              │     └─ Q-table lookup → edge list (drives TRSP path)
              │     └─ _find_edge_route() for SP distance (fuel charge)
              └─ _find_edge_route()        [if trsp.agent: dijkstra]
                    └─ nx.shortest_path → edge list + SP distance
```

---

## Constraints and Assumptions

- **Same network:** The TRSP policy must be trained on the same SUMO network as the TA-EVRP simulation. `TRSPLocalPlanner` builds its graph from the TRSP config's `net.net_path` — mismatched networks cause `KeyError` on node lookup.
- **SARSA only:** `TRSPLocalPlanner` loads a Q-table pickle directly. DQN/PPO policies are not yet supported.
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
| `TRSPLocalPlanner` with cycle detection and Dijkstra fallback | ✅ Done |
| `_resolve_route()` in `EVRPSimulator` dispatches to TRSP planner | ✅ Done |
| Fuel debited by Dijkstra SP distance (not actual TRSP path length) | ✅ Done |
| `runtime_refuel` — emergency depot detour when range is insufficient | ✅ Done |
| PSL traffic model unified: per-edge speeds in both EVRP and TRSP | ✅ Done |
| `traci.edge.setMaxSpeed()` per-edge slot transitions in EVRP | ✅ Done |
| Pygame renderer: per-edge color from PSL profile | ✅ Done |
| `_RouteLogger` correctly unpacks `(edge_ids, dist)` for per-leg logging | ✅ Done |
| DQN/PPO policy support in `TRSPLocalPlanner` | ❌ Not yet |
| End-to-end test: MILP plan + TRSP navigation + metrics | ✅ Done |
| TRSP time slot aligned with EVRP sim clock for K>1 | ⚠️ Works for K=1; needs validation for K>1 |
