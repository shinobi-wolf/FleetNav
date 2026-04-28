# TA-EVRP Integration — Current State

This document covers the connection between the two modules: how the EVRP outer planner hands legs to the TRSP inner navigator, what fuel accounting applies, and how the traffic models relate.

---

## What Is Integrated

The EVRP simulators (`EVRPSequentialSimulator`, `EVRPConcurrentSimulator`) can use a trained TRSP RL agent to navigate between every pair of consecutive stops, replacing the default Dijkstra shortest-path routing. The change is config-driven and fully backward-compatible: the default behavior (`trsp.agent: dijkstra`) is unchanged.

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

Constructed in `infer_ta_evrp.py` when `trsp.agent != dijkstra`, then injected into the simulator. Steps:

1. Loads the TRSP config from `runs/trsp/<policy_id>/config.yaml`.
2. Calls `gym.make("SumoTRSPEnv-v0", config=trsp_config)` to build the road graph and PSL traffic model. **SUMO is not started** — `gym.make()` only reads the net file via `sumolib`.
3. Loads the Q-table from `runs/trsp/<policy_id>/policy.pkl`.
4. Exposes `plan(src_node, dst_node, sim_time_s) → list[str]`.

### `plan()` — Graph-Only Navigation

No SUMO call. Pure Python graph traversal:

```
curr_node = src_node
while curr_node != dst_node:
    compute steps-to-dest for each neighbour (directed SPL)
    build observation: (graph_node, target_node, time_slot)
    look up Q[state][:num_neighbours], argmax → action
    if state not in Q: fall back to best_dir (greedy Dijkstra neighbour)
    append G_dir edge ID to output list
    advance curr_node
```

Cycle detection: if any node is visited twice, the planner falls back to Dijkstra for the full leg. Also falls back if max steps (500) are exceeded without arriving.

### Dispatcher in Simulators

`EVRPSequentialSimulator._resolve_route(src, dst, sim_time_s)`:
- If `TRSPLocalPlanner` is set → calls `plan()`
- Otherwise → calls `_find_edge_route()` (NetworkX Dijkstra)

Used for all **planned delivery legs** (depot→customer, customer→customer, customer→AFS, etc.). Emergency trackbacks to depot (when fuel is insufficient for the round-trip) always use Dijkstra regardless of `local_planner`.

`EVRPConcurrentSimulator._dispatch_leg()` calls `_resolve_route()` for normal dispatch; it inherits the method from the sequential base class.

---

## Fuel Accounting

**Rule:** Fuel is always debited by the **Dijkstra shortest-path distance** between the two stops, regardless of the actual path the TRSP agent takes.

This ensures:
- The fuel gate check (`vs.range_m < dist_to_next`) uses the same distance that is later consumed → no double-accounting errors.
- If the TRSP agent takes a longer path (due to cycling or traffic-avoidance), it does not strand the vehicle faster than the planner expected.
- Metrics (`vs.distance_m`, `vs.energy_kwh`) reflect Dijkstra distances, not actual traversed distances.

### Implementation

Sequential (`_execute_route`):
```python
dist_to_next = self._route_distance_m(src_node, dst_node)   # Dijkstra
edge_list = self._resolve_route(src_node, dst_node, sim_time_s)
_, sim_time_s = self._drive(vs, edge_list, sim_time_s, traffic)
vs.consume(dist_to_next)   # Dijkstra distance, not actual SUMO distance
```

Concurrent (`_dispatch_leg`): `vs.pending_leg_dist_m = dist_to_next` (Dijkstra); `_on_arrival()` consumes it.

---

## Traffic Model Unification

Both TA-EVRP and TRSP now use the **same PSL (Piecewise Speed Limit) model** — per-edge independent speed profiles over K time slots.

| | EVRP `PSLTrafficModel` | TRSP `PSLTrafficModel` |
|---|---|---|
| File | `dronesim/evrp/traffic/segmented.py` | `dronesim/trsp/traffic.py` |
| Speed scope | Per-edge | Per-edge |
| Interface | `get_slot(t)`, `get_speed(edge_id, slot)` | `get_slot(t)`, `get_speed(edge_id, slot)` |
| SUMO integration | `traci.edge.setMaxSpeed()` on slot change | `traci.edge.setMaxSpeed()` on slot change |
| Config source | `config/ta_evrp/base.yaml` `traffic.psl:` | `config/trsp/train.yaml` `traffic:` |
| RNG | `random.Random(seed)` | `random.Random(seed)` |

Both models generate the same speed level set `{v_min, v_min+step, ..., v_max}` and draw independently per edge. With identical seed and identical edge set they produce identical profiles — but the TRSP training config and TA-EVRP config can use different seeds and different K values.

Key difference from the old architecture: EVRP's old `SegmentedSpeedModel` used a **single global speed** for all edges at each time slot. The new PSL gives each edge its own independent speed, matching TRSP semantics and enabling proper per-edge edge coloring in the renderer.

---

## Data Flow Diagram

```
infer_ta_evrp.py
  │
  ├─ Planner (milp / heuristic / mtsp / td_milp)
  │     └─ list[Route] (stops with sumo_node_ids)
  │
  ├─ [optional] TRSPLocalPlanner(policy_id)      ← dronesim/ta_evrp/local_planner.py
  │     ├─ gym.make(SumoTRSPEnv-v0)  ← reads net, no SUMO
  │     └─ load Q-table from policy.pkl
  │
  └─ Simulator (sequential / concurrent)
        │
        ├─ build PSLTrafficModel from edge_ids + config
        ├─ traci.start(sumo_cmd)
        │
        └─ per leg: _resolve_route(src, dst, sim_time_s)
              ├─ TRSPLocalPlanner.plan()  [if trsp.agent != dijkstra]
              │     └─ iterative Q-table lookup on graph → edge list
              └─ _find_edge_route()       [if trsp.agent: dijkstra]
                    └─ nx.shortest_path on routing graph → edge list
```

---

## Constraints and Assumptions

- **Same network:** The TRSP policy must be trained on the same SUMO network as the TA-EVRP simulation. `TRSPLocalPlanner` builds its graph from the TRSP config's `net.net_path` — mismatched networks cause `KeyError` on node lookup.
- **SARSA only:** `TRSPLocalPlanner` loads a Q-table pickle directly. DQN/PPO policies are not yet supported.
- **No SUMO during plan:** `plan()` is pure Python — no TraCI calls. It runs inside the EVRP SUMO session without conflict.
- **Time slot approximation:** `plan()` receives `sim_time_s` and maps it to a slot via the TRSP PSL model. For `K=1` this is always slot 0 (exact). For `K>1`, the slot reflects the TA-EVRP simulation clock, not the TRSP training clock — consistency depends on matching `T` (derived from `max_time_s`) and `K` values between the two configs.

---

## Current Status

| Feature | Status |
|---------|--------|
| `TRSPLocalPlanner` with cycle detection and Dijkstra fallback | ✅ Done |
| `_resolve_route()` in both simulators | ✅ Done |
| Fuel debited by Dijkstra distance (not actual path) | ✅ Done |
| PSL traffic model unified: per-edge speeds in both EVRP and TRSP | ✅ Done |
| `traci.edge.setMaxSpeed()` per-edge slot transitions in EVRP | ✅ Done |
| Pygame renderer: per-edge color from PSL profile | ✅ Done |
| DQN/PPO policy support in `TRSPLocalPlanner` | ❌ Not yet |
| End-to-end test: MILP plan + TRSP navigation + metrics | ❌ Not yet run |
| TRSP time slot aligned with EVRP sim clock for K>1 | ⚠️ Works for K=1; needs validation for K>1 |
