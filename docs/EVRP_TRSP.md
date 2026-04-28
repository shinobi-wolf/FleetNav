# EVRP + TRSP Integration — Current State

This document covers the connection between the two modules: how the EVRP outer planner hands legs to the TRSP inner navigator, what fuel accounting applies, and how the traffic configs relate.

---

## What Is Integrated

The EVRP simulators (`EVRPSequentialSimulator`, `EVRPConcurrentSimulator`) can now use a trained TRSP RL agent to navigate between every pair of consecutive stops, replacing the default Dijkstra shortest-path routing. The change is config-driven and fully backward-compatible: the default behavior (`local_planner: dijkstra`) is unchanged.

---

## Config

Set in `config/evrp/base.yaml` under `simulation`:

```yaml
simulation:
  local_planner: dijkstra   # dijkstra | trsp
  trsp_policy: null         # policy ID under runs/trsp/; required when local_planner: trsp
```

Example — switch to TRSP navigation with the SARSA policy:

```yaml
simulation:
  local_planner: trsp
  trsp_policy: sarsa        # loads runs/trsp/sarsa/config.yaml + policy.pkl
```

The policy ID must correspond to a directory under `runs/trsp/` containing `config.yaml` and `policy.pkl`.

---

## How It Works

### `TRSPLocalPlanner` (`dronesim/evrp/local_planner.py`)

Instantiated once during `EVRPSequentialSimulator.__init__()` — before SUMO starts. Steps:

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

Used for all **planned delivery legs** (depot→customer, customer→customer, customer→AFS, etc.). Emergency trackbacks to depot (when fuel is insufficient for the round-trip) always use Dijkstra regardless of `local_planner` — they are not delivery legs.

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

Concurrent (`_dispatch_leg`): already set `vs.pending_leg_dist_m = dist_to_next` (Dijkstra); `_on_arrival()` reads it — no change needed.

---

## Traffic Config Unification

EVRP's `SegmentedSpeedModel` and TRSP's `PSLTrafficModel` use **identical config key names** (`T`, `K`, `v_min`, `v_max`, `speed_step`, `seed`). However, they remain **separate models** with different semantics:

| | `SegmentedSpeedModel` (EVRP) | `PSLTrafficModel` (TRSP) |
|---|---|---|
| Speed scope | Global — same speed for all edges at a given time | Per-edge — each edge has its own speed profile |
| Interface | `update(t) → bool`, `commanded_speed_ms` property | `get_slot(t) → int`, `get_speed(edge_id, slot) → float` |
| Config source | `config/evrp/base.yaml` `traffic:` block | `config/trsp/train.yaml` `traffic:` block |
| Speed values | Drawn via `np.random.default_rng(seed)` | Drawn via `random.Random(seed)` (isolated from global RNG) |

The config key unification means a single `traffic:` block format works for both modules — no separate `seg_duration`/`speed_limits` keys to maintain. But the actual speed values sampled for EVRP and TRSP are independent (different RNGs, different configs, different seeds allowed).

---

## Data Flow Diagram

```
run_sim.py
  │
  ├─ Planner (milp / heuristic / mtsp / td_evrp_milp)
  │     └─ list[Route] (stops with sumo_node_ids)
  │
  └─ Simulator (sequential / concurrent)
        │
        ├─ EVRPSequentialSimulator.__init__()
        │     └─ TRSPLocalPlanner.__init__(policy_id)
        │           ├─ gym.make(SumoTRSPEnv-v0)  ← reads net, no SUMO
        │           └─ load Q-table from policy.pkl
        │
        └─ per leg: _resolve_route(src, dst, sim_time_s)
              ├─ TRSPLocalPlanner.plan()  [if local_planner: trsp]
              │     └─ iterative Q-table lookup on graph → edge list
              └─ _find_edge_route()       [if local_planner: dijkstra]
                    └─ nx.shortest_path on routing graph → edge list
```

---

## Constraints and Assumptions

- **Same network:** The TRSP policy must be trained on the same SUMO network as the EVRP simulation. `TRSPLocalPlanner` builds its graph from the TRSP config's `net.net_path` — mismatched networks cause `KeyError` on node lookup.
- **SARSA only:** `TRSPLocalPlanner` loads a Q-table pickle directly. DQN/PPO policies are not yet supported (DQN would require loading the neural network weights and doing a forward pass).
- **No SUMO during plan:** `plan()` is pure Python — no TraCI calls. It runs inside the EVRP SUMO session without conflict.
- **Time slot approximation:** `plan()` receives `sim_time_s` and maps it to a slot via the TRSP PSL model. For `K=1` this is always slot 0 (exact). For `K>1`, the slot reflects the EVRP simulation clock, not the TRSP training clock — consistency depends on matching `T` and `K` values between the two configs.

---

## Current Status

| Feature | Status |
|---------|--------|
| Config keys: `local_planner`, `trsp_policy` in base.yaml | ✅ Done |
| `TRSPLocalPlanner` with cycle detection and Dijkstra fallback | ✅ Done |
| `_resolve_route()` in both simulators | ✅ Done |
| Fuel debited by Dijkstra distance (not actual path) | ✅ Done |
| Traffic config unified to PSL-style keys | ✅ Done |
| DQN/PPO policy support in `TRSPLocalPlanner` | ❌ Not yet |
| End-to-end test: MILP plan + TRSP navigation + metrics | ❌ Not yet run |
| TRSP time slot aligned with EVRP sim clock for K>1 | ⚠️ Works for K=1; needs validation for K>1 |
