# Heuristic Planner Bug Fixes

Two bugs in `dronesim/evrp/planners/evrp_heuristic.py` caused RoutingBlocks (rb) to
report routes as feasible even when vehicles ran out of fuel before returning to the
depot in simulation.

---

## Bug 1 — Arc resource consumption missing `gal_per_mile` factor

### What was wrong

`rb.adptw.ArcData(distance, travel_time, resource_consumption)` takes three
arguments. The third is the amount of battery/fuel consumed when traversing that arc.

The original code passed the same value `d` (distance in miles) for all three:

```python
builder.add_arc(a_id, b_id, rb.adptw.ArcData(d, d, d))
```

This made arc resource consumption numerically equal to arc distance (miles), while
`rb.adptw.Evaluation(fuel_capacity, ...)` received `fuel_capacity` in **gallons**.
The units were inconsistent: rb compared a resource measured in *miles* against a
budget measured in *gallons*.

For `consumption_rate = 1 gal/mile` the numbers happened to be identical so there
was no observable effect. For any other rate the feasibility check would have been
wrong in proportion to the rate.

### Fix

Multiply the resource argument by `gal_per_mile` so it is in gallons:

```python
builder.add_arc(a_id, b_id, rb.adptw.ArcData(d, d, d * gal_per_mile))
```

Now the arc resource consumption and the evaluation battery capacity are both in
gallons, and rb's feasibility check is dimensionally correct.

---

## Bug 2 — Planning distances ~2-3× shorter than simulation distances

### What was wrong

After fixing Bug 1, routes were still infeasible. The rb makespan was `4.9` (miles)
for vehicle 1's route, but SUMO measured `19,828 m ≈ 12.3 miles` for the same route.
rb thought the route consumed `4.9 gal < 7 gal` (fuel capacity), so it reported
feasible — in reality the vehicle used all 7 gallons and still had to return to depot.

The discrepancy came from a mismatch between how the planning and simulation measure
distances between nodes.

**Planning** used node-to-node shortest paths from the NetworkX graph
(`build_distance_matrix_from_shortest_paths`). For example, depot (C2) → customer C4
(C1) = **985.6 m** (one direct junction-to-junction hop).

**Simulation** routed with `traci.simulation.findRoute(src_edge, dst_edge)` where:

- `src_edge` = first non-internal **outgoing** edge from the source junction
- `dst_edge` = first non-internal **incoming** edge into the destination junction

The route therefore includes the **full traversal of `src_edge`** and the **full
traversal of `dst_edge`** — at least two extra edge lengths (~985.6 m each) on top
of the junction-to-junction path. The same depot → C4 leg measured **~2,977 m** in
simulation (roughly three edge-lengths instead of one).

This factor was non-uniform across legs (ranging from ~1.25× to ~3×), so a fixed
scaling constant could not fix it.

### Fix

The fix is in the simulator (`dronesim/evrp/simulator/evrp_sim.py`), not in the
planner. Instead of adapting the planning distances to match a broken routing method,
the simulator is made to route the same way the planner computes distances.

**`EVRPSimulator._build_routing_graph`** (new method)

Called once in `run()` after the SUMO net is loaded, this builds a `nx.DiGraph`
from the same net — identical in structure to what `load_sumo_graph` builds in
`evrp_exact.py`. Every non-internal SUMO edge becomes a directed graph edge with its
SUMO edge ID stored as a `"id"` attribute and its length in metres as `"length"`:

```python
for edge in self._net.getEdges():
    if edge.getID().startswith(":"):
        continue
    g.add_edge(from_id, to_id, id=edge.getID(), length=edge.getLength())
```

**`EVRPSimulator._find_edge_route`** (reimplemented)

The old implementation called `traci.simulation.findRoute(outgoing_edge(src),
incoming_edge(dst))`, which included the framing edges in the measured distance.
The new implementation uses the nx graph directly:

```python
node_path = nx.shortest_path(self._routing_graph, src_node, dst_node, weight="length")
edge_ids  = [graph[u][v]["id"] for u, v in zip(node_path[:-1], node_path[1:])]
```

`nx.shortest_path` with `weight="length"` finds the exact same minimum-cost node
sequence that `build_distance_matrix_from_shortest_paths` uses in the planner, so the
SUMO edges the simulator traverses sum to the same distance that rb used when deciding
the route was feasible.  The planning–simulation distance gap is eliminated at its
source.
