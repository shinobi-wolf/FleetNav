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

Two new pieces were added to `dronesim/evrp/evrp_exact.py`:

**1. Canonical edge maps in `load_sumo_graph`**

Using the same per-node edge-iteration logic as the simulator's
`_node_to_outgoing_edge` / `_node_to_incoming_edge`, `load_sumo_graph` now computes:

```
canonical_outgoing[node_id] = (to_node_id, edge_length_m)
canonical_incoming[node_id] = (from_node_id, edge_length_m)
```

These identify exactly which SUMO edge the simulator will depart from / arrive on for
each junction.

**2. `build_simulation_dist_matrix`**

A new function computes a **directed** distance matrix that matches simulation routing:

```
dist[i, j] = out_edge(src).length
           + node_to_node_shortest_path(out_edge.to_node, in_edge.from_node)
           + in_edge(dst).length
```

This is the exact distance `traci.simulation.findRoute` will return for that leg.

**In the heuristic planner**, the old symmetric node-to-node `dist_miles` matrix was
replaced by this simulation-consistent directed matrix:

```python
graph_data = load_sumo_graph(net_xml_path)
sim_dist_m  = build_simulation_dist_matrix(sumo_ids, graph_data, "meters")
dist_miles  = sim_dist_m * 0.000621371
```

With realistic distances, rb's fuel budget is evaluated against the same distances the
simulator will actually traverse, so the feasibility flag now reflects whether vehicles
can complete their routes — including the return to depot — without running out of fuel.
