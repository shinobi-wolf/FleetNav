"""Gurobi MILP formulation for the Green VRP (Erdogan & Miller-Hooks 2012)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

try:
    import gurobipy as gp  # type: ignore[import-not-found]
    from gurobipy import GRB  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment dependent
    gp = None
    GRB = None
    GUROBI_IMPORT_ERROR = exc
else:
    GUROBI_IMPORT_ERROR = None

from ..instance.dummy_expansion import expand_with_dummy_vertices


def build_gvrp_model(dist: np.ndarray, params: dict[str, Any], physical_meta: dict[str, Any] | None = None):
    """Build and return the paper-style G-VRP MILP model in Gurobi."""
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {GUROBI_IMPORT_ERROR}")

    if physical_meta is None:
        physical_meta = params.get("_physical_meta")

    if physical_meta is None:
        raise ValueError("physical_meta is required to build dynamic G-VRP model")

    instance = expand_with_dummy_vertices(dist, physical_meta, params)
    d = instance["dist"]
    t = instance["travel_time"]
    p = instance["service_times"]
    V = instance["V"]
    arcs = instance["arcs"]
    customers = instance["customers"]
    f_prime = instance["f_prime"]
    refuel_nodes = instance["refuel_nodes"]
    non_depot = instance["non_depot"]
    depot = instance["depot"]

    r = float(params["r"])
    Q = float(params["Q"])
    Tmax = float(params["Tmax"])
    m = int(params["m"])

    model = gp.Model("gvrp_exact_erdogan_miller_hooks_2012")
    model.Params.TimeLimit = float(params.get("time_limit", 300.0))
    model.Params.OutputFlag = int(params.get("output_flag", 1))
    model.Params.MIPGap = float(params.get("mip_gap", 0.0))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    out_neighbors: dict[int, list[int]] = {i: [] for i in V}
    in_neighbors: dict[int, list[int]] = {i: [] for i in V}
    for i, j in arcs:
        out_neighbors[i].append(j)
        in_neighbors[j].append(i)

    y = model.addVars(V, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="y")
    s = model.addVars(V, lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name="s")

    model.setObjective(gp.quicksum(d[i, j] * x[i, j] for i, j in arcs), GRB.MINIMIZE)

    for i in customers:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_neighbors[i]) == 1,
            name=f"c2_customer_successor_{i}",
        )

    for i in f_prime:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_neighbors[i]) <= 1,
            name=f"c3_afs_dummy_successor_{i}",
        )

    for j in V:
        incoming = gp.quicksum(x[i, j] for i in in_neighbors[j])
        outgoing = gp.quicksum(x[j, k] for k in out_neighbors[j])
        model.addConstr(incoming - outgoing == 0, name=f"c4_flow_{j}")

    model.addConstr(
        gp.quicksum(x[depot, j] for j in out_neighbors[depot]) <= m,
        name="c5_depot_depart_limit",
    )

    model.addConstr(
        gp.quicksum(x[i, depot] for i in in_neighbors[depot]) <= m,
        name="c6_depot_return_limit",
    )

    for i, j in arcs:
        if j in non_depot:
            model.addConstr(
                s[j] >= s[i] + (t[i, j] + p[j]) * x[i, j] - Tmax * (1 - x[i, j]),
                name=f"c7_time_track_{i}_{j}",
            )

    model.addConstr(s[depot] >= 0.0, name="c8_depot_time_lb")
    model.addConstr(s[depot] <= Tmax, name="c8_depot_time_ub")

    for j in non_depot:
        model.addConstr(s[j] >= t[depot, j], name=f"c9_arrival_lb_{j}")
        model.addConstr(
            s[j] <= Tmax - (t[j, depot] + p[j]),
            name=f"c9_arrival_ub_{j}",
        )

    for j in customers:
        for i in in_neighbors[j]:
            model.addConstr(
                y[j] <= y[i] - r * d[i, j] * x[i, j] + Q * (1 - x[i, j]),
                name=f"c10_fuel_track_{i}_{j}",
            )

    for j in refuel_nodes:
        model.addConstr(y[j] == Q, name=f"c11_fuel_reset_{j}")

    for j in customers:
        direct_to_depot = r * d[j, depot]
        via_station = [r * (d[j, l] + d[l, depot]) for l in f_prime]
        reserve_fuel = min([direct_to_depot] + via_station)
        model.addConstr(y[j] >= reserve_fuel, name=f"c12_fuel_reserve_{j}")

    model._x = x
    model._y = y
    model._s = s
    model._instance = instance
    return model


def _status_name(status_code: int) -> str:
    if gp is None:
        return "UNKNOWN"

    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


def solve_and_extract(model, dist: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Solve model and extract route sequences plus node-level fuel/time details."""
    _ = dist, params
    model.optimize()

    status = model.Status
    status_text = _status_name(status)
    has_feasible = model.SolCount > 0

    print("\n" + "=" * 88)
    print(f"Solver status: {status_text} (code {status})")
    print(f"Feasible solutions found: {model.SolCount}")

    if not has_feasible:
        print("No feasible solution available to extract.")
        return {
            "has_solution": False,
            "status": status_text,
            "objective": None,
            "routes": [],
            "instance": model._instance,
        }

    if status == GRB.OPTIMAL:
        print("Optimal solution found.")
    elif status == GRB.TIME_LIMIT:
        print("Time limit reached; reporting best feasible incumbent.")
    else:
        print("Reporting best available feasible incumbent solution.")

    objective = float(model.ObjVal)
    print(f"Objective value (total distance): {objective:.3f} {params.get('distance_unit', 'units')}")

    x = model._x
    y = model._y
    s = model._s
    inst = model._instance
    V = inst["V"]
    depot = inst["depot"]
    d = inst["dist"]
    node_names = inst["node_names"]
    node_sumo_ids = inst["node_sumo_ids"]

    active_arcs = [(i, j) for (i, j) in inst["arcs"] if x[i, j].X > 0.5]
    succ = defaultdict(list)
    for i, j in active_arcs:
        succ[i].append(j)

    routes: list[dict[str, Any]] = []
    covered_arcs: set[tuple[int, int]] = set()
    starts = succ.get(depot, [])

    for vehicle_id, first_stop in enumerate(starts, start=1):
        route_nodes = [depot, first_stop]
        covered_arcs.add((depot, first_stop))
        current = first_stop
        guard = 0
        max_hops = len(V) + 5

        while current != depot and guard < max_hops:
            next_candidates = succ.get(current, [])
            if not next_candidates:
                break
            nxt = next_candidates[0]
            route_nodes.append(nxt)
            covered_arcs.add((current, nxt))
            if nxt in route_nodes[:-1]:
                break
            current = nxt
            guard += 1

        route_distance = float(sum(d[u, v] for u, v in zip(route_nodes[:-1], route_nodes[1:])))
        route_stops = []
        for node in route_nodes:
            route_stops.append(
                {
                    "node_index": node,
                    "node_name": node_names[node],
                    "sumo_node_id": node_sumo_ids[node],
                    "fuel": float(y[node].X),
                    "arrival_time": float(s[node].X),
                }
            )

        routes.append(
            {
                "vehicle": vehicle_id,
                "nodes": route_nodes,
                "distance": route_distance,
                "stops": route_stops,
            }
        )

    uncovered_arcs = [arc for arc in active_arcs if arc not in covered_arcs and arc[0] != depot]
    if uncovered_arcs:
        print("Warning: active arcs outside depot-anchored routes were found:")
        for i, j in uncovered_arcs:
            print(f"  {node_names[i]} -> {node_names[j]}")

    if not routes:
        print("No depot-anchored routes were extracted from the incumbent.")
    else:
        print("\nExtracted routes:")
        for route in routes:
            seq = " -> ".join(node_names[node] for node in route["nodes"])
            print(f"\nVehicle {route['vehicle']} route: {seq}")
            print(f"Route distance: {route['distance']:.3f} {params.get('distance_unit', 'units')}")
            print("Node details (arrival time in hours, fuel in gallons):")
            for stop in route["stops"]:
                print(
                    f"  {stop['node_name']:<10s} ({stop['sumo_node_id']}) | "
                    f"time = {stop['arrival_time']:.3f} | fuel = {stop['fuel']:.3f}"
                )

    return {
        "has_solution": True,
        "status": status_text,
        "objective": objective,
        "routes": routes,
        "instance": inst,
    }


def run_single_vehicle_check(
    config: dict[str, Any],
    physical_dist: np.ndarray,
    params: dict[str, Any],
    physical_meta: dict[str, Any],
) -> None:
    """Optional m=1 validation run on the generated network instance."""
    if not bool(config.get("gvrp", {}).get("run_single_vehicle_benchmark", False)):
        return

    print("\nRunning optional single-vehicle check (m=1)...")
    single_params = dict(params)
    single_params["m"] = 1
    single_params["output_flag"] = int(config.get("solver", {}).get("output_flag", 0))
    single_params["_physical_meta"] = physical_meta

    model_m1 = build_gvrp_model(physical_dist, single_params, physical_meta)
    result_m1 = solve_and_extract(model_m1, physical_dist, single_params)

    if result_m1["has_solution"]:
        target = float(config.get("gvrp", {}).get("benchmark_target_distance_miles", 354.0))
        tolerance = float(config.get("gvrp", {}).get("benchmark_tolerance", 1e-3))
        value = float(result_m1["objective"])
        diff = value - target

        print(f"Single-vehicle objective: {value:.3f} {params.get('distance_unit', 'units')}")
        if abs(diff) <= tolerance:
            print(f"Benchmark confirmation: objective matches target {target:.3f}.")
        else:
            print(
                "Benchmark confirmation: objective does not match target "
                f"{target:.3f} (difference {diff:+.3f})."
            )
