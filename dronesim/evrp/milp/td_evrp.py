"""Gurobi MILP for the Time-Dependent EVRP — Lera-Romero et al. (2024).

Key differences from static G-VRP:
  - Travel times τ_ij(t) depend on departure time via piecewise-constant speed
    (Ichoua et al. 2003 FIFO model). Precharacterized per arc and departure period.
  - Binary period-indicator variables y[i,j,k] link departure time to the correct τ.
  - Battery consumption is static (m_u = 0): r * d_ij per arc.
  - Full constant-time recharging (replaces paper's partial SoC model).
  - No time windows (paper's TDEVRPTW → TDEVRP).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

try:
    import gurobipy as gp  # type: ignore[import-not-found]
    from gurobipy import GRB  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover
    gp = None
    GRB = None
    GUROBI_IMPORT_ERROR = exc
else:
    GUROBI_IMPORT_ERROR = None

from ..instance.dummy_expansion import expand_with_dummy_vertices
from ..traffic.segmented import build_traffic_model


def _speed_unit_factor(distance_unit: str) -> float:
    """Convert 1 m/s → distance_unit / hour."""
    unit = distance_unit.lower()
    if unit in {"meters", "meter", "m"}:
        return 3600.0
    if unit in {"kilometers", "kilometer", "km"}:
        return 3.6
    if unit in {"miles", "mile", "mi"}:
        return 2.23694
    raise ValueError(f"Unsupported distance unit: {distance_unit}")


def _ichoua_travel_time(
    t_depart_h: float,
    distance: float,
    speed_h: np.ndarray,
    seg_duration_h: float,
) -> float:
    """FIFO travel time using Ichoua et al. (2003) step-speed model.

    Integrates across speed-period boundaries as the vehicle traverses the arc.
    Produces piecewise-linear τ_ij(t) that satisfies FIFO.
    """
    K = len(speed_h)
    remaining = float(distance)
    elapsed = 0.0
    t = float(t_depart_h)

    while remaining > 1e-10:
        k = min(int(t / seg_duration_h), K - 1)
        v = max(float(speed_h[k]), 1e-9)
        t_seg_end = (k + 1) * seg_duration_h
        dist_in_seg = v * (t_seg_end - t)
        if dist_in_seg >= remaining:
            elapsed += remaining / v
            remaining = 0.0
        else:
            elapsed += t_seg_end - t
            remaining -= dist_in_seg
            t = t_seg_end

    return elapsed


def precompute_tau(
    dist: np.ndarray,
    arcs: list[tuple[int, int]],
    speed_h: np.ndarray,
    seg_duration_h: float,
    K: int,
) -> np.ndarray:
    """Return tau[i, j, k] = travel time for arc (i,j) departing at start of period k.

    Evaluating τ at the period boundary (T_k) is the standard precharacterization
    approach for period-discretized TD-VRP exact solvers.
    """
    n = dist.shape[0]
    tau = np.zeros((n, n, K), dtype=float)
    for (i, j) in arcs:
        d_ij = float(dist[i, j])
        if d_ij < 1e-10:
            continue
        for k in range(K):
            tau[i, j, k] = _ichoua_travel_time(k * seg_duration_h, d_ij, speed_h, seg_duration_h)
    return tau


def build_td_evrp_model(
    dist: np.ndarray,
    params: dict[str, Any],
    physical_meta: dict[str, Any],
    config: dict[str, Any],
):
    """Build Gurobi MILP for TD-EVRP (Lera-Romero et al. 2024).

    Period-indicator binary variables y[i,j,k] select the departure period for
    each arc. Combined with precomputed τ[i,j,k], this embeds traffic properly
    in the MIP constraints — τ is not an external subroutine but a parameter.
    """
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {GUROBI_IMPORT_ERROR}")

    instance = expand_with_dummy_vertices(dist, physical_meta, params)
    d = instance["dist"]
    p = instance["service_times"]
    V = instance["V"]
    arcs = instance["arcs"]
    customers = instance["customers"]
    f_prime = instance["f_prime"]
    refuel_nodes = instance["refuel_nodes"]
    non_depot = instance["non_depot"]
    depot = instance["depot"]
    afs_nodes = instance["afs_nodes"]
    node_types = instance["node_types"]

    r = float(params["r"])
    Q = float(params["Q"])
    Tmax = float(params["Tmax"])
    m = int(params["m"])
    distance_unit = str(params.get("distance_unit", "meters"))
    unit_factor = _speed_unit_factor(distance_unit)

    # ── Build traffic model and precompute τ ──────────────────────────────────
    free_flow_ms = float(params["speed"]) / unit_factor
    traffic = build_traffic_model(config, free_flow_ms, Tmax * 3600.0)
    seg_duration_h = float(traffic.seg_duration_s) / 3600.0
    K = len(traffic.speed_segments)
    speed_h = traffic.speed_segments.astype(float) * unit_factor  # distance_unit/hour

    print(f"TD-EVRP: K={K} traffic periods, seg={seg_duration_h*60:.1f} min each")

    tau = precompute_tau(d, arcs, speed_h, seg_duration_h, K)
    BIG_M = Tmax + seg_duration_h

    # Depot-adjacent arcs: need explicit return-time feasibility bound
    depot_family = {i for i, t in enumerate(node_types) if t in {"depot", "depot_dummy"}}

    # ── Gurobi model ──────────────────────────────────────────────────────────
    model = gp.Model("td_evrp_lera_romero_2024")
    model.Params.TimeLimit = float(params.get("time_limit", 300.0))
    model.Params.OutputFlag = int(params.get("output_flag", 1))
    model.Params.MIPGap = float(params.get("mip_gap", 0.0))

    # arc indicators
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    # period indicators: y[i,j,k]=1 iff arc (i,j) is used with departure in period k
    y = model.addVars(
        [(i, j, k) for (i, j) in arcs for k in range(K)],
        vtype=GRB.BINARY, name="y",
    )
    # departure time from node i (after service)
    s = model.addVars(V, lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name="s")
    # battery level at arrival at node i
    b = model.addVars(V, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="b")

    out_nbrs: dict[int, list[int]] = {i: [] for i in V}
    in_nbrs: dict[int, list[int]] = {i: [] for i in V}
    for (i, j) in arcs:
        out_nbrs[i].append(j)
        in_nbrs[j].append(i)

    # Minimize total arc distance
    model.setObjective(
        gp.quicksum(d[i, j] * x[i, j] for (i, j) in arcs),
        GRB.MINIMIZE,
    )

    # ── Routing constraints ───────────────────────────────────────────────────
    for i in customers:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_nbrs[i]) == 1,
            name=f"visit_{i}",
        )

    for i in f_prime:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_nbrs[i]) <= 1,
            name=f"afs_{i}",
        )

    for j in V:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nbrs[j]) ==
            gp.quicksum(x[j, k] for k in out_nbrs[j]),
            name=f"flow_{j}",
        )

    model.addConstr(
        gp.quicksum(x[depot, j] for j in out_nbrs[depot]) <= m,
        name="vehicles",
    )

    # ── Period-indicator constraints ──────────────────────────────────────────
    for (i, j) in arcs:
        # Exactly one period selected iff arc used
        model.addConstr(
            gp.quicksum(y[i, j, k] for k in range(K)) == x[i, j],
            name=f"psum_{i}_{j}",
        )
        for k in range(K):
            t_lo = k * seg_duration_h
            t_hi = (k + 1) * seg_duration_h
            # Departure time s[i] must fall in [T_k, T_{k+1}) when y_ijk=1
            model.addConstr(
                s[i] >= t_lo - BIG_M * (1 - y[i, j, k]),
                name=f"plb_{i}_{j}_{k}",
            )
            model.addConstr(
                s[i] <= t_hi + BIG_M * (1 - y[i, j, k]),
                name=f"pub_{i}_{j}_{k}",
            )

    # ── Time propagation ──────────────────────────────────────────────────────
    for (i, j) in arcs:
        if j in depot_family:
            # Arrival at a depot node must complete within Tmax
            for k in range(K):
                model.addConstr(
                    s[i] + float(tau[i, j, k]) <= Tmax + BIG_M * (1 - y[i, j, k]),
                    name=f"return_{i}_{j}_{k}",
                )
        else:
            # Departure from j ≥ departure from i + travel time + service at j
            for k in range(K):
                model.addConstr(
                    s[j] >= s[i] + float(tau[i, j, k]) + p[j] - BIG_M * (1 - y[i, j, k]),
                    name=f"time_{i}_{j}_{k}",
                )

    # Depot departs at time 0; all other nodes depart before Tmax - service
    model.addConstr(s[depot] == 0.0, name="depot_start")
    for i in non_depot:
        if node_types[i] not in {"depot", "depot_dummy"}:
            model.addConstr(s[i] <= Tmax - p[i], name=f"tmax_{i}")

    # ── Battery constraints ───────────────────────────────────────────────────
    for (i, j) in arcs:
        if j in customers:
            model.addConstr(
                b[j] <= b[i] - r * d[i, j] * x[i, j] + Q * (1 - x[i, j]),
                name=f"batt_{i}_{j}",
            )

    for i in refuel_nodes:
        model.addConstr(b[i] == Q, name=f"recharge_{i}")

    for j in customers:
        direct = r * d[j, depot]
        via = [r * (d[j, ell] + d[ell, depot]) for ell in afs_nodes]
        reserve = min([direct] + via)
        model.addConstr(b[j] >= reserve, name=f"reserve_{j}")

    model._x = x
    model._b = b
    model._s = s
    model._instance = instance
    model._K = K
    return model


def _status_name(status_code: int) -> str:
    if gp is None:
        return "UNKNOWN"
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(status_code, f"STATUS_{status_code}")


def solve_and_extract_td(model, params: dict[str, Any]) -> dict[str, Any]:
    """Solve and extract routes from the TD-EVRP model."""
    model.optimize()
    status = model.Status
    status_text = _status_name(status)
    has_feasible = model.SolCount > 0

    print("\n" + "=" * 88)
    print(f"Solver status: {status_text} (code {status}), solutions: {model.SolCount}")

    if not has_feasible:
        print("No feasible solution.")
        return {
            "has_solution": False,
            "status": status_text,
            "objective": None,
            "routes": [],
            "instance": model._instance,
        }

    objective = float(model.ObjVal)
    x = model._x
    b = model._b
    s = model._s
    inst = model._instance
    d = inst["dist"]
    depot = inst["depot"]
    node_names = inst["node_names"]
    node_sumo_ids = inst["node_sumo_ids"]
    V = inst["V"]

    active_arcs = [(i, j) for (i, j) in inst["arcs"] if x[i, j].X > 0.5]
    succ: dict[int, list[int]] = defaultdict(list)
    for i, j in active_arcs:
        succ[i].append(j)

    routes: list[dict[str, Any]] = []
    covered: set[tuple[int, int]] = set()

    for vehicle_id, first in enumerate(succ.get(depot, []), start=1):
        nodes = [depot, first]
        covered.add((depot, first))
        cur = first
        for _ in range(len(V) + 5):
            if cur == depot:
                break
            nxt_list = succ.get(cur, [])
            if not nxt_list:
                break
            nxt = nxt_list[0]
            nodes.append(nxt)
            covered.add((cur, nxt))
            if nxt in nodes[:-1]:
                break
            cur = nxt

        dist_total = sum(float(d[u, v]) for u, v in zip(nodes[:-1], nodes[1:]))
        stops = [
            {
                "node_index": nd,
                "node_name": node_names[nd],
                "sumo_node_id": node_sumo_ids[nd],
                "fuel": float(b[nd].X),
                "arrival_time": float(s[nd].X),
            }
            for nd in nodes
        ]
        routes.append({"vehicle": vehicle_id, "nodes": nodes, "distance": dist_total, "stops": stops})

    print(f"Objective (total distance): {objective:.3f} {params.get('distance_unit', 'units')}")
    for route in routes:
        seq = " -> ".join(node_names[nd] for nd in route["nodes"])
        print(f"  Vehicle {route['vehicle']}: {seq}  [{route['distance']:.3f}]")

    return {
        "has_solution": True,
        "status": status_text,
        "objective": objective,
        "routes": routes,
        "instance": inst,
    }
