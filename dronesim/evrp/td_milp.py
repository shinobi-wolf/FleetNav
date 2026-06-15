"""Time-Dependent EVRP MILP planner — Lu et al. (2020) (cargo relaxed).

Adapted from dronesim/evrp/milp/td_evrp.py for use with ta_evrp Instance objects.
Key unit differences vs. the original:
  - params["Tmax"] is in SECONDS (not hours)
  - expand_with_dummy_vertices service_times are in SECONDS (not hours)
  - params["Q"] is in metres; params["r"] = 1.0 (range consumed per metre)
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import EVRPPlanner, Route, Stop, expand_with_dummy_vertices

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance

try:
    import gurobipy as gp  # type: ignore[import-not-found]
    from gurobipy import GRB  # type: ignore[import-not-found]
except Exception as _exc:
    gp = None
    GRB = None
    _GUROBI_ERR = _exc
else:
    _GUROBI_ERR = None


# ---------------------------------------------------------------------------
# Travel-time interval helpers (Lu et al. Eq. 4)
# ---------------------------------------------------------------------------

def _interval_bounds(
    m: int, d: float, vc: float, vr: float, T1: float, T2: float, u0: float,
) -> tuple[float, float]:
    if m == 1:
        b_prev, b_curr = 0.0, T1 - d / vc
    elif m == 2:
        b_prev, b_curr = T1 - d / vc, T1
    elif m == 3:
        b_prev, b_curr = T1, T2 - d / vr
    elif m == 4:
        b_prev, b_curr = T2 - d / vr, T2
    elif m == 5:
        b_prev, b_curr = T2, u0
    else:
        return (0.0, 0.0)
    return (max(0.0, b_prev), min(u0, b_curr))


def _travel_coefs(m: int, d: float, vc: float, vr: float, T1: float, T2: float) -> tuple[float, float]:
    if m in (1, 5):
        return (d / vc, 0.0)
    if m == 3:
        return (d / vr, 0.0)
    if m == 2:
        return (T1 * (1.0 - vc / vr) + d / vr, vc / vr - 1.0)
    if m == 4:
        return (T2 * (1.0 - vr / vc) + d / vc, vr / vc - 1.0)
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Config readers
# ---------------------------------------------------------------------------

def _read_traffic_lu(ta_config: dict[str, Any], u0_s: float) -> dict[str, Any]:
    cfg = ta_config.get("traffic_lu", {})
    vc_ms = float(cfg.get("v_congestion_kmh", 15.0)) / 3.6
    vf_ms = float(cfg.get("v_freeflow_kmh", 50.0)) / 3.6
    if vc_ms <= 0.0 or vf_ms <= vc_ms:
        raise ValueError("traffic_lu: require 0 < v_congestion_kmh < v_freeflow_kmh")
    k = max(2, int(cfg.get("num_speed_levels", 3)))
    levels = np.linspace(vc_ms, vf_ms, k)
    T1_s = float(cfg.get("T1_hours", 1.0)) * 3600.0
    T2_s = float(cfg.get("T2_hours", 3.0)) * 3600.0
    if not (0.0 < T1_s < T2_s < u0_s):
        raise ValueError(
            f"traffic_lu: require 0 < T1 < T2 < u0 (T1={T1_s:.0f}s T2={T2_s:.0f}s u0={u0_s:.0f}s)"
        )
    return {
        "vc_ms":           vc_ms,
        "speed_levels_ms": levels,
        "T1_s":            T1_s,
        "T2_s":            T2_s,
        "u0_s":            u0_s,
    }


def _read_cost_model(ta_config: dict[str, Any]) -> dict[str, float]:
    cfg = ta_config.get("cost_model", {})
    return {
        "w_energy":   float(cfg.get("w_energy",   1.0)),
        "w_makespan": float(cfg.get("w_makespan", 1.0)),
    }


# ---------------------------------------------------------------------------
# Internal: build Gurobi model
# ---------------------------------------------------------------------------

def _build_td_evrp_model(
    expanded: dict[str, Any],
    params: dict[str, Any],
    ta_config: dict[str, Any],
):
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {_GUROBI_ERR}")

    d_m          = expanded["dist"]
    p_s          = expanded["service_times"]   # already in seconds
    V            = expanded["V"]
    arcs         = expanded["arcs"]
    customers    = expanded["customers"]
    afs_nodes    = expanded["afs_nodes"]
    refuel_nodes = expanded["refuel_nodes"]
    non_depot    = expanded["non_depot"]
    depot        = expanded["depot"]
    node_types   = expanded["node_types"]
    depot_dummies = [i for i, t in enumerate(node_types) if t == "depot_dummy"]

    u0_s       = float(params["Tmax"])   # already in seconds
    K_vehicles = int(params["m"])
    Q          = float(params["Q"])      # metres (range)
    e_per_m    = float(params["r"])      # = 1.0 (range consumed per metre)

    traffic     = _read_traffic_lu(ta_config, u0_s)
    vc          = traffic["vc_ms"]
    speed_levels = traffic["speed_levels_ms"]
    T1          = traffic["T1_s"]
    T2          = traffic["T2_s"]
    cost        = _read_cost_model(ta_config)
    w_energy    = cost["w_energy"]
    w_makespan  = cost["w_makespan"]

    M_set = [1, 2, 3, 4, 5]
    R_set = list(range(len(speed_levels)))
    print(
        f"TD-EVRP-MILP: |V|={len(V)} |A|={len(arcs)} K={K_vehicles} "
        f"|R|={len(R_set)} vc={vc:.2f}m/s vf={speed_levels[-1]:.2f}m/s "
        f"T1={T1/3600:.2f}h T2={T2/3600:.2f}h Q={Q:.1f}m"
    )

    # Precompute per-arc (m, r) coefficients
    EPS = 1e-9
    valid_mr: dict[tuple[int, int], list[tuple[int, int]]] = {}
    b_lo: dict[tuple[int, int, int, int], float] = {}
    b_hi: dict[tuple[int, int, int, int], float] = {}
    A_T:  dict[tuple[int, int, int, int], float] = {}
    B_T:  dict[tuple[int, int, int, int], float] = {}

    for (i, j) in arcs:
        d_ij = float(d_m[i, j])
        combos: list[tuple[int, int]] = []
        for r_idx, vr in enumerate(speed_levels[1:]):
            for m in M_set:
                lo, hi = _interval_bounds(m, d_ij, vc, float(vr), T1, T2, u0_s)
                if hi - lo <= EPS:
                    continue
                combos.append((m, r_idx))
                b_lo[(i, j, m, r_idx)] = lo
                b_hi[(i, j, m, r_idx)] = hi
                at, bt = _travel_coefs(m, d_ij, vc, float(vr), T1, T2)
                A_T[(i, j, m, r_idx)] = at
                B_T[(i, j, m, r_idx)] = bt
        valid_mr[(i, j)] = combos

    model = gp.Model("td_evrp_milp_lu2020")
    model.Params.TimeLimit = float(params.get("time_limit", 300.0))
    model.Params.OutputFlag = int(params.get("output_flag", 1))
    model.Params.MIPGap = float(params.get("mip_gap", 0.0))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    zmr_keys = [(i, j, m, r) for (i, j) in arcs for (m, r) in valid_mr[(i, j)]]
    z = model.addVars(zmr_keys, vtype=GRB.BINARY, name="z")
    t = model.addVars(zmr_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
    w = model.addVars(V, lb=0.0, ub=u0_s, vtype=GRB.CONTINUOUS, name="w")
    y = model.addVars(V, lb=0.0, ub=Q,    vtype=GRB.CONTINUOUS, name="y")

    out_nb: dict[int, list[int]] = {i: [] for i in V}
    in_nb:  dict[int, list[int]] = {i: [] for i in V}
    for (i, j) in arcs:
        out_nb[i].append(j)
        in_nb[j].append(i)

    # Objective
    total_energy = gp.quicksum(e_per_m * float(d_m[i, j]) * x[i, j] for (i, j) in arcs)
    depot_set = set([depot] + depot_dummies)
    T_max = model.addVar(lb=0.0, ub=u0_s, vtype=GRB.CONTINUOUS, name="T_max")
    for (i, j) in arcs:
        if j not in depot_set:
            continue
        for (m, r) in valid_mr[(i, j)]:
            arrival = A_T[(i, j, m, r)] * z[i, j, m, r] + (1.0 + B_T[(i, j, m, r)]) * t[i, j, m, r]
            model.addConstr(T_max >= arrival, name=f"c_mksp_{i}_{j}_{m}_{r}")

    model.setObjective(w_energy * total_energy + w_makespan * T_max, GRB.MINIMIZE)

    model.addConstr(gp.quicksum(x[depot, j] for j in out_nb[depot]) <= K_vehicles, name="c10")
    for j in customers:
        model.addConstr(gp.quicksum(x[i, j] for i in in_nb[j]) == 1, name=f"c11_{j}")
    for j in afs_nodes + depot_dummies:
        model.addConstr(gp.quicksum(x[i, j] for i in in_nb[j]) <= 1, name=f"c12_{j}")
    for j in V:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nb[j]) == gp.quicksum(x[j, k] for k in out_nb[j]),
            name=f"c13_{j}",
        )
    for j in non_depot:
        arrival = gp.quicksum(
            t[i, j, m, r] + A_T[(i, j, m, r)] * z[i, j, m, r] + B_T[(i, j, m, r)] * t[i, j, m, r]
            for i in in_nb[j] for (m, r) in valid_mr[(i, j)]
        )
        model.addConstr(arrival <= w[j], name=f"c16_{j}")
    for i in non_depot:
        dep = gp.quicksum(t[i, j, m, r] for j in out_nb[i] for (m, r) in valid_mr[(i, j)])
        visited = gp.quicksum(x[i, j] for j in out_nb[i])
        model.addConstr(w[i] + p_s[i] * visited <= dep + u0_s * (1 - visited), name=f"c17_{i}")
    for (i, j) in arcs:
        if j not in customers:
            continue
        E = e_per_m * float(d_m[i, j])
        model.addConstr(y[j] <= y[i] - E * x[i, j] + Q * (1 - x[i, j]), name=f"c19_{i}_{j}")
    for (i, j) in arcs:
        E = e_per_m * float(d_m[i, j])
        model.addConstr(y[i] >= E * x[i, j], name=f"c_efeas_{i}_{j}")
    for i in refuel_nodes:
        model.addConstr(y[i] == Q, name=f"c21_{i}")
    for (i, j, m, r) in zmr_keys:
        model.addConstr(t[i, j, m, r] >= b_lo[(i, j, m, r)] * z[i, j, m, r], name=f"c22_lo_{i}_{j}_{m}_{r}")
        model.addConstr(t[i, j, m, r] <= b_hi[(i, j, m, r)] * z[i, j, m, r], name=f"c22_hi_{i}_{j}_{m}_{r}")
    for (i, j) in arcs:
        if valid_mr[(i, j)]:
            model.addConstr(
                gp.quicksum(z[i, j, m, r] for (m, r) in valid_mr[(i, j)]) == x[i, j],
                name=f"c23_{i}_{j}",
            )
        else:
            model.addConstr(x[i, j] == 0, name=f"c23_forbid_{i}_{j}")
    model.addConstr(w[depot] == 0.0, name="c_depot_start")

    model._x        = x
    model._z        = z
    model._t        = t
    model._w        = w
    model._y        = y
    model._T_max    = T_max
    model._expanded = expanded
    model._valid_mr = valid_mr
    model._A_T      = A_T
    model._B_T      = B_T
    model._Q        = Q
    model._depot    = depot
    return model


# ---------------------------------------------------------------------------
# Internal: solve and extract
# ---------------------------------------------------------------------------

def _status_name(code: int) -> str:
    if gp is None:
        return "UNKNOWN"
    return {
        GRB.OPTIMAL:     "OPTIMAL",
        GRB.TIME_LIMIT:  "TIME_LIMIT",
        GRB.INFEASIBLE:  "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED:   "UNBOUNDED",
        GRB.SUBOPTIMAL:  "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(code, f"STATUS_{code}")


def _solve_and_extract_td(model) -> list[Route]:
    model.printStats()
    try:
        model.optimize()
    except Exception as exc:
        if gp is not None and isinstance(exc, gp.GurobiError) and "size-limited" in str(exc):
            inst = getattr(model, "_expanded", {}) or {}
            raise RuntimeError(
                "TD-EVRP MILP exceeds Gurobi size-limited license "
                f"(|V|={len(inst.get('V', []))}, |A|={len(inst.get('arcs', []))}). "
                "Reduce model size via config (customer_count, afs_count, num_speed_levels, dummy_copies)."
            ) from exc
        raise

    status = _status_name(model.Status)
    print(f"\nSolver status: {status}  solutions: {model.SolCount}")
    if model.SolCount == 0:
        raise RuntimeError(f"TD-EVRP MILP found no feasible solution (status={status})")

    print(f"Objective: {float(model.ObjVal):.4f}  makespan={float(model._T_max.X):.1f}s")

    x        = model._x
    z        = model._z
    t        = model._t
    w        = model._w
    y        = model._y
    expanded = model._expanded
    depot    = model._depot
    d        = expanded["dist"]
    names    = expanded["node_names"]
    types    = expanded["node_types"]
    sumo_ids = expanded["node_sumo_ids"]
    V        = expanded["V"]
    valid_mr = model._valid_mr
    A_T      = model._A_T
    B_T      = model._B_T

    active = [(i, j) for (i, j) in expanded["arcs"] if x[i, j].X > 0.5]
    succ: dict[int, list[int]] = defaultdict(list)
    for i, j in active:
        succ[i].append(j)

    routes: list[Route] = []
    for vehicle_id, first in enumerate(succ.get(depot, []), start=1):
        nodes = [depot, first]
        cur   = first
        for _ in range(len(V) + 5):
            if cur == depot:
                break
            nxt_list = succ.get(cur, [])
            if not nxt_list:
                break
            nxt = nxt_list[0]
            nodes.append(nxt)
            if nxt in nodes[:-1] and nxt != depot:
                break
            cur = nxt

        stops: list[Stop] = []
        for k, nd in enumerate(nodes):
            if k == 0:
                arrival_s = 0.0
            else:
                prev = nodes[k - 1]
                active_mr = next(
                    ((m, r) for (m, r) in valid_mr[(prev, nd)] if z[prev, nd, m, r].X > 0.5),
                    None,
                )
                if active_mr is None:
                    arrival_s = float(w[nd].X)
                else:
                    m, r = active_mr
                    t_val = float(t[prev, nd, m, r].X)
                    arrival_s = t_val + A_T[(prev, nd, m, r)] + B_T[(prev, nd, m, r)] * t_val

            if k + 1 < len(nodes):
                nxt = nodes[k + 1]
                dep_s = sum(t[nd, nxt, m, r].X for (m, r) in valid_mr.get((nd, nxt), []))
            else:
                dep_s = 0.0
            wait_s = max(0.0, dep_s - float(w[nd].X))

            stops.append(Stop(
                sumo_node_id=sumo_ids[nd],
                node_name=names[nd],
                node_type=types[nd],
                planned_fuel=float(y[nd].X),
                planned_time=arrival_s,
                planned_wait_s=wait_s,
            ))

        dist_total = sum(float(d[u, v]) for u, v in zip(nodes[:-1], nodes[1:]))
        print(f"  Vehicle {vehicle_id}: " + " → ".join(names[nd] for nd in nodes) + f"  dist={dist_total:.1f}m")
        routes.append(Route(vehicle_id=vehicle_id, stops=stops))

    return routes


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------

class TDMILPPlanner(EVRPPlanner):
    def plan(self, instance: "Instance") -> list[Route]:
        p        = instance.problem
        expanded = expand_with_dummy_vertices(p.physical_dist, p.physical_meta, p.params)
        model    = _build_td_evrp_model(expanded, p.params, instance.ta_config)
        return _solve_and_extract_td(model)
