"""MILP planner for the Time-Dependent Electric VRP (Lu et al. 2020).

This is the standard G-VRP MILP (see ``milp.py``) with **one** structural change:
the arc travel time is a function of the *departure time* from the tail node,
following the three-period time-of-day congestion profile of Lu et al. (2020):

    morning peak  [0,  T1)  at congestion speed v_c
    free-flow     [T1, T2)  at free-flow speed  v_f  (> v_c)
    evening peak  [T2, u0]  at congestion speed v_c  (== morning)

It reuses, unchanged from ``milp.py``:
  * the AFS/depot dummy-vertex expansion, and
  * the distance-based fuel constraints (range ``Q`` in metres, rate ``r``).
Cargo / demand constraints are not modelled (relaxed), exactly as in the base
G-VRP. The vehicle speed is fixed to each period's speed limit — there is **no**
per-arc speed optimisation. The departure-time dependence of the travel time is
linearised with per-arc departure-interval variables ``z_ij^m`` / ``t_ij^m``
(paper Eqs. 16, 17, 22, 23).

**Objective — makespan first.** Distance is what ``milp`` already optimises, and
because ``td_milp``'s time-dependent constraints are *stricter* than ``milp``'s
constant-speed ones, ``td_milp`` can never beat ``milp`` on distance. Its payoff
is instead the metric the congestion profile actually informs: the **makespan**
(time-dependent completion time). ``td_milp`` therefore minimises

    w_makespan · makespan_s  +  w_distance · total_distance_m

with ``w_makespan=1.0`` and a light ``w_distance=0.001`` regulariser by default.
Minimising the makespan drives the optimal schedule to *depart ASAP / no
waiting*, which is exactly how the Simulator drives the routes — so the planned
makespan advantage carries over to the simulated metric. Set ``w_makespan=0`` to
recover the pure min-distance, feasibility-only variant.

Reference:
    Lu J., Chen Y., Hao J.-K., He R. (2020). "The Time-dependent Electric
    Vehicle Routing Problem: Model and solution." Expert Systems with
    Applications 161, 113593. https://doi.org/10.1016/j.eswa.2020.113593
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .base import EVRPPlanner, Route, Stop, expand_with_dummy_vertices

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance

try:
    import gurobipy as gp  # type: ignore[import-not-found]
    from gurobipy import GRB  # type: ignore[import-not-found]
except Exception as _exc:  # pragma: no cover
    gp = None
    GRB = None
    _GUROBI_ERR = _exc
else:
    _GUROBI_ERR = None


# ---------------------------------------------------------------------------
# Time-dependent travel-time helpers (paper Eq. 1, single free-flow speed v_f)
# ---------------------------------------------------------------------------
#
# A trip on an arc of length d departing the tail node at time t falls into one
# of five departure intervals m ∈ {1..5}, split by the period boundaries T1, T2
# and the FIFO crossing points (Ichoua et al. 2003). On interval m the travel
# time is affine in the departure time t:   T(t) = A_T + B_T * t.

_MSET = (1, 2, 3, 4, 5)
_EPS = 1e-9


def _interval_bounds(
    m: int, d: float, vc: float, vf: float, T1: float, T2: float, u0: float
) -> tuple[float, float]:
    """Departure-time bounds (b_prev, b_curr) of interval ``m`` for arc length d."""
    if m == 1:
        lo, hi = 0.0, T1 - d / vc
    elif m == 2:
        lo, hi = T1 - d / vc, T1
    elif m == 3:
        lo, hi = T1, T2 - d / vf
    elif m == 4:
        lo, hi = T2 - d / vf, T2
    elif m == 5:
        lo, hi = T2, u0
    else:
        return 0.0, 0.0
    return max(0.0, lo), min(u0, hi)


def _travel_coefs(
    m: int, d: float, vc: float, vf: float, T1: float, T2: float
) -> tuple[float, float]:
    """Return (A_T, B_T) so that travel time on interval m is A_T + B_T * t."""
    if m in (1, 5):           # whole trip inside a congestion period
        return d / vc, 0.0
    if m == 3:                # whole trip inside the free-flow period
        return d / vf, 0.0
    if m == 2:                # start in morning congestion, cross into free-flow
        # T = (T1 - t)(1 - vc/vf) + d/vf
        return T1 * (1.0 - vc / vf) + d / vf, vc / vf - 1.0
    if m == 4:                # start in free-flow, cross into evening congestion
        # T = (T2 - t)(1 - vf/vc) + d/vc
        return T2 * (1.0 - vf / vc) + d / vc, vf / vc - 1.0
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Config: time-dependent traffic parameters
# ---------------------------------------------------------------------------

def _read_td_traffic(ta_config: dict[str, Any], u0_s: float) -> dict[str, float]:
    """Read ``traffic.td.*`` (the same block the Simulator uses) into SI units.

    Boundaries default to T1 = u0/3, T2 = 2·u0/3 when not specified.
    """
    td = (ta_config.get("traffic", {}) or {}).get("td", {}) or {}
    vc = float(td.get("v_congestion_kmh", 15.0)) / 3.6
    vf = float(td.get("v_freeflow_kmh", 50.0)) / 3.6
    if not (0.0 < vc < vf):
        raise ValueError("traffic.td: require 0 < v_congestion_kmh < v_freeflow_kmh")
    T1 = float(td["T1_hours"]) * 3600.0 if "T1_hours" in td else u0_s / 3.0
    T2 = float(td["T2_hours"]) * 3600.0 if "T2_hours" in td else 2.0 * u0_s / 3.0
    if not (0.0 < T1 < T2 < u0_s):
        raise ValueError(
            f"traffic.td: require 0 < T1 < T2 < horizon "
            f"(T1={T1/3600:.2f}h, T2={T2/3600:.2f}h, horizon={u0_s/3600:.2f}h)"
        )
    return {"vc": vc, "vf": vf, "T1": T1, "T2": T2, "u0": u0_s}


# ---------------------------------------------------------------------------
# Internal: build Gurobi model
# ---------------------------------------------------------------------------

def _build_td_model(
    expanded: dict[str, Any],
    params: dict[str, Any],
    traffic: dict[str, float],
    w_makespan: float,
    w_distance: float,
):
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {_GUROBI_ERR}")

    d = expanded["dist"]
    p = expanded["service_times"]          # seconds
    V = expanded["V"]
    arcs          = expanded["arcs"]
    customers     = expanded["customers"]
    f_prime       = expanded["f_prime"]
    refuel_nodes  = expanded["refuel_nodes"]
    non_depot     = expanded["non_depot"]
    depot         = expanded["depot"]

    r    = float(params["r"])
    Q    = float(params["Q"])
    m    = int(params["m"])

    vc = traffic["vc"]; vf = traffic["vf"]
    T1 = traffic["T1"]; T2 = traffic["T2"]; u0 = traffic["u0"]

    # ── Per-arc valid departure intervals + (b_lo, b_hi, A_T, B_T) ────────────
    valid_m: dict[tuple[int, int], list[int]] = {}
    b_lo: dict[tuple[int, int, int], float] = {}
    b_hi: dict[tuple[int, int, int], float] = {}
    A_T:  dict[tuple[int, int, int], float] = {}
    B_T:  dict[tuple[int, int, int], float] = {}
    for (i, j) in arcs:
        d_ij = float(d[i, j])
        combos: list[int] = []
        for mm in _MSET:
            lo, hi = _interval_bounds(mm, d_ij, vc, vf, T1, T2, u0)
            if hi - lo <= _EPS:
                continue
            combos.append(mm)
            b_lo[(i, j, mm)] = lo
            b_hi[(i, j, mm)] = hi
            at, bt = _travel_coefs(mm, d_ij, vc, vf, T1, T2)
            A_T[(i, j, mm)] = at
            B_T[(i, j, mm)] = bt
        valid_m[(i, j)] = combos

    fifo_limit = min((T2 - T1) * vf, T1 * vc, (u0 - T2) * vc)
    max_d = max((float(d[i, j]) for (i, j) in arcs), default=0.0)

    zkeys = [(i, j, mm) for (i, j) in arcs for mm in valid_m[(i, j)]]
    print(
        f"TD-MILP (Lu 2020): |V|={len(V)} |A|={len(arcs)} m={m}  "
        f"vc={vc:.2f} vf={vf:.2f} m/s  T1={T1/3600:.2f}h T2={T2/3600:.2f}h "
        f"u0={u0/3600:.2f}h  z-vars={len(zkeys)}  "
        f"obj={w_makespan:g}·makespan+{w_distance:g}·dist"
    )
    if max_d >= fifo_limit:
        print(
            f"  [warn] longest arc {max_d:.0f}m ≥ single-period limit {fifo_limit:.0f}m: "
            f"time-dependent travel time is approximate for arcs spanning >2 periods "
            f"(raise v_freeflow_kmh or widen the T1–T2 gap to remove this)."
        )

    # ── Gurobi model ──────────────────────────────────────────────────────────
    model = gp.Model("td_evrp_milp")
    model.Params.TimeLimit = float(params.get("time_limit", 300.0))
    model.Params.OutputFlag = int(params.get("output_flag", 1))
    model.Params.MIPGap = float(params.get("mip_gap", 0.0))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    out_nb: dict[int, list[int]] = {i: [] for i in V}
    in_nb:  dict[int, list[int]] = {i: [] for i in V}
    for i, j in arcs:
        out_nb[i].append(j)
        in_nb[j].append(i)

    y = model.addVars(V, lb=0.0, ub=Q,  vtype=GRB.CONTINUOUS, name="y")   # fuel (range, m)
    w = model.addVars(V, lb=0.0, ub=u0, vtype=GRB.CONTINUOUS, name="w")   # service-start time (s)
    z  = model.addVars(zkeys, vtype=GRB.BINARY, name="z")                  # departure-interval pick
    td = model.addVars(zkeys, lb=0.0, vtype=GRB.CONTINUOUS, name="td")     # departure time from i (s)

    # ── Objective: w_makespan·makespan + w_distance·distance ──────────────────
    # Makespan (max return time) is the metric the time-dependent profile informs,
    # so it is the primary term by default; distance is a light regulariser that
    # keeps routes from wandering among equal-makespan solutions.
    dist_expr = gp.quicksum(d[i, j] * x[i, j] for i, j in arcs)
    T_max = model.addVar(lb=0.0, ub=u0, vtype=GRB.CONTINUOUS, name="T_max")
    model.setObjective(w_makespan * T_max + w_distance * dist_expr, GRB.MINIMIZE)

    # ── Routing constraints (identical to milp.py) ────────────────────────────
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in out_nb[i]) == 1, name=f"c2_{i}")
    for i in f_prime:
        model.addConstr(gp.quicksum(x[i, j] for j in out_nb[i]) <= 1, name=f"c3_{i}")
    for j in V:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nb[j]) - gp.quicksum(x[j, k] for k in out_nb[j]) == 0,
            name=f"c4_{j}",
        )
    model.addConstr(gp.quicksum(x[depot, j] for j in out_nb[depot]) <= m, name="c5")
    model.addConstr(gp.quicksum(x[i, depot] for i in in_nb[depot]) <= m, name="c6")

    # ── Fuel constraints (identical to milp.py) ───────────────────────────────
    for j in customers:
        for i in in_nb[j]:
            model.addConstr(
                y[j] <= y[i] - r * d[i, j] * x[i, j] + Q * (1 - x[i, j]),
                name=f"c10_{i}_{j}",
            )
    for j in refuel_nodes:
        model.addConstr(y[j] == Q, name=f"c11_{j}")
    for j in customers:
        direct  = r * d[j, depot]
        via     = [r * (d[j, l] + d[l, depot]) for l in f_prime]
        reserve = min([direct] + via)
        model.addConstr(y[j] >= reserve, name=f"c12_{j}")

    # ── Time-dependent travel time (replaces milp.py's C7/C8/C9) ──────────────
    # Eq (23): exactly one departure interval is picked iff the arc is used.
    for (i, j) in arcs:
        if valid_m[(i, j)]:
            model.addConstr(
                gp.quicksum(z[i, j, mm] for mm in valid_m[(i, j)]) == x[i, j],
                name=f"c23_{i}_{j}",
            )
        else:
            model.addConstr(x[i, j] == 0, name=f"c23_forbid_{i}_{j}")

    # Eq (22): the departure time lies within the selected interval's bounds.
    for (i, j, mm) in zkeys:
        model.addConstr(td[i, j, mm] >= b_lo[(i, j, mm)] * z[i, j, mm], name=f"c22lo_{i}_{j}_{mm}")
        model.addConstr(td[i, j, mm] <= b_hi[(i, j, mm)] * z[i, j, mm], name=f"c22hi_{i}_{j}_{mm}")

    # Eq (16): arrival at j ≤ service start w[j].  arrival = Σ_m (1+B_T)·td + A_T·z.
    for j in non_depot:
        arrival = gp.quicksum(
            (1.0 + B_T[(i, j, mm)]) * td[i, j, mm] + A_T[(i, j, mm)] * z[i, j, mm]
            for i in in_nb[j] for mm in valid_m[(i, j)]
        )
        model.addConstr(arrival <= w[j], name=f"c16_{j}")

    # Eq (17): departure from i ≥ service start + service time, when i is visited.
    for i in non_depot:
        departure = gp.quicksum(td[i, j, mm] for j in out_nb[i] for mm in valid_m[(i, j)])
        visited   = gp.quicksum(x[i, j] for j in out_nb[i])
        model.addConstr(
            w[i] + p[i] * visited <= departure + u0 * (1 - visited),
            name=f"c17_{i}",
        )

    # Vehicles start at the depot at t = 0; each return must finish within the
    # horizon and defines the makespan T_max (max over all returning vehicles).
    model.addConstr(w[depot] == 0.0, name="c_depot_start")
    for i in in_nb[depot]:
        ret = gp.quicksum(
            (1.0 + B_T[(i, depot, mm)]) * td[i, depot, mm] + A_T[(i, depot, mm)] * z[i, depot, mm]
            for mm in valid_m[(i, depot)]
        )
        model.addConstr(ret <= u0, name=f"c_return_{i}")
        model.addConstr(T_max >= ret, name=f"c_mksp_{i}")

    # Valid lower bound: the vehicle serving the farthest customer cannot return
    # faster than its free-flow round trip (vf is the fastest possible speed).
    # This tightens the otherwise-weak min-max makespan LP relaxation a lot.
    if customers:
        lb_mksp = max(2.0 * float(d[depot, c]) / vf for c in customers)
        model.addConstr(T_max >= lb_mksp, name="c_mksp_lb")

    # ── Stash for extraction ──────────────────────────────────────────────────
    model._x = x
    model._y = y
    model._w = w
    model._z = z
    model._td = td
    model._valid_m = valid_m
    model._A_T = A_T
    model._B_T = B_T
    model._expanded = expanded
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


def _solve_and_extract(model, params: dict[str, Any]) -> list[Route]:
    model.optimize()
    status = _status_name(model.Status)
    print(f"\nSolver status: {status}  solutions: {model.SolCount}")

    if model.SolCount == 0:
        raise RuntimeError(f"TD-MILP found no feasible solution (status={status})")

    print(f"Objective: {float(model.ObjVal):.3f}")

    x        = model._x
    w        = model._w
    z        = model._z
    td       = model._td
    valid_m  = model._valid_m
    A_T      = model._A_T
    B_T      = model._B_T
    expanded = model._expanded
    V        = expanded["V"]
    depot    = expanded["depot"]
    d        = expanded["dist"]
    names    = expanded["node_names"]
    types    = expanded["node_types"]
    sumo_ids = expanded["node_sumo_ids"]
    refuel_set = set(expanded["refuel_nodes"])

    r = float(params["r"])
    Q = float(params["Q"])

    def arc_arrival(i: int, j: int) -> float:
        return sum(
            (1.0 + B_T[(i, j, mm)]) * td[i, j, mm].X + A_T[(i, j, mm)] * z[i, j, mm].X
            for mm in valid_m[(i, j)]
        )

    def arc_departure(i: int, j: int) -> float:
        return sum(td[i, j, mm].X for mm in valid_m[(i, j)])

    active = [(i, j) for (i, j) in expanded["arcs"] if x[i, j].X > 0.5]
    succ: dict[int, list[int]] = defaultdict(list)
    for i, j in active:
        succ[i].append(j)

    routes: list[Route] = []
    for vehicle_id, first in enumerate(succ.get(depot, []), start=1):
        nodes = [depot, first]
        cur   = first
        guard = 0
        while cur != depot and guard < len(V) + 5:
            nxt_list = succ.get(cur, [])
            if not nxt_list:
                break
            nxt = nxt_list[0]
            nodes.append(nxt)
            if nxt in nodes[:-1]:
                break
            cur = nxt
            guard += 1

        prop_fuel = Q
        stops: list[Stop] = []
        for k, nd in enumerate(nodes):
            arrival = 0.0 if k == 0 else arc_arrival(nodes[k - 1], nd)
            if k + 1 < len(nodes):
                departure = arc_departure(nd, nodes[k + 1])
                svc_start = 0.0 if k == 0 else float(w[nd].X)
                wait_s    = max(0.0, departure - svc_start)
            else:
                wait_s = 0.0
            stops.append(Stop(
                sumo_node_id=sumo_ids[nd],
                node_name=names[nd],
                node_type=types[nd],
                planned_fuel=prop_fuel,
                planned_time=arrival,
                planned_wait_s=wait_s,
            ))
            if k + 1 < len(nodes):
                nxt = nodes[k + 1]
                leg = float(d[nd, nxt])
                prop_fuel = Q if nxt in refuel_set else prop_fuel - r * leg

        route_dist = sum(d[u, v] for u, v in zip(nodes[:-1], nodes[1:]))
        finish_s   = arc_arrival(nodes[-2], nodes[-1]) if len(nodes) >= 2 else 0.0
        print(
            f"  Vehicle {vehicle_id}: " + " → ".join(names[nd] for nd in nodes)
            + f"  dist={route_dist:.1f}m  finish={finish_s/3600:.2f}h"
        )
        routes.append(Route(vehicle_id=vehicle_id, stops=stops))

    return routes


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------

class TDMILPPlanner(EVRPPlanner):
    def plan(self, instance: "Instance") -> list[Route]:
        p = instance.problem
        u0_s = float(p.params["Tmax"])
        traffic = _read_td_traffic(instance.ta_config, u0_s)
        td_cfg = instance.ta_config.get("evrp", {}).get("td_milp", {}) or {}
        w_makespan = float(td_cfg.get("w_makespan", 1.0))   # primary: completion time
        w_distance = float(td_cfg.get("w_distance", 0.001))  # light distance regulariser
        expanded = expand_with_dummy_vertices(p.physical_dist, p.physical_meta, p.params)
        model = _build_td_model(expanded, p.params, traffic, w_makespan, w_distance)
        return _solve_and_extract(model, p.params)
