"""MILP-based G-VRP planner (Erdogan & Miller-Hooks 2012)."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

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
# Internal: build Gurobi model
# ---------------------------------------------------------------------------

def _build_gvrp_model(expanded: dict[str, Any], params: dict[str, Any]):
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {_GUROBI_ERR}")

    d = expanded["dist"]
    t = expanded["travel_time"]
    p = expanded["service_times"]
    V = expanded["V"]
    arcs          = expanded["arcs"]
    customers     = expanded["customers"]
    f_prime       = expanded["f_prime"]
    refuel_nodes  = expanded["refuel_nodes"]
    non_depot     = expanded["non_depot"]
    depot         = expanded["depot"]

    r    = float(params["r"])
    Q    = float(params["Q"])
    Tmax = float(params["Tmax"])
    m    = int(params["m"])

    model = gp.Model("gvrp_milp")
    model.Params.TimeLimit = float(params.get("time_limit", 300.0))
    model.Params.OutputFlag = int(params.get("output_flag", 1))
    model.Params.MIPGap = float(params.get("mip_gap", 0.0))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    out_nb: dict[int, list[int]] = {i: [] for i in V}
    in_nb:  dict[int, list[int]] = {i: [] for i in V}
    for i, j in arcs:
        out_nb[i].append(j)
        in_nb[j].append(i)

    y = model.addVars(V, lb=0.0, ub=Q,    vtype=GRB.CONTINUOUS, name="y")
    s = model.addVars(V, lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name="s")

    model.setObjective(gp.quicksum(d[i, j] * x[i, j] for i, j in arcs), GRB.MINIMIZE)

    for i in customers:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_nb[i]) == 1,
            name=f"c2_{i}",
        )
    for i in f_prime:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_nb[i]) <= 1,
            name=f"c3_{i}",
        )
    for j in V:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nb[j]) - gp.quicksum(x[j, k] for k in out_nb[j]) == 0,
            name=f"c4_{j}",
        )
    model.addConstr(gp.quicksum(x[depot, j] for j in out_nb[depot]) <= m, name="c5")
    model.addConstr(gp.quicksum(x[i, depot] for i in in_nb[depot]) <= m, name="c6")

    for i, j in arcs:
        if j in non_depot:
            model.addConstr(
                s[j] >= s[i] + (t[i, j] + p[j]) * x[i, j] - Tmax * (1 - x[i, j]),
                name=f"c7_{i}_{j}",
            )
    model.addConstr(s[depot] >= 0.0,   name="c8_lb")
    model.addConstr(s[depot] <= Tmax,  name="c8_ub")
    for j in non_depot:
        model.addConstr(s[j] >= t[depot, j],                name=f"c9_lb_{j}")
        model.addConstr(s[j] <= Tmax - (t[j, depot] + p[j]), name=f"c9_ub_{j}")

    for j in customers:
        for i in in_nb[j]:
            model.addConstr(
                y[j] <= y[i] - r * d[i, j] * x[i, j] + Q * (1 - x[i, j]),
                name=f"c10_{i}_{j}",
            )
    for j in refuel_nodes:
        model.addConstr(y[j] == Q, name=f"c11_{j}")
    for j in customers:
        direct = r * d[j, depot]
        via    = [r * (d[j, l] + d[l, depot]) for l in f_prime]
        reserve = min([direct] + via)
        model.addConstr(y[j] >= reserve, name=f"c12_{j}")

    model._x        = x
    model._y        = y
    model._s        = s
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
        raise RuntimeError(f"MILP found no feasible solution (status={status})")

    objective = float(model.ObjVal)
    print(f"Objective (total distance): {objective:.3f} m")

    x        = model._x
    expanded = model._expanded
    V        = expanded["V"]
    depot    = expanded["depot"]
    d        = expanded["dist"]
    p        = expanded["service_times"]
    names    = expanded["node_names"]
    types    = expanded["node_types"]
    sumo_ids = expanded["node_sumo_ids"]
    refuel_set = set(expanded["refuel_nodes"])

    r      = float(params["r"])
    Q      = float(params["Q"])
    speed  = float(params["speed"])

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

        # Forward-propagate fuel and time
        prop_fuel = Q
        prop_time = 0.0
        stops: list[Stop] = []
        for k, nd in enumerate(nodes):
            stops.append(Stop(
                sumo_node_id=sumo_ids[nd],
                node_name=names[nd],
                node_type=types[nd],
                planned_fuel=prop_fuel,
                planned_time=prop_time,
            ))
            if k + 1 < len(nodes):
                nxt = nodes[k + 1]
                leg = float(d[nd, nxt])
                prop_time += leg / speed + float(p[nxt])
                prop_fuel = Q if nxt in refuel_set else prop_fuel - r * leg

        route_dist = sum(d[u, v] for u, v in zip(nodes[:-1], nodes[1:]))
        print(f"  Vehicle {vehicle_id}: " + " → ".join(names[nd] for nd in nodes) + f"  dist={route_dist:.1f}m")
        routes.append(Route(vehicle_id=vehicle_id, stops=stops))

    return routes


# ---------------------------------------------------------------------------
# Public planner
# ---------------------------------------------------------------------------

class MILPPlanner(EVRPPlanner):
    def plan(self, instance: "Instance") -> list[Route]:
        p = instance.problem
        expanded = expand_with_dummy_vertices(p.physical_dist, p.physical_meta, p.params)
        model    = _build_gvrp_model(expanded, p.params)
        return _solve_and_extract(model, p.params)
