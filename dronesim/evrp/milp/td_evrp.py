"""Gurobi ILP for the Time-Dependent Electric VRP — Lu et al. (2020).

Reference:
    Lu J., Chen Y., Hao J.-K., He R. (2020). "The Time-dependent Electric
    Vehicle Routing Problem: Model and solution." Expert Systems with
    Applications. https://doi.org/10.1016/j.eswa.2020.113593

Formulation notes:
  * Three-level time-of-day speed profile with morning congestion [0, T1],
    free-flow period [T1, T2], and evening congestion [T2, u0].
  * Discrete free-flow speed levels R = {v^1, ..., v^k} with v^1 = v_c and
    v^k = v_f (the optimizer picks one per used arc).
  * Five departure intervals M = {1, ..., 5} split by the 3-level profile and
    the Ichoua (2003) FIFO boundary crossings at T1 and T2.
  * Energy model (Franceschetti 2013 / Bektas-Laporte 2011):
        E(d, v, f) = alpha * (mu + f) * d  +  beta * d / v  +  gamma * d * v^2
    CARGO IS RELAXED HERE: f_ij = 0 across all arcs, so the weight module is
    simply alpha * mu * d.
  * Departure-time / departure-mode variables t_ij^{mr}, z_ij^{mr} linearize
    the time-dependent travel time T_ij^{mr} and energy E_ij^{mr} — both are
    affine in t_ij^{mr}. The identity z_ij^{mr} * t_ij^{mr} = t_ij^{mr}
    (enforced by constraint (22)) linearizes every T*z and E*z product.

Objective (£): energy cost + driver wage + acquisition cost.
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


# ── Unit helpers ──────────────────────────────────────────────────────────────

def _kmh_to_ms(kmh: float) -> float:
    return float(kmh) / 3.6


def _distance_to_meters(value: float, unit: str) -> float:
    u = unit.lower()
    if u in {"meters", "meter", "m"}:
        return float(value)
    if u in {"kilometers", "kilometer", "km"}:
        return float(value) * 1000.0
    if u in {"miles", "mile", "mi"}:
        return float(value) * 1609.344
    raise ValueError(f"Unsupported distance unit: {unit}")


# ── Per-arc coefficients for T^{mr} and E^{mr} ────────────────────────────────
#
# Given an arc (i,j) with distance d (m), congestion speed vc (m/s), free-flow
# speed level v^r (m/s), and boundaries T1, T2 (s), each (m, r) combination is
# characterised by:
#
#     interval bounds    b^{m-1}, b^m   (clipped to [0, u0])
#     travel time        T = A_T + B_T * t
#     energy consumed    E = A_E + B_E * t   (with f = 0)
#
# where t is the departure-from-i time (seconds).

def _interval_bounds(
    m: int,
    d: float,
    vc: float,
    vr: float,
    T1: float,
    T2: float,
    u0: float,
) -> tuple[float, float]:
    """(b_prev, b_curr) for departure interval m on arc of length d, level r."""
    if m == 1:
        b_prev = 0.0
        b_curr = T1 - d / vc
    elif m == 2:
        b_prev = T1 - d / vc
        b_curr = T1
    elif m == 3:
        b_prev = T1
        b_curr = T2 - d / vr
    elif m == 4:
        b_prev = T2 - d / vr
        b_curr = T2
    elif m == 5:
        b_prev = T2
        b_curr = u0
    else:
        return (0.0, 0.0)

    b_prev = max(0.0, b_prev)
    b_curr = min(u0, b_curr)
    return (b_prev, b_curr)


def _travel_coefs(
    m: int,
    d: float,
    vc: float,
    vr: float,
    T1: float,
    T2: float,
) -> tuple[float, float]:
    """Return (A_T, B_T) such that T(t) = A_T + B_T * t for interval m.

    Derived from Lu et al. Eq. (4) via algebraic expansion.
    """
    if m in (1, 5):
        return (d / vc, 0.0)
    if m == 3:
        return (d / vr, 0.0)
    if m == 2:
        # T = (T1 - t)(1 - vc/vr) + d/vr
        a = T1 * (1.0 - vc / vr) + d / vr
        b = vc / vr - 1.0
        return (a, b)
    if m == 4:
        # T = (T2 - t)(1 - vr/vc) + d/vc
        a = T2 * (1.0 - vr / vc) + d / vc
        b = vr / vc - 1.0
        return (a, b)
    return (0.0, 0.0)


def _energy_coefs(
    m: int,
    d: float,
    vc: float,
    vr: float,
    T1: float,
    T2: float,
    alpha: float,
    beta: float,
    gamma: float,
    mu: float,
) -> tuple[float, float]:
    """Return (A_E, B_E) such that E(t) = A_E + B_E * t for interval m (f=0).

    Derived from Lu et al. Eq. (5) with f_ij = 0.
    """
    weight_term = alpha * mu * d

    if m in (1, 5):
        # Constant speed vc throughout
        a = weight_term + beta * d / vc + gamma * d * vc * vc
        return (a, 0.0)
    if m == 3:
        # Constant speed vr throughout
        a = weight_term + beta * d / vr + gamma * d * vr * vr
        return (a, 0.0)
    if m == 2:
        # E = α μ d + β d/vr + γ d vr^2  +  (T1 - t) · C2
        # where C2 = β(1 - vc/vr) + γ · vc · (vc^2 - vr^2)
        c2 = beta * (1.0 - vc / vr) + gamma * vc * (vc * vc - vr * vr)
        a = weight_term + beta * d / vr + gamma * d * vr * vr + c2 * T1
        b = -c2
        return (a, b)
    if m == 4:
        # E = α μ d + β d/vc + γ d vc^2  +  (T2 - t) · C4
        # where C4 = β(1 - vr/vc) + γ · vr · (vr^2 - vc^2)
        c4 = beta * (1.0 - vr / vc) + gamma * vr * (vr * vr - vc * vc)
        a = weight_term + beta * d / vc + gamma * d * vc * vc + c4 * T2
        b = -c4
        return (a, b)
    return (0.0, 0.0)


# ── Config extraction ─────────────────────────────────────────────────────────

def _read_traffic_lu(config: dict[str, Any], u0_s: float) -> dict[str, Any]:
    cfg = config.get("traffic_lu", {})
    vc_ms = _kmh_to_ms(float(cfg.get("v_congestion_kmh", 15.0)))
    vf_ms = _kmh_to_ms(float(cfg.get("v_freeflow_kmh", 50.0)))
    if vc_ms <= 0.0 or vf_ms <= vc_ms:
        raise ValueError("traffic_lu: require 0 < v_congestion_kmh < v_freeflow_kmh")

    k = max(2, int(cfg.get("num_speed_levels", 3)))
    levels = np.linspace(vc_ms, vf_ms, k)

    T1_s = float(cfg.get("T1_hours", 1.0)) * 3600.0
    T2_s = float(cfg.get("T2_hours", 3.0)) * 3600.0
    if not (0.0 < T1_s < T2_s < u0_s):
        raise ValueError(
            f"traffic_lu: require 0 < T1 < T2 < u0 (got T1={T1_s}, T2={T2_s}, u0={u0_s})"
        )
    return {
        "vc_ms": vc_ms,
        "vf_ms": vf_ms,
        "k_levels": k,
        "speed_levels_ms": levels,
        "T1_s": T1_s,
        "T2_s": T2_s,
        "u0_s": u0_s,
    }


def _read_cost_model(config: dict[str, Any]) -> dict[str, float]:
    cfg = config.get("cost_model", {})
    return {
        "alpha": float(cfg.get("alpha", 0.27272)),              # m/s^2
        "beta": float(cfg.get("beta", 33000.0)),                # J/s  (= 33 kJ/s)
        "gamma": float(cfg.get("gamma", 4.58339)),              # kg/m
        "lambda_energy": float(cfg.get("lambda_energy", 0.14 / 3.6e6)),  # £/J
        "delta_wage": float(cfg.get("delta_wage", 0.0022)),     # £/s
        "mu_curb": float(cfg.get("mu_curb", 6350.0)),           # kg
        "h_acquisition": float(cfg.get("h_acquisition", 200.0)),  # £
        "Q_energy_j": float(cfg.get("energy_capacity_j", 7.2e7)),  # J (20 kWh)
    }


# ── Model builder ─────────────────────────────────────────────────────────────

def build_td_evrp_milp_model(
    dist: np.ndarray,
    params: dict[str, Any],
    physical_meta: dict[str, Any],
    config: dict[str, Any],
):
    """Build the Lu et al. (2020) TDEVRP ILP (cargo relaxed) in Gurobi.

    Returns the Gurobi Model with instance metadata attached via model._*.
    """
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {GUROBI_IMPORT_ERROR}")

    # ── Instance expansion (AFS + depot dummies) ──────────────────────────────
    instance = expand_with_dummy_vertices(dist, physical_meta, params)
    d_raw = instance["dist"]
    p_raw = instance["service_times"]           # hours
    V = instance["V"]
    arcs: list[tuple[int, int]] = instance["arcs"]
    customers: list[int] = instance["customers"]
    afs_nodes: list[int] = instance["afs_nodes"]
    refuel_nodes: list[int] = instance["refuel_nodes"]
    non_depot: list[int] = instance["non_depot"]
    depot: int = instance["depot"]
    node_types: list[str] = instance["node_types"]
    depot_dummies = [i for i, t in enumerate(node_types) if t == "depot_dummy"]

    # ── Unit conversions to SI (meters, seconds) ──────────────────────────────
    distance_unit = str(params.get("distance_unit", "meters"))
    d_m = np.array(
        [[_distance_to_meters(d_raw[i, j], distance_unit) for j in range(len(V))]
         for i in range(len(V))],
        dtype=float,
    )
    p_s = np.array([float(p_raw[i]) * 3600.0 for i in range(len(V))], dtype=float)

    Tmax_h = float(params["Tmax"])
    u0_s = Tmax_h * 3600.0
    K_vehicles = int(params["m"])

    # ── Traffic and cost parameters ───────────────────────────────────────────
    traffic = _read_traffic_lu(config, u0_s)
    vc = traffic["vc_ms"]
    speed_levels = traffic["speed_levels_ms"]
    T1 = traffic["T1_s"]
    T2 = traffic["T2_s"]

    cost = _read_cost_model(config)
    alpha = cost["alpha"]
    beta = cost["beta"]
    gamma = cost["gamma"]
    lam = cost["lambda_energy"]
    delta = cost["delta_wage"]
    mu_curb = cost["mu_curb"]
    h_acq = cost["h_acquisition"]
    Q = cost["Q_energy_j"]

    M_set = [1, 2, 3, 4, 5]
    R_set = list(range(len(speed_levels)))
    print(
        f"TD-EVRP-MILP (Lu 2020): |V|={len(V)} |A|={len(arcs)} "
        f"K={K_vehicles} |R|={len(R_set)} vc={vc:.2f} m/s "
        f"vf={speed_levels[-1]:.2f} m/s T1={T1/3600:.2f}h T2={T2/3600:.2f}h"
    )

    # ── Precompute per-arc (m, r) data, skipping empty intervals ──────────────
    valid_mr: dict[tuple[int, int], list[tuple[int, int]]] = {}
    b_lo: dict[tuple[int, int, int, int], float] = {}
    b_hi: dict[tuple[int, int, int, int], float] = {}
    A_T: dict[tuple[int, int, int, int], float] = {}
    B_T: dict[tuple[int, int, int, int], float] = {}
    A_E: dict[tuple[int, int, int, int], float] = {}
    B_E: dict[tuple[int, int, int, int], float] = {}

    EPS = 1e-9
    for (i, j) in arcs:
        d_ij = float(d_m[i, j])
        combos: list[tuple[int, int]] = []
        for r_idx, vr in enumerate(speed_levels):
            for m in M_set:
                lo, hi = _interval_bounds(m, d_ij, vc, float(vr), T1, T2, u0_s)
                if hi - lo <= EPS:
                    continue
                combos.append((m, r_idx))
                b_lo[(i, j, m, r_idx)] = lo
                b_hi[(i, j, m, r_idx)] = hi
                at, bt = _travel_coefs(m, d_ij, vc, float(vr), T1, T2)
                ae, be = _energy_coefs(
                    m, d_ij, vc, float(vr), T1, T2, alpha, beta, gamma, mu_curb,
                )
                A_T[(i, j, m, r_idx)] = at
                B_T[(i, j, m, r_idx)] = bt
                A_E[(i, j, m, r_idx)] = ae
                B_E[(i, j, m, r_idx)] = be
        valid_mr[(i, j)] = combos

    # ── Gurobi model ──────────────────────────────────────────────────────────
    model = gp.Model("td_evrp_milp_lu_2020")
    model.Params.TimeLimit = float(params.get("time_limit", 300.0))
    model.Params.OutputFlag = int(params.get("output_flag", 1))
    model.Params.MIPGap = float(params.get("mip_gap", 0.0))

    # Decision variables
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    zmr_keys = [
        (i, j, m, r) for (i, j) in arcs for (m, r) in valid_mr[(i, j)]
    ]
    z = model.addVars(zmr_keys, vtype=GRB.BINARY, name="z")
    t = model.addVars(zmr_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

    w = model.addVars(V, lb=0.0, ub=u0_s, vtype=GRB.CONTINUOUS, name="w")
    y = model.addVars(V, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="y")

    out_nbrs: dict[int, list[int]] = {i: [] for i in V}
    in_nbrs: dict[int, list[int]] = {i: [] for i in V}
    for (i, j) in arcs:
        out_nbrs[i].append(j)
        in_nbrs[j].append(i)

    # ── Objective ─────────────────────────────────────────────────────────────
    # Energy cost: λ · Σ E*z = λ · Σ (A_E · z + B_E · t)  (since z·t = t)
    energy_cost = gp.quicksum(
        lam * (A_E[k] * z[k] + B_E[k] * t[k]) for k in zmr_keys
    )

    # Driver wage — Lu et al. Eq. (7) and (8), extended so the main depot
    # participates too (vehicles can return directly to the main depot as
    # well as via depot dummies).
    # wage at node i = δ · (service_start_i − dep_from_pred_of_i)
    # For i ∈ C ∪ afs_nodes: dep_from_i = Σ t_{i,·,m,r}; we sum that form.
    wage_cf = gp.LinExpr()
    for i in customers + afs_nodes:
        dep_from_i = gp.quicksum(
            t[i, j, m, r]
            for j in out_nbrs[i] for (m, r) in valid_mr[(i, j)]
        )
        dep_to_i = gp.quicksum(
            t[k, i, m, r]
            for k in in_nbrs[i] for (m, r) in valid_mr[(k, i)]
        )
        wage_cf += delta * (dep_from_i - dep_to_i)

    wage_depot = gp.LinExpr()
    for i in [depot] + depot_dummies:
        dep_to_i = gp.quicksum(
            t[k, i, m, r]
            for k in in_nbrs[i] for (m, r) in valid_mr[(k, i)]
        )
        arrival_at_i = gp.quicksum(
            t[k, i, m, r] + A_T[(k, i, m, r)] * z[k, i, m, r]
            + B_T[(k, i, m, r)] * t[k, i, m, r]
            for k in in_nbrs[i] for (m, r) in valid_mr[(k, i)]
        )
        # For depot dummies mid-route: w_i ≥ arrival (constraint 16). Wage
        # contribution is δ·(w_i − dep_to_i) which bounds the travel-from-pred
        # duration. For the main depot (return endpoint) use arrival directly
        # so the last leg's duration is charged.
        if i == depot:
            wage_depot += delta * (arrival_at_i - dep_to_i)
        else:
            wage_depot += delta * (w[i] - dep_to_i)

    acquisition_cost = gp.quicksum(h_acq * x[depot, j] for j in out_nbrs[depot])

    model.setObjective(energy_cost + wage_cf + wage_depot + acquisition_cost, GRB.MINIMIZE)

    # ── Constraint (10): vehicle count ────────────────────────────────────────
    model.addConstr(
        gp.quicksum(x[depot, j] for j in out_nbrs[depot]) <= K_vehicles,
        name="c10_vehicle_limit",
    )

    # ── Constraint (11): each customer visited exactly once ───────────────────
    for j in customers:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nbrs[j]) == 1,
            name=f"c11_cust_visit_{j}",
        )

    # ── Constraint (12): each station/depot dummy at most once ────────────────
    for j in afs_nodes + depot_dummies:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nbrs[j]) <= 1,
            name=f"c12_station_visit_{j}",
        )

    # ── Constraint (13): flow conservation ────────────────────────────────────
    for j in V:
        model.addConstr(
            gp.quicksum(x[i, j] for i in in_nbrs[j])
            == gp.quicksum(x[j, k] for k in out_nbrs[j]),
            name=f"c13_flow_{j}",
        )

    # (14), (15): cargo/demand constraints — DROPPED per user (cargo relaxed).

    # ── Constraint (16): arrival ≤ service_start, for all non-depot nodes ─────
    # Arrival at j via arc (i,j) and mode (m,r) = t + T*z = t + A_T·z + B_T·t.
    for j in non_depot:
        arrival = gp.quicksum(
            t[i, j, m, r] + A_T[(i, j, m, r)] * z[i, j, m, r]
            + B_T[(i, j, m, r)] * t[i, j, m, r]
            for i in in_nbrs[j] for (m, r) in valid_mr[(i, j)]
        )
        model.addConstr(arrival <= w[j], name=f"c16_arrival_{j}")

    # ── Constraint (17): service_start + service_time ≤ departure ────────────
    for i in non_depot:
        dep_from_i = gp.quicksum(
            t[i, j, m, r]
            for j in out_nbrs[i] for (m, r) in valid_mr[(i, j)]
        )
        # dep_from_i is active only if the node is used (flow in = 1). When
        # unused, all t are 0 and w_i ≤ dep_from_i = 0 via this constraint.
        # To avoid forcing w_i = 0 when i is unused, gate with flow:
        visited_flag = gp.quicksum(x[i, j] for j in out_nbrs[i])
        model.addConstr(
            w[i] + p_s[i] * visited_flag <= dep_from_i + u0_s * (1 - visited_flag),
            name=f"c17_service_{i}",
        )

    # (18): time windows — relaxed to [0, u0] via w's lb/ub. No extra constraint.

    # ── Constraint (19)/(20): energy balance to customer successors ───────────
    # For any predecessor i → customer j:  y_j ≤ y_i − E_ij^{mr}·z + Q·(1 − z)
    # where y_i already equals Q when i is a refuel node (constraint 21), so
    # this unifies Lu's (19) and (20).
    for (i, j) in arcs:
        if j not in customers:
            continue
        for (m, r) in valid_mr[(i, j)]:
            E_lin = A_E[(i, j, m, r)] * z[i, j, m, r] + B_E[(i, j, m, r)] * t[i, j, m, r]
            model.addConstr(
                y[j] <= y[i] - E_lin + Q * (1 - z[i, j, m, r]),
                name=f"c19_energy_{i}_{j}_{m}_{r}",
            )

    # Feasibility: vehicle must have enough energy to traverse any used arc.
    # (Required for arcs ending at refuel nodes, which otherwise bypass 19/20
    # because y[j] is pinned to Q irrespective of y[i].)
    for (i, j, m, r) in zmr_keys:
        E_lin = A_E[(i, j, m, r)] * z[i, j, m, r] + B_E[(i, j, m, r)] * t[i, j, m, r]
        model.addConstr(
            y[i] >= E_lin,
            name=f"c_energy_feasible_{i}_{j}_{m}_{r}",
        )

    # ── Constraint (21): full charge at depot and every refuel station ───────
    for i in refuel_nodes:
        model.addConstr(y[i] == Q, name=f"c21_full_charge_{i}")

    # ── Constraint (22): z · b^{m-1} ≤ t ≤ z · b^m ────────────────────────────
    for (i, j, m, r) in zmr_keys:
        model.addConstr(
            t[i, j, m, r] >= b_lo[(i, j, m, r)] * z[i, j, m, r],
            name=f"c22_lb_{i}_{j}_{m}_{r}",
        )
        model.addConstr(
            t[i, j, m, r] <= b_hi[(i, j, m, r)] * z[i, j, m, r],
            name=f"c22_ub_{i}_{j}_{m}_{r}",
        )

    # ── Constraint (23): Σ_m Σ_r z_ij^{mr} = x_ij ─────────────────────────────
    for (i, j) in arcs:
        if valid_mr[(i, j)]:
            model.addConstr(
                gp.quicksum(z[i, j, m, r] for (m, r) in valid_mr[(i, j)])
                == x[i, j],
                name=f"c23_z_sum_{i}_{j}",
            )
        else:
            # No valid departure interval for this arc — it cannot be used.
            model.addConstr(x[i, j] == 0, name=f"c23_arc_forbidden_{i}_{j}")

    # ── Depot start: service starts at t = 0 ──────────────────────────────────
    model.addConstr(w[depot] == 0.0, name="c_depot_start")

    # ── Stash for solution extraction ────────────────────────────────────────
    model._x = x
    model._z = z
    model._t = t
    model._w = w
    model._y = y
    model._instance = instance
    model._valid_mr = valid_mr
    model._A_T = A_T
    model._B_T = B_T
    model._speed_levels = speed_levels
    model._Q = Q
    model._depot = depot
    return model


# ── Solver / extraction ───────────────────────────────────────────────────────

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


def solve_and_extract_td_milp(model, params: dict[str, Any]) -> dict[str, Any]:
    """Solve the Lu et al. ILP and return reconstructed routes."""
    _ = params  # reserved for future reporting hooks
    model.optimize()
    status = model.Status
    status_text = _status_name(status)
    has_feasible = model.SolCount > 0

    print("\n" + "=" * 88)
    print(f"Solver status: {status_text} (code {status}), solutions: {model.SolCount}")

    if not has_feasible:
        return {
            "has_solution": False,
            "status": status_text,
            "objective": None,
            "routes": [],
            "instance": model._instance,
        }

    objective = float(model.ObjVal)
    x = model._x
    z = model._z
    t = model._t
    w = model._w
    y = model._y
    inst = model._instance
    depot = model._depot
    d = inst["dist"]
    node_names = inst["node_names"]
    node_sumo_ids = inst["node_sumo_ids"]
    V = inst["V"]
    valid_mr = model._valid_mr
    A_T = model._A_T
    B_T = model._B_T

    active_arcs = [(i, j) for (i, j) in inst["arcs"] if x[i, j].X > 0.5]
    succ: dict[int, list[int]] = defaultdict(list)
    for i, j in active_arcs:
        succ[i].append(j)

    routes: list[dict[str, Any]] = []
    for vehicle_id, first in enumerate(succ.get(depot, []), start=1):
        nodes = [depot, first]
        cur = first
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

        dist_total = sum(float(d[u, v]) for u, v in zip(nodes[:-1], nodes[1:]))

        stops: list[dict[str, Any]] = []
        for idx_in_route, nd in enumerate(nodes):
            if idx_in_route == 0:
                # Depot departure: find active outgoing (m,r) for dep time
                arrival_s = 0.0
            else:
                prev = nodes[idx_in_route - 1]
                active = None
                for (m, r) in valid_mr[(prev, nd)]:
                    if z[prev, nd, m, r].X > 0.5:
                        active = (m, r)
                        break
                if active is None:
                    arrival_s = float(w[nd].X)
                else:
                    m, r = active
                    t_val = float(t[prev, nd, m, r].X)
                    T_val = A_T[(prev, nd, m, r)] + B_T[(prev, nd, m, r)] * t_val
                    arrival_s = t_val + T_val
            stops.append(
                {
                    "node_index": nd,
                    "node_name": node_names[nd],
                    "sumo_node_id": node_sumo_ids[nd],
                    "fuel": float(y[nd].X),
                    "arrival_time": arrival_s / 3600.0,  # convert s → hours
                }
            )

        routes.append(
            {
                "vehicle": vehicle_id,
                "nodes": nodes,
                "distance": dist_total,
                "stops": stops,
            }
        )

    print(f"Objective: {objective:.4f}  (energy £ + wage £ + acquisition £)")
    for route in routes:
        seq = " -> ".join(node_names[nd] for nd in route["nodes"])
        print(f"  Vehicle {route['vehicle']}: {seq}  [dist={route['distance']:.2f}]")

    return {
        "has_solution": True,
        "status": status_text,
        "objective": objective,
        "routes": routes,
        "instance": inst,
    }
