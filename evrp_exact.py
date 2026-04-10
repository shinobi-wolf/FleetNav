"""Config-driven exact MILP implementation of the Green Vehicle Routing Problem (G-VRP).

This script builds a G-VRP physical instance from a SUMO network (net.xml), then
solves the Erdogan and Miller-Hooks style MILP with Gurobi.

Workflow:
1) Load YAML config (default: config/gvrp.yaml).
2) Build directed graph from SUMO net file (skip internal edges).
3) Select depot/customers/AFSs according to config.
4) Build complete physical distance matrix from shortest-path distances.
5) Expand AFS/depot dummy vertices.
6) Build and solve MILP; print routes, fuel levels, and arrival times.
7) Plot routes using SUMO coordinates when available.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from sklearn.manifold import MDS

try:
    import gurobipy as gp  # type: ignore[import-not-found]
    from gurobipy import GRB  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment dependent
    gp = None
    GRB = None
    GUROBI_IMPORT_ERROR = exc
else:
    GUROBI_IMPORT_ERROR = None

try:
    import sumolib  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment dependent
    sumolib = None
    SUMOLIB_IMPORT_ERROR = exc
else:
    SUMOLIB_IMPORT_ERROR = None


DEFAULT_CONFIG_PATH = "config/gvrp.yaml"
INF = 1e15


def load_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    return config


def _resolve_net_xml_path(net_path: str) -> str:
    """Resolve SUMO net path from either .sumocfg or .net.xml input."""
    src = Path(net_path)

    if src.suffix == ".sumocfg":
        return str(src.with_suffix(".net.xml"))

    if src.suffix == ".xml" and src.name.endswith(".net.xml"):
        return str(src)

    # Fallback keeps behavior explicit in error messages.
    return str(src)


def _node_position(node_obj: Any) -> tuple[float, float]:
    """Get a stable (x, y) position for a SUMO node object."""
    shape = node_obj.getShape()
    if shape:
        x_coords = [p[0] for p in shape]
        y_coords = [p[1] for p in shape]
        return float(np.mean(x_coords)), float(np.mean(y_coords))

    coord = node_obj.getCoord()
    return float(coord[0]), float(coord[1])


def load_sumo_graph(net_path: str) -> dict[str, Any]:
    """Build directed network graph and shortest-path matrix from SUMO net."""
    if sumolib is None:
        raise RuntimeError(f"sumolib import failed: {SUMOLIB_IMPORT_ERROR}")

    net_xml_path = _resolve_net_xml_path(net_path)
    if not Path(net_xml_path).exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_xml_path}")

    net = sumolib.net.readNet(net_xml_path, withInternal=False)
    g_dir = nx.DiGraph()

    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(":"):
            continue

        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        from_id = from_node.getID()
        to_id = to_node.getID()

        fx, fy = _node_position(from_node)
        tx, ty = _node_position(to_node)

        g_dir.add_node(from_id, x=fx, y=fy)
        g_dir.add_node(to_id, x=tx, y=ty)

        edge_length = float(edge.getLength())
        if edge_length <= 0.0:
            edge_length = float(math.hypot(tx - fx, ty - fy))

        g_dir.add_edge(from_id, to_id, id=edge_id, length=edge_length)

    if len(g_dir.nodes) == 0:
        raise ValueError("No non-internal edges/nodes found while building SUMO graph")

    shortest = dict(nx.all_pairs_dijkstra_path_length(g_dir, weight="length"))
    sorted_nodes = sorted(g_dir.nodes)
    node2idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    idx2node = {idx: node for node, idx in node2idx.items()}

    node_positions = {
        node_id: (float(g_dir.nodes[node_id]["x"]), float(g_dir.nodes[node_id]["y"]))
        for node_id in g_dir.nodes
    }

    return {
        "net_xml_path": net_xml_path,
        "graph": g_dir,
        "shortest_paths": shortest,
        "node2idx": node2idx,
        "idx2node": idx2node,
        "node_positions": node_positions,
    }


def _resolve_depot_node_id(
    graph: nx.DiGraph,
    warehouse_choice: str,
    rng: np.random.Generator,
) -> str:
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Cannot resolve depot: graph has no nodes")

    choice = str(warehouse_choice).strip()
    choice_lower = choice.lower()

    if choice_lower == "center":
        xs = np.array([graph.nodes[n]["x"] for n in nodes], dtype=float)
        ys = np.array([graph.nodes[n]["y"] for n in nodes], dtype=float)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        best_node = min(nodes, key=lambda n: (graph.nodes[n]["x"] - cx) ** 2 + (graph.nodes[n]["y"] - cy) ** 2)
        return str(best_node)

    if choice_lower == "random":
        return str(rng.choice(np.array(nodes, dtype=object)))

    if choice in graph.nodes:
        return choice

    raise ValueError(f"Configured depot node '{warehouse_choice}' does not exist in SUMO graph")


def _determine_customer_count(config: dict[str, Any], candidate_count: int) -> int:
    tasks_cfg = config.get("tasks", {})
    gvrp_cfg = config.get("gvrp", {})

    mode = str(gvrp_cfg.get("customer_sampling", tasks_cfg.get("customer_mode", "density"))).lower()

    if mode == "count":
        count = int(gvrp_cfg.get("customer_count", tasks_cfg.get("customer_count", 0)))
        if count <= 0:
            raise ValueError("customer_count must be > 0 when customer sampling mode is 'count'")
    else:
        density = float(gvrp_cfg.get("customer_density", tasks_cfg.get("task_density", 30.0)))
        if density <= 0:
            raise ValueError("customer density must be > 0")
        count = max(1, int((density * candidate_count) // 100))

    if count > candidate_count:
        raise ValueError(
            f"Requested {count} customers but only {candidate_count} eligible graph nodes are available"
        )

    return count


def _extract_pair_distance(
    src: str,
    dst: str,
    shortest_paths: dict[str, dict[str, float]],
) -> float:
    """Return symmetric distance for an undirected MILP from directed shortest paths."""
    forward = float(shortest_paths.get(src, {}).get(dst, INF))
    backward = float(shortest_paths.get(dst, {}).get(src, INF))

    forward_ok = np.isfinite(forward) and forward < INF
    backward_ok = np.isfinite(backward) and backward < INF

    if not forward_ok and not backward_ok:
        raise ValueError(f"No path between selected SUMO nodes '{src}' and '{dst}'")

    if forward_ok and backward_ok:
        return 0.5 * (forward + backward)

    return forward if forward_ok else backward


def _distance_unit_factor(distance_unit: str) -> float:
    unit = distance_unit.lower()
    if unit in {"meters", "meter", "m"}:
        return 1.0
    if unit in {"kilometers", "kilometer", "km"}:
        return 1e-3
    if unit in {"miles", "mile", "mi"}:
        return 0.000621371192237334

    raise ValueError(f"Unsupported distance unit '{distance_unit}'")


def _convert_consumption_rate(base_gal_per_mile: float, distance_unit: str) -> float:
    """Convert fuel consumption rate to gallons per configured distance unit."""
    unit = distance_unit.lower()
    if unit in {"miles", "mile", "mi"}:
        return base_gal_per_mile
    if unit in {"kilometers", "kilometer", "km"}:
        return base_gal_per_mile * 0.621371192237334
    if unit in {"meters", "meter", "m"}:
        return base_gal_per_mile * 0.000621371192237334

    raise ValueError(f"Unsupported distance unit '{distance_unit}'")


def _convert_speed_from_mph(speed_mph: float, distance_unit: str) -> float:
    """Convert mph to configured distance unit per hour."""
    unit = distance_unit.lower()
    if unit in {"miles", "mile", "mi"}:
        return speed_mph
    if unit in {"kilometers", "kilometer", "km"}:
        return speed_mph * 1.609344
    if unit in {"meters", "meter", "m"}:
        return speed_mph * 1609.344

    raise ValueError(f"Unsupported distance unit '{distance_unit}'")


def build_distance_matrix_from_shortest_paths(
    selected_node_ids: list[str],
    shortest_paths: dict[str, dict[str, float]],
    distance_unit: str,
) -> np.ndarray:
    """Build complete symmetric physical distance matrix from graph shortest paths."""
    n_nodes = len(selected_node_ids)
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    factor = _distance_unit_factor(distance_unit)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dij_meters = _extract_pair_distance(selected_node_ids[i], selected_node_ids[j], shortest_paths)
            dij = dij_meters * factor
            dist[i, j] = dij
            dist[j, i] = dij

    return dist


def build_problem_instance_from_config(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Generate physical G-VRP node set and distance matrix from SUMO graph + config."""
    net_cfg = config.get("net", {})
    if "net_path" not in net_cfg:
        raise KeyError("Config missing required key: net.net_path")

    graph_data = load_sumo_graph(str(net_cfg["net_path"]))
    graph: nx.DiGraph = graph_data["graph"]
    shortest_paths = graph_data["shortest_paths"]

    gvrp_cfg = config.get("gvrp", {})
    solver_cfg = config.get("solver", {})
    fleet_cfg = config.get("fleet", {})

    seed = int(config.get("generation_seed", gvrp_cfg.get("generation_seed", 42)))
    rng = np.random.default_rng(seed)

    depot_choice = str(net_cfg.get("warehouse", gvrp_cfg.get("depot", "center")))
    depot_node_id = _resolve_depot_node_id(graph, depot_choice, rng)

    candidate_nodes = [node_id for node_id in graph.nodes if node_id != depot_node_id]
    if not candidate_nodes:
        raise ValueError("No candidate nodes available after selecting depot")

    customer_count = _determine_customer_count(config, len(candidate_nodes))
    afs_count = int(gvrp_cfg.get("afs_count", 3))
    if afs_count <= 0:
        raise ValueError("gvrp.afs_count must be > 0")

    if customer_count + afs_count > len(candidate_nodes):
        raise ValueError(
            "Requested customers + AFSs exceeds available non-depot nodes: "
            f"{customer_count} + {afs_count} > {len(candidate_nodes)}"
        )

    customer_nodes = rng.choice(np.array(candidate_nodes, dtype=object), size=customer_count, replace=False)
    customer_nodes = [str(node) for node in customer_nodes.tolist()]

    remaining_for_afs = [node_id for node_id in candidate_nodes if node_id not in customer_nodes]
    afs_nodes = rng.choice(np.array(remaining_for_afs, dtype=object), size=afs_count, replace=False)
    afs_nodes = [str(node) for node in afs_nodes.tolist()]

    physical_sumo_ids = [depot_node_id] + customer_nodes + afs_nodes
    physical_node_names = ["D"] + [f"C{i + 1}" for i in range(customer_count)] + [f"F{i + 1}" for i in range(afs_count)]
    physical_node_types = ["depot"] + ["customer"] * customer_count + ["afs"] * afs_count

    physical_dist = build_distance_matrix_from_shortest_paths(
        selected_node_ids=physical_sumo_ids,
        shortest_paths=shortest_paths,
        distance_unit=str(gvrp_cfg.get("distance_unit", "miles")),
    )

    physical_coords = np.array([graph_data["node_positions"][node_id] for node_id in physical_sumo_ids], dtype=float)

    distance_unit = str(gvrp_cfg.get("distance_unit", "miles"))
    speed_mph = float(gvrp_cfg.get("speed_mph", 40.0))

    params: dict[str, Any] = {
        "Q": float(gvrp_cfg.get("fuel_capacity", 50.0)),
        "r": _convert_consumption_rate(float(gvrp_cfg.get("consumption_rate", 0.2)), distance_unit),
        "speed": _convert_speed_from_mph(speed_mph, distance_unit),
        "Tmax": float(gvrp_cfg.get("max_duration_hours", 11.0)),
        "m": int(gvrp_cfg.get("num_vehicles", fleet_cfg.get("size", 3))),
        "afs_dummy_copies": int(gvrp_cfg.get("afs_dummy_copies", gvrp_cfg.get("nf", 3))),
        "depot_dummy_copies": int(gvrp_cfg.get("depot_dummy_copies", gvrp_cfg.get("nf", 3))),
        "customer_service_time": float(gvrp_cfg.get("customer_service_time_hours", 0.5)),
        "refuel_time": float(gvrp_cfg.get("refuel_time_hours", 0.25)),
        "time_limit": float(solver_cfg.get("time_limit_seconds", 300.0)),
        "output_flag": int(solver_cfg.get("output_flag", 1)),
        "mip_gap": float(solver_cfg.get("mip_gap", 0.0)),
        "distance_unit": distance_unit,
        "plot": bool(gvrp_cfg.get("plot", True)),
    }

    physical_meta = {
        "physical_node_names": physical_node_names,
        "physical_node_types": physical_node_types,
        "physical_sumo_ids": physical_sumo_ids,
        "depot_index": 0,
    }

    generation_info = {
        "seed": seed,
        "net_xml_path": graph_data["net_xml_path"],
        "depot_sumo_id": depot_node_id,
        "customer_sumo_ids": customer_nodes,
        "afs_sumo_ids": afs_nodes,
        "physical_coords": physical_coords,
    }

    return physical_dist, params, physical_meta, generation_info


def build_distance_matrix(config: dict[str, Any]) -> np.ndarray:
    """Compatibility wrapper for config-driven distance matrix generation."""
    dist, _, _, _ = build_problem_instance_from_config(config)
    return dist


def _expand_with_dummy_vertices(
    physical_dist: np.ndarray,
    physical_meta: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Expand dynamic physical nodes with AFS and depot dummy copies."""
    afs_dummy_copies = int(params["afs_dummy_copies"])
    depot_dummy_copies = int(params["depot_dummy_copies"])

    physical_names: list[str] = list(physical_meta["physical_node_names"])
    physical_types: list[str] = list(physical_meta["physical_node_types"])
    physical_sumo_ids: list[str] = list(physical_meta["physical_sumo_ids"])
    depot = int(physical_meta["depot_index"])

    afs_indices = [idx for idx, node_type in enumerate(physical_types) if node_type == "afs"]

    node_names = list(physical_names)
    node_types = list(physical_types)
    node_sumo_ids = list(physical_sumo_ids)
    parent_index = list(range(len(physical_names)))

    for afs_idx in afs_indices:
        afs_name = physical_names[afs_idx]
        afs_node_id = physical_sumo_ids[afs_idx]
        for copy_id in range(1, afs_dummy_copies + 1):
            node_names.append(f"{afs_name}_d{copy_id}")
            node_types.append("afs_dummy")
            node_sumo_ids.append(afs_node_id)
            parent_index.append(afs_idx)

    depot_name = physical_names[depot]
    depot_node_id = physical_sumo_ids[depot]
    for copy_id in range(1, depot_dummy_copies + 1):
        node_names.append(f"{depot_name}_d{copy_id}")
        node_types.append("depot_dummy")
        node_sumo_ids.append(depot_node_id)
        parent_index.append(depot)

    n_nodes = len(node_names)
    expanded_dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        pi = parent_index[i]
        for j in range(n_nodes):
            pj = parent_index[j]
            expanded_dist[i, j] = physical_dist[pi, pj]

    speed = float(params["speed"])
    travel_time = expanded_dist / speed

    service_times = np.zeros(n_nodes, dtype=float)
    customer_service = float(params["customer_service_time"])
    refuel_service = float(params["refuel_time"])
    for node in range(n_nodes):
        if node_types[node] == "customer":
            service_times[node] = customer_service
        elif node_types[node] in {"afs", "afs_dummy", "depot_dummy"}:
            service_times[node] = refuel_service
        else:
            service_times[node] = 0.0

    all_nodes = list(range(n_nodes))
    customers = [i for i, t in enumerate(node_types) if t == "customer"]
    afs_nodes = [i for i, t in enumerate(node_types) if t in {"afs", "afs_dummy"}]
    depot_dummy_nodes = [i for i, t in enumerate(node_types) if t == "depot_dummy"]
    refuel_nodes = [depot] + afs_nodes + depot_dummy_nodes

    f_prime = afs_nodes + depot_dummy_nodes
    non_depot = [i for i in all_nodes if i != depot]
    arcs = [
        (i, j)
        for i in all_nodes
        for j in all_nodes
        if i != j and parent_index[i] != parent_index[j]
    ]

    return {
        "dist": expanded_dist,
        "travel_time": travel_time,
        "service_times": service_times,
        "node_names": node_names,
        "node_types": node_types,
        "node_sumo_ids": node_sumo_ids,
        "parent_index": parent_index,
        "V": all_nodes,
        "arcs": arcs,
        "depot": depot,
        "customers": customers,
        "f_prime": f_prime,
        "afs_nodes": afs_nodes,
        "refuel_nodes": refuel_nodes,
        "non_depot": non_depot,
    }


def build_gvrp_model(dist: np.ndarray, params: dict[str, Any], physical_meta: dict[str, Any] | None = None):
    """Build and return the paper-style G-VRP MILP model in Gurobi."""
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {GUROBI_IMPORT_ERROR}")

    if physical_meta is None:
        physical_meta = params.get("_physical_meta")

    if physical_meta is None:
        raise ValueError("physical_meta is required to build dynamic G-VRP model")

    instance = _expand_with_dummy_vertices(dist, physical_meta, params)
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

    # Decision variables:
    # x_ij = 1 if arc (i,j) is traversed, else 0.
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    out_neighbors: dict[int, list[int]] = {i: [] for i in V}
    in_neighbors: dict[int, list[int]] = {i: [] for i in V}
    for i, j in arcs:
        out_neighbors[i].append(j)
        in_neighbors[j].append(i)

    # y_j = fuel level upon arrival at node j.
    y = model.addVars(V, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="y")

    # s_j = arrival time at node j.
    s = model.addVars(V, lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name="s")

    # (1) Objective: minimize total traveled distance.
    model.setObjective(gp.quicksum(d[i, j] * x[i, j] for i, j in arcs), GRB.MINIMIZE)

    # (2) Each customer has exactly one successor.
    for i in customers:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_neighbors[i]) == 1,
            name=f"c2_customer_successor_{i}",
        )

    # (3) Each AFS and dummy vertex has at most one successor.
    for i in f_prime:
        model.addConstr(
            gp.quicksum(x[i, j] for j in out_neighbors[i]) <= 1,
            name=f"c3_afs_dummy_successor_{i}",
        )

    # (4) Flow conservation for all nodes in V'.
    for j in V:
        incoming = gp.quicksum(x[i, j] for i in in_neighbors[j])
        outgoing = gp.quicksum(x[j, k] for k in out_neighbors[j])
        model.addConstr(incoming - outgoing == 0, name=f"c4_flow_{j}")

    # (5) At most m vehicles depart depot.
    model.addConstr(
        gp.quicksum(x[depot, j] for j in out_neighbors[depot]) <= m,
        name="c5_depot_depart_limit",
    )

    # (6) At most m vehicles return to depot.
    model.addConstr(
        gp.quicksum(x[i, depot] for i in in_neighbors[depot]) <= m,
        name="c6_depot_return_limit",
    )

    # (7) Time tracking and subtour elimination.
    for i, j in arcs:
        if j in non_depot:
            model.addConstr(
                s[j] >= s[i] + (t[i, j] + p[j]) * x[i, j] - Tmax * (1 - x[i, j]),
                name=f"c7_time_track_{i}_{j}",
            )

    # (8) Depot departure time bound.
    model.addConstr(s[depot] >= 0.0, name="c8_depot_time_lb")
    model.addConstr(s[depot] <= Tmax, name="c8_depot_time_ub")

    # (9) Arrival time bounds for all non-depot nodes.
    for j in non_depot:
        model.addConstr(s[j] >= t[depot, j], name=f"c9_arrival_lb_{j}")
        model.addConstr(
            s[j] <= Tmax - (t[j, depot] + p[j]),
            name=f"c9_arrival_ub_{j}",
        )

    # (10) Fuel tracking to customer nodes.
    for j in customers:
        for i in in_neighbors[j]:
            model.addConstr(
                y[j] <= y[i] - r * d[i, j] * x[i, j] + Q * (1 - x[i, j]),
                name=f"c10_fuel_track_{i}_{j}",
            )

    # (11) Fuel reset to full capacity at AFS/depot/refuel-copy nodes.
    for j in refuel_nodes:
        model.addConstr(y[j] == Q, name=f"c11_fuel_reset_{j}")

    # (12) Customer must keep enough fuel to reach depot directly or via nearest AFS.
    for j in customers:
        direct_to_depot = r * d[j, depot]
        via_station = [r * (d[j, l] + d[l, depot]) for l in f_prime]
        reserve_fuel = min([direct_to_depot] + via_station)
        model.addConstr(y[j] >= reserve_fuel, name=f"c12_fuel_reserve_{j}")

    # (13) Binary integrality on x_ij is enforced directly in variable type.

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


def _build_plot_coordinates(
    physical_dist: np.ndarray,
    instance: dict[str, Any],
    physical_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Build plotting coordinates from SUMO coordinates when available, otherwise MDS."""
    if physical_coords is not None:
        base_coords = np.array(physical_coords, dtype=float)
    else:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            n_init=4,
            normalized_stress="auto",
        )
        base_coords = mds.fit_transform(physical_dist)

    node_coords = np.zeros((len(instance["V"]), 2), dtype=float)
    parent_seen_count: dict[int, int] = defaultdict(int)

    for node, parent in enumerate(instance["parent_index"]):
        coord = base_coords[parent].copy()
        node_type = instance["node_types"][node]
        if node_type in {"afs_dummy", "depot_dummy"}:
            parent_seen_count[parent] += 1
            k = parent_seen_count[parent]
            angle = (2.0 * math.pi * k) / 6.0
            radius = 5.0 + 1.5 * (k - 1)
            coord += np.array([radius * math.cos(angle), radius * math.sin(angle)])
        node_coords[node] = coord

    return node_coords


def plot_routes(routes: list[dict[str, Any]], coords: np.ndarray, node_types: list[str]) -> None:
    """Visualize node classes and routes in 2D."""
    if len(routes) == 0:
        print("No routes to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 8))

    class_styles = {
        "depot": {"marker": "s", "color": "black", "label": "Depot", "size": 130},
        "depot_dummy": {
            "marker": "s",
            "color": "0.45",
            "label": "Depot dummy",
            "size": 85,
        },
        "customer": {
            "marker": "o",
            "color": "tab:blue",
            "label": "Customer",
            "size": 95,
        },
        "afs": {"marker": "^", "color": "tab:green", "label": "AFS", "size": 130},
        "afs_dummy": {
            "marker": "^",
            "color": "tab:olive",
            "label": "AFS dummy",
            "size": 85,
        },
    }

    for node_class, style in class_styles.items():
        idxs = [i for i, t in enumerate(node_types) if t == node_class]
        if not idxs:
            continue
        ax.scatter(
            coords[idxs, 0],
            coords[idxs, 1],
            marker=style["marker"],
            s=style["size"],
            c=style["color"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    route_palette = plt.colormaps.get_cmap("tab10")
    for ridx, route in enumerate(routes):
        nodes = route["nodes"]
        x_line = [coords[n, 0] for n in nodes]
        y_line = [coords[n, 1] for n in nodes]
        ax.plot(
            x_line,
            y_line,
            color=route_palette(ridx % 10),
            linewidth=2.4,
            alpha=0.95,
            label=f"Vehicle {route['vehicle']} ({route['distance']:.1f})",
            zorder=2,
        )

    for node_idx in range(coords.shape[0]):
        ax.text(
            coords[node_idx, 0] + 1.2,
            coords[node_idx, 1] + 1.2,
            str(node_idx),
            fontsize=8,
            color="0.2",
        )

    ax.set_title("G-VRP solution routes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.28)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        out_file = "gvrp_routes.png"
        plt.savefig(out_file, dpi=200)
        print(f"Plot saved to {out_file} (non-GUI backend detected).")
        plt.close(fig)
    else:
        plt.show()


def maybe_save_instance_json(
    config: dict[str, Any],
    physical_dist: np.ndarray,
    physical_meta: dict[str, Any],
    generation_info: dict[str, Any],
) -> None:
    output_cfg = config.get("output", {})
    out_path = output_cfg.get("save_instance_json", None)
    if not out_path:
        return

    payload = {
        "distance_unit": config.get("gvrp", {}).get("distance_unit", "miles"),
        "physical_node_names": physical_meta["physical_node_names"],
        "physical_node_types": physical_meta["physical_node_types"],
        "physical_sumo_ids": physical_meta["physical_sumo_ids"],
        "distance_matrix": physical_dist.tolist(),
        "generation_info": {
            "seed": generation_info["seed"],
            "net_xml_path": generation_info["net_xml_path"],
            "depot_sumo_id": generation_info["depot_sumo_id"],
            "customer_sumo_ids": generation_info["customer_sumo_ids"],
            "afs_sumo_ids": generation_info["afs_sumo_ids"],
        },
    }

    out_file = Path(out_path)
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved generated instance to {out_file}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and solve config-driven SUMO G-VRP instance")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to G-VRP YAML config",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting even when config enables it",
    )
    args = parser.parse_args()

    if gp is None:
        print("gurobipy is required but could not be imported.")
        print(f"Import error: {GUROBI_IMPORT_ERROR}")
        return

    if sumolib is None:
        print("sumolib is required but could not be imported.")
        print(f"Import error: {SUMOLIB_IMPORT_ERROR}")
        return

    config = load_yaml_config(args.config)
    physical_dist, params, physical_meta, generation_info = build_problem_instance_from_config(config)

    params["_physical_meta"] = physical_meta
    if args.no_plot:
        params["plot"] = False

    print("\nGenerated SUMO-driven G-VRP physical instance:")
    print(f"  net: {generation_info['net_xml_path']}")
    print(f"  seed: {generation_info['seed']}")
    print(f"  depot: {generation_info['depot_sumo_id']}")
    print(f"  customers ({len(generation_info['customer_sumo_ids'])}): {generation_info['customer_sumo_ids']}")
    print(f"  afss ({len(generation_info['afs_sumo_ids'])}): {generation_info['afs_sumo_ids']}")
    print(f"  distance unit: {params['distance_unit']}")

    maybe_save_instance_json(config, physical_dist, physical_meta, generation_info)

    print("\nSolving generated case...")
    model = build_gvrp_model(physical_dist, params, physical_meta)
    result = solve_and_extract(model, physical_dist, params)

    if result["has_solution"] and bool(params.get("plot", True)):
        coords = _build_plot_coordinates(
            physical_dist,
            result["instance"],
            generation_info["physical_coords"],
        )
        plot_routes(result["routes"], coords, result["instance"]["node_types"])

    run_single_vehicle_check(config, physical_dist, params, physical_meta)


if __name__ == "__main__":
    main()
