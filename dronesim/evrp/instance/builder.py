"""Build G-VRP problem instances from SUMO networks and config dicts."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

try:
    import sumolib  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment dependent
    sumolib = None
    SUMOLIB_IMPORT_ERROR = exc
else:
    SUMOLIB_IMPORT_ERROR = None


INF = 1e15


def _resolve_net_xml_path(net_path: str) -> str:
    """Resolve SUMO net path from either .sumocfg or .net.xml input."""
    src = Path(net_path)

    if src.suffix == ".sumocfg":
        return str(src.with_suffix(".net.xml"))

    if src.suffix == ".xml" and src.name.endswith(".net.xml"):
        return str(src)

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
