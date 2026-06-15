"""Problem instance sampler for the TA-EVRP Instance block.

Native implementation — does not import from dronesim.evrp.
Samples depot, customers, and AFS from the road network and builds the
all-pairs distance matrix needed by the EVRP planners.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .network import Network

INF = 1e15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_depot(
    graph: nx.DiGraph,
    warehouse_choice: str,
    rng: np.random.Generator,
) -> str:
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Cannot resolve depot: graph has no nodes")

    choice = str(warehouse_choice).strip()

    if choice.lower() == "center":
        xs = np.array([graph.nodes[n]["x"] for n in nodes], dtype=float)
        ys = np.array([graph.nodes[n]["y"] for n in nodes], dtype=float)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        return str(min(nodes, key=lambda n: (graph.nodes[n]["x"] - cx) ** 2 + (graph.nodes[n]["y"] - cy) ** 2))

    if choice.lower() == "random":
        return str(rng.choice(np.array(nodes, dtype=object)))

    if choice in graph.nodes:
        return choice

    raise ValueError(f"Configured depot node '{warehouse_choice}' does not exist in graph")


def _determine_customer_count(ta_config: dict[str, Any], candidate_count: int) -> int:
    tasks = ta_config.get("simulation", {}).get("tasks", {})
    mode  = str(tasks.get("customer_sampling", "density")).lower()

    if mode == "count":
        count = int(tasks.get("customer_count", 0))
        if count <= 0:
            raise ValueError("customer_count must be > 0 when customer_sampling is 'count'")
    else:
        density = float(tasks.get("task_density", 30.0))
        if density <= 0:
            raise ValueError("task_density must be > 0")
        count = max(1, int((density * candidate_count) // 100))

    if count > candidate_count:
        raise ValueError(
            f"Requested {count} customers but only {candidate_count} eligible nodes available"
        )
    return count


def _extract_pair_distance(
    src: str,
    dst: str,
    shortest_paths: dict[str, dict[str, float]],
) -> float:
    forward  = float(shortest_paths.get(src, {}).get(dst, INF))
    backward = float(shortest_paths.get(dst, {}).get(src, INF))
    fwd_ok   = np.isfinite(forward)  and forward  < INF
    bwd_ok   = np.isfinite(backward) and backward < INF
    if not fwd_ok and not bwd_ok:
        raise ValueError(f"No path between SUMO nodes '{src}' and '{dst}'")
    if fwd_ok and bwd_ok:
        return 0.5 * (forward + backward)
    return forward if fwd_ok else backward


def _build_distance_matrix(
    selected_ids: list[str],
    shortest_paths: dict[str, dict[str, float]],
) -> np.ndarray:
    n = len(selected_ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _extract_pair_distance(selected_ids[i], selected_ids[j], shortest_paths)
            dist[i, j] = d
            dist[j, i] = d
    return dist


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class Problem:
    """Sampled G-VRP problem instance.

    Attributes
    ----------
    physical_dist   : (n × n) numpy array of inter-node distances in metres (symmetric).
    params          : fleet / solver parameters in SI units (Q in m, Tmax in s, etc.).
    physical_meta   : node names, types, SUMO IDs, and depot index.
    generation_info : seed, net path, and sampled node IDs for reproducibility.
    """

    def __init__(self, ta_config: dict[str, Any], network: "Network") -> None:
        graph          = network.graph
        node_positions = network.node_positions
        net_xml_path   = network.net_xml_path

        net_cfg  = ta_config.get("network", {})
        ugv_cfg  = ta_config.get("simulation", {}).get("ugv", {})
        tasks    = ta_config.get("simulation", {}).get("tasks", {})
        milp_cfg = ta_config.get("evrp", {}).get("milp", {})

        seed = int(ta_config.get("generation_seed", 42))
        rng  = np.random.default_rng(seed)

        # ── Sampling ─────────────────────────────────────────────────────────
        depot_choice = str(net_cfg.get("warehouse", "center"))
        depot_id     = _resolve_depot(graph, depot_choice, rng)

        candidates      = [n for n in graph.nodes if n != depot_id]
        customer_count  = _determine_customer_count(ta_config, len(candidates))
        afs_count       = int(tasks.get("afs_count", 3))

        if afs_count <= 0:
            raise ValueError("simulation.tasks.afs_count must be > 0")
        if customer_count + afs_count > len(candidates):
            raise ValueError(
                f"customers ({customer_count}) + AFS ({afs_count}) > available nodes ({len(candidates)})"
            )

        customer_nodes = [
            str(n) for n in
            rng.choice(np.array(candidates, dtype=object), size=customer_count, replace=False).tolist()
        ]
        remaining = [n for n in candidates if n not in customer_nodes]
        afs_nodes  = [
            str(n) for n in
            rng.choice(np.array(remaining, dtype=object), size=afs_count, replace=False).tolist()
        ]

        # ── Distance matrix ───────────────────────────────────────────────────
        selected     = [depot_id] + customer_nodes + afs_nodes
        shortest     = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
        dist_matrix  = _build_distance_matrix(selected, shortest)

        node_names   = ["D"] + [f"C{i+1}" for i in range(customer_count)] + [f"F{i+1}" for i in range(afs_count)]
        node_types   = ["depot"] + ["customer"] * customer_count + ["afs"] * afs_count
        coords       = np.array([node_positions[n] for n in selected], dtype=float)

        # ── Unit conversions (all → SI) ───────────────────────────────────────
        battery_L      = float(ugv_cfg.get("battery_capacity_L", 50.0))
        consume_L_km   = float(ugv_cfg.get("consumption_rate_L_per_km", 5.0))
        speed_kmh      = float(ugv_cfg.get("free_flow_speed_kmh", 50.0))
        max_dur_h      = float(ugv_cfg.get("max_duration_h", 6.0))
        svc_time_h     = float(ugv_cfg.get("customer_service_time_h", 0.0))
        recharge_h     = float(ugv_cfg.get("recharge_time_h", 0.0))
        fleet_size     = int(ugv_cfg.get("fleet_size", 3))

        # ── Store all outputs ─────────────────────────────────────────────────
        self.physical_dist: np.ndarray = dist_matrix

        self.params: dict[str, Any] = {
            "Q":                    battery_L * 1000.0 / consume_L_km,  # m (range)
            "r":                    1.0,                                 # dimensionless
            "speed":                speed_kmh / 3.6,                    # m/s
            "Tmax":                 max_dur_h * 3600.0,                 # s
            "m":                    fleet_size,
            "afs_dummy_copies":     int(milp_cfg.get("afs_dummy_copies", 3)),
            "depot_dummy_copies":   int(milp_cfg.get("depot_dummy_copies", 3)),
            "customer_service_time": svc_time_h * 3600.0,              # s
            "refuel_time":          recharge_h * 3600.0,               # s
            "time_limit":           float(milp_cfg.get("time_limit_s", 300)),
            "output_flag":          int(milp_cfg.get("output_flag", 1)),
            "mip_gap":              float(milp_cfg.get("mip_gap", 0.01)),
        }

        self.physical_meta: dict[str, Any] = {
            "physical_node_names": node_names,
            "physical_node_types": node_types,
            "physical_sumo_ids":   selected,
            "depot_index":         0,
        }

        self.generation_info: dict[str, Any] = {
            "seed":             seed,
            "net_xml_path":     net_xml_path,
            "depot_sumo_id":    depot_id,
            "customer_sumo_ids": customer_nodes,
            "afs_sumo_ids":     afs_nodes,
            "physical_coords":  coords,
        }

    # ── Clean property accessors ──────────────────────────────────────────────

    @property
    def depot_id(self) -> str:
        return self.generation_info["depot_sumo_id"]

    @property
    def customer_ids(self) -> list[str]:
        return self.generation_info["customer_sumo_ids"]

    @property
    def afs_ids(self) -> list[str]:
        return self.generation_info["afs_sumo_ids"]

    @property
    def node_names(self) -> list[str]:
        return self.physical_meta["physical_node_names"]

    @property
    def node_types(self) -> list[str]:
        return self.physical_meta["physical_node_types"]

    @property
    def coords(self) -> np.ndarray:
        return self.generation_info["physical_coords"]
