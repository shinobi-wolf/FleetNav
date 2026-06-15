"""Erdogan & Miller-Hooks dummy-vertex expansion for the G-VRP MILP."""
from __future__ import annotations

from typing import Any

import numpy as np


def expand_with_dummy_vertices(
    physical_dist: np.ndarray,
    physical_meta: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Expand physical nodes with AFS and depot dummy copies."""
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
