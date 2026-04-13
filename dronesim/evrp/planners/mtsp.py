"""MTSP planner: KMeans cluster assignment + nearest-neighbour TSP sequencing.

Ignores fuel constraints — treats the problem as a plain multi-vehicle TSP.
Useful as a fast baseline to compare against the MILP and heuristic EVRP.

Algorithm
---------
1. Cluster customer nodes into `fleet.size` groups with KMeans on their
   SUMO (x, y) coordinates.
2. Sequence each cluster with a nearest-neighbour greedy tour starting
   from the depot.
3. Wrap each tour as depot → [customers] → depot with no AFS visits.
"""
from __future__ import annotations

import sys
import os
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evrp_exact import build_problem_instance_from_config
from planners.base import BasePlanner, Route, Stop


def _nearest_neighbour_order(
    depot_idx: int,
    customer_indices: list[int],
    dist: np.ndarray,
) -> list[int]:
    """Return customer_indices reordered by nearest-neighbour from depot."""
    if len(customer_indices) <= 1:
        return list(customer_indices)

    remaining = list(customer_indices)
    ordered: list[int] = []
    current = depot_idx
    while remaining:
        nearest = min(remaining, key=lambda j: dist[current, j])
        ordered.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return ordered


class MTSPPlanner(BasePlanner):
    """KMeans + nearest-neighbour TSP baseline (no fuel constraints)."""

    def plan(self, config: dict[str, Any]) -> list[Route]:
        physical_dist, params, physical_meta, generation_info = (
            build_problem_instance_from_config(config)
        )

        names: list[str] = physical_meta["physical_node_names"]
        types: list[str] = physical_meta["physical_node_types"]
        sumo_ids: list[str] = physical_meta["physical_sumo_ids"]
        depot_idx: int = physical_meta["depot_index"]
        coords: np.ndarray = generation_info["physical_coords"]

        customer_indices = [i for i, t in enumerate(types) if t == "customer"]
        num_vehicles = int(params["m"])

        if not customer_indices:
            raise ValueError("No customer nodes in instance")

        # ── KMeans cluster assignment ────────────────────────────────────────
        k = min(num_vehicles, len(customer_indices))
        customer_coords = coords[customer_indices]

        if k == 1:
            labels = np.zeros(len(customer_indices), dtype=int)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(customer_coords)

        clusters: list[list[int]] = [[] for _ in range(k)]
        for local_idx, global_idx in enumerate(customer_indices):
            clusters[labels[local_idx]].append(global_idx)

        # ── Nearest-neighbour sequencing per cluster ─────────────────────────
        routes: list[Route] = []
        vehicle_id = 1

        for cluster in clusters:
            if not cluster:
                continue

            ordered = _nearest_neighbour_order(depot_idx, cluster, physical_dist)

            stops: list[Stop] = []
            # depot start
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type=types[depot_idx],
                planned_fuel=float(params["Q"]),
                planned_time=0.0,
            ))
            # customers in NN order
            for idx in ordered:
                stops.append(Stop(
                    sumo_node_id=sumo_ids[idx],
                    node_name=names[idx],
                    node_type=types[idx],
                    planned_fuel=0.0,   # not computed by this planner
                    planned_time=0.0,
                ))
            # depot return
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type="depot",
                planned_fuel=0.0,
                planned_time=0.0,
            ))

            routes.append(Route(vehicle_id=vehicle_id, stops=stops))
            vehicle_id += 1

        self._last_physical_dist = physical_dist
        self._last_physical_meta = physical_meta
        self._last_generation_info = generation_info

        return routes
