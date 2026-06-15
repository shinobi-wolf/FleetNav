"""KMeans cluster + nearest-neighbour TSP baseline (ignores fuel constraints)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from .base import EVRPPlanner, Route, Stop

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance


def _nearest_neighbour_order(
    depot_idx: int, customer_indices: list[int], dist: np.ndarray
) -> list[int]:
    if len(customer_indices) <= 1:
        return list(customer_indices)
    remaining = list(customer_indices)
    ordered: list[int] = []
    cur = depot_idx
    while remaining:
        nearest = min(remaining, key=lambda j: dist[cur, j])
        ordered.append(nearest)
        remaining.remove(nearest)
        cur = nearest
    return ordered


class MTSPPlanner(EVRPPlanner):
    def plan(self, instance: "Instance") -> list[Route]:
        p = instance.problem

        names     = p.physical_meta["physical_node_names"]
        types     = p.physical_meta["physical_node_types"]
        sumo_ids  = p.physical_meta["physical_sumo_ids"]
        depot_idx = p.physical_meta["depot_index"]
        coords    = p.generation_info["physical_coords"]

        customer_indices = [i for i, t in enumerate(types) if t == "customer"]
        num_vehicles     = int(p.params["m"])

        if not customer_indices:
            raise ValueError("No customer nodes in instance")

        k = min(num_vehicles, len(customer_indices))
        cust_coords = coords[customer_indices]

        if k == 1:
            labels = np.zeros(len(customer_indices), dtype=int)
        else:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(cust_coords)

        clusters: list[list[int]] = [[] for _ in range(k)]
        for local_idx, global_idx in enumerate(customer_indices):
            clusters[labels[local_idx]].append(global_idx)

        routes: list[Route] = []
        vehicle_id = 1

        for cluster in clusters:
            if not cluster:
                continue
            ordered = _nearest_neighbour_order(depot_idx, cluster, p.physical_dist)

            stops: list[Stop] = []
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type=types[depot_idx],
                planned_fuel=float(p.params["Q"]),
                planned_time=0.0,
            ))
            for idx in ordered:
                stops.append(Stop(
                    sumo_node_id=sumo_ids[idx],
                    node_name=names[idx],
                    node_type=types[idx],
                    planned_fuel=0.0,
                    planned_time=0.0,
                ))
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type="depot",
                planned_fuel=0.0,
                planned_time=0.0,
            ))

            routes.append(Route(vehicle_id=vehicle_id, stops=stops))
            vehicle_id += 1

        return routes
