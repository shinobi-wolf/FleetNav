"""Shared types, expansion, and IO utilities for evrp_ planners."""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance


# ---------------------------------------------------------------------------
# Route data types
# ---------------------------------------------------------------------------

@dataclass
class Stop:
    sumo_node_id: str
    node_name: str
    node_type: str       # depot | customer | afs | afs_dummy | depot_dummy
    planned_fuel: float  # remaining range in metres at arrival
    planned_time: float  # seconds at arrival
    planned_wait_s: float = 0.0  # td_evrp only: wait from service-start to departure


@dataclass
class Route:
    vehicle_id: int
    stops: list[Stop] = field(default_factory=list)

    @property
    def customer_stops(self) -> list[Stop]:
        return [s for s in self.stops if s.node_type == "customer"]

    @property
    def afs_stops(self) -> list[Stop]:
        return [s for s in self.stops if s.node_type in {"afs", "afs_dummy"}]


# ---------------------------------------------------------------------------
# Abstract planner interface
# ---------------------------------------------------------------------------

class EVRPPlanner(ABC):
    @abstractmethod
    def plan(self, instance: "Instance") -> list[Route]:
        ...


# ---------------------------------------------------------------------------
# Dummy-vertex expansion (Erdogan & Miller-Hooks 2012)
# ---------------------------------------------------------------------------

def expand_with_dummy_vertices(
    physical_dist: np.ndarray,
    physical_meta: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    afs_dummy_copies  = int(params["afs_dummy_copies"])
    depot_dummy_copies = int(params["depot_dummy_copies"])

    physical_names   = list(physical_meta["physical_node_names"])
    physical_types   = list(physical_meta["physical_node_types"])
    physical_sumo_ids = list(physical_meta["physical_sumo_ids"])
    depot             = int(physical_meta["depot_index"])

    afs_indices = [i for i, t in enumerate(physical_types) if t == "afs"]

    node_names    = list(physical_names)
    node_types    = list(physical_types)
    node_sumo_ids = list(physical_sumo_ids)
    parent_index  = list(range(len(physical_names)))

    for afs_idx in afs_indices:
        afs_name = physical_names[afs_idx]
        afs_sid  = physical_sumo_ids[afs_idx]
        for copy_id in range(1, afs_dummy_copies + 1):
            node_names.append(f"{afs_name}_d{copy_id}")
            node_types.append("afs_dummy")
            node_sumo_ids.append(afs_sid)
            parent_index.append(afs_idx)

    depot_name = physical_names[depot]
    depot_sid  = physical_sumo_ids[depot]
    for copy_id in range(1, depot_dummy_copies + 1):
        node_names.append(f"{depot_name}_d{copy_id}")
        node_types.append("depot_dummy")
        node_sumo_ids.append(depot_sid)
        parent_index.append(depot)

    n = len(node_names)
    expanded_dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        pi = parent_index[i]
        for j in range(n):
            pj = parent_index[j]
            expanded_dist[i, j] = physical_dist[pi, pj]

    speed        = float(params["speed"])
    travel_time  = expanded_dist / speed

    service_times = np.zeros(n, dtype=float)
    cust_svc  = float(params["customer_service_time"])
    refuel_svc = float(params["refuel_time"])
    for node in range(n):
        nt = node_types[node]
        if nt == "customer":
            service_times[node] = cust_svc
        elif nt in {"afs", "afs_dummy", "depot_dummy"}:
            service_times[node] = refuel_svc

    all_nodes    = list(range(n))
    customers    = [i for i, t in enumerate(node_types) if t == "customer"]
    afs_nodes    = [i for i, t in enumerate(node_types) if t in {"afs", "afs_dummy"}]
    depot_dummies = [i for i, t in enumerate(node_types) if t == "depot_dummy"]
    refuel_nodes = [depot] + afs_nodes + depot_dummies
    f_prime      = afs_nodes + depot_dummies
    non_depot    = [i for i in all_nodes if i != depot]
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


# ---------------------------------------------------------------------------
# Save instance JSON
# ---------------------------------------------------------------------------

def save_instance_json(instance: "Instance", out_dir: Path) -> None:
    out_dir = Path(out_dir)
    problem = instance.problem
    gi = problem.generation_info
    payload = {
        "distance_unit": "meters",
        "physical_node_names": problem.physical_meta["physical_node_names"],
        "physical_node_types": problem.physical_meta["physical_node_types"],
        "physical_sumo_ids":   problem.physical_meta["physical_sumo_ids"],
        "distance_matrix":     problem.physical_dist.tolist(),
        "generation_info": {
            "seed":              gi["seed"],
            "net_xml_path":      gi["net_xml_path"],
            "depot_sumo_id":     gi["depot_sumo_id"],
            "customer_sumo_ids": gi["customer_sumo_ids"],
            "afs_sumo_ids":      gi["afs_sumo_ids"],
        },
    }
    out_path = out_dir / "generated_instance.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved generated instance to {out_path}")


# ---------------------------------------------------------------------------
# Route plot
# ---------------------------------------------------------------------------

def plot_routes(
    routes: list[Route],
    instance: "Instance",
    out_path: str | None = None,
) -> None:
    if not routes:
        print("No routes to plot.")
        return

    problem     = instance.problem
    sumo_ids    = problem.physical_meta["physical_sumo_ids"]
    coords      = problem.generation_info["physical_coords"]  # (n_physical, 2)

    id_to_coord = {sid: coords[i] for i, sid in enumerate(sumo_ids)}

    # Assign a small positional jitter per dummy copy of the same parent
    _dummy_count: dict[str, int] = defaultdict(int)

    def _coord(stop: Stop) -> tuple[float, float]:
        base = id_to_coord.get(stop.sumo_node_id, (0.0, 0.0))
        if stop.node_type in {"afs_dummy", "depot_dummy"}:
            _dummy_count[stop.sumo_node_id] += 1
            k = _dummy_count[stop.sumo_node_id]
            angle  = (2.0 * math.pi * k) / 6.0
            radius = 5.0 + 1.5 * (k - 1)
            return (
                float(base[0]) + radius * math.cos(angle),
                float(base[1]) + radius * math.sin(angle),
            )
        return (float(base[0]), float(base[1]))

    # Gather node scatter points by type
    type_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for route in routes:
        for stop in route.stops:
            c = _coord(stop)
            type_points[stop.node_type].append(c)
    # reset counter for route drawing
    _dummy_count.clear()

    class_styles = {
        "depot":       {"marker": "s", "color": "black",     "label": "Depot",       "size": 130},
        "depot_dummy": {"marker": "s", "color": "0.45",      "label": "Depot dummy", "size": 85},
        "customer":    {"marker": "o", "color": "tab:blue",  "label": "Customer",    "size": 95},
        "afs":         {"marker": "^", "color": "tab:green", "label": "AFS",         "size": 130},
        "afs_dummy":   {"marker": "^", "color": "tab:olive", "label": "AFS dummy",   "size": 85},
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for node_class, style in class_styles.items():
        pts = type_points.get(node_class, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(
            xs, ys,
            marker=style["marker"],
            s=style["size"],
            c=style["color"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    palette = plt.colormaps.get_cmap("tab10")
    for ridx, route in enumerate(routes):
        stop_coords = [_coord(s) for s in route.stops]
        xs = [c[0] for c in stop_coords]
        ys = [c[1] for c in stop_coords]
        ax.plot(
            xs, ys,
            color=palette(ridx % 10),
            linewidth=2.4,
            alpha=0.95,
            label=f"Vehicle {route.vehicle_id}",
            zorder=2,
        )
    _dummy_count.clear()

    # Label physical nodes
    for sid, coord in id_to_coord.items():
        ax.text(
            coord[0] + 1.2, coord[1] + 1.2,
            sid[:6],
            fontsize=7,
            color="0.4",
        )

    ax.set_title("G-VRP solution routes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.28)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        out_file = out_path or "gvrp_routes.png"
        plt.savefig(out_file, dpi=200)
        print(f"Plot saved to {out_file} (non-GUI backend detected).")
        plt.close(fig)
    else:
        plt.show()
