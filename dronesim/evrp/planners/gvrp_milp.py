"""MILP-based G-VRP planner wrapping evrp_exact.py."""
from __future__ import annotations

from typing import Any

from ..evrp_exact import (
    build_problem_instance_from_config,
    build_gvrp_model,
    solve_and_extract,
)
from .base import BasePlanner, Route, Stop


class GVRPMILPPlanner(BasePlanner):
    """Solves G-VRP exactly with Gurobi and returns Route objects."""

    def plan(self, config: dict[str, Any]) -> list[Route]:
        physical_dist, params, physical_meta, generation_info = (
            build_problem_instance_from_config(config)
        )
        params["_physical_meta"] = physical_meta
        params["plot"] = False  # plotting is the simulator's job

        model = build_gvrp_model(physical_dist, params, physical_meta)
        result = solve_and_extract(model, physical_dist, params)

        if not result["has_solution"]:
            raise RuntimeError(
                f"MILP planner found no feasible solution (status={result['status']})"
            )

        inst = result["instance"]
        node_names: list[str] = inst["node_names"]
        node_types: list[str] = inst["node_types"]
        node_sumo_ids: list[str] = inst["node_sumo_ids"]

        routes: list[Route] = []
        for raw in result["routes"]:
            stops: list[Stop] = []
            for s in raw["stops"]:
                idx = s["node_index"]
                stops.append(
                    Stop(
                        sumo_node_id=node_sumo_ids[idx],
                        node_name=node_names[idx],
                        node_type=node_types[idx],
                        planned_fuel=s["fuel"],
                        planned_time=s["arrival_time"],
                    )
                )
            routes.append(Route(vehicle_id=raw["vehicle"], stops=stops))

        # Store artefacts for downstream use (plotting, debugging, instance export)
        self._last_generation_info = generation_info
        self._last_physical_dist = physical_dist
        self._last_physical_meta = physical_meta
        self._last_instance = inst
        self._last_raw_routes = result["routes"]  # list of dicts from solve_and_extract

        return routes
