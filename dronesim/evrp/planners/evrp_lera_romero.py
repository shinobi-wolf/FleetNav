"""TD-EVRP planner — Lera-Romero et al. (2024) exact formulation."""
from __future__ import annotations

from typing import Any

from ..instance.builder import build_problem_instance_from_config
from ..milp.td_evrp import build_td_evrp_model, solve_and_extract_td
from ..routes.base import Route, Stop
from .base import BasePlanner


class EVRPLeraRomeroPlanner(BasePlanner):
    """Solves TD-EVRP exactly with Gurobi (Lera-Romero et al. 2024).

    Traffic is embedded directly in the MILP via period-indicator binary
    variables and Ichoua-precharacterized travel times. Requires Gurobi license.
    """

    def plan(self, config: dict[str, Any]) -> list[Route]:
        physical_dist, params, physical_meta, generation_info = (
            build_problem_instance_from_config(config)
        )
        params["_physical_meta"] = physical_meta

        model = build_td_evrp_model(physical_dist, params, physical_meta, config)
        result = solve_and_extract_td(model, params)

        if not result["has_solution"]:
            raise RuntimeError(
                f"TD-EVRP planner found no feasible solution (status={result['status']})"
            )

        inst = result["instance"]
        node_names: list[str] = inst["node_names"]
        node_types: list[str] = inst["node_types"]
        node_sumo_ids: list[str] = inst["node_sumo_ids"]

        routes: list[Route] = []
        for raw in result["routes"]:
            stops: list[Stop] = []
            for stop_data in raw["stops"]:
                idx = stop_data["node_index"]
                stops.append(
                    Stop(
                        sumo_node_id=node_sumo_ids[idx],
                        node_name=node_names[idx],
                        node_type=node_types[idx],
                        planned_fuel=stop_data["fuel"],
                        planned_time=stop_data["arrival_time"],
                    )
                )
            routes.append(Route(vehicle_id=raw["vehicle"], stops=stops))

        self._last_generation_info = generation_info
        self._last_physical_dist = physical_dist
        self._last_physical_meta = physical_meta
        self._last_instance = inst
        self._last_raw_routes = result["routes"]

        return routes
