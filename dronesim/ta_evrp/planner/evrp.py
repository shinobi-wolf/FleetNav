from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dronesim.evrp import REGISTRY, Route
from dronesim.evrp.base import plot_routes, save_instance_json

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance


class GlobalPlanner:
    def __init__(self) -> None:
        self._instance: "Instance | None" = None

    def plan(self, instance: "Instance") -> list[Route]:
        self._instance = instance
        planner_name = instance.planner_name
        if planner_name not in REGISTRY:
            raise ValueError(
                f"Unknown EVRP planner '{planner_name}'. "
                f"Available: {sorted(REGISTRY)}"
            )
        return REGISTRY[planner_name]().plan(instance)

    def save_instance_json(self, out_dir: Path | str) -> None:
        if self._instance is None:
            raise RuntimeError("Call plan() before save_instance_json()")
        save_instance_json(self._instance, Path(out_dir))

    def plot(self, routes: list[Route], out_path: str | None = None) -> None:
        if self._instance is None:
            raise RuntimeError("Call plan() before plot()")
        plot_routes(routes, self._instance, out_path)