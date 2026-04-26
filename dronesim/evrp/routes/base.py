"""Shared route data types for planners and simulator."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stop:
    """A single stop on a vehicle route."""
    sumo_node_id: str
    node_name: str          # logical name: D, C1..Cn, F1..Fm
    node_type: str          # depot | customer | afs | afs_dummy | depot_dummy
    planned_fuel: float     # gallons at arrival (from planner)
    planned_time: float     # hours at arrival (from planner)
    planned_wait_s: float = 0.0  # seconds from service-start to departure (td_evrp only)


@dataclass
class Route:
    """Planned route for one vehicle."""
    vehicle_id: int
    stops: list[Stop] = field(default_factory=list)

    @property
    def customer_stops(self) -> list[Stop]:
        return [s for s in self.stops if s.node_type == "customer"]

    @property
    def afs_stops(self) -> list[Stop]:
        return [s for s in self.stops if s.node_type in {"afs", "afs_dummy"}]
