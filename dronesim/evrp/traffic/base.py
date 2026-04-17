"""Abstract base class for traffic/speed models."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrafficModel(ABC):

    @abstractmethod
    def update(self, sim_time_s: float) -> bool:
        """Advance to sim_time_s; return True if commanded speed changed."""
        ...

    @property
    @abstractmethod
    def commanded_speed_ms(self) -> float:
        """Current commanded vehicle speed in m/s."""
        ...

    def travel_time(self, dist_m: float) -> float:
        """Estimated travel time in seconds for dist_m metres."""
        speed = self.commanded_speed_ms
        if speed <= 0:
            return float("inf")
        return dist_m / speed
