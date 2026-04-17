"""Abstract base class for EVRP planners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..routes.base import Route


class BasePlanner(ABC):
    """Interface every planner must implement."""

    @abstractmethod
    def plan(self, config: dict[str, Any]) -> list[Route]:
        """
        Given a config dict, return a list of Route objects.
        Each route starts and ends at the depot.
        """
        ...
