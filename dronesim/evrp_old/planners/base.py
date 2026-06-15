"""Abstract base class for EVRP planners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from ..routes.base import Route

if TYPE_CHECKING:
    from ..network import NetworkContext


class BasePlanner(ABC):
    """Interface every planner must implement."""

    @abstractmethod
    def plan(
        self,
        config: dict[str, Any],
        network: Optional["NetworkContext"] = None,
    ) -> list[Route]:
        """
        Given a config dict, return a list of Route objects.
        Each route starts and ends at the depot.

        If a pre-built NetworkContext is supplied it is forwarded to the
        instance builder so the SUMO network is not reloaded from disk.
        """
        ...
