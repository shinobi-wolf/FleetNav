"""TA-EVRP integration layer: wires EVRP outer planner with TRSP inner agent."""
from .local_planner import TRSPLocalPlanner

__all__ = ["TRSPLocalPlanner"]
