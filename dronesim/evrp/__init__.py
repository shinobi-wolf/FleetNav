from .base import EVRPPlanner, Route, Stop
from .milp import MILPPlanner
from .td_milp import TDMILPPlanner
from .heuristic import HeuristicPlanner
from .mtsp import MTSPPlanner

REGISTRY: dict[str, type[EVRPPlanner]] = {
    "milp":      MILPPlanner,
    "td_milp":   TDMILPPlanner,
    "heuristic": HeuristicPlanner,
    "mtsp":      MTSPPlanner,
}

__all__ = ["EVRPPlanner", "Route", "Stop", "REGISTRY",
           "MILPPlanner", "TDMILPPlanner", "HeuristicPlanner", "MTSPPlanner"]
