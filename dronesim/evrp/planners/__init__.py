"""Planner registry for EVRP."""
from .base import BasePlanner, Route, Stop
from .gvrp_milp import GVRPMILPPlanner
from .mtsp import MTSPPlanner
from .evrp_heuristic import EVRPHeuristicPlanner

REGISTRY: dict[str, type[BasePlanner]] = {
    "milp": GVRPMILPPlanner,
    "mtsp": MTSPPlanner,
    "heuristic": EVRPHeuristicPlanner,
}


def get_planner(name: str) -> BasePlanner:
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown planner '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
