"""Planner registry for EVRP."""
from ..routes.base import Route, Stop
from .base import BasePlanner
from .gvrp_milp import GVRPMILPPlanner
from .mtsp import MTSPPlanner
from .evrp_heuristic import EVRPHeuristicPlanner
from .evrp_ctvtt_ga import EVRPCTVTTGAPlanner

REGISTRY: dict[str, type[BasePlanner]] = {
    "milp": GVRPMILPPlanner,
    "mtsp": MTSPPlanner,
    "heuristic": EVRPHeuristicPlanner,
    "ctvtt_ga": EVRPCTVTTGAPlanner,
}


def get_planner(name: str) -> BasePlanner:
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown planner '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
