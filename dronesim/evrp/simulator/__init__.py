"""SUMO-based EVRP simulators."""
from .evrp_sim_sequential import EVRPSequentialSimulator
from .evrp_sim_concurrent import EVRPConcurrentSimulator


def get_simulator(config: dict, local_planner=None):
    """Return the simulator instance selected by config['simulation']['simulator']."""
    sim_type = config.get("simulation", {}).get("simulator", "sequential").strip().lower()
    if sim_type == "concurrent":
        return EVRPConcurrentSimulator(config, local_planner=local_planner)
    if sim_type == "sequential":
        return EVRPSequentialSimulator(config, local_planner=local_planner)
    raise ValueError(
        f"Unknown simulation.simulator '{sim_type}'. Valid options: sequential, concurrent"
    )
