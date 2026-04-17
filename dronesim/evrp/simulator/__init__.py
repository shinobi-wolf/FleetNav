"""SUMO-based EVRP simulators."""
from .evrp_sim_sequential import EVRPSequentialSimulator
from .evrp_sim_concurrent import EVRPConcurrentSimulator


def get_simulator(config: dict):
    """Return the simulator instance selected by config['simulation']['simulator']."""
    sim_type = config.get("simulation", {}).get("simulator", "sequential").strip().lower()
    if sim_type == "concurrent":
        return EVRPConcurrentSimulator(config)
    if sim_type == "sequential":
        return EVRPSequentialSimulator(config)
    raise ValueError(
        f"Unknown simulation.simulator '{sim_type}'. Valid options: sequential, concurrent"
    )
