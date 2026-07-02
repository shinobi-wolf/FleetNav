"""Shared pedestrian flow–speed relationship for delivery-robot (ped) mode.

Kept traci-free and pure so the training env (`SumoTRSPEnv`), the DSP-PED baseline, and
the eval pipeline all compute the robot's density-coupled walking speed identically —
callers do their own TraCI queries and pass the resulting person count and lane area here.
"""
import math


def weidmann_speed(n_persons: int, area: float, v_free: float,
                   gamma: float, density_jam: float, v_min: float) -> float:
    """Robot walking speed (m/s) on a sidewalk given pedestrian count and lane area.

    Weidmann: v = v_free * (1 - exp(-gamma * (1/rho - 1/rho_jam))), clamped to
    [v_min, v_free], with areal density rho = n_persons / area [ped/m²]. Returns free-flow
    when the sidewalk is empty or the area is invalid, and the v_min floor at/above jam
    density (avoids a hard stall).
    """
    if n_persons <= 0 or area <= 0:
        return v_free
    rho = n_persons / area
    if rho >= density_jam:
        return v_min
    f = 1.0 - math.exp(-gamma * (1.0 / rho - 1.0 / density_jam))
    return max(v_free * min(max(f, 0.0), 1.0), v_min)
