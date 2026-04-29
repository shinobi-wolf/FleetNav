"""Piecewise Speed Limit (PSL) traffic model for TA-EVRP.

Each edge gets an independent random speed profile over K time slots.
Mirrors dronesim/trsp/traffic.py but operates at the EVRP simulation level.
"""
from __future__ import annotations

import json
import math
import random as _random
from typing import Any


def _kmh_to_ms(kmh: float) -> float:
    return float(kmh) / 3.6


def _mph_to_ms(mph: float) -> float:
    return float(mph) * 0.44704


class PSLTrafficModel:
    """Per-edge piecewise-constant speed limit model.

    Parameters
    ----------
    enabled : bool
        When False, get_speed() always returns free_flow_ms.
    free_flow_ms : float
        Fallback speed (m/s) used when disabled.
    K : int
        Number of time slots.
    delta_t : float
        Duration of each slot in seconds (= T / K).
    v_min, v_max : float
        Speed range in m/s (for normalization; profiles are already clipped).
    profiles : dict[edge_id, list[float]]
        Per-edge speed in m/s for each of the K slots.
    """

    def __init__(
        self,
        enabled: bool,
        free_flow_ms: float,
        K: int = 1,
        delta_t: float = 86400.0,
        v_min: float = 0.0,
        v_max: float = 0.0,
        profiles: dict[str, list[float]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.free_flow_ms = free_flow_ms
        self.K = K
        self.delta_t = delta_t
        self.v_min = v_min
        self.v_max = v_max
        self.profiles: dict[str, list[float]] = profiles or {}

    def get_slot(self, t: float) -> int:
        """Active time slot index for simulation time t (seconds)."""
        if self.K == 1:
            return 0
        return min(self.K - 1, int(math.floor(t / self.delta_t)))

    def get_speed(self, edge_id: str, slot: int) -> float:
        """Speed in m/s for edge_id at the given slot."""
        if not self.enabled:
            return self.free_flow_ms
        return self.profiles[edge_id][slot]


def build_traffic_model(
    config: dict[str, Any],
    free_flow_ms: float,
    max_time_s: float,
    edge_ids: list[str],
) -> PSLTrafficModel:
    """Build a PSLTrafficModel from the internal config dict.

    Args:
        config:       internal config dict (key "traffic" holds PSL params)
        free_flow_ms: fallback speed when traffic is disabled
        max_time_s:   simulation horizon; used as T (total profile duration)
        edge_ids:     non-internal SUMO edge IDs to generate profiles for
    """
    tcfg = config.get("traffic", {})
    enabled = bool(tcfg.get("enabled", False))

    if not enabled:
        return PSLTrafficModel(enabled=False, free_flow_ms=free_flow_ms)

    K = int(tcfg.get("K", 1))
    if K < 1:
        raise ValueError("traffic.K must be >= 1")
    delta_t = max_time_s / K

    v_min_ms = _kmh_to_ms(float(tcfg.get("v_min", 40.0)))
    v_max_ms = _kmh_to_ms(float(tcfg.get("v_max", 100.0)))
    step_ms = _kmh_to_ms(float(tcfg.get("speed_step", 10.0)))
    if step_ms <= 0:
        raise ValueError("traffic.speed_step must be > 0")

    seed = int(tcfg.get("seed", 42))
    # support both profile_type (new TA-EVRP key) and speed_profile_type (legacy)
    profile_type = str(tcfg.get("profile_type", tcfg.get("speed_profile_type", "random")))

    # Build discrete speed levels
    levels: list[float] = []
    v = v_min_ms
    while v <= v_max_ms + 1e-9:
        levels.append(round(v, 8))
        v += step_ms
    if not levels:
        levels = [v_max_ms]

    rng = _random.Random(seed)

    if profile_type == "random":
        profiles = {
            eid: [rng.choice(levels) for _ in range(K)]
            for eid in sorted(edge_ids)
        }
    elif profile_type == "function":
        fn = tcfg["speed_profile_fn"]
        profiles = {
            eid: [float(max(v_min_ms, min(v_max_ms, fn(eid, j) / 3.6)))
                  for j in range(K)]
            for eid in edge_ids
        }
    elif profile_type == "file":
        path = tcfg["speed_profile_path"]
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        profiles = {
            eid: [float(max(v_min_ms, min(v_max_ms, s / 3.6)))
                  for s in data.get(eid, [v_max_ms * 3.6] * K)]
            for eid in edge_ids
        }
    else:
        raise ValueError(f"Unknown traffic profile_type: '{profile_type}'")

    return PSLTrafficModel(
        enabled=True,
        free_flow_ms=free_flow_ms,
        K=K,
        delta_t=delta_t,
        v_min=v_min_ms,
        v_max=v_max_ms,
        profiles=profiles,
    )
