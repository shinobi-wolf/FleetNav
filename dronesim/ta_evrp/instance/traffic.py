"""PSL traffic model for the TA-EVRP Instance block.

Native implementation — does not import from dronesim.evrp.
Reads ta_config directly (human-friendly units) and converts km/h → m/s internally.
"""
from __future__ import annotations

import json
import math
import random as _random
from typing import Any


class PSLTrafficModel:
    """Per-edge piecewise-constant speed limit model.

    Each edge has an independent speed for each of K time slots.
    Slot index: j = floor(t / delta_t) % K.
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
        self.enabled      = enabled
        self.free_flow_ms = free_flow_ms
        self.K            = K
        self.delta_t      = delta_t
        self.v_min        = v_min
        self.v_max        = v_max
        self.profiles: dict[str, list[float]] = profiles or {}

    def get_slot(self, t: float) -> int:
        if self.K == 1:
            return 0
        return int(math.floor(t / self.delta_t)) % self.K

    def get_speed(self, edge_id: str, slot: int) -> float:
        if not self.enabled:
            return self.free_flow_ms
        return self.profiles[edge_id][slot]


class Traffic:
    """Builds a PSLTrafficModel from raw ta_config.

    Reads traffic.psl.* keys (in km/h) and converts to m/s internally.
    Requires the network's edge_ids to generate per-edge speed profiles.
    """

    def __init__(self, ta_config: dict[str, Any], edge_ids: list[str]) -> None:
        psl = ta_config.get("traffic", {}).get("psl", {})
        ugv = ta_config.get("simulation", {}).get("ugv", {})

        free_flow_ms = float(ugv.get("free_flow_speed_kmh", 50.0)) / 3.6
        max_time_s   = float(ugv.get("max_duration_h", 6.0)) * 3600.0
        enabled      = bool(psl.get("enabled", False))

        if not enabled:
            self.model = PSLTrafficModel(enabled=False, free_flow_ms=free_flow_ms)
            return

        K = int(psl.get("K", 1))
        if K < 1:
            raise ValueError("traffic.psl.K must be >= 1")

        slot_dur = psl.get("slot_duration")
        delta_t  = float(slot_dur) if slot_dur is not None else max_time_s / K

        v_min_ms = float(psl.get("v_min_kmh", 40.0))  / 3.6
        v_max_ms = float(psl.get("v_max_kmh", 100.0)) / 3.6
        step_ms  = float(psl.get("speed_step_kmh", 60.0)) / 3.6
        if step_ms <= 0:
            raise ValueError("traffic.psl.speed_step_kmh must be > 0")

        seed         = int(psl.get("seed", 42))
        profile_type = str(psl.get("profile_type", "random"))

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
            fn = psl["speed_profile_fn"]
            profiles = {
                eid: [float(max(v_min_ms, min(v_max_ms, fn(eid, j)))) for j in range(K)]
                for eid in edge_ids
            }
        elif profile_type == "file":
            path = psl["speed_profile_path"]
            with open(path) as f:
                if path.endswith((".yaml", ".yml")):
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            profiles = {
                eid: [float(max(v_min_ms, min(v_max_ms, s)))
                      for s in data.get(eid, [v_max_ms] * K)]
                for eid in edge_ids
            }
        else:
            raise ValueError(f"Unknown traffic.psl.profile_type: '{profile_type}'")

        self.model = PSLTrafficModel(
            enabled=True,
            free_flow_ms=free_flow_ms,
            K=K,
            delta_t=delta_t,
            v_min=v_min_ms,
            v_max=v_max_ms,
            profiles=profiles,
        )
