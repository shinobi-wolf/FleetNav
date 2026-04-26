"""Piecewise-constant speed traffic model (moved from simulator/evrp_sim.py)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseTrafficModel


def _kmh_to_ms(kmh: float) -> float:
    return kmh / 3.6


def _mph_to_ms(mph: float) -> float:
    return mph * 0.44704


@dataclass
class SegmentedSpeedModel(BaseTrafficModel):
    enabled: bool
    free_flow_ms: float          # used when disabled
    speed_segments: np.ndarray   # m/s per segment (length = num_segments)
    seg_duration_s: float        # seconds per segment
    current_segment: int = -1
    current_speed_ms: float = 0.0

    def update(self, sim_time_s: float) -> bool:
        """Return True if speed changed."""
        if not self.enabled:
            return False
        seg = int(sim_time_s / self.seg_duration_s)
        seg = min(seg, len(self.speed_segments) - 1)
        if seg != self.current_segment:
            self.current_segment = seg
            self.current_speed_ms = float(self.speed_segments[seg])
            return True
        return False

    @property
    def commanded_speed_ms(self) -> float:
        if not self.enabled:
            return self.free_flow_ms
        if self.current_segment < 0:
            return float(self.speed_segments[0]) if len(self.speed_segments) else self.free_flow_ms
        return self.current_speed_ms


def build_traffic_model(
    config: dict[str, Any],
    free_flow_ms: float,
    max_time_s: float,
) -> BaseTrafficModel:
    from .lu import build_lu_traffic_model  # avoid circular import at module level

    tcfg = config.get("traffic", {})
    enabled = bool(tcfg.get("enabled", True))
    model_name = str(tcfg.get("model", "segmented")).strip().lower()

    if enabled and model_name == "lu":
        return build_lu_traffic_model(config)

    if not enabled:
        return SegmentedSpeedModel(
            enabled=False,
            free_flow_ms=free_flow_ms,
            speed_segments=np.array([free_flow_ms]),
            seg_duration_s=1e9,
        )

    speed_limits_kmh = tcfg.get("speed_limits", [20, 50])
    speed_step_kmh = float(tcfg.get("speed_step", 5))
    seg_duration_min = float(tcfg.get("seg_duration", 5))
    seed = int(tcfg.get("seed", 42))

    lo_ms = _kmh_to_ms(float(speed_limits_kmh[0]))
    hi_ms = _kmh_to_ms(float(speed_limits_kmh[1]))
    step_ms = _kmh_to_ms(speed_step_kmh)
    seg_duration_s = seg_duration_min * 60.0

    speed_range = np.arange(lo_ms, hi_ms + 1e-9, step_ms, dtype=np.float32)
    num_segs = max(1, int(math.ceil(max_time_s / seg_duration_s)))
    rng = np.random.default_rng(seed)
    segments = rng.choice(speed_range, size=num_segs)

    return SegmentedSpeedModel(
        enabled=True,
        free_flow_ms=free_flow_ms,
        speed_segments=segments,
        seg_duration_s=seg_duration_s,
    )
