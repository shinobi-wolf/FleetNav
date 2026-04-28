"""Piecewise-constant speed traffic model (moved from simulator/evrp_sim.py)."""
from __future__ import annotations

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

    # PSL-style keys: T (seconds), K (slots), v_min/v_max (km/h), speed_step (km/h)
    T = float(tcfg.get("T", 86400.0))
    K = int(tcfg.get("K", 288))
    v_min_kmh = float(tcfg.get("v_min", 20.0))
    v_max_kmh = float(tcfg.get("v_max", 50.0))
    speed_step_kmh = float(tcfg.get("speed_step", 10.0))
    seed = int(tcfg.get("seed", 42))

    lo_ms = _kmh_to_ms(v_min_kmh)
    hi_ms = _kmh_to_ms(v_max_kmh)
    step_ms = _kmh_to_ms(speed_step_kmh)
    delta_t = T / max(K, 1)

    speed_range = np.arange(lo_ms, hi_ms + 1e-9, step_ms, dtype=np.float32)
    rng = np.random.default_rng(seed)
    segments = rng.choice(speed_range, size=K)

    return SegmentedSpeedModel(
        enabled=True,
        free_flow_ms=free_flow_ms,
        speed_segments=segments,
        seg_duration_s=delta_t,
    )
