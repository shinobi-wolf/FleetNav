"""Three-segment time-dependent speed model — Lu et al. (2020), Figure 1.

Speed profile over the planning horizon [0, u0]:
  [0,  T1)   morning congestion  → vc  (low speed)
  [T1, T2)   free-flow period    → vf  (high speed)
  [T2, u0]   evening congestion  → vc  (low speed)

Parameters are read from the `traffic_lu` config section, which is shared
with the td_evrp MILP so simulation and planner use identical boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseTrafficModel


def _kmh_to_ms(kmh: float) -> float:
    return float(kmh) / 3.6


@dataclass
class LuTrafficModel(BaseTrafficModel):
    """Piecewise-constant 3-segment speed model (Lu et al. 2020, Fig. 1)."""
    vc_ms: float  # congestion speed in m/s
    vf_ms: float  # free-flow speed in m/s
    T1_s: float   # end of morning congestion in seconds
    T2_s: float   # end of free-flow period in seconds
    _speed: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._speed = self.vc_ms  # t=0 starts in morning congestion

    def update(self, sim_time_s: float) -> bool:
        """Advance model to sim_time_s; return True if speed changed."""
        new = self.vf_ms if self.T1_s <= sim_time_s < self.T2_s else self.vc_ms
        if new != self._speed:
            self._speed = new
            return True
        return False

    @property
    def commanded_speed_ms(self) -> float:
        return self._speed


def build_lu_traffic_model(config: dict[str, Any]) -> LuTrafficModel:
    """Build a LuTrafficModel from the `traffic_lu` config section."""
    cfg = config.get("traffic_lu", {})
    vc_ms = _kmh_to_ms(float(cfg.get("v_congestion_kmh", 15.0)))
    vf_ms = _kmh_to_ms(float(cfg.get("v_freeflow_kmh", 50.0)))
    T1_s = float(cfg.get("T1_hours", 1.0)) * 3600.0
    T2_s = float(cfg.get("T2_hours", 3.0)) * 3600.0
    return LuTrafficModel(vc_ms=vc_ms, vf_ms=vf_ms, T1_s=T1_s, T2_s=T2_s)
