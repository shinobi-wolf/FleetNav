"""Traffic models for the TA-EVRP Instance block.

Native implementation — does not import from dronesim.evrp.
Reads ta_config directly (human-friendly units) and converts km/h → m/s internally.

Supported models (traffic.model):
  psl  — Piecewise Speed Limit: per-edge independent speed profiles over K time slots.
  td   — Time-Dependent congestion (Lu et al. 2020, Fig. 1): three periods with a
          global congestion speed v_c in the first and last slot and a global
          free-flow speed v_f in the middle slot. Slot boundaries at T1, T2.
  svc  — Stochastic Vehicular Congestion: SUMO background vehicles loaded via route file;
          PSL is disabled so the Simulator skips slot-based speed overrides.
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


class TDTrafficModel(PSLTrafficModel):
    """Three-period time-of-day congestion model (Lu et al. 2020, Fig. 1).

    Slot 0 = morning peak  [0,  T1)  at congestion speed v_c
    Slot 1 = free-flow     [T1, T2)  at free-flow speed  v_f  (> v_c)
    Slot 2 = evening peak  [T2, u0]  at congestion speed v_c  (== slot 0)

    Speeds are GLOBAL: every edge gets the identical profile [v_c, v_f, v_c], so
    the td_milp planner and the Simulator use the same speeds end-to-end. Only
    get_slot() differs from PSL — slot boundaries are at T1/T2 instead of the
    PSL cyclic delta_t grid.
    """

    def __init__(
        self,
        enabled: bool,
        free_flow_ms: float,
        vc_ms: float,
        vf_ms: float,
        T1_s: float,
        T2_s: float,
        edge_ids: list[str],
    ) -> None:
        profiles = {eid: [vc_ms, vf_ms, vc_ms] for eid in sorted(edge_ids)}
        super().__init__(
            enabled=enabled,
            free_flow_ms=free_flow_ms,
            K=3,
            v_min=vc_ms,
            v_max=vf_ms,
            profiles=profiles,
        )
        self.T1_s = float(T1_s)
        self.T2_s = float(T2_s)

    def get_slot(self, t: float) -> int:
        if t < self.T1_s:
            return 0
        if t < self.T2_s:
            return 1
        return 2


class Traffic:
    """Builds a traffic model from raw ta_config.

    Reads traffic.model to select between PSL and SVC, then reads the
    appropriate sub-section (traffic.psl.* or traffic.svc.*).
    Requires the network's edge_ids for PSL profile generation.

    Attributes
    ----------
    model       : PSLTrafficModel — speed-limit model (disabled when traffic.model == svc)
    model_type  : str             — "psl" | "td" | "svc"
    route_file  : str | None      — path to SVC rvt_routes.xml, or None
    """

    def __init__(self, ta_config: dict[str, Any], edge_ids: list[str]) -> None:
        traffic_cfg  = ta_config.get("traffic", {})
        psl          = traffic_cfg.get("psl", {})
        ugv          = ta_config.get("simulation", {}).get("ugv", {})

        free_flow_ms    = float(ugv.get("free_flow_speed_kmh", 50.0)) / 3.6
        max_time_s      = float(ugv.get("max_duration_h", 6.0)) * 3600.0
        self.model_type = str(traffic_cfg.get("model", "psl"))
        self.route_file: str | None = None

        if self.model_type == "svc":
            svc_cfg             = traffic_cfg.get("svc", {})
            self.route_file     = svc_cfg.get("route_file") or None
            self.traffic_scale  = float(svc_cfg.get("traffic_scale", 1.0))
            self.model          = PSLTrafficModel(enabled=False, free_flow_ms=free_flow_ms)
            return

        self.traffic_scale = 1.0

        if self.model_type == "td":
            td = traffic_cfg.get("td", {})
            if not bool(td.get("enabled", True)):
                self.model = PSLTrafficModel(enabled=False, free_flow_ms=free_flow_ms)
                return
            vc_ms = float(td.get("v_congestion_kmh", 15.0)) / 3.6
            vf_ms = float(td.get("v_freeflow_kmh", 50.0)) / 3.6
            if not (0.0 < vc_ms < vf_ms):
                raise ValueError(
                    "traffic.td: require 0 < v_congestion_kmh < v_freeflow_kmh"
                )
            T1_s = float(td["T1_hours"]) * 3600.0 if "T1_hours" in td else max_time_s / 3.0
            T2_s = float(td["T2_hours"]) * 3600.0 if "T2_hours" in td else 2.0 * max_time_s / 3.0
            if not (0.0 < T1_s < T2_s < max_time_s):
                raise ValueError(
                    f"traffic.td: require 0 < T1 < T2 < max_duration "
                    f"(T1={T1_s/3600:.2f}h, T2={T2_s/3600:.2f}h, horizon={max_time_s/3600:.2f}h)"
                )
            self.model = TDTrafficModel(
                enabled=True,
                free_flow_ms=free_flow_ms,
                vc_ms=vc_ms,
                vf_ms=vf_ms,
                T1_s=T1_s,
                T2_s=T2_s,
                edge_ids=edge_ids,
            )
            return

        enabled = bool(psl.get("enabled", False))

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
