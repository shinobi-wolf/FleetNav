"""Instance orchestrator for the TA-EVRP pipeline.

Builds Network, Traffic, and Problem from a raw ta_config dict and exposes
a clean interface to the Planner, Simulator, and runner script.
"""
from __future__ import annotations

from typing import Any, Optional

import networkx as nx

from .network  import Network
from .traffic  import Traffic, PSLTrafficModel
from .problem  import Problem


def _build_sim_cfg(ta: dict[str, Any]) -> dict[str, Any]:
    """Translate ta_config into the format EVRPSimulator expects.

    This replaces the deleted _reformat_cfg_sim from infer_ta_evrp.py.
    All unit conversions (km/h → m/s, h → s) are done here.
    """
    ugv         = ta.get("simulation", {}).get("ugv", {})
    tasks       = ta.get("simulation", {}).get("tasks", {})
    traffic_cfg = ta.get("traffic", {})
    psl         = traffic_cfg.get("psl", {})
    td          = traffic_cfg.get("td", {})
    svc         = traffic_cfg.get("svc", {})
    model       = str(traffic_cfg.get("model", "psl"))
    sumo        = ta.get("sumo", {})
    out         = ta.get("output", {})

    battery_L    = float(ugv.get("battery_capacity_L", 50.0))
    consume_L_km = float(ugv.get("consumption_rate_L_per_km", 5.0))
    speed_kmh    = float(ugv.get("free_flow_speed_kmh", 50.0))
    svc_time_h   = float(ugv.get("customer_service_time_h", 0.0))
    recharge_h   = float(ugv.get("recharge_time_h", 0.0))

    # Speed-scale bounds for the renderer and the enabled flag depend on the model.
    if model == "td":
        traffic_enabled = bool(td.get("enabled", True))
        v_min_ms = float(td.get("v_congestion_kmh", 15.0)) / 3.6
        v_max_ms = float(td.get("v_freeflow_kmh", 50.0)) / 3.6
    else:
        traffic_enabled = False if model == "svc" else bool(psl.get("enabled", False))
        v_min_ms = float(psl.get("v_min_kmh", 40.0))  / 3.6
        v_max_ms = float(psl.get("v_max_kmh", 100.0)) / 3.6

    return {
        "model": {
            "evrp": ta.get("evrp", {}),
            "trsp": ta.get("trsp", {}),
        },
        "net": ta.get("network", {}),
        "instance": {
            "fleet_size":              int(ugv.get("fleet_size", 3)),
            "customer_count":          int(tasks.get("customer_count", 10)),
            "ugv_range":               battery_L * 1000.0 / consume_L_km,  # m
            "ugv_speed":               speed_kmh / 3.6,                     # m/s
            "customer_service_time_s": svc_time_h * 3600.0,                # s
            "recharge_time_s":         recharge_h * 3600.0,                # s
            "runtime_refuel":          bool(ugv.get("runtime_refuel", False)),
        },
        "traffic": {
            "model":         model,
            "route_file":    svc.get("route_file") if model == "svc" else None,
            "traffic_scale": float(svc.get("traffic_scale", 1.0)) if model == "svc" else 1.0,
            "enabled":       traffic_enabled,
            "K":             int(psl.get("K", 1)),
            "slot_duration": psl.get("slot_duration"),
            "v_min":         v_min_ms,   # m/s — renderer speed-color floor
            "v_max":         v_max_ms,   # m/s — renderer speed-color ceiling
            "speed_step":    float(psl.get("speed_step_kmh", 60.0)) / 3.6,  # m/s
            "seed":          int(psl.get("seed", 42)),
        },
        "hps": {
            "stop_time": float(sumo.get("max_time_h", 12.0)) * 3600.0,  # s — SUMO wall-clock budget
            "sumo":      sumo,
            "output":    out,
        },
    }


class Instance:
    """Fully built TA-EVRP problem instance.

    Orchestrates Network, Traffic, and Problem construction from a raw
    ta_config dict (as returned by load_config in infer_ta_evrp.py).

    Attributes exposed to the runner
    ---------------------------------
    planner_name, agent_name, policy_id   -- from config
    net_path, traffic_enabled, sim_type   -- from config
    net_xml_path, num_nodes, num_edges    -- from Network

    Attributes exposed to Planner
    --------------------------------
    network   : Network          -- full Network object (graph, net, positions)
    graph     : nx.DiGraph
    problem   : Problem          -- sampled depot/customers/AFS + distance matrix

    Attributes exposed to Simulator
    ---------------------------------
    network   : Network
    traffic   : PSLTrafficModel  -- per-edge speed profiles
    sim_cfg   : dict             -- pre-formatted config dict for EVRPSimulator
    """

    def __init__(self, ta_config: dict[str, Any]) -> None:
        self.ta_config = ta_config

        evrp        = ta_config.get("evrp", {})
        trsp        = ta_config.get("trsp", {})
        net         = ta_config.get("network", {})
        sumo        = ta_config.get("sumo", {})
        traffic_cfg = ta_config.get("traffic", {})
        psl         = traffic_cfg.get("psl", {})
        td          = traffic_cfg.get("td", {})
        traffic_model = str(traffic_cfg.get("model", "psl"))

        # ── Flat identifiers ─────────────────────────────────────────────────
        self.planner_name:    str           = str(evrp.get("planner", "milp"))
        self.agent_name:      str           = str(trsp.get("agent", "dijkstra"))
        self.policy_id:       Optional[str] = trsp.get("policy_id") or None
        self.net_path:        str           = str(net.get("net_path", ""))
        self.traffic_model:   str           = traffic_model
        if traffic_model == "svc":
            self.traffic_enabled: bool = True
        elif traffic_model == "td":
            self.traffic_enabled = bool(td.get("enabled", True))
        else:
            self.traffic_enabled = bool(psl.get("enabled", False))
        self.sim_type:        str           = str(sumo.get("simulator", "concurrent"))

        # ── Three submodules ─────────────────────────────────────────────────
        self._network  = Network(self.net_path)
        self._traffic  = Traffic(ta_config, self._network.edge_ids)
        self._problem  = Problem(ta_config, self._network)

        # ── Simulator config ─────────────────────────────────────────────────
        self.sim_cfg: dict[str, Any] = _build_sim_cfg(ta_config)

    # ── Runner properties ────────────────────────────────────────────────────

    @property
    def net_xml_path(self) -> str:
        return self._network.net_xml_path

    @property
    def num_nodes(self) -> int:
        return self._network.num_nodes

    @property
    def num_edges(self) -> int:
        return self._network.num_edges

    # ── Planner / Simulator properties ───────────────────────────────────────

    @property
    def network(self) -> Network:
        return self._network

    @property
    def graph(self) -> nx.DiGraph:
        return self._network.graph

    @property
    def traffic(self) -> PSLTrafficModel:
        return self._traffic.model

    @property
    def problem(self) -> Problem:
        return self._problem
