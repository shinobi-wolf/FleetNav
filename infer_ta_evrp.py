"""TA-EVRP inference: plan → (optional) simulate → write metrics.

Handles both pure-EVRP (dijkstra, no traffic) and full TA-EVRP (RL agent,
with traffic) based on config. The planner and TRSP agent are declared inside
the config file; no --planner flag is needed.

Usage
-----
  # Pure EVRP (MILP + static Dijkstra, no traffic)
  python infer_ta_evrp.py -c config/ta_evrp/milp.yaml

  # Full TA-EVRP (MILP + SARSA, with traffic)
  python infer_ta_evrp.py -c config/ta_evrp/milp_sarsa.yaml

  # Plan only, skip SUMO simulation
  python infer_ta_evrp.py -c config/ta_evrp/milp.yaml --no-sim

  # Suppress route plot
  python infer_ta_evrp.py -c config/ta_evrp/milp.yaml --no-plot

Config inheritance
------------------
  A config file may declare `base: path/to/base.yaml`.
  The base is loaded first; the child deep-merges on top.

Outputs
-------
  All result files are written under runs/ta_evrp/<run_id>/ where run_id is
  formatted as  {planner}_{agent}_{YYYYMMDD_HHMMSS}.
"""
from __future__ import annotations

import argparse
import json
import inspect
import sys
import threading
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dronesim.evrp.io.plot import build_plot_coordinates, plot_routes
from dronesim.evrp.io.instance_json import save_instance_json
from dronesim.evrp.planners import get_planner
from dronesim.evrp.planners.gvrp_milp import GVRPMILPPlanner

RUNS_DIR = Path("runs/ta_evrp")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_planned_stops(
    *,
    routes: list,
    run_dir: Path,
    run_id: str,
    planner_name: str,
    agent_name: str,
    policy_id: Optional[str],
    traffic_enabled: bool,
) -> Path:
    vehicles = []
    for route in routes:
        stops = []
        for i, s in enumerate(route.stops):
            stops.append(
                {
                    "stop_idx": i,
                    "node_name": getattr(s, "node_name", None),
                    "node_type": getattr(s, "node_type", None),
                    "sumo_node_id": getattr(s, "sumo_node_id", None),
                }
            )
        vehicles.append({"vehicle_id": route.vehicle_id, "stops": stops})

    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": planner_name,
        "trsp_agent": agent_name,
        "trsp_policy_id": policy_id,
        "traffic_enabled": bool(traffic_enabled),
        "vehicles": vehicles,
    }

    out_path = run_dir / "planned_stops.json"
    _write_json(out_path, payload)
    return out_path


def _resolve_net_xml_path(net_path: str) -> Optional[Path]:
    """Best-effort resolve a SUMO `.net.xml` path from a config net_path."""
    p = Path(net_path)
    if not p.exists():
        return None

    if p.suffixes[-2:] == [".net", ".xml"]:
        return p

    if p.suffix == ".sumocfg":
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            return None
        net_el = root.find("./input/net-file") or root.find(".//net-file")
        if net_el is None:
            return None
        value = net_el.get("value")
        if not value:
            return None
        net_xml = (p.parent / value).resolve()
        return net_xml if net_xml.exists() else None

    return None


def _edges_to_nodes_from_net(edge_ids: list[str], net) -> Optional[list[str]]:
    if not edge_ids:
        return []
    node_ids: list[str] = []
    for i, eid in enumerate(edge_ids):
        try:
            e = net.getEdge(eid)
        except Exception:
            return None
        if i == 0:
            node_ids.append(e.getFromNode().getID())
        node_ids.append(e.getToNode().getID())
    return node_ids


def _extract_vehicle_id(vs: Any) -> Optional[int]:
    if vs is None:
        return None
    for attr in ("vehicle_id", "ugv_id", "vid", "id"):
        if hasattr(vs, attr):
            try:
                return int(getattr(vs, attr))
            except Exception:
                continue
    return None


class _RouteLogger:
    """Runtime logger for per-leg driven routes.

    Attaches a wrapper around the simulator's private `_drive` method (if present)
    without requiring edits to the simulator implementation.
    """

    def __init__(self, *, net_path: str) -> None:
        self._lock = threading.Lock()
        self.legs: list[dict[str, Any]] = []

        self._net = None
        net_xml = _resolve_net_xml_path(net_path)
        if net_xml is None:
            return
        try:
            import sumolib  # type: ignore

            self._net = sumolib.net.readNet(str(net_xml))
        except Exception:
            self._net = None

    def _log_leg(
        self,
        *,
        vehicle_id: Optional[int],
        sim_time_s: Optional[float],
        edge_ids: list[str],
        src_node: Optional[str] = None,
        dst_node: Optional[str] = None,
    ) -> None:
        node_ids = None
        if self._net is not None:
            node_ids = _edges_to_nodes_from_net(edge_ids, self._net)
        # Prefer the mapped endpoints when available.
        mapped_src = node_ids[0] if isinstance(node_ids, list) and node_ids else None
        mapped_dst = node_ids[-1] if isinstance(node_ids, list) and node_ids else None
        with self._lock:
            self.legs.append(
                {
                    "vehicle_id": vehicle_id,
                    "start_time_s": float(sim_time_s) if sim_time_s is not None else None,
                    "src_node": mapped_src or src_node,
                    "dst_node": mapped_dst or dst_node,
                    "edge_ids": list(edge_ids),
                    "node_ids": node_ids,
                    "thread": threading.current_thread().name,
                }
            )

    def _vehicle_id_from_stack(self) -> Optional[int]:
        frame = inspect.currentframe()
        # Skip ourselves + wrapper frame.
        for _ in range(12):
            if frame is None:
                break
            for key in ("vs", "vehicle_state", "vehicle", "ugv", "state"):
                if key in frame.f_locals:
                    vid = _extract_vehicle_id(frame.f_locals.get(key))
                    if vid is not None:
                        return vid
            for v in frame.f_locals.values():
                vid = _extract_vehicle_id(v)
                if vid is not None:
                    return vid
            frame = frame.f_back
        return None

    def attach(self, sim: Any) -> bool:
        # Prefer hooking `_resolve_route(src, dst, sim_time_s) -> edge_ids`, which is
        # used in both sequential and concurrent simulators.
        if hasattr(sim, "_resolve_route") and callable(getattr(sim, "_resolve_route")):
            original = getattr(sim, "_resolve_route")
            sig = inspect.signature(original)

            def _wrapper(*args, **kwargs):
                bound = sig.bind_partial(*args, **kwargs)
                src = bound.arguments.get("src_node")
                dst = bound.arguments.get("dst_node")
                sim_time_s = bound.arguments.get("sim_time_s")

                if src is None and len(args) >= 1:
                    src = args[0]
                if dst is None and len(args) >= 2:
                    dst = args[1]
                if sim_time_s is None and len(args) >= 3:
                    sim_time_s = args[2]

                edge_ids = original(*args, **kwargs)
                if edge_ids is None:
                    edge_ids = []
                if isinstance(edge_ids, tuple):
                    edge_ids = list(edge_ids)

                vid = self._vehicle_id_from_stack()
                # Store src/dst even if we cannot map edges->nodes.
                self._log_leg(
                    vehicle_id=vid,
                    sim_time_s=float(sim_time_s) if sim_time_s is not None else None,
                    edge_ids=list(edge_ids),
                    src_node=src,
                    dst_node=dst,
                )
                return edge_ids

            setattr(sim, "_resolve_route", _wrapper)
            return True

        # Fallback: hook `_drive(...)` if `_resolve_route` is unavailable.
        if hasattr(sim, "_drive") and callable(getattr(sim, "_drive")):
            original = getattr(sim, "_drive")
            sig = inspect.signature(original)

            def _wrapper(*args, **kwargs):
                bound = sig.bind_partial(*args, **kwargs)
                vs = bound.arguments.get("vs")
                edge_list = bound.arguments.get("edge_list")
                sim_time_s = bound.arguments.get("sim_time_s")

                if vs is None and args:
                    vs = args[0]
                if edge_list is None:
                    for v in list(bound.arguments.values()) + list(args):
                        if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                            edge_list = v
                            break
                if sim_time_s is None:
                    for name in ("t", "time_s", "start_time_s"):
                        if name in bound.arguments:
                            sim_time_s = bound.arguments[name]
                            break

                if isinstance(edge_list, tuple):
                    edge_list = list(edge_list)
                if not isinstance(edge_list, list):
                    edge_list = []

                self._log_leg(
                    vehicle_id=_extract_vehicle_id(vs),
                    sim_time_s=float(sim_time_s) if sim_time_s is not None else None,
                    edge_ids=edge_list,
                )
                return original(*args, **kwargs)

            setattr(sim, "_drive", _wrapper)
            return True

        return False


def _build_local_routes_payload(
    *,
    run_id: str,
    cfg: dict,
    routes: list,
    leg_logs: list[dict[str, Any]],
    local_planner: Any,
) -> dict:
    traffic_enabled = bool(cfg.get("traffic", {}).get("enabled", False))
    sim_type = cfg.get("simulation", {}).get("simulator", "unknown")

    planned_by_vehicle: dict[int, list[str]] = {}
    for r in routes:
        seq = [getattr(s, "sumo_node_id", None) for s in r.stops]
        planned_by_vehicle[int(r.vehicle_id)] = [x for x in seq if isinstance(x, str)]

    vehicles_out: dict[str, dict[str, Any]] = {}
    for entry in leg_logs:
        vid = entry.get("vehicle_id")
        vid_key = str(vid) if vid is not None else "unknown"
        vehicles_out.setdefault(vid_key, {"vehicle_id": vid, "legs": [], "warnings": []})

        edge_ids = entry.get("edge_ids") or []
        node_ids = entry.get("node_ids")

        from_node = (
            node_ids[0]
            if isinstance(node_ids, list) and node_ids
            else entry.get("src_node")
        )
        to_node = (
            node_ids[-1]
            if isinstance(node_ids, list) and node_ids
            else entry.get("dst_node")
        )
        start_time_s = entry.get("start_time_s")

        leg_kind = "unknown"
        if vid is not None and vid in planned_by_vehicle and from_node and to_node:
            planned = planned_by_vehicle[vid]
            for i in range(len(planned) - 1):
                if planned[i] == from_node and planned[i + 1] == to_node:
                    leg_kind = "planned"
                    break
            else:
                leg_kind = "emergency"

        traffic_slot = 0
        if local_planner is not None and hasattr(local_planner, "traffic"):
            try:
                traffic_slot = int(local_planner.traffic.get_slot(float(start_time_s or 0.0)))
            except Exception:
                traffic_slot = 0

        planner_kind = "dijkstra"
        fallback_reason = None
        if local_planner is not None and from_node and to_node and start_time_s is not None:
            if hasattr(local_planner, "plan_edges_and_nodes"):
                try:
                    pred_edges, _pred_nodes, meta = local_planner.plan_edges_and_nodes(
                        from_node, to_node, float(start_time_s)
                    )
                    if list(pred_edges) == list(edge_ids):
                        planner_kind = "trsp"
                        fallback_reason = meta.get("fallback_reason")
                except Exception:
                    pass

        vehicles_out[vid_key]["legs"].append(
            {
                "leg_idx": len(vehicles_out[vid_key]["legs"]),
                "leg_kind": leg_kind,
                "planner_kind": planner_kind,
                "from_sumo_node_id": from_node,
                "to_sumo_node_id": to_node,
                "start_time_s": start_time_s,
                "traffic_slot": traffic_slot if traffic_enabled else 0,
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "fallback_reason": fallback_reason,
            }
        )

    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "simulator": sim_type,
        "traffic_enabled": traffic_enabled,
        "vehicles": list(vehicles_out.values()),
    }
    return payload


# ---------------------------------------------------------------------------
# Config loader with base-inheritance and deep-merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as fh:
        child = yaml.safe_load(fh) or {}
    base_ref = child.pop("base", None)
    if base_ref is None:
        return child
    base_path = Path(base_ref)
    with base_path.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    base.pop("base", None)
    return _deep_merge(base, child)


# ---------------------------------------------------------------------------
# Translate new TA-EVRP config schema → internal format used by simulators/planners
# ---------------------------------------------------------------------------

def _normalize_config(ta: dict) -> dict:
    """Map TA-EVRP config keys to the internal format expected by the simulator/planner stack."""
    net = ta.get("network", {})
    evrp_cfg = ta.get("evrp", {})
    milp_cfg = evrp_cfg.get("milp", {})
    heuristic_cfg = evrp_cfg.get("heuristic", {})
    trsp = ta.get("trsp", {})
    sim = ta.get("simulation", {})
    ugv = sim.get("ugv", {})
    tasks = sim.get("tasks", {})
    traffic_cfg = ta.get("traffic", {})
    psl = traffic_cfg.get("psl", {})
    simulated = traffic_cfg.get("simulated", {})
    vehicular = simulated.get("vehicular", {})
    sumo_cfg = ta.get("sumo", {})
    out = ta.get("output", {})

    # Battery range → fuel_capacity + consumption_rate
    # battery_range_m sets: consumption_rate_per_m = 1.0 gal/m so range = fuel_capacity
    # This is achieved by setting consumption_rate = 1609.344 gal/mile (= 1 gal/m after /1609.344)
    battery_range_m = float(ugv.get("battery_range_m", 10000.0))

    cfg: dict = {}

    cfg["generation_seed"] = ta.get("generation_seed", 42)

    cfg["net"] = {
        "net_path": net.get("net_path", "data/grid_road/grid_road.sumocfg"),
        "warehouse": net.get("warehouse", "center"),
        "max_degree": net.get("max_degree", 4),
    }

    cfg["planner"] = evrp_cfg.get("planner", "milp")

    cfg["fleet"] = {
        "size": ugv.get("fleet_size", 3),
        "max_speed": ugv.get("free_flow_speed_mph", 40.0),
    }

    cfg["gvrp"] = {
        "customer_count": tasks.get("customer_count", 10),
        "customer_sampling": tasks.get("customer_sampling", "count"),
        "task_density": tasks.get("task_density", 30),
        "afs_count": tasks.get("afs_count", 3),
        "distance_unit": "meters",
        "fuel_capacity": battery_range_m,
        "consumption_rate": 1609.344,        # gal/mile → 1.0 gal/m after internal /1609.344
        "speed_mph": ugv.get("free_flow_speed_mph", 40.0),
        "max_duration_hours": ugv.get("max_duration_hours", 6.0),
        "customer_service_time_hours": ugv.get("customer_service_time_hours", 0),
        "refuel_time_hours": ugv.get("recharge_time_hours", 0),
        "afs_dummy_copies": milp_cfg.get("afs_dummy_copies", 3),
        "depot_dummy_copies": milp_cfg.get("depot_dummy_copies", 3),
        "plot": bool(out.get("plot", True)),
        "run_single_vehicle_benchmark": False,
    }

    cfg["solver"] = {
        "time_limit_seconds": milp_cfg.get("time_limit_s", 300),
        "mip_gap": milp_cfg.get("mip_gap", 0.01),
        "output_flag": milp_cfg.get("output_flag", 1),
    }

    cfg["heuristic"] = {
        "alns_iterations": heuristic_cfg.get("alns_iterations", 300),
        "seed": heuristic_cfg.get("seed", 42),
    }

    # Traffic: PSL config (T derived at simulation time from max_time_s)
    psl_enabled = bool(psl.get("enabled", False))
    cfg["traffic"] = {
        "enabled": psl_enabled,
        "K": psl.get("K", 1),
        "v_min": psl.get("v_min", 40.0),
        "v_max": psl.get("v_max", 100.0),
        "speed_step": psl.get("speed_step", 60.0),
        "seed": psl.get("seed", 42),
        "profile_type": psl.get("profile_type", "random"),
    }

    # Cost model (td_milp only)
    if "cost_model" in ta:
        cfg["cost_model"] = ta["cost_model"]

    # Background SUMO traffic scale
    vehicle_scale = float(vehicular.get("scale", 0.0)) if vehicular.get("enabled") else 0.0

    cfg["sumo"] = {
        "step_length": sumo_cfg.get("step_length", 1.0),
        "traffic_scale": vehicle_scale,
        "delay": sumo_cfg.get("delay", 10),
    }

    agent = trsp.get("agent", "dijkstra")
    policy_id = trsp.get("policy_id", None)
    cfg["simulation"] = {
        "simulator": sumo_cfg.get("simulator", "concurrent"),
        "max_time": sumo_cfg.get("max_time", 12),
        "render_mode": sumo_cfg.get("render_mode", "quiet"),
        "runtime_refuel": bool(ugv.get("runtime_refuel", False)),
        "use_planned_wait_times": False,
        # kept for reference but no longer drives planner construction
        "_trsp_agent": agent,
        "_trsp_policy_id": policy_id,
    }

    cfg["output"] = {
        "save_instance_json": "generated_instance.json" if out.get("save_instance_json", True) else None,
        "save_metrics_json": "sim_metrics.json" if out.get("save_metrics_json", True) else None,
    }

    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id(planner: str, agent: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{planner}_{agent}_{ts}"


def _plot_routes(planner, routes, out_path: str) -> None:
    if not hasattr(planner, "_last_generation_info"):
        print("  [plot] planner did not store instance data; skipping plot.")
        return

    gen_info = planner._last_generation_info
    physical_coords = gen_info.get("physical_coords")

    if isinstance(planner, GVRPMILPPlanner):
        coords = build_plot_coordinates(
            planner._last_physical_dist,
            planner._last_instance,
            physical_coords,
        )
        plot_routes(planner._last_raw_routes, coords, planner._last_instance["node_types"], out_path=out_path)
    else:
        meta = planner._last_physical_meta
        sumo_ids: list[str] = meta["physical_sumo_ids"]
        node_types: list[str] = meta["physical_node_types"]
        sumo_to_phys = {sid: i for i, sid in enumerate(sumo_ids)}
        raw_routes = []
        for route in routes:
            nodes = [
                sumo_to_phys[s.sumo_node_id]
                for s in route.stops
                if s.sumo_node_id in sumo_to_phys
            ]
            dist = float(sum(
                planner._last_physical_dist[nodes[i], nodes[i + 1]]
                for i in range(len(nodes) - 1)
            ))
            raw_routes.append({"vehicle": route.vehicle_id, "nodes": nodes, "distance": dist})
        plot_routes(raw_routes, physical_coords, node_types, out_path=out_path)


def _print_routes(routes) -> None:
    print("\n── Planned routes ──────────────────────────────────────────────")
    for route in routes:
        seq = " → ".join(f"{s.node_name}({s.node_type[0].upper()})" for s in route.stops)
        print(f"  Vehicle {route.vehicle_id}: {seq}")
        print(f"    customers={len(route.customer_stops)}  recharges={len(route.afs_stops)}")


def _print_metrics(metrics: dict) -> None:
    print("\n── Simulation metrics ──────────────────────────────────────────")
    print(f"  makespan          : {metrics['makespan_s']:.1f} s  ({metrics['makespan_h']:.3f} h)")
    print(f"  total distance    : {metrics['total_distance_m']:.1f} m  ({metrics['total_distance_km']:.3f} km)")
    print(f"  total energy      : {metrics['total_energy_kwh']:.3f} kWh")
    print(f"  total recharges   : {metrics['total_recharges']}")
    print(f"  customers served  : {metrics['customers_served']}")
    if metrics["avg_inter_customer_gap_s"] is not None:
        print(f"  avg service gap   : {metrics['avg_inter_customer_gap_s']:.1f} s")
    print("\n  Per-vehicle breakdown:")
    for vid, v in metrics["per_vehicle"].items():
        print(
            f"    V{vid}: dist={v['distance_m']:.1f}m  "
            f"energy={v['energy_kwh']:.3f}kWh  "
            f"recharges={v['num_recharges']}  "
            f"customers={v['customers_served']}  "
            f"finish={v['finish_time_s']:.1f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TA-EVRP planner + SUMO simulator")
    parser.add_argument("-c", "--config", default="config/ta_evrp/milp.yaml")
    parser.add_argument("--no-sim", action="store_true", help="Skip SUMO simulation")
    parser.add_argument("--no-plot", action="store_true", help="Suppress route plot")
    parser.add_argument("--save-metrics", action="store_true", help="Save TA-EVRP metrics to ta_evrp_metrics.json")
    args = parser.parse_args()

    ta_config = load_config(args.config)
    cfg = _normalize_config(ta_config)

    planner_name = cfg["planner"]
    agent_name = cfg["simulation"]["_trsp_agent"]
    policy_id = cfg["simulation"]["_trsp_policy_id"]
    run_id = _make_run_id(planner_name, agent_name)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTA-EVRP run: {run_id}")
    print(f"  planner : {planner_name}")
    print(f"  agent   : {agent_name}" + (f"  policy: {policy_id}" if policy_id else ""))
    print(f"  traffic : {'PSL enabled' if cfg['traffic']['enabled'] else 'disabled'}")
    print(f"  output  : {run_dir}/")

    # ── Redirect output paths ─────────────────────────────────────────────────
    if cfg["output"].get("save_instance_json"):
        cfg["output"]["save_instance_json"] = str(run_dir / Path(cfg["output"]["save_instance_json"]).name)
    
    if args.save_metrics:
        cfg["output"]["save_metrics_json"] = str(run_dir / "ta_evrp_metrics.json")
    elif cfg["output"].get("save_metrics_json"):
        cfg["output"]["save_metrics_json"] = str(run_dir / Path(cfg["output"]["save_metrics_json"]).name)

    # ── Plan ─────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Planning with '{planner_name}' planner...")
    planner = get_planner(planner_name)
    routes = planner.plan(cfg)
    _print_routes(routes)

    planned_stops_path = _export_planned_stops(
        routes=routes,
        run_dir=run_dir,
        run_id=run_id,
        planner_name=planner_name,
        agent_name=agent_name,
        policy_id=policy_id,
        traffic_enabled=cfg.get("traffic", {}).get("enabled", False),
    )
    print(f"\n  Planned stops saved to {planned_stops_path}")

    save_instance_json(
        cfg,
        planner._last_physical_dist,
        planner._last_physical_meta,
        planner._last_generation_info,
    )

    # ── Optional plot ─────────────────────────────────────────────────────────
    do_plot = bool(cfg.get("gvrp", {}).get("plot", True)) and not args.no_plot
    if do_plot:
        plot_path = str(run_dir / "gvrp_routes.png")
        _plot_routes(planner, routes, out_path=plot_path)

    # ── Build TRSP local planner if RL agent requested ────────────────────────
    local_planner = None
    if agent_name not in ("dijkstra", "dynamic_dijkstra"):
        if not policy_id:
            raise ValueError(
                f"trsp.policy_id must be set in config when trsp.agent is '{agent_name}'"
            )
        import dronesim.trsp  # noqa: F401 — registers SumoTRSPEnv-v0
        from dronesim.ta_evrp import TRSPLocalPlanner
        local_planner = TRSPLocalPlanner(policy_id)

    # ── Simulate ─────────────────────────────────────────────────────────────
    if args.no_sim:
        print("\n[2/3] Simulation skipped (--no-sim).")
        return

    sim_type = cfg.get("simulation", {}).get("simulator", "sequential")
    print(f"\n[2/3] Starting SUMO simulation ({sim_type})...")
    from dronesim.evrp.simulator import get_simulator
    sim = get_simulator(cfg, local_planner=local_planner)

    route_logger = _RouteLogger(net_path=cfg.get("net", {}).get("net_path", ""))
    logger_attached = route_logger.attach(sim)
    if not logger_attached:
        print("  [route-log] Warning: simulator has no _drive() hook; local_routes.json may be empty.")

    metrics = sim.run(routes)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/3] Results:")
    _print_metrics(metrics)

    metrics_path = cfg["output"].get("save_metrics_json")
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(json.dumps(metrics, indent=2))
        print(f"\n  Metrics saved to {metrics_path}")

    local_routes_payload = _build_local_routes_payload(
        run_id=run_id,
        cfg=cfg,
        routes=routes,
        leg_logs=route_logger.legs,
        local_planner=local_planner,
    )
    local_routes_path = run_dir / "local_routes.json"
    _write_json(local_routes_path, local_routes_payload)
    print(f"\n  Local routes saved to {local_routes_path}")


if __name__ == "__main__":
    main()
