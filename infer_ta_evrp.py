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
import inspect
import json
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

from dronesim.ta_evrp import Instance, GlobalPlanner, LocalPlanner, Simulator

RUNS_DIR = Path("runs/ta_evrp")


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
    base = load_config(base_ref)  # recurse to follow the full chain
    return _deep_merge(base, child)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_run_id(planner_name: str, agent_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{planner_name}_{agent_name}_{ts}"


def _export_planned_stops(
    *,
    routes: list,
    run_dir: Path,
    run_id: str,
    instance: Instance,
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
        "planner": instance.planner_name,
        "trsp_agent": instance.agent_name,
        "trsp_policy_id": instance.policy_id,
        "traffic_enabled": instance.traffic_enabled,
        "vehicles": vehicles,
    }

    out_path = run_dir / "planned_stops.json"
    _write_json(out_path, payload)
    return out_path


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
    print(f"  total recharges   : {metrics['total_recharges']}")
    print(f"  customers served  : {metrics['customers_served']}")
    if metrics["avg_inter_customer_gap_s"] is not None:
        print(f"  avg service gap   : {metrics['avg_inter_customer_gap_s']:.1f} s")
    print("\n  Per-vehicle breakdown:")
    for vid, v in metrics["per_vehicle"].items():
        print(
            f"    V{vid}: dist={v['distance_m']:.1f}m  "
            f"recharges={v['num_recharges']}  "
            f"customers={v['customers_served']}  "
            f"finish={v['finish_time_s']:.1f}s"
        )


# ---------------------------------------------------------------------------
# Route logger (attaches to simulator internals to record per-leg edges)
# ---------------------------------------------------------------------------

def _resolve_net_xml_path(net_path: str) -> Optional[Path]:
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
    """Records per-leg driven edge lists by hooking into the simulator."""

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
                result = original(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    raw_edges, _ = result
                    edge_ids_to_log = list(raw_edges) if raw_edges else []
                elif result is None:
                    edge_ids_to_log = []
                else:
                    edge_ids_to_log = list(result)
                vid = self._vehicle_id_from_stack()
                self._log_leg(
                    vehicle_id=vid,
                    sim_time_s=float(sim_time_s) if sim_time_s is not None else None,
                    edge_ids=edge_ids_to_log,
                    src_node=src,
                    dst_node=dst,
                )
                return result

            setattr(sim, "_resolve_route", _wrapper)
            return True

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
    instance: Instance,
    routes: list,
    leg_logs: list[dict[str, Any]],
    local_planner: Any,
) -> dict:
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
            node_ids[0] if isinstance(node_ids, list) and node_ids else entry.get("src_node")
        )
        to_node = (
            node_ids[-1] if isinstance(node_ids, list) and node_ids else entry.get("dst_node")
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
                "traffic_slot": traffic_slot if instance.traffic_enabled else 0,
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "fallback_reason": fallback_reason,
            }
        )

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "simulator": instance.sim_type,
        "traffic_enabled": instance.traffic_enabled,
        "vehicles": list(vehicles_out.values()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TA-EVRP planner + SUMO simulator")
    parser.add_argument("-c", "--config", default="config/ta_evrp/milp.yaml")
    parser.add_argument("--no-sim", action="store_true", help="Skip SUMO simulation")
    parser.add_argument("--no-plot", action="store_true", help="Suppress route plot")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics to ta_evrp_metrics.json")
    args = parser.parse_args()

    ta_config = load_config(args.config)

    # ── Instance ─────────────────────────────────────────────────────────────
    print(f"\n[0/3] Building instance from {ta_config['network']['net_path']} ...")
    instance = Instance(ta_config)
    print(f"       {instance.net_xml_path}  ({instance.num_nodes} nodes, {instance.num_edges} edges)")

    run_id = _make_run_id(instance.planner_name, instance.agent_name)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTA-EVRP run: {run_id}")
    print(f"  planner : {instance.planner_name}")
    print(f"  agent   : {instance.agent_name}" + (f"  policy: {instance.policy_id}" if instance.policy_id else ""))
    print(f"  traffic : {instance.traffic_model + ' enabled' if instance.traffic_enabled else 'disabled'}")
    print(f"  output  : {run_dir}/")

    # ── Plan ─────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Planning with '{instance.planner_name}' planner...")
    planner = GlobalPlanner()
    trsp_planner = LocalPlanner(instance.policy_id) if instance.policy_id else None
    routes = planner.plan(instance)
    _print_routes(routes)

    planned_stops_path = _export_planned_stops(
        routes=routes, run_dir=run_dir, run_id=run_id, instance=instance,
    )
    print(f"\n  Planned stops saved to {planned_stops_path}")

    planner.save_instance_json(out_dir=run_dir)

    if not args.no_plot:
        planner.plot(routes, out_path=str(run_dir / "gvrp_routes.png"))

    # ── Simulate ─────────────────────────────────────────────────────────────
    if args.no_sim:
        print("\n[2/3] Simulation skipped (--no-sim).")
        return

    print(f"\n[2/3] Starting SUMO simulation ...")
    simulator = Simulator(instance, trsp_planner)

    route_logger = _RouteLogger(net_path=instance.net_path)
    logger_attached = route_logger.attach(simulator.sim)
    if not logger_attached:
        print("  [route-log] Warning: simulator has no hook; local_routes.json may be empty.")

    metrics = simulator.run(routes)
    raps_log        = metrics.pop("raps_log", [])
    traffic_log     = metrics.pop("traffic_log", [])
    traffic_profile = metrics.pop("traffic_profile", {})

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/3] Results:")
    _print_metrics(metrics)

    metrics_filename = "ta_evrp_metrics.json" if args.save_metrics else "sim_metrics.json"
    metrics_path = run_dir / metrics_filename
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Metrics saved to {metrics_path}")

    if traffic_profile:
        profile_path = run_dir / "traffic_profile.json"
        profile_path.write_text(json.dumps(traffic_profile, indent=2))
        print(f"  Traffic profile saved to {profile_path}")

    if traffic_log:
        log_path = run_dir / "traffic_log.jsonl"
        with open(log_path, "w") as f:
            for event in traffic_log:
                f.write(json.dumps(event) + "\n")
        print(f"  Traffic event log saved to {log_path}")

    local_routes_payload = _build_local_routes_payload(
        run_id=run_id,
        instance=instance,
        routes=routes,
        leg_logs=route_logger.legs,
        local_planner=simulator.local_planner,
    )
    local_routes_path = run_dir / "local_routes.json"
    _write_json(local_routes_path, local_routes_payload)
    print(f"\n  Local routes saved to {local_routes_path}")

    if raps_log:
        raps_log_path = run_dir / "raps_log.json"
        _write_json(raps_log_path, raps_log)
        print(f"  RAPS log → {raps_log_path}  ({len(raps_log)} legs)")


if __name__ == "__main__":
    main()
