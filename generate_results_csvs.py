#!/usr/bin/env python3
"""
Generate/update results CSVs from all completed runs/ta_evrp/ directories.

Writes four files to results/:
  networks.csv          — unique road network archetypes
  traffic.csv           — unique PSL traffic parameter sets
  problem_instances.csv — unique problem instance configurations
  ta_evrp_results.csv   — per-run metrics (one row per run)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Network archetype metadata not derivable from the .net.xml itself.
# Keyed on the net.xml filename (basename).
# ---------------------------------------------------------------------------
NETWORK_ARCHETYPES: dict[str, dict] = {
    "grid_road.net.xml": {
        "topology": "Grid",
        "connectivity_lambda": 0.3,
        "vehicle_type": "Car",
    },
}

PLANNER_DISPLAY = {
    "milp":       "MILP",
    "heuristic":  "Heuristic",
    "mtsp":       "mTSP",
    "td_milp":    "trEVRP Exact",
}

AGENT_DISPLAY = {
    "dijkstra":         "SSP",
    "dynamic_dijkstra": "Dyn-SP",
    "sarsa":            "SARSA",
    "sarsa_lambda":     "SARSA-λ",
    "dqn":              "DQN",
    "a2c":              "A2C",
    "ppo":              "PPO",
}

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs" / "ta_evrp"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "config" / "ta_evrp"


# ---------------------------------------------------------------------------
# Config helpers (mirrors infer_ta_evrp.py)
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: Path) -> dict:
    with path.open() as fh:
        child = yaml.safe_load(fh) or {}
    base_ref = child.pop("base", None)
    if base_ref is None:
        return child
    base_path = ROOT / base_ref
    with base_path.open() as fh:
        base = yaml.safe_load(fh) or {}
    base.pop("base", None)
    return _deep_merge(base, child)


def _build_config_index() -> dict[tuple[str, str], dict]:
    """Return {(planner, agent): resolved_config} for all non-base config files."""
    index: dict[tuple[str, str], dict] = {}
    for p in CONFIG_DIR.glob("*.yaml"):
        if p.name == "base.yaml":
            continue
        cfg = load_config(p)
        planner = cfg.get("evrp", {}).get("planner", "")
        agent = cfg.get("trsp", {}).get("agent", "")
        if planner and agent:
            index[(planner, agent)] = cfg
    return index


# ---------------------------------------------------------------------------
# Network node/edge counts via sumolib
# ---------------------------------------------------------------------------

_net_cache: dict[str, tuple[int, int]] = {}


def _net_stats(net_xml: Path) -> tuple[int, int]:
    key = str(net_xml)
    if key not in _net_cache:
        try:
            import sumolib  # type: ignore
            net = sumolib.net.readNet(str(net_xml))
            _net_cache[key] = (len(net.getNodes()), len(net.getEdges()))
        except Exception:
            _net_cache[key] = (-1, -1)
    return _net_cache[key]


# ---------------------------------------------------------------------------
# Fingerprint helpers → stable sortable IDs
# ---------------------------------------------------------------------------

def _network_fp(net_xml_path: str) -> tuple:
    return (Path(net_xml_path).name,)


def _traffic_fp(psl: dict) -> tuple:
    return (
        int(psl.get("K", 1)),
        float(psl.get("v_min", 0)),
        float(psl.get("v_max", 0)),
        float(psl.get("speed_step", 0)),
        str(psl.get("profile_type", "random")),
    )


def _problem_fp(num_ugvs: int, num_tasks: int, num_afs: int,
                battery_m: float, svc_s: float, rch_s: float) -> tuple:
    return (num_ugvs, num_tasks, num_afs, battery_m, svc_s, rch_s)


def _assign_ids(fingerprints: list[tuple], prefix: str) -> dict[tuple, str]:
    unique = sorted(set(fingerprints))
    return {fp: f"{prefix}{i+1}" for i, fp in enumerate(unique)}


# ---------------------------------------------------------------------------
# Parse one run directory → row dict
# ---------------------------------------------------------------------------

def _parse_run(run_dir: Path, config_index: dict) -> dict[str, Any] | None:
    metrics_path = run_dir / "sim_metrics.json"
    instance_path = run_dir / "generated_instance.json"

    if not metrics_path.exists() or not instance_path.exists():
        return None

    with metrics_path.open() as fh:
        m = json.load(fh)
    with instance_path.open() as fh:
        inst = json.load(fh)

    parts = run_dir.name.split("_")
    if len(parts) < 3:
        return None

    # Run dirs: {planner}_{agent}_{YYYYMMDD}_{HHMMSS}
    # agent may be multi-word like "sarsa_lambda" → parse by exclusion of date suffix
    # Last two parts are date+time; everything before is planner_agent
    timestamp = "_".join(parts[-2:])
    planner_agent = "_".join(parts[:-2])

    # Match against known config index (longest matching planner prefix wins)
    planner, agent = "", ""
    for (p, a) in config_index:
        candidate = f"{p}_{a}"
        if planner_agent == candidate:
            planner, agent = p, a
            break
    if not planner:
        # Fallback: first token = planner, rest = agent
        planner = parts[0]
        agent = "_".join(parts[1:-2])

    cfg = config_index.get((planner, agent), {})
    if not cfg:
        cfg = load_config(CONFIG_DIR / "base.yaml")

    sim_cfg = cfg.get("simulation", {})
    ugv_cfg = sim_cfg.get("ugv", {})
    tasks_cfg = sim_cfg.get("tasks", {})
    psl_cfg = cfg.get("traffic", {}).get("psl", {})
    uav_cfg = cfg.get("uav_bms", {})

    net_xml_path = inst.get("generation_info", {}).get("net_xml_path", "")

    # Problem params
    node_types = inst.get("physical_node_types", [])
    num_tasks = node_types.count("customer")
    num_afs = node_types.count("afs")
    num_ugvs = ugv_cfg.get("fleet_size", len(m.get("per_vehicle", {})))
    battery_m = float(ugv_cfg.get("battery_range_m", 0))
    svc_s = float(ugv_cfg.get("customer_service_time_hours", 0)) * 3600
    rch_s = float(ugv_cfg.get("recharge_time_hours", 0)) * 3600

    uav_enabled = bool(uav_cfg.get("enabled", False))
    uav_dist = float(m.get("uav_distance_m", 0))
    uav_energy = float(m.get("uav_energy_kwh", 0))
    ugv_dist = float(m.get("total_distance_m", 0)) - uav_dist
    ugv_energy = float(m.get("total_energy_kwh", 0)) - uav_energy

    makespan_s = float(m["makespan_s"])
    makespan_h = makespan_s / 3600
    customers_served = int(m.get("customers_served", num_tasks))
    throughput = round(customers_served / makespan_h, 2) if makespan_h > 0 else None

    return {
        "run_id": run_dir.name,
        "timestamp": timestamp,
        "planner": planner,
        "agent": agent,
        "evrp_planner": PLANNER_DISPLAY.get(planner, planner),
        "trsp_agent": AGENT_DISPLAY.get(agent, agent),
        "uav_bms": "Yes" if uav_enabled else "No",
        # fingerprint keys
        "net_xml_path": net_xml_path,
        "traffic_fp": _traffic_fp(psl_cfg),
        "problem_fp": _problem_fp(num_ugvs, num_tasks, num_afs, battery_m, svc_s, rch_s),
        # metrics
        "makespan_s": makespan_s,
        "ugv_distance_m": ugv_dist,
        "uav_distance_m": uav_dist,
        "total_distance_m": float(m["total_distance_m"]),
        "ugv_energy_kwh": ugv_energy,
        "uav_energy_kwh": uav_energy,
        "total_energy_kwh": float(m["total_energy_kwh"]),
        "recharges": int(m["total_recharges"]),
        "throughput_tasks_per_h": throughput,
        "gap_pct": m.get("milp_gap_pct", ""),
        # for other tables
        "num_ugvs": num_ugvs,
        "num_tasks": num_tasks,
        "num_afs": num_afs,
        "battery_range_m": battery_m,
        "service_time_s": svc_s,
        "recharge_time_s": rch_s,
        "psl": psl_cfg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    config_index = _build_config_index()

    runs = sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []
    rows = [r for d in runs if d.is_dir() and (r := _parse_run(d, config_index))]

    if not rows:
        print("No completed runs found.")
        return

    # Assign IDs
    net_ids = _assign_ids([r["net_xml_path"] for r in rows], "N")
    traffic_ids = _assign_ids([r["traffic_fp"] for r in rows], "T")
    problem_ids = _assign_ids([r["problem_fp"] for r in rows], "P")

    for r in rows:
        r["network_id"] = net_ids[r["net_xml_path"]]
        r["traffic_id"] = traffic_ids[r["traffic_fp"]]
        r["problem_id"] = problem_ids[r["problem_fp"]]

    # ── networks.csv ──────────────────────────────────────────────────────────
    seen_nets: set[str] = set()
    net_rows = []
    for r in rows:
        nid = r["network_id"]
        if nid in seen_nets:
            continue
        seen_nets.add(nid)
        net_xml = ROOT / r["net_xml_path"]
        num_nodes, num_edges = _net_stats(net_xml)
        meta = NETWORK_ARCHETYPES.get(net_xml.name, {})
        net_rows.append({
            "network_id": nid,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "topology": meta.get("topology", ""),
            "connectivity_lambda": meta.get("connectivity_lambda", ""),
            "vehicle_type": meta.get("vehicle_type", ""),
        })
    net_rows.sort(key=lambda x: x["network_id"])

    _write_csv(RESULTS_DIR / "networks.csv", net_rows,
               ["network_id", "num_nodes", "num_edges",
                "topology", "connectivity_lambda", "vehicle_type"])

    # ── traffic.csv ───────────────────────────────────────────────────────────
    seen_tr: set[str] = set()
    traffic_rows = []
    for r in rows:
        tid = r["traffic_id"]
        if tid in seen_tr:
            continue
        seen_tr.add(tid)
        psl = r["psl"]
        traffic_rows.append({
            "traffic_id": tid,
            "K": int(psl.get("K", 1)),
            "v_min_kmh": float(psl.get("v_min", 0)),
            "v_max_kmh": float(psl.get("v_max", 0)),
            "step_kmh": float(psl.get("speed_step", 0)),
            "type": "PSL" if psl.get("enabled", True) else "None",
        })
    traffic_rows.sort(key=lambda x: x["traffic_id"])

    _write_csv(RESULTS_DIR / "traffic.csv", traffic_rows,
               ["traffic_id", "K", "v_min_kmh", "v_max_kmh", "step_kmh", "type"])

    # ── problem_instances.csv ─────────────────────────────────────────────────
    seen_pr: set[str] = set()
    problem_rows = []
    for r in rows:
        pid = r["problem_id"]
        if pid in seen_pr:
            continue
        seen_pr.add(pid)
        problem_rows.append({
            "problem_id": pid,
            "num_ugvs": r["num_ugvs"],
            "num_tasks": r["num_tasks"],
            "num_afs": r["num_afs"],
            "battery_range_m": int(r["battery_range_m"]),
            "service_time_s": int(r["service_time_s"]),
            "recharge_time_s": int(r["recharge_time_s"]),
        })
    problem_rows.sort(key=lambda x: x["problem_id"])

    _write_csv(RESULTS_DIR / "problem_instances.csv", problem_rows,
               ["problem_id", "num_ugvs", "num_tasks", "num_afs",
                "battery_range_m", "service_time_s", "recharge_time_s"])

    # ── ta_evrp_results.csv ───────────────────────────────────────────────────
    result_rows = sorted(rows, key=lambda r: (r["evrp_planner"], r["trsp_agent"], r["timestamp"]))
    _write_csv(
        RESULTS_DIR / "ta_evrp_results.csv",
        [{
            "evrp_planner":           r["evrp_planner"],
            "trsp_agent":             r["trsp_agent"],
            "problem":                r["problem_id"],
            "network":                r["network_id"],
            "traffic":                r["traffic_id"],
            "uav_bms":                r["uav_bms"],
            "makespan_s":             r["makespan_s"],
            "ugv_distance_m":         r["ugv_distance_m"],
            "uav_distance_m":         r["uav_distance_m"],
            "total_distance_m":       r["total_distance_m"],
            "ugv_energy_kwh":         r["ugv_energy_kwh"],
            "uav_energy_kwh":         r["uav_energy_kwh"],
            "total_energy_kwh":       r["total_energy_kwh"],
            "recharges":              r["recharges"],
            "throughput_tasks_per_h": r["throughput_tasks_per_h"],
            "gap_pct":                r["gap_pct"],
            "run_id":                 r["run_id"],
        } for r in result_rows],
        ["evrp_planner", "trsp_agent", "problem", "network", "traffic", "uav_bms",
         "makespan_s", "ugv_distance_m", "uav_distance_m", "total_distance_m",
         "ugv_energy_kwh", "uav_energy_kwh", "total_energy_kwh",
         "recharges", "throughput_tasks_per_h", "gap_pct", "run_id"],
    )

    print(f"Wrote {len(net_rows)} network(s), {len(traffic_rows)} traffic config(s), "
          f"{len(problem_rows)} problem instance(s), {len(result_rows)} result row(s) "
          f"to {RESULTS_DIR}/")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
