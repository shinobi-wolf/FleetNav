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
import sys
from datetime import datetime
from pathlib import Path

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
    metrics = sim.run(routes)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/3] Results:")
    _print_metrics(metrics)

    metrics_path = cfg["output"].get("save_metrics_json")
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(json.dumps(metrics, indent=2))
        print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
