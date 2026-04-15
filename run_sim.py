"""EVRP entry point: plan → simulate → report metrics.

Usage
-----
  python run_sim.py -c config/evrp/milp.yaml      [--planner milp]
  python run_sim.py -c config/evrp/mtsp.yaml      --planner mtsp
  python run_sim.py -c config/evrp/heuristic.yaml --planner heuristic
  python run_sim.py -c config/evrp/milp.yaml      --no-sim

Flags
-----
  --no-sim    Run the planner only; skip SUMO simulation.
  --no-plot   Suppress the route plot even when config enables it.
  --planner   Planner name from registry (default: milp).

Config inheritance
------------------
  A config file may declare `base: path/to/base.yaml`.
  The base is loaded first; the child config deep-merges on top.
  Nested dicts are merged recursively; scalar values are overridden.

Outputs
-------
  All result files are written under runs/evrp/:
    gvrp_routes.png          route visualisation
    sim_metrics.json         simulation metrics
    generated_gvrp_instance.json  problem instance (if enabled in config)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend so plots always save to PNG

import yaml

# ── make dronesim importable without an installed package ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dronesim.evrp.evrp_exact import _build_plot_coordinates, plot_routes, maybe_save_instance_json
from dronesim.evrp.planners import get_planner
from dronesim.evrp.planners.gvrp_milp import GVRPMILPPlanner

RUNS_DIR = Path("runs/evrp")


# ---------------------------------------------------------------------------
# Config loader with base-inheritance and deep-merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: override merged on top of base, recursively for dicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str) -> dict:
    """Load a YAML config, merging from `base:` if declared."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        child = yaml.safe_load(fh) or {}

    base_ref = child.pop("base", None)
    if base_ref is None:
        return child

    # Resolve base path relative to CWD (same convention as -c flag)
    base_path = Path(base_ref)
    with base_path.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    base.pop("base", None)  # bases don't chain (keep it simple)

    return _deep_merge(base, child)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_routes(planner, routes, out_path: str) -> None:
    """Save route plot for any planner that stored physical instance data."""
    if not hasattr(planner, "_last_generation_info"):
        print("  [plot] planner did not store instance data; skipping plot.")
        return

    gen_info = planner._last_generation_info
    physical_coords = gen_info.get("physical_coords")

    if isinstance(planner, GVRPMILPPlanner):
        coords = _build_plot_coordinates(
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
    parser = argparse.ArgumentParser(description="EVRP planner + SUMO simulator")
    parser.add_argument("-c", "--config", default="config/evrp/milp.yaml")
    parser.add_argument("--no-sim", action="store_true", help="Skip SUMO simulation")
    parser.add_argument("--no-plot", action="store_true", help="Suppress route plot")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    # Redirect output paths in config to runs/evrp/
    config.setdefault("output", {})
    if config["output"].get("save_instance_json"):
        config["output"]["save_instance_json"] = str(RUNS_DIR / Path(config["output"]["save_instance_json"]).name)
    if config["output"].get("save_metrics_json"):
        config["output"]["save_metrics_json"] = str(RUNS_DIR / Path(config["output"]["save_metrics_json"]).name)

    # ── Plan ─────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Planning with '{config['planner']}' planner...")
    planner = get_planner(config['planner'])
    routes = planner.plan(config)
    _print_routes(routes)

    maybe_save_instance_json(
        config,
        planner._last_physical_dist,
        planner._last_physical_meta,
        planner._last_generation_info,
    )

    # ── Optional plot ─────────────────────────────────────────────────────────
    do_plot = bool(config.get("gvrp", {}).get("plot", True)) and not args.no_plot
    if do_plot:
        plot_path = str(RUNS_DIR / "gvrp_routes.png")
        _plot_routes(planner, routes, out_path=plot_path)

    # ── Simulate ─────────────────────────────────────────────────────────────
    if args.no_sim:
        print("\n[2/3] Simulation skipped (--no-sim).")
        return

    print("\n[2/3] Starting SUMO simulation...")
    from dronesim.evrp.simulator import EVRPSimulator
    sim = EVRPSimulator(config)
    metrics = sim.run(routes)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/3] Results:")
    _print_metrics(metrics)

    metrics_path = config["output"].get("save_metrics_json", str(RUNS_DIR / "sim_metrics.json"))
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
