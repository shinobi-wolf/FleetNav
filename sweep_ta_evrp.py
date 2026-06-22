"""sweep_ta_evrp.py — Exhaustive parameter sweep for TA-EVRP.

Reads a superset config (e.g. config/ta_evrp/sweep.yaml) where any leaf
value that is a list defines a sweep axis. Computes the Cartesian product of
all axes and runs one full TA-EVRP simulation per combination in quiet mode.
Results are appended incrementally to results/ta_evrp_results.csv.

Parallelises via ProcessPoolExecutor: SUMO/TraCI requires separate OS
processes (threads share file descriptors and break TraCI connections).
Each worker gets its own SUMO instance on an independent socket.

Usage
-----
  python sweep_ta_evrp.py                                     # default sweep.yaml
  python sweep_ta_evrp.py -c config/ta_evrp/sweep.yaml
  python sweep_ta_evrp.py --workers 8                         # parallel processes
  python sweep_ta_evrp.py --dry-run                           # print combos, exit
  python sweep_ta_evrp.py --no-sim                            # plan only, skip SUMO
  python sweep_ta_evrp.py --output results/custom_sweep.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dronesim.ta_evrp import Instance, GlobalPlanner, LocalPlanner, Simulator

# ---------------------------------------------------------------------------
# Axis — unit of sweep variation (single dotpath or zip group)
# ---------------------------------------------------------------------------

@dataclass
class Axis:
    dotpaths: list[str]
    values: list  # scalars for single axes, tuples for zip axes

    @property
    def is_zip(self) -> bool:
        return len(self.dotpaths) > 1


def _combo_to_row(all_axes: list[Axis], combo: tuple) -> dict[str, Any]:
    """Expand a combo tuple to a flat {col_name: value} dict."""
    row: dict[str, Any] = {}
    for ax, val in zip(all_axes, combo):
        if ax.is_zip:
            for dp, v in zip(ax.dotpaths, val):
                row[_dotpath_to_col(dp)] = v
        else:
            row[_dotpath_to_col(ax.dotpaths[0])] = val
    return row


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_CONFIG_DEFAULT = "config/ta_evrp/sweep.yaml"
RESULTS_CSV_DEFAULT = "results/ta_evrp_results.csv"
RUNS_DIR = Path("runs/ta_evrp")
RUNS_TRSP_DIR = Path("runs/trsp")

# Agents that require a trained policy loaded from runs/trsp/
RL_AGENTS = frozenset({"sarsa", "sarsa_lambda", "dqn", "ppo", "a2c"})

METRIC_COLUMNS = [
    "makespan_s",
    "makespan_h",
    "total_distance_m",
    "total_distance_km",
    "total_recharges",
    "customers_served",
    "avg_inter_customer_gap_s",
]

META_COLUMNS = ["run_id", "timestamp", "status", "error"]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_base_config(base_yaml: str) -> dict:
    raw = _load_yaml(base_yaml)
    raw.pop("base", None)
    return raw


# ---------------------------------------------------------------------------
# Sweep axis extraction
# ---------------------------------------------------------------------------

def _extract_axes(d: dict, prefix: str = "") -> dict[str, list]:
    """Collect dotpaths whose leaf is a list — these are sweep axes."""
    axes: dict[str, list] = {}
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, list):
            axes[path] = val
        elif isinstance(val, dict):
            axes.update(_extract_axes(val, path))
    return axes


def _extract_fixed(d: dict, prefix: str = "") -> dict[str, Any]:
    """Collect dotpaths whose leaf is a scalar — these are fixed overrides."""
    fixed: dict[str, Any] = {}
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, list):
            pass
        elif isinstance(val, dict):
            fixed.update(_extract_fixed(val, path))
        else:
            fixed[path] = val
    return fixed


# ---------------------------------------------------------------------------
# Dotpath helpers
# ---------------------------------------------------------------------------

def _set_dotpath(d: dict, dotpath: str, value: Any) -> None:
    parts = dotpath.split(".")
    node = d
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _dotpath_to_col(dotpath: str) -> str:
    """Convert 'a.b.c' to 'a_b_c' for use as a CSV column name."""
    return dotpath.replace(".", "_")


# ---------------------------------------------------------------------------
# Per-combo config construction
# ---------------------------------------------------------------------------

def _build_run_config(
    base_cfg: dict,
    fixed: dict[str, Any],
    combo: dict[str, Any],
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for dotpath, val in fixed.items():
        _set_dotpath(cfg, dotpath, val)
    for dotpath, val in combo.items():
        _set_dotpath(cfg, dotpath, val)
    # Always force quiet, no-plot, no-instance-json in sweep mode
    _set_dotpath(cfg, "sumo.render_mode", "quiet")
    _set_dotpath(cfg, "output.plot", False)
    _set_dotpath(cfg, "output.save_instance_json", False)
    return cfg


# ---------------------------------------------------------------------------
# RL policy auto-discovery
# ---------------------------------------------------------------------------

def _discover_policies(agent: str, net_path: str, k: int | None = None) -> list[str]:
    """Return policy_ids from runs/trsp/ matching algo, network, and optionally K.

    First tries a strict match (algo + net stem + K); if that yields nothing and K
    was provided, falls back to matching on algo + net stem only.
    """
    if agent not in RL_AGENTS or not RUNS_TRSP_DIR.exists():
        return []
    net_stem = Path(net_path).stem

    def _scan(require_k: bool) -> list[str]:
        matches = []
        for policy_dir in sorted(RUNS_TRSP_DIR.iterdir()):
            cfg_path = policy_dir / "config.yaml"
            if not cfg_path.exists():
                continue
            try:
                pcfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                if (pcfg.get("agent") or {}).get("algo", "") != agent:
                    continue
                if Path((pcfg.get("net") or {}).get("net_path", "")).stem != net_stem:
                    continue
                if require_k and k is not None:
                    if int((pcfg.get("traffic") or {}).get("K", -1)) != int(k):
                        continue
                matches.append(policy_dir.name)
            except Exception:
                pass
        return matches

    found = _scan(require_k=True)
    if not found and k is not None:
        found = _scan(require_k=False)   # K-relaxed fallback
    return found


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _combo_fingerprint(combo_row: dict[str, Any]) -> str:
    """Stable string key for a combo — used to detect already-done runs."""
    return "|".join(f"{k}={combo_row[k]}" for k in sorted(combo_row))


def _load_done_keys(csv_path: str, axis_keys: list[str]) -> set[str]:
    """Return fingerprints of combos already recorded in the CSV."""
    p = Path(csv_path)
    if not p.exists():
        return set()
    done: set[str] = set()
    try:
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("status") not in ("ok", "planned", "error"):
                    continue
                key = _combo_fingerprint(
                    {_dotpath_to_col(k): row.get(_dotpath_to_col(k), "") for k in axis_keys}
                )
                done.add(key)
    except Exception:
        pass
    return done


def _append_csv_row(
    csv_path: str,
    row: dict,
    fieldnames: list[str],
    *,
    write_header: bool,
) -> None:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with p.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fieldnames})


# ---------------------------------------------------------------------------
# Worker — must be a top-level function for multiprocessing pickling
# ---------------------------------------------------------------------------

def _run_single(args: tuple) -> dict:  # noqa: C901
    run_cfg, sweep_row, _run_idx, _total, no_sim = args
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        instance = Instance(run_cfg)
        run_id = f"{instance.planner_name}_{instance.agent_name}_{timestamp}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        planner = GlobalPlanner()
        local_planner = LocalPlanner(instance.policy_id) if instance.policy_id else None
        routes = planner.plan(instance)

        if no_sim:
            return {
                **sweep_row,
                "run_id": run_id,
                "timestamp": timestamp,
                "status": "planned",
                **{col: "" for col in METRIC_COLUMNS},
                "error": "",
            }

        sim = Simulator(instance, local_planner)
        metrics = sim.run(routes)
        metrics.pop("traffic_log", None)
        metrics.pop("traffic_profile", None)

        return {
            **sweep_row,
            "run_id": run_id,
            "timestamp": timestamp,
            "status": "ok",
            "makespan_s": metrics.get("makespan_s", ""),
            "makespan_h": metrics.get("makespan_h", ""),
            "total_distance_m": metrics.get("total_distance_m", ""),
            "total_distance_km": metrics.get("total_distance_km", ""),
            "total_recharges": metrics.get("total_recharges", ""),
            "customers_served": metrics.get("customers_served", ""),
            "avg_inter_customer_gap_s": metrics.get("avg_inter_customer_gap_s", ""),
            "error": "",
        }

    except Exception:
        tb = traceback.format_exc()
        brief = tb.splitlines()[-1] if tb.splitlines() else "unknown error"
        return {
            **sweep_row,
            "run_id": "",
            "timestamp": timestamp,
            "status": "error",
            **{col: "" for col in METRIC_COLUMNS},
            "error": brief,
        }


# ---------------------------------------------------------------------------
# Dry-run display
# ---------------------------------------------------------------------------

def _print_combo_table(
    col_names: list[str],
    all_combos: list[tuple],
    n_skip: int,
) -> None:
    col_widths = [max(len(cn), 4) for cn in col_names]
    for combo in all_combos:
        for i, v in enumerate(combo):
            col_widths[i] = max(col_widths[i], len(str(v)))

    total_w = sum(col_widths) + 3 * (len(col_widths) - 1)
    bar = "─" * (total_w + 2)

    print(f"\n{bar}")
    print(f"  Axes         : {len(col_names)}")
    print(f"  Combinations : {len(all_combos)}")
    if n_skip:
        print(f"  Already done : {n_skip}  (will be skipped)")
    print(f"  To run       : {len(all_combos) - n_skip}")
    print(bar)

    header_row = "   ".join(cn.ljust(w) for cn, w in zip(col_names, col_widths))
    sep_row = "   ".join("-" * w for w in col_widths)
    print(f"  {header_row}")
    print(f"  {sep_row}")
    for combo in all_combos:
        row_str = "   ".join(str(v).ljust(w) for v, w in zip(combo, col_widths))
        print(f"  {row_str}")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TA-EVRP parameter sweep → results/ta_evrp_results.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", default=SWEEP_CONFIG_DEFAULT,
        help="Sweep superset config YAML",
    )
    parser.add_argument(
        "--output", default=RESULTS_CSV_DEFAULT,
        help="Output CSV path (rows appended incrementally)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print combo table and exit without running any simulations",
    )
    parser.add_argument(
        "--no-sim", action="store_true",
        help="Plan only; skip SUMO simulation (fast check of planner coverage)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("sweep")

    # ── Load sweep config ────────────────────────────────────────────────────
    sweep_raw = _load_yaml(args.config)
    base_path = sweep_raw.pop("base", "config/ta_evrp/base.yaml")
    base_cfg = _load_base_config(base_path)

    zip_groups = sweep_raw.pop("_zip_", []) or []
    axes = _extract_axes(sweep_raw)
    fixed = _extract_fixed(sweep_raw)

    all_axes: list[Axis] = [Axis([k], v) for k, v in axes.items()]
    for group in zip_groups:
        all_axes.append(Axis(group["keys"], [tuple(row) for row in group["values"]]))

    if not all_axes:
        log.error("No sweep axes found in %s — nothing to sweep.", args.config)
        sys.exit(1)

    all_dotpaths = [dp for ax in all_axes for dp in ax.dotpaths]
    all_combos = list(itertools.product(*[ax.values for ax in all_axes]))

    total_runs = len(all_combos)
    log.info("Sweep config  : %s", args.config)
    log.info("Axes          : %d  →  %d combinations", len(all_axes), total_runs)
    for ax in all_axes:
        if ax.is_zip:
            log.info("  zip(%-36s) %d rows", ", ".join(ax.dotpaths), len(ax.values))
        else:
            log.info("  %-40s %s", ax.dotpaths[0], ax.values)

    # ── Skip already-done ────────────────────────────────────────────────────
    done_keys = _load_done_keys(args.output, all_dotpaths)
    pending: list[tuple] = [
        c for c in all_combos
        if _combo_fingerprint(_combo_to_row(all_axes, c)) not in done_keys
    ]
    n_skip = total_runs - len(pending)
    if n_skip:
        log.info("Already done  : %d (skipping)", n_skip)
    log.info("To run        : %d", len(pending))

    if args.dry_run:
        col_names = [_dotpath_to_col(dp) for ax in all_axes for dp in ax.dotpaths]
        flat_combos = [tuple(_combo_to_row(all_axes, c).values()) for c in all_combos]
        _print_combo_table(col_names, flat_combos, n_skip)
        return

    if not pending:
        log.info("All combinations already recorded in %s.", args.output)
        return

    # ── CSV column schema ────────────────────────────────────────────────────
    sweep_col_names = [_dotpath_to_col(dp) for ax in all_axes for dp in ax.dotpaths]
    all_columns = sweep_col_names + METRIC_COLUMNS + META_COLUMNS

    csv_is_new = not Path(args.output).exists() or Path(args.output).stat().st_size == 0
    write_header_next = csv_is_new

    # ── Build work items (auto-expand RL combos with discovered policy_ids) ──
    work_items: list[tuple] = []
    for combo in pending:
        combo_dict: dict[str, Any] = {}
        for ax, val in zip(all_axes, combo):
            if ax.is_zip:
                for dp, v in zip(ax.dotpaths, val):
                    combo_dict[dp] = v
            else:
                combo_dict[ax.dotpaths[0]] = val
        cfg = _build_run_config(base_cfg, fixed, combo_dict)

        agent     = (cfg.get("trsp") or {}).get("agent", "dijkstra")
        policy_id = (cfg.get("trsp") or {}).get("policy_id")

        if agent in RL_AGENTS and not policy_id:
            net_path = (cfg.get("network") or {}).get("net_path", "")
            k_val    = (cfg.get("traffic") or {}).get("psl", {}).get("K")
            policies = _discover_policies(agent, net_path, k=k_val)
            if not policies:
                log.warning(
                    "  skip  no trained '%s' policy for net=%s K=%s",
                    agent, Path(net_path).name, k_val,
                )
                continue
            for pid in policies:
                cfg_copy = copy.deepcopy(cfg)
                _set_dotpath(cfg_copy, "trsp.policy_id", pid)
                sweep_row = _combo_to_row(all_axes, combo)
                sweep_row["trsp_policy_id"] = pid
                work_items.append((cfg_copy, sweep_row, 0, 0, args.no_sim))
        else:
            sweep_row = _combo_to_row(all_axes, combo)
            work_items.append((cfg, sweep_row, 0, 0, args.no_sim))

    # ── Execute ──────────────────────────────────────────────────────────────
    n_workers = args.workers if args.workers else min(4, os.cpu_count() or 1)
    n_workers = min(n_workers, len(work_items))

    log.info("Workers       : %d", n_workers)
    log.info("Output CSV    : %s", args.output)
    if args.no_sim:
        log.info("Mode          : plan-only (--no-sim)")
    log.info("")

    # Optional tqdm progress bar
    try:
        from tqdm import tqdm as _tqdm
        progress = _tqdm(total=len(work_items), unit="run", dynamic_ncols=True, leave=True)
    except ImportError:
        progress = None

    n_ok = n_err = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(_run_single, w): w for w in work_items}

        for fut in as_completed(future_map):
            row = fut.result()
            status = row.get("status", "?")
            run_id = row.get("run_id") or "—"

            if status == "ok":
                n_ok += 1
                log.info(
                    "  ok   %s  makespan=%.1f s  dist=%.1f m  recharges=%s",
                    run_id,
                    float(row.get("makespan_s") or 0),
                    float(row.get("total_distance_m") or 0),
                    row.get("total_recharges", "?"),
                )
            elif status == "planned":
                n_ok += 1
                log.info("  ok   %s  (plan only)", run_id)
            else:
                n_err += 1
                log.warning("  err  %s", row.get("error", "unknown"))

            _append_csv_row(args.output, row, all_columns, write_header=write_header_next)
            write_header_next = False

            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    log.info("")
    log.info("Sweep complete: %d ok, %d errors.", n_ok, n_err)
    log.info("Results       : %s", args.output)


if __name__ == "__main__":
    main()
