"""Verify the DroneSim environment is set up correctly on this machine.

Run after scripts/setup_ubuntu.sh (inside the activated conda env):
    python scripts/smoke_test_env.py

Checks imports, the SUMO binary, torch/CUDA, and (non-fatally) whether a
Gurobi license is active. Exits non-zero only on hard failures (missing
packages / SUMO binary) — a missing Gurobi license is reported but does not
fail the run, since the heuristic/mTSP planners work without one.
"""

import shutil
import subprocess
import sys

HARD_FAILURES = []
WARNINGS = []

print("Checking core imports...")
for mod in [
    "numpy", "scipy", "sklearn", "networkx", "matplotlib", "yaml", "tqdm",
    "gymnasium", "torch", "tyro", "wandb",
    "shapely", "geopandas", "osmnx", "SumoNetVis",
    "sumolib", "traci", "libsumo",
    "gurobipy", "routingblocks", "pygame",
]:
    try:
        __import__(mod)
        print(f"  ok  {mod}")
    except ImportError as e:
        HARD_FAILURES.append(f"import {mod} failed: {e}")
        print(f"  FAIL {mod}: {e}")

print("\nChecking SUMO binary...")
sumo_bin = shutil.which("sumo")
if sumo_bin is None:
    HARD_FAILURES.append("`sumo` binary not found on PATH")
    print("  FAIL sumo not on PATH")
else:
    out = subprocess.run([sumo_bin, "--version"], capture_output=True, text=True)
    print(f"  ok  {sumo_bin} -> {out.stdout.splitlines()[0] if out.stdout else out.stderr.splitlines()[0]}")

print("\nChecking torch / CUDA...")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  torch {torch.__version__}, CUDA available: {cuda_ok}")
    if not cuda_ok:
        WARNINGS.append("CUDA not available to torch — training will run on CPU")
except ImportError:
    pass

print("\nChecking Gurobi license...")
try:
    import gurobipy as gp
    try:
        m = gp.Model()
        m.dispose()
        print("  ok  Gurobi license active")
    except gp.GurobiError as e:
        WARNINGS.append(f"Gurobi license not active ({e}) — milp/td_milp planners unavailable")
        print(f"  WARN no active Gurobi license: {e}")
except ImportError:
    pass

print("\n" + "=" * 60)
if WARNINGS:
    print("Warnings:")
    for w in WARNINGS:
        print(f"  - {w}")
if HARD_FAILURES:
    print("Failures:")
    for f in HARD_FAILURES:
        print(f"  - {f}")
    sys.exit(1)

print("Environment looks good.")
