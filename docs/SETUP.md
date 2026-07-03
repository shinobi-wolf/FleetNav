# Environment Setup — Fresh Ubuntu Machine

How to stand up DroneSim (EVRP + TRSP) on a new Ubuntu box, e.g. to move
training off a laptop and onto a dedicated machine.

## Quick start

```bash
git clone <your-fork-or-remote-url> DroneSim
cd DroneSim
scripts/setup_ubuntu.sh
conda activate dronesim
```

That's it for everything except Gurobi's license (below). The script is
idempotent — re-run it any time to pick up requirements.txt changes.

Options:

```bash
scripts/setup_ubuntu.sh --env-name dronesim_train   # custom conda env name
scripts/setup_ubuntu.sh --cuda cu121                # force a CUDA build (default: auto-detect)
scripts/setup_ubuntu.sh --cuda cpu                  # force CPU-only torch
scripts/setup_ubuntu.sh --notebooks                 # also install jupyter/pandas for notebooks/
```

## What the script does

1. **System packages** (`apt`, needs `sudo`): `build-essential`, `cmake`,
   `git`, `curl`, `wget`, `libgl1`, `libglib2.0-0`. Everything else is
   installed from pip wheels — no PPA, no compiling SUMO or Gurobi from
   source.
2. **Miniconda**, if `conda` isn't already on the machine.
3. A conda env (`dronesim` by default) with **Python 3.10** — pinned
   because `routingblocks` only ships a `cp310` wheel.
4. **torch**, CPU or CUDA build depending on whether `nvidia-smi` is
   present (override with `--cuda`).
5. Everything in `environment/requirements.txt`, pinned to versions
   verified against this codebase.
6. **SUMO**, via the `eclipse-sumo` pip wheel. This is a fully
   self-contained SUMO build (binaries + `tools/` + `data/` under
   `site-packages/sumo/`) — no apt package or PPA needed. The script wires
   `SUMO_HOME` to point at it automatically whenever the conda env is
   activated (via `etc/conda/activate.d/`).
7. A smoke test (`scripts/smoke_test_env.py`) that imports everything,
   checks the `sumo` binary, checks CUDA, and checks for an active Gurobi
   license (non-fatal if missing).

## Gurobi license (manual step)

`gurobipy` is installed by the script, but Gurobi requires a license tied
to your account/machine — this can't be scripted blindly. The MILP planner
(`evrp.planner: milp` / `td_milp`) needs it; the heuristic (`RoutingBlocks`
ALNS) and `mtsp` planners do **not**.

1. Get a license from https://www.gurobi.com/ (academic licenses are free
   for `.edu` accounts).
2. Depending on license type:
   - **Named-user / node-locked**: run `grbgetkey <your-key>` — this tool
     ships with the full Gurobi distribution, not the pip package. Either
     download the distribution tarball from Gurobi and run it once, or use
     the WLS flow below instead.
   - **WLS (Web License Service)**, recommended for a headless training
     box you don't control physically: download the `gurobi.lic` from your
     Gurobi account portal directly (no `grbgetkey` needed) and place it at
     `~/gurobi.lic`, or point `GRB_LICENSE_FILE` at it:
     ```bash
     export GRB_LICENSE_FILE=/path/to/gurobi.lic
     ```
3. Verify: `python scripts/smoke_test_env.py` should print
   `ok  Gurobi license active`.

Do not commit `gurobi.lic` to the repo — it's tied to your account/machine.

## Data and trained policies

`data/` (SUMO networks) and `runs/trsp/` (trained policies) are tracked in
git, so a plain `git clone` brings them over — no separate transfer step.
`runs/ta_evrp/` (simulation outputs) is gitignored and regenerated per run.

## Verifying the move

```bash
conda activate dronesim
python scripts/smoke_test_env.py
python infer_ta_evrp.py -c config/ta_evrp/heuristic.yaml   # no Gurobi/license needed
python train_trsp.py -c config/trsp/train.yaml             # short run to confirm the RL loop starts
```

## Troubleshooting

- **`sumo: command not found` after activating the env** — the
  `activate.d` hook only takes effect on `conda activate`, not in a shell
  that was already open when setup ran. Open a new shell or re-run
  `conda activate dronesim`.
- **`routingblocks` fails to install** — it only publishes wheels for
  CPython 3.10 on manylinux x86_64. Confirm `python --version` inside the
  env is 3.10.x and the machine is x86_64.
- **torch can't see the GPU** — check `nvidia-smi` works outside the conda
  env first (driver issue, not a Python issue), then re-run
  `scripts/setup_ubuntu.sh --cuda cu128` (or whichever CUDA version matches
  your driver) inside the existing env.
