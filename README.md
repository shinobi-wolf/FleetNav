# DroneSim: Reproducing IROS 2026 Results

This repository contains the simulation code used for the attached paper.
The primary goal is to reproduce the paper's experimental results.

## Reproducibility Pipeline

The intended workflow is:

1. Run batch simulations with `param_simulation.py`.
2. Generate CSV outputs in `runs/iros_26/` (mainly `results_nrlp.csv`).
3. Visualize and compare outcomes in `results.ipynb`.

## Project Structure

```text
DroneSim/
├── param_simulation.py      # Main batch runner for experiments
├── server.py                # Runs one simulation and appends one CSV row
├── results.ipynb            # Main visualization notebook for produced CSV files
├── analysis.ipynb           # Additional exploratory analysis
├── main.py                  # Interactive entry point (pygame frontend)
├── console.py               # Visualization frontend logic
├── requirements.txt         # Pinned direct dependencies from conda env iros_26
├── config/
│   └── world1.yaml
├── runs/
│   └── iros_26/
│       ├── results.csv
│       └── results_nrlp.csv
└── world_1/
    ├── world.py
    ├── planner.py
    └── vehicles/
```

## Environment Setup (conda)

Use the same conda environment used for this project:

```bash
conda activate iros_26
pip install -r requirements.txt
```

If `runs/iros_26` does not exist in a fresh clone, create it:

```bash
mkdir -p runs/iros_26
```

## Run Experiments

Main command:

```bash
python param_simulation.py
```

This script sweeps combinations of:

- number of EVs
- number of tasks
- EV range
- maze loop percentage
- routing algorithm (`mtsp`, `evrp`)
- BMS on/off

and writes rows to:

- `runs/iros_26/results_nrlp.csv`

## Visualize Results

Open and run:

- `results.ipynb`

The notebook reads from:

- `runs/iros_26/results.csv`
- `runs/iros_26/results_nrlp.csv`

and computes metrics such as makespan (`timesteps`) and an energy.

## Note on Existing CSV Files

To reduce iteration volume during testing, the parameter sweep was split into two separate runs.
The CSV files currently present in `runs/iros_26/` are the outputs from those split runs.

For full reproduction, rerun `param_simulation.py` with the intended sweep ranges.

## Practical Runtime Notes

- `param_simulation.py` uses parallel workers (`ProcessPoolExecutor`) with a fixed window size.
- Full sweeps can take a long time depending on CPU resources.
- For quick sanity checks, temporarily reduce parameter ranges in `param_simulation.py`.

## Output Schema

Each simulation row written by `server.py` has columns:

- `algo`
- `num_evs`
- `num_tasks`
- `ev_range`
- `maze_loop_percentage`
- `maze_size`
- `bms`
- `timesteps`
- `num_tasks_finished`
- `ev_distance`
- `drone_distance`
