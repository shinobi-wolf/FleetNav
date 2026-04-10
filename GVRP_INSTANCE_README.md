# Config-Driven SUMO G-VRP (`evrp_exact.py`)

This runner builds a G-VRP problem instance from a SUMO network and then solves it with Gurobi.

## What Changed

- The old hardcoded Figure-1 node/distance table is replaced with a generated instance.
- Instance generation now uses:
  - `net.net_path` to load SUMO graph
  - `net.warehouse` to choose depot (`center`, `random`, or explicit node id)
  - task/fleet/gvrp sections to sample customers and AFS nodes
- Distances are computed from all-pairs Dijkstra shortest paths over the SUMO graph.

## Config

Use `config/gvrp.yaml`.

Key fields:

- `net.net_path`: SUMO `.sumocfg` or `.net.xml` path
- `net.warehouse`: `center`, `random`, or explicit node id
- `gvrp.customer_sampling`: `density` or `count`
- `gvrp.customer_count` / `tasks.task_density`
- `gvrp.afs_count`
- `gvrp.fuel_capacity`, `gvrp.consumption_rate`, `gvrp.speed_mph`, `gvrp.max_duration_hours`
- `gvrp.afs_dummy_copies`, `gvrp.depot_dummy_copies`
- `solver.time_limit_seconds`, `solver.output_flag`, `solver.mip_gap`

## Run

```bash
python evrp_exact.py -c config/gvrp.yaml
```

Disable plotting:

```bash
python evrp_exact.py -c config/gvrp.yaml --no-plot
```

## Outputs

- Console: generated node sets + solve status + routes + fuel + arrival times
- Plot: `gvrp_routes.png` when non-GUI backend is active
- Optional JSON instance dump if `output.save_instance_json` is configured
