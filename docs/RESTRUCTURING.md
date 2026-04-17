# EVRP Module Restructuring Plan

## Motivation

The current `dronesim/evrp/` module has two structural problems:

1. `evrp_exact.py` is a kitchen sink — it mixes instance generation, MILP model
   construction, plotting, and JSON I/O into a single 900-line file.
2. `Route`/`Stop` live under `planners/base.py`, but the simulator also imports
   them — a layering violation (planners should not own data types that the
   simulator depends on).
3. The traffic model is defined and owned by the simulator, making it impossible
   to use time-varying travel times in any planner without importing from the
   simulator.

The goal is a clean layered structure where each folder has a single
responsibility and nothing imports "upward" through the layers.

---

## Target Directory Layout

```
dronesim/evrp/
├── routes/
│   ├── __init__.py
│   └── base.py
├── instance/
│   ├── __init__.py
│   ├── builder.py
│   └── dummy_expansion.py
├── traffic/
│   ├── __init__.py
│   ├── base.py
│   └── segmented.py
├── io/
│   ├── __init__.py
│   ├── plot.py
│   └── instance_json.py
├── milp/
│   ├── __init__.py
│   └── gvrp.py
├── planners/
│   ├── __init__.py
│   ├── base.py
│   ├── gvrp_milp.py
│   ├── mtsp.py
│   └── evrp_heuristic.py
├── simulator/
│   ├── __init__.py
│   ├── evrp_sim.py
│   └── pygame_renderer.py
└── __init__.py
```

Dependency direction (nothing imports upward):

```
routes/   <──  instance/  <──  milp/
              traffic/         planners/
              routes/          simulator/
              io/
```

---

## File-by-File Specification

### `routes/base.py`
**Moved from**: `planners/base.py`

Contains exactly:
- `Stop` dataclass
- `Route` dataclass (with `customer_stops` and `afs_stops` properties)

No other code. No planner imports. This is the shared data contract between
planners and the simulator.

---

### `instance/builder.py`
**Moved from**: `evrp_exact.py`

Contains:
- `load_sumo_graph(net_path) -> dict`
- `build_problem_instance_from_config(config) -> tuple[ndarray, dict, dict, dict]`
- `build_distance_matrix_from_shortest_paths(...) -> ndarray`
- All private helpers: `_resolve_net_xml_path`, `_node_position`,
  `_resolve_depot_node_id`, `_determine_customer_count`,
  `_extract_pair_distance`, `_distance_unit_factor`,
  `_convert_consumption_rate`, `_convert_speed_from_mph`
- `build_distance_matrix(config)` compatibility wrapper

No MILP, no plotting, no Gurobi imports.

---

### `instance/dummy_expansion.py`
**Moved from**: `evrp_exact.py`

Contains:
- `expand_with_dummy_vertices(physical_dist, physical_meta, params) -> dict`

Separated from `builder.py` because dummy expansion is specific to the
Erdogan & Miller-Hooks MILP formulation — it is not a general instance
concept. Keeping it in its own file makes it clear that other planners
(heuristic, future CTVTT) do not need it.

---

### `traffic/base.py`
**New file**

Contains:
- `BaseTrafficModel` — abstract class with:
  - `update(sim_time_s: float) -> bool`
  - `commanded_speed_ms` property
  - `travel_time(dist_m: float) -> float`  ← new method, distance → time

The `travel_time` method is the hook for future time-dependent planners. The
base implementation uses `dist_m / commanded_speed_ms`.

---

### `traffic/segmented.py`
**Moved from**: `simulator/evrp_sim.py`

Contains:
- `SegmentedSpeedModel` (rename of `_TrafficModel` dataclass, implementing
  `BaseTrafficModel`)
- `build_traffic_model(config, free_flow_ms, max_time_s) -> SegmentedSpeedModel`
  (rename of `_build_traffic_model`)
- `_kmh_to_ms` and `_mph_to_ms` helpers (move here from simulator)

By living in `traffic/` rather than `simulator/`, this model can be
instantiated by a planner for travel-time estimation during planning, without
importing anything from the simulator layer.

---

### `io/plot.py`
**Moved from**: `evrp_exact.py`

Contains:
- `plot_routes(routes, coords, node_types, out_path) -> None`
- `build_plot_coordinates(physical_dist, instance, physical_coords) -> ndarray`
  (rename of `_build_plot_coordinates`)

---

### `io/instance_json.py`
**Moved from**: `evrp_exact.py`

Contains:
- `save_instance_json(config, physical_dist, physical_meta, generation_info)`
  (rename of `maybe_save_instance_json`)

---

### `milp/gvrp.py`
**Moved from**: `evrp_exact.py`

Contains:
- `build_gvrp_model(dist, params, physical_meta) -> gp.Model`
- `solve_and_extract(model, dist, params) -> dict`
- `run_single_vehicle_check(config, physical_dist, params, physical_meta)`
- `_status_name(status_code) -> str`

Imports from `instance/dummy_expansion.py` only. Gurobi import guard stays
here.

---

### `planners/base.py`
**Trimmed from current version**

Contains only:
- `BasePlanner` abstract class

`Route` and `Stop` are removed — they now live in `routes/base.py`. All
planners and the simulator import from there.

---

### `planners/gvrp_milp.py`
**Updated imports only**

```python
from ..routes.base import Route, Stop
from ..instance.builder import build_problem_instance_from_config
from ..milp.gvrp import build_gvrp_model, solve_and_extract
from ..planners.base import BasePlanner
```

No logic changes.

---

### `planners/mtsp.py`
**Updated imports only**

```python
from ..routes.base import Route, Stop
from ..instance.builder import build_problem_instance_from_config
from ..planners.base import BasePlanner
```

No logic changes.

---

### `planners/evrp_heuristic.py`
**Updated imports only**

```python
from ..routes.base import Route, Stop
from ..instance.builder import build_problem_instance_from_config
from ..planners.base import BasePlanner
```

No logic changes.

---

### `simulator/evrp_sim.py`
**Updated imports only**

```python
from ..routes.base import Route, Stop
from ..traffic.segmented import SegmentedSpeedModel, build_traffic_model
```

Remove the inline `_TrafficModel`, `_build_traffic_model`, `_kmh_to_ms`, and
`_mph_to_ms` definitions — they now come from `traffic/segmented.py`.

No logic changes.

---

### `run_sim.py`
**Updated imports**

```python
from dronesim.evrp.instance.builder import build_problem_instance_from_config
from dronesim.evrp.io.plot import plot_routes, build_plot_coordinates
from dronesim.evrp.io.instance_json import save_instance_json
from dronesim.evrp.planners import get_planner
from dronesim.evrp.planners.gvrp_milp import GVRPMILPPlanner
```

The `_plot_routes` helper in `run_sim.py` that calls `_build_plot_coordinates`
and `plot_routes` from `evrp_exact` is updated to call the new `io/` paths.

---

## Files Deleted

| File | Reason |
|------|--------|
| `dronesim/evrp/evrp_exact.py` | Fully dismantled; all contents redistributed |

---

## `__init__.py` Files

Each new folder gets a minimal `__init__.py` that re-exports the public API
of that module. Example for `routes/`:

```python
from .base import Route, Stop
```

Example for `traffic/`:

```python
from .base import BaseTrafficModel
from .segmented import SegmentedSpeedModel, build_traffic_model
```

This allows callers to write `from dronesim.evrp.routes import Route, Stop`
rather than spelling out the submodule path.

---

## Config Changes

None required. All config keys (`traffic.*`, `gvrp.*`, etc.) remain identical.
The new `traffic/segmented.py` reads the same config dict structure as the
current `_build_traffic_model`.

---

## Execution Order

Do these steps in order. Each step should leave the test suite green before
moving to the next.

1. Create `routes/base.py` with `Route` and `Stop`. Update all imports in
   `planners/base.py`, `gvrp_milp.py`, `mtsp.py`, `evrp_heuristic.py`, and
   `simulator/evrp_sim.py`. Delete `Route`/`Stop` from `planners/base.py`.

2. Create `traffic/base.py` and `traffic/segmented.py`. Move `_TrafficModel`,
   `_build_traffic_model`, `_kmh_to_ms`, `_mph_to_ms` from `evrp_sim.py`.
   Update `evrp_sim.py` imports.

3. Create `instance/builder.py`. Move all instance-generation functions from
   `evrp_exact.py`. Update `planners/gvrp_milp.py`, `mtsp.py`,
   `evrp_heuristic.py` imports.

4. Create `instance/dummy_expansion.py`. Move `_expand_with_dummy_vertices`
   from `evrp_exact.py`.

5. Create `milp/gvrp.py`. Move `build_gvrp_model`, `solve_and_extract`,
   `run_single_vehicle_check` from `evrp_exact.py`. Update
   `planners/gvrp_milp.py` imports.

6. Create `io/plot.py` and `io/instance_json.py`. Move plotting and JSON
   functions from `evrp_exact.py`. Update `run_sim.py` imports.

7. Delete `evrp_exact.py`. Run smoke test to confirm nothing is broken.

---

## Smoke Test After Each Step

```bash
python run_sim.py -c config/evrp/milp.yaml --no-sim --no-plot
python scripts/smoke_test_lmd.py --config config/evrp/base.yaml --steps 8
```
