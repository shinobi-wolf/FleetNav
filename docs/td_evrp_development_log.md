# TD-EVRP Development Log

This document traces the evolution of the TD-EVRP planner from commit `f1ea83c` (td_evrp v0)
through the work done in the current session, and explains why the planner is not yet producing
results better than the heuristic in simulation.

---

## Commit History Since td_evrp v0

### `f1ea83c` — td_evrp v0 (baseline)

Placeholder / scaffolding commit. The `EVRPTDMILPPlanner` class existed but the MILP model
was a stub derived from the earlier Lera-Romero formulation. Config file was
`config/evrp/lera_romero.yaml`.

---

### `e4247be` — td_evrp started with claude code (2026-04-20)

**What changed:**
- `config/evrp/lera_romero.yaml` deleted; replaced by `config/evrp/td_evrp_milp.yaml`.
- `dronesim/evrp/milp/td_evrp.py` rewritten from scratch (+695 / -253 lines) to implement
  the Lu et al. (2020) ILP formulation:
  - Five departure intervals M={1..5} via Ichoua (2003) FIFO boundary crossings at T1, T2.
  - Discrete free-flow speed levels R = linspace(vc, vf, k).
  - Per-arc travel-time coefficients A_T, B_T (linear in departure time).
  - Decision variables: x (arc binary), z (arc-mode binary), t (departure time), w (service
    start), y (fuel level).
  - Constraints (10)–(13), (16)–(17), (19)–(23) from the paper; (14)–(15) dropped (cargo
    relaxed); (18) relaxed to global [0, u0].
  - Makespan variable T_max added (not in Lu et al. — paper minimises total cost).
- `dronesim/evrp/planners/evrp_td_milp.py` renamed from `evrp_lera_romero.py`; updated to
  call `build_td_evrp_milp_model` and `solve_and_extract_td_milp`.
- Initial config used Franceschetti (2013) energy parameters (α, β, γ, λ, δ, μ, h) and a
  3-level speed grid (num_speed_levels: 3).

**Status:** Model ran but solutions were not compared against simulation.

---

### `5543805` — td_evrp_milp (buggy) (2026-04-22)

**What changed:**
- `docs/td_evrp_milp_formulation_notes.md` added — documents every deviation from the paper.
- `config/evrp/td_evrp_milp_smoke.yaml` added — tiny instance (3 customers, 1 AFS) for fast
  iteration.
- Energy model **simplified**: Franceschetti multi-parameter model dropped entirely. Replaced
  with a single constant `e_per_m = energy_per_meter * d_ij` (J/m × m). All α/β/γ/λ/δ/h
  parameters removed from config. See §5 of formulation notes for rationale.
- Objective changed to `w_energy * total_energy + w_makespan * T_max` — a two-weight
  linear combination with units (J, s). This is a new objective not in Lu et al.
- Speed level count reduced: num_speed_levels 3 → 2 (vc and vf only), shrinking the
  variable/constraint count.
- Solver time limit raised to 200 s, MIP gap set to 0.01.

**Status:** Model produced routes. Noted as "buggy" in commit message — specific bugs
not yet isolated.

---

### `0feb90f` — unified gvrp fuel constraints in tdevrp (2026-04-24)

**What changed:**
- Fuel/energy source of truth unified across all planners. The MILP previously read
  `cost_model.energy_per_meter` and `cost_model.energy_capacity_j`. These are now derived
  from `gvrp.consumption_rate` (gal/mile → gal/m) and `gvrp.fuel_capacity` (gallons),
  matching the GVRP MILP and heuristic planners. Ensures consistent vehicle range across
  all three planners for fair comparison.
- Speed level loop fixed: iterate over `speed_levels[1:]` instead of `speed_levels` to skip
  the congestion speed vc at index 0. The congestion speed is used for intervals m=1,5 but
  does not define a free-flow interval, so including it duplicated constraints incorrectly.
- Removed verbose Gurobi `printStats()` dump at solve time.
- Updated formulation notes (section 5 rewritten).

---

### Current session (uncommitted) — simulation integration

**What changed:**

1. **Service-start and departure-time wait extraction** (`dronesim/evrp/milp/td_evrp.py`,
   `dronesim/evrp/routes/base.py`, `dronesim/evrp/planners/evrp_td_milp.py`)
   - `Stop` dataclass gains `planned_wait_s: float = 0.0` (seconds from service-start
     to planned departure, per stop).
   - In `solve_and_extract_td_milp`, for each stop: `wait_s = departure_s − w[nd].X`
     where `departure_s = Σ t[nd, nxt, m, r].X` over the active outgoing arc mode.
   - Planner stores it as `Stop.planned_wait_s`.

2. **Simulation toggle** (`config/evrp/base.yaml`, `config/evrp/td_evrp_milp.yaml`,
   `dronesim/evrp/simulator/evrp_sim.py`)
   - `simulation.use_planned_wait_times` (default false). When true, the simulator uses
     `stop.planned_wait_s` as the dwell time at each stop instead of the fixed
     `service_time_s` / `refuel_time_s`. Runtime trackback refuels are unaffected.
   - Enabled by default in `td_evrp_milp.yaml`.

3. **Lu traffic model** (`dronesim/evrp/traffic/lu.py`, `__init__.py`, `segmented.py`)
   - `LuTrafficModel`: three-segment piecewise-constant speed model matching the MILP
     profile. Returns vc_ms for t < T1, vf_ms for T1 ≤ t < T2, vc_ms for t ≥ T2.
   - `build_traffic_model()` dispatches to `LuTrafficModel` when `traffic.model: lu`.
   - Simulator type annotations widened from `SegmentedSpeedModel` to `BaseTrafficModel`.
   - `td_evrp_milp.yaml` sets `traffic: {enabled: true, model: lu}`.

4. **departure_s extraction bug fixed** (`dronesim/evrp/milp/td_evrp.py`)
   - See next section.

---

## Why td_evrp Is Not Outperforming the Heuristic

### The Bug That Caused Catastrophic Simulation Results (now fixed)

In `solve_and_extract_td_milp`, the `departure_s` for each stop was computed as:

```python
departure_s = sum(
    t[nd, j, m, r].X
    for j in succ.get(nd, [])       # ← BUG: global succ
    for (m, r) in valid_mr.get((nd, j), [])
)
```

`succ[depot]` holds the first stop of **every** vehicle, not just the current one.
So for the depot node (stop 0 in every route), `departure_s` accumulated the planned
departure times of all K vehicles. With K=3 vehicles departing at times t₁, t₂, t₃:

```
departure_s for ALL vehicles' depot stop = t₁ + t₂ + t₃
planned_wait_s at depot = t₁ + t₂ + t₃
```

Every vehicle sat at the depot for the combined departure time of all three. Vehicle 3
(assigned 4 customers) was held until well past T2=2160s, forcing it into evening
congestion (15 km/h) for its entire route. This produced a simulation makespan of 4269s
vs the MILP's planned 1364s.

**Fix:** Replace `succ.get(nd, [])` with `[nodes[idx_in_route + 1]]` — the next node
in this specific vehicle's route:

```python
if idx_in_route + 1 < len(nodes):
    nxt = nodes[idx_in_route + 1]
    departure_s = sum(t[nd, nxt, m, r].X for (m, r) in valid_mr.get((nd, nxt), []))
else:
    departure_s = 0.0
```

---

### Remaining Structural Issues

Even with the bug fixed, td_evrp will not dramatically outperform the heuristic until the
following issues are addressed. The table below classifies each issue by origin.

| # | Issue | Cause | Origin |
|---|-------|-------|--------|
| 1 | LP relaxation bound is permanently 0 | Time-indexed formulations admit fractional z/x that trivially achieve T_max=0 | **Lu's formulation** |
| 2 | 100% MIP gap throughout 200s solve | Direct consequence of issue 1; solver cannot prune the tree | **Lu's formulation** |
| 3 | Planned travel times ≠ simulated travel times | MILP uses Ichoua (2003) travel time (blends speeds mid-arc); simulation uses piecewise-constant Lu model (speed changes instantly) | **Architectural gap** |
| 4 | Energy model decoupled from speed | Simplified to `e_per_m * d` (constant per metre). The speed-energy tradeoff that makes td-evrp harder than EVRP is absent. | **Implementation simplification** |
| 5 | Objective is a new formulation, not Lu's | Lu minimises `λ·E + δ·time + h·vehicles`; our objective is `w_energy·fuel + w_makespan·T_max`. The two are not equivalent. | **Implementation choice** |

#### Issue 1 & 2: LP Relaxation Always 0

The Gurobi log shows the LP bound never moves from 0:

```
Best objective 1.364e+03, best bound 0.000e+00, gap 100.0000%
```

In a proper MILP for VRP makespan, the LP relaxation should give a tight lower bound.
Here it gives 0 because, with continuous relaxation, fractional x and z values can satisfy
customer-visit constraints (Eq. 11) with z=0 on all depot-return arcs, making T_max ≥ 0
trivially satisfied. Lu et al. acknowledge this: the paper reports that CPLEX "can only
solve instances of very limited sizes (no more than 15 customers)" and proposes IVNS as the
practical solver precisely because the ILP bound is weak. This is a property of the
formulation class, not a coding error.

**Consequence:** The branch-and-bound tree explored 37,357 nodes in 200s. The solver found
a 1364s solution near the end but cannot prove it is within any percentage of optimal.

#### Issue 3: Ichoua vs Piecewise-Constant Travel Time

The MILP plans arcs using the Ichoua formula (Eq. 1 in Lu et al.), which blends speeds
when a trip crosses T1 or T2. For example, a vehicle departing just before T1 partially
travels at vc and partially at vf; the MILP computes the exact blend. The simulation uses
the Lu traffic model, which switches speed instantaneously at T1 and T2 regardless of
where a vehicle is on its arc.

The consequence: the MILP's predicted arrival time at each node will differ from the
simulation's actual arrival time, even after correctly implementing `use_planned_wait_times`.
Vehicles arrive later than planned if they depart during a transition interval.

**This gap cannot be fully closed without implementing the Ichoua formula in the
simulator**, which would require tracking vehicle position continuously across time
boundaries rather than applying one speed per SUMO leg.

#### Issue 4: Energy Model Does Not Depend on Speed

The full Franceschetti model (Lu et al. Eq. 2) penalises both slow driving
(high β/v engine cost) and fast driving (high γv² aerodynamic cost). This speed-energy
tradeoff is the reason td-evrp requires choosing a free-flow speed level vr rather than
always using vf: sometimes driving slower saves enough energy to avoid an extra AFS stop.

With the simplified model E_ij = e_per_m · d_ij (constant per metre, speed-independent),
the speed choice has zero effect on fuel consumption. The z and t variables still
affect travel time (and thus makespan), but energy is determined only by distance. The
MILP is effectively a time-dependent VRP with distance-proportional fuel constraints,
not a true TD-EVRP.

#### Issue 5: Different Objective Function

Lu et al. combine driver wage (proportional to trip duration), energy cost, and vehicle
acquisition cost into a single monetary objective. This naturally balances speed against
cost. Our `w_makespan · T_max` term only cares about the latest return time across
all vehicles — if two vehicles finish early and one finishes late, nothing penalises
the early finishers waiting. The heuristic (RoutingBlocks ALNS) optimises total route
length, which correlates more directly with what the simulator measures.

---

### Summary

The td_evrp implementation is a partial implementation of Lu et al. (2020). The MILP
structure (variables, interval bounds, linearisation) is correctly coded, but the
formulation is:

- **Weaker than necessary** in its LP relaxation (Lu's design), making the solver
  slow regardless of implementation quality.
- **Simplified** in energy model and objective (implementation choices) in ways that
  decouple the speed decision from the energy decision, removing the core difficulty
  and benefit of the TD-EVRP formulation.
- **Disconnected from simulation** because the MILP travel-time model (Ichoua) and the
  simulation traffic model (piecewise-constant Lu) are different.

The departure_s bug fixed in this session was the immediate cause of catastrophic
simulation results. Fixing it will close the gap between planned makespan (1364s) and
simulated makespan significantly. However, without addressing the architectural issues
above, td_evrp is unlikely to consistently outperform the heuristic.
