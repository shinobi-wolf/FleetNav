# TD-EVRP MILP: Formulation Notes and Deviations from Lu et al. (2020)

**Reference:** Lu J., Chen Y., Hao J.-K., He R. (2020). "The Time-dependent Electric Vehicle Routing Problem: Model and solution." *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2020.113593

**Implementation file:** `dronesim/evrp/milp/td_evrp.py`

---

## 1. What We Implement

We implement the paper's ILP-based exact formulation (Section 3, Equations 1–23). The paper also proposes an Iterated Variable Neighbourhood Search (IVNS) heuristic (Section 4) — **that is not implemented here**. Only the constrained optimisation model is used, solved via Gurobi.

---

## 2. Relaxations (Constraints Dropped or Simplified)

### 2.1 Cargo / Load Constraints — Constraints (14) and (15) DROPPED

The paper models cargo flow `f_ij` on each arc (the amount of goods carried from node `i` to `j`). This leads to two constraints:

- **(14)** Demand satisfaction: `Σ_j f_ij − Σ_j f_ji = q_i` for each customer `i` (flow balance).
- **(15)** Capacity bound: `0 ≤ f_ij ≤ C · x_ij` (cargo cannot exceed vehicle capacity `C`).

**What we do:** Both constraints are dropped entirely. The cargo variable `f_ij` is eliminated and fixed to zero everywhere: `f_ij = 0` for all `(i, j)`.

**Impact on the energy model (Eq. 5):** The Franceschetti (2013) / Bektas-Laporte (2011) energy formula is:

```
E(d, v, f) = α(μ + f)d  +  β(d/v)  +  γ·d·v²
```

With `f = 0`, the weight-dependent term simplifies to `α·μ·d` where `μ` is the vehicle curb weight. The energy model is otherwise unchanged.

**Justification:** The routing problem here is for ground delivery robots carrying packages of known small weight that do not materially affect energy consumption. The cargo relaxation decouples routing from load tracking.

### 2.2 Time Windows — Constraint (18) Simplified

The paper's constraint (18) enforces customer-specific time windows `[a_i, b_i]`:

```
a_i ≤ w_i ≤ b_i
```

**What we do:** We do not model customer-specific time windows. Instead, the service-start variable `w_i` is bounded globally by `[0, u0]` (where `u0 = Tmax` in seconds). This is equivalent to setting `a_i = 0` and `b_i = u0` for all nodes.

**Justification:** DroneSim's problem instances are generated randomly and do not carry time-window annotations. The global time horizon is enforced implicitly via `u0`.

---

## 3. Additions Beyond the Paper

### 3.1 Arc Energy Feasibility Constraint (New — Not in Lu et al.)

Lu et al.'s energy balance constraints (19) and (20) only bound `y_j` (remaining energy at node `j`) when `j` is a customer:

```
y_j ≤ y_i − E_ij^{mr}·z  +  Q·(1 − z)
```

For arcs `(i, j)` where `j` is an AFS node, the paper's constraint (21) pins `y_j = Q` (full recharge), which bypasses the energy propagation. This means a vehicle could "teleport" to a refuelling station without having sufficient energy to reach it.

**What we add:** For every arc `(i, j, m, r)` — regardless of node type — we add:

```
y_i ≥ A_E·z_{i,j,m,r}  +  B_E·t_{i,j,m,r}
```

This ensures the vehicle has enough energy at `i` to actually traverse the arc. It tightens the LP relaxation and prevents infeasible solutions that exploit the pinned recharge.

### 3.2 Dummy Vertex Expansion for AFS Revisits

The paper assumes each AFS can be visited at most once per vehicle, which it handles via constraint (12). To allow multiple visits to the same physical AFS across the planning horizon, we use the Erdogan & Miller-Hooks (2012) dummy vertex technique:

- Each physical AFS with `k` permitted visits generates `k` dummy nodes in the expanded graph.
- Distances between dummies and other nodes mirror the physical distance.
- The depot is similarly duplicated with depot-dummy copies.

This expansion is handled by `dronesim/evrp/instance/dummy_expansion.py` before the model is built.

### 3.3 Depot Start Time Pinned to Zero

The paper does not explicitly state `w_depot = 0`. We add:

```
w_depot = 0
```

This ensures all vehicles depart at time `t = 0`.

---

## 4. Driver Wage Objective — Extended

The paper's wage objective (Eqs. 7–8) counts driving time between customer and AFS nodes only. We extend it to also account for the final leg returning to the depot:

- For intermediate nodes (customers, AFS): wage contribution = `δ · (departure_from_i − departure_to_i)`.
- For the main depot (return endpoint): wage contribution = `δ · (arrival_at_depot − departure_from_last_node)`.
- For depot-dummy copies (mid-route depot revisits): `δ · (w_i − departure_to_i)`.

This ensures the wage accounts for the full route duration, not just the customer-serving portion.

---

## 5. Simplified Energy Model and Objective

The original Lu et al. objective is:

```
min  λ · Σ E_ij · z_ij  +  δ · total_duration  +  h · vehicles_used
```

where `E_ij` follows the Franceschetti (2013) / Bektas-Laporte (2011) model:

```
E(d, v, f) = α(μ + f)d  +  β(d/v)  +  γ·d·v²
```

**What we do instead:** Replace the entire cost model with a two-term weighted objective:

```
min  w_energy · (Σ_{(i,j)} e_per_m · d_ij · x_ij)  +  w_makespan · T_max
```

where:
- `e_per_m` (J/m) — constant energy consumption per metre, independent of speed or load.
- `T_max` — makespan: the completion time of the last vehicle to return to any depot node, enforced by `T_max ≥ arrival_{i→j}` for all arcs `(i,j)` with `j` a depot-type node.
- `w_energy`, `w_makespan` — dimensionless objective weights (tune to balance J vs s).

**Consequences:**
- The Franceschetti parameters α, β, γ, the driver wage δ, and the acquisition cost `h` are all dropped.
- The energy balance constraints (19)/(20) and the arc feasibility constraint simplify: arc energy `E_ij = e_per_m · d_ij` is now a constant coefficient on `x_ij`, not a function of `(m, r)`. This reduces the constraint count from O(|A|·|M|·|R|) to O(|A|) for energy.
- The speed-level and departure-interval variables `z, t` are still needed — they determine the time-dependent travel times that control the makespan.

## 6. Speed Level Discretisation

The paper uses `R = {v¹, ..., vᵏ}` as a discrete set of free-flow speed levels, with `v¹ = vc` (congestion speed) and `vᵏ = vf` (maximum free-flow speed).

**Our implementation:** We generate `k` uniformly-spaced levels between `vc` and `vf` using `numpy.linspace(vc, vf, k)`. The config parameter `traffic_lu.num_speed_levels` controls `k` (default 2, minimum 2).

---

## 7. Five Departure Intervals — Boundary Clipping

Lu et al. define five departure intervals `M = {1, 2, 3, 4, 5}` split by the Ichoua (2003) FIFO travel time boundary crossings at `T1` and `T2`. For a given arc of length `d` and free-flow speed `vr`, the interval bounds are:

| Interval | Departure range |
|----------|----------------|
| m=1 | `[0, T1 − d/vc]` — departs and arrives in morning congestion |
| m=2 | `[T1 − d/vc, T1]` — crosses the T1 boundary |
| m=3 | `[T1, T2 − d/vr]` — departs and arrives in free-flow |
| m=4 | `[T2 − d/vr, T2]` — crosses the T2 boundary |
| m=5 | `[T2, u0]` — departs and arrives in evening congestion |

For long arcs, some intervals collapse (e.g. if `d/vc > T1`, interval m=1 becomes empty). We clip all bounds to `[0, u0]` and skip intervals where `b_hi − b_lo ≤ ε`. This prevents Gurobi from receiving constraints over empty sets and avoids numerical issues with negative bounds.

**Continuity verified:** The travel time `T(t)` function is continuous at all four boundaries (m=1↔2, m=2↔3, m=3↔4, m=4↔5).

---

## 8. Linearisation of T·z Products

The travel time `T_ij^{mr}` is affine in the departure time `t_ij^{mr}`:

```
T = A_T + B_T · t
```

Constraint (22) enforces `z · b_prev ≤ t ≤ z · b_curr`, which implies `t = 0` when `z = 0`. Therefore `z · t = t` always, and the product `T · z` linearises without auxiliary variables:

```
T · z = A_T · z + B_T · t
```

This keeps the model linear (ILP, not MIQP). Note: with the simplified energy model, there is no longer an `E · z` product — energy depends only on `x_ij`, which is already binary.

---

## 9. Infeasible Arcs

If no valid departure interval `(m, r)` exists for arc `(i, j)` (e.g. the arc length exceeds the entire time horizon at any speed), we add:

```
x[i, j] = 0
```

The arc is forbidden, preventing it from appearing in any solution.

---

## 10. Config Parameters

| Section | Key | Default | Unit | Description |
|---------|-----|---------|------|-------------|
| `traffic_lu` | `v_congestion_kmh` | 15.0 | km/h | Speed during [0,T1] and [T2,u0] |
| `traffic_lu` | `v_freeflow_kmh` | 50.0 | km/h | Maximum free-flow speed |
| `traffic_lu` | `num_speed_levels` | 2 | — | Discrete levels in R |
| `traffic_lu` | `T1_hours` | 1.0 | h | End of morning congestion |
| `traffic_lu` | `T2_hours` | 4.0 | h | Start of evening congestion |
| `cost_model` | `energy_per_meter` | 100.0 | J/m | Energy consumed per metre travelled |
| `cost_model` | `w_energy` | 1.0 | — | Objective weight for total energy |
| `cost_model` | `w_makespan` | 1.0 | — | Objective weight for makespan |
| `cost_model` | `energy_capacity_j` | 7.2e7 | J | Battery capacity (≈ 20 kWh) |

All distances are converted to SI (meters) and times to seconds before model construction.

---

## 10. Summary of Constraint Mapping

| Paper constraint | Description | Status |
|-----------------|-------------|--------|
| (10) | Vehicle count ≤ K | ✅ Implemented |
| (11) | Each customer visited exactly once | ✅ Implemented |
| (12) | Each station/depot dummy visited at most once | ✅ Implemented |
| (13) | Flow conservation at every node | ✅ Implemented |
| (14) | Cargo flow balance | ❌ **Dropped** (cargo relaxed) |
| (15) | Cargo capacity bound | ❌ **Dropped** (cargo relaxed) |
| (16) | Arrival ≤ service-start | ✅ Implemented |
| (17) | Service-start + service-time ≤ departure | ✅ Implemented |
| (18) | Customer time windows | ⚠️ **Relaxed** to global [0, u0] |
| (19) | Energy balance (customer predecessor) | ✅ **Simplified** — `E_ij = e_per_m·d_ij·x_ij` (per arc, not per (m,r)) |
| (20) | Energy balance (refuel predecessor) | ✅ **Simplified** — unified with (19) |
| (21) | Full charge at depot and refuel stations | ✅ Implemented |
| (22) | Departure interval bounds on t | ✅ Implemented |
| (23) | Σ z = x (arc-mode consistency) | ✅ Implemented |
| — | Arc energy feasibility (new) | ✅ **Added** — `y_i ≥ e_per_m·d_ij·x_ij` (per arc) |
| — | Makespan variable T_max (new) | ✅ **Added** — `T_max ≥ arrival` at all depot-return arcs |
| — | Depot start time = 0 | ✅ **Added** (implicit in paper) |
