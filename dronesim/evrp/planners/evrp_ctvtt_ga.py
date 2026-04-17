"""Genetic Algorithm planner for EVRP-CTVTT (Shao et al., 2017).

Implements the GA from Section 4 of the paper with these simplifications
(all intentional adaptations, see config/evrp/ctvtt_ga.yaml for rationale):

    - Static travel time: t_ij = d_ij / v  (no dynamic Dijkstra)
    - No soft time windows (no Cp penalty term, no T_i^early / T_i^delay)
    - No cargo weight constraints (W_max, w_j dropped)
    - T_Ok fixed at 0 (no depot-departure-time enumeration)
    - No max_duration_hours constraint inside the GA

Objective (paper §3, eqs. 3-6 without Cp):
    min  Cf + Ct + Cr
        Cf = c_f * (number of effective routes)
        Ct = c_t * total travel time in minutes
        Cr = c_c * total recharge events

Chromosome encoding (paper §4, steps 1-5):
    Gene IDs:
        0 .. nn-1              → customer k maps to physical index k+1
        nn .. nn+mm-1          → station k maps to physical index k+1
        nn+mm .. nn+mm+2kk-1   → depot markers (2kk total: kk start + kk end)

    Construction:
        1. A = random permutation of all customer+station gene IDs
        2. Place marker nn+mm at position 0
           Place marker nn+mm+1 at the last position
        3. Insert remaining 2kk-2 markers at random internal positions

    Decoding:
        Depot markers are paired left-to-right in groups of two. Each
        consecutive pair (start, end) defines one vehicle route from the
        content between them. Routes with no customers are 'ineffective'
        and dropped from the output.

Genetic operators (paper does not specify — chosen from standard VRP GA):
    - Crossover: Order Crossover (OX). Permutation-preserving; handles
      depot markers naturally since all gene IDs are unique.
    - Mutation: swap two random positions.
    - Feasibility: infeasible children/mutants are retried up to
      `max_feasibility_retries`; on exhaustion we fall back to parents.

Unit conventions inside this file:
    All GA-internal computation happens in paper units (km, minutes, yuan).
    Unit conversion from the config's distance_unit happens once at the
    boundary in plan(). Route objects returned use hours (planned_time) and
    gallons (planned_fuel) to stay compatible with the simulator.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..instance.builder import build_problem_instance_from_config
from ..routes.base import Route, Stop
from .base import BasePlanner


_KM_PER_MILE = 1.609344


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _unit_to_km_factor(distance_unit: str) -> float:
    unit = distance_unit.lower()
    if unit in {"miles", "mile", "mi"}:
        return _KM_PER_MILE
    if unit in {"meters", "meter", "m"}:
        return 0.001
    if unit in {"kilometers", "kilometer", "km"}:
        return 1.0
    raise ValueError(f"Unsupported distance unit: {distance_unit}")


# ---------------------------------------------------------------------------
# Internal GA instance (all quantities in paper units)
# ---------------------------------------------------------------------------

@dataclass
class _GAInstance:
    nn: int                      # customers
    mm: int                      # charging stations
    kk: int                      # vehicles
    depot_idx: int               # physical index of the depot (always 0)
    dist_km: np.ndarray          # complete distance matrix in km
    range_km: float              # D_max in km (full-battery range)
    speed_kmh: float             # constant driving speed
    t_c_min: float               # constant charging time per recharge
    c_f: float                   # per-vehicle fixed cost (yuan)
    c_t: float                   # travel-time cost (yuan/min)
    c_c: float                   # charging cost (yuan/recharge)
    fuel_capacity_gal: float     # Q, used only to report gallons on Stops


# ---------------------------------------------------------------------------
# Gene ↔ physical node mapping
# ---------------------------------------------------------------------------

def _gene_to_physical(gene: int, nn: int, mm: int) -> int:
    """Map gene ID to physical node index.

    Physical node layout (from build_problem_instance_from_config):
        0                  → depot
        1 .. nn            → customers
        nn+1 .. nn+mm      → stations

    Customer gene k (0 ≤ k < nn)    → physical index k + 1
    Station  gene k (nn ≤ k < nn+mm)→ physical index k + 1
    Depot marker (k ≥ nn+mm)        → physical index 0 (depot)
    """
    if gene < nn + mm:
        return gene + 1
    return 0


def _is_station_gene(gene: int, nn: int, mm: int) -> bool:
    return nn <= gene < nn + mm


def _is_marker(gene: int, nn: int, mm: int) -> bool:
    return gene >= nn + mm


# ---------------------------------------------------------------------------
# Chromosome construction and decoding
# ---------------------------------------------------------------------------

def _make_individual(nn: int, mm: int, kk: int, rng: random.Random) -> list[int]:
    """Generate one individual per paper §4 steps 1-5.

    Internal markers are inserted as *adjacent pairs* (end-of-route i, start-of-
    route i+1). This guarantees every customer lands inside some route segment
    (no gaps between pairs) and matches the structure visible in the paper's
    own Figure 4 example.
    """
    body = list(range(nn + mm))  # customers + stations
    rng.shuffle(body)

    marker_base = nn + mm
    # S1 at position 0, E_k at the last position
    individual = [marker_base] + body + [marker_base + 1]

    # Insert (E_i, S_{i+1}) pairs as adjacent markers at random internal positions
    for pair_idx in range(kk - 1):
        e_i = marker_base + 2 + 2 * pair_idx
        s_next = marker_base + 3 + 2 * pair_idx
        pos = rng.randint(1, len(individual) - 1)
        individual.insert(pos, s_next)
        individual.insert(pos, e_i)

    return individual


def _decode_routes(individual: list[int], nn: int, mm: int, kk: int) -> list[list[int]]:
    """Split chromosome into kk route segments by pairing depot markers
    left-to-right in groups of two."""
    marker_base = nn + mm
    marker_positions = [i for i, g in enumerate(individual) if g >= marker_base]
    if len(marker_positions) != 2 * kk:
        raise ValueError(
            f"Chromosome has {len(marker_positions)} depot markers, expected {2 * kk}"
        )

    routes: list[list[int]] = []
    for i in range(0, 2 * kk, 2):
        start = marker_positions[i]
        end = marker_positions[i + 1]
        routes.append(individual[start + 1:end])
    return routes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class _EvalResult:
    cost: float
    effective_routes: int
    total_recharges: int
    total_time_min: float


_INFEASIBLE = _EvalResult(float("inf"), 0, 0, 0.0)


def _evaluate(individual: list[int], inst: _GAInstance) -> _EvalResult:
    """Compute the paper's objective for this chromosome.

    Returns _INFEASIBLE if:
      - any customer is in a 'gap' between consecutive route pairs
        (not covered by any vehicle), OR
      - any route violates battery range.
    """
    routes = _decode_routes(individual, inst.nn, inst.mm, inst.kk)

    # Coverage: every customer gene must appear inside some paired route.
    customers_covered = 0
    for segment in routes:
        for gene in segment:
            if gene < inst.nn:
                customers_covered += 1
    if customers_covered != inst.nn:
        return _INFEASIBLE  # some customers fell into inter-pair gaps

    total_time_min = 0.0
    total_recharges = 0
    effective_routes = 0

    for segment in routes:
        has_customer = any(g < inst.nn for g in segment)
        if not has_customer:
            continue  # ineffective route

        effective_routes += 1
        cur = inst.depot_idx
        remaining_km = inst.range_km
        route_time = 0.0

        for gene in segment:
            nxt = _gene_to_physical(gene, inst.nn, inst.mm)
            d = inst.dist_km[cur, nxt]
            if remaining_km < d:
                return _INFEASIBLE
            remaining_km -= d
            route_time += (d / inst.speed_kmh) * 60.0

            if _is_station_gene(gene, inst.nn, inst.mm):
                remaining_km = inst.range_km
                route_time += inst.t_c_min
                total_recharges += 1

            cur = nxt

        # Return leg to depot
        d = inst.dist_km[cur, inst.depot_idx]
        if remaining_km < d:
            return _INFEASIBLE
        route_time += (d / inst.speed_kmh) * 60.0

        total_time_min += route_time

    if effective_routes == 0:
        return _INFEASIBLE

    cost = (
        inst.c_f * effective_routes
        + inst.c_t * total_time_min
        + inst.c_c * total_recharges
    )
    return _EvalResult(cost, effective_routes, total_recharges, total_time_min)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _ox_crossover(
    p1: list[int], p2: list[int], rng: random.Random
) -> tuple[list[int], list[int]]:
    """Order Crossover (OX). Preserves the permutation property."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))

    def build(parent: list[int], other: list[int]) -> list[int]:
        child: list[int | None] = [None] * n
        child[a:b + 1] = parent[a:b + 1]
        kept = set(parent[a:b + 1])

        other_idx = (b + 1) % n
        child_idx = (b + 1) % n
        filled = b - a + 1
        while filled < n:
            gene = other[other_idx]
            if gene not in kept:
                child[child_idx] = gene
                child_idx = (child_idx + 1) % n
                filled += 1
            other_idx = (other_idx + 1) % n
        return child  # type: ignore[return-value]

    return build(p1, p2), build(p2, p1)


def _swap_mutation(individual: list[int], rng: random.Random) -> list[int]:
    new = list(individual)
    i, j = rng.sample(range(len(new)), 2)
    new[i], new[j] = new[j], new[i]
    return new


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class EVRPCTVTTGAPlanner(BasePlanner):
    """GA planner for the EVRP-CTVTT model of Shao et al., 2017."""

    def plan(self, config: dict[str, Any]) -> list[Route]:
        physical_dist, params, physical_meta, generation_info = (
            build_problem_instance_from_config(config)
        )

        names: list[str] = physical_meta["physical_node_names"]
        types: list[str] = physical_meta["physical_node_types"]
        sumo_ids: list[str] = physical_meta["physical_sumo_ids"]
        depot_idx: int = physical_meta["depot_index"]

        customer_indices = [i for i, t in enumerate(types) if t == "customer"]
        station_indices = [i for i, t in enumerate(types) if t == "afs"]
        nn = len(customer_indices)
        mm = len(station_indices)
        kk = int(params["m"])

        if nn == 0:
            raise ValueError("GA planner requires at least one customer")
        if kk <= 0:
            raise ValueError("GA planner requires at least one vehicle")

        # Unit conversion to paper units (km, min, yuan)
        to_km = _unit_to_km_factor(params["distance_unit"])
        dist_km = physical_dist * to_km
        range_km = (float(params["Q"]) / float(params["r"])) * to_km
        speed_kmh = float(config.get("gvrp", {}).get("speed_mph", 40.0)) * _KM_PER_MILE
        t_c_min = float(params["refuel_time"]) * 60.0

        ctvtt_cfg = config.get("ctvtt", {})
        c_f = float(ctvtt_cfg.get("vehicle_fixed_cost", 100.0))
        c_t = float(ctvtt_cfg.get("travel_time_cost", 1.0))
        c_c = float(ctvtt_cfg.get("charging_cost", 30.0))

        inst = _GAInstance(
            nn=nn,
            mm=mm,
            kk=kk,
            depot_idx=depot_idx,
            dist_km=dist_km,
            range_km=range_km,
            speed_kmh=speed_kmh,
            t_c_min=t_c_min,
            c_f=c_f,
            c_t=c_t,
            c_c=c_c,
            fuel_capacity_gal=float(params["Q"]),
        )

        ga_cfg = config.get("ga", {})
        N = int(ga_cfg.get("population_size", 50))
        T = int(ga_cfg.get("generations", 300))
        Ne = int(ga_cfg.get("elite_count", 20))
        Pc = float(ga_cfg.get("crossover_rate", 0.9))
        Pm = float(ga_cfg.get("mutation_rate", 0.1))
        max_retries = int(ga_cfg.get("max_feasibility_retries", 20))
        seed = ga_cfg.get("seed", None)
        rng = random.Random(seed)

        print(
            f"  [GA] instance: nn={nn} customers, mm={mm} stations, kk={kk} vehicles; "
            f"range={range_km:.2f}km, speed={speed_kmh:.2f}km/h, t_c={t_c_min:.1f}min"
        )
        print(
            f"  [GA] config: N={N}, T={T}, Ne={Ne}, Pc={Pc}, Pm={Pm}, "
            f"max_retries={max_retries}"
        )

        best_ind, best_eval = self._run_ga(inst, N, T, Ne, Pc, Pm, max_retries, rng)

        if best_eval.cost >= float("inf"):
            raise RuntimeError(
                "GA found no feasible solution. The instance may be infeasible "
                "with the current battery range, or you may need more generations."
            )

        print(
            f"  [GA] best: cost={best_eval.cost:.2f} yuan "
            f"(Cf={inst.c_f * best_eval.effective_routes:.2f} + "
            f"Ct={inst.c_t * best_eval.total_time_min:.2f} + "
            f"Cr={inst.c_c * best_eval.total_recharges:.2f}), "
            f"routes={best_eval.effective_routes}, "
            f"recharges={best_eval.total_recharges}, "
            f"time={best_eval.total_time_min:.1f}min"
        )

        output_routes = self._individual_to_routes(best_ind, inst, names, types, sumo_ids)

        self._last_physical_dist = physical_dist
        self._last_physical_meta = physical_meta
        self._last_generation_info = generation_info

        return output_routes

    # ------------------------------------------------------------------
    # GA main loop
    # ------------------------------------------------------------------

    def _run_ga(
        self,
        inst: _GAInstance,
        N: int,
        T: int,
        Ne: int,
        Pc: float,
        Pm: float,
        max_retries: int,
        rng: random.Random,
    ) -> tuple[list[int], _EvalResult]:
        # Step 1: Initial population (retry for feasibility)
        population: list[list[int]] = []
        for _ in range(N):
            chosen = _make_individual(inst.nn, inst.mm, inst.kk, rng)
            for _ in range(max_retries):
                if _evaluate(chosen, inst).cost < float("inf"):
                    break
                chosen = _make_individual(inst.nn, inst.mm, inst.kk, rng)
            population.append(chosen)

        evals = [_evaluate(ind, inst) for ind in population]
        feasible_init = sum(1 for e in evals if e.cost < float("inf"))
        print(f"  [GA] initial population: {feasible_init}/{N} feasible")

        best_ind = min(zip(evals, population), key=lambda x: x[0].cost)[1]
        best_eval = _evaluate(best_ind, inst)

        for gen in range(1, T + 1):
            # Step 3: Elites + remainder
            order = sorted(range(N), key=lambda i: evals[i].cost)
            elites = [population[i] for i in order[:Ne]]
            rest = [population[i] for i in order[Ne:]]
            rng.shuffle(rest)

            # Step 4: Crossover on rest in pairs
            offspring: list[list[int]] = []
            for i in range(0, len(rest) - 1, 2):
                p1, p2 = rest[i], rest[i + 1]
                if rng.random() < Pc:
                    c1, c2 = p1, p2
                    for _ in range(max_retries):
                        c1t, c2t = _ox_crossover(p1, p2, rng)
                        if (_evaluate(c1t, inst).cost < float("inf")
                                and _evaluate(c2t, inst).cost < float("inf")):
                            c1, c2 = c1t, c2t
                            break
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
            if len(rest) % 2 == 1:
                offspring.append(rest[-1])

            # Step 5: Mutation
            mutated: list[list[int]] = []
            for ind in offspring:
                if rng.random() < Pm:
                    new_ind = ind
                    for _ in range(max_retries):
                        candidate = _swap_mutation(ind, rng)
                        if _evaluate(candidate, inst).cost < float("inf"):
                            new_ind = candidate
                            break
                    mutated.append(new_ind)
                else:
                    mutated.append(ind)

            # Step 6: Merge elites with offspring
            population = elites + mutated
            evals = [_evaluate(ind, inst) for ind in population]

            # Track overall best across generations (elitism preserves it anyway)
            gen_best_idx = min(range(N), key=lambda i: evals[i].cost)
            if evals[gen_best_idx].cost < best_eval.cost:
                best_ind = population[gen_best_idx]
                best_eval = evals[gen_best_idx]

            if gen % 50 == 0 or gen == 1:
                print(f"  [GA] gen {gen:4d}: best cost = {best_eval.cost:.2f} yuan")

        return best_ind, best_eval

    # ------------------------------------------------------------------
    # Chromosome → Route objects
    # ------------------------------------------------------------------

    def _individual_to_routes(
        self,
        individual: list[int],
        inst: _GAInstance,
        names: list[str],
        types: list[str],
        sumo_ids: list[str],
    ) -> list[Route]:
        segments = _decode_routes(individual, inst.nn, inst.mm, inst.kk)
        output: list[Route] = []
        vehicle_id = 1

        for segment in segments:
            if not any(g < inst.nn for g in segment):
                continue  # skip ineffective routes

            stops: list[Stop] = [Stop(
                sumo_node_id=sumo_ids[inst.depot_idx],
                node_name=names[inst.depot_idx],
                node_type=types[inst.depot_idx],
                planned_fuel=inst.fuel_capacity_gal,
                planned_time=0.0,
            )]

            remaining_km = inst.range_km
            cumulative_hours = 0.0
            cur = inst.depot_idx

            for gene in segment:
                nxt = _gene_to_physical(gene, inst.nn, inst.mm)
                d_km = float(inst.dist_km[cur, nxt])
                remaining_km -= d_km
                cumulative_hours += d_km / inst.speed_kmh

                stops.append(Stop(
                    sumo_node_id=sumo_ids[nxt],
                    node_name=names[nxt],
                    node_type=types[nxt],
                    planned_fuel=_km_to_gal(remaining_km, inst),
                    planned_time=cumulative_hours,
                ))

                if _is_station_gene(gene, inst.nn, inst.mm):
                    remaining_km = inst.range_km
                    cumulative_hours += inst.t_c_min / 60.0

                cur = nxt

            d_km = float(inst.dist_km[cur, inst.depot_idx])
            remaining_km -= d_km
            cumulative_hours += d_km / inst.speed_kmh
            stops.append(Stop(
                sumo_node_id=sumo_ids[inst.depot_idx],
                node_name=names[inst.depot_idx],
                node_type="depot",
                planned_fuel=_km_to_gal(remaining_km, inst),
                planned_time=cumulative_hours,
            ))

            output.append(Route(vehicle_id=vehicle_id, stops=stops))
            vehicle_id += 1

        return output


def _km_to_gal(remaining_km: float, inst: _GAInstance) -> float:
    """Convert remaining battery range in km to equivalent gallons for
    reporting in Stop.planned_fuel."""
    if inst.range_km <= 0:
        return 0.0
    fraction = max(0.0, remaining_km / inst.range_km)
    return fraction * inst.fuel_capacity_gal
