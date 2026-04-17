"""Heuristic EVRP planner using RoutingBlocks ALNS.

Uses the ADPTW (Adaptive Departure-time with Time Windows) model from
RoutingBlocks, which natively handles battery constraints and AFS insertion.
Station visits are inserted/removed by dedicated ALNS operators so the
solver decides when and where to refuel.

Key differences from the MILP planner
--------------------------------------
- No optimality guarantee; quality depends on iteration budget.
- Runs in seconds (vs minutes for the MILP on large instances).
- Objective follows the tmp.py heuristic: minimise route makespan with a
    fixed penalty when fewer than fleet.size routes are used.
- AFS stations are passed as RoutingBlocks "stations"; the solver inserts
    them automatically — you don't need dummy copies.

Config keys used (same section as gvrp.yaml)
---------------------------------------------
  fleet.size              number of vehicles
  gvrp.fuel_capacity      battery capacity (gallons)
  gvrp.consumption_rate   gallons per mile (converted internally)
  gvrp.distance_unit      meters | kilometers | miles
  heuristic.alns_iterations   (optional, default 300)
  heuristic.seed              (optional, default None → time-based)
"""
from __future__ import annotations

import copy
import random
import time
from typing import Any

import numpy as np
import routingblocks as rb

from ..instance.builder import build_problem_instance_from_config
from ..routes.base import Route, Stop
from .base import BasePlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_dist_to_miles(dist_m: np.ndarray, distance_unit: str) -> np.ndarray:
    """Convert distance matrix to miles for battery accounting."""
    unit = distance_unit.lower()
    if unit in {"miles", "mile", "mi"}:
        return dist_m
    if unit in {"meters", "meter", "m"}:
        return dist_m * 0.000621371
    if unit in {"kilometers", "kilometer", "km"}:
        return dist_m * 0.621371
    raise ValueError(f"Unknown distance unit: {distance_unit}")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class EVRPHeuristicPlanner(BasePlanner):
    """RoutingBlocks ALNS heuristic for EVRP with explicit AFS nodes."""

    def plan(self, config: dict[str, Any]) -> list[Route]:
        physical_dist, params, physical_meta, generation_info = (
            build_problem_instance_from_config(config)
        )

        names: list[str] = physical_meta["physical_node_names"]
        types: list[str] = physical_meta["physical_node_types"]
        sumo_ids: list[str] = physical_meta["physical_sumo_ids"]
        depot_idx: int = physical_meta["depot_index"]
        coords: np.ndarray = generation_info["physical_coords"]

        num_vehicles = int(params["m"])
        fuel_capacity = float(params["Q"])          # gallons
        gal_per_mile = float(config.get("gvrp", {}).get("consumption_rate", 1.0))
        distance_unit = str(params.get("distance_unit", "meters"))

        heuristic_cfg = config.get("heuristic", {})
        alns_iterations = int(heuristic_cfg.get("alns_iterations", 300))
        seed_val = heuristic_cfg.get("seed", None)

        if seed_val is not None:
            random.seed(int(seed_val))
            np.random.seed(int(seed_val))

        customer_indices = [i for i, t in enumerate(types) if t == "customer"]
        afs_indices      = [i for i, t in enumerate(types) if t == "afs"]

        if not customer_indices:
            raise ValueError("No customer nodes in instance")

        # Convert distance matrix to miles for consistent battery accounting
        dist_miles = _scale_dist_to_miles(physical_dist, distance_unit)

        BIG = 1e9
        BIG_TIME = 1e9

        def makespan(solution: rb.Solution, fleet_count=0) -> float:
            """Return max route cost across all routes."""
            costs = [route.cost for route in solution]
            return max(costs) if costs else 0.0

        def objective_with_fleet_penalty(
            solution: rb.Solution,
            fleet_count: int,
            penalty: float = 1e6,
        ) -> float:
            """Tmp-parity objective: makespan + underused-fleet penalty."""
            non_empty = sum(1 for route in solution if len(route) > 2)
            obj = makespan(solution)
            if non_empty < fleet_count:
                obj += penalty * (fleet_count - non_empty)
            return obj

        # ── Node ID strings for RoutingBlocks ─────────────────────────────
        # rb needs unique string IDs; we suffix by role to avoid collisions
        # when depot and station share the same physical SUMO node.
        def rb_id(idx: int, role: str) -> str:
            return f"{sumo_ids[idx]}_{role}"

        depot_rb_id = rb_id(depot_idx, "depot")

        customer_rb_ids = [rb_id(i, "customer") for i in customer_indices]

        # Each AFS becomes one station in rb; rb's FPO handles how many times
        # a vehicle visits it.
        station_rb_ids = [rb_id(i, "station") for i in afs_indices]
        # Also add the depot as a charging station (vehicles can return mid-route)
        depot_station_rb_id = rb_id(depot_idx, "station")
        all_station_rb_ids = station_rb_ids + [depot_station_rb_id]

        # Map rb id → physical index
        rb_id_to_idx: dict[str, int] = {
            depot_rb_id: depot_idx,
            depot_station_rb_id: depot_idx,
        }
        for rb_cid, cidx in zip(customer_rb_ids, customer_indices):
            rb_id_to_idx[rb_cid] = cidx
        for rb_sid, sidx in zip(station_rb_ids, afs_indices):
            rb_id_to_idx[rb_sid] = sidx

        all_rb_ids = [depot_rb_id] + customer_rb_ids + all_station_rb_ids

        # ── Arc distance matrix in rb id space ────────────────────────────
        def arc_dist(a_rb: str, b_rb: str) -> float:
            ai = rb_id_to_idx[a_rb]
            bi = rb_id_to_idx[b_rb]
            return float(dist_miles[ai, bi])

        # ── Build RoutingBlocks instance ───────────────────────────────────
        builder = rb.utility.InstanceBuilder(
            create_vertex=rb.adptw.create_adptw_vertex,
            create_arc=rb.adptw.create_adptw_arc,
        )

        dx, dy = float(coords[depot_idx, 0]), float(coords[depot_idx, 1])
        builder.set_depot(
            depot_rb_id,
            rb.adptw.VertexData(dx, dy, 0.0, 0.0, BIG_TIME, 0.0),
        )

        for rb_cid, cidx in zip(customer_rb_ids, customer_indices):
            cx, cy = float(coords[cidx, 0]), float(coords[cidx, 1])
            builder.add_customer(
                rb_cid,
                rb.adptw.VertexData(cx, cy, 1.0, 0.0, BIG_TIME, 0.0),
            )

        for rb_sid in all_station_rb_ids:
            sidx = rb_id_to_idx[rb_sid]
            sx, sy = float(coords[sidx, 0]), float(coords[sidx, 1])
            builder.add_station(
                rb_sid,
                rb.adptw.VertexData(sx, sy, 0.0, 0.0, BIG_TIME, 0.0),
            )

        for a_id in all_rb_ids:
            for b_id in all_rb_ids:
                d = arc_dist(a_id, b_id)
                builder.add_arc(a_id, b_id, rb.adptw.ArcData(d, d, d * gal_per_mile))

        builder.fleet_size = num_vehicles
        instance = builder.build()

        # ── Evaluation ────────────────────────────────────────────────────
        evaluation = rb.adptw.Evaluation(fuel_capacity, 1e7)
        evaluation.overload_penalty_factor = 100.0
        evaluation.resource_penalty_factor = 100.0
        evaluation.time_shift_penalty_factor = 100.0

        # ── Local search operators ─────────────────────────────────────────
        arc_set = rb.ArcSet(instance.number_of_vertices)
        ls_operators = [
            rb.operators.SwapOperator_0_1(instance, arc_set),
            rb.operators.SwapOperator_0_2(instance, arc_set),
            rb.operators.SwapOperator_1_1(instance, arc_set),
            rb.operators.InsertStationOperator(instance),
            rb.operators.RemoveStationOperator(instance),
        ]
        local_search = rb.LocalSearch(
            instance, evaluation, None, rb.BestImprovementPivotingRule()
        )

        # ── ALNS ──────────────────────────────────────────────────────────
        rng_seed = seed_val if seed_val is not None else time.time_ns()
        randgen = rb.Random(int(rng_seed))
        alns = rb.AdaptiveLargeNeighborhood(randgen, 0.4)

        alns.add_repair_operator(rb.operators.RandomInsertionOperator(randgen))
        alns.add_repair_operator(
            rb.operators.BestInsertionOperator(
                instance,
                rb.operators.blink_selector_factory(0.1, randgen),
            )
        )
        alns.add_destroy_operator(rb.operators.RandomRemovalOperator(randgen))
        alns.add_destroy_operator(
            rb.operators.WorstRemovalOperator(
                instance,
                rb.operators.blink_selector_factory(0.1, randgen),
            )
        )

        fpo = rb.adptw.FacilityPlacementOptimizer(instance, fuel_capacity)

        # ── Initial solution: round-robin ──────────────────────────────────
        cust_vids = [v.vertex_id for v in instance.customers]
        random.shuffle(cust_vids)
        k = min(num_vehicles, len(cust_vids))
        buckets: list[list] = [[] for _ in range(k)]
        for i, vid in enumerate(cust_vids):
            buckets[i % k].append(vid)
        init_routes = [
            rb.create_route(evaluation, instance, b) for b in buckets if b
        ]
        best_sol = rb.Solution(evaluation, instance, init_routes)
        best_objective = makespan(best_sol, num_vehicles)

        # ── Main ALNS loop ─────────────────────────────────────────────────
        for it in range(1, alns_iterations + 1):
            current = copy.copy(best_sol)

            n_in_sol = sum(max(0, len(r) - 2) for r in current)
            if n_in_sol == 0:
                continue
            n_remove = max(1, int(random.uniform(0.2, 0.4) * n_in_sol))

            picked = alns.generate(evaluation, current, n_remove)
            local_search.optimize(current, ls_operators)

            if it % 50 == 0 and best_sol.feasible:
                try:
                    opt_routes = []
                    for route in best_sol:
                        vids = [node.vertex_id for node in route]
                        opt_vids = fpo.optimize(vids)
                        opt_routes.append(
                            rb.create_route(evaluation, instance, opt_vids[1:-1])
                        )
                    fpo_sol = rb.Solution(
                        evaluation, instance,
                        [r for r in opt_routes if not r.empty],
                    )
                    fpo_objective = makespan(
                        fpo_sol,
                        num_vehicles,
                    )
                    if fpo_objective < best_objective:
                        best_sol = fpo_sol
                        best_objective = fpo_objective
                except Exception:
                    pass

            current_objective = makespan(
                current,
                num_vehicles,
            )
            if current_objective < best_objective:
                best_sol = current
                best_objective = current_objective
                alns.collect_score(*picked, 4)
            else:
                alns.collect_score(*picked, 0)

            if it % 20 == 0:
                alns.adapt_operator_weights()

        final_routes = sum(1 for r in best_sol if len(r) > 2)
        print(
            f"ALNS complete: makespan={best_objective:.1f}, "
            f"routes={final_routes}/{num_vehicles}, "
            f"feasible={best_sol.feasible}"
        )

        # ── Extract routes → list[Route] ───────────────────────────────────
        customer_rb_id_set = set(customer_rb_ids)
        station_rb_id_set  = set(all_station_rb_ids)

        routes: list[Route] = []
        vehicle_id = 1

        for rb_route in best_sol:
            stops_raw = list(rb_route)  # sequence of rb nodes including depot ends

            # Collect only customer + station stops (skip the depot bookends)
            middle_nodes = []
            for node in stops_raw:
                sid = node.vertex_strid
                if sid in customer_rb_id_set or sid in station_rb_id_set:
                    middle_nodes.append(sid)

            # Skip empty routes
            if not any(sid in customer_rb_id_set for sid in middle_nodes):
                continue

            stops: list[Stop] = []

            # depot start
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type=types[depot_idx],
                planned_fuel=fuel_capacity,
                planned_time=0.0,
            ))

            for sid in middle_nodes:
                phys_idx = rb_id_to_idx[sid]
                if sid in customer_rb_id_set:
                    node_type = "customer"
                else:
                    # depot used as mid-route charging station → depot_dummy
                    # so the simulator knows to refuel there
                    node_type = "afs" if phys_idx != depot_idx else "depot_dummy"
                stops.append(Stop(
                    sumo_node_id=sumo_ids[phys_idx],
                    node_name=names[phys_idx],
                    node_type=node_type,
                    planned_fuel=0.0,   # rb doesn't expose per-stop fuel easily
                    planned_time=0.0,
                ))

            # depot return
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type="depot",
                planned_fuel=0.0,
                planned_time=0.0,
            ))

            routes.append(Route(vehicle_id=vehicle_id, stops=stops))
            vehicle_id += 1

        if not routes:
            raise RuntimeError(
                "ALNS produced no non-empty routes. "
                "Try increasing heuristic.alns_iterations or checking fuel_capacity."
            )

        self._last_physical_dist = physical_dist
        self._last_physical_meta = physical_meta
        self._last_generation_info = generation_info

        return routes
