"""RoutingBlocks ALNS heuristic for EVRP with explicit AFS nodes."""
from __future__ import annotations

import copy
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import routingblocks as rb

from .base import EVRPPlanner, Route, Stop

if TYPE_CHECKING:
    from dronesim.ta_evrp.instance.instance import Instance


class HeuristicPlanner(EVRPPlanner):
    def plan(self, instance: "Instance") -> list[Route]:
        p = instance.problem

        names    = p.physical_meta["physical_node_names"]
        types    = p.physical_meta["physical_node_types"]
        sumo_ids = p.physical_meta["physical_sumo_ids"]
        depot_idx = p.physical_meta["depot_index"]
        coords   = p.generation_info["physical_coords"]

        num_vehicles    = int(p.params["m"])
        fuel_capacity   = float(p.params["Q"])
        consumption_rate = float(p.params.get("r", 1.0))

        heur_cfg       = instance.ta_config.get("heuristic", {})
        alns_iterations = int(heur_cfg.get("alns_iterations", 300))
        seed_val        = heur_cfg.get("seed", None)

        if seed_val is not None:
            random.seed(int(seed_val))
            np.random.seed(int(seed_val))

        customer_indices = [i for i, t in enumerate(types) if t == "customer"]
        afs_indices      = [i for i, t in enumerate(types) if t == "afs"]

        if not customer_indices:
            raise ValueError("No customer nodes in instance")

        BIG_TIME = 1e9

        def makespan(solution: rb.Solution) -> float:
            costs = [route.cost for route in solution]
            return max(costs) if costs else 0.0

        def rb_id(idx: int, role: str) -> str:
            return f"{sumo_ids[idx]}_{role}"

        depot_rb_id      = rb_id(depot_idx, "depot")
        customer_rb_ids  = [rb_id(i, "customer") for i in customer_indices]
        station_rb_ids   = [rb_id(i, "station")  for i in afs_indices]
        depot_station_id = rb_id(depot_idx, "station")
        all_station_ids  = station_rb_ids + [depot_station_id]

        rb_id_to_idx: dict[str, int] = {
            depot_rb_id:      depot_idx,
            depot_station_id: depot_idx,
        }
        for rid, cidx in zip(customer_rb_ids, customer_indices):
            rb_id_to_idx[rid] = cidx
        for rid, sidx in zip(station_rb_ids, afs_indices):
            rb_id_to_idx[rid] = sidx

        all_rb_ids = [depot_rb_id] + customer_rb_ids + all_station_ids

        def arc_dist(a: str, b: str) -> float:
            return float(p.physical_dist[rb_id_to_idx[a], rb_id_to_idx[b]])

        builder = rb.utility.InstanceBuilder(
            create_vertex=rb.adptw.create_adptw_vertex,
            create_arc=rb.adptw.create_adptw_arc,
        )

        dx, dy = float(coords[depot_idx, 0]), float(coords[depot_idx, 1])
        builder.set_depot(depot_rb_id, rb.adptw.VertexData(dx, dy, 0.0, 0.0, BIG_TIME, 0.0))

        for rid, cidx in zip(customer_rb_ids, customer_indices):
            cx, cy = float(coords[cidx, 0]), float(coords[cidx, 1])
            builder.add_customer(rid, rb.adptw.VertexData(cx, cy, 1.0, 0.0, BIG_TIME, 0.0))

        for rid in all_station_ids:
            sidx = rb_id_to_idx[rid]
            sx, sy = float(coords[sidx, 0]), float(coords[sidx, 1])
            builder.add_station(rid, rb.adptw.VertexData(sx, sy, 0.0, 0.0, BIG_TIME, 0.0))

        for a in all_rb_ids:
            for b in all_rb_ids:
                d = arc_dist(a, b)
                builder.add_arc(a, b, rb.adptw.ArcData(d, d, d * consumption_rate))

        builder.fleet_size = num_vehicles
        rb_instance = builder.build()

        evaluation = rb.adptw.Evaluation(fuel_capacity, 1e7)
        evaluation.overload_penalty_factor    = 100.0
        evaluation.resource_penalty_factor    = 100.0
        evaluation.time_shift_penalty_factor  = 100.0

        arc_set = rb.ArcSet(rb_instance.number_of_vertices)
        ls_ops  = [
            rb.operators.SwapOperator_0_1(rb_instance, arc_set),
            rb.operators.SwapOperator_0_2(rb_instance, arc_set),
            rb.operators.SwapOperator_1_1(rb_instance, arc_set),
            rb.operators.InsertStationOperator(rb_instance),
            rb.operators.RemoveStationOperator(rb_instance),
        ]
        local_search = rb.LocalSearch(rb_instance, evaluation, None, rb.BestImprovementPivotingRule())

        rng_seed = seed_val if seed_val is not None else time.time_ns()
        randgen  = rb.Random(int(rng_seed))
        alns     = rb.AdaptiveLargeNeighborhood(randgen, 0.4)
        alns.add_repair_operator(rb.operators.RandomInsertionOperator(randgen))
        alns.add_repair_operator(
            rb.operators.BestInsertionOperator(
                rb_instance, rb.operators.blink_selector_factory(0.1, randgen)
            )
        )
        alns.add_destroy_operator(rb.operators.RandomRemovalOperator(randgen))
        alns.add_destroy_operator(
            rb.operators.WorstRemovalOperator(
                rb_instance, rb.operators.blink_selector_factory(0.1, randgen)
            )
        )

        fpo = rb.adptw.FacilityPlacementOptimizer(rb_instance, fuel_capacity)

        # Initial round-robin solution
        cust_vids = [v.vertex_id for v in rb_instance.customers]
        random.shuffle(cust_vids)
        k       = min(num_vehicles, len(cust_vids))
        buckets: list[list] = [[] for _ in range(k)]
        for i, vid in enumerate(cust_vids):
            buckets[i % k].append(vid)
        init_routes = [rb.create_route(evaluation, rb_instance, b) for b in buckets if b]
        best_sol    = rb.Solution(evaluation, rb_instance, init_routes)
        best_obj    = makespan(best_sol)

        for it in range(1, alns_iterations + 1):
            current  = copy.copy(best_sol)
            n_in_sol = sum(max(0, len(r) - 2) for r in current)
            if n_in_sol == 0:
                continue
            n_remove = max(1, int(random.uniform(0.2, 0.4) * n_in_sol))

            picked = alns.generate(evaluation, current, n_remove)
            local_search.optimize(current, ls_ops)

            if it % 50 == 0 and best_sol.feasible:
                try:
                    opt_routes = []
                    for route in best_sol:
                        vids     = [node.vertex_id for node in route]
                        opt_vids = fpo.optimize(vids)
                        opt_routes.append(rb.create_route(evaluation, rb_instance, opt_vids[1:-1]))
                    fpo_sol = rb.Solution(
                        evaluation, rb_instance, [r for r in opt_routes if not r.empty]
                    )
                    fpo_obj = makespan(fpo_sol)
                    if fpo_obj < best_obj:
                        best_sol, best_obj = fpo_sol, fpo_obj
                except Exception:
                    pass

            curr_obj = makespan(current)
            if curr_obj < best_obj:
                best_sol, best_obj = current, curr_obj
                alns.collect_score(*picked, 4)
            else:
                alns.collect_score(*picked, 0)

            if it % 20 == 0:
                alns.adapt_operator_weights()

        final_routes = sum(1 for r in best_sol if len(r) > 2)
        print(
            f"ALNS complete: makespan={best_obj:.1f}  "
            f"routes={final_routes}/{num_vehicles}  feasible={best_sol.feasible}"
        )

        customer_id_set = set(customer_rb_ids)
        station_id_set  = set(all_station_ids)

        routes: list[Route] = []
        vehicle_id = 1

        for rb_route in best_sol:
            raw_nodes = list(rb_route)
            middle = [
                node.vertex_strid for node in raw_nodes
                if node.vertex_strid in customer_id_set or node.vertex_strid in station_id_set
            ]
            if not any(sid in customer_id_set for sid in middle):
                continue

            stops: list[Stop] = []
            stops.append(Stop(
                sumo_node_id=sumo_ids[depot_idx],
                node_name=names[depot_idx],
                node_type=types[depot_idx],
                planned_fuel=fuel_capacity,
                planned_time=0.0,
            ))
            for sid in middle:
                phys_idx = rb_id_to_idx[sid]
                if sid in customer_id_set:
                    node_type = "customer"
                else:
                    node_type = "afs" if phys_idx != depot_idx else "depot_dummy"
                stops.append(Stop(
                    sumo_node_id=sumo_ids[phys_idx],
                    node_name=names[phys_idx],
                    node_type=node_type,
                    planned_fuel=0.0,
                    planned_time=0.0,
                ))
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
                "Try increasing heuristic.alns_iterations or checking fuel capacity."
            )

        return routes
