import copy
import random
import sys
import time

# Ensure the installed routingblocks package is used, not a local source tree
sys.path = [p for p in sys.path if not p.endswith('RoutingBlocks')]

import networkx as nx
import routingblocks as rb
from sklearn.cluster import KMeans


def mtsp_planner(all_tasks, num_patrols, patrol_positions, G, edge_weight, R):
    """
    Assigns tasks to patrols via KMeans clustering, then sequences each patrol's
    tasks by selecting the closest reachable task and computing the initial path.

    Parameters
    ----------
    all_tasks : list of (int, int)
        All task positions sampled from the maze.
    num_patrols : int
        Number of patrol vehicles (EVs).
    patrol_positions : list of (int, int)
        Current position of each patrol (typically the warehouse at init).
    G : nx.Graph
        The maze graph.
    edge_weight : float
        Cost multiplier per graph edge.
    R : float
        Full battery range of an EV.

    Returns
    -------
    patrol_tasks : list of list of (int, int)
        Tasks assigned to each patrol after clustering.
    backcost : list of list of float
        Shortest-path distances from each patrol's start to each of its tasks.
    active_tasks : list of (int, int) or None
        The first target task selected for each patrol.
    patrol_paths : list of list of (int, int)
        The initial BFS path (excluding start node) for each patrol.
    """
    kmeans = KMeans(n_clusters=num_patrols).fit(all_tasks)
    task_clusters = kmeans.labels_

    patrol_tasks = [[] for _ in range(num_patrols)]
    backcost = [[] for _ in range(num_patrols)]
    active_tasks = []
    patrol_paths = [[] for _ in range(num_patrols)]

    # Distribute tasks into clusters
    for i, t in enumerate(all_tasks):
        cluster_idx = task_clusters[i]
        patrol_tasks[cluster_idx].append(t)

    # For each patrol, find the closest task and build the initial path
    for i in range(num_patrols):
        current_pos = patrol_positions[i]
        closest_dist = float('inf')
        closest_task = None

        for task in patrol_tasks[i]:
            try:
                dist = nx.shortest_path_length(
                    G, source=current_pos, target=(task[0], task[1])
                )
                backcost[i].append(dist)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_task = (task[0], task[1])
            except nx.NetworkXNoPath:
                continue

        active_tasks.append(closest_task)
        if closest_task is not None and 2 * (closest_dist * edge_weight) <= R:
            patrol_paths[i] = nx.shortest_path(
                G, source=current_pos, target=closest_task
            )[1:]
        else:
            patrol_paths[i] = []

    return patrol_tasks, backcost, active_tasks, patrol_paths

# def mtsp_planner_1(all_tasks, num_patrols, patrol_positions, G, edge_weight, R, warehouse_pos):
#     """
#     EVRP planner that assigns and sequences tasks while respecting battery
#     constraints and using the warehouse as the sole charging station.

#     Tasks are first partitioned among patrols via KMeans clustering (same as
#     mtsp_planner).  Within each patrol the tasks are then ordered using a
#     greedy nearest-neighbour heuristic:

#         while unvisited tasks remain:
#             1. Among all unvisited tasks, find the nearest one from the
#                current position such that the EV can reach it AND still
#                return to the warehouse on its remaining battery.
#             2. If such a task exists  -> visit it (update position & battery).
#             3. If no feasible task    -> return to warehouse, recharge to
#                full, and retry from step 1.
#             4. If after a fresh recharge no task is feasible, the remaining
#                tasks are unreachable within range and are dropped.

#     The ordered task list is returned so that the parallel ``backcost`` array
#     (warehouse-to-task distance for each task) is aligned with the visit
#     sequence, giving the runtime ``lmd_simulate`` the best ordering to work
#     with.

#     Parameters
#     ----------
#     all_tasks : list of (int, int)
#         All task positions sampled from the maze.
#     num_patrols : int
#         Number of patrol vehicles (EVs).
#     patrol_positions : list of (int, int)
#         Current position of each patrol (typically the warehouse at init).
#     G : nx.Graph
#         The maze graph.
#     edge_weight : float
#         Cost multiplier per graph edge.
#     R : float
#         Full battery range of an EV.
#     warehouse_pos : (int, int)
#         The warehouse / depot position used as the charging station.

#     Returns
#     -------
#     patrol_tasks : list of list of (int, int)
#         Tasks assigned to each patrol, ordered in the planned visit sequence.
#     backcost : list of list of float
#         For each task in patrol_tasks[i], the shortest-path distance from the
#         warehouse to that task (= return cost on an undirected graph).
#     active_tasks : list of (int, int) or None
#         The first target task selected for each patrol.
#     patrol_paths : list of list of (int, int)
#         The initial shortest path (excluding start node) for each patrol.
#     """

#     # --- 1. Cluster tasks among patrols (spatial partitioning) -------------
#     kmeans = KMeans(n_clusters=num_patrols).fit(all_tasks)
#     task_clusters = kmeans.labels_

#     raw_tasks = [[] for _ in range(num_patrols)]
#     for i, t in enumerate(all_tasks):
#         raw_tasks[task_clusters[i]].append(t)

#     # --- 2. Pre-compute warehouse ↔ task distances -------------------------
#     #     Cache shortest-path lengths to avoid redundant computations.
#     _dist_cache = {}

#     def _sp_length(src, dst):
#         key = (src, dst)
#         if key not in _dist_cache:
#             try:
#                 _dist_cache[key] = nx.shortest_path_length(G, source=src, target=dst)
#             except nx.NetworkXNoPath:
#                 _dist_cache[key] = float('inf')
#         return _dist_cache[key]

#     # --- 3. Greedy nearest-neighbour sequencing per patrol ------------------
#     patrol_tasks = [[] for _ in range(num_patrols)]
#     backcost = [[] for _ in range(num_patrols)]
#     active_tasks = []
#     patrol_paths = [[] for _ in range(num_patrols)]

#     for i in range(num_patrols):
#         remaining = list(raw_tasks[i])
#         current_pos = patrol_positions[i]  # starts at warehouse
#         remaining_battery = R
#         ordered_tasks = []
#         ordered_backcost = []

#         while remaining:
#             # Find the nearest feasible task:
#             #   feasible = can reach task AND return to warehouse afterwards
#             best_task = None
#             best_dist = float('inf')
#             best_return = 0.0

#             for task in remaining:
#                 d_to_task = _sp_length(current_pos, (task[0], task[1]))
#                 d_task_to_wh = _sp_length((task[0], task[1]), warehouse_pos)

#                 total_cost = (d_to_task + d_task_to_wh) * edge_weight
#                 if total_cost <= remaining_battery and d_to_task < best_dist:
#                     best_dist = d_to_task
#                     best_task = task
#                     best_return = d_task_to_wh

#             if best_task is not None:
#                 # Visit this task
#                 ordered_tasks.append(best_task)
#                 ordered_backcost.append(best_return)  # warehouse ↔ task
#                 remaining_battery -= best_dist * edge_weight
#                 current_pos = (best_task[0], best_task[1])
#                 remaining.remove(best_task)
#             else:
#                 # No feasible task from current position – return to warehouse
#                 # and recharge.
#                 d_to_wh = _sp_length(current_pos, warehouse_pos)
#                 if d_to_wh * edge_weight > remaining_battery:
#                     # Cannot even reach the warehouse – drop remaining tasks
#                     break

#                 current_pos = warehouse_pos
#                 remaining_battery = R

#                 # After recharging, check if ANY remaining task is reachable
#                 any_feasible = False
#                 for task in remaining:
#                     d_to_task = _sp_length(warehouse_pos, (task[0], task[1]))
#                     d_task_to_wh = _sp_length((task[0], task[1]), warehouse_pos)
#                     if (d_to_task + d_task_to_wh) * edge_weight <= R:
#                         any_feasible = True
#                         break

#                 if not any_feasible:
#                     # Remaining tasks are unreachable within range – drop them
#                     break

#         patrol_tasks[i] = ordered_tasks
#         backcost[i] = ordered_backcost

#         # --- 4. Set the initial active task and path -----------------------
#         if ordered_tasks:
#             first_task = ordered_tasks[0]
#             start_pos = patrol_positions[i]
#             d = _sp_length(start_pos, (first_task[0], first_task[1]))
#             d_ret = _sp_length((first_task[0], first_task[1]), warehouse_pos)

#             if (d + d_ret) * edge_weight <= R:
#                 try:
#                     patrol_paths[i] = nx.shortest_path(
#                         G, source=start_pos, target=(first_task[0], first_task[1])
#                     )[1:]
#                 except nx.NetworkXNoPath:
#                     patrol_paths[i] = []
#                 active_tasks.append((first_task[0], first_task[1]))
#             else:
#                 patrol_paths[i] = []
#                 active_tasks.append(None)
#         else:
#             active_tasks.append(None)

#     return patrol_tasks, backcost, active_tasks, patrol_paths

def evrp_planner(all_tasks, num_patrols, patrol_positions, G, edge_weight, R, warehouse_pos,
                 alns_iterations=200):
    """
    EVRP planner using RoutingBlocks ALNS on the maze graph.

    Solves an Electric Vehicle Routing Problem where:
      - The warehouse is both depot and the sole charging station.
      - Battery is measured in edge_weight-scaled units (each hop costs
        ``edge_weight``; full battery = ``R``).
      - The ADPTW model handles battery feasibility and station insertion.
      - Objective: minimise **makespan** (max route cost) with a penalty
        for using fewer routes than ``num_patrols``.

    After ALNS optimisation the routes are translated back into the same
    4-tuple format that ``mtsp_planner`` returns so ``lmd_simulate`` works
    without modification.

    Parameters
    ----------
    all_tasks : list of (int, int)
        All task positions sampled from the maze.
    num_patrols : int
        Number of patrol vehicles (EVs).
    patrol_positions : list of (int, int)
        Current position of each patrol (typically the warehouse at init).
    G : nx.Graph
        The maze graph (unweighted; each edge = 1 hop).
    edge_weight : float
        Battery cost per hop.
    R : float
        Full battery range of an EV (in edge_weight-scaled units).
    warehouse_pos : (int, int)
        Depot / charging-station position.
    alns_iterations : int
        Number of ALNS iterations (default 200).

    Returns
    -------
    patrol_tasks : list of list of (int, int)
        Tasks assigned to each patrol, ordered by planned visit sequence.
    backcost : list of list of float
        ``backcost[i][j]`` = shortest-path hop distance from
        ``patrol_tasks[i][j]`` back to the warehouse.
    active_tasks : list of (int, int) or None
        The first target task for each patrol.
    patrol_paths : list of list of (int, int)
        Initial BFS path (excluding start node) for each patrol.
    """
    # ------------------------------------------------------------------
    # Makespan helpers
    # ------------------------------------------------------------------
    def _makespan(solution):
        """Max route cost across all routes."""
        costs = [route.cost for route in solution]
        return max(costs) if costs else 0.0

    def _makespan_with_penalty(solution, fleet_count, penalty=1e6):
        """Makespan + penalty for using fewer routes than *fleet_count*."""
        non_empty = sum(1 for route in solution if len(route) > 2)
        ms = _makespan(solution)
        if non_empty < fleet_count:
            ms += penalty * (fleet_count - non_empty)
        return ms

    # ------------------------------------------------------------------
    # 0. String-ID helpers  (rb needs string vertex IDs)
    #    Suffixes: _d = depot, _c = customer, _s = station
    # ------------------------------------------------------------------
    def _node_to_str(n):
        return f"{n[0]},{n[1]}"

    def _str_to_node(s):
        parts = s.split(",")
        return (int(parts[0]), int(parts[1]))

    base_depot_str = _node_to_str(warehouse_pos)
    depot_str = base_depot_str + "_d"
    station_str = base_depot_str + "_s"
    customer_strs = [_node_to_str(t) + "_c" for t in all_tasks]

    # Mapping: suffixed string ID → (int, int) tuple
    str_to_tuple = {depot_str: warehouse_pos, station_str: warehouse_pos}
    for t, s in zip(all_tasks, customer_strs):
        str_to_tuple[s] = t

    # print("\n" + "="*60)
    # print("EVRP PLANNER — RoutingBlocks ALNS")
    # print("="*60)
    # print(f"  Depot/Station : {warehouse_pos}")
    # print(f"  Tasks         : {len(all_tasks)}")
    # print(f"  EVs (patrols) : {num_patrols}")
    # print(f"  Battery range : {R}  (edge_weight={edge_weight})")
    # print(f"  Max hops/full : {R / edge_weight:.0f}")
    # print(f"  ALNS iters    : {alns_iterations}")

    # ------------------------------------------------------------------
    # 1. All-pairs shortest paths between relevant nodes
    # ------------------------------------------------------------------
    # print("\n[1/5] Computing all-pairs shortest paths ...")
    relevant_tuples = [warehouse_pos] + list(all_tasks)
    seen = set()
    unique_tuples = []
    for n in relevant_tuples:
        if n not in seen:
            seen.add(n)
            unique_tuples.append(n)

    # Single-source shortest-path lengths from each relevant node (unweighted)
    sp_raw = {}   # (tuple, tuple) -> int  (raw hops)
    for src in unique_tuples:
        lengths = nx.single_source_shortest_path_length(G, src)
        for dst in unique_tuples:
            hops = lengths.get(dst, None)
            if src == dst:
                sp_raw[(src, dst)] = 0
            elif hops is not None:
                sp_raw[(src, dst)] = hops
            else:
                sp_raw[(src, dst)] = 10**9

    n_pairs = len(sp_raw)
    unreachable = sum(1 for v in sp_raw.values() if v >= 10**9)
    # print(f"  {n_pairs} pairs computed, {unreachable} unreachable")

    # Build string-keyed sp_dist scaled by edge_weight for rb arcs.
    BIG = 1e9
    all_str_ids = [depot_str] + customer_strs + [station_str]

    sp_dist = {}
    for a_str in all_str_ids:
        a_tup = str_to_tuple[a_str]
        for b_str in all_str_ids:
            b_tup = str_to_tuple[b_str]
            raw = sp_raw.get((a_tup, b_tup), 10**9)
            sp_dist[(a_str, b_str)] = raw * edge_weight if raw < 10**9 else BIG

    # ------------------------------------------------------------------
    # 2. Build RoutingBlocks Instance
    # ------------------------------------------------------------------
    # print("\n[2/5] Building RoutingBlocks instance ...")
    # print(f"  Vertices: 1 depot + {len(customer_strs)} customers + 1 station = {len(all_str_ids)}")
    # print(f"  Arcs: {len(all_str_ids)}x{len(all_str_ids)} = {len(all_str_ids)**2}")
    BIG_TIME = 1e7
    builder = rb.utility.InstanceBuilder(
        create_vertex=rb.adptw.create_adptw_vertex,
        create_arc=rb.adptw.create_adptw_arc,
    )

    # Depot
    dx, dy = float(warehouse_pos[0]), float(warehouse_pos[1])
    builder.set_depot(
        depot_str,
        rb.adptw.VertexData(dx, dy, 0.0, 0.0, BIG_TIME, 0.0),
    )

    # Customers (one per task)
    for cid in customer_strs:
        t = str_to_tuple[cid]
        builder.add_customer(
            cid,
            rb.adptw.VertexData(float(t[0]), float(t[1]), 1.0, 0.0, BIG_TIME, 0.0),
        )

    # Station (virtual copy of warehouse for mid-route recharging)
    builder.add_station(
        station_str,
        rb.adptw.VertexData(dx, dy, 0.0, 0.0, BIG_TIME, 0.0),
    )

    # Arcs (full N×N matrix required by RoutingBlocks)
    for i_id in all_str_ids:
        for j_id in all_str_ids:
            dist = sp_dist.get((i_id, j_id), BIG)
            builder.add_arc(
                i_id, j_id,
                rb.adptw.ArcData(dist, dist, dist),
            )

    builder.fleet_size = num_patrols
    instance = builder.build()

    battery_capacity = float(R)
    # print(f"  Instance built: {instance.number_of_vertices} vertices "
    #       f"({instance.number_of_customers} customers, "
    #       f"{instance.number_of_stations} stations), "
    #       f"fleet_size={instance.fleet_size}, battery_cap={battery_capacity}")

    # ------------------------------------------------------------------
    # 3. ALNS Solver
    # ------------------------------------------------------------------
    # print(f"\n[3/5] Running ALNS ({alns_iterations} iterations) ...")
    vehicle_storage_capacity = 1e7
    evaluation = rb.adptw.Evaluation(battery_capacity, vehicle_storage_capacity)
    evaluation.overload_penalty_factor = 100.0
    evaluation.resource_penalty_factor = 100.0
    evaluation.time_shift_penalty_factor = 100.0

    # Local search
    local_search = rb.LocalSearch(
        instance, evaluation, None, rb.BestImprovementPivotingRule()
    )
    arc_set = rb.ArcSet(instance.number_of_vertices)
    ls_operators = [
        rb.operators.SwapOperator_0_1(instance, arc_set),
        rb.operators.SwapOperator_0_2(instance, arc_set),
        rb.operators.SwapOperator_1_1(instance, arc_set),
        rb.operators.InsertStationOperator(instance),
        rb.operators.RemoveStationOperator(instance),
    ]

    # ALNS operators
    randgen = rb.Random(time.time_ns())
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

    # Facility placement optimizer
    fpo = rb.adptw.FacilityPlacementOptimizer(instance, battery_capacity)

    # Initial solution: round-robin across exactly num_patrols routes
    customer_ids = [v.vertex_id for v in instance.customers]
    random.shuffle(customer_ids)
    n_cust = len(customer_ids)
    k = min(num_patrols, n_cust)
    buckets = [[] for _ in range(k)]
    for i, cid in enumerate(customer_ids):
        buckets[i % k].append(cid)
    init_routes = []
    for bucket in buckets:
        if bucket:
            init_routes.append(rb.create_route(evaluation, instance, bucket))
    best_solution = rb.Solution(evaluation, instance, init_routes)
    best_ms = _makespan_with_penalty(best_solution, num_patrols)
    n_init_routes = sum(1 for r in best_solution if len(r) > 2)
    # print(f"  Initial solution: makespan={best_ms:.1f}, "
    #       f"sum_cost={best_solution.cost:.1f}, "
    #       f"routes={n_init_routes}/{num_patrols}, "
    #       f"feasible={best_solution.feasible}")

    # Main ALNS loop (makespan-based acceptance)
    for it in range(1, alns_iterations + 1):
        current = copy.copy(best_solution)

        n_customers_in_sol = sum(
            max(0, len(route) - 2) for route in current
        )
        if n_customers_in_sol == 0:
            continue
        n_remove = max(1, int(random.uniform(0.2, 0.4) * n_customers_in_sol))

        picked_ops = alns.generate(evaluation, current, n_remove)
        local_search.optimize(current, ls_operators)

        # Periodically run facility placement optimizer
        if it % 50 == 0 and best_solution.feasible:
            try:
                opt_routes = []
                for route in best_solution:
                    vids = [node.vertex_id for node in route]
                    opt_vids = fpo.optimize(vids)
                    opt_routes.append(
                        rb.create_route(evaluation, instance, opt_vids[1:-1])
                    )
                fpo_solution = rb.Solution(
                    evaluation, instance,
                    [r for r in opt_routes if not r.empty],
                )
                fpo_ms = _makespan_with_penalty(fpo_solution, num_patrols)
                if fpo_ms < best_ms:
                    best_solution = fpo_solution
                    best_ms = fpo_ms
                    # print(f"    [FPO iter {it}] Improved makespan to {best_ms:.1f}")
            except Exception:
                pass

        # Acceptance (makespan-based)
        current_ms = _makespan_with_penalty(current, num_patrols)
        if current_ms < best_ms:
            best_solution = current
            best_ms = current_ms
            alns.collect_score(*picked_ops, 4)
            n_active = sum(1 for r in best_solution if len(r) > 2)
            # print(f"    iter {it:>4d}: new best makespan={best_ms:.1f}, "
            #       f"routes={n_active}/{num_patrols}, "
            #       f"feasible={best_solution.feasible}")
        else:
            alns.collect_score(*picked_ops, 0)

        if it % 20 == 0:
            alns.adapt_operator_weights()

    final_routes = sum(1 for r in best_solution if len(r) > 2)
    # print(f"  ALNS complete: makespan={best_ms:.1f}, "
    #       f"sum_cost={best_solution.cost:.1f}, "
    #       f"routes={final_routes}/{num_patrols}, "
    #       f"feasible={best_solution.feasible}")

    # ------------------------------------------------------------------
    # 4. Extract routes → planner return format
    # ------------------------------------------------------------------
    # print(f"\n[4/5] Extracting routes ...")
    customer_str_set = set(customer_strs)
    route_task_lists = []
    for route in best_solution:
        tasks_in_route = []
        for node in route:
            sid = node.vertex_strid
            if sid in customer_str_set:
                tasks_in_route.append(str_to_tuple[sid])
        if tasks_in_route:
            route_task_lists.append(tasks_in_route)

    # Pad or merge to exactly num_patrols routes
    patrol_tasks = [[] for _ in range(num_patrols)]
    backcost_out = [[] for _ in range(num_patrols)]
    active_tasks = []
    patrol_paths = [[] for _ in range(num_patrols)]

    for i in range(min(len(route_task_lists), num_patrols)):
        patrol_tasks[i] = route_task_lists[i]

    # If ALNS produced more routes than patrols, merge extras
    for i in range(num_patrols, len(route_task_lists)):
        min_idx = min(range(num_patrols), key=lambda x: len(patrol_tasks[x]))
        patrol_tasks[min_idx].extend(route_task_lists[i])

    total_assigned = sum(len(pt) for pt in patrol_tasks)
    # print(f"  {total_assigned}/{len(all_tasks)} tasks assigned across {num_patrols} patrols")
    # for i in range(num_patrols):
    #     print(f"    EV {i}: {len(patrol_tasks[i])} tasks")

    # ------------------------------------------------------------------
    # 5. Compute backcost, active_tasks, patrol_paths
    # ------------------------------------------------------------------
    # print(f"\n[5/5] Computing backcosts & initial paths ...")
    for i in range(num_patrols):
        bc = []
        for task in patrol_tasks[i]:
            raw = sp_raw.get((task, warehouse_pos), None)
            if raw is None:
                try:
                    raw = nx.shortest_path_length(G, source=task, target=warehouse_pos)
                except nx.NetworkXNoPath:
                    raw = 10**9
            bc.append(raw)
        backcost_out[i] = bc

        if patrol_tasks[i]:
            first_task = patrol_tasks[i][0]
            current_pos = patrol_positions[i]
            try:
                dist_to_task = nx.shortest_path_length(G, source=current_pos, target=first_task)
            except nx.NetworkXNoPath:
                dist_to_task = 10**9

            dist_back = bc[0] if bc else 10**9

            if (dist_to_task + dist_back) * edge_weight <= R:
                try:
                    patrol_paths[i] = nx.shortest_path(G, source=current_pos, target=first_task)[1:]
                except nx.NetworkXNoPath:
                    patrol_paths[i] = []
                active_tasks.append(first_task)
            else:
                active_tasks.append(None)
                patrol_paths[i] = []
        else:
            active_tasks.append(None)
            patrol_paths[i] = []

    # print("\n--- Final Assignment ---")
    # for i in range(num_patrols):
    #     task_str = " -> ".join(f"({t[0]},{t[1]})" for t in patrol_tasks[i]) if patrol_tasks[i] else "(none)"
    #     active_str = f"({active_tasks[i][0]},{active_tasks[i][1]})" if active_tasks[i] else "None"
    #     print(f"  EV {i}: {len(patrol_tasks[i])} tasks [{task_str}], "
    #           f"active={active_str}, path_len={len(patrol_paths[i])}, "
    #           f"backcosts={backcost_out[i]}")
    # print("="*60 + "\n")

    return patrol_tasks, backcost_out, active_tasks, patrol_paths