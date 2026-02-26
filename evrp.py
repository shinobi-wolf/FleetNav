
import copy
import json
import os
import random
import sys
import time

# Ensure the installed routingblocks package is used, not the local source tree
sys.path = [p for p in sys.path if not p.endswith('RoutingBlocks')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sumolib
import yaml
import routingblocks as rb


# ---------------------------------------------------------------------------
# Makespan helper
# ---------------------------------------------------------------------------

def makespan(solution):
    """Return the makespan (max route cost) of a solution."""
    costs = [route.cost for route in solution]
    return max(costs) if costs else 0.0


def makespan_with_penalty(solution, fleet_count, penalty=1e6):
    """
    Makespan objective with a penalty for using fewer routes than *fleet_count*.

    This discourages the ALNS destroy/repair from collapsing routes.
    """
    non_empty = sum(1 for route in solution if len(route) > 2)
    ms = makespan(solution)
    if non_empty < fleet_count:
        ms += penalty * (fleet_count - non_empty)
    return ms


# ---------------------------------------------------------------------------
# 3b. Build nx graph from SUMO network
# ---------------------------------------------------------------------------

def get_centroid(shape):
    """Return the centroid (avg x, avg y) of a shape polygon."""
    xs = [p[0] for p in shape]
    ys = [p[1] for p in shape]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def build_graph(net_path):
    """
    Build an undirected networkx graph from a SUMO network.

    Derives the ``.net.xml`` path from the ``.sumocfg`` path, parses the
    network with *sumolib*, and creates a graph whose nodes are the
    walking-area entry points of each (non-internal) edge.  Node positions
    are the centroid of the edge shape.

    Returns
    -------
    G_undirected : nx.Graph
    node2idx : dict  – graph-node-id → integer index
    idx2node : dict  – integer index → graph-node-id
    """
    net_xml_path = net_path.replace('.sumocfg', '.net.xml')
    net = sumolib.net.readNet(net_xml_path, withPedestrianConnections=False)

    G = nx.Graph()
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(':'):
            continue
        n1 = edge.getToNode()
        shape = n1.getShape()
        if not shape:
            continue
        cx, cy = get_centroid(shape)
        G.add_node(n1.getID(), x=cx, y=cy)
        n2 = edge.getFromNode()
        shape2 = n2.getShape()
        if not shape2:
            continue
        cx2, cy2 = get_centroid(shape2)
        G.add_node(n2.getID(), x=cx2, y=cy2)

        G.add_edge(n1.getID(), n2.getID(), length=edge.getLength())

    node2idx = {}
    idx2node = {}
    for i, node in enumerate(G.nodes()):
        node2idx[node] = i
        idx2node[i] = node

    return G, node2idx, idx2node


# ---------------------------------------------------------------------------
# Shortest-path computation between relevant vertices
# ---------------------------------------------------------------------------

def compute_shortest_paths(G, relevant_nodes):
    """
    Compute all-pairs shortest path distances and paths between
    *relevant_nodes* on graph *G* using Dijkstra (weight='length').

    Returns
    -------
    sp_dist : dict[(str,str) -> float]
    sp_path : dict[(str,str) -> list[str]]   (node-id sequences)
    """
    BIG = 1e9
    sp_dist = {}
    sp_path = {}

    # pre-compute from each relevant source
    for src in relevant_nodes:
        lengths, paths = nx.single_source_dijkstra(G, src, weight='length')
        for dst in relevant_nodes:
            if src == dst:
                sp_dist[(src, dst)] = 0.0
                sp_path[(src, dst)] = [src]
            elif dst in lengths:
                sp_dist[(src, dst)] = lengths[dst]
                sp_path[(src, dst)] = paths[dst]
            else:
                sp_dist[(src, dst)] = BIG
                sp_path[(src, dst)] = []

    return sp_dist, sp_path


# ---------------------------------------------------------------------------
# Build RoutingBlocks Instance
# ---------------------------------------------------------------------------

def build_rb_instance(G, depot_node, customer_nodes, station_nodes,
                      sp_dist, fleet_count, fleet_range, max_speed):
    """
    Construct a routingblocks Instance using the ADPTW evaluation model.

    Battery is modelled in *distance* units:
      - consumption on an arc = shortest-path distance
      - battery_capacity_time = fleet_range  (max distance on a full charge)
      - travel_time = distance / max_speed

    Time windows are disabled (set to [0, very large]).
    Demand is set to 1 for customers, 0 for depot/stations; vehicle storage
    capacity is set very large so the capacity constraint is inactive.

    Returns
    -------
    instance : rb.Instance
    battery_capacity_time : float
    """
    BIG_TIME = 1e7
    builder = rb.utility.InstanceBuilder(
        create_vertex=rb.adptw.create_adptw_vertex,
        create_arc=rb.adptw.create_adptw_arc,
    )

    # Make Vertiex IDs by appending suffixes to the original node IDs, to distinguish
    depot_node = depot_node + '_d'
    customer_nodes = [n + '_c' for n in customer_nodes]
    station_nodes = [n + '_s' for n in station_nodes]

    # Collect all vertex string IDs in the order we'll register them
    all_vertex_ids = [depot_node] + customer_nodes + station_nodes

    # -- depot --
    dx = G.nodes[depot_node[:-2]]['x']
    dy = G.nodes[depot_node[:-2]]['y']
    builder.set_depot(
        depot_node,
        rb.adptw.VertexData(dx, dy, 0.0, 0.0, BIG_TIME, 0.0),
    )

    # -- customers --
    for cid in customer_nodes:
        cx = G.nodes[cid[:-2]]['x']
        cy = G.nodes[cid[:-2]]['y']
        builder.add_customer(
            cid,
            rb.adptw.VertexData(cx, cy, 1.0, 0.0, BIG_TIME, 0.0),
        )

    # -- stations --
    for sid in station_nodes:
        sx = G.nodes[sid[:-2]]['x']
        sy = G.nodes[sid[:-2]]['y']
        builder.add_station(
            sid,
            rb.adptw.VertexData(sx, sy, 0.0, 0.0, BIG_TIME, 0.0),
        )

    # -- arcs (N×N, required by RoutingBlocks) --
    for i_id in all_vertex_ids:
        for j_id in all_vertex_ids:
            dist = sp_dist.get((i_id[:-2], j_id[:-2]), 1e9)
            travel_time = dist / max_speed if max_speed > 0 else dist
            # consumption = distance  (battery measured in distance units)
            consumption = dist
            builder.add_arc(i_id, j_id, rb.adptw.ArcData(dist, consumption, travel_time))

    builder.fleet_size = fleet_count
    instance = builder.build()

    # battery_capacity_time = fleet_range (in distance units, matching consumption)
    battery_capacity_time = float(fleet_range)

    return instance, battery_capacity_time


# ---------------------------------------------------------------------------
# Create a random initial solution
# ---------------------------------------------------------------------------

def create_random_solution(evaluation, instance, fleet_count):
    """Split customers round-robin across exactly *fleet_count* routes."""
    customer_ids = [v.vertex_id for v in instance.customers]
    random.shuffle(customer_ids)

    n = len(customer_ids)
    k = min(fleet_count, n)
    # round-robin assignment to guarantee k routes with balanced sizes
    buckets = [[] for _ in range(k)]
    for i, cid in enumerate(customer_ids):
        buckets[i % k].append(cid)
    routes = []
    for bucket in buckets:
        if bucket:
            routes.append(rb.create_route(evaluation, instance, bucket))
    return rb.Solution(evaluation, instance, routes)


# ---------------------------------------------------------------------------
# ALNS solver
# ---------------------------------------------------------------------------

def solve_alns(instance, fleet_count, battery_capacity_time,
               num_iterations=300,
               min_removal=0.2, max_removal=0.4):
    """
    Solve the fuel-constrained VRP using Adaptive Large Neighborhood Search.

    Returns the best Solution found.
    """
    # -- evaluation --
    vehicle_storage_capacity = 1e7          # effectively unconstrained
    evaluation = rb.adptw.Evaluation(battery_capacity_time, vehicle_storage_capacity)
    evaluation.overload_penalty_factor = 100.0
    evaluation.resource_penalty_factor = 100.0
    evaluation.time_shift_penalty_factor = 100.0

    # -- local search --
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

    # -- ALNS operators --
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

    # -- facility placement optimizer (optimal station insertion) --
    fpo = rb.adptw.FacilityPlacementOptimizer(instance, battery_capacity_time)

    # -- initial solution --
    best_solution = create_random_solution(evaluation, instance, fleet_count)
    best_ms = makespan_with_penalty(best_solution, fleet_count)
    print(f"Initial solution: makespan={best_ms:.2f}, sum_cost={best_solution.cost:.2f}, "
          f"feasible={best_solution.feasible}, "
          f"routes={sum(1 for r in best_solution if len(r)>2)}")

    # -- main ALNS loop --
    for it in range(1, num_iterations + 1):
        current = copy.copy(best_solution)

        # number of customer vertices to remove
        n_customers_in_sol = sum(
            max(0, len(route) - 2) for route in current
        )
        if n_customers_in_sol == 0:
            continue
        n_remove = max(1, int(random.uniform(min_removal, max_removal) * n_customers_in_sol))

        # destroy + repair
        picked_ops = alns.generate(evaluation, current, n_remove)

        # local search
        local_search.optimize(current, ls_operators)

        # periodically run facility placement optimiser on best solution
        if it % 50 == 0 and best_solution.feasible:
            try:
                opt_routes = []
                for route in best_solution:
                    vids = [node.vertex_id for node in route]
                    opt_vids = fpo.optimize(vids)
                    # optimize() returns ids including depot bookends → strip them
                    opt_routes.append(
                        rb.create_route(evaluation, instance, opt_vids[1:-1])
                    )
                fpo_solution = rb.Solution(
                    evaluation, instance,
                    [r for r in opt_routes if not r.empty],
                )
                fpo_ms = makespan_with_penalty(fpo_solution, fleet_count)
                if fpo_ms < best_ms:
                    best_solution = fpo_solution
                    best_ms = fpo_ms
                    print(f"  [FPO] Improved: makespan={best_ms:.2f}, "
                          f"feasible={best_solution.feasible}")
            except Exception:
                pass  # FPO may fail on degenerate routes; skip silently

        # acceptance (makespan-based)
        current_ms = makespan_with_penalty(current, fleet_count)
        if current_ms < best_ms:
            best_solution = current
            best_ms = current_ms
            alns.collect_score(*picked_ops, 4)
            print(f"  iter {it:>4d}: new best makespan={best_ms:.2f}, "
                  f"routes={sum(1 for r in best_solution if len(r)>2)}, "
                  f"feasible={best_solution.feasible}")
        else:
            alns.collect_score(*picked_ops, 0)

        # adapt weights
        if it % 20 == 0:
            alns.adapt_operator_weights()

    return best_solution, evaluation


# ---------------------------------------------------------------------------
# Extract route information
# ---------------------------------------------------------------------------

def extract_routes(solution):
    """
    Return a list of dicts, each with:
      - 'vertex_ids' : list of vertex string-ids (including depot bookends)
      - 'cost'       : float
      - 'feasible'   : bool
    """
    routes_info = []
    for route in solution:
        vids = [node.vertex_strid for node in route]
        routes_info.append({
            'vertex_ids': vids,
            'cost': route.cost,
            'feasible': route.feasible,
        })
    return routes_info


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_routes(G, depot_node, customer_nodes, station_nodes,
                routes_info, sp_path, save_path='routes.png'):
    """Draw routes on the network graph and save to *save_path*."""
    pos = {n: (G.nodes[n]['x'], G.nodes[n]['y'])
           for n in G.nodes() if 'x' in G.nodes[n]}

    fig, ax = plt.subplots(figsize=(10, 10))

    # draw full graph faintly
    nx.draw_networkx_edges(G, pos, alpha=0.10, ax=ax)

    # colour palette for routes
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colours = [cmap(i % 10) for i in range(len(routes_info))]

    for idx, rinfo in enumerate(routes_info):
        vids = rinfo['vertex_ids']
        colour = colours[idx]
        # expand each consecutive pair through the shortest-path edges
        route_edges = []
        for a, b in zip(vids[:-1], vids[1:]):
            path = sp_path.get((a[:-2], b[:-2]), [])
            for u, v in zip(path[:-1], path[1:]):
                route_edges.append((u, v))
        if route_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=route_edges,
                edge_color=[colour] * len(route_edges),
                width=2.5, alpha=0.8, ax=ax,
                label=f"Route {idx} (cost={rinfo['cost']:.1f})",
            )

    # draw nodes by type
    depot_pos = {depot_node: pos[depot_node]}
    cust_pos = {n: pos[n] for n in customer_nodes if n in pos}
    stat_pos = {n: pos[n] for n in station_nodes if n in pos}
    other_pos = {n: p for n, p in pos.items()
                 if n not in cust_pos and n not in stat_pos and n != depot_node}

    nx.draw_networkx_nodes(G, other_pos, nodelist=list(other_pos.keys()),
                           node_size=15, node_color='lightgrey', ax=ax)
    nx.draw_networkx_nodes(G, cust_pos, nodelist=list(cust_pos.keys()),
                           node_size=80, node_color='dodgerblue',
                           edgecolors='black', label='Customers', ax=ax)
    nx.draw_networkx_nodes(G, stat_pos, nodelist=list(stat_pos.keys()),
                           node_size=120, node_color='limegreen',
                           node_shape='s',
                           edgecolors='black', label='Stations', ax=ax)
    nx.draw_networkx_nodes(G, depot_pos, nodelist=[depot_node],
                           node_size=200, node_color='red',
                           node_shape='*',
                           edgecolors='black', label='Depot', ax=ax)

    # labels for special nodes (only those in pos)
    special = {}
    for d in (cust_pos, stat_pos, depot_pos):
        special.update(d)
    special = {n: p for n, p in special.items() if n in pos}
    nx.draw_networkx_labels(G, special, labels={n: n for n in special},
                            font_size=6, ax=ax)

    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Vehicle Routes (Fuel-Constrained VRP)')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Route plot saved to {save_path}")
    plt.close(fig)


# ===================================================================
# MAIN
# ===================================================================

if __name__ == '__main__':
    # ---- 3a. Config loading ------------------------------------------------
    with open('example2.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    fleet_count  = cfg['fleet']['count']
    fleet_range  = cfg['fleet']['range']
    max_speed    = cfg['fleet'].get('max_speed', 5)
    net_path     = cfg['net']['net_path']
    depot_cfg    = cfg['net']['depot']
    demand_per   = cfg['task']['demand_per']

    random.seed(42)
    np.random.seed(42)

    # ---- 3b. Build nx graph ------------------------------------------------
    G, node2idx, idx2node = build_graph(net_path)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- 3c. Sample task & station nodes -----------------------------------
    all_nodes = list(G.nodes())
    num_customers = 8

    num_charging_stations = 4

    # depot_node = 'C2'

    # customer_nodes = ['A1', 'A3', 'B0', 'B4', 'D0', 'D4', 'E1', 'E3']
    # station_nodes  = ['B1', 'B3', 'D1', 'D3']

    depot_node = 'C2'

    customer_nodes = ['A0', 'A4', 'E0', 'E4']
    station_nodes  = ['C2']

    print(f"Depot      : {depot_node}")
    print(f"Customers  : {len(customer_nodes)}")
    print(f"Stations   : {len(station_nodes)}  {station_nodes}")
    print(f"Fleet      : {fleet_count} vehicles, range={fleet_range}")

    # ---- 3d. Compute shortest paths ----------------------------------------
    relevant_nodes = [depot_node] + customer_nodes + station_nodes
    print("Computing shortest paths between relevant vertices …")
    sp_dist, sp_path = compute_shortest_paths(G, relevant_nodes)
    print(f"  {len(sp_dist)} shortest-path pairs computed.")

    # ---- 3e. Build RoutingBlocks instance & solve --------------------------
    print("Building RoutingBlocks instance …")
    instance, battery_cap = build_rb_instance(
        G, depot_node, customer_nodes, station_nodes,
        sp_dist, fleet_count, fleet_range, max_speed,
    )
    print(f"  Instance: {instance.number_of_vertices} vertices "
          f"({instance.number_of_customers} customers, "
          f"{instance.number_of_stations} stations), "
          f"fleet_size={instance.fleet_size}")

    print(f"\nRunning ALNS (battery_capacity={battery_cap}) …")
    best_solution, evaluation = solve_alns(
        instance, fleet_count, battery_cap, num_iterations=300,
    )

    # ---- 3f. Print & plot routes -------------------------------------------
    routes_info = extract_routes(best_solution)
    final_ms = makespan(best_solution)
    print(f"\n{'='*60}")
    print(f"BEST SOLUTION  —  makespan = {final_ms:.2f}, "
          f"sum_cost = {best_solution.cost:.2f}, "
          f"feasible = {best_solution.feasible}")
    print(f"{'='*60}")
    for i, r in enumerate(routes_info):
        print(f"  Route {i}: {' → '.join(r['vertex_ids'])}  "
              f"(cost={r['cost']:.2f}, feasible={r['feasible']})")

    plot_routes(G, depot_node, customer_nodes, station_nodes,
                routes_info, sp_path, save_path='routes.png')

    # ---- 3g. Store results -------------------------------------------------
    result = {
        'makespan': final_ms,
        'cost': best_solution.cost,
        'feasible': best_solution.feasible,
        'fleet_count': fleet_count,
        'fleet_range': fleet_range,
        'depot': depot_node,
        'customers': customer_nodes,
        'stations': station_nodes,
        'routes': [
            {'vertices': r['vertex_ids'], 'cost': r['cost'], 'feasible': r['feasible']}
            for r in routes_info
        ],
    }
    with open('solution.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Solution written to solution.json")