import argparse
import os
import time
from collections import defaultdict
from datetime import timedelta

import networkx as nx
import numpy as np
import traci
import sumolib
import yaml
from tqdm import tqdm

from agents import get_agent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.sarsa import SARSAAgent

INF = 100000


def get_centroid(shape):
    """Calculates the center point of a shape (list of x,y tuples)."""
    if not shape:
        return 0, 0
    x_coords = [p[0] for p in shape]
    y_coords = [p[1] for p in shape]
    return np.mean(x_coords), np.mean(y_coords)


def build_graph(net_path):
    net_path = net_path.replace('.sumocfg', '.net.xml')
    net = sumolib.net.readNet(net_path, withPedestrianConnections=False)

    G = nx.DiGraph()
    junc2nodes_map = defaultdict(list)

    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(':'):
            continue

        from_node = edge.getFromNode()
        to_node = edge.getToNode()

        from_shape = from_node.getShape()
        to_shape = to_node.getShape()
        if not from_shape or not to_shape:
            continue

        from_cx, from_cy = get_centroid(from_shape)
        to_cx, to_cy = get_centroid(to_shape)

        G.add_node(from_node.getID(), junc_id=from_node.getID(), x=from_cx, y=from_cy)
        G.add_node(to_node.getID(), junc_id=to_node.getID(), x=to_cx, y=to_cy)

        if from_node.getID() not in junc2nodes_map[from_node.getID()]:
            junc2nodes_map[from_node.getID()].append(from_node.getID())
        if to_node.getID() not in junc2nodes_map[to_node.getID()]:
            junc2nodes_map[to_node.getID()].append(to_node.getID())

        G.add_edge(from_node.getID(), to_node.getID(), id=edge_id, length=edge.getLength(), type='road')

    G_und = G.to_undirected()

    all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    all_shortest_path_lengths_dir = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    num_junctions = len(junc2nodes_map)
    junction_list = list(junc2nodes_map.keys())
    junc2idx = {junc: idx for idx, junc in enumerate(sorted(junction_list))}
    idx2junc = {idx: junc for junc, idx in junc2idx.items()}
    node2idx = {node: idx for idx, node in enumerate(sorted(G_und.nodes))}
    idx2node = {idx: node for node, idx in node2idx.items()}

    sg_pairs = []
    for node_idx in range(len(idx2node)):
        for junc_id in junction_list:
            node = idx2node[node_idx]
            if G_und.nodes[node]['junc_id'] != junc_id:
                sg_pairs.append((node_idx, junc2idx[junc_id]))

    graph_info = {
        'G': G_und,
        'G_dir': G,
        'all_shortest_path_lengths': all_shortest_path_lengths,
        'all_shortest_path_lengths_dir': all_shortest_path_lengths_dir,
        'junc2nodes_map': junc2nodes_map,
        'num_junctions': num_junctions,
        'junction_list': junction_list,
        'junc2idx': junc2idx,
        'idx2junc': idx2junc,
        'node2idx': node2idx,
        'idx2node': idx2node,
        'sg_pairs': sg_pairs,
    }
    return graph_info


def main():
    parser = argparse.ArgumentParser(description="Road Vehicle Simulation Configuration")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()

    if not args.policy:
        raise ValueError("Please provide a policy ID with -p/--policy")

    config_path = f"runs/{args.policy}/config.yaml"
    config = yaml.safe_load(open(config_path, 'r'))

    test_counter = 0
    while os.path.exists(f"runs/{args.policy}/test_{test_counter}"):
        test_counter += 1
    os.makedirs(f"runs/{args.policy}/test_{test_counter}", exist_ok=True)
    test_dir = f"runs/{args.policy}/test_{test_counter}"

    test_config = config.copy()
    test_config['test'] = {'mode': 'road_vehicle'}
    with open(f"{test_dir}/config.yaml", 'w') as f:
        yaml.dump(test_config, f)

    results_path = f"{test_dir}/simulation_results.csv"
    with open(results_path, 'w') as f:
        f.write("start_idx,target_idx,warm_up_time,tour_start_time,total_time_taken,path_length,path_edges,failed_error\n")

    if config['agent']['algo'] == 'dqn':
        agent = DQNAgent(config)
    elif config['agent']['algo'] == 'ppo':
        agent = PPOAgent(config)
    elif config['agent']['algo'] == 'sarsa':
        agent = SARSAAgent(config)
    else:
        agent = get_agent(None, config['agent'], 0)

    net_path = config['net']['net_path']
    graph_info = build_graph(net_path)
    G = graph_info['G']
    G_dir = graph_info['G_dir']
    all_shortest_path_lengths = graph_info['all_shortest_path_lengths']
    all_shortest_path_lengths_dir = graph_info['all_shortest_path_lengths_dir']
    junc2nodes_map = graph_info['junc2nodes_map']
    junction_list = graph_info['junction_list']
    junc2idx = graph_info['junc2idx']
    node2idx = graph_info['node2idx']
    idx2node = graph_info['idx2node']
    sg_pairs = graph_info['sg_pairs']

    nodes = sorted(list(G.nodes))
    num_nodes = len(nodes)
    max_steps = config['simulation']['max_steps']
    max_time = config['simulation']['max_time']
    step_length = config['sumo']['step_length']
    traffic_scale = config['sumo'].get('traffic_scale', 1.0)
    num_actions = config['net']['max_degree']

    sumo_config = [
        'sumo',
        '-c', config['net']['net_path'],
        '--step-length', str(step_length),
        '--scale', str(traffic_scale),
        '--start',
        '--no-warnings',
        '--no-step-log',
        '--no-duration-log',
    ]

    warm_up_times = [1000]

    route_times = []
    sim_times = []

    print(f"\nStarting road vehicle simulation with {num_nodes} nodes")

    overall_start_time = time.time()

    for start_idx, start_node in tqdm(idx2node.items(), desc="Start Nodes", position=0):
        for target_idx, target_node in tqdm(idx2node.items(), desc="  Targets (Start={start_idx})", position=1, leave=False):
            if start_node == target_node:
                continue

            route_start_time = time.time()
            curr_step = 0
            curr_pos = start_idx
            curr_node = start_node

            route_edges = []
            route_dirs = []

            try:
                while curr_step < max_steps and curr_node != target_node:
                    nb_nodes = list(G_dir.neighbors(curr_node))
                    nb_node_idxs = sorted([node2idx[nb] for nb in nb_nodes])[:num_actions]
                    if not nb_node_idxs:
                        break

                    steps2dest = []
                    for nb_node_idx in nb_node_idxs:
                        nb_node = idx2node[nb_node_idx]
                        if target_node in all_shortest_path_lengths_dir.get(nb_node, {}):
                            steps2dest.append(all_shortest_path_lengths_dir[nb_node][target_node])
                        else:
                            steps2dest.append(INF)
                    steps2dest += [INF] * (num_actions - len(steps2dest))

                    junction_idx = junc2idx[G.nodes[target_node]['junc_id']]
                    sg_idx = sg_pairs.index((curr_pos, junction_idx))
                    best_dir = int(np.argmin(steps2dest))

                    obs = {
                        'graph_node': curr_pos,
                        'target_node': junction_idx,
                        'best_dir': best_dir,
                        'steps2dest': np.array(steps2dest, dtype=np.float32),
                        'sg_pair_idx': sg_idx,
                    }
                    action = int(agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx))

                    if action >= len(nb_node_idxs):
                        action = best_dir

                    next_pos_idx = nb_node_idxs[action]
                    next_pos = idx2node[next_pos_idx]

                    route_edges.append(G_dir.edges[curr_node, next_pos]['id'])
                    if (curr_node, next_pos) not in G_dir.edges:
                        route_dirs.append(1)
                    else:
                        route_dirs.append(0)

                    curr_step += 1
                    curr_pos = node2idx[next_pos]
                    curr_node = next_pos
            except KeyError:
                with open(results_path, 'a') as f:
                    f.write(f"{start_idx},{target_idx},-1,-1,-1,-1,-1,KeyError\n")
                continue

            route_calc_time = time.time() - route_start_time
            route_times.append(route_calc_time)

            if curr_node == target_node and route_edges:
                for warmup_time in tqdm(warm_up_times, desc="    Warmup times", position=2, leave=False):
                    sim_start_time_real = time.time()
                    traci.start(sumo_config)
                    sim_warmup_time = warmup_time
                    sim_time = 0.0

                    while sim_time < sim_warmup_time:
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    veh_id = f"ugv_{start_idx}_{target_idx}_{int(sim_warmup_time)}"
                    route_id = f"{veh_id}_route"

                    try:
                        traci.route.add(route_id, route_edges)
                        traci.vehicle.add(veh_id, route_id)
                    except traci.TraCIException:
                        print(f"Error adding vehicle for route {start_node} -> {target_node} with warmup {sim_warmup_time}s. Skipping simulation.")
                        with open(results_path, 'a') as f:
                            f.write(f"{start_idx},{target_idx},-1,-1,-1,-1,-1,Route can't be added\n")
                        traci.close()
                        continue
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))

                    traci.simulationStep()
                    sim_time = traci.simulation.getTime()
                    sim_start_time = sim_time

                    while veh_id in traci.vehicle.getIDList() and sim_time < max_time:
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    total_time_taken = sim_time - sim_start_time

                    sim_elapsed_time = time.time() - sim_start_time_real
                    sim_times.append(sim_elapsed_time)

                    route_edges_str = ';'.join(route_edges)
                    with open(results_path, 'a') as f:
                        f.write(
                            f"{start_idx},{target_idx},{sim_warmup_time},{sim_start_time},{total_time_taken},{curr_step},{route_edges_str},NA\n"
                        )
                    traci.close()
            else:
                with open(results_path, 'a') as f:
                    f.write(f"{start_idx},{target_idx},-1,-1,-1,-1,-1,No Path\n")

    overall_elapsed_time = time.time() - overall_start_time

    print("\n" + "=" * 60)
    print("ROAD VEHICLE SIMULATION COMPLETE - Performance Statistics")
    print("=" * 60)
    print(f"Total elapsed time: {timedelta(seconds=int(overall_elapsed_time))}")
    print(f"Total elapsed time (seconds): {overall_elapsed_time:.2f}s")

    if route_times:
        avg_route_time = np.mean(route_times)
        print("\nRoute Calculation:")
        print(f"  Average time per route: {avg_route_time * 1000:.2f}ms")
        print(f"  Total routes calculated: {len(route_times)}")
        print(f"  Total route calc time: {sum(route_times):.2f}s")

    if sim_times:
        avg_sim_time = np.mean(sim_times)
        print("\nSUMO Simulations:")
        print(f"  Average time per simulation: {avg_sim_time:.2f}s")
        print(f"  Total simulations run: {len(sim_times)}")
        print(f"  Total simulation time: {sum(sim_times):.2f}s")
        print(f"  Min simulation time: {min(sim_times):.2f}s")
        print(f"  Max simulation time: {max(sim_times):.2f}s")

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
