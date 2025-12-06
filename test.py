import argparse
import yaml
import os
import gymnasium as gym
import numpy as np
import time
import networkx as nx
import random
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from collections import defaultdict

import traci
import sumolib

import envs
from agents import get_agent

def get_centroid(shape):
    """Calculates the center point of a shape (list of x,y tuples)."""
    if not shape:
        return 0, 0
    x_coords = [p[0] for p in shape]
    y_coords = [p[1] for p in shape]
    return np.mean(x_coords), np.mean(y_coords)

def build_graph(net_path,):
    net_path = net_path.replace('.sumocfg', '.net.xml')
    net = sumolib.net.readNet(net_path, withPedestrianConnections=True)
    junctions = net.getNodes()
    edge_eps = {}
    junc2nodes_map = defaultdict(list)
    
    for junction in junctions:
        connections = junction.getConnections()
        for connection in connections:
            # 1. Inspect the source of the connection
            ep = None
            edge = None
            dir = None
            if connection._from.getFunction() == 'walkingarea':
                ep = connection._from
                edge = connection._to
                dir = 0
            elif connection._to.getFunction() == 'walkingarea':
                ep = connection._to
                edge = connection._from
                dir = 1
            if ep and edge:
                edge_id = edge.getID()
                eps = [None, None]
                eps[dir] = ep
                if edge_id not in edge_eps:
                    edge_eps[edge_id] = {
                        "edge": edge,
                        "eps": eps
                    }
                else:
                    edge_eps[edge_id]["eps"][dir] = ep
                    if len([ep for ep in edge_eps[edge_id]["eps"] if ep is not None]) > 2:
                        print("More than 2 entry points for edge ", edge_id)
                        raise ValueError("More than 2 entry points for edge ", edge_id)

    G = nx.DiGraph()
    for edge_id, edge_data in edge_eps.items():
        entry_points = edge_data["eps"]
        for ep in entry_points:
            shape = ep.getShape()
            cx, cy = get_centroid(shape)
            G.add_node(ep._id, junc_id=ep._from._id,junc_coord=ep._from._coord, x=cx, y=cy)
            junc2nodes_map[ep._from._id].append(ep._id)
        edge_obj = edge_data["edge"]
        edge_length = None
        edge_type = edge_obj.getFunction()
        if edge_type == 'crossing':
            edge_length = 5.0
            G.add_edge(entry_points[0]._id, entry_points[1]._id, id=edge_obj._id, length=edge_length, type=edge_type)
            # G.add_edge(entry_points[1]._id, entry_points[0]._id, edge=edge_obj._id, length=edge_length, type=edge_type)
        else:
            edge_length = edge_obj.getLength()
            edge_type = 'sidewalk'
            G.add_edge(entry_points[0]._id, entry_points[1]._id, id=edge_obj._id, length=edge_length, type=edge_type)
    
    G_und = G.to_undirected()



    all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_und, weight='length'))
    return G_und, G, all_shortest_path_lengths, junc2nodes_map

def main():
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()

    config_path = f"runs/{args.policy}/config.yaml"
    config = yaml.safe_load(open(config_path, 'r'))

    test_counter = 0
    while os.path.exists(f"runs/{args.policy}/test_{test_counter}"):
        test_counter += 1
    os.makedirs(f"runs/{args.policy}/test_{test_counter}", exist_ok=True)
    test_dir = f"runs/{args.policy}/test_{test_counter}"

    test_config = config.copy()
    test_config['test'] = {
        'mode': 'smooth'
    }
    with open(f"{test_dir}/config.yaml", 'w') as f:
        yaml.dump(test_config, f)

    results_path = f"{test_dir}/simulation_results.csv"
    with open(results_path, 'w') as f:
        f.write("start_idx,target_idx,warm_up_time,tour_start_time,total_time_taken,path_length,path_edges\n")
    
    agent = get_agent(None, config['agent'], 0)

    net_path = config['net']['net_path']
    G, G_dir, all_shortest_path_lengths, junc2nodes_map = build_graph(net_path,)
    junction_list = list(junc2nodes_map.keys())
    node2idx = {node: idx for idx, node in enumerate(sorted(G.nodes))}
    idx2node = {idx: node for node, idx in node2idx.items()}

    nodes = sorted(list(G.nodes))
    num_nodes = len(nodes)
    max_steps = config['simulation']['max_steps']
    max_time = config['simulation']['max_time']
    step_length = config['sumo']['step_length']
    traffic_scale = config['sumo'].get('traffic_scale', 1.0)
    num_actions = config['net']['max_degree']

    Sumo_config = [
        'sumo',
        '-c', config["net"]["net_path"],
        '--step-length', str(step_length),
        '--scale', str(traffic_scale),
        # '--delay', '200',
        '--start',
        "--no-warnings",
        "--no-step-log",
        "--no-duration-log",
    ]
    # traci.close()

    warm_up_times = [1000,]

    # Timing statistics
    route_times = []
    sim_times = []
    
    print(f"\nStarting simulation with {num_nodes} nodes")
    
    overall_start_time = time.time()
    
    for start_idx, start_node in tqdm(idx2node.items(), desc="Start Nodes", position=0):
        for junc_id in tqdm(junction_list, desc=f"  Targets (Start={start_idx})", position=1, leave=False):
            if G.nodes[start_node]['junc_id'] == junc_id:
                continue
            # if start_idx!= 0 or junc_id != '1':
            #     continue

            route_start_time = time.time()
            curr_step = 0
            curr_pos = start_idx
            curr_node = start_node
            target_nodes = junc2nodes_map[junc_id]
            target_node = target_nodes[np.argmin([all_shortest_path_lengths[start_node][tn] for tn in target_nodes])]
            target_idx = node2idx[target_node]
            route_edges = []
            route_dirs = []
            while curr_step < max_steps and G.nodes[curr_node]['junc_id'] != G.nodes[target_node]['junc_id']:
                steps2dest = []
                nb_nodes = list(G.neighbors(curr_node))  # Get up to num_actions neighbors
                nb_node_idxs = sorted([node2idx[nb] for nb in nb_nodes])[:num_actions]
                for nb_node_idx in nb_node_idxs:
                    steps2dest.append(all_shortest_path_lengths[idx2node[nb_node_idx]][target_node])
                best_dir = np.argmin(steps2dest) if steps2dest else 4
                obs = {
                    'graph_node': curr_pos,
                    'target_node': int(G.nodes[target_node]['junc_id']),
                    'best_dir': best_dir
                }
                action = agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx)
                # print("Action",action)
                # action = best_dir
                next_pos_idx = nb_node_idxs[action]
                next_pos = idx2node[next_pos_idx]
                route_edges.append(G.edges[curr_node, next_pos]['id'])
                if (curr_node, next_pos) not in G_dir.edges:
                    route_dirs.append(1)
                else:
                    route_dirs.append(0)
                curr_step += 1
                curr_pos = node2idx[next_pos]
                curr_node = next_pos
            
            route_calc_time = time.time() - route_start_time
            route_times.append(route_calc_time)
            
            if G.nodes[curr_node]['junc_id'] == G.nodes[target_node]['junc_id']:
                for warmup_time in tqdm(warm_up_times, desc=f"    Warmup times", position=2, leave=False):
                    sim_start_time_real = time.time()
                    traci.start(Sumo_config)
                    sim_warmup_time = warmup_time
                    sim_time = 0.0
                    while sim_time < sim_warmup_time:
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    veh_id = f"ugv_{start_idx}_{target_idx}"
                    if route_dirs[0] == 0:
                        traci.person.add(veh_id, route_edges[0], 0.0)
                    else:
                        traci.person.add(veh_id, route_edges[0], -1)
                    traci.person.setColor(veh_id, (255, 0, 0, 255))
                    if route_dirs[-1] == 0:
                        traci.person.appendWalkingStage(veh_id, route_edges, 
                                                        arrivalPos=-1, speed=5.0)
                    else:
                        traci.person.appendWalkingStage(veh_id, route_edges, 
                                                        arrivalPos=0.0, speed=5.0)
                    traci.simulationStep()
                    sim_time = traci.simulation.getTime()
                    sim_start_time = sim_time
                    while veh_id in traci.person.getIDList() and sim_time < max_time:
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()
                    # if veh_id in traci.person.getIDList():
                    #     total_time_taken = -1
                    # else:
                    total_time_taken = sim_time - sim_start_time
                    
                    sim_elapsed_time = time.time() - sim_start_time_real
                    sim_times.append(sim_elapsed_time)
                    
                    route_edges_str = ";".join(route_edges)
                    with open(results_path, 'a') as f:
                        f.write(f"{start_idx},{junc_id},{sim_warmup_time},{sim_start_time},{total_time_taken},{curr_step},{route_edges_str}\n")
                    traci.close()
            else:
                with open(results_path, 'a') as f:
                    f.write(f"{start_idx},{junc_id},-1,-1,-1,-1,-1\n")

    # Calculate and display final statistics
    overall_elapsed_time = time.time() - overall_start_time
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - Performance Statistics")
    print("="*60)
    print(f"Total elapsed time: {timedelta(seconds=int(overall_elapsed_time))}")
    print(f"Total elapsed time (seconds): {overall_elapsed_time:.2f}s")
    
    if route_times:
        avg_route_time = np.mean(route_times)
        print(f"\nRoute Calculation:")
        print(f"  Average time per route: {avg_route_time*1000:.2f}ms")
        print(f"  Total routes calculated: {len(route_times)}")
        print(f"  Total route calc time: {sum(route_times):.2f}s")
    
    if sim_times:
        avg_sim_time = np.mean(sim_times)
        print(f"\nSUMO Simulations:")
        print(f"  Average time per simulation: {avg_sim_time:.2f}s")
        print(f"  Total simulations run: {len(sim_times)}")
        print(f"  Total simulation time: {sum(sim_times):.2f}s")
        print(f"  Min simulation time: {min(sim_times):.2f}s")
        print(f"  Max simulation time: {max(sim_times):.2f}s")
    
    print(f"\nResults saved to: {results_path}")
    print("="*60)

    # env.close()
    # traci.close()

if __name__ == '__main__':
    main()