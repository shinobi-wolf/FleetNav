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

    results_path = f"runs/{args.policy}/simulation_results_big.csv"
    with open(results_path, 'w') as f:
        f.write("start_idx,target_idx,warm_up_time,tour_start_time,total_time_taken,path_length\n")
    
    env = gym.make('SumoTRSPEnv-v0', config=config)
    agent = get_agent(env, config['agent'], 0)

    net_path = config['net']['net_path']
    G, G_dir, all_shortest_path_lengths, junc2nodes_map = build_graph(net_path,)
    junction_list = list(junc2nodes_map.keys())
    node2idx = {node: idx for idx, node in enumerate(sorted(G.nodes))}
    idx2node = {idx: node for node, idx in node2idx.items()}

    nodes = sorted(list(G.nodes))
    num_nodes = len(nodes)
    all_sg_pairs = []

    
    print(f"\nStarting simulation with {num_nodes} nodes")
    
    overall_start_time = time.time()
    
    for start_idx, start_node in idx2node.items():
        for junc_id in junction_list:
            if G.nodes[start_node]['junc_id'] == junc_id:
                continue
            all_sg_pairs.append((start_idx, junc_id))

    for sg_idx, (start_idx, junc_id) in enumerate(tqdm(all_sg_pairs, desc="All Start-Goal Pairs", position=0)):
        sumo_start_time = 1000
        config['test'] = {}
        config['test']['warm_up_time'] = sumo_start_time
        config['test']['start_node_idx'] = start_idx
        config['test']['target_junc_id'] = junc_id

        env = gym.make('SumoTRSPEnv-v0', config=config)
        obs, world_state = env.reset(seed=42)
        truncated = False
        terminated = False
        while not terminated and not truncated:
            action = agent.predict(obs,G=G, idx2node=idx2node, node2idx=node2idx)
            print(f"Agent selected action: {action}")

            # Execute step
            obs, reward, terminated, truncated, world_state = env.step(action)
        
        total_time_taken = world_state.get("time_elapsed", -1)
        total_time_taken = total_time_taken - sumo_start_time if total_time_taken != -1 else -1
        if truncated:
            with open(results_path, 'a') as f:
                f.write(f"{start_idx},{junc_id},{sumo_start_time},{sumo_start_time},{total_time_taken}\n")
        else:
            with open(results_path, 'a') as f:
                f.write(f"{start_idx},{junc_id},-1,-1,-1\n")
        env.close()

    # # Calculate and display final statistics
    # overall_elapsed_time = time.time() - overall_start_time
    
    # print("\n" + "="*60)
    # print("SIMULATION COMPLETE - Performance Statistics")
    # print("="*60)
    # print(f"Total elapsed time: {timedelta(seconds=int(overall_elapsed_time))}")
    # print(f"Total elapsed time (seconds): {overall_elapsed_time:.2f}s")
    
    # if route_times:
    #     avg_route_time = np.mean(route_times)
    #     print(f"\nRoute Calculation:")
    #     print(f"  Average time per route: {avg_route_time*1000:.2f}ms")
    #     print(f"  Total routes calculated: {len(route_times)}")
    #     print(f"  Total route calc time: {sum(route_times):.2f}s")
    
    # if sim_times:
    #     avg_sim_time = np.mean(sim_times)
    #     print(f"\nSUMO Simulations:")
    #     print(f"  Average time per simulation: {avg_sim_time:.2f}s")
    #     print(f"  Total simulations run: {len(sim_times)}")
    #     print(f"  Total simulation time: {sum(sim_times):.2f}s")
    #     print(f"  Min simulation time: {min(sim_times):.2f}s")
    #     print(f"  Max simulation time: {max(sim_times):.2f}s")
    
    print(f"\nResults saved to: {results_path}")
    print("="*60)

    # env.close()
    # traci.close()

if __name__ == '__main__':
    main()