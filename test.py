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

import traci

import envs
from agents import get_agent

def main():
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()

    config_path = f"runs/{args.policy}/config.yaml"
    config = yaml.safe_load(open(config_path, 'r'))

    results_path = f"runs/{args.policy}/simulation_results_dij.csv"
    with open(results_path, 'w') as f:
        f.write("start_idx,target_idx,start_node,target_node,warm_up_time,tour_start_time,total_time_taken\n")
    
    env = gym.make('SumoTrainEnv-v0', config=config)
    env.reset(seed=42)
    G = env.unwrapped.G
    all_shortest_path_lengths = env.unwrapped.all_shortest_path_lengths
    agent = get_agent(env, config['agent'], 0)

    nodes = sorted(list(G.nodes))
    num_nodes = len(nodes)
    # num_nodes = 2
    max_steps = 80
    max_time = 20000
    step_length = 500
    traffic_scale = 1
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
    traci.close()

    curr_step = 0
    warmup_time_idx = 0
    warm_up_times = [1000,5000,9000]

    # Timing statistics
    route_times = []
    sim_times = []
    total_routes = num_nodes * (num_nodes - 1)
    total_simulations = total_routes * len(warm_up_times)
    
    print(f"\nStarting simulation with {num_nodes} nodes")
    print(f"Total routes to test: {total_routes}")
    print(f"Total simulations (with warmup variations): {total_simulations}\n")
    
    overall_start_time = time.time()
    
    for start_idx in tqdm(range(num_nodes), desc="Start Nodes", position=0):
        for target_idx in tqdm(range(num_nodes), desc=f"  Targets (Start={start_idx})", position=1, leave=False):
            if start_idx == target_idx:
                continue

            route_start_time = time.time()
            curr_step = 0
            curr_pos = start_idx
            route_edges = []
            while curr_step < max_steps and curr_pos != target_idx:
                steps2dest = []
                target_pos = nodes[target_idx]
                nb_nodes = sorted(G.neighbors(nodes[curr_pos]))[:num_actions]  # Get up to num_actions neighbors
                for nb_node in nb_nodes:
                    steps2dest.append(all_shortest_path_lengths[nb_node][target_pos])
                best_dir = np.argmin(steps2dest) if steps2dest else 4
                obs = {
                    'graph_node': curr_pos,
                    'target_node': target_idx,
                    'best_dir': best_dir
                }
                # action = agent.predict(obs)
                action = best_dir
                next_pos = nb_nodes[action]
                route_edges.append(G.edges[nodes[curr_pos], next_pos]['id'])
                curr_step += 1
                curr_pos = nodes.index(next_pos)
            
            route_calc_time = time.time() - route_start_time
            route_times.append(route_calc_time)
            
            if curr_pos == target_idx:
                for warmup_time in tqdm(warm_up_times, desc=f"    Warmup times", position=2, leave=False):
                    sim_start_time_real = time.time()
                    traci.start(Sumo_config)
                    sim_warmup_time = warmup_time
                    sim_time = 0.0
                    while sim_time < sim_warmup_time:
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    veh_id = f"ugv_{start_idx}_{target_idx}"
                    traci.person.add(veh_id, route_edges[0], 0.0)
                    traci.person.setColor(veh_id, (255, 0, 0, 255))
                    traci.person.appendWalkingStage(veh_id, route_edges, 
                                                    arrivalPos=-1, speed=3.0)
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
                    
                    with open(results_path, 'a') as f:
                        f.write(f"{start_idx},{target_idx},{nodes[start_idx]},{nodes[target_idx]},{sim_warmup_time},{sim_start_time},{total_time_taken}\n")
                    traci.close()
            else:
                with open(results_path, 'a') as f:
                    f.write(f"{start_idx},{target_idx},{nodes[start_idx]},{nodes[target_idx]},-1,-1,-1\n")
            warmup_time_idx += 1

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