import argparse
import yaml
import os
import gymnasium as gym
import numpy as np
import time
import networkx as nx

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
        f.write("start_idx,target_idx,start_node,target_node,total_time_taken\n")
    
    env = gym.make('SumoTrainEnv-v0', config=config)
    env.reset(seed=42)
    agent = get_agent(env, config['agent'], 0)

    G = env.unwrapped.G
    all_shortest_path_lengths = env.unwrapped.all_shortest_path_lengths
    nodes = sorted(list(G.nodes))
    num_nodes = len(nodes)
    # num_nodes = 2
    max_steps = 25
    max_time = 10000
    step_length = 50
    traffic_scale = 1
    num_actions = 4

    Sumo_config = [
        'sumo',
        '-c', config["net"]["net_path"],
        '--step-length', str(step_length),
        '--scale', str(traffic_scale),
        # '--delay', '200',
        '--start',
        "--no-warnings"
    ]
    traci.close()
    traci.start(Sumo_config)

    sim_time = 0.0
    curr_step = 0

    for start_idx in range(num_nodes):
        for target_idx in range(num_nodes):
            if start_idx == target_idx:
                continue
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
            print(f"Start: {start_idx}, Target: {target_idx}, Route: {route_edges}")
            if curr_pos == target_idx:
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
                total_time_taken = sim_time - sim_start_time
                with open(results_path, 'a') as f:
                    f.write(f"{start_idx},{target_idx},{nodes[start_idx]},{nodes[target_idx]},{total_time_taken}\n")
                if sim_time >= max_time:
                    traci.close()
                    traci.start(Sumo_config)
                    sim_time = 0.0
            else:
                with open(results_path, 'a') as f:
                    f.write(f"{start_idx},{target_idx},{nodes[start_idx]},{nodes[target_idx]},-1\n")

    env.close()
    # traci.close()

if __name__ == '__main__':
    main()