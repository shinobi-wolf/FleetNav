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

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration
from dronesim.trsp.agents import get_agent
from dronesim.trsp.traffic import PSLTrafficModel
from dronesim.trsp.agents.dqn import DQNAgent
from dronesim.trsp.agents.ppo import PPOAgent
from dronesim.trsp.agents.sarsa import SARSAAgent

INF = 100000


def is_traffic_enabled(config: dict) -> bool:
    traffic_cfg = config.get('traffic')
    if traffic_cfg is None:
        return False
    return bool(traffic_cfg.get('enabled', True))


def get_centroid(shape):
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

    edge_ids = [data['id'] for _, _, data in G.edges(data=True)]
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

    return {
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
        'edge_ids': edge_ids,
    }




def append_result_row(results_path, start_idx, target_idx, warm_up_time, tour_start_time,
                      total_time_taken, path_length, path_edges, traffic_segment_start,
                      traffic_segment_end, traffic_speed_start_ms, traffic_speed_end_ms, failed_error):
    with open(results_path, 'a') as f:
        f.write(
            f"{start_idx},{target_idx},{warm_up_time},{tour_start_time},{total_time_taken},"
            f"{path_length},{path_edges},{traffic_segment_start},{traffic_segment_end},"
            f"{traffic_speed_start_ms},{traffic_speed_end_ms},{failed_error}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="TRSP Evaluation")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()

    if not args.policy:
        raise ValueError("Please provide a policy ID with -p/--policy")

    config_path = f"runs/trsp/{args.policy}/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_counter = 0
    while os.path.exists(f"runs/trsp/{args.policy}/test_{test_counter}"):
        test_counter += 1
    test_dir = f"runs/trsp/{args.policy}/test_{test_counter}"
    os.makedirs(test_dir, exist_ok=True)

    test_config = config.copy()
    test_config['test'] = {'mode': 'road_vehicle'}
    with open(f"{test_dir}/config.yaml", 'w') as f:
        yaml.dump(test_config, f)

    results_path = f"{test_dir}/simulation_results.csv"
    with open(results_path, 'w') as f:
        f.write(
            "start_idx,target_idx,warm_up_time,tour_start_time,total_time_taken,path_length,path_edges,"
            "traffic_segment_start,traffic_segment_end,traffic_speed_start_ms,traffic_speed_end_ms,failed_error\n"
        )

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
    junction_list = graph_info['junction_list']
    junc2idx = graph_info['junc2idx']
    node2idx = graph_info['node2idx']
    idx2node = graph_info['idx2node']
    sg_pairs = graph_info['sg_pairs']

    nodes = sorted(list(G.nodes))
    num_nodes = len(nodes)
    max_steps = config['simulation']['max_steps']
    max_time = float(config['simulation']['max_time']) * 3600.0
    step_length = config['sumo']['step_length']
    traffic_scale = config['sumo'].get('traffic_scale', 1.0)
    num_actions = config['net']['max_degree']
    warm_up_time = float(config['simulation'].get('warm_up_time', 1000))

    traffic_enabled = is_traffic_enabled(config)
    edge_ids = graph_info['edge_ids']
    psl_model = None

    if traffic_enabled:
        psl_model = PSLTrafficModel(
            config['traffic'], edge_ids, seed=config['traffic'].get('seed')
        )
        print(
            f"Traffic model: PSL  K={psl_model.K}  "
            f"v=[{psl_model.v_min*3.6:.0f},{psl_model.v_max*3.6:.0f}]km/h  "
            f"delta_t={psl_model.delta_t:.1f}s"
        )
    else:
        print("Traffic model: DISABLED (no speed override)")

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

    warm_up_times = [warm_up_time]
    route_times = []
    sim_times = []

    print(f"\nStarting TRSP evaluation with {num_nodes} nodes")
    overall_start_time = time.time()

    for start_idx, start_node in tqdm(idx2node.items(), desc="Start Nodes", position=0):
        for target_idx, target_node in tqdm(idx2node.items(), desc="  Targets", position=1, leave=False):
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
                        'time_slot': psl_model.get_slot(0.0) if traffic_enabled else 0,
                    }
                    action = int(agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx))

                    if action >= len(nb_node_idxs):
                        action = best_dir

                    next_pos_idx = nb_node_idxs[action]
                    next_pos = idx2node[next_pos_idx]

                    route_edges.append(G_dir.edges[curr_node, next_pos]['id'])
                    route_dirs.append(0 if (curr_node, next_pos) in G_dir.edges else 1)

                    curr_step += 1
                    curr_pos = node2idx[next_pos]
                    curr_node = next_pos
            except KeyError:
                append_result_row(results_path, start_idx, target_idx,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 'KeyError')
                continue

            route_times.append(time.time() - route_start_time)

            if curr_node == target_node and route_edges:
                for warmup_time in tqdm(warm_up_times, desc="    Warmup", position=2, leave=False):
                    sim_start_time_real = time.time()
                    traci.start(sumo_config)
                    sim_time = 0.0
                    while sim_time < warmup_time:
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    veh_id = f"ugv_{start_idx}_{target_idx}_{int(warmup_time)}"
                    route_id = f"{veh_id}_route"

                    if traffic_enabled:
                        traffic_segment_start = psl_model.get_slot(sim_time)
                        first_edge = route_edges[0]
                        traffic_speed_start = (
                            psl_model.get_speed(first_edge, traffic_segment_start)
                            if first_edge in psl_model.profiles else -1.0
                        )
                    else:
                        traffic_segment_start = -1
                        traffic_speed_start = -1.0

                    try:
                        traci.route.add(route_id, route_edges)
                        depart_spd = traffic_speed_start if traffic_enabled else 'max'
                        traci.vehicle.add(veh_id, route_id, departSpeed=depart_spd)
                    except traci.TraCIException:
                        append_result_row(results_path, start_idx, target_idx,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, "Route can't be added")
                        traci.close()
                        continue

                    if traffic_enabled:
                        traci.vehicle.setSpeed(veh_id, traffic_speed_start)
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
                    traci.simulationStep()
                    sim_time = traci.simulation.getTime()
                    sim_start_time = sim_time

                    while veh_id in traci.vehicle.getIDList() and sim_time < max_time:
                        if traffic_enabled:
                            slot = psl_model.get_slot(sim_time)
                            road_id = traci.vehicle.getRoadID(veh_id)
                            if not road_id.startswith(':') and road_id in psl_model.profiles:
                                traci.vehicle.setSpeed(veh_id, psl_model.get_speed(road_id, slot))
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    if traffic_enabled:
                        traffic_segment_end = psl_model.get_slot(sim_time)
                        last_edge = route_edges[-1]
                        traffic_speed_end = (
                            psl_model.get_speed(last_edge, traffic_segment_end)
                            if last_edge in psl_model.profiles else -1.0
                        )
                    else:
                        traffic_segment_end = -1
                        traffic_speed_end = -1.0
                    total_time_taken = sim_time - sim_start_time
                    sim_times.append(time.time() - sim_start_time_real)

                    append_result_row(results_path, start_idx, target_idx, warmup_time, sim_start_time,
                                      total_time_taken, curr_step, ';'.join(route_edges),
                                      traffic_segment_start, traffic_segment_end,
                                      traffic_speed_start, traffic_speed_end, 'NA')
                    traci.close()
            else:
                append_result_row(results_path, start_idx, target_idx,
                                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 'No Path')

    overall_elapsed = time.time() - overall_start_time
    print("\n" + "=" * 60)
    print(f"TRSP EVALUATION COMPLETE — {timedelta(seconds=int(overall_elapsed))}")
    print("=" * 60)
    if route_times:
        print(f"Route calc: avg {np.mean(route_times)*1000:.2f}ms over {len(route_times)} routes")
    if sim_times:
        print(f"SUMO sims:  avg {np.mean(sim_times):.2f}s over {len(sim_times)} simulations")
    print(f"Results saved to: {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
