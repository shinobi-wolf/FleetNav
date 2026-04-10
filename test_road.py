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

from dronesim.agents import get_agent
from dronesim.agents.dqn import DQNAgent
from dronesim.agents.ppo import PPOAgent
from dronesim.agents.sarsa import SARSAAgent

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


def build_traffic_model(config, max_time):
    traffic_cfg = config.get('traffic')
    if traffic_cfg is None:
        raise ValueError("Missing 'traffic' section in config")

    required_keys = ['speed_limits', 'speed_step', 'seg_duration', 'seed']
    missing = [key for key in required_keys if key not in traffic_cfg]
    if missing:
        raise ValueError(f"Missing traffic config keys: {missing}")

    speed_limits = traffic_cfg['speed_limits']
    if not isinstance(speed_limits, (list, tuple)) or len(speed_limits) != 2:
        raise ValueError("traffic.speed_limits must be [min_kmh, max_kmh]")

    min_speed_kmh = float(speed_limits[0])
    max_speed_kmh = float(speed_limits[1])
    speed_step_kmh = float(traffic_cfg['speed_step'])
    seg_duration_min = float(traffic_cfg['seg_duration'])
    traffic_seed = int(traffic_cfg['seed'])

    if speed_step_kmh <= 0:
        raise ValueError("traffic.speed_step must be > 0")
    if seg_duration_min <= 0:
        raise ValueError("traffic.seg_duration must be > 0")
    if max_speed_kmh < min_speed_kmh:
        raise ValueError("traffic.speed_limits must satisfy min <= max")

    min_speed_ms = min_speed_kmh * 1000.0 / 3600.0
    max_speed_ms = max_speed_kmh * 1000.0 / 3600.0
    speed_step_ms = speed_step_kmh * 1000.0 / 3600.0
    seg_duration_s = seg_duration_min * 60.0

    traffic_speed_range = np.arange(min_speed_ms, max_speed_ms + 1e-9, speed_step_ms, dtype=np.float32)
    if traffic_speed_range.size == 0:
        raise ValueError("traffic speed range is empty; check speed limits and speed step")

    num_segments = max(1, int(np.ceil(max_time / seg_duration_s)))
    rng = np.random.RandomState(traffic_seed)
    traffic_speed_segments = rng.choice(traffic_speed_range, size=num_segments)

    return {
        'speed_limits_ms': (float(min_speed_ms), float(max_speed_ms)),
        'seg_duration_s': float(seg_duration_s),
        'speed_segments_ms': traffic_speed_segments,
        'num_segments': num_segments,
    }


def get_current_traffic_segment(current_time, traffic_seg_duration, traffic_speed_segments):
    if len(traffic_speed_segments) == 0:
        return -1
    if current_time < 0:
        return 0

    segment_index = int(current_time / traffic_seg_duration)
    return min(segment_index, len(traffic_speed_segments) - 1)


def get_current_traffic_speed(current_time, traffic_seg_duration, traffic_speed_segments):
    segment_index = get_current_traffic_segment(current_time, traffic_seg_duration, traffic_speed_segments)
    if segment_index < 0:
        return 0.0
    return float(traffic_speed_segments[segment_index])


def append_result_row(
    results_path,
    start_idx,
    target_idx,
    warm_up_time,
    tour_start_time,
    total_time_taken,
    path_length,
    path_edges,
    traffic_segment_start,
    traffic_segment_end,
    traffic_speed_start_ms,
    traffic_speed_end_ms,
    failed_error,
):
    with open(results_path, 'a') as f:
        f.write(
            f"{start_idx},{target_idx},{warm_up_time},{tour_start_time},{total_time_taken},"
            f"{path_length},{path_edges},{traffic_segment_start},{traffic_segment_end},"
            f"{traffic_speed_start_ms},{traffic_speed_end_ms},{failed_error}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Road Vehicle Simulation Configuration")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()

    if not args.policy:
        raise ValueError("Please provide a policy ID with -p/--policy")

    config_path = f"runs/{args.policy}/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    junc2nodes_map = graph_info['junc2nodes_map']
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

    traffic_model = build_traffic_model(config, max_time)
    traffic_seg_duration = traffic_model['seg_duration_s']
    traffic_speed_segments = traffic_model['speed_segments_ms']
    speed_min_ms, speed_max_ms = traffic_model['speed_limits_ms']

    print(
        "Traffic model:",
        f"segment_duration={traffic_seg_duration:.1f}s",
        f"segments={traffic_model['num_segments']}",
        f"speed_limits={speed_min_ms:.2f}-{speed_max_ms:.2f}m/s",
    )

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
                        # SARSAAgent reads time_slot from observation.
                        'time_slot': get_current_traffic_segment(0.0, traffic_seg_duration, traffic_speed_segments),
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
                append_result_row(
                    results_path,
                    start_idx,
                    target_idx,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    'KeyError',
                )
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
                    traffic_segment_start = get_current_traffic_segment(
                        sim_time,
                        traffic_seg_duration,
                        traffic_speed_segments,
                    )
                    traffic_speed_start = get_current_traffic_speed(
                        sim_time,
                        traffic_seg_duration,
                        traffic_speed_segments,
                    )

                    try:
                        traci.route.add(route_id, route_edges)
                        traci.vehicle.add(veh_id, route_id, departSpeed=traffic_speed_start)
                    except traci.TraCIException:
                        print(f"Error adding vehicle for route {start_node} -> {target_node} with warmup {sim_warmup_time}s. Skipping simulation.")
                        append_result_row(
                            results_path,
                            start_idx,
                            target_idx,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            "Route can't be added",
                        )
                        traci.close()
                        continue
                    traci.vehicle.setSpeed(veh_id, traffic_speed_start)
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))

                    traci.simulationStep()
                    sim_time = traci.simulation.getTime()
                    sim_start_time = sim_time

                    while veh_id in traci.vehicle.getIDList() and sim_time < max_time:
                        curr_speed = get_current_traffic_speed(
                            sim_time,
                            traffic_seg_duration,
                            traffic_speed_segments,
                        )
                        traci.vehicle.setSpeed(veh_id, curr_speed)
                        traci.simulationStep()
                        sim_time = traci.simulation.getTime()

                    traffic_segment_end = get_current_traffic_segment(
                        sim_time,
                        traffic_seg_duration,
                        traffic_speed_segments,
                    )
                    traffic_speed_end = get_current_traffic_speed(
                        sim_time,
                        traffic_seg_duration,
                        traffic_speed_segments,
                    )

                    total_time_taken = sim_time - sim_start_time

                    sim_elapsed_time = time.time() - sim_start_time_real
                    sim_times.append(sim_elapsed_time)

                    route_edges_str = ';'.join(route_edges)
                    append_result_row(
                        results_path,
                        start_idx,
                        target_idx,
                        sim_warmup_time,
                        sim_start_time,
                        total_time_taken,
                        curr_step,
                        route_edges_str,
                        traffic_segment_start,
                        traffic_segment_end,
                        traffic_speed_start,
                        traffic_speed_end,
                        'NA',
                    )
                    traci.close()
            else:
                append_result_row(
                    results_path,
                    start_idx,
                    target_idx,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    'No Path',
                )

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
