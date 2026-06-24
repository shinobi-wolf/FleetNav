import argparse
import copy
import os
import random
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
from dronesim.trsp.agents.a2c_psl      import A2CAgent
from dronesim.trsp.agents.a2c_svc      import A2CSVCAgent
from dronesim.trsp.agents.dijkstra_psl import DijkstraAgent
from dronesim.trsp.agents.dsp          import DSPAgent
from dronesim.trsp.agents.dqn_psl      import DQNAgent
from dronesim.trsp.agents.dqn_svc      import DQNSVCAgent
from dronesim.trsp.agents.ppo_psl      import PPOAgent
from dronesim.trsp.agents.ppo_svc      import PPOSVCAgent
from dronesim.trsp.agents.sarsa_psl    import SARSAAgent
from dronesim.trsp.agents.sarsa_svc    import SARSASVCAgent
from dronesim.trsp.traffic import PSLTrafficModel

INF = 100000


def get_traffic_model(config: dict) -> str:
    """Return 'svc', 'psl', or 'none'."""
    traffic_cfg = config.get('traffic')
    if traffic_cfg is None:
        return 'none'
    model = traffic_cfg.get('model', 'psl')
    if model == 'svc':
        return 'svc'
    if not traffic_cfg.get('enabled', True):
        return 'none'
    return 'psl'


def _neighbor_speeds_from_traci(curr_node, nb_node_idxs, G_dir, idx2node,
                                 ugv_speed: float, num_actions: int) -> np.ndarray:
    """Query TraCI for mean speed on each outgoing edge; normalize by ugv_speed."""
    speeds = []
    for nb_idx in nb_node_idxs:
        nb_node = idx2node[nb_idx]
        eid = G_dir.edges[curr_node, nb_node]['id']
        raw = traci.edge.getLastStepMeanSpeed(eid)
        raw = ugv_speed if raw < 0 else raw
        speeds.append(min(raw / ugv_speed, 2.0))
    speeds += [1.0] * (num_actions - len(speeds))
    return np.array(speeds, dtype=np.float32)


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

    all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_und, weight='length'))
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


def select_nodes(num_nodes, seed=42, n_pairs=100):
    """Select n_pairs+1 node indices with a fixed seed; no two consecutive are the same."""
    rng = random.Random(seed)
    all_idxs = list(range(num_nodes))
    selected = [rng.choice(all_idxs)]
    for _ in range(n_pairs):
        choices = [i for i in all_idxs if i != selected[-1]]
        selected.append(rng.choice(choices))
    return selected  # length n_pairs+1


def plan_route(agent, action_mode, start_node, target_node, time_slot,
               G, G_dir, idx2node, node2idx, junc2idx, sg_pairs,
               all_shortest_path_lengths_dir, max_steps, num_actions,
               traffic_model='psl', ugv_speed=None):
    """Plan a graph-level route using the agent. Returns (route_edges, error_str_or_None)."""
    curr_step = 0
    curr_pos = node2idx[start_node]
    curr_node = start_node
    route_edges = []

    if hasattr(agent, 'start_leg'):
        agent.start_leg(start_node, target_node)

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
            try:
                sg_idx = sg_pairs.index((curr_pos, junction_idx))
            except ValueError:
                sg_idx = 0
            best_dir = int(np.argmin(steps2dest))

            obs = {
                'graph_node': curr_pos,
                'target_node': junction_idx,
                'best_dir': best_dir,
                'steps2dest': np.array(steps2dest, dtype=np.float32),
                'sg_pair_idx': sg_idx,
            }
            if traffic_model == 'svc':
                obs['neighbor_speeds'] = _neighbor_speeds_from_traci(
                    curr_node, nb_node_idxs, G_dir, idx2node, ugv_speed, num_actions
                )
            else:
                obs['time_slot'] = time_slot

            action = int(agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx))

            if action_mode == 'node_id':
                if action not in nb_node_idxs:
                    action = nb_node_idxs[best_dir]
                next_pos_idx = action
            else:
                if action >= len(nb_node_idxs):
                    action = best_dir
                next_pos_idx = nb_node_idxs[action]

            next_pos = idx2node[next_pos_idx]
            route_edges.append(G_dir.edges[curr_node, next_pos]['id'])

            curr_step += 1
            curr_pos = node2idx[next_pos]
            curr_node = next_pos
    except KeyError:
        return [], 'KeyError'

    if curr_node != target_node or not route_edges:
        return [], 'No Path'

    return route_edges, None


def make_test_dir(base_dir):
    """Create and return the next test_N subdirectory under base_dir."""
    counter = 0
    while os.path.exists(f"{base_dir}/test_{counter}"):
        counter += 1
    test_dir = f"{base_dir}/test_{counter}"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


_CSV_HEADER = (
    "start_idx,target_idx,warm_up_time,tour_start_time,total_time_taken,path_length,path_edges,"
    "traffic_segment_start,traffic_segment_end,traffic_speed_start_ms,traffic_speed_end_ms,failed_error\n"
)


def init_results(test_dir, config):
    """Write config snapshot and initialize the policy CSV with headers."""
    with open(f"{test_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    results_path = f"{test_dir}/simulation_results.csv"
    with open(results_path, 'w') as f:
        f.write(_CSV_HEADER)
    return results_path


def init_baseline_csv(test_dir, suffix):
    """Initialize a baseline-results CSV (no config.yaml written)."""
    results_path = f"{test_dir}/simulation_results_{suffix}.csv"
    with open(results_path, 'w') as f:
        f.write(_CSV_HEADER)
    return results_path


def run_evaluation(agent, action_mode, selected_idxs, idx2node, graph_info, config,
                   sumo_config_cmd, psl_model, traffic_model, results_path, desc='Eval'):
    """
    Run all 100 OG pairs sequentially within a single continuous SUMO session.
    SUMO is only restarted when the simulation clock hits max_time before all pairs finish.
    This ensures traffic slots evolve naturally across pairs (critical for K>1 policies).
    """
    G = graph_info['G']
    G_dir = graph_info['G_dir']
    all_shortest_path_lengths_dir = graph_info['all_shortest_path_lengths_dir']
    junc2idx = graph_info['junc2idx']
    node2idx = graph_info['node2idx']
    sg_pairs = graph_info['sg_pairs']

    max_steps = config['simulation']['max_steps']
    max_time = float(config['simulation']['max_time']) * 3600.0
    num_actions = config['net']['max_degree']
    warmup_time = float(config['simulation'].get('warm_up_time', 0))

    ugv_speed = config['fleet']['max_speed'] * 1000.0 / 3600.0  # km/h → m/s
    psl_active = (traffic_model == 'psl')
    n_pairs = len(selected_idxs) - 1

    def start_sumo():
        traci.start(sumo_config_cmd)
        t = 0.0
        while t < warmup_time:
            traci.simulationStep()
            t = traci.simulation.getTime()
        return t

    sim_time = start_sumo()

    for pair_idx in tqdm(range(n_pairs), desc=desc):
        start_idx = selected_idxs[pair_idx]
        target_idx = selected_idxs[pair_idx + 1]
        start_node = idx2node[start_idx]
        target_node = idx2node[target_idx]

        current_slot = psl_model.get_slot(sim_time) if psl_active else 0
        route_edges, err = plan_route(
            agent, action_mode, start_node, target_node, current_slot,
            G, G_dir, idx2node, node2idx, junc2idx, sg_pairs,
            all_shortest_path_lengths_dir, max_steps, num_actions,
            traffic_model=traffic_model, ugv_speed=ugv_speed,
        )

        if err:
            append_result_row(results_path, start_idx, target_idx,
                              warmup_time, -1, -1, -1, -1, -1, -1, -1, -1, err)
            continue

        tour_start_time = sim_time
        if psl_active:
            traffic_segment_start = psl_model.get_slot(sim_time)
            first_edge = route_edges[0]
            traffic_speed_start = (
                psl_model.get_speed(first_edge, traffic_segment_start)
                if first_edge in psl_model.profiles else -1.0
            )
        else:
            traffic_segment_start = -1
            traffic_speed_start = -1.0

        veh_id = f"ugv_{pair_idx}"
        route_id = f"route_{pair_idx}"
        failed_error = 'NA'

        try:
            traci.route.add(route_id, route_edges)
            depart_spd = traffic_speed_start if (psl_active and traffic_speed_start > 0) else 'max'
            traci.vehicle.add(veh_id, route_id, departSpeed=depart_spd)
            if psl_active and traffic_speed_start > 0:
                traci.vehicle.setSpeed(veh_id, traffic_speed_start)
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
        except traci.TraCIException:
            append_result_row(results_path, start_idx, target_idx,
                              warmup_time, -1, -1, -1, -1, -1, -1, -1, -1, "Route can't be added")
            continue

        while veh_id in traci.vehicle.getIDList() and sim_time < max_time:
            if psl_active:
                slot = psl_model.get_slot(sim_time)
                road_id = traci.vehicle.getRoadID(veh_id)
                if not road_id.startswith(':') and road_id in psl_model.profiles:
                    traci.vehicle.setSpeed(veh_id, psl_model.get_speed(road_id, slot))
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

        if veh_id in traci.vehicle.getIDList():
            failed_error = 'Timeout'

        total_time_taken = sim_time - tour_start_time

        if psl_active:
            traffic_segment_end = psl_model.get_slot(sim_time)
            last_edge = route_edges[-1]
            traffic_speed_end = (
                psl_model.get_speed(last_edge, traffic_segment_end)
                if last_edge in psl_model.profiles else -1.0
            )
        else:
            traffic_segment_end = -1
            traffic_speed_end = -1.0

        append_result_row(
            results_path, start_idx, target_idx, warmup_time, tour_start_time,
            total_time_taken, len(route_edges), ';'.join(route_edges),
            traffic_segment_start, traffic_segment_end,
            traffic_speed_start, traffic_speed_end, failed_error
        )

        if sim_time >= max_time and pair_idx < n_pairs - 1:
            traci.close()
            sim_time = start_sumo()

    traci.close()


def main():
    parser = argparse.ArgumentParser(description="TRSP Sequential Evaluation (K>1 compatible)")
    parser.add_argument('-p', '--policy', type=str, required=True,
                        help="Policy ID under runs/trsp/")
    args = parser.parse_args()

    config_path = f"runs/trsp/{args.policy}/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    net_path = config['net']['net_path']
    print(f"Building graph from {net_path} ...")
    graph_info = build_graph(net_path)
    idx2node = graph_info['idx2node']
    num_nodes = len(idx2node)

    # 101 nodes → 100 OG pairs, reproducible and shared across policy + dijkstra runs
    selected_idxs = select_nodes(num_nodes, seed=42, n_pairs=100)
    print(f"Selected 101 nodes ({num_nodes} total) → 100 evaluation pairs")

    traffic_model = get_traffic_model(config)
    edge_ids = graph_info['edge_ids']
    psl_model = None
    if traffic_model == 'psl':
        psl_model = PSLTrafficModel(
            config['traffic'], edge_ids, seed=config['traffic'].get('seed')
        )
        print(
            f"Traffic model: PSL  K={psl_model.K}  "
            f"v=[{psl_model.v_min*3.6:.0f},{psl_model.v_max*3.6:.0f}]km/h  "
            f"delta_t={psl_model.delta_t:.1f}s"
        )
    elif traffic_model == 'svc':
        print(f"Traffic model: SVC  route_file={config['traffic']['route_file']}")
    else:
        print("Traffic model: DISABLED")

    step_length = config['sumo']['step_length']
    traffic_scale = config['sumo'].get('traffic_scale', 1.0)
    sumo_config_cmd = [
        'sumo',
        '-c', config['net']['net_path'],
        '--step-length', str(step_length),
        '--scale', str(traffic_scale),
        '--start',
        '--no-warnings',
        '--no-step-log',
        '--no-duration-log',
    ]
    if traffic_model == 'svc':
        sumo_config_cmd += ['--route-files', config['traffic']['route_file']]

    ugv_speed = config['fleet']['max_speed'] * 1000.0 / 3600.0  # km/h → m/s

    # --- Policy evaluation ---
    algo = config['agent']['algo']
    if algo == 'dqn':
        policy_agent = DQNAgent(config)
    elif algo == 'dqn_svc':
        policy_agent = DQNSVCAgent(config)
    elif algo == 'sarsa':
        policy_agent = SARSAAgent(config)
    elif algo == 'sarsa_svc':
        policy_agent = SARSASVCAgent(config)
    elif algo == 'ppo':
        policy_agent = PPOAgent(config)
    elif algo == 'ppo_svc':
        policy_agent = PPOSVCAgent(config)
    elif algo == 'a2c':
        policy_agent = A2CAgent(config)
    elif algo == 'a2c_svc':
        policy_agent = A2CSVCAgent(config)
    elif algo == 'dsp':
        policy_agent = DSPAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                                graph_info['idx2node'], ugv_speed)
    else:
        policy_agent = get_agent(config['agent'], 0)
    action_mode = config['agent'].get('action_mode', 'neighbor_index')
    if algo == 'dsp':
        action_mode = 'node_id'

    policy_dir = f"runs/trsp/{args.policy}"
    test_dir = make_test_dir(policy_dir)
    test_config = copy.deepcopy(config)
    test_config['test'] = {'mode': 'sequential_eval'}
    results_path = init_results(test_dir, test_config)

    print(f"\n[Policy: {args.policy}] → {test_dir}")
    t0 = time.time()
    run_evaluation(
        policy_agent, action_mode, selected_idxs, idx2node, graph_info, config,
        sumo_config_cmd, psl_model, traffic_model, results_path, desc=args.policy
    )
    print(f"Policy eval done in {timedelta(seconds=int(time.time() - t0))} → {results_path}")

    # --- Dijkstra baseline evaluation ---
    dij_agent = DijkstraAgent(0)
    dij_results_path = init_baseline_csv(test_dir, 'dijkstra')

    print(f"\n[Dijkstra] → {dij_results_path}")
    t0 = time.time()
    run_evaluation(
        dij_agent, 'neighbor_index', selected_idxs, idx2node, graph_info, config,
        sumo_config_cmd, psl_model, traffic_model, dij_results_path, desc='Dijkstra'
    )
    print(f"Dijkstra eval done in {timedelta(seconds=int(time.time() - t0))} → {dij_results_path}")

    # --- DSP baseline evaluation ---
    dsp_agent = DSPAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                         graph_info['idx2node'], ugv_speed)
    dsp_results_path = init_baseline_csv(test_dir, 'dsp')

    print(f"\n[DSP] → {dsp_results_path}")
    t0 = time.time()
    run_evaluation(
        dsp_agent, 'node_id', selected_idxs, idx2node, graph_info, config,
        sumo_config_cmd, psl_model, traffic_model, dsp_results_path, desc='DSP'
    )
    print(f"DSP eval done in {timedelta(seconds=int(time.time() - t0))} → {dsp_results_path}")


if __name__ == '__main__':
    main()
