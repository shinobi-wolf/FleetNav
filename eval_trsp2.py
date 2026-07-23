"""Sequential closed-loop evaluation for PSL (piecewise speed-limit) TRSP policies.

Runs 100 fixed OD pairs one after another inside a single continuous SUMO session so
the traffic slot evolves naturally across pairs (essential for K>1). For every pair the
agent is driven exactly as in training: at each junction it observes the LIVE slot,
picks one hop, that single edge is driven in SUMO, then it re-decides. This is the key
difference from a plan-whole-route-then-drive evaluator, which froze the slot at
departure and never let a K>1 policy adapt mid-route.

Scope: PSL and TD traffic only (both are slot-based and share the same code path). SVC
and pedestrian modes are intentionally not handled here.
"""
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
from dronesim.trsp.agents.dijkstra_psl import DijkstraAgent
from dronesim.trsp.agents.dsp          import DSPAgent
from dronesim.trsp.agents.dqn_psl      import DQNAgent
from dronesim.trsp.agents.ppo_psl      import PPOAgent
from dronesim.trsp.agents.a2c_psl      import A2CAgent
from dronesim.trsp.agents.sarsa_psl    import SARSAAgent
from dronesim.trsp.traffic import PSLTrafficModel, TDTrafficModel

INF = 100000


def get_traffic_model(config: dict) -> str:
    """Return 'td', 'psl', or 'none' (svc/ped are out of scope for this evaluator)."""
    traffic_cfg = config.get('traffic')
    if traffic_cfg is None or not traffic_cfg.get('enabled', True):
        return 'none'
    return 'td' if traffic_cfg.get('model') == 'td' else 'psl'


def get_centroid(shape):
    if not shape:
        return 0, 0
    x_coords = [p[0] for p in shape]
    y_coords = [p[1] for p in shape]
    return np.mean(x_coords), np.mean(y_coords)


def build_graph(net_path):
    net_path = net_path.replace('.sumocfg', '.net.xml')
    net = sumolib.net.readNet(net_path)

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

        G.add_edge(from_node.getID(), to_node.getID(),
                   id=edge_id, length=edge.getLength(), type='road')

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


def agent_decide(agent, action_mode, curr_node, target_node, live_slot,
                 G, G_dir, idx2node, node2idx, junc2idx, sg_pairs,
                 all_shortest_path_lengths_dir, num_actions):
    """One closed-loop decision at `curr_node`. Builds the same observation the agent saw
    during training, with the LIVE slot, and returns (next_node, edge_id) — or
    (None, None) if the node is a dead end. Invalid / out-of-range actions fall back to
    the greedy shortest-path neighbour (`best_dir`), matching the training env's masking."""
    curr_pos = node2idx[curr_node]
    nb_nodes = list(G_dir.neighbors(curr_node))
    nb_node_idxs = sorted([node2idx[nb] for nb in nb_nodes])[:num_actions]
    if not nb_node_idxs:
        return None, None

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
        'time_slot': live_slot,
        'best_dir': best_dir,
        'steps2dest': np.array(steps2dest, dtype=np.float32),
        'sg_pair_idx': sg_idx,
    }

    action = int(agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx))

    if action_mode == 'node_id':
        if action not in nb_node_idxs:
            action = nb_node_idxs[best_dir]
        next_pos_idx = action
    else:
        if action >= len(nb_node_idxs):
            action = best_dir
        next_pos_idx = nb_node_idxs[action]

    next_node = idx2node[next_pos_idx]
    return next_node, G_dir.edges[curr_node, next_node]['id']


def drive_single_edge(edge_id, sim_time, max_time, entity_id, psl_model, psl_active, ugv_speed):
    """Insert one vehicle on a single edge and step SUMO until it crosses, exactly as the
    training env does per hop (dronesim/trsp/envs/sumo_trsp_env.py). Speed is set to the
    edge's slot speed and re-updated on every slot change mid-edge. Returns
    (new_sim_time, arrived_bool); arrived_bool is False only on wall-clock timeout."""
    slot = psl_model.get_slot(sim_time) if psl_active else 0
    depart_speed = (psl_model.get_speed(edge_id, slot)
                    if (psl_active and edge_id in psl_model.profiles) else ugv_speed)

    route_id = f"{entity_id}_route"
    traci.route.add(route_id, [edge_id])
    traci.vehicle.add(entity_id, route_id, departSpeed=depart_speed)
    traci.vehicle.setSpeed(entity_id, depart_speed)

    traci.simulationStep()
    sim_time = traci.simulation.getTime()

    inserted = False
    while sim_time < max_time:
        active_ids = set(traci.vehicle.getIDList())
        arrived_ids = set(traci.simulation.getArrivedIDList())
        if entity_id in active_ids:
            inserted = True
            if psl_active:
                new_slot = psl_model.get_slot(sim_time)
                if new_slot != slot:
                    slot = new_slot
                    road_id = traci.vehicle.getRoadID(entity_id)
                    if not road_id.startswith(':') and road_id in psl_model.profiles:
                        traci.vehicle.setSpeed(entity_id, psl_model.get_speed(road_id, new_slot))
        elif entity_id in arrived_ids or inserted:
            return sim_time, True

        traci.simulationStep()
        sim_time = traci.simulation.getTime()

    return sim_time, False   # wall-clock budget exhausted mid-edge


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
    """Run all 100 OD pairs sequentially within a single continuous SUMO session, driving
    each pair closed-loop (decide one hop under the live slot, drive that edge, re-decide).
    SUMO is only restarted when the sim clock reaches max_time before all pairs finish, so
    traffic slots evolve naturally across pairs (critical for K>1)."""
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
    psl_active = traffic_model in ('psl', 'td')  # both use psl_model.get_slot / get_speed
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

        if hasattr(agent, 'start_leg'):
            agent.start_leg(start_node, target_node)

        tour_start_time = sim_time
        traffic_segment_start = psl_model.get_slot(sim_time) if psl_active else -1
        traffic_speed_start = -1.0

        # ── Closed loop: at every junction, observe the LIVE slot, let the agent choose
        # the next hop, drive that single edge, then re-decide — exactly the per-hop MDP
        # the policy was trained on. This lets the slot evolve *between* decisions, which
        # the old plan-whole-route-then-drive path could not do (it froze the slot at
        # departure). K=1 is unchanged (constant slot); K>1 now truly adapts.
        curr_node = start_node
        route_edges = []
        failed_error = 'NA'
        try:
            while curr_node != target_node and len(route_edges) < max_steps:
                live_slot = psl_model.get_slot(sim_time) if psl_active else 0
                next_node, edge_id = agent_decide(
                    agent, action_mode, curr_node, target_node, live_slot,
                    G, G_dir, idx2node, node2idx, junc2idx, sg_pairs,
                    all_shortest_path_lengths_dir, num_actions,
                )
                if next_node is None:
                    failed_error = 'No Path'
                    break
                if not route_edges and psl_active:
                    traffic_speed_start = (
                        psl_model.get_speed(edge_id, traffic_segment_start)
                        if edge_id in psl_model.profiles else -1.0
                    )
                entity_id = f"ugv_{pair_idx}_{len(route_edges)}"
                sim_time, arrived = drive_single_edge(
                    edge_id, sim_time, max_time, entity_id, psl_model, psl_active, ugv_speed,
                )
                route_edges.append(edge_id)
                curr_node = next_node
                if not arrived:                 # wall-clock budget hit mid-edge
                    failed_error = 'Timeout'
                    break
        except traci.TraCIException:
            failed_error = "Route can't be added"
        except KeyError:
            failed_error = 'KeyError'

        # Reached max_steps without arriving → treat as no valid route (matches the old
        # evaluator, which discarded an over-length walk).
        if failed_error == 'NA' and curr_node != target_node:
            failed_error = 'No Path'

        if failed_error != 'NA':
            append_result_row(results_path, start_idx, target_idx,
                              warmup_time, -1, -1, -1, -1, -1, -1, -1, -1, failed_error)
            if sim_time >= max_time and pair_idx < n_pairs - 1:
                traci.close()
                sim_time = start_sumo()
            continue

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
    parser = argparse.ArgumentParser(description="TRSP Sequential Closed-Loop Evaluation (PSL/TD)")
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

    # 101 nodes → 100 OD pairs, reproducible and shared across policy + baselines
    selected_idxs = select_nodes(num_nodes, seed=42, n_pairs=100)
    print(f"Selected 101 nodes ({num_nodes} total) → 100 evaluation pairs")

    traffic_model = get_traffic_model(config)
    edge_ids = graph_info['edge_ids']
    if traffic_model == 'td':
        psl_model = TDTrafficModel(config['traffic'], edge_ids, seed=config['traffic'].get('seed'))
        tcfg = config['traffic']
        print(f"Traffic model: TD  K=3  v_c={tcfg['v_congestion']:.0f}km/h  "
              f"v_f={tcfg['v_freeflow']:.0f}km/h  T1={tcfg['T1']}s  T2={tcfg['T2']}s")
    else:
        psl_model = PSLTrafficModel(config.get('traffic', {'enabled': False}), edge_ids,
                                    seed=config.get('traffic', {}).get('seed'))
        if traffic_model == 'psl':
            print(f"Traffic model: PSL  K={psl_model.K}  "
                  f"v=[{psl_model.v_min*3.6:.0f},{psl_model.v_max*3.6:.0f}]km/h  "
                  f"delta_t={psl_model.delta_t:.1f}s")
        else:
            print("Traffic model: DISABLED")

    step_length = config['sumo']['step_length']
    traffic_scale = config['sumo'].get('traffic_scale', 0.0)
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

    ugv_speed = config['fleet']['max_speed'] * 1000.0 / 3600.0  # km/h → m/s

    # --- Policy evaluation ---
    algo = config['agent']['algo']
    if algo == 'dqn':
        policy_agent = DQNAgent(config)
    elif algo == 'sarsa':
        policy_agent = SARSAAgent(config)
    elif algo == 'ppo':
        policy_agent = PPOAgent(config)
    elif algo == 'a2c':
        policy_agent = A2CAgent(config)
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

    run_kwargs = dict(
        selected_idxs=selected_idxs, idx2node=idx2node, graph_info=graph_info,
        config=config, sumo_config_cmd=sumo_config_cmd, psl_model=psl_model,
        traffic_model=traffic_model,
    )

    print(f"\n[Policy: {args.policy}] → {test_dir}")
    t0 = time.time()
    run_evaluation(policy_agent, action_mode, results_path=results_path,
                   desc=args.policy, **run_kwargs)
    print(f"Policy eval done in {timedelta(seconds=int(time.time() - t0))} → {results_path}")

    # --- Dijkstra baseline (static shortest path) ---
    dij_agent = DijkstraAgent(0)
    dij_results_path = init_baseline_csv(test_dir, 'dijkstra')
    print(f"\n[Dijkstra] → {dij_results_path}")
    t0 = time.time()
    run_evaluation(dij_agent, 'neighbor_index', results_path=dij_results_path,
                   desc='Dijkstra', **run_kwargs)
    print(f"Dijkstra eval done in {timedelta(seconds=int(time.time() - t0))} → {dij_results_path}")

    # --- DSP baseline (live-speed Dijkstra) ---
    dsp_agent = DSPAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                         graph_info['idx2node'], ugv_speed)
    dsp_results_path = init_baseline_csv(test_dir, 'dsp')
    print(f"\n[DSP] → {dsp_results_path}")
    t0 = time.time()
    run_evaluation(dsp_agent, 'node_id', results_path=dsp_results_path,
                   desc='DSP', **run_kwargs)
    print(f"DSP eval done in {timedelta(seconds=int(time.time() - t0))} → {dsp_results_path}")


if __name__ == '__main__':
    main()
