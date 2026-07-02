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
from dronesim.trsp.agents.dsp_ped      import DSPPEDAgent
from dronesim.trsp.agents.dqn_psl      import DQNAgent
from dronesim.trsp.agents.dqn_svc      import DQNSVCAgent
from dronesim.trsp.agents.ppo_psl      import PPOAgent
from dronesim.trsp.agents.ppo_svc      import PPOSVCAgent
from dronesim.trsp.agents.sarsa_psl    import SARSAAgent
from dronesim.trsp.agents.sarsa_svc    import SARSASVCAgent
from dronesim.trsp.agents.sarsa_ped    import SARSAPEDAgent
from dronesim.trsp.agents.dqn_ped      import DQNPEDAgent
from dronesim.trsp.traffic import PSLTrafficModel, TDTrafficModel
from dronesim.trsp.ped_speed import weidmann_speed

INF = 100000

# Cache of sidewalk lane footprints (m²) for the Weidmann ped-speed override; lane
# length/width are static within a SUMO session.
_LANE_AREA_CACHE: dict[str, float] = {}


def _ped_sidewalk_speed(lane_id, ugv_speed, ped_params, exclude_self=False):
    """Robot Weidmann speed (m/s) on a sidewalk lane from live TraCI person counts.

    `ped_params` = (gamma, density_jam, v_min). Mirrors SumoTRSPEnv._weidmann_speed so
    eval simulates the same density-coupled dynamics the agents were trained under.
    """
    gamma, density_jam, v_min = ped_params
    n = len(traci.edge.getLastStepPersonIDs(traci.lane.getEdgeID(lane_id)))
    if exclude_self:
        n = max(n - 1, 0)
    area = _LANE_AREA_CACHE.get(lane_id)
    if area is None:
        area = traci.lane.getLength(lane_id) * traci.lane.getWidth(lane_id)
        _LANE_AREA_CACHE[lane_id] = area
    return weidmann_speed(n, area, ugv_speed, gamma, density_jam, v_min)


def get_traffic_model(config: dict) -> str:
    """Return 'td', 'svc', 'ped', 'psl', or 'none'."""
    traffic_cfg = config.get('traffic')
    if traffic_cfg is None:
        return 'none'
    if not traffic_cfg.get('enabled', True):
        return 'none'
    model = traffic_cfg.get('model', 'psl')
    if model == 'svc':
        return 'svc'
    if model == 'ped':
        return 'ped'
    if model == 'td':
        return 'td'
    return 'psl'


def _neighbor_speeds_from_traci(curr_node, nb_node_idxs, G_dir, idx2node,
                                 ugv_speed: float, num_actions: int,
                                 ped_mode: bool = False, ped_params=None) -> np.ndarray:
    """Query TraCI for each outgoing edge's speed; normalize by ugv_speed.

    For ped mode with the override active (`ped_params` given) this matches the training
    env: the neighbour sidewalk speed is the robot's Weidmann density-based speed, not the
    pedestrian mean speed — so the agent observes the speed it will actually get.
    """
    speeds = []
    for nb_idx in nb_node_idxs:
        nb_node = idx2node[nb_idx]
        edge_data = G_dir.edges[curr_node, nb_node]
        if ped_mode:
            lane_id = edge_data.get('sidewalk_lane')
            if ped_params is not None:
                raw = _ped_sidewalk_speed(lane_id, ugv_speed, ped_params) if lane_id else ugv_speed
            else:
                raw = traci.lane.getLastStepMeanSpeed(lane_id) if lane_id else ugv_speed
        else:
            raw = traci.edge.getLastStepMeanSpeed(edge_data['id'])
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


def build_graph(net_path, ped_mode=False):
    net_path = net_path.replace('.sumocfg', '.net.xml')
    net = sumolib.net.readNet(net_path, withPedestrianConnections=ped_mode)

    G = nx.DiGraph()
    junc2nodes_map = defaultdict(list)

    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(':'):
            continue
        if ped_mode and not any(lane.allows('pedestrian') for lane in edge.getLanes()):
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

        if ped_mode:
            sidewalk_lane = next(
                (lane.getID() for lane in edge.getLanes() if lane.allows('pedestrian')),
                None,
            )
            G.add_edge(from_node.getID(), to_node.getID(),
                       id=edge_id, length=edge.getLength(), type='road',
                       sidewalk_lane=sidewalk_lane)
        else:
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


def plan_route(agent, action_mode, start_node, target_node, time_slot,
               G, G_dir, idx2node, node2idx, junc2idx, sg_pairs,
               all_shortest_path_lengths_dir, max_steps, num_actions,
               traffic_model='psl', ugv_speed=None, ped_mode=False, ped_params=None):
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
            if traffic_model in ('svc', 'ped'):
                obs['neighbor_speeds'] = _neighbor_speeds_from_traci(
                    curr_node, nb_node_idxs, G_dir, idx2node, ugv_speed, num_actions,
                    ped_mode=ped_mode, ped_params=ped_params,
                )
            else:  # 'psl' or 'td' — both expose time_slot ∈ {0..K-1}
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
                   sumo_config_cmd, psl_model, traffic_model, results_path, desc='Eval',
                   fleet_type='ev', dr_cfg=None):
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
    psl_active = traffic_model in ('psl', 'td')  # both use psl_model.get_slot / get_speed
    ped_mode = fleet_type == 'delivery_robot'
    adr_vtype_id = 'adr_vtype'
    n_pairs = len(selected_idxs) - 1

    # Weidmann ped-speed override (matches the training env). When enabled, ped_params
    # feeds both the observation (neighbor speeds) and the robot's per-step speed.
    pso = config.get('traffic', {}).get('ped_speed_override', {})
    ped_override = ped_mode and bool(pso.get('enabled', False))
    ped_params = ((pso.get('gamma', 1.913), pso.get('density_jam', 5.4),
                   pso.get('v_min', 0.1)) if ped_override else None)

    def start_sumo():
        traci.start(sumo_config_cmd)
        if ped_mode and dr_cfg is not None:
            traci.vehicletype.copy('DEFAULT_PEDTYPE', adr_vtype_id)
            traci.vehicletype.setWidth(adr_vtype_id, dr_cfg.get('width', 1.0))
            traci.vehicletype.setLength(adr_vtype_id, dr_cfg.get('length', 0.8))
            traci.vehicletype.setSpeedFactor(adr_vtype_id, dr_cfg.get('max_speed_factor', 1.0))
            # Cap the ADR at the configured robot speed (DEFAULT_PEDTYPE's inherited
            # maxSpeed is a large SUMO default ~10 m/s) — same as SumoTRSPEnv.
            traci.vehicletype.setMaxSpeed(adr_vtype_id, ugv_speed)
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
            traffic_model=traffic_model, ugv_speed=ugv_speed, ped_mode=ped_mode,
            ped_params=ped_params,
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

        entity_id = f"ugv_{pair_idx}"
        failed_error = 'NA'

        try:
            if ped_mode:
                traci.person.add(entity_id, route_edges[0], pos=0.0,
                                 depart=-1, typeID=adr_vtype_id)
                traci.person.appendWalkingStage(entity_id, route_edges, arrivalPos=-1)
            else:
                route_id = f"route_{pair_idx}"
                depart_spd = traffic_speed_start if (psl_active and traffic_speed_start > 0) else 'max'
                traci.route.add(route_id, route_edges)
                traci.vehicle.add(entity_id, route_id, departSpeed=depart_spd)
                if psl_active and traffic_speed_start > 0:
                    traci.vehicle.setSpeed(entity_id, traffic_speed_start)
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
        except traci.TraCIException:
            append_result_row(results_path, start_idx, target_idx,
                              warmup_time, -1, -1, -1, -1, -1, -1, -1, -1, "Route can't be added")
            continue

        if ped_mode:
            while entity_id in traci.person.getIDList() and sim_time < max_time:
                if ped_params is not None:
                    lane_id = traci.person.getLaneID(entity_id)
                    if lane_id and not lane_id.startswith(':'):  # skip internal/junction lanes
                        traci.person.setSpeed(entity_id, _ped_sidewalk_speed(
                            lane_id, ugv_speed, ped_params, exclude_self=True))
                traci.simulationStep()
                sim_time = traci.simulation.getTime()
            if entity_id in traci.person.getIDList():
                failed_error = 'Timeout'
        else:
            while entity_id in traci.vehicle.getIDList() and sim_time < max_time:
                if psl_active:
                    slot = psl_model.get_slot(sim_time)
                    road_id = traci.vehicle.getRoadID(entity_id)
                    if not road_id.startswith(':') and road_id in psl_model.profiles:
                        traci.vehicle.setSpeed(entity_id, psl_model.get_speed(road_id, slot))
                traci.simulationStep()
                sim_time = traci.simulation.getTime()
            if entity_id in traci.vehicle.getIDList():
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

    fleet_type = config['fleet'].get('type', 'ev')
    ped_mode = fleet_type == 'delivery_robot'
    dr_cfg = config['fleet'].get('delivery_robot', {}) if ped_mode else None

    net_path = config['net']['net_path']
    print(f"Building graph from {net_path} ...")
    graph_info = build_graph(net_path, ped_mode=ped_mode)
    idx2node = graph_info['idx2node']
    num_nodes = len(idx2node)

    # 101 nodes → 100 OG pairs, reproducible and shared across policy + dijkstra runs
    selected_idxs = select_nodes(num_nodes, seed=42, n_pairs=100)
    print(f"Selected 101 nodes ({num_nodes} total) → 100 evaluation pairs")

    traffic_model = get_traffic_model(config)
    if ped_mode:
        traffic_model = 'ped'
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
    elif traffic_model == 'td':
        psl_model = TDTrafficModel(
            config['traffic'], edge_ids, seed=config['traffic'].get('seed')
        )
        tcfg = config['traffic']
        print(
            f"Traffic model: TD  K=3  "
            f"v_c={tcfg['v_congestion']:.0f}km/h  v_f={tcfg['v_freeflow']:.0f}km/h  "
            f"T1={tcfg['T1']}s  T2={tcfg['T2']}s"
        )
    elif traffic_model == 'svc':
        route_file = config.get('traffic', {}).get('route_file', 'from sumocfg')
        print(f"Traffic model: SVC  route_file={route_file}")
    elif traffic_model == 'ped':
        print("Traffic model: PED  (live sidewalk pedestrian speeds)")
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
    # if traffic_model == 'svc':
    #     sumo_config_cmd += ['--route-files', config['traffic']['route_file']]

    ugv_speed = config['fleet']['max_speed'] * 1000.0 / 3600.0  # km/h → m/s

    # --- Policy evaluation ---
    algo = config['agent']['algo']
    if algo == 'dqn':
        policy_agent = DQNAgent(config)
    elif algo == 'dqn_svc':
        policy_agent = DQNSVCAgent(config)
    elif algo == 'dqn_ped':
        policy_agent = DQNPEDAgent(config)
    elif algo == 'sarsa':
        policy_agent = SARSAAgent(config)
    elif algo == 'sarsa_svc':
        policy_agent = SARSASVCAgent(config)
    elif algo == 'sarsa_ped':
        policy_agent = SARSAPEDAgent(config)
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
    elif algo == 'dsp_ped':
        pso = config.get('traffic', {}).get('ped_speed_override', {})
        policy_agent = DSPPEDAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                                   graph_info['idx2node'], ugv_speed,
                                   gamma=pso.get('gamma', 1.913),
                                   density_jam=pso.get('density_jam', 5.4),
                                   v_min=pso.get('v_min', 0.1))
    else:
        policy_agent = get_agent(config['agent'], 0)
    action_mode = config['agent'].get('action_mode', 'neighbor_index')
    if algo in ('dsp', 'dsp_ped'):
        action_mode = 'node_id'

    policy_dir = f"runs/trsp/{args.policy}"
    test_dir = make_test_dir(policy_dir)
    test_config = copy.deepcopy(config)
    test_config['test'] = {'mode': 'sequential_eval'}
    results_path = init_results(test_dir, test_config)

    run_kwargs = dict(
        selected_idxs=selected_idxs, idx2node=idx2node, graph_info=graph_info,
        config=config, sumo_config_cmd=sumo_config_cmd, psl_model=psl_model,
        traffic_model=traffic_model, fleet_type=fleet_type, dr_cfg=dr_cfg,
    )

    print(f"\n[Policy: {args.policy}] → {test_dir}")
    t0 = time.time()
    run_evaluation(policy_agent, action_mode, results_path=results_path,
                   desc=args.policy, **run_kwargs)
    print(f"Policy eval done in {timedelta(seconds=int(time.time() - t0))} → {results_path}")

    # --- Dijkstra baseline evaluation ---
    dij_agent = DijkstraAgent(0)
    dij_results_path = init_baseline_csv(test_dir, 'dijkstra')

    print(f"\n[Dijkstra] → {dij_results_path}")
    t0 = time.time()
    run_evaluation(dij_agent, 'neighbor_index', results_path=dij_results_path,
                   desc='Dijkstra', **run_kwargs)
    print(f"Dijkstra eval done in {timedelta(seconds=int(time.time() - t0))} → {dij_results_path}")

    # --- DSP baseline evaluation ---
    # EV: vehicular DSP; ped: DSP-PED (edge weights use the robot's Weidmann sidewalk speed).
    if not ped_mode:
        dsp_agent = DSPAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                             graph_info['idx2node'], ugv_speed)
    else:
        pso = config.get('traffic', {}).get('ped_speed_override', {})
        dsp_agent = DSPPEDAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                                graph_info['idx2node'], ugv_speed,
                                gamma=pso.get('gamma', 1.913),
                                density_jam=pso.get('density_jam', 5.4),
                                v_min=pso.get('v_min', 0.1))
    dsp_results_path = init_baseline_csv(test_dir, 'dsp')

    print(f"\n[DSP] → {dsp_results_path}")
    t0 = time.time()
    run_evaluation(dsp_agent, 'node_id', results_path=dsp_results_path,
                   desc='DSP', **run_kwargs)
    print(f"DSP eval done in {timedelta(seconds=int(time.time() - t0))} → {dsp_results_path}")


if __name__ == '__main__':
    main()
