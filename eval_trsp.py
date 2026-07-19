"""
eval_trsp.py — TRSP Sequential Evaluation (K>1 compatible).

Replaces the old eval_trsp.py / eval_trsp2.py with three fixes:

1. OD-pair sampling: 100 DISTINCT (start, target) pairs are sampled independently
   up front (select_od_pairs), instead of deriving 100 overlapping pairs from a
   101-node consecutive walk. The old scheme let the same (o,d) recur (pair i's
   target was always pair i+1's start), which silently collapsed independent
   trials under any (start_idx,target_idx)-keyed dedup downstream. With distinct
   pairs guaranteed at sampling time, no such collapse can happen.

2. Live-traffic route planning advances real SUMO time between hops. For agents
   trained on live per-edge speeds (SVC/PED: sarsa_svc, dqn_svc, a2c_svc,
   ppo_svc, sarsa_ped, dqn_ped — see REACTIVE_ALGOS), drive_reactive_route()
   inserts a single-edge vehicle/person per hop and steps SUMO until it clears
   before computing the next hop's observation — mirroring
   SumoTRSPEnv.ugv_handler's training-time dynamics exactly. Previously the
   whole multi-hop route was planned against one frozen simulated instant.
   Dijkstra and DSP/DSP-PED are NOT reactive: DSPAgent.start_leg() already
   solves a live-speed-weighted Dijkstra ONCE at leg start and then follows
   that fixed path deterministically (same one-shot-then-walk shape as static
   Dijkstra, just with live-derived weights) — that design is correct as-is
   and must not be changed to per-hop replanning.

3. Optional offline oscillation-fix preprocessing (--fix-oscillations) for the
   policy agent only. Generalizes SARSAAgent._fix_oscillations()
   (dronesim/trsp/agents/sarsa_psl.py) — which only handles strict 2-node
   A<->B cycles in a tabular Q-table — to any predict()-based agent (tabular
   or network) and any cycle length: OscillationFixWrapper probes the agent's
   greedy rollout across every (node, target-junction) combination once at
   run start, and wherever the walk would revisit an already-visited node it
   overrides that single decision with the best not-yet-visited neighbour
   (ranked by static shortest-path distance to target), continuing the probe
   so downstream loops toward the same target are caught too.

Outputs three CSVs per test_N/ dir with the same schema as before:
  simulation_results.csv, simulation_results_dijkstra.csv, simulation_results_dsp.csv
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
N_PAIRS = 100
SEED = 42

# Algos whose predict() consumes live per-edge neighbor_speeds and were trained under
# a genuinely evolving-traffic step-and-observe loop (SumoTRSPEnv.ugv_handler). These
# get drive_reactive_route(); everything else (dijkstra, dsp, dsp_ped, and the PSL/TD
# variants sarsa/dqn/ppo/a2c) keeps the one-shot plan-then-drive path.
REACTIVE_ALGOS = {'sarsa_svc', 'dqn_svc', 'a2c_svc', 'ppo_svc', 'sarsa_ped', 'dqn_ped'}

# Cache of sidewalk lane footprints (m²) for the Weidmann ped-speed override; lane
# length/width are static within a SUMO session.
_LANE_AREA_CACHE: dict[str, float] = {}


def _ped_sidewalk_speed(lane_id, ugv_speed, ped_params, exclude_self=False):
    """Robot Weidmann speed (m/s) on a sidewalk lane from live TraCI person counts."""
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
    """Query TraCI for each outgoing edge's speed *right now*; normalize by ugv_speed."""
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
    """Parse the SUMO net and precompute everything plan/drive routines need."""
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

    sg_idx_map = {}
    counter = 0
    for node_idx in range(len(idx2node)):
        for junc_id in junction_list:
            node = idx2node[node_idx]
            if G_und.nodes[node]['junc_id'] != junc_id:
                sg_idx_map[(node_idx, junc2idx[junc_id])] = counter
                counter += 1

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
        'sg_idx_map': sg_idx_map,
        'edge_ids': edge_ids,
    }


# ── OD-pair sampling (flaw #1) ──────────────────────────────────────────────

def select_od_pairs(num_nodes, seed=SEED, n_pairs=N_PAIRS):
    """Sample n_pairs DISTINCT (start_idx, target_idx) node-index pairs, independently —
    no consecutive-walk chaining, no possibility of the same pair recurring."""
    rng = random.Random(seed)
    all_idxs = list(range(num_nodes))
    if num_nodes * (num_nodes - 1) < n_pairs:
        raise ValueError(f"Network has only {num_nodes} nodes — cannot draw {n_pairs} "
                          f"distinct (o,d) pairs (max {num_nodes * (num_nodes - 1)}).")
    seen = set()
    pairs = []
    while len(pairs) < n_pairs:
        o = rng.choice(all_idxs)
        d = rng.choice(all_idxs)
        if o == d or (o, d) in seen:
            continue
        seen.add((o, d))
        pairs.append((o, d))
    return pairs


# ── One-shot planning (dijkstra, dsp, dsp_ped, psl/td RL agents) ───────────

def plan_route_oneshot(agent, action_mode, start_node, target_node, time_slot,
                       G, G_dir, idx2node, node2idx, junc2idx, sg_idx_map,
                       all_shortest_path_lengths_dir, max_steps, num_actions,
                       traffic_model='psl', ugv_speed=None, ped_mode=False, ped_params=None):
    """Plan a graph-level route in one shot (no SUMO time advance). Correct for agents
    that decide the whole path up front — DSP/DSP-PED via start_leg(), Dijkstra via a
    static distance gradient, and PSL/TD agents whose observation is a coarse time_slot
    fixed for the whole leg. Returns (route_edges, error_str_or_None)."""
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
            sg_idx = sg_idx_map.get((curr_pos, junction_idx), 0)
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


def drive_fixed_route(entity_id, route_edges, sim_time, max_time, psl_active, psl_model,
                       ped_mode, ped_params, ugv_speed, traffic_speed_start):
    """Insert the already-decided route_edges as one fixed multi-edge SUMO route/walk and
    step until arrival. Used after plan_route_oneshot; never re-plans mid-route."""
    try:
        if ped_mode:
            traci.person.add(entity_id, route_edges[0], pos=0.0, depart=-1, typeID='adr_vtype')
            traci.person.appendWalkingStage(entity_id, route_edges, arrivalPos=-1)
        else:
            route_id = f"{entity_id}_route"
            depart_spd = traffic_speed_start if (psl_active and traffic_speed_start > 0) else 'max'
            traci.route.add(route_id, route_edges)
            traci.vehicle.add(entity_id, route_id, departSpeed=depart_spd)
            if psl_active and traffic_speed_start > 0:
                traci.vehicle.setSpeed(entity_id, traffic_speed_start)
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
    except traci.TraCIException:
        return sim_time, "Route can't be added"

    if ped_mode:
        while entity_id in traci.person.getIDList() and sim_time < max_time:
            if ped_params is not None:
                lane_id = traci.person.getLaneID(entity_id)
                if lane_id and not lane_id.startswith(':'):
                    traci.person.setSpeed(entity_id, _ped_sidewalk_speed(
                        lane_id, ugv_speed, ped_params, exclude_self=True))
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
        if entity_id in traci.person.getIDList():
            return sim_time, 'Timeout'
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
            return sim_time, 'Timeout'

    return sim_time, None


# ── Reactive step-and-observe planning (flaw #2, SVC/PED agents only) ──────

def _drive_single_edge(entity_id, edge_id, sim_time, max_time, ped_mode, ped_params, ugv_speed):
    """Insert a single-edge vehicle/person, step until it clears — mirrors
    SumoTRSPEnv.ugv_handler's per-hop insert/step/arrive pattern exactly, so a fresh
    hop's traffic snapshot has genuinely advanced by the previous hop's travel time."""
    try:
        if ped_mode:
            traci.person.add(entity_id, edge_id, pos=0.0, depart=-1, typeID='adr_vtype')
            traci.person.appendWalkingStage(entity_id, [edge_id], arrivalPos=-1)
        else:
            route_id = f"{entity_id}_route"
            traci.route.add(route_id, [edge_id])
            traci.vehicle.add(entity_id, route_id, departSpeed='max')
    except traci.TraCIException:
        return sim_time, "Route can't be added"

    traci.simulationStep()
    sim_time = traci.simulation.getTime()

    inserted = False
    while sim_time < max_time:
        if ped_mode:
            active_ids = set(traci.person.getIDList())
            if entity_id in active_ids:
                inserted = True
                if ped_params is not None:
                    lane_id = traci.person.getLaneID(entity_id)
                    if lane_id and not lane_id.startswith(':'):
                        traci.person.setSpeed(entity_id, _ped_sidewalk_speed(
                            lane_id, ugv_speed, ped_params, exclude_self=True))
            elif inserted:
                break
        else:
            active_ids = set(traci.vehicle.getIDList())
            arrived_ids = set(traci.simulation.getArrivedIDList())
            if entity_id in active_ids:
                inserted = True
            elif entity_id in arrived_ids or inserted:
                break
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

    if sim_time >= max_time:
        return sim_time, 'Timeout'
    return sim_time, None


def drive_reactive_route(agent, action_mode, start_node, target_node, G, G_dir,
                         idx2node, node2idx, junc2idx, sg_idx_map,
                         all_shortest_path_lengths_dir, max_steps, num_actions,
                         ugv_speed, sim_time, max_time, entity_prefix,
                         ped_mode=False, ped_params=None):
    """Interleaved plan+drive for live-speed (svc/ped) RL agents: each hop's
    neighbor_speeds observation is queried live, the chosen edge is driven with a
    single-edge SUMO entity, and only once it clears does the NEXT hop's observation
    get built — so traffic has genuinely evolved between hops, matching training.
    Returns (route_edges, tour_start_time, sim_time, error_or_None)."""
    curr_node = start_node
    curr_pos = node2idx[start_node]
    route_edges = []
    tour_start_time = sim_time
    curr_step = 0

    if hasattr(agent, 'start_leg'):
        agent.start_leg(start_node, target_node)

    while curr_step < max_steps and curr_node != target_node and sim_time < max_time:
        nb_nodes = list(G_dir.neighbors(curr_node))
        nb_node_idxs = sorted([node2idx[nb] for nb in nb_nodes])[:num_actions]
        if not nb_node_idxs:
            return route_edges, tour_start_time, sim_time, 'No Path'

        steps2dest = []
        for nb_idx in nb_node_idxs:
            nb_node = idx2node[nb_idx]
            steps2dest.append(all_shortest_path_lengths_dir.get(nb_node, {}).get(target_node, INF))
        steps2dest += [INF] * (num_actions - len(steps2dest))
        junction_idx = junc2idx[G.nodes[target_node]['junc_id']]
        sg_idx = sg_idx_map.get((curr_pos, junction_idx), 0)
        best_dir = int(np.argmin(steps2dest))

        obs = {
            'graph_node': curr_pos,
            'target_node': junction_idx,
            'best_dir': best_dir,
            'steps2dest': np.array(steps2dest, dtype=np.float32),
            'sg_pair_idx': sg_idx,
            'neighbor_speeds': _neighbor_speeds_from_traci(
                curr_node, nb_node_idxs, G_dir, idx2node, ugv_speed, num_actions,
                ped_mode=ped_mode, ped_params=ped_params),
        }
        try:
            action = int(agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx))
        except KeyError:
            return route_edges, tour_start_time, sim_time, 'KeyError'

        if action_mode == 'node_id':
            if action not in nb_node_idxs:
                action = nb_node_idxs[best_dir]
            next_pos_idx = action
        else:
            if action >= len(nb_node_idxs):
                action = best_dir
            next_pos_idx = nb_node_idxs[action]
        next_pos = idx2node[next_pos_idx]
        edge_id = G_dir.edges[curr_node, next_pos]['id']

        entity_id = f"{entity_prefix}_{curr_step}"
        sim_time, err = _drive_single_edge(entity_id, edge_id, sim_time, max_time,
                                           ped_mode, ped_params, ugv_speed)
        if err:
            return route_edges, tour_start_time, sim_time, err

        route_edges.append(edge_id)
        curr_step += 1
        curr_pos = next_pos_idx
        curr_node = next_pos

    if curr_node == target_node and route_edges:
        return route_edges, tour_start_time, sim_time, None
    if sim_time >= max_time:
        return route_edges, tour_start_time, sim_time, 'Timeout'
    return route_edges, tour_start_time, sim_time, 'No Path'


# ── Offline oscillation-fix preprocessing (flaw #3 / option) ───────────────

class OscillationFixWrapper:
    """Offline preprocessing pass over a loaded agent's greedy policy: probes every
    (node, target-junction) combination once at run start and records an override
    wherever the induced greedy rollout would revisit an already-visited node —
    generalizing SARSAAgent._fix_oscillations (dronesim/trsp/agents/sarsa_psl.py,
    strict tabular 2-cycles only) to any predict()-based agent (tabular or network)
    and any cycle length. At the offending step, the override picks the
    static-shortest-path-best neighbour that is NOT already visited on that walk; the
    probe continues from there so downstream loops toward the same target are also
    caught. Drop-in replacement for the wrapped agent (same predict()/start_leg()
    interface) — apply only to the policy agent, never to Dijkstra/DSP baselines
    (both are already structurally acyclic: DSPAgent.start_leg() runs nx.dijkstra_path,
    which never revisits a node by construction).
    """

    def __init__(self, agent, action_mode, G, G_dir, idx2node, node2idx, junc2idx,
                sg_idx_map, all_shortest_path_lengths_dir, max_steps, num_actions,
                traffic_model, ugv_speed, ped_mode=False, ped_params=None, time_slot=0):
        self.agent = agent
        self.action_mode = action_mode
        self.override: dict[tuple, int] = {}
        self.n_fixed = 0
        # Live neighbor-speed reads are cached per curr_node for the whole probe: the
        # probe deliberately uses ONE representative snapshot (consistent with "offline
        # preprocessing"), and this cuts what would be tens of thousands of TraCI
        # round-trips down to at most one per graph node.
        self._speed_cache: dict[str, np.ndarray] = {}
        self._build_overrides(G, G_dir, idx2node, node2idx, junc2idx, sg_idx_map,
                              all_shortest_path_lengths_dir, max_steps, num_actions,
                              traffic_model, ugv_speed, ped_mode, ped_params, time_slot)

    def start_leg(self, src, dst):
        if hasattr(self.agent, 'start_leg'):
            self.agent.start_leg(src, dst)

    def predict(self, obs, G=None, idx2node=None, node2idx=None):
        key = (int(obs['graph_node']), int(obs['target_node']))
        if key in self.override:
            return self.override[key]
        return int(self.agent.predict(obs, G=G, idx2node=idx2node, node2idx=node2idx))

    def _get_neighbor_speeds(self, curr_node, nb_node_idxs, G_dir, idx2node, ugv_speed,
                             num_actions, ped_mode, ped_params):
        cached = self._speed_cache.get(curr_node)
        if cached is None:
            cached = _neighbor_speeds_from_traci(
                curr_node, nb_node_idxs, G_dir, idx2node, ugv_speed, num_actions,
                ped_mode=ped_mode, ped_params=ped_params)
            self._speed_cache[curr_node] = cached
        return cached

    def _build_overrides(self, G, G_dir, idx2node, node2idx, junc2idx, sg_idx_map,
                         all_shortest_path_lengths_dir, max_steps, num_actions,
                         traffic_model, ugv_speed, ped_mode, ped_params, time_slot):
        for target_node in G.nodes:
            target_junc_id = G.nodes[target_node]['junc_id']
            junction_idx = junc2idx[target_junc_id]

            for start_node in G.nodes:
                if G.nodes[start_node]['junc_id'] == target_junc_id:
                    continue
                curr_node = start_node
                visited = {node2idx[start_node]}

                for _ in range(max_steps):
                    if G.nodes[curr_node]['junc_id'] == target_junc_id:
                        break
                    nb_nodes = list(G_dir.neighbors(curr_node))
                    nb_node_idxs = sorted([node2idx[nb] for nb in nb_nodes])[:num_actions]
                    if not nb_node_idxs:
                        break
                    curr_pos = node2idx[curr_node]
                    steps2dest = [
                        all_shortest_path_lengths_dir.get(idx2node[ni], {}).get(target_node, INF)
                        for ni in nb_node_idxs
                    ]
                    steps2dest += [INF] * (num_actions - len(steps2dest))
                    best_dir = int(np.argmin(steps2dest))
                    key = (curr_pos, junction_idx)

                    if key in self.override:
                        raw_action = self.override[key]
                    else:
                        obs = {
                            'graph_node': curr_pos, 'target_node': junction_idx,
                            'best_dir': best_dir,
                            'steps2dest': np.array(steps2dest, dtype=np.float32),
                            'sg_pair_idx': sg_idx_map.get(key, 0),
                        }
                        if traffic_model in ('svc', 'ped'):
                            obs['neighbor_speeds'] = self._get_neighbor_speeds(
                                curr_node, nb_node_idxs, G_dir, idx2node, ugv_speed,
                                num_actions, ped_mode, ped_params)
                        else:
                            obs['time_slot'] = time_slot
                        try:
                            raw_action = int(self.agent.predict(
                                obs, G=G, idx2node=idx2node, node2idx=node2idx))
                        except KeyError:
                            break

                    if self.action_mode == 'node_id':
                        action_idx = raw_action if raw_action in nb_node_idxs else nb_node_idxs[best_dir]
                    else:
                        action_idx = (nb_node_idxs[raw_action] if raw_action < len(nb_node_idxs)
                                      else nb_node_idxs[best_dir])

                    if action_idx in visited:
                        candidates = sorted(
                            (steps2dest[i], nb_node_idxs[i]) for i in range(len(nb_node_idxs))
                            if nb_node_idxs[i] not in visited
                        )
                        if candidates:
                            alt_node_idx = candidates[0][1]
                            self.override[key] = (
                                alt_node_idx if self.action_mode == 'node_id'
                                else nb_node_idxs.index(alt_node_idx)
                            )
                            action_idx = alt_node_idx
                            self.n_fixed += 1
                        # else: every neighbour already visited on this walk (boxed in)
                        # — nothing better available, leave the agent's own choice.

                    visited.add(action_idx)
                    curr_node = idx2node[action_idx]


# ── CSV helpers ──────────────────────────────────────────────────────────────

def append_result_row(results_path, start_idx, target_idx, warm_up_time, tour_start_time,
                      total_time_taken, path_length, path_edges, traffic_segment_start,
                      traffic_segment_end, traffic_speed_start_ms, traffic_speed_end_ms, failed_error):
    with open(results_path, 'a') as f:
        f.write(
            f"{start_idx},{target_idx},{warm_up_time},{tour_start_time},{total_time_taken},"
            f"{path_length},{path_edges},{traffic_segment_start},{traffic_segment_end},"
            f"{traffic_speed_start_ms},{traffic_speed_end_ms},{failed_error}\n"
        )


def make_test_dir(base_dir):
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
    with open(f"{test_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    results_path = f"{test_dir}/simulation_results.csv"
    with open(results_path, 'w') as f:
        f.write(_CSV_HEADER)
    return results_path


def init_baseline_csv(test_dir, suffix):
    results_path = f"{test_dir}/simulation_results_{suffix}.csv"
    with open(results_path, 'w') as f:
        f.write(_CSV_HEADER)
    return results_path


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(agent, algo, action_mode, od_pairs, idx2node, graph_info, config,
                   sumo_config_cmd, psl_model, traffic_model, results_path, desc='Eval',
                   fleet_type='ev', dr_cfg=None, fix_oscillations=False):
    """Run all N_PAIRS distinct OD pairs sequentially within a single continuous SUMO
    session (restarted only if the sim clock hits max_time before all pairs finish, so
    K>1 traffic slots evolve naturally across pairs)."""
    G = graph_info['G']
    G_dir = graph_info['G_dir']
    all_shortest_path_lengths_dir = graph_info['all_shortest_path_lengths_dir']
    junc2idx = graph_info['junc2idx']
    node2idx = graph_info['node2idx']
    sg_idx_map = graph_info['sg_idx_map']

    max_steps = config['simulation']['max_steps']
    max_time = float(config['simulation']['max_time']) * 3600.0
    num_actions = config['net']['max_degree']
    warmup_time = float(config['simulation'].get('warm_up_time', 0))

    ugv_speed = config['fleet']['max_speed'] * 1000.0 / 3600.0  # km/h → m/s
    psl_active = traffic_model in ('psl', 'td')
    reactive = algo in REACTIVE_ALGOS
    ped_mode = fleet_type == 'delivery_robot'
    adr_vtype_id = 'adr_vtype'
    n_pairs = len(od_pairs)

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
            traci.vehicletype.setMaxSpeed(adr_vtype_id, ugv_speed)
        t = 0.0
        while t < warmup_time:
            traci.simulationStep()
            t = traci.simulation.getTime()
        return t

    sim_time = start_sumo()

    if fix_oscillations:
        current_slot = psl_model.get_slot(sim_time) if psl_active else 0
        agent = OscillationFixWrapper(
            agent, action_mode, G, G_dir, idx2node, node2idx, junc2idx, sg_idx_map,
            all_shortest_path_lengths_dir, max_steps, num_actions, traffic_model,
            ugv_speed, ped_mode=ped_mode, ped_params=ped_params, time_slot=current_slot,
        )
        print(f"  [{desc}] oscillation-fix: {agent.n_fixed} (node,target) states overridden")

    for pair_idx in tqdm(range(n_pairs), desc=desc):
        start_idx, target_idx = od_pairs[pair_idx]
        start_node = idx2node[start_idx]
        target_node = idx2node[target_idx]

        current_slot = psl_model.get_slot(sim_time) if psl_active else 0

        if reactive:
            route_edges, tour_start_time, sim_time, err = drive_reactive_route(
                agent, action_mode, start_node, target_node, G, G_dir, idx2node,
                node2idx, junc2idx, sg_idx_map, all_shortest_path_lengths_dir,
                max_steps, num_actions, ugv_speed, sim_time, max_time,
                entity_prefix=f"ugv_{pair_idx}", ped_mode=ped_mode, ped_params=ped_params,
            )
            if err:
                append_result_row(results_path, start_idx, target_idx, warmup_time,
                                  tour_start_time, -1, len(route_edges),
                                  ';'.join(route_edges), -1, -1, -1, -1, err)
            else:
                total_time_taken = sim_time - tour_start_time
                append_result_row(results_path, start_idx, target_idx, warmup_time,
                                  tour_start_time, total_time_taken, len(route_edges),
                                  ';'.join(route_edges), -1, -1, -1, -1, 'NA')
        else:
            route_edges, err = plan_route_oneshot(
                agent, action_mode, start_node, target_node, current_slot,
                G, G_dir, idx2node, node2idx, junc2idx, sg_idx_map,
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
            sim_time, err = drive_fixed_route(
                entity_id, route_edges, sim_time, max_time, psl_active, psl_model,
                ped_mode, ped_params, ugv_speed, traffic_speed_start,
            )
            if err:
                append_result_row(results_path, start_idx, target_idx, warmup_time,
                                  tour_start_time, -1, len(route_edges),
                                  ';'.join(route_edges), traffic_segment_start, -1,
                                  traffic_speed_start, -1, err)
            else:
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
                    traffic_speed_start, traffic_speed_end, 'NA'
                )

        if sim_time >= max_time and pair_idx < n_pairs - 1:
            traci.close()
            sim_time = start_sumo()

    traci.close()


def main():
    parser = argparse.ArgumentParser(description="TRSP Sequential Evaluation (K>1 compatible)")
    parser.add_argument('-p', '--policy', type=str, required=True,
                        help="Policy ID under runs/trsp/")
    parser.add_argument('--fix-oscillations', action='store_true',
                        help="Offline-preprocess the policy agent to remove greedy-policy "
                             "oscillations/loops before evaluating (see OscillationFixWrapper).")
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

    od_pairs = select_od_pairs(num_nodes, seed=SEED, n_pairs=N_PAIRS)
    print(f"Selected {N_PAIRS} distinct evaluation OD pairs ({num_nodes} nodes total)")

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
    test_config['test'] = {'mode': 'sequential_eval', 'fix_oscillations': args.fix_oscillations}
    results_path = init_results(test_dir, test_config)

    run_kwargs = dict(
        od_pairs=od_pairs, idx2node=idx2node, graph_info=graph_info,
        config=config, sumo_config_cmd=sumo_config_cmd, psl_model=psl_model,
        traffic_model=traffic_model, fleet_type=fleet_type, dr_cfg=dr_cfg,
    )

    print(f"\n[Policy: {args.policy}] → {test_dir}")
    t0 = time.time()
    run_evaluation(policy_agent, algo, action_mode, results_path=results_path,
                   desc=args.policy, fix_oscillations=args.fix_oscillations, **run_kwargs)
    print(f"Policy eval done in {timedelta(seconds=int(time.time() - t0))} → {results_path}")

    # --- Dijkstra baseline evaluation (never reactive, never oscillation-fixed) ---
    dij_agent = DijkstraAgent(0)
    dij_results_path = init_baseline_csv(test_dir, 'dijkstra')

    print(f"\n[Dijkstra] → {dij_results_path}")
    t0 = time.time()
    run_evaluation(dij_agent, 'dijkstra', 'neighbor_index', results_path=dij_results_path,
                   desc='Dijkstra', **run_kwargs)
    print(f"Dijkstra eval done in {timedelta(seconds=int(time.time() - t0))} → {dij_results_path}")

    # --- DSP baseline evaluation (never reactive, never oscillation-fixed) ---
    if not ped_mode:
        dsp_agent = DSPAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                             graph_info['idx2node'], ugv_speed)
        dsp_algo = 'dsp'
    else:
        pso = config.get('traffic', {}).get('ped_speed_override', {})
        dsp_agent = DSPPEDAgent(0, graph_info['G_dir'], graph_info['node2idx'],
                                graph_info['idx2node'], ugv_speed,
                                gamma=pso.get('gamma', 1.913),
                                density_jam=pso.get('density_jam', 5.4),
                                v_min=pso.get('v_min', 0.1))
        dsp_algo = 'dsp_ped'
    dsp_results_path = init_baseline_csv(test_dir, 'dsp')

    print(f"\n[DSP] → {dsp_results_path}")
    t0 = time.time()
    run_evaluation(dsp_agent, dsp_algo, 'node_id', results_path=dsp_results_path,
                   desc='DSP', **run_kwargs)
    print(f"DSP eval done in {timedelta(seconds=int(time.time() - t0))} → {dsp_results_path}")


if __name__ == '__main__':
    main()
