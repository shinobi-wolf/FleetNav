import math
import os
import random
import numpy as np
import networkx as nx
import pygame

import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib

from dronesim.trsp.fleet.UGVSumo import UGVSumo
from dronesim.trsp.traffic import PSLTrafficModel, TDTrafficModel
from dronesim.trsp.ped_speed import weidmann_speed

INF = 100000

COMPLETION_REWARD = 50.0
TRUNCATION_PENALTY = -10.0      # was 0.0 in reference — zero gave no urgency signal
INVALID_ACTION_PENALTY = -10.0
MOVE_TIME_PENALTY_FAC = -0.3    # per second; encourages shorter traversal paths
PROGRESS_REWARD_FAC = 0.05      # per metre of progress towards target


def get_centroid(shape):
    if not shape:
        return 0, 0
    x_coords = [p[0] for p in shape]
    y_coords = [p[1] for p in shape]
    return np.mean(x_coords), np.mean(y_coords)


class SumoTRSPEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'quiet']}

    def __init__(self, config, render_mode=None):
        super().__init__()

        self.net_path = config['net']['net_path']
        self.num_actions = config['net']['max_degree']

        self.ugv_speed = config['fleet']['max_speed'] * 1000.0 / 3600.0  # km/h → m/s
        # NOTE: for ped mode this is provisional (only feeds the unused edge_tt); it is
        # overwritten from the ADR vtype's max desired speed in _register_adr_vtype().
        self.fleet_type = config['fleet'].get('type', 'ev')

        if self.fleet_type == 'delivery_robot':
            dr_cfg = config['fleet'].get('delivery_robot', {})
            self.dr_width = dr_cfg.get('width', 1.0)
            self.dr_length = dr_cfg.get('length', 0.8)
            self.dr_max_speed_factor = dr_cfg.get('max_speed_factor', 1.0)
            self.adr_vtype_id = 'adr_vtype'

        self.step_length = config['sumo']['step_length']
        self.traffic_scale = config['sumo'].get('traffic_scale', 1)
        self.delay = config['sumo'].get('delay', 0)

        self.max_time = config['simulation']['max_time'] * 3600.0  # hours → seconds
        self.max_steps = config['simulation']['max_steps']
        self.render_mode = config['simulation'].get('render_mode', 'quiet')
        self.warmup_time = config['simulation'].get('warm_up_time', 0)
        self.random_sg_pair = config['simulation']['random_sg_pair']

        if not self.random_sg_pair:
            self.start_idx = config['simulation']['start_node_idx']
            self.target_junc_id = config['simulation']['target_junc_id']

        self.build_graph()
        print(f"Graph: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        self.num_sg_pairs = len(self.sg_pairs)

        traffic_cfg = config.get('traffic', {'enabled': False})
        self.traffic_model = traffic_cfg.get('model', 'psl')   # 'psl' | 'td' | 'svc' | 'ped'
        if self.fleet_type == 'delivery_robot':
            self.traffic_model = 'ped'

        # svc (live vehicular speeds from a route file) and ped (live pedestrian speeds on
        # sidewalks) both derive edge speeds from live SUMO state, not a PSL/TD profile.
        self.live_speed_model = self.traffic_model in ('svc', 'ped')

        if self.live_speed_model:
            self.route_file = traffic_cfg.get('route_file', None)
            self.traffic = PSLTrafficModel({'enabled': False}, self.edge_ids)
        elif self.traffic_model == 'td':
            self.traffic = TDTrafficModel(
                traffic_cfg, self.edge_ids, seed=traffic_cfg.get('seed')
            )
        else:
            self.traffic = PSLTrafficModel(
                traffic_cfg, self.edge_ids, seed=traffic_cfg.get('seed')
            )
        self.current_slot = 0
        self.traffic_log: list[dict] = []

        # Pedestrian-density speed override (Weidmann): couples sidewalk crowding to the
        # robot's walking speed. Delivery-robot only; defaults off so other fleets/runs
        # are unaffected.
        pso = traffic_cfg.get('ped_speed_override', {})
        self.ped_speed_override = bool(pso.get('enabled', False)) and self.fleet_type == 'delivery_robot'
        self.ped_gamma = pso.get('gamma', 1.913)          # Weidmann shape parameter
        self.ped_density_jam = pso.get('density_jam', 5.4)  # ped/m² jam density (v → v_min)
        self.ped_v_min = pso.get('v_min', 0.1)            # m/s floor, avoids permanent stall
        self._lane_area_cache: dict[str, float] = {}

        num_nodes = len(self.node2idx)
        psl_K = self.traffic.K if self.traffic.enabled else 1

        # action_mode controls what the agent's action integer means:
        #   'neighbor_index' (default) — index into sorted neighbor list; SARSA/Dijkstra
        #   'node_id'                  — global node index with masking; DQN
        self.action_mode = config.get('agent', {}).get('action_mode', 'neighbor_index')

        if self.action_mode == 'node_id':
            self.action_space = spaces.Discrete(num_nodes)
            if self.live_speed_model:
                self.observation_space = spaces.Dict({
                    'graph_node':      spaces.Discrete(num_nodes),
                    'target_node':     spaces.Discrete(num_nodes),
                    'neighbor_speeds': spaces.Box(0.0, 2.0, shape=(self.num_actions,), dtype=np.float32),
                })
            else:
                self.observation_space = spaces.Dict({
                    'graph_node':  spaces.Discrete(num_nodes),
                    'target_node': spaces.Discrete(num_nodes),
                    'time_slot':   spaces.Discrete(psl_K),
                })
        else:
            self.action_space = spaces.Discrete(self.num_actions)
            if self.live_speed_model:
                self.observation_space = spaces.Dict({
                    'graph_node':      spaces.Discrete(num_nodes),
                    'target_node':     spaces.Discrete(num_nodes),
                    'neighbor_speeds': spaces.Box(0.0, 2.0, shape=(self.num_actions,), dtype=np.float32),
                    'best_dir':        spaces.Discrete(self.num_actions),
                })
            else:
                self.observation_space = spaces.Dict({
                    'graph_node':  spaces.Discrete(num_nodes),
                    'target_node': spaces.Discrete(num_nodes),
                    'time_slot':   spaces.Discrete(psl_K),
                    'best_dir':    spaces.Discrete(self.num_actions),
                })

        # runtime state — populated in reset()
        self.ugv = None
        self.screen = None
        self.clock = None
        self.time_elapsed = 0.0
        self.last_move_time = 0.0
        self.current_timestep = 0
        self.terminated = False
        self.truncated = False
        self.total_ev_distance = 0.0
        self.next_action_manual = False
        self.next_action = None
        self.info = {}
        self._dist_delta = 0.0   # metres of progress this step (positive = closer)
        self.last_reward = 0.0

    # ------------------------------------------------------------------ graph

    def _resolve_net_path(self):
        """Path to the .net.xml. When net_path is a .sumocfg, read its
        <net-file> entry so a config whose basename differs from the net
        (e.g. a *_train variant that shares one net) still resolves; fall
        back to swapping the suffix."""
        if not self.net_path.endswith('.sumocfg'):
            return self.net_path
        try:
            import xml.etree.ElementTree as ET
            nf = ET.parse(self.net_path).getroot().find('.//net-file')
            if nf is not None and nf.get('value'):
                return os.path.join(os.path.dirname(self.net_path), nf.get('value'))
        except Exception:
            pass
        return self.net_path.replace('.sumocfg', '.net.xml')

    def build_graph(self):
        net_path = self._resolve_net_path()
        ped_mode = self.fleet_type == 'delivery_robot'
        net = sumolib.net.readNet(net_path, withPedestrianConnections=ped_mode)

        G = nx.DiGraph()
        junc2nodes_map = {}
        for edge in net.getEdges():
            eid = edge.getID()
            if eid.startswith(':'):
                continue
            if ped_mode and not any(lane.allows('pedestrian') for lane in edge.getLanes()):
                continue
            n1 = edge.getToNode()
            shape1 = n1.getShape()
            if not shape1:
                continue
            cx1, cy1 = get_centroid(shape1)
            G.add_node(n1.getID(), junc_id=n1.getID(), x=cx1, y=cy1)
            junc2nodes_map[n1.getID()] = [n1.getID()]

            n2 = edge.getFromNode()
            shape2 = n2.getShape()
            if not shape2:
                continue
            cx2, cy2 = get_centroid(shape2)
            G.add_node(n2.getID(), junc_id=n2.getID(), x=cx2, y=cy2)
            junc2nodes_map[n2.getID()] = [n2.getID()]

            edge_length = edge.getLength()
            edge_tt = edge_length / self.ugv_speed
            if ped_mode:
                sidewalk_lane = next(
                    (lane.getID() for lane in edge.getLanes() if lane.allows('pedestrian')),
                    None,
                )
                G.add_edge(n2.getID(), n1.getID(),
                           length=edge_length, id=eid, edge_tt=edge_tt,
                           sidewalk_lane=sidewalk_lane)
            else:
                G.add_edge(n2.getID(), n1.getID(), length=edge_length, id=eid, edge_tt=edge_tt)

        self.edge_ids = [data['id'] for _, _, data in G.edges(data=True)]
        G_und = G.to_undirected()
        all_spl = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))

        self.G = G_und
        self.G_dir = G
        self.all_shortest_path_lengths = all_spl
        self.junc2nodes_map = junc2nodes_map
        self.junction_list = list(junc2nodes_map.keys())
        self.junc2idx = {j: i for i, j in enumerate(sorted(self.junction_list))}
        self.idx2junc = {i: j for j, i in self.junc2idx.items()}
        self.node2idx = {n: i for i, n in enumerate(sorted(G_und.nodes))}
        self.idx2node = {i: n for n, i in self.node2idx.items()}
        self.sg_pairs = [
            (self.node2idx[n], self.junc2idx[j])
            for n in G_und.nodes
            for j in self.junction_list
            if G_und.nodes[n]['junc_id'] != j
        ]

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        try:
            if traci.isLoaded():
                traci.close(False)
        except traci.TraCIException:
            pass

        self.current_timestep = 0
        self.terminated = False
        self.truncated = False
        self.time_elapsed = 0.0
        self.last_move_time = 0.0
        self.total_ev_distance = 0.0
        self.screen = None
        self.clock = None
        self.next_action_manual = False
        self.next_action = None

        sumo_bin = 'sumo-gui' if self.render_mode == 'human' else 'sumo'
        delay = self.delay if self.render_mode == 'human' else 0
        sumo_cmd = [
            sumo_bin,
            '-c', self.net_path,
            '--step-length', str(self.step_length),
            '--scale', str(self.traffic_scale),
            '--start',
            '--delay', str(delay),
            '--no-warnings',
            '--no-step-log',
            '--no-duration-log',
        ]
        if self.live_speed_model and self.route_file:
            sumo_cmd += ['--route-files', self.route_file]
        traci.start(sumo_cmd)

        if self.fleet_type == 'delivery_robot':
            self._register_adr_vtype()

        while self.time_elapsed < self.warmup_time:
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()

        self.current_slot = (
            self.traffic.get_slot(self.time_elapsed) if self.traffic.enabled else 0
        )

        if self.random_sg_pair:
            init_node, target_junction_id = self._sample_sg_pair()
        else:
            init_node = self.idx2node[self.start_idx]
            target_junction_id = self.target_junc_id

        init_node_idx = self.node2idx[init_node]
        target_nodes = self.junc2nodes_map[target_junction_id]
        target_node = target_nodes[int(np.argmin([
            self.all_shortest_path_lengths.get(init_node, {}).get(tn, INF)
            for tn in target_nodes
        ]))]

        self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.ugv_speed)
        self.ugv.task_list = [self.node2idx[target_node]]

        obs = self._get_observation()
        self.info = self._analyzer()
        return obs, self.info

    def step(self, action):
        if self.terminated:
            return None, None, self.terminated, False, self.info

        if self.next_action_manual:
            action = self.next_action
            self.next_action_manual = False

        # snapshot distance to target before the move
        task_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        ugv_node_before = self.idx2node[self.ugv.position]
        dist_before = (
            self.all_shortest_path_lengths.get(ugv_node_before, {}).get(task_node, INF)
            if task_node else 0.0
        )

        self.ugv_handler(action)

        ugv_node_after = self.idx2node[self.ugv.position]
        dist_after = (
            self.all_shortest_path_lengths.get(ugv_node_after, {}).get(task_node, INF)
            if task_node else 0.0
        )
        self._dist_delta = dist_before - dist_after  # positive = moved closer

        self.current_timestep += 1
        truncated = self.task_handler()
        reward = self._get_reward()
        self.last_reward = reward
        self._check_termination_condition()

        obs = self._get_observation()
        self.info = self._analyzer()
        return obs, reward, self.terminated, truncated, self.info

    # ------------------------------------------------------------------ internals

    def _register_adr_vtype(self):
        traci.vehicletype.copy('DEFAULT_PEDTYPE', self.adr_vtype_id)
        traci.vehicletype.setWidth(self.adr_vtype_id, self.dr_width)
        traci.vehicletype.setLength(self.adr_vtype_id, self.dr_length)
        traci.vehicletype.setSpeedFactor(self.adr_vtype_id, self.dr_max_speed_factor)
        # Cap the ADR vtype at the configured robot speed (DEFAULT_PEDTYPE's inherited
        # maxSpeed is a large SUMO default, ~10 m/s). SUMO then enforces the real speed,
        # and we source the env's free-flow speed back from the vtype so v_free (Weidmann)
        # and the neighbor-speed normalization match what SUMO actually caps the robot at.
        traci.vehicletype.setMaxSpeed(self.adr_vtype_id, self.ugv_speed)
        self.ugv_speed = traci.vehicletype.getMaxSpeed(self.adr_vtype_id)

    def get_valid_action_mask(self) -> np.ndarray:
        """Boolean mask of shape (num_nodes,) — True for valid next-hop node indices.

        Only meaningful when action_mode == 'node_id'.
        """
        ugv_node = self.idx2node[self.ugv.position]
        mask = np.zeros(len(self.node2idx), dtype=bool)
        for nb in self.G.neighbors(ugv_node):
            mask[self.node2idx[nb]] = True
        return mask

    def ugv_handler(self, action):
        if self.ugv.status == 2:
            return
        # entity still traversing previous edge
        active_check = (traci.person.getIDList() if self.fleet_type == 'delivery_robot'
                        else traci.vehicle.getIDList())
        if self.ugv.veh_id and self.ugv.veh_id in active_check:
            self.ugv.status = 0
            return
        self.ugv.status = 1

        self.ugv.last_position = self.ugv.position

        # In node_id mode the action is a global node index; convert to neighbor
        # list index before passing to UGVSumo, which expects the latter.
        if self.action_mode == 'node_id':
            current_node = self.idx2node[self.ugv.position]
            nb_idxs = sorted([self.node2idx[nb] for nb in self.G.neighbors(current_node)])
            if action not in nb_idxs:
                self.ugv.invalid_action_this_step = True
                return
            neighbor_action = nb_idxs.index(action)
        else:
            neighbor_action = action

        move_flag, edge_id, edge_dir = self.ugv.move(
            neighbor_action, self.time_elapsed, self.G, self.G_dir, self.node2idx, self.idx2node
        )
        self.last_move_time = self.time_elapsed

        if not move_flag:
            return

        depart_speed = self._edge_speed(edge_id, self.current_slot)
        eid = f"ugv_{self.ugv.agent_id}_{int(self.time_elapsed)}"
        self.ugv.veh_id = eid
        if self.fleet_type == 'delivery_robot':
            traci.person.add(eid, edge_id, pos=0.0, depart=-1, typeID=self.adr_vtype_id)
            traci.person.appendWalkingStage(eid, [edge_id], arrivalPos=-1)
            traci.person.setSpeed(eid, depart_speed)
            traci.person.setColor(eid, (255, 0, 0, 255))
        else:
            route_id = f"{eid}_route"
            traci.route.add(route_id, [edge_id])
            traci.vehicle.add(eid, route_id, departSpeed=depart_speed)
            traci.vehicle.setSpeed(eid, depart_speed)
            traci.vehicle.setColor(eid, (255, 0, 0, 255))

        traci.simulationStep()
        self.time_elapsed = traci.simulation.getTime()

        inserted = False
        while self.time_elapsed < self.max_time:
            if self.fleet_type == 'delivery_robot':
                active_ids = set(traci.person.getIDList())
                if self.ugv.veh_id in active_ids:
                    inserted = True
                    if self.ped_speed_override:
                        lane_id = traci.person.getLaneID(self.ugv.veh_id)
                        if lane_id and not lane_id.startswith(':'):  # skip internal/junction lanes
                            traci.person.setSpeed(
                                self.ugv.veh_id,
                                self._weidmann_speed(lane_id, exclude_self=True),
                            )
                elif inserted:
                    self.ugv.status = 1
                    break
            else:
                active_ids  = set(traci.vehicle.getIDList())
                arrived_ids = set(traci.simulation.getArrivedIDList())
                if self.ugv.veh_id in active_ids:
                    inserted = True
                    if self.traffic_model == 'psl' or self.traffic_model == 'td':
                        new_slot = self.traffic.get_slot(self.time_elapsed)
                        if new_slot != self.current_slot:
                            old_slot = self.current_slot
                            self.current_slot = new_slot
                            new_speed = self._edge_speed(edge_id, new_slot)
                            traci.vehicle.setSpeed(self.ugv.veh_id, new_speed)
                            if self.render_mode != "quiet":
                                print(f"[{self.time_elapsed:.1f}s] SLOT CHANGE: {old_slot} → {new_slot}"
                                      f"  edge={edge_id}  speed={new_speed:.2f} m/s")
                            self.traffic_log.append({"t": self.time_elapsed, "type": "slot_change",
                                                     "old_slot": old_slot, "new_slot": new_slot,
                                                     "edge": edge_id, "speed_ms": round(new_speed, 4)})
                elif self.ugv.veh_id in arrived_ids or inserted:
                    # Arrived this step, or was confirmed in-network and has since left
                    self.ugv.status = 1
                    break
            # else: not yet inserted into SUMO — keep stepping

            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()

    def task_handler(self):
        self.truncated = False
        at_target = (
            self.ugv.task_list and
            self.G.nodes[self.idx2node[self.ugv.position]]['junc_id'] ==
            self.G.nodes[self.idx2node[self.ugv.task_list[0]]]['junc_id']
        )
        timed_out = self.current_timestep > self.max_steps

        if timed_out:
            self.truncated = True
            self._reset_episode()
        elif at_target:
            self.ugv.task_completed_this_step = True
            self.truncated = True
            self._reset_episode()

        return self.truncated

    def _reset_episode(self):
        self.current_timestep = 0
        init_node, target_junction_id = self._sample_sg_pair()
        init_node_idx = self.node2idx[init_node]
        target_nodes = self.junc2nodes_map[target_junction_id]
        target_node = target_nodes[int(np.argmin([
            self.all_shortest_path_lengths.get(init_node, {}).get(tn, INF)
            for tn in target_nodes
        ]))]
        self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.ugv_speed)
        self.ugv.task_list = [self.node2idx[target_node]]

    def _sample_sg_pair(self):
        init_node = random.choice(list(self.G.nodes))
        current_junc = self.G.nodes[init_node]['junc_id']
        available = [j for j in self.junction_list if j != current_junc]
        target_junction_id = random.choice(available)
        return init_node, target_junction_id

    def _edge_speed(self, edge_id: str, slot: int) -> float:
        """Speed (m/s) for an edge at a time slot, falling back to fleet max speed."""
        if self.traffic.enabled and edge_id in self.traffic.profiles:
            return self.traffic.get_speed(edge_id, slot)
        return self.ugv_speed

    def _lane_area(self, lane_id: str) -> float:
        """Cached sidewalk-lane footprint (m²); lane length/width are static."""
        a = self._lane_area_cache.get(lane_id)
        if a is None:
            a = traci.lane.getLength(lane_id) * traci.lane.getWidth(lane_id)
            self._lane_area_cache[lane_id] = a
        return a

    def _weidmann_speed(self, lane_id: str, exclude_self: bool = False) -> float:
        """Robot walking speed (m/s) on a sidewalk given its pedestrian density (Weidmann).

        Person count comes from the edge (this TraCI build has no per-lane person counter);
        on a pedestrian edge that count is exactly the sidewalk lane's pedestrians. Shared
        formula in `dronesim.trsp.ped_speed` so the DSP-PED baseline and eval match exactly.
        """
        n = len(traci.edge.getLastStepPersonIDs(traci.lane.getEdgeID(lane_id)))
        if exclude_self:
            n = max(n - 1, 0)                 # discount the robot itself
        return weidmann_speed(n, self._lane_area(lane_id), self.ugv_speed,
                              self.ped_gamma, self.ped_density_jam, self.ped_v_min)

    def _get_reward(self):
        if self.ugv.invalid_action_this_step:
            self.ugv.invalid_action_this_step = False
            return INVALID_ACTION_PENALTY

        move_time = self.time_elapsed - self.last_move_time
        reward = MOVE_TIME_PENALTY_FAC * move_time + PROGRESS_REWARD_FAC * self._dist_delta

        if self.truncated:
            self.truncated = False
            if self.ugv.task_completed_this_step:
                self.ugv.task_completed_this_step = False
                reward += COMPLETION_REWARD
            else:
                reward += TRUNCATION_PENALTY

        return reward

    def _check_termination_condition(self):
        self.terminated = self.time_elapsed >= self.max_time

    def _get_neighbor_speeds(self) -> np.ndarray:
        """Normalized mean speeds on outgoing edges, sorted by node2idx, padded to max_degree."""
        current_node = self.idx2node[self.ugv.position]
        nb_sorted = sorted(self.G.neighbors(current_node), key=lambda n: self.node2idx[n])
        speeds = []
        for nb in nb_sorted:
            edge_data = self.G_dir[current_node][nb]
            if self.fleet_type == 'delivery_robot':
                lane_id = edge_data.get('sidewalk_lane')
                if self.ped_speed_override:
                    # Effective robot speed via the same Weidmann f(density) it will be
                    # subject to; exclude_self=False since the robot is not on neighbor lanes.
                    raw = self._weidmann_speed(lane_id) if lane_id else self.ugv_speed
                else:
                    raw = traci.lane.getLastStepMeanSpeed(lane_id) if lane_id else self.ugv_speed
            else:
                raw = traci.edge.getLastStepMeanSpeed(edge_data['id'])
            raw = self.ugv_speed if raw < 0 else raw        # -1 → no entities → free-flow
            speeds.append(min(raw / self.ugv_speed, 2.0))   # normalize, cap at 2×
        speeds += [1.0] * (self.num_actions - len(speeds))  # pad to max_degree
        return np.array(speeds, dtype=np.float32)

    def _get_observation(self):
        ugv_node = self.idx2node[self.ugv.position]
        task_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        target_node_idx = self.node2idx[task_node] if task_node else 0

        if self.action_mode == 'node_id':
            if self.live_speed_model:
                return {
                    'graph_node':      self.ugv.position,
                    'target_node':     target_node_idx,
                    'neighbor_speeds': self._get_neighbor_speeds(),
                }
            return {
                'graph_node':  self.ugv.position,
                'target_node': target_node_idx,
                'time_slot':   self.current_slot,
            }

        # neighbor_index mode: compute best_dir hint for Dijkstra/SARSA
        best_dir = 0
        if task_node:
            nb_idxs = sorted([self.node2idx[nb] for nb in self.G.neighbors(ugv_node)])
            steps = [
                self.all_shortest_path_lengths.get(self.idx2node[ni], {}).get(task_node, INF)
                for ni in nb_idxs
            ]
            steps += [INF] * (self.num_actions - len(steps))
            best_dir = int(np.argmin(steps[:self.num_actions]))

        if self.live_speed_model:
            return {
                'graph_node':      self.ugv.position,
                'target_node':     target_node_idx,
                'neighbor_speeds': self._get_neighbor_speeds(),
                'best_dir':        best_dir,
            }
        return {
            'graph_node':  self.ugv.position,
            'target_node': target_node_idx,
            'time_slot':   self.current_slot,
            'best_dir':    best_dir,
        }

    def _analyzer(self):
        self.total_ev_distance = self.ugv.distance_traveled
        ugv_pos_node = self.idx2node[self.ugv.position]
        task_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        return {
            'time_elapsed':      self.time_elapsed,
            'current_timestep':  self.current_timestep,
            'ugv_position':      ugv_pos_node,
            'ugv_target':        task_node,
            'ugv_position_idx':  self.ugv.position,
            'ugv_target_idx':    self.ugv.task_list[0] if self.ugv.task_list else None,
            'ugv_speed_ms':      self.ugv_speed,
            'ugv_speed_kmh':     self.ms_to_kmh(self.ugv_speed),
            'distance_traveled': self.ugv.distance_traveled,
            'time_slot':         self.current_slot,
        }

    # ------------------------------------------------------------------ render

    def render(self):
        if self.render_mode == 'quiet':
            return

        ugv_pos_node = self.idx2node[self.ugv.position]
        task_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None

        if self.render_mode == 'print':
            print(
                f"t={self.time_elapsed:8.1f}s | slot={self.current_slot} | "
                f"step={self.current_timestep:4d}/{self.max_steps} | "
                f"pos={ugv_pos_node} → target={task_node} | "
                f"dist={self.total_ev_distance:.1f}m | "
                f"r={self.last_reward:+7.2f} | "
                f"{'DONE' if self.terminated else 'trunc' if self.truncated else '     '}"
            )
            return

        # human: pygame overlay (SUMO GUI already open)
        if self.screen is None:
            pygame.init()
            self.screen_width = 1000
            self.screen_height = 800
            self.clock = pygame.time.Clock()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('TRSP Environment')

        self.screen.fill((255, 255, 255))

        pos_x = [self.G.nodes[n]['x'] for n in self.G.nodes]
        pos_y = [self.G.nodes[n]['y'] for n in self.G.nodes]
        min_x, max_x = min(pos_x), max(pos_x)
        min_y, max_y = min(pos_y), max(pos_y)
        margin = 50
        scale = min(
            (self.screen_width - 2 * margin) / max(1, max_x - min_x),
            (self.screen_height - 2 * margin) / max(1, max_y - min_y),
        )

        def tx(x, y):
            return (int(margin + (x - min_x) * scale), int(margin + (max_y - y) * scale))

        _OFFSET = 3      # perpendicular offset (px)
        _ARROW  = 8      # arrowhead length (px)
        _ARROW_HW = 3    # arrowhead half-width (px)

        for u, v, edata in self.G_dir.edges(data=True):
            eid = edata['id']
            speed = (
                self.traffic.get_speed(eid, self.current_slot)
                if self.traffic.enabled and eid in self.traffic.profiles
                else self.ugv_speed
            )
            color = self._speed_to_color(speed)
            x1, y1 = tx(self.G.nodes[u]['x'], self.G.nodes[u]['y'])
            x2, y2 = tx(self.G.nodes[v]['x'], self.G.nodes[v]['y'])
            dx, dy = x2 - x1, y2 - y1
            ln = math.hypot(dx, dy)
            if ln < 1.0:
                continue
            ux, uy = dx / ln, dy / ln
            ox, oy = -uy * _OFFSET, ux * _OFFSET
            ax, ay = x1 + ox, y1 + oy
            bx, by = x2 + ox, y2 + oy
            pygame.draw.line(self.screen, color,
                             (int(ax), int(ay)), (int(bx), int(by)), 2)
            if ln > 2.5 * _ARROW:
                t = 0.6
                tip_x = ax + (bx - ax) * t
                tip_y = ay + (by - ay) * t
                base_x = tip_x - ux * _ARROW
                base_y = tip_y - uy * _ARROW
                left  = (base_x - uy * _ARROW_HW, base_y + ux * _ARROW_HW)
                right = (base_x + uy * _ARROW_HW, base_y - ux * _ARROW_HW)
                pygame.draw.polygon(self.screen, color, [
                    (int(tip_x), int(tip_y)),
                    (int(left[0]),  int(left[1])),
                    (int(right[0]), int(right[1])),
                ])

        for node in self.G.nodes:
            x, y = tx(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
            pygame.draw.circle(self.screen, (80, 80, 80), (x, y), 10)
            if pygame.font:
                font = pygame.font.Font(None, 16)
                lbl = font.render(str(self.node2idx[node]), True, (255, 255, 255))
                self.screen.blit(lbl, lbl.get_rect(center=(x, y)))

        if task_node:
            tx_, ty_ = tx(self.G.nodes[task_node]['x'], self.G.nodes[task_node]['y'])
            surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 0, 0, 140), (10, 10), 10)
            self.screen.blit(surf, (tx_ - 10, ty_ - 10))

        ugv_node = self.idx2node[self.ugv.position]
        ux, uy = tx(self.G.nodes[ugv_node]['x'], self.G.nodes[ugv_node]['y'])
        color = (0, 200, 0) if self.ugv.status == 1 else (255, 200, 0)
        surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, 180), (10, 10), 10)
        self.screen.blit(surf, (ux - 10, uy - 10))

        if pygame.font:
            font = pygame.font.Font(None, 26)
            xo, yo = self.screen_width - 320, 10
            for line in [
                f"t = {self.time_elapsed:.1f} s",
                f"slot = {self.current_slot}",
                f"step = {self.current_timestep}/{self.max_steps}",
                f"pos = {ugv_pos_node}",
                f"target = {task_node}",
                f"dist = {self.total_ev_distance:.1f} m",
                f"speed = {self.format_speed(self.ugv_speed)}",
            ]:
                self.screen.blit(font.render(line, True, (0, 0, 0)), (xo, yo))
                yo += 28

        pygame.display.flip()

        print("Press Return to continue…")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                    elif event.key in (pygame.K_KP0, pygame.K_KP1, pygame.K_KP2, pygame.K_KP3):
                        self.next_action = event.key - pygame.K_KP0
                        self.next_action_manual = True
                        waiting = False
            self.clock.tick(30)

    # ------------------------------------------------------------------ helpers

    def _speed_to_color(self, speed_ms: float) -> tuple:
        """Red (slow) → green (fast) linear interpolation over the PSL speed range."""
        if not self.traffic.enabled:
            return (180, 180, 180)
        v_range = self.traffic.v_max - self.traffic.v_min
        t = (speed_ms - self.traffic.v_min) / v_range if v_range > 1e-9 else 1.0
        t = max(0.0, min(1.0, t))
        return (int(255 * (1 - t)), int(255 * t), 0)

    def ms_to_kmh(self, speed_ms):
        return None if speed_ms is None else speed_ms * 3.6

    def format_speed(self, speed_ms):
        if speed_ms is None:
            return 'N/A'
        return f'{speed_ms:.2f} m/s ({self.ms_to_kmh(speed_ms):.1f} km/h)'

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        try:
            traci.close()
        except Exception:
            pass
