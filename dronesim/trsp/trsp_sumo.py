from collections import defaultdict
from shutil import move
import numpy as np
import networkx as nx
import random
import pygame # Added pygame
import os # Added os for path joining
import time

import gymnasium as gym
from gymnasium import spaces

import traci
import sumolib

from dronesim.envs.fleet.UGVSumo import UGVSumo
import numpy as np

INF = 100000

def get_centroid(shape):
    """Calculates the center point of a shape (list of x,y tuples)."""
    if not shape:
        return 0, 0
    x_coords = [p[0] for p in shape]
    y_coords = [p[1] for p in shape]
    return np.mean(x_coords), np.mean(y_coords)

class SumoTRSPEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'quiet']}

    def __init__(self, config, render_mode=None): # Default render_mode is None
        super(SumoTRSPEnv, self).__init__()

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.net_path = config["net"]["net_path"]

        # Agent vars
        self.ugv_speed = config["fleet"]['max_speed'] * 1000 / 3600 # in m/s
        self.num_actions = config['net']['max_degree']

        # Sumo vars
        self.step_length = config['sumo']['step_length']
        self.traffic_scale = config['sumo'].get('traffic_scale', 1)
        self.delay = config['sumo'].get('delay', 0)

        # Simulation vars
        self.max_time = config['simulation']['max_time']*3600 # in seconds
        self.max_steps = config['simulation']['max_steps']
        self.render_mode = config['simulation'].get('render_mode')
        self.warmup_time = config['simulation']['warm_up_time']
        self.random_sg_pair = config['simulation']['random_sg_pair']

        # Traffic vars
        self.traffic_speed_limits = tuple(np.array(config['traffic']['speed_limits']) * 1000 / 3600) # in m/s
        self.traffic_speed_step = config['traffic']['speed_step'] * 1000 / 3600 # in m/s
        self.traffic_seg_duration = config['traffic']['seg_duration'] * 60 # in seconds
        self.traffic_seed = config['traffic']['seed']

        min_speed, max_speed = self.traffic_speed_limits
        self.traffic_speed_range = np.arange(
            min_speed, 
            max_speed + 1e-9, 
            self.traffic_speed_step
        )
        segment_duration = self.traffic_seg_duration 
        num_segments = int(np.ceil(self.max_time / segment_duration))
        random.seed(self.traffic_seed)
        np.random.seed(self.traffic_seed)
        self.traffic_speed_segments = np.random.choice(self.traffic_speed_range, size=num_segments)

        print(
            f"Traffic segment duration: {self.traffic_seg_duration:.0f} s "
            f"({self.traffic_seg_duration / 60.0:.1f} min)"
            f"Number of segments: {num_segments}"
        )
        print("First 10 traffic speed segments:")
        for seg_idx, speed_ms in enumerate(self.traffic_speed_segments[:10]):
            print(
                f"  seg {seg_idx}: {speed_ms:.2f} m/s "
                f"({self.ms_to_kmh(speed_ms):.1f} km/h)"
            )

        self.speed_changed_this_step = False
        self.last_speed_change_time = None
        self.last_speed_change_from = None
        self.last_speed_change_to = None
        self.last_speed_change_segment_from = None
        self.last_speed_change_segment_to = None

        self.build_graph()
        print(f"Graph built with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")
        self.num_sg_pairs = len(self.sg_pairs)

        if not self.random_sg_pair:
            self.start_idx = config['simulation']['start_node_idx']
            self.target_junc_id = config['simulation']['target_junc_id']

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.Discrete(self.num_actions)
        
        steps2dest_space = spaces.Box(
            low=0,
            high=INF,
            shape=(self.num_actions,),
            dtype=np.float32
        )
        graph_node_space = spaces.Discrete(1000)
        best_dir_space = spaces.Discrete(self.num_actions)
        sg_pair_idx_space = spaces.Discrete(self.num_sg_pairs)
        time_slot_space = spaces.Discrete(len(self.traffic_speed_segments))
        self.observation_space = spaces.Dict({
            'steps2dest': steps2dest_space,
            'graph_node': graph_node_space,
            'target_node': graph_node_space,
            'best_dir': best_dir_space,
            'sg_pair_idx': sg_pair_idx_space,
            'time_slot': time_slot_space,
        })

    # # Crosswalk graph       
    # def build_graph(self,):
    #     net_path = self.net_path.replace('.sumocfg', '.net.xml')
    #     net = sumolib.net.readNet(net_path, withPedestrianConnections=True)
    #     junctions = net.getNodes()
    #     edge_eps = {}
    #     junc2nodes_map = defaultdict(list)
        
    #     for junction in junctions:
    #         connections = junction.getConnections()
    #         for connection in connections:
    #             # 1. Inspect the source of the connection
    #             ep = None
    #             edge = None
    #             dir = None
    #             if connection._from.getFunction() == 'walkingarea':
    #                 ep = connection._from
    #                 edge = connection._to
    #                 dir = 0
    #             elif connection._to.getFunction() == 'walkingarea':
    #                 ep = connection._to
    #                 edge = connection._from
    #                 dir = 1
    #             if ep and edge:
    #                 edge_id = edge.getID()
    #                 eps = [None, None]
    #                 eps[dir] = ep
    #                 if edge_id not in edge_eps:
    #                     edge_eps[edge_id] = {
    #                         "edge": edge,
    #                         "eps": eps
    #                     }
    #                 else:
    #                     edge_eps[edge_id]["eps"][dir] = ep
    #                     if len([ep for ep in edge_eps[edge_id]["eps"] if ep is not None]) > 2:
    #                         print("More than 2 entry points for edge ", edge_id)
    #                         raise ValueError("More than 2 entry points for edge ", edge_id)

    #     G = nx.DiGraph()
    #     for edge_id, edge_data in edge_eps.items():
    #         entry_points = edge_data["eps"]
    #         for ep in entry_points:
    #             shape = ep.getShape()
    #             cx, cy = get_centroid(shape)
    #             G.add_node(ep._id, junc_id=ep._from._id,junc_coord=ep._from._coord, x=cx, y=cy)
    #             junc2nodes_map[ep._from._id].append(ep._id)
    #         edge_obj = edge_data["edge"]
    #         edge_length = None
    #         edge_type = edge_obj.getFunction()
    #         if edge_type == 'crossing':
    #             edge_length = 5.0
    #             G.add_edge(entry_points[0]._id, entry_points[1]._id, id=edge_obj._id, length=edge_length, type=edge_type)
    #             # G.add_edge(entry_points[1]._id, entry_points[0]._id, edge=edge_obj._id, length=edge_length, type=edge_type)
    #         else:
    #             edge_length = edge_obj.getLength()
    #             edge_type = 'sidewalk'
    #             G.add_edge(entry_points[0]._id, entry_points[1]._id, id=edge_obj._id, length=edge_length, type=edge_type)
        
    #     G_und = G.to_undirected()

    #     all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_und, weight='length'))
    #     self.G, self.G_dir, self.all_shortest_path_lengths, self.junc2nodes_map = G_und, G, all_shortest_path_lengths, junc2nodes_map

    #     self.num_junctions = len(self.junc2nodes_map)
    #     self.junction_list = self.junc2nodes_map.keys()
    #     self.node2idx = {node: idx for idx, node in enumerate(sorted(self.G.nodes))}
    #     self.idx2node = {idx: node for node, idx in self.node2idx.items()}
    #     self.sg_pairs = []
    #     for node_idx in range(len(self.idx2node)):
    #         for junc_id in self.junction_list:
    #             node = self.idx2node[node_idx]
    #             if self.G.nodes[node]['junc_id'] != junc_id:
    #                 self.sg_pairs.append((node_idx, junc_id))

    def build_graph(self,):
        net_path = self.net_path.replace('.sumocfg', '.net.xml')
        net = sumolib.net.readNet(net_path, withPedestrianConnections=False)

        G = nx.DiGraph()
        junc2nodes_map = {}
        for edge in net.getEdges():
            eid = edge.getID()
            if eid.startswith(':'):
                continue
            n1 = edge.getToNode()
            shape = n1.getShape()
            if not shape:
                continue
            cx, cy = get_centroid(shape)
            G.add_node(n1.getID(), junc_id=n1.getID(), x=cx, y=cy)
            junc2nodes_map[n1.getID()] = [n1.getID()]
            n2 = edge.getFromNode()
            shape2 = n2.getShape()
            if not shape2:
                continue
            cx2, cy2 = get_centroid(shape2)
            G.add_node(n2.getID(), junc_id=n2.getID(), x=cx2, y=cy2)
            junc2nodes_map[n2.getID()] = [n2.getID()]
            
            edge_speed = self.traffic_speed_range[0]
            edge_length = edge.getLength()
            edge_tt = edge_length / edge_speed
            G.add_edge(n2.getID(), n1.getID(), length=edge_length, id=eid, edge_speed=edge_speed, edge_tt=edge_tt)

        G_und = G.to_undirected()
        all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))

        self.G, self.G_dir, self.all_shortest_path_lengths, self.junc2nodes_map = G_und, G, all_shortest_path_lengths, junc2nodes_map
        self.junction_list = self.junc2nodes_map.keys()
        self.junc2idx = {junc: idx for idx, junc in enumerate(sorted(self.junction_list))}
        self.idx2junc = {idx: junc for junc, idx in self.junc2idx.items()}
        self.node2idx = {node: idx for idx, node in enumerate(sorted(self.G.nodes))}
        self.idx2node = {idx: node for node, idx in self.node2idx.items()}
        self.sg_pairs = []
        for node in self.G.nodes:
            for junc_id in self.junction_list:
                if self.G.nodes[node]['junc_id'] != junc_id:
                    self.sg_pairs.append((self.node2idx[node], self.junc2idx[junc_id]))
    
    def get_current_speed(self):
        """
        Returns the target traffic speed for a given current_time.
        """
        current_time = self.time_elapsed
        # 1. Handle edge case: time hasn't started yet
        if current_time < 0:
            return self.traffic_speed_segments[0]

        # 2. Calculate the corresponding index for the current time
        segment_index = int(current_time / self.traffic_seg_duration)
        
        # 3. Handle edge case: current_time exceeds the pre-calculated segments (max_time)
        if segment_index >= len(self.traffic_speed_segments):
            # Clamp to the final generated segment to avoid an IndexError
            segment_index = len(self.traffic_speed_segments) - 1
            
        return self.traffic_speed_segments[segment_index]

    def ms_to_kmh(self, speed_ms):
        if speed_ms is None:
            return None
        return speed_ms * 3.6

    def format_speed(self, speed_ms):
        if speed_ms is None:
            return "N/A"
        return f"{speed_ms:.2f} m/s ({self.ms_to_kmh(speed_ms):.1f} km/h)"

    def get_current_segment_index(self):
        if len(self.traffic_speed_segments) == 0:
            return -1
        current_time = max(0.0, self.time_elapsed)
        segment_index = int(current_time / self.traffic_seg_duration)
        return min(segment_index, len(self.traffic_speed_segments) - 1)

    def get_actual_vehicle_speed(self):
        veh_id = getattr(self.ugv, 'veh_id', None) if hasattr(self, 'ugv') else None
        if not veh_id:
            return None
        try:
            if veh_id in traci.vehicle.getIDList():
                return traci.vehicle.getSpeed(veh_id)
        except traci.TraCIException:
            return None
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call parent reset for seeding RNG
        if seed: # Seeding is handled by super().reset()
            random.seed(seed)
            np.random.seed(seed) # Also seed numpy for consistent sampling if used

        try:
            if traci.isLoaded():
                traci.close(False)  # or traci.close(); False = don't wait
        except traci.TraCIException:
            # If no active connection or already closed, just ignore
            pass

        # Simulation State
        self.current_timestep = 0
        self.terminated = False
        self.truncated = False
        self.time_elapsed = 0.0
        self.last_move_time = 0.0
        self.speed_changed_this_step = False
        self.last_speed_change_time = None
        self.last_speed_change_from = None
        self.last_speed_change_to = None
        self.last_speed_change_segment_from = None
        self.last_speed_change_segment_to = None

        # Initialize pygame display if not already done
        self.screen = None
        self.clock = None
        self.next_action_manual = False
        self.next_action = None

        sumo_mode = "sumo-gui" if self.render_mode == 'human' else "sumo"
        delay = self.delay if self.render_mode == 'human' else 0
        Sumo_config = [
            sumo_mode,
            '-c', self.net_path,
            '--step-length', str(self.step_length),
            '--scale', str(self.traffic_scale),
            '--start',
            '--delay', str(delay),
            "--no-warnings"
        ]
        traci.start(Sumo_config)

        while self.time_elapsed < self.warmup_time:
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()

        # UGV States
        if self.random_sg_pair:
            init_node = random.choice(list(self.G.nodes))
            init_node_idx = self.node2idx[init_node]
            target_junction_id = random.choice(list(self.junction_list - {self.G.nodes[init_node]['junc_id']}))
            target_junc_idx = self.junc2idx[target_junction_id]
        else:
            init_node_idx = self.start_idx
            init_node = self.idx2node[init_node_idx]
            target_junction_id = self.target_junc_id
            target_junc_idx = self.junc2idx[target_junction_id]

        self.current_speed = self.get_current_speed()

        self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.current_speed)
        target_nodes = self.junc2nodes_map[target_junction_id]
        target_node = target_nodes[np.argmin([self.all_shortest_path_lengths[init_node][tn] for tn in target_nodes])]
        self.ugv.task_list = [self.node2idx[target_node]]

        # Metrics
        obs = self._get_observation()
        info = self.analyzer()

        return obs, info
 
    def _get_observation(self):
        ugv_node = self.idx2node[self.ugv.position]
        ugv_node_idx = self.ugv.position

        task_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        task_node_idx = self.ugv.task_list[0] if self.ugv.task_list else None
        
        junction_idx = -1
        steps2dest = [INF] * self.num_actions
        best_dir = 4
        sg_idx = -1
        time_slot = self.get_current_segment_index()

        if task_node:
            junction_idx = self.junc2idx[self.G.nodes[task_node]['junc_id']]

            nb_nodes_idx = sorted([self.node2idx[nb] for nb in list(self.G.neighbors(ugv_node))])
            nb_nodes = [self.idx2node[idx] for idx in nb_nodes_idx]
            steps2dest = []
            for nb_node in nb_nodes:
                steps2dest.append(self.all_shortest_path_lengths[nb_node][task_node])
            steps2dest += [INF] * (self.num_actions - len(steps2dest))
            
            best_dir = np.argmin(steps2dest)

            sg_idx = self.sg_pairs.index((ugv_node_idx, junction_idx))
            
            
        obs_dict = {
            'graph_node': ugv_node_idx,
            'target_node': junction_idx,
            'steps2dest': np.array(steps2dest, dtype=np.float32),
            'best_dir': best_dir,
            'sg_pair_idx': sg_idx,
            'time_slot': time_slot,
        }

        return obs_dict
    
    def analyzer(self):
        """Analyzes the current state and returns an info dictionary."""
        self.total_ev_distance = self.ugv.distance_traveled

        ugv_pos_node = self.idx2node[self.ugv.position] if self.ugv.position is not None else None
        task_pos_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        junction_id = self.G.nodes[task_pos_node]['junc_id'] if task_pos_node else None
        traffic_speed_cmd_ms = self.current_speed if hasattr(self, 'current_speed') else self.get_current_speed()
        vehicle_speed_actual_ms = self.get_actual_vehicle_speed()
        info = {
            'time_elapsed': self.time_elapsed,
            'current_timestep': self.current_timestep,
            
            'ugv_position': ugv_pos_node,
            'ugv_target': task_pos_node,
            'ugv_position_idx': self.ugv.position,
            'ugv_target_idx': self.ugv.task_list[0] if self.ugv.task_list else None,
            'ugv_target_junction': junction_id,
            'current_traffic_segment_idx': self.get_current_segment_index(),
            'traffic_speed_cmd_ms': traffic_speed_cmd_ms,
            'traffic_speed_cmd_kmh': self.ms_to_kmh(traffic_speed_cmd_ms),
            'vehicle_speed_actual_ms': vehicle_speed_actual_ms,
            'vehicle_speed_actual_kmh': self.ms_to_kmh(vehicle_speed_actual_ms),
            'speed_changed_this_step': self.speed_changed_this_step,
            'last_speed_change_time': self.last_speed_change_time,
            'last_speed_change_from': self.last_speed_change_from,
            'last_speed_change_to': self.last_speed_change_to,
            'last_speed_change_segment_from': self.last_speed_change_segment_from,
            'last_speed_change_segment_to': self.last_speed_change_segment_to,
        }
        return info
    
    def step(self, action):
        if self.terminated:
            return None, None, self.terminated, None, self.info # Return None if already terminated

        self.speed_changed_this_step = False
        
        if self.next_action_manual:
            action = self.next_action
            self.next_action_manual = False
        
        self.ugv_handler(action)   
        self.current_timestep += 1
        truncated = self.task_handler() # Handle task completion and updates
        
        reward = self._get_reward()
        self._check_termination_condition() # Check termination based on new state


        obs = self._get_observation() # Gets observation *after* action/reward
        self.info = self.analyzer() # Update info with current state

        return obs, reward, self.terminated, truncated, self.info # Return standard gym step tuple

    def ugv_handler(self, action):
        if self.ugv.status==2:
            return
        if self.ugv.veh_id and self.ugv.veh_id in traci.vehicle.getIDList():
            self.ugv.status = 0
            return
        self.ugv.status = 1
        act = action
        self.ugv.last_position = self.ugv.position    
        if act == self.num_actions:
            return
        move_flag, edge_id, edge_dir = self.ugv.move(act, self.time_elapsed, self.G, self.G_dir, self.node2idx, self.idx2node)
        self.last_move_time = self.time_elapsed
        self.current_speed = self.get_current_speed()
        
        if move_flag:
            # Only track move time if move was successful
            route_id = f"ugv_{self.ugv.agent_id}_{int(self.time_elapsed)}_route"
            self.ugv.veh_id = f"ugv_{self.ugv.agent_id}_{int(self.time_elapsed)}"
            # # add person and set a visible color based on UGV id
            # if edge_dir==0:
            #     traci.person.add(self.ugv.veh_id, edge_id, 0.0)
            #     traci.person.setColor(self.ugv.veh_id, (255,0,0,255))
            #     traci.person.appendWalkingStage(self.ugv.veh_id, [edge_id], arrivalPos=-1, speed=self.ugv.max_speed)
            # else:
            #     traci.person.add(self.ugv.veh_id, edge_id, -1)
            #     traci.person.setColor(self.ugv.veh_id, (255,0,0,255))
            #     traci.person.appendWalkingStage(self.ugv.veh_id, [edge_id], arrivalPos=0, speed=self.ugv.max_speed)
            # add a vehicle
            traci.route.add(route_id, [edge_id])
            traci.vehicle.add(self.ugv.veh_id, route_id, departSpeed=self.current_speed)
            traci.vehicle.setSpeed(self.ugv.veh_id, self.current_speed)
            traci.vehicle.setColor(self.ugv.veh_id, (255,0,0,255))
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()
            reached_flag = False
            while not reached_flag and self.time_elapsed < self.max_time:
                prev_segment_idx = self.get_current_segment_index()
                prev_speed = self.current_speed
                # if self.ugv.veh_id and self.ugv.veh_id not in traci.person.getIDList():
                if self.ugv.veh_id and self.ugv.veh_id not in traci.vehicle.getIDList():
                    self.ugv.status = 1
                    reached_flag = True
                traci.simulationStep()
                self.time_elapsed = traci.simulation.getTime()
                next_speed = self.get_current_speed()

                # veh_sim_speed = traci.vehicle.getSpeed(self.ugv.veh_id) if self.ugv.veh_id and self.ugv.veh_id in traci.vehicle.getIDList() else None
                # print(
                #     f"Sim time: {self.time_elapsed:.1f} s | "
                #     f"UGV speed: {self.format_speed(veh_sim_speed)} | "
                # )
                # next_segment_idx = self.get_current_segment_index()
                # if next_segment_idx != prev_segment_idx:
                #     self.speed_changed_this_step = True
                #     self.last_speed_change_time = self.time_elapsed
                #     self.last_speed_change_from = prev_speed
                #     self.last_speed_change_to = next_speed
                #     self.last_speed_change_segment_from = prev_segment_idx
                #     self.last_speed_change_segment_to = next_segment_idx
                #     print(
                #         "[TRAFFIC SPEED CHANGE] "
                #         f"t={self.time_elapsed:.1f} s | "
                #         f"segment {prev_segment_idx} -> {next_segment_idx} | "
                #         f"{self.format_speed(prev_speed)} -> {self.format_speed(next_speed)}"
                #     )

                if abs(next_speed - self.current_speed) > 1e-3:
                    self.current_speed = next_speed
                    if self.ugv.veh_id and self.ugv.veh_id in traci.vehicle.getIDList():
                        traci.vehicle.setSpeed(self.ugv.veh_id, self.current_speed)
            
            # travel_time = min(traci.edge.getTraveltime(edge_id), 25.0)
            # eta = self.time_elapsed + travel_time
            # while self.time_elapsed < eta and self.time_elapsed < self.max_time:
            #     traci.simulationStep()
            #     self.time_elapsed = traci.simulation.getTime()
            # self.ugv.status = 1
            
    def task_handler(self):
        self.truncated = False
        if self.ugv.status==1:
            if self.current_timestep > self.max_steps:
                self.truncated = True
                self.current_timestep = 0
                init_node = random.choice(list(self.G.nodes))
                init_node_idx = self.node2idx[init_node]
                self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.current_speed)
                target_junction_id = random.choice(list(self.junction_list - {self.G.nodes[init_node]['junc_id']}))
                target_nodes = self.junc2nodes_map[target_junction_id]
                target_node = target_nodes[np.argmin([self.all_shortest_path_lengths[init_node][tn] for tn in target_nodes])]
                self.ugv.task_list = [self.node2idx[target_node]]
            elif self.G.nodes[self.idx2node[self.ugv.position]]['junc_id'] == self.G.nodes[self.idx2node[self.ugv.task_list[0]]]['junc_id']:
                self.truncated = True
                self.current_timestep = 0
                init_node = random.choice(list(self.G.nodes))
                init_node_idx = self.node2idx[init_node]
                self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.current_speed)
                target_junction_id = random.choice(list(self.junction_list - {self.G.nodes[init_node]['junc_id']}))
                target_nodes = self.junc2nodes_map[target_junction_id]
                target_node = target_nodes[np.argmin([self.all_shortest_path_lengths[init_node][tn] for tn in target_nodes])]
                self.ugv.task_list = [self.node2idx[target_node]]
                self.ugv.task_completed_this_step = True  # Mark task completion for reward
        return self.truncated
    
    def _get_reward(self):
        """
        Calculate reward for each UGV based on:
        1. Completion reward: High positive reward when task is reached
        2. Invalid action reward: High negative reward for invalid actions
        3. Move reward: Low negative reward proportional to move time
        """
        total_reward = 0.0
        
        # Reward constants
        COMPLETION_REWARD = 50.0  # High positive reward for task completion
        INCOMPLETE_PENALTY = 0.0  # High negative reward for invalid actions
        INVALID_ACTION_PENALTY = -10.0  # High negative reward for invalid actions
        MOVE_TIME_PENALTY_FAC = -1  # Low negative reward per unit time (proportional to move time)
        
        if self.ugv.invalid_action_this_step:
            total_reward += INVALID_ACTION_PENALTY
            self.ugv.invalid_action_this_step = False  # Reset invalid action flag after rewarding

        else:
            move_time = self.time_elapsed - self.last_move_time
            total_reward += MOVE_TIME_PENALTY_FAC * move_time
            if self.truncated:
                self.truncated = False  # Reset truncated flag after rewarding
                if self.ugv.task_completed_this_step:
                    total_reward += COMPLETION_REWARD
                    self.ugv.task_completed_this_step = False  # Reset task completion flag after rewarding
                else:
                    total_reward += INCOMPLETE_PENALTY
        
        return total_reward


    def _check_termination_condition(self):
        if self.time_elapsed >= self.max_time:
            self.terminated = True
        else:
            self.terminated = False

    def render(self):
        """
        Render the environment.
        - 'human': pygame visualization showing the road network and UGV agents
        - 'print': print relevant environment information
        - 'quiet': no rendering
        """
        if self.render_mode in ['human', 'print']:
            segment_idx = self.get_current_segment_index()
            actual_speed_ms = self.get_actual_vehicle_speed()
            ugv_pos_node = self.idx2node[self.ugv.position] if self.ugv.position is not None else None
            task_pos_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None

            print("\n" + "=" * 80)
            print(f"Time elapsed: {self.time_elapsed:.2f} s")
            print(f"Timestep: {self.current_timestep}/{self.max_steps}")
            print(f"Current traffic segment: {segment_idx}")
            print(f"Commanded traffic speed: {self.format_speed(self.current_speed)}")
            print(f"Actual vehicle speed: {self.format_speed(actual_speed_ms)}")
            print(f"Speed changed this step: {'YES' if self.speed_changed_this_step else 'NO'}")
            if self.last_speed_change_time is not None:
                print(
                    "Last speed change: "
                    f"t={self.last_speed_change_time:.1f} s | "
                    f"segment {self.last_speed_change_segment_from} -> {self.last_speed_change_segment_to} | "
                    f"{self.format_speed(self.last_speed_change_from)} -> "
                    f"{self.format_speed(self.last_speed_change_to)}"
                )
            print(f"Vehicle node: {ugv_pos_node} (idx={self.ugv.position})")
            print(
                f"Target node: {task_pos_node} "
                f"(idx={self.ugv.task_list[0] if self.ugv.task_list else None})"
            )
            print(f"Distance traveled: {self.total_ev_distance:.1f} m")
            print(f"Terminated: {self.terminated} | Truncated: {self.truncated}")
            print("=" * 80)

        if self.render_mode == 'quiet':
            return

        # Human mode: pygame visualization
        if self.render_mode == 'human':
            # Initialize pygame display if not already done
            if self.screen is None:
                pygame.init()
                self.screen_width = 1000
                self.screen_height = 800
                self.clock = pygame.time.Clock()
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("SUMO LMD Environment")

            # Clear the screen
            self.screen.fill((255, 255, 255))  # White background
            
            # Get all node positions to calculate scaling factors
            pos_x = [self.G.nodes[n]['x'] for n in self.G.nodes]
            pos_y = [self.G.nodes[n]['y'] for n in self.G.nodes]
            min_x, max_x = min(pos_x), max(pos_x)
            min_y, max_y = min(pos_y), max(pos_y)
            
            # Add margins
            margin = 50
            scale_x = (self.screen_width - 2*margin) / max(1, max_x - min_x)
            scale_y = (self.screen_height - 2*margin) / max(1, max_y - min_y)
            scale = min(scale_x, scale_y)
            
            # Transform coordinate function
            def transform(x, y):
                return (int(margin + (x - min_x) * scale), int(margin + (max_y - y) * scale))
            
            # Draw edges (roads)
            for u, v in self.G.edges():
                start_x, start_y = transform(self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                end_x, end_y = transform(self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                # Draw roads in gray
                pygame.draw.line(self.screen, (150, 150, 150), (start_x, start_y), (end_x, end_y), 2)
            
            # Draw nodes (junctions)
            node_radius = 12
            sorted_nodes = sorted(self.G.nodes)
            
            for node in self.G.nodes:
                x, y = transform(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
                pygame.draw.circle(self.screen, (80, 80, 80), (x, y), node_radius)
                
                # Draw node index inside the circle
                if pygame.font:
                    font_node = pygame.font.Font(None, 16)
                    node_idx_text = font_node.render(str(self.node2idx[node]), True, (255, 255, 255))
                    text_rect = node_idx_text.get_rect(center=(x, y))
                    self.screen.blit(node_idx_text, text_rect)
            
            
            # Draw task locations and UGVs
            task_radius = 7
            ugv_radius = 9
            if self.ugv.position is not None:
                # Draw active task location (transparent red)
                if self.ugv.task_list:
                    active_task = self.idx2node[self.ugv.task_list[0]]
                    task_x, task_y = transform(self.G.nodes[active_task]['x'], self.G.nodes[active_task]['y'])
                    # Create a transparent surface for the task
                    task_surface = pygame.Surface((task_radius*2, task_radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(task_surface, (255, 0, 0, 100), (task_radius, task_radius), task_radius)
                    self.screen.blit(task_surface, (task_x - task_radius, task_y - task_radius))
                
                # Draw UGV
                ugv_x, ugv_y = transform(self.G.nodes[self.idx2node[self.ugv.position]]['x'], self.G.nodes[self.idx2node[self.ugv.position]]['y'])
                # Color based on status: Green=Moving, Yellow=Idle, Gray=Charging
                if self.ugv.status == 1:
                    color = (0, 200, 0)  # Green - moving
                elif self.ugv.status == 2:
                    color = (128, 128, 128)  # Gray - charging
                else:
                    color = (255, 200, 0)  # Yellow - idle
                
                # Create a transparent surface for the UGV
                ugv_surface = pygame.Surface((ugv_radius*2, ugv_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(ugv_surface, (*color, 150), (ugv_radius, ugv_radius), ugv_radius)
                self.screen.blit(ugv_surface, (ugv_x - ugv_radius, ugv_y - ugv_radius))

            # Draw info panel on the right side
            if pygame.font:
                font = pygame.font.Font(None, 28)
                y_offset = 10
                x_offset = self.screen_width - 350  # Position on the right
                segment_idx = self.get_current_segment_index()
                actual_speed_ms = self.get_actual_vehicle_speed()
                
                text = font.render(f"Time: {self.time_elapsed:.1f}s", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                text = font.render(f"Step: {self.current_timestep}/{self.max_steps}", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                text = font.render(f"Seg: {segment_idx}", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                text = font.render(f"Cmd: {self.format_speed(self.current_speed)}", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                text = font.render(f"Actual: {self.format_speed(actual_speed_ms)}", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                text = font.render(f"Distance: {self.total_ev_distance:.1f}m", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30

                font_small = pygame.font.Font(None, 20)
                text = font_small.render(
                    f"Speed changed this step: {'YES' if self.speed_changed_this_step else 'NO'}",
                    True,
                    (0, 0, 0)
                )
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 25

                if self.last_speed_change_time is not None:
                    last_change_text = (
                        f"Last change: seg {self.last_speed_change_segment_from}"
                        f"->{self.last_speed_change_segment_to} @ {self.last_speed_change_time:.1f}s"
                    )
                else:
                    last_change_text = "Last change: N/A"
                text = font_small.render(last_change_text, True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 22

                text = font_small.render("Upcoming segments:", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 20
                for i in range(1, 4):
                    future_idx = segment_idx + i
                    if future_idx >= len(self.traffic_speed_segments):
                        break
                    future_speed = self.traffic_speed_segments[future_idx]
                    text = font_small.render(
                        f"+{i} seg {future_idx}: {self.format_speed(future_speed)}",
                        True,
                        (0, 0, 0)
                    )
                    self.screen.blit(text, (x_offset, y_offset))
                    y_offset += 18
                
                # Legend on the bottom left
                legend_y = self.screen_height - 100
                font_small = pygame.font.Font(None, 22)
                pygame.draw.circle(self.screen, (0, 0, 255), (20, legend_y), 8)
                
                pygame.draw.circle(self.screen, (255, 0, 0), (20, legend_y + 25), 6)
                text = font_small.render("Task", True, (0, 0, 0))
                self.screen.blit(text, (35, legend_y + 17))
                
                pygame.draw.circle(self.screen, (0, 200, 0), (20, legend_y + 50), 8)
                text = font_small.render("UGV (Moving)", True, (0, 0, 0))
                self.screen.blit(text, (35, legend_y + 42))
            
            # Update display and handle events
            pygame.display.flip()
            
            # Wait for Return key press after each render
            print("\nPress Return to continue...")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        waiting = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_KP0:
                            self.next_action = 0  # Numpad 0 -> action 0
                            self.next_action_manual = True
                            waiting = False
                        elif event.key == pygame.K_KP1:
                            self.next_action = 1  # Numpad 1 -> action 1
                            self.next_action_manual = True
                            waiting = False
                        elif event.key == pygame.K_KP2:
                            self.next_action = 2  # Numpad 2 -> action 2
                            self.next_action_manual = True
                            waiting = False
                        elif event.key == pygame.K_KP3:
                            self.next_action = 3  # Numpad 3 -> action 3
                            self.next_action_manual = True
                            waiting = False
                self.clock.tick(30)
            
    def close(self):
        """Cleans up pygame resources."""
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        traci.close()

