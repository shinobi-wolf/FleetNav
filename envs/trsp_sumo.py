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

from envs.fleet.UGVSumo import UGVSumo
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
        self.build_graph()
        print(f"Graph built with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")
        self.num_sg_pairs = len(self.sg_pairs)

        # Agent vars
        self.ugv_speed = config["fleet"]['max_speed']
        self.num_actions = config['net']['max_degree']

        # Sumo vars
        self.step_length = config['sumo']['step_length']
        self.traffic_scale = config['sumo'].get('traffic_scale', 1)
        self.delay = config['sumo'].get('delay', 0)

        # Simulation vars
        self.max_time = config['simulation']['max_time']
        self.max_steps = config['simulation']['max_steps']
        self.render_mode = config['simulation'].get('render_mode')
        self.warmup_time = config['simulation']['warm_up_time']
        self.random_sg_pair = config['simulation']['random_sg_pair']

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
        self.observation_space = spaces.Dict({
            'steps2dest': steps2dest_space,
            'graph_node': graph_node_space,
            'target_node': graph_node_space,
            'best_dir': best_dir_space,
            'sg_pair_idx': sg_pair_idx_space,
        })
        
    def build_graph(self,):
        net_path = self.net_path.replace('.sumocfg', '.net.xml')
        net = sumolib.net.readNet(net_path, withPedestrianConnections=True)
        junctions = net.getNodes()
        edge_eps = {}
        junc2nodes_map = defaultdict(list)
        
        for junction in junctions:
            connections = junction.getConnections()
            for connection in connections:
                # 1. Inspect the source of the connection
                ep = None
                edge = None
                dir = None
                if connection._from.getFunction() == 'walkingarea':
                    ep = connection._from
                    edge = connection._to
                    dir = 0
                elif connection._to.getFunction() == 'walkingarea':
                    ep = connection._to
                    edge = connection._from
                    dir = 1
                if ep and edge:
                    edge_id = edge.getID()
                    eps = [None, None]
                    eps[dir] = ep
                    if edge_id not in edge_eps:
                        edge_eps[edge_id] = {
                            "edge": edge,
                            "eps": eps
                        }
                    else:
                        edge_eps[edge_id]["eps"][dir] = ep
                        if len([ep for ep in edge_eps[edge_id]["eps"] if ep is not None]) > 2:
                            print("More than 2 entry points for edge ", edge_id)
                            raise ValueError("More than 2 entry points for edge ", edge_id)

        G = nx.DiGraph()
        for edge_id, edge_data in edge_eps.items():
            entry_points = edge_data["eps"]
            for ep in entry_points:
                shape = ep.getShape()
                cx, cy = get_centroid(shape)
                G.add_node(ep._id, junc_id=ep._from._id,junc_coord=ep._from._coord, x=cx, y=cy)
                junc2nodes_map[ep._from._id].append(ep._id)
            edge_obj = edge_data["edge"]
            edge_length = None
            edge_type = edge_obj.getFunction()
            if edge_type == 'crossing':
                edge_length = 5.0
                G.add_edge(entry_points[0]._id, entry_points[1]._id, id=edge_obj._id, length=edge_length, type=edge_type)
                # G.add_edge(entry_points[1]._id, entry_points[0]._id, edge=edge_obj._id, length=edge_length, type=edge_type)
            else:
                edge_length = edge_obj.getLength()
                edge_type = 'sidewalk'
                G.add_edge(entry_points[0]._id, entry_points[1]._id, id=edge_obj._id, length=edge_length, type=edge_type)
        
        G_und = G.to_undirected()

        all_shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_und, weight='length'))
        self.G, self.G_dir, self.all_shortest_path_lengths, self.junc2nodes_map = G_und, G, all_shortest_path_lengths, junc2nodes_map

        self.num_junctions = len(self.junc2nodes_map)
        self.junction_list = self.junc2nodes_map.keys()
        self.node2idx = {node: idx for idx, node in enumerate(sorted(self.G.nodes))}
        self.idx2node = {idx: node for node, idx in self.node2idx.items()}
        self.sg_pairs = []
        for node_idx in range(len(self.idx2node)):
            for junc_id in self.junction_list:
                node = self.idx2node[node_idx]
                if self.G.nodes[node]['junc_id'] != junc_id:
                    self.sg_pairs.append((node_idx, junc_id))

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
            target_junction_id = random.choice(list(self.junction_list - {self.G.nodes[init_node]['junc_id']}))
            init_node_idx = self.node2idx[init_node]
        else:
            init_node_idx = self.start_idx
            init_node = self.idx2node[init_node_idx]
            target_junction_id = self.target_junc_id
        self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.ugv_speed)
        target_nodes = self.junc2nodes_map[target_junction_id]
        target_node = target_nodes[np.argmin([self.all_shortest_path_lengths[init_node][tn] for tn in target_nodes])]
        self.ugv.task_list = [self.node2idx[target_node]]

        # Metrics
        obs = self._get_observation()
        info = self.analyzer()

        return obs, info
 
    def _get_observation(self):
        steps2dest = []
        ugv_node = self.idx2node[self.ugv.position]
        ugv_node_idx = self.ugv.position
        task_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        task_node_idx = self.ugv.task_list[0] if self.ugv.task_list else None
        junction_idx = None
        if task_node:
            nb_nodes = list(self.G.neighbors(ugv_node))  # Get up to num_actions neighbors
            nb_node_idxs = sorted([self.node2idx[nb] for nb in nb_nodes])[:self.num_actions]
            for nb_node_idx in nb_node_idxs:
                steps2dest.append(self.all_shortest_path_lengths[self.idx2node[nb_node_idx]][task_node])
            steps2dest += [INF] * (self.num_actions - len(steps2dest))
            junction_idx = self.G.nodes[task_node]['junc_id']
            sg_idx = self.sg_pairs.index((ugv_node_idx, junction_idx))
            best_dir = np.argmin(steps2dest)
        else:
            junction_idx = -1
            steps2dest = [INF] * self.num_actions
            sg_idx = -1
            best_dir = 4
        obs_dict = {
            'graph_node': ugv_node_idx,
            'target_node': int(junction_idx),
            'best_dir': best_dir,
            'steps2dest': np.array(steps2dest, dtype=np.float32),
            'sg_pair_idx': sg_idx
        }

        return obs_dict
    
    def analyzer(self):
        """Analyzes the current state and returns an info dictionary."""
        self.total_ev_distance = self.ugv.distance_traveled

        ugv_pos_node = self.idx2node[self.ugv.position] if self.ugv.position is not None else None
        task_pos_node = self.idx2node[self.ugv.task_list[0]] if self.ugv.task_list else None
        junction_idx = int(self.G.nodes[task_pos_node]['junc_id']) if task_pos_node else None
        info = {
            'time_elapsed': self.time_elapsed,
            'current_timestep': self.current_timestep,
            
            'ugv_position': ugv_pos_node,
            'ugv_target': task_pos_node,
            'ugv_position_idx': self.ugv.position,
            'ugv_target_idx': self.ugv.task_list[0] if self.ugv.task_list else None,
            'ugv_target_junction': junction_idx,
        }
        return info
    
    def step(self, action):
        if self.terminated:
            return None, None, self.terminated, None, self.info # Return None if already terminated
        
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
        if self.ugv.veh_id and self.ugv.veh_id in traci.person.getIDList():
            self.ugv.status = 0
            return
        self.ugv.status = 1
        act = action
        self.ugv.last_position = self.ugv.position    
        if act == self.num_actions:
            return
        move_flag, edge_id, edge_dir = self.ugv.move(act, self.time_elapsed, self.G, self.G_dir, self.node2idx, self.idx2node)
        self.last_move_time = self.time_elapsed
        
        if move_flag:
            # Only track move time if move was successful
            route_id = f"ugv_{self.ugv.agent_id}_{self.current_timestep}_route"
            self.ugv.veh_id = f"ugv_{self.ugv.agent_id}_{self.current_timestep}"
            # add person and set a visible color based on UGV id
            if edge_dir==0:
                traci.person.add(self.ugv.veh_id, edge_id, 0.0)
                traci.person.setColor(self.ugv.veh_id, (255,0,0,255))
                traci.person.appendWalkingStage(self.ugv.veh_id, [edge_id], arrivalPos=-1, speed=self.ugv.max_speed)
            else:
                traci.person.add(self.ugv.veh_id, edge_id, -1)
                traci.person.setColor(self.ugv.veh_id, (255,0,0,255))
                traci.person.appendWalkingStage(self.ugv.veh_id, [edge_id], arrivalPos=0, speed=self.ugv.max_speed)
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()
            reached_flag = False
            while not reached_flag and self.time_elapsed < self.max_time:
                if self.ugv.veh_id and self.ugv.veh_id not in traci.person.getIDList():
                    self.ugv.status = 1
                    reached_flag = True
                traci.simulationStep()
                self.time_elapsed = traci.simulation.getTime()
            
    def task_handler(self):
        self.truncated = False
        if self.ugv.status==1:
            if self.current_timestep > self.max_steps:
                self.truncated = True
                self.current_timestep = 0
                init_node = random.choice(list(self.G.nodes))
                init_node_idx = self.node2idx[init_node]
                self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.ugv_speed)
                target_junction_id = random.choice(list(self.junction_list - {self.G.nodes[init_node]['junc_id']}))
                target_nodes = self.junc2nodes_map[target_junction_id]
                target_node = target_nodes[np.argmin([self.all_shortest_path_lengths[init_node][tn] for tn in target_nodes])]
                self.ugv.task_list = [self.node2idx[target_node]]
            elif self.G.nodes[self.idx2node[self.ugv.position]]['junc_id'] == self.G.nodes[self.idx2node[self.ugv.task_list[0]]]['junc_id']:
                self.truncated = True
                self.current_timestep = 0
                init_node = random.choice(list(self.G.nodes))
                init_node_idx = self.node2idx[init_node]
                self.ugv = UGVSumo(ugv_id=0, base_position=init_node_idx, max_speed=self.ugv_speed)
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
        MOVE_TIME_PENALTY_FAC = -0.1  # Low negative reward per unit time (proportional to move time)
        
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
        print("\n" + "="*80)
        print(f"RENDER - Time: {self.time_elapsed:.2f}s | Timestep: {self.current_timestep}/{self.max_steps}")
        print(f"Terminated: {self.terminated}")
        print("-"*80)
        print("DETAILED UGV STATUS:")
        print(f"  UGV : Position: {self.ugv.position}m")
        if self.ugv.task_list:
            print(f"         Active Task: {self.ugv.task_list[0]} | Tasks Remaining: {len(self.ugv.task_list)}")
            print(f"         Active Task Junction: {self.G.nodes[self.idx2node[self.ugv.task_list[0]]]['junc_id']}")
            print(f"         Full Task List: {self.ugv.task_list}")
        else:
            print(f"         No tasks remaining")
        # Check if vehicle exists in SUMO
        print("-"*80)
        print(f"ENVIRONMENT STATE:")
        print(f"  Graph Nodes: {len(self.G.nodes)}, Edges: {len(self.G.edges)}")
        print(f"  Max Time: {self.max_time}s, Max Steps: {self.max_steps}")
        print(f"  Step Length: {self.step_length}s")
        print("="*80)
        # return
        
        # Human mode: pygame visualization
        if True:
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
                
                # Time and tasks
                text = font.render(f"Time: {self.time_elapsed:.1f}s | Step: {self.current_timestep}", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                text = font.render(f"Distance: {self.total_ev_distance:.1f}m", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 30
                
                # Draw agent status
                font_small = pygame.font.Font(None, 20)
                text = font_small.render("AGENT STATUS:", True, (0, 0, 0))
                self.screen.blit(text, (x_offset, y_offset))
                y_offset += 25
                
                # for i, ugv in enumerate(self.ugv_states):
                #     status_text = "Moving" if ugv.status == 1 else ("Charging" if ugv.status == 2 else "Idle")
                #     range_text = f"UGV {i}: {status_text} | Range: {ugv.current_range:.1f}/{self.max_ugv_range}m"
                #     text = font_small.render(range_text, True, (0, 0, 0))
                #     self.screen.blit(text, (x_offset + 5, y_offset))
                #     y_offset += 22
                
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

