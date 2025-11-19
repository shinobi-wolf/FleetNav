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

from envs.fleet.UGVSumo import UGVSumo
import numpy as np

class SumoLMDEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'quiet'], "render_fps": 30} # Adjusted FPS

    def __init__(self, config, render_mode=None): # Default render_mode is None
        super(SumoLMDEnv, self).__init__()

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.net_path = config["net"]["net_path"]
        self.warehouse_choice = config["net"]["warehouse"]

        # Task vars
        self.demand_per = config['task']['demand_per']

        # Agent vars
        self.num_ugvs = config["fleet"]["count"]
        self.max_ugv_range = config["fleet"]['range']
        self.ugv_speed = config["fleet"]['max_speed']
        self.num_actions = config['net']['max_degree']

        # Simulation vars
        self.max_time = config['simulation']['max_time']
        self.max_steps = config['simulation']['max_steps']
        self.render_mode = config['simulation']['render_mode']  # Render mode from config
        self.render_fps = config['simulation'].get('render_fps', 30)  # Default to 30 if not specified
        self.step_length = config['simulation']['step_length']
        self.train_mode = config['simulation']['train']
        self.traffic_scale = config['simulation'].get('traffic_scale', 1)
        self.ff = config['debug']['fast_forward']
        self.delay = config['debug'].get('delay', 0)

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([self.num_actions]*self.num_ugvs)
        
        ugv_status_space = spaces.MultiDiscrete([3]*self.num_ugvs) # 0: idle, 1: moving, 2: charging
        max_moves = 100
        steps2dest_space = spaces.MultiDiscrete(np.array([max_moves]*self.num_actions*self.num_ugvs))
        graph_node_space = spaces.MultiDiscrete([1000]*self.num_ugvs)
        nb_traffic_space = spaces.MultiDiscrete(np.array([3,3,3,3]*self.num_ugvs))
        self.observation_space = spaces.Dict({
            'ugv_status': ugv_status_space,
            'steps2dest': steps2dest_space,
            'graph_node': graph_node_space,
            'target_node': graph_node_space,
            'nb_traffic': nb_traffic_space,
        })
        

    def distribute_tasks(self):
        num_tasks_per_ugv = max(1, len(self.task_list) // self.num_ugvs)
        for ugv_id, ugv in enumerate(self.ugv_states):
            start_idx = ugv_id * num_tasks_per_ugv
            end_idx = start_idx + num_tasks_per_ugv
            ugv.task_list = self.task_list[start_idx:end_idx]

    def build_graph(self,):
        G = nx.DiGraph()

        # Get all edges (skip internal)
        edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(":")]

        for eid in edge_ids:
            from_node = traci.edge.getFromJunction(eid)
            to_node = traci.edge.getToJunction(eid)

            for lane_idx in range(traci.edge.getLaneNumber(eid)):
                lid = f"{eid}_{lane_idx}"
                if 'pedestrian' in traci.lane.getAllowed(lid):
                    length = traci.lane.getLength(lid)
                    break

            # Add nodes (junctions)
            fx, fy = traci.junction.getPosition(from_node)
            tx, ty = traci.junction.getPosition(to_node)
            G.add_node(from_node, x=fx, y=fy)
            G.add_node(to_node, x=tx, y=ty)

            # Add directed edge with attributes
            G.add_edge(
                from_node,
                to_node,
                id=eid,
                length=length,
            )

        all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        return G, all_shortest_path_lengths

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call parent reset for seeding RNG
        if seed: # Seeding is handled by super().reset()
            random.seed(seed)
            np.random.seed(seed) # Also seed numpy for consistent sampling if used

        # Simulation State
        self.current_timestep = 0
        self.terminated = False
        self.time_elapsed = 0.0

        # Initialize pygame display if not already done
        self.screen = None
        self.clock = None
        self.render_mode = getattr(self, 'render_mode', None)

        sumo_mode = 'sumo-gui' if self.render_mode == 'human' else 'sumo'
        Sumo_config = [
            sumo_mode,
            '-c', self.net_path,
            '--step-length', str(self.step_length),
            '--delay', str(self.delay),
            '--scale', str(self.traffic_scale),
            '--start',
            "--no-warnings"
        ]
        traci.start(Sumo_config)

        # Maze State
        self.G, self.all_shortest_path_lengths = self.build_graph() # Rebuild graph with new maze
        self.warehouse_node = random.choice(list(self.G.nodes))

        # UGV States
        self.ugv_states = [UGVSumo(ugv_id=i, base_position=self.warehouse_node, max_range=self.max_ugv_range, max_speed=self.ugv_speed,
                             G=self.G) for i in range(self.num_ugvs)]
        self.num_tasks = (self.demand_per * len(self.G.nodes))//100
        self.task_list = random.choices(list(self.G.nodes), k=self.num_tasks) # Randomly select tasks from graph nodes
        self.distribute_tasks() # Distribute tasks to UGVs

        # Metrics
        self.num_tasks_completed = 0
        self.task_completion_times = []
        self.total_ev_distance = 0.0

        obs = self._get_observation()
        info = self.analyzer()

        return obs, info

    def _get_nb_traffic(self, position, nb_nodes):
        # mean speed in last step
        nb_traffic = []
        for nb_node in nb_nodes:
            edge = self.G.edges[(position, nb_node)]['id']
            v_mean = traci.edge.getLastStepMeanSpeed(edge)

            v_limit = traci.lane.getMaxSpeed(edge+'_0')

            # congestion percent (0% = free flow, 100% = stopped)
            cong = max(0.0, min(1.0 - v_mean / v_limit, 1.0)) * 100.0

            # 3-bin discretization
            if cong < 33:
                nb_traffic.append(0)
            elif cong < 66:
                nb_traffic.append(1)     # medium congestion
            else:
                nb_traffic.append(2)     # high congestion
        while len(nb_traffic) < 4:
            nb_traffic.append(2)
        return nb_traffic
    
    def _get_observation(self):
        steps2dest = []
        nb_traffic = []
        ugv_status = []
        graph_node = []
        target_node = []

        for ugv_id, ugv in enumerate(self.ugv_states):
            ugv_pos = ugv.position
            # Steps to destination in each direction
            steps2dest_ = []
            nb_nodes = sorted(self.G.neighbors(ugv_pos))[:self.num_actions]  # Get up to num_actions neighbors
            if ugv.task_list:  # Check if task list is not empty
                task_pos = ugv.task_list[0]
                for nb_node in nb_nodes:
                    steps2dest_.append(self.all_shortest_path_lengths[nb_node][task_pos])
                for _ in range(self.num_actions - len(nb_nodes)):
                    steps2dest_.append(100)
            else:
                # If no tasks, use zeros for steps
                steps2dest_ = [100] * self.num_actions
            steps2dest.append(steps2dest_)

            # Neighbor Traffic
            nb_traffic.append(self._get_nb_traffic(ugv_pos, nb_nodes))

            ugv_status.append(ugv.status)
            graph_node.append(sorted(list(self.G.nodes)).index(ugv.position))
            if ugv.task_list:
                target_node.append(sorted(list(self.G.nodes)).index(ugv.task_list[0]))
            else:
                target_node.append(-1)


        steps2dest_flat = np.array(steps2dest, dtype=np.float32).flatten()
        nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()
        ugv_status_flat = np.array(ugv_status, dtype=np.int32).flatten()
        graph_node_flat = np.array(graph_node, dtype=np.int32).flatten()
        target_node_flat = np.array(target_node, dtype=np.int32).flatten()
        obs_dict = {
            'steps2dest': steps2dest_flat,
            'nb_traffic': nb_traffic_flat,
            'ugv_status': ugv_status_flat,
            'graph_node': graph_node_flat,
            'target_node': target_node_flat,
        }

        return obs_dict
    
    def analyzer(self):
        """Analyzes the current state and returns an info dictionary."""
        self.total_ev_distance = sum(ugv.distance_traveled for ugv in self.ugv_states)
        task_completion_time_diffs = [self.task_completion_times[i] - self.task_completion_times[i-1] for i in range(1, len(self.task_completion_times))]
        avg_task_completion_time = np.mean(task_completion_time_diffs) if task_completion_time_diffs else 0.0

        active_tasks = [ugv.task_list[0] if ugv.task_list else None for ugv in self.ugv_states]
        ugv_status = [ugv.status for ugv in self.ugv_states]
        ugv_pos = [ugv.position for ugv in self.ugv_states]
        info = {
            'time_elapsed': self.time_elapsed,
            'current_timestep': self.current_timestep,
            
            'num_tasks_completed': self.num_tasks_completed,
            'total_ev_distance': self.total_ev_distance,
            'avg_task_completion_time': avg_task_completion_time,

            'active_tasks': active_tasks,
            'ugv_status': ugv_status,
            'ugv_pos': ugv_pos,
        }
        return info
    
    def step(self, action):
        if self.terminated:
            return None, None, self.terminated, None, self.info # Return None if already terminated
        
        self.ugv_handler(action)
        self.current_timestep += 1
        traci.simulationStep()
        self.time_elapsed = traci.simulation.getTime()
        reached_flag = False
        while not reached_flag and self.time_elapsed < self.max_time:
            for ugv in self.ugv_states:
                if ugv.veh_id and ugv.veh_id not in traci.person.getIDList():
                    ugv.status = 1
                    reached_flag = True
            traci.simulationStep()
            self.time_elapsed = traci.simulation.getTime()
        truncated = self.task_handler() # Handle task completion and updates
        
        reward = self._get_reward()

        self._check_termination_condition() # Check termination based on new state

        obs = self._get_observation() # Gets observation *after* action/reward
        self.info = self.analyzer() # Update info with current state

        return obs, reward, self.terminated, truncated, self.info # Return standard gym step tuple

    def ugv_handler(self, action):
        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status==2:
                continue
            if ugv.veh_id and ugv.veh_id in traci.person.getIDList():
                ugv.status = 0
                continue
            ugv.status = 1
            act = action[ugv_id]
            ugv.last_position = ugv.position    
            if act == self.num_actions:
                continue
            move_flag, edge_id = ugv.move(act, self.time_elapsed, self.G)
            
            if move_flag:
                # Only track move time if move was successful
                route_id = f"ugv_{ugv_id}_{self.current_timestep}_route"
                ugv.veh_id = f"ugv_{ugv_id}_{self.current_timestep}"
                # add person and set a visible color based on UGV id
                traci.person.add(ugv.veh_id, edge_id, 0.0)
                traci.person.setColor(ugv.veh_id, (255,0,0,255))
                traci.person.appendWalkingStage(ugv.veh_id, [edge_id], arrivalPos=-1, speed=3.0)
            
            if ugv.position == self.warehouse_node:
                ugv.recharge()

    def task_handler(self):
        truncated = False
        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status==1:
                if ugv.task_list and ugv.position == ugv.task_list[0]:
                    truncated = True
                    ugv.task_list.pop(0)
                    ugv.task_completed_this_step = True  # Mark task completion for reward
                    self.num_tasks_completed += 1
                    self.task_completion_times.append(self.time_elapsed)
                    if len(ugv.task_list) == 0:
                        ugv.status = 2

        return truncated # Return if any UGV completed a task
    
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
        INVALID_ACTION_PENALTY = -15.0  # High negative reward for invalid actions
        MOVE_TIME_PENALTY_SCALE = -0.1  # Low negative reward per unit time (proportional to move time)
        
        for ugv in self.ugv_states:
            ugv_reward = 0.0
            
            # 1. Completion reward: High positive reward if task was completed
            if ugv.task_completed_this_step:
                ugv_reward += COMPLETION_REWARD
            
            # 2. Invalid action reward: High negative reward if invalid action was performed
            if ugv.invalid_action_this_step:
                ugv_reward += INVALID_ACTION_PENALTY
            
            # 3. Move reward: Low negative reward proportional to move time
            # Only apply if a move was actually made (last_move_time > 0)
            move_time = self.time_elapsed - ugv.last_move_time
            if move_time > 0:
                ugv_reward += MOVE_TIME_PENALTY_SCALE * move_time

            # if ugv.status!=2 and not ugv.task_completed_this_step:
            #     curr_dist = self.all_shortest_path_lengths[ugv.position][ugv.task_list[0]]
            #     prev_dist = self.all_shortest_path_lengths[ugv.last_position][ugv.task_list[0]]
            #     ugv_reward += 5.0*(prev_dist-curr_dist)
            
            total_reward += ugv_reward
        
        return total_reward


    def _check_termination_condition(self):
        done_status_flag = True
        for ugv in self.ugv_states:
            if ugv.status!=2:
                done_status_flag = False
                break
        if self.time_elapsed >= self.max_time or self.current_timestep >= self.max_steps or done_status_flag:
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
        if not hasattr(self, 'render_mode') or self.render_mode == 'quiet':
            return
        
        # Print mode: display environment information
        if True:
            print("\n" + "="*80)
            print(f"RENDER - Time: {self.time_elapsed:.2f}s | Timestep: {self.current_timestep}/{self.max_steps}")
            print(f"Tasks Completed: {self.num_tasks_completed}/{self.num_tasks}")
            print(f"Total Distance Traveled: {self.total_ev_distance:.2f}m")
            print(f"Terminated: {self.terminated}")
            print("-"*80)
            print("DETAILED UGV STATUS:")
            for i, ugv in enumerate(self.ugv_states):
                print(f"  UGV {i}: {ugv.status} | Position: {ugv.position} | Range: {ugv.current_range:.1f}/{self.max_ugv_range}m")
                print(f"         Distance Traveled: {ugv.distance_traveled:.2f}m")
                if ugv.task_list:
                    print(f"         Active Task: {ugv.task_list[0]} | Tasks Remaining: {len(ugv.task_list)}")
                    print(f"         Full Task List: {ugv.task_list}")
                else:
                    print(f"         No tasks remaining")
                # Check if vehicle exists in SUMO
                veh_id = f"ugv_{i}"
                try:
                    if veh_id in traci.vehicle.getIDList():
                        sumo_pos = traci.vehicle.getPosition(veh_id)
                        sumo_speed = traci.vehicle.getSpeed(veh_id)
                        sumo_route = traci.vehicle.getRoute(veh_id)
                        print(f"         SUMO Position: {sumo_pos}, Speed: {sumo_speed:.2f}m/s")
                        print(f"         SUMO Route: {sumo_route}")
                    else:
                        print(f"         Vehicle not in SUMO simulation")
                except:
                    print(f"         SUMO data unavailable")
            print("-"*80)
            print(f"ENVIRONMENT STATE:")
            print(f"  Warehouse Node: {self.warehouse_node}")
            print(f"  Graph Nodes: {len(self.G.nodes)}, Edges: {len(self.G.edges)}")
            print(f"  Max Time: {self.max_time}s, Max Steps: {self.max_steps}")
            print(f"  Step Length: {self.step_length}s")
            print(f"  Task Completion Times: {self.task_completion_times}")
            print("="*80)
            # return
        
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
            node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}
            
            for node in self.G.nodes:
                x, y = transform(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
                pygame.draw.circle(self.screen, (80, 80, 80), (x, y), node_radius)
                
                # Draw node index inside the circle
                if pygame.font:
                    font_node = pygame.font.Font(None, 16)
                    node_idx_text = font_node.render(str(node_to_index[node]), True, (255, 255, 255))
                    text_rect = node_idx_text.get_rect(center=(x, y))
                    self.screen.blit(node_idx_text, text_rect)
            
            # Draw warehouse (base station) - blue
            wx, wy = transform(self.G.nodes[self.warehouse_node]['x'], self.G.nodes[self.warehouse_node]['y'])
            pygame.draw.circle(self.screen, (0, 0, 255), (wx, wy), 10)
            
            # Draw task locations and UGVs
            task_radius = 7
            ugv_radius = 9
            for i, ugv in enumerate(self.ugv_states):
                if ugv.position is not None:
                    # Draw active task location (transparent red)
                    if ugv.task_list:
                        active_task = ugv.task_list[0]
                        task_x, task_y = transform(self.G.nodes[active_task]['x'], self.G.nodes[active_task]['y'])
                        # Create a transparent surface for the task
                        task_surface = pygame.Surface((task_radius*2, task_radius*2), pygame.SRCALPHA)
                        pygame.draw.circle(task_surface, (255, 0, 0, 100), (task_radius, task_radius), task_radius)
                        self.screen.blit(task_surface, (task_x - task_radius, task_y - task_radius))
                    
                    # Draw UGV
                    ugv_x, ugv_y = transform(self.G.nodes[ugv.position]['x'], self.G.nodes[ugv.position]['y'])
                    # Color based on status: Green=Moving, Yellow=Idle, Gray=Charging
                    if ugv.status == 1:
                        color = (0, 200, 0)  # Green - moving
                    elif ugv.status == 2:
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
                text = font.render(f"Tasks: {self.num_tasks_completed}/{self.num_tasks}", True, (0, 0, 0))
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
                
                for i, ugv in enumerate(self.ugv_states):
                    status_text = "Moving" if ugv.status == 1 else ("Charging" if ugv.status == 2 else "Idle")
                    range_text = f"UGV {i}: {status_text} | Range: {ugv.current_range:.1f}/{self.max_ugv_range}m"
                    text = font_small.render(range_text, True, (0, 0, 0))
                    self.screen.blit(text, (x_offset + 5, y_offset))
                    y_offset += 22
                
                # Legend on the bottom left
                legend_y = self.screen_height - 100
                font_small = pygame.font.Font(None, 22)
                pygame.draw.circle(self.screen, (0, 0, 255), (20, legend_y), 8)
                text = font_small.render("Warehouse", True, (0, 0, 0))
                self.screen.blit(text, (35, legend_y - 8))
                
                pygame.draw.circle(self.screen, (255, 0, 0), (20, legend_y + 25), 6)
                text = font_small.render("Task", True, (0, 0, 0))
                self.screen.blit(text, (35, legend_y + 17))
                
                pygame.draw.circle(self.screen, (0, 200, 0), (20, legend_y + 50), 8)
                text = font_small.render("UGV (Moving)", True, (0, 0, 0))
                self.screen.blit(text, (35, legend_y + 42))
            
            # Update display and handle events
            pygame.display.flip()
            
            # Control frame rate
            if self.num_tasks_completed < self.ff:
                self.clock.tick(self.metadata["render_fps"])
            else:
                # Pause and wait for user input after fast-forward threshold
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.close()
                            waiting = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                            waiting = False
                    self.clock.tick(self.metadata["render_fps"])
            
    def close(self):
        """Cleans up pygame resources."""
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        traci.close()

