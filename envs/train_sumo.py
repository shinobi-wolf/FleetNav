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

class SumoTrainEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'quiet'], "render_fps": 30} # Adjusted FPS

    def __init__(self, config, render_mode=None): # Default render_mode is None
        super(SumoTrainEnv, self).__init__()

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.net_path = config["net"]["net_path"]

        # Agent vars
        self.ugv_speed = config["fleet"]['max_speed']
        self.num_actions = config['net']['max_degree']

        # Simulation vars
        self.max_time = config['simulation']['max_time']
        self.max_steps = config['simulation']['max_steps']
        self.step_length = config['simulation']['step_length']
        self.traffic_scale = config['simulation'].get('traffic_scale', 1)

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.Discrete(self.num_actions)
        
        max_moves = 100
        # steps2dest_space = spaces.MultiDiscrete(np.array([max_moves]*self.num_actions))
        graph_node_space = spaces.Discrete(1000)
        # nb_traffic_space = spaces.MultiDiscrete(np.array([3,3,3,3]))
        best_dir_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict({
            # 'steps2dest': steps2dest_space,
            'graph_node': graph_node_space,
            'target_node': graph_node_space,
            'best_dir': best_dir_space,
        })
        
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
        self.truncated = False
        self.time_elapsed = 0.0

        # Initialize pygame display if not already done
        self.screen = None
        self.clock = None
        self.render_mode = getattr(self, 'render_mode', None)
        self.next_action_manual = False
        self.next_action = None

        Sumo_config = [
            'sumo',
            '-c', self.net_path,
            '--step-length', str(self.step_length),
            '--scale', str(self.traffic_scale),
            '--start',
            "--no-warnings"
        ]
        traci.start(Sumo_config)

        # Maze State
        self.G, self.all_shortest_path_lengths = self.build_graph() # Rebuild graph with new maze

        # UGV States
        init_node, target_node = random.sample(list(self.G.nodes), 2)
        self.ugv = UGVSumo(ugv_id=0, base_position=init_node, max_speed=self.ugv_speed, G=self.G)
        self.ugv.task_list = [target_node]

        # Metrics
        obs = self._get_observation()
        info = self.analyzer()

        return obs, info
 
    def _get_observation(self):
        ugv_pos = sorted(list(self.G.nodes)).index(self.ugv.position)
        steps2dest = []
        if self.ugv.task_list:
            target_pos = sorted(list(self.G.nodes)).index(self.ugv.task_list[0])
            nb_nodes = sorted(self.G.neighbors(self.ugv.position))[:self.num_actions]  # Get up to num_actions neighbors
            for nb_node in nb_nodes:
                steps2dest.append(self.all_shortest_path_lengths[nb_node][self.ugv.task_list[0]])
        else:
            target_pos = ugv_pos
            steps2dest = []
        best_dir = np.argmin(steps2dest) if steps2dest else 4
        obs_dict = {
            'graph_node': ugv_pos,
            'target_node': target_pos,
            'best_dir': best_dir
        }

        return obs_dict
    
    def analyzer(self):
        """Analyzes the current state and returns an info dictionary."""
        self.total_ev_distance = self.ugv.distance_traveled

        ugv_pos = self.ugv.position
        task_pos = self.ugv.task_list[0] if self.ugv.task_list else None
        info = {
            'time_elapsed': self.time_elapsed,
            'current_timestep': self.current_timestep,
            
            'ugv_position': ugv_pos,
            'ugv_target': task_pos,
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
        traci.simulationStep()
        self.time_elapsed = traci.simulation.getTime()
        reached_flag = False
        while not reached_flag and self.time_elapsed < self.max_time:
            if self.ugv.veh_id and self.ugv.veh_id not in traci.person.getIDList():
                self.ugv.status = 1
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
        move_flag, edge_id = self.ugv.move(act, self.time_elapsed, self.G)
        
        if move_flag:
            # Only track move time if move was successful
            route_id = f"ugv_{self.ugv.agent_id}_{self.current_timestep}_route"
            self.ugv.veh_id = f"ugv_{self.ugv.agent_id}_{self.current_timestep}"
            # add person and set a visible color based on UGV id
            traci.person.add(self.ugv.veh_id, edge_id, 0.0)
            traci.person.setColor(self.ugv.veh_id, (255,0,0,255))
            traci.person.appendWalkingStage(self.ugv.veh_id, [edge_id], arrivalPos=-1, speed=self.ugv.max_speed)
            
    def task_handler(self):
        self.truncated = False
        if self.ugv.status==1:
            if self.current_timestep > self.max_steps:
                self.truncated = True
                self.current_timestep = 0
                init_node, target_node = random.sample(list(self.G.nodes), 2)
                self.ugv = UGVSumo(ugv_id=0, base_position=init_node, max_speed=self.ugv_speed, G=self.G)
                self.ugv.task_list = [target_node]
            elif self.ugv.position == self.ugv.task_list[0]:
                self.truncated = True
                self.current_timestep = 0
                init_node, target_node = random.sample(list(self.G.nodes), 2)
                self.ugv = UGVSumo(ugv_id=0, base_position=init_node, max_speed=self.ugv_speed, G=self.G)
                self.ugv.task_list = [target_node]
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
        INVALID_ACTION_PENALTY = -15.0  # High negative reward for invalid actions
        MOVE_TIME_PENALTY_SCALE = -0.1  # Low negative reward per unit time (proportional to move time)
        
        # 1. Completion reward: High positive reward if task was completed
        if self.truncated:
            self.truncated = False  # Reset truncated flag after rewarding
            if self.ugv.task_completed_this_step:
                total_reward += COMPLETION_REWARD
                self.ugv.task_completed_this_step = False  # Reset task completion flag after rewarding
            else:
                total_reward -= 10  # Partial penalty if terminated early without task completion
        else:
            move_time = self.time_elapsed - self.ugv.last_move_time
            total_reward += MOVE_TIME_PENALTY_SCALE * move_time
        
        # 2. Invalid action reward: High negative reward if invalid action was performed
        if self.ugv.invalid_action_this_step:
            total_reward += INVALID_ACTION_PENALTY
        
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
            
            
            # Draw task locations and UGVs
            task_radius = 7
            ugv_radius = 9
            if self.ugv.position is not None:
                # Draw active task location (transparent red)
                if self.ugv.task_list:
                    active_task = self.ugv.task_list[0]
                    task_x, task_y = transform(self.G.nodes[active_task]['x'], self.G.nodes[active_task]['y'])
                    # Create a transparent surface for the task
                    task_surface = pygame.Surface((task_radius*2, task_radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(task_surface, (255, 0, 0, 100), (task_radius, task_radius), task_radius)
                    self.screen.blit(task_surface, (task_x - task_radius, task_y - task_radius))
                
                # Draw UGV
                ugv_x, ugv_y = transform(self.G.nodes[self.ugv.position]['x'], self.G.nodes[self.ugv.position]['y'])
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

