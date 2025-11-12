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

        # Simulation vars
        self.max_time = config['simulation']['max_time']
        self.max_steps = config['simulation']['max_steps']
        self.render_mode = config['simulation']['render_mode']  # Render mode from config
        self.render_fps = config['simulation'].get('render_fps', 30)  # Default to 30 if not specified
        self.step_length = config['simulation']['step_length']
        self.ff = config['debug']['fast_forward']

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([4]*self.num_ugvs)
        
        ugv_status_space = spaces.MultiDiscrete([3]*self.num_ugvs) # 0: idle, 1: moving, 2: charging
        max_moves = 50
        steps2dest_space = spaces.MultiDiscrete(np.array([max_moves]*5*self.num_ugvs))
        self.observation_space = spaces.Dict({
            'ugv_status': ugv_status_space,
            'steps2dest': steps2dest_space,
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

            lid = f"{eid}_0"  # Assume single lane per edge for simplicity
            length = traci.lane.getLength(lid)
            speed = traci.edge.getLastStepMeanSpeed(eid)

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
                speed=speed,
                weight=length / speed if speed > 0 else float("inf")
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

        Sumo_config = [
            'sumo',
            '-c', self.net_path,
            '--step-length', str(self.step_length),
            # '--delay', '500',
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

    # def _get_nb_traffic(self, position, nb_nodes):
    #     """
    #     Get the traffic density in the North, East, South, and West directions
    #     from the given UGV's position using the traffic_centroids as Gaussian centers.
    #     Returns a list of four discrete traffic levels (0, 1, 2) corresponding to [N, E, S, W].
    #     """
    #     traffic_levels = [self.G.edges.get((position, neighbor), {}).get('traffic', 0) for neighbor in nb_nodes]
    #     traffic_levels += [0] * (4 - len(traffic_levels))  # Pad to 4 directions
    #     return traffic_levels
    
    def _get_observation(self):
        steps2dest = []
        nb_traffic = []
        ugv_status = []

        for ugv_id, ugv in enumerate(self.ugv_states):
            ugv_pos = ugv.position
            # Steps to destination in each direction
            steps2dest_ = []
            if ugv.task_list:  # Check if task list is not empty
                task_pos = ugv.task_list[0]
                nb_nodes = sorted(self.G.neighbors(ugv_pos))[:4]
                for nb_node in nb_nodes:
                    steps2dest_.append(self.all_shortest_path_lengths[nb_node][task_pos])
                for _ in range(4 - len(nb_nodes)):
                    steps2dest_.append(100)
            else:
                # If no tasks, use zeros for steps
                steps2dest_ = [100] * 4
            steps2dest.append(steps2dest_)

            # # Neighbor Traffic
            # nb_traffic.append(self._get_nb_traffic(ugv_pos, nb_nodes))

            ugv_status.append(ugv.status)

        steps2dest_flat = np.array(steps2dest, dtype=np.float32).flatten()
        # nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()
        ugv_status_flat = np.array(ugv_status, dtype=np.int32).flatten()
        obs_dict = {
            'steps2dest': steps2dest_flat,
            # 'nb_traffic': nb_traffic_flat,
            'ugv_status': ugv_status_flat,
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
        truncated = self.task_handler() # Handle task completion and updates
        # reward = self._get_reward()
        
        self.current_timestep += 1
        traci.simulationStep()
        self.time_elapsed = traci.simulation.getTime()

        self._check_termination_condition() # Check termination based on new state

        obs = self._get_observation() # Gets observation *after* action/reward
        self.info = self.analyzer() # Update info with current state
        # self.last_move_time = [0 for _ in range(self.num_ugvs)]
        # self.action_response = [None for _ in range(self.num_ugvs)]

        return obs, 0, self.terminated, truncated, self.info # Return standard gym step tuple

    def ugv_handler(self, action):
        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status==2:
                continue
            if ugv.veh_id and ugv.veh_id in traci.vehicle.getIDList():
                ugv.status = 0
            else:
                ugv.status = 1
            act = action[ugv_id]
            move_flag, edge_id = ugv.move(act, self.G)
            if move_flag:
                route_id = f"ugv_{ugv_id}_{self.current_timestep}_route"
                ugv.veh_id = f"ugv_{ugv_id}_{self.current_timestep}"
                traci.route.add(route_id, [edge_id])
                traci.vehicle.add(ugv.veh_id, route_id)
            if ugv.position == self.warehouse_node:
                ugv.recharge()

    def task_handler(self):
        truncated = False
        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status==1:
                if ugv.task_list and ugv.position == ugv.task_list[0]:
                    truncated = True
                    ugv.task_list.pop(0)
                    self.num_tasks_completed += 1
                    self.task_completion_times.append(self.time_elapsed)
                    if len(ugv.task_list) == 0:
                        ugv.status = 2

        return truncated # Return if any UGV completed a task
    
    def _get_reward(self):
        # Calculate reward based on the outcome of the action in the previous state
        reward = 0.0

        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status and ugv.task_list:
                # --- Progress-Based Shaping Reward ---
                # Reward for moving closer to the nearest task
                # Note: This reward requires tracking prev_position which is not currently in UGVSumo
                # Commenting out until prev_position is added to UGVSumo class
                # bonus_factor = 5.0 
                # current_distance = self.all_shortest_path_lengths[ugv.position][ugv.task_list[0]]
                # prev_distance = self.all_shortest_path_lengths[ugv.prev_position][ugv.task_list[0]]
                # reward += bonus_factor * (prev_distance - current_distance)
                
                # --- Task Completion Reward ---
                # Note: This reward requires tracking task_completed which is not currently in UGVSumo
                # Commenting out until task_completed is added to UGVSumo class
                # task_completion_reward = 50.0
                # if ugv.task_completed:
                #     reward += task_completion_reward
                # --- Time Penalty ---
                # Note: This penalty requires tracking move_time which is not currently in UGVSumo
                # time_penalty_factor = 5.0
                # reward -= time_penalty_factor * ugv.move_time/self.cell_size

                # --- Invalid Move Penalty ---
                # Note: This penalty requires tracking act_response which is not currently in UGVSumo
                # invalid_move_penalty = 40.0
                # if ugv.act_response == 0:
                #     reward -= invalid_move_penalty

                # If UGV is out of range, apply a penalty
                out_of_range_penalty = 20.0
                if ugv.current_range < 0:
                    reward -= out_of_range_penalty

        return reward


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
                return (int(margin + (x - min_x) * scale), int(margin + (y - min_y) * scale))
            
            # Draw edges (roads)
            for u, v in self.G.edges():
                start_x, start_y = transform(self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                end_x, end_y = transform(self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                # Draw roads in gray
                pygame.draw.line(self.screen, (150, 150, 150), (start_x, start_y), (end_x, end_y), 2)
            
            # Draw nodes (junctions)
            node_radius = 4
            for node in self.G.nodes:
                x, y = transform(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
                pygame.draw.circle(self.screen, (80, 80, 80), (x, y), node_radius)
            
            # Draw warehouse (base station) - blue
            wx, wy = transform(self.G.nodes[self.warehouse_node]['x'], self.G.nodes[self.warehouse_node]['y'])
            pygame.draw.circle(self.screen, (0, 0, 255), (wx, wy), 10)
            
            # Draw task locations and UGVs
            task_radius = 7
            ugv_radius = 9
            for i, ugv in enumerate(self.ugv_states):
                if ugv.position is not None:
                    # Draw active task location (red)
                    if ugv.task_list:
                        active_task = ugv.task_list[0]
                        task_x, task_y = transform(self.G.nodes[active_task]['x'], self.G.nodes[active_task]['y'])
                        pygame.draw.circle(self.screen, (255, 0, 0), (task_x, task_y), task_radius)
                    
                    # Draw UGV
                    ugv_x, ugv_y = transform(self.G.nodes[ugv.position]['x'], self.G.nodes[ugv.position]['y'])
                    # Color based on status: Green=Moving, Yellow=Idle, Gray=Charging
                    if ugv.status == 1:
                        color = (0, 200, 0)  # Green - moving
                    elif ugv.status == 2:
                        color = (128, 128, 128)  # Gray - charging
                    else:
                        color = (255, 200, 0)  # Yellow - idle
                    pygame.draw.circle(self.screen, color, (ugv_x, ugv_y), ugv_radius)
                    # Draw UGV ID
                    if pygame.font:
                        font = pygame.font.Font(None, 18)
                        text = font.render(str(i), True, (255, 255, 255))
                        text_rect = text.get_rect(center=(ugv_x, ugv_y))
                        self.screen.blit(text, text_rect)

            # Draw info panel
            if pygame.font:
                font = pygame.font.Font(None, 28)
                y_offset = 10
                # Time and tasks
                text = font.render(f"Time: {self.time_elapsed:.1f}s | Step: {self.current_timestep}", True, (0, 0, 0))
                self.screen.blit(text, (10, y_offset))
                y_offset += 30
                text = font.render(f"Tasks: {self.num_tasks_completed}/{self.num_tasks}", True, (0, 0, 0))
                self.screen.blit(text, (10, y_offset))
                y_offset += 30
                text = font.render(f"Distance: {self.total_ev_distance:.1f}m", True, (0, 0, 0))
                self.screen.blit(text, (10, y_offset))
                
                # Legend
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

