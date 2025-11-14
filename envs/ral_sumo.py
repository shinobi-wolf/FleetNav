import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
import math
import pygame # Added pygame
import os # Added os for path joining
import time
import traci

import gymnasium as gym
from gymnasium import spaces

from envs.fleet.UGVSumo import UGVSumo
from envs.fleet.UAV import UAV
from sklearn.cluster import KMeans
import numpy as np

class SumoLMDEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array', 'quiet'], "render_fps": 30} # Adjusted FPS

    def __init__(self, config, render_mode=None): # Default render_mode is None
        super(SumoLMDEnv, self).__init__()

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.map = config["maze"]["map"]
        self.cell_size = config["maze"]["cell_size"]

        # Task vars
        self.num_tasks = config["task"]["num_tasks"]

        # Agent vars
        self.num_ugvs = config["fleet"]["count"]
        self.max_ugv_range = config["fleet"]["range"]
        self.drain_rate = config["fleet"]["drain_rate"]
        self.ugv_speed = config["fleet"]["speed"]
        self.num_primitives = config["fleet"]["num_primitives"]

        # Simulation vars
        self.max_time = config['simulation']['max_time']
        self.traffic_b = config['simulation']['traffic']
        self.ff = config['debug']['fast_forward'] # Fast forward factor for rendering
        self.render_mode = config['simulation']['render_mode']  # Render mode from config

        # Traffic vars
        self.traffic_std_dev = config['traffic']['std_dev']  # Standard deviation for Gaussian distribution
        self.traffic_reset_dur = config['traffic']['reset_dur'] 
        self.traffic_prob = config['traffic']['prob']
        self.traffic_num_centroids = config['traffic']['num_centroids']
        self.traffic_delay_factor = config['traffic']['delay_factor']  # Factor to adjust traffic delay


        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([self.num_primitives]*self.num_ugvs)
        
        ugv_status_space = spaces.MultiDiscrete([3]*self.num_ugvs) # 0: idle, 1: moving, 2: charging
        max_moves = 100
        steps2dest_space = spaces.MultiDiscrete(np.array([max_moves]*self.num_primitives*self.num_ugvs))
        nb_traffic_space = spaces.MultiDiscrete(np.array([3]*(self.num_primitives-1)*self.num_ugvs))
        state_info_space = spaces.MultiDiscrete(np.array([1000]*2*self.num_ugvs))
        self.observation_space = spaces.Dict({
            'ugv_status': ugv_status_space,
            'steps2dest': steps2dest_space,
            'nb_traffic': nb_traffic_space,
            'state_info': state_info_space,
        })

    def distribute_tasks(self):
        if self.task_list:
            # Convert task tuples to a NumPy array of shape (n_tasks, 2)
            tasks_array = np.array([[self.G.nodes[i]['x'], self.G.nodes[i]['y']] for i in self.task_list])
            # Cluster tasks into self.num_ugvs clusters
            kmeans = KMeans(n_clusters=self.num_ugvs)
            labels = kmeans.fit_predict(tasks_array)
            # Assign tasks to each UGV based on clustering labels
            ugv_task_list = [[] for _ in range(self.num_ugvs)]
            for task, label in zip(self.task_list, labels):
                ugv_task_list[label].append(task)
            for ugv in self.ugv_states:
                ugv.task_list = ugv_task_list[ugv.agent_id]
                self.assign_ugv_task(ugv.agent_id, 0)  # Assign tasks to UGVs

    def assign_ugv_task(self, ugv_id, mode):
        ugv = self.ugv_states[ugv_id]
        if not ugv.task_list:
            return
        
        ugv.active_task_list = []
        if mode == 0:
            active_task = ugv.task_list[0]  # Get the first task
        else:
            active_task = self.warehouse_node
        ugv_pos = ugv.position
        shortest_path = nx.shortest_path(self.G, source=ugv_pos, target=active_task)
        while shortest_path:
            next_pos = shortest_path.pop(0)
            if self.all_shortest_path_lengths[ugv_pos][next_pos] > 7 or next_pos == active_task:
                ugv.active_task_list.append(next_pos)
                ugv_pos = next_pos

    def fill_traffic_centroids(self):
        # Reset previous roads and weights
        # self.red_roads = []
        # self.yellow_roads = []
        # for u, v in self.G.edges():
        #     self.G[u][v]['traffic'] = 0

        # if random.random() < self.traffic_prob:
        #     self.traffic_centeroids = [[self.G.nodes[i]['x'], self.G.nodes[i]['y']] for i in random.sample(list(self.G.nodes), self.traffic_num_centroids)]
        #     centroid_array = np.array(self.traffic_centeroids)

        #     for u, v in self.G.edges():
        #         midpoint = np.array([(self.G.nodes[u]['x'] + self.G.nodes[v]['x']) / 2, (self.G.nodes[u]['y'] + self.G.nodes[v]['y']) / 2])
        #         total_contrib = 0
        #         if centroid_array.size > 0: # Check if centroids exist
        #             for centroid in centroid_array:
        #                 dist_sq = np.sum((midpoint - centroid)**2)
        #                 contrib = np.exp(-dist_sq / (2.0 * self.traffic_std_dev**2))
        #                 total_contrib += contrib

        #         if total_contrib < 0.33:
        #             self.G[u][v]['traffic'] = 0
        #         elif total_contrib < 0.66:
        #             self.G[u][v]['traffic'] = 1
        #             self.yellow_roads.append((u, v))
        #         else:
        #             self.G[u][v]['traffic'] = 2
        #             self.red_roads.append((u, v))
        # else:
        #     self.traffic_centeroids = []
        #     # Ensure weights are reset even if no traffic is generated
        #     for u, v in self.G.edges():
        #          self.G[u][v]['traffic'] = 0

        # Reset previous roads and weights
        self.red_roads = []
        self.yellow_roads = []
        for edge in self.G.edges():
            edge_id = self.G.edges[edge]['id']
            v_mean = traci.edge.getLastStepMeanSpeed(edge_id)
            v_limit = traci.lane.getMaxSpeed(edge_id+'_0')

            # congestion percent (0% = free flow, 100% = stopped)
            cong = max(0.0, min(1.0 - v_mean / v_limit, 1.0)) * 100.0

            # 3-bin discretization
            if cong < 33:
                self.G.edges[edge]['traffic'] = 0
            elif cong < 66:
                self.G.edges[edge]['traffic'] = 1
                self.yellow_roads.append(edge)
            else:
                self.G.edges[edge]['traffic'] = 2 
                self.red_roads.append(edge)

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
            '-c', self.map,
            '--step-length', '1',
            # '--delay', '500',
        ]
        traci.start(Sumo_config)
        traci.simulationStep()

        # Maze State
        self.G, self.all_shortest_path_lengths = self.build_graph() # Rebuild graph with new maze
        self.warehouse_node = random.choice(list(self.G.nodes))
        for u, v in self.G.edges():
            self.G[u][v]['traffic'] = 0 # Reset traffic on graph edges
        self.red_roads = []
        self.yellow_roads = []
        self.traffic_centeroids = []
        if self.traffic_b:
            self.fill_traffic_centroids() # Generate initial traffic if enabled

        # UGV States
        self.ugv_states = [UGVSumo(ugv_id=i, base_position=self.warehouse_node, cell_dist=self.cell_size, max_range=self.max_ugv_range,drain_rate=self.drain_rate,
                       max_speed=self.ugv_speed, traffic_delay_factor= self.traffic_delay_factor, G=self.G, num_primitives=self.num_primitives) for i in range(self.num_ugvs)]
        self.task_list = random.choices(list(self.G.nodes), k=self.num_tasks) # Randomly select tasks from graph nodes
        self.distribute_tasks() # Distribute tasks to UGVs

        # Metrics
        self.num_tasks_completed = 0
        self.task_completion_times = []
        self.total_ev_distance = 0.0
        self.total_energy_consumed = 0.0

        obs = self._get_observation()
        info = self.analyzer()

        return obs, info

    def _get_nb_traffic(self, position, nb_nodes):
        """
        Get the traffic density in the North, East, South, and West directions
        from the given UGV's position using the traffic_centroids as Gaussian centers.
        Returns a list of four discrete traffic levels (0, 1, 2) corresponding to [N, E, S, W].
        """
        traffic_levels = [self.G.edges.get((position, neighbor), {}).get('traffic', 2) for neighbor in nb_nodes]
        traffic_levels += [2] * (self.num_primitives-1 - len(traffic_levels))  # Pad to 4 directions
        return traffic_levels
    
    def _get_observation(self):
        steps2dest = []
        nb_traffic = []
        ugv_status = []
        state_info = []

        for ugv_id, ugv in enumerate(self.ugv_states):
            ugv_pos = ugv.position
            if ugv.active_task_list:
                task_pos = ugv.active_task_list[0]

                # Steps to destination in each direction
                steps2dest_ = []
                # nb_nodes = sorted(self.G.neighbors(ugv_pos), key=lambda x: np.linalg.norm(np.array([self.G.nodes[x]['x'], self.G.nodes[x]['y']]) - np.array([self.G.nodes[task_pos]['x'], self.G.nodes[task_pos]['y']])))[:4]
                nb_nodes = sorted(self.G.neighbors(ugv_pos))[:self.num_primitives-1]
                for nb_node in nb_nodes:
                    steps2dest_.append(self.all_shortest_path_lengths[nb_node][task_pos])
                steps2dest_ += [100] * (self.num_primitives-1 - len(steps2dest_))  # Pad to 5 directions
                steps2dest_.append(self.all_shortest_path_lengths[ugv_pos][task_pos])
                steps2dest.append(steps2dest_)

                # Neighbor Traffic
                nb_traffic.append(self._get_nb_traffic(ugv_pos, nb_nodes))

                ugv_status.append(ugv.status)

                ugv_pos_idx = list(self.G.nodes).index(ugv_pos)
                task_pos_idx = list(self.G.nodes).index(task_pos)
                state_info.append([ugv_pos_idx, task_pos_idx])
            else:
                steps2dest.append([100]*self.num_primitives)
                nb_traffic.append([2]*(self.num_primitives-1))
                ugv_status.append(0)
                ugv_pos_idx = list(self.G.nodes).index(ugv_pos)
                state_info.append([ugv_pos_idx, ugv_pos_idx]) # No task assigned


        steps2dest_flat = np.array(steps2dest, dtype=np.float32).flatten()
        nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()
        ugv_status_flat = np.array(ugv_status, dtype=np.int32).flatten()
        state_info_flat = np.array(state_info, dtype=np.int32).flatten()
        obs_dict = {
            'steps2dest': steps2dest_flat,
            'nb_traffic': nb_traffic_flat,
            'ugv_status': ugv_status_flat,
            'state_info': state_info_flat
        }

        return obs_dict
    
    def analyzer(self):
        """Analyzes the current state and returns an info dictionary."""
        self.total_ev_distance = sum(ugv.distance_traveled for ugv in self.ugv_states)
        self.total_energy_consumed = sum(ugv.energy_consumed for ugv in self.ugv_states)
        # if self.bms:

        task_completion_time_diffs = [self.task_completion_times[i] - self.task_completion_times[i-1] for i in range(1, len(self.task_completion_times))]
        avg_task_completion_time = np.mean(task_completion_time_diffs) if task_completion_time_diffs else 0.0

        active_tasks = [ugv.active_task_list[-1] if ugv.active_task_list else None for ugv in self.ugv_states]
        info = {
            'time_elapsed': self.time_elapsed,
            'current_timestep': self.current_timestep,
            
            'num_tasks_completed': self.num_tasks_completed,
            'total_ev_distance': self.total_ev_distance,
            'total_energy_consumed': self.total_energy_consumed,
            'avg_task_completion_time': avg_task_completion_time,

            'active_tasks': active_tasks,
            'red_roads': self.red_roads,
            'yellow_roads': self.yellow_roads,
        }
        return info

    def step(self, action):
        if self.terminated:
            return None, None, self.terminated, None, self.info # Return None if already terminated
        traci.simulationStep()
        self.ugv_handler(action)
        self.time_elapsed = min([ugv.local_time for ugv in self.ugv_states])
        truncated = self.task_handler() # Handle task completion and updates
        reward = self._get_reward()
        
        self.current_timestep += 1

        # Update traffic periodically *before* updating info for the next step's rendering
        if self.traffic_b:
            if self.current_timestep % self.traffic_reset_dur == 0:
                self.fill_traffic_centroids()

        self._check_termination_condition() # Check termination based on new state

        obs = self._get_observation() # Gets observation *after* action/reward

        self.info = self.analyzer() # Update info with current state
        self.last_move_time = [0 for _ in range(self.num_ugvs)]
        self.action_response = [None for _ in range(self.num_ugvs)]

        return obs, reward, self.terminated, truncated, self.info # Return standard gym step tuple

    def ugv_handler(self, action):
        truncated = False
        for ugv_id, ugv in enumerate(self.ugv_states):
            act = action[ugv_id]
            self.ugv_states[ugv_id].move(act, self.G)
            if ugv.position == self.warehouse_node:
                ugv.recharge()

    def task_handler(self):
        truncated = False
        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status == 2: # If UGV is charging, skip task handling
                continue
            if ugv.local_time == self.time_elapsed:
                ugv.status = 1
                if ugv.position == ugv.active_task_list[0]:
                    ugv.task_completed = True
                    truncated = True
                    ugv.active_task_list.pop(0)
                    if ugv.position == ugv.task_list[0]:
                        ugv.task_list.pop(0)
                        self.num_tasks_completed += 1
                        self.task_completion_times.append(self.time_elapsed)
                    if len(ugv.active_task_list) == 0:
                        self.assign_ugv_task(ugv_id, 0) # Assign new task if available
                        ugv.charging_status = 0
                    ugv.prev_position = ugv.position
                else:
                    ugv.task_completed = False
            else:
                ugv.status = 0

        return truncated # Return if any UGV completed a task
    
    def _get_reward(self):
        # Calculate reward based on the outcome of the action in the previous state
        reward = 0.0

        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.status:
                # print("Computing reward for UGV:", ugv_id)
                # --- Progress-Based Shaping Reward ---
                # Reward for moving closer to the nearest task
                bonus_factor = 5.0 
                if ugv.active_task_list:
                    current_distance = self.all_shortest_path_lengths[ugv.position][ugv.active_task_list[0]]
                    prev_distance = self.all_shortest_path_lengths[ugv.prev_position][ugv.active_task_list[0]]
                    reward += bonus_factor * (prev_distance - current_distance)
                    print("Progress Reward:", bonus_factor * (prev_distance - current_distance))
                
                # --- Task Completion Reward ---
                task_completion_reward = 50.0
                if ugv.task_completed:
                    reward += task_completion_reward
                    print("Task Completion Reward:", task_completion_reward)

                # --- Time Penalty ---
                # Penalize based on the time taken for the action
                time_penalty_factor = 5.0 # Adjust this factor based on desired behavior
                reward -= time_penalty_factor * ugv.move_time/self.cell_size
                print("Time Penalty:", -1*time_penalty_factor * ugv.move_time/self.cell_size)

                # --- Invalid Move Penalty ---
                invalid_move_penalty = 40.0
                if ugv.act_response == 0:
                    reward -= invalid_move_penalty
                    print("Invalid Move Penalty:", -1*invalid_move_penalty)

                # If UGV is out of range, apply a penalty
                # out_of_range_penalty = 20.0
                # if ugv.current_range < nx.shortest_path_length(self.G, ugv.position, self.warehouse_node)*50:
                #     reward -= out_of_range_penalty
                    # print("Out of Range Penalty:", out_of_range_penalty)

        return reward


    def _check_termination_condition(self):
        no_fuel_flag = True
        for ugv in self.ugv_states:
            if ugv.current_range >= 50:
                no_fuel_flag = False
                break
        if self.time_elapsed >= self.max_time or no_fuel_flag:
            self.terminated = True
        else:
            self.terminated = False
        if not self.ugv_states[0].active_task_list:
            self.terminated = True

    def render(self):
        """
        Render the environment using pygame.
        Shows the road network and UGV agents.
        """
        if not hasattr(self, 'render_mode') or self.render_mode == 'quiet':
            return
        
        print(f"Time: {self.time_elapsed:.2f}, Tasks completed: {self.num_tasks_completed}")
        print(f"Current timestep: {self.current_timestep}, Fast forward: {self.ff}")
        print("UGV Info:")
        for ugv in self.ugv_states:
            print(str(ugv))

        # Initialize pygame display if not already done
        if self.screen is None and self.render_mode in ['human', 'rgb_array']:
            pygame.init()
            self.screen_width = 1000
            self.screen_height = 800
            self.clock = pygame.time.Clock()
            
            if self.render_mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("SUMO LMD Environment")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.screen is None:
            return

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
        
        # Draw edges (roads) first so nodes are on top
        for u, v in self.G.edges():
            start_x, start_y = transform(self.G.nodes[u]['x'], self.G.nodes[u]['y'])
            end_x, end_y = transform(self.G.nodes[v]['x'], self.G.nodes[v]['y'])
            
            # Color based on traffic
            color = (100, 100, 100)  # Default gray for no traffic
            if (u, v) in self.red_roads or (v, u) in self.red_roads:
                color = (255, 0, 0)  # Red for high traffic
            elif (u, v) in self.yellow_roads or (v, u) in self.yellow_roads:
                color = (255, 255, 0)  # Yellow for medium traffic
            
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 4)
        
        # Draw nodes (junctions)
        node_radius = 5
        for node in self.G.nodes:
            x, y = transform(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), node_radius)
        
        # Draw warehouse (base station) with different color and size
        wx, wy = transform(self.G.nodes[self.warehouse_node]['x'], self.G.nodes[self.warehouse_node]['y'])
        pygame.draw.circle(self.screen, (0, 0, 255), (wx, wy), node_radius * 2)
        
        # Draw UGVs
        ugv_radius = 8
        for ugv in self.ugv_states:
            if ugv.position is not None:
                ugv_x, ugv_y = transform(self.G.nodes[ugv.position]['x'], self.G.nodes[ugv.position]['y'])
                # Color UGV based on status: Green for active, Yellow for charging
                color = (0, 255, 0) if ugv.status == 1 else (255, 255, 0)
                pygame.draw.circle(self.screen, color, (ugv_x, ugv_y), ugv_radius)

                active_task = ugv.active_task_list[-1]
                task_x, task_y = transform(self.G.nodes[active_task]['x'], self.G.nodes[active_task]['y'])
                pygame.draw.circle(self.screen, (255, 0, 0), (task_x, task_y), node_radius * 2)  # Draw task location in red

        # Add text for time and tasks completed
        if pygame.font:
            font = pygame.font.Font(None, 24)
            text = font.render(f"Time: {self.time_elapsed:.2f}, Tasks: {self.num_tasks_completed}", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))
        
        if self.render_mode == 'human':
            pygame.display.flip()
            if self.num_tasks_completed < self.ff:
                self.clock.tick(self.metadata["render_fps"])
            else:
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.close()
                            waiting = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                            waiting = False
                    self.clock.tick(self.metadata["render_fps"])

            
        if self.render_mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
            
    def close(self):
        """Cleans up pygame resources."""
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        traci.close()

