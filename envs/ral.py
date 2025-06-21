import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
from pyamaze import maze
import tkinter as tk
tk.Tk.state = lambda self, s=None: self.wm_state('normal' if s == 'zoomed' else s)
import math
import pygame # Added pygame
import os # Added os for path joining
import time

import gymnasium as gym
from gymnasium import spaces

from envs.fleet.UGV import UGV
from sklearn.cluster import KMeans
import numpy as np
# from agents.UAV import UAVAgent

def generate_df_maze(rows=20, cols=20, lp=80):
    # Create a Pyamaze maze
    m = maze(rows, cols)
    m.CreateMaze(loopPercent=lp)  # No loops to match a grid-like maze

    # List to store cell data
    maze_data = []
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            cell = (x, y)
            cell_walls = m.maze_map[cell]  # Get wall info for each cell

            E = 1 if cell_walls['E'] else 0
            W = 1 if cell_walls['W'] else 0
            N = 1 if cell_walls['N'] else 0
            S = 1 if cell_walls['S'] else 0

            maze_data.append([f"({x},{y})", E, W, N, S])

    df = pd.DataFrame(maze_data, columns=["  cell  ", "E", "W", "N", "S"])
    if '  cell  ' in df.columns:
        df[['row', 'col']] = (
            df['  cell  '].astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.split(',', expand=True)
            .astype(int)
        )
    return df

# Define colors for pygame
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
DARK_GREEN = (0, 100, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Map agent IDs to colors (example)
AGENT_COLORS = [
    (50, 50, 255),    # Blue-ish
    (255, 50, 50),    # Red-ish
    (50, 255, 50),    # Green-ish
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
]

def get_agent_color(agent_id):
    """Returns a distinct color for a given agent ID."""
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


class LMDEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array', 'quiet'], "render_fps": 30} # Adjusted FPS

    def __init__(self, config, render_mode=None): # Default render_mode is None
        super(LMDEnv, self).__init__()

        '''GUI CONFIGURATION'''
        self.render_mode = render_mode
        if not render_mode:
            self.render_mode = config['simulation']['render_mode']
        if self.render_mode == 'human':
            self.metadata['render_fps'] = config['simulation'].get('render_fps', 30) # Default to 30 FPS if not specified
            self.manual_control = config['debug']['manual_control']
            self.ff = config['debug']['fast_forward'] # Fast forward mode

        self.screen = None
        self.clock = None
        self.font = None
        self.cell_size_px = 70 #config["world"].get("cell_size", 30) # Pixel size for rendering cells
        self.wall_thickness = 4
        self.screen_width = None
        self.screen_height = None
        self.truck_image = None
        self.warehouse_image = None # Optional: image for warehouse
        self.task_image = None      # Optional: image for tasks
        self.bs_image = None        # Optional: image for base stations

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.max_row = config["maze"]["maze_size"]
        self.max_col = config["maze"]["maze_size"]
        self.maze_lp = config["maze"]["maze_loop_percentage"]
        self.cell_size = config["maze"]["cell_size"] # Physical cell size

        # Task vars
        self.num_tasks = config["task"]["num_tasks"]
        self.task_prob = config["task"]["task_prob"]

        # Warehouse vars
        self.warehouse_opt = config["maze"]["warehouse"]
        if self.warehouse_opt == "center":
            center_row = self.max_row // 2 + self.max_row % 2
            center_col = self.max_col // 2 + self.max_row % 2
            self.warehouse_pos = (int(center_row), int(center_col))
        elif self.warehouse_opt == "random":
            self.warehouse_pos = (
                random.randint(1, self.max_row),
                random.randint(1, self.max_col)
            )

        # # Base station vars
        # self.num_base_stations = config["world"]["num_base_stations"]
        # bs_df = self.df_maze.sample(self.num_base_stations, random_state=47)[['row', 'col']]
        # self.base_stations = []
        # for t in bs_df.values.tolist():
        #     self.base_stations.append( (int(t[0]), int(t[1])) )

        # Agent vars
        self.num_ugvs = config["fleet"]['ugv']["count"]
        self.max_ugv_range = config["fleet"]['ugv']['range']
        self.drain_rate = config["fleet"]['ugv']['drain_rate']
        self.ugv_speed = config["fleet"]['ugv']['speed']
        
        # Simulation vars
        self.bms = config['simulation']['bms']
        self.max_time = config['simulation']['max_time']
        self.traffic_b = config['simulation']['traffic']
        self.algo = config['simulation']['algo']

        # Traffic vars
        self.traffic_std_dev = config['traffic']['std_dev']  # Standard deviation for Gaussian distribution
        self.traffic_reset_dur = config['traffic']['reset_dur'] 
        self.traffic_prob = config['traffic']['prob']
        self.traffic_num_centroids = config['traffic']['num_centroids']
        self.traffic_delay_factor = config['traffic']['delay_factor']  # Factor to adjust traffic delay

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([config["fleet"]['ugv']["num_primitives"]]*self.num_ugvs)
        
        ugv_status_space = spaces.MultiDiscrete([3]*self.num_ugvs) # 0: idle, 1: moving, 2: charging
        max_moves = self.max_row * self.max_col+1
        steps2dest_space = spaces.MultiDiscrete(np.array([max_moves]*5*self.num_ugvs))
        nb_traffic_space = spaces.MultiDiscrete(np.array([3,3,3,3]*self.num_ugvs))
        self.observation_space = spaces.Dict({
            'ugv_status': ugv_status_space,
            'steps2dest': steps2dest_space,
            'nb_traffic': nb_traffic_space,
        })

        # Initialize rendering if mode is 'human'
        if self.render_mode == "human":
            self._initialize_render()

    def _initialize_render(self):
        """Initializes pygame, screen, clock, fonts, and loads assets."""
        if self.screen is not None: # Avoid re-initialization
            return
        try:
            pygame.init()
            pygame.display.set_caption("LMD Simulation")
            self.clock = pygame.time.Clock()

            # Calculate screen dimensions based on maze size and cell pixel size
            # Add padding for potential info text display later
            padding = 200
            self.screen_width = self.max_col * self.cell_size_px
            self.screen_height = self.max_row * self.cell_size_px + padding # Add padding at the bottom
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            # Load font
            try:
                self.font = pygame.font.SysFont(None, 24) # Adjust font size as needed
            except Exception as e:
                print(f"Warning: Could not load system font. Text rendering might fail. Error: {e}")
                self.font = pygame.font.Font(None, 24) # Fallback to default font

            # Load images (adjust paths as necessary)
            try:
                # Assuming 'data' directory is in the same parent directory as 'envs'
                truck_img_path = os.path.join('data/truck.png')
                self.truck_image = pygame.image.load(truck_img_path).convert_alpha() # Use convert_alpha for transparency
                # Scale truck image to fit within a cell (e.g., 80% of cell size)
                scale_factor = 0.8
                img_size = int(self.cell_size_px * scale_factor)
                self.truck_image = pygame.transform.scale(self.truck_image, (img_size, img_size))
                # Optional: Load other images (warehouse, task, base station) here
                # self.warehouse_image = pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'warehouse.png')).convert_alpha(), (img_size, img_size))
                # self.task_image = pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'task.png')).convert_alpha(), (img_size // 2, img_size // 2))
                # self.bs_image = pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'base_station.png')).convert_alpha(), (img_size, img_size))

            except pygame.error as e:
                print(f"Warning: Could not load image assets. Rendering with shapes. Error: {e}")
                self.truck_image = None
                # self.warehouse_image = None
                # self.task_image = None
                # self.bs_image = None
        except Exception as e:
             print(f"Error initializing Pygame: {e}")
             self.render_mode = None # Disable rendering if initialization fails

    def _cell_to_pygame(self, row, col):
        """Converts maze (row, col) to pygame pixel coordinates (top-left of cell)."""
        # Pygame origin (0,0) is top-left
        px = (col - 1) * self.cell_size_px
        py = (row - 1) * self.cell_size_px # Adjusted for top-left origin
        return int(px), int(py)
    
    def render(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.sleep_flag = True
            if self.screen is None:
                self._initialize_render() # Initialize if not already done
                if self.screen is None: # Check if initialization failed
                    return None # Cannot render, return None

            # For rgb_array mode, we need a separate surface to render on
            if self.render_mode == "rgb_array":
                surface = pygame.Surface((self.screen_width, self.screen_height-200))
                render_surface = surface
            else:  # For human mode, render directly to screen
                render_surface = self.screen

                # --- Event Handling (only for human mode) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close() # Proper cleanup
                        return None
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.terminated = True

            # --- Drawing ---
            render_surface.fill(WHITE) # Clear surface

            # --- Draw Maze Area (Grid and Walls) ---
            maze_area_height = self.max_row * self.cell_size_px
            # --- Draw Maze Area (Grid and Walls) using self.G ---
            for r in range(1, self.max_row + 1):
                for c in range(1, self.max_col + 1):
                    current_node = (r, c)
                    # Skip if node somehow doesn't exist in the graph (should not happen for grid)
                    if current_node not in self.G:
                        continue

                    px_top_left, py_top_left = self._cell_to_pygame(r, c)

                    # Check North Wall: Draw wall if no edge exists between current node and North neighbor (r-1, c)
                    # G.has_edge handles boundary cases (e.g., r=1, neighbor (0,c) doesn't exist, so has_edge is False)
                    north_neighbor = (r - 1, c)
                    if not self.G.has_edge(current_node, north_neighbor):
                        # Draw the North wall of cell (r, c)
                        pygame.draw.line(render_surface, BLACK, (px_top_left, py_top_left), (px_top_left + self.cell_size_px, py_top_left), self.wall_thickness)
                    else:
                        pygame.draw.line(render_surface, GRAY, (px_top_left, py_top_left), (px_top_left + self.cell_size_px, py_top_left), self.wall_thickness)

                    # Check West Wall: Draw wall if no edge exists between current node and West neighbor (r, c-1)
                    west_neighbor = (r, c - 1)
                    if not self.G.has_edge(current_node, west_neighbor):
                        # Draw the West wall of cell (r, c)
                        pygame.draw.line(render_surface, BLACK, (px_top_left, py_top_left), (px_top_left, py_top_left + self.cell_size_px), self.wall_thickness)
                    else:
                        pygame.draw.line(render_surface, GRAY, (px_top_left, py_top_left), (px_top_left, py_top_left + self.cell_size_px), self.wall_thickness)

            # Draw outer boundary walls (South and East edges explicitly)
            # This ensures the bottom border of the last row and the right border of the last column are drawn.
            pygame.draw.line(render_surface, BLACK, (0, maze_area_height), (self.screen_width, maze_area_height), self.wall_thickness) # Bottom boundary
            pygame.draw.line(render_surface, BLACK, (self.screen_width, 0), (self.screen_width, maze_area_height), self.wall_thickness) # Right boundary


            # --- Draw Traffic Roads ---
            road_thickness = max(1, int(self.cell_size_px * 0.15)) # Adjust thickness relative to cell size
            for u, v in self.red_roads:
                u_px_c, u_py_c = self._cell_to_pygame_center(u[0], u[1])
                v_px_c, v_py_c = self._cell_to_pygame_center(v[0], v[1])
                pygame.draw.line(render_surface, RED, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)
            for u, v in self.yellow_roads:
                u_px_c, u_py_c = self._cell_to_pygame_center(u[0], u[1])
                v_px_c, v_py_c = self._cell_to_pygame_center(v[0], v[1])
                pygame.draw.line(render_surface, YELLOW, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)


            # --- Draw Warehouse ---
            wh_r, wh_c = self.warehouse_pos
            wh_px_c, wh_py_c = self._cell_to_pygame_center(wh_r, wh_c)
            warehouse_radius = int(self.cell_size_px * 0.4)
            pygame.draw.circle(render_surface, BLUE, (wh_px_c, wh_py_c), warehouse_radius)

            for ugv_id, ugv in enumerate(self.ugv_states):
                # --- Draw Tasks ---
                task_r, task_c = ugv.active_task
                task_px_c, task_py_c = self._cell_to_pygame_center(task_r, task_c)
                task_radius = int(self.cell_size_px * 0.25)
                pygame.draw.circle(render_surface, MAGENTA, (task_px_c, task_py_c), task_radius)

                # --- Draw UGVs and Battery ---
                ugv_r, ugv_c = ugv.get_actual_position(self.time_elapsed)

                ugv_px_c, ugv_py_c = self._cell_to_pygame_center(ugv_r, ugv_c)
                agent_color = 0

                # Draw UGV (Image or Shape)
                if self.truck_image:
                    ugv_heading_dir = ugv.current_heading
                    truck_img_rotated = pygame.transform.rotate(self.truck_image, (ugv_heading_dir-1)%4 * -90) # Rotate image based on heading

                    img_rect = truck_img_rotated.get_rect(center=(ugv_px_c, ugv_py_c))
                    render_surface.blit(truck_img_rotated, img_rect)
                else:
                    # Fallback to drawing a colored rectangle
                    ugv_size = int(self.cell_size_px * 0.7)
                    ugv_rect = pygame.Rect(0, 0, ugv_size, ugv_size)
                    ugv_rect.center = (ugv_px_c, ugv_py_c)
                    pygame.draw.rect(render_surface, agent_color, ugv_rect)
                    pygame.draw.rect(render_surface, BLACK, ugv_rect, 1) # Border

                # # Draw Battery Bar below UGV
                # battery_width = int(self.cell_size_px * 0.8)
                # battery_height = max(3, int(self.cell_size_px * 0.1)) # Ensure minimum height
                # battery_x = ugv_px_c - battery_width // 2
                # # Position below the center, slightly offset
                # battery_y = ugv_py_c + (self.truck_image.get_height() // 2 if self.truck_image else self.cell_size_px // 2) + 2

                # filled_width = int(ugv.current_range_percent * battery_width)

                # # Background of battery bar (e.g., light gray)
                # pygame.draw.rect(render_surface, GRAY, (battery_x, battery_y, battery_width, battery_height))
                # # Filled portion (Green)
                # pygame.draw.rect(render_surface, GREEN, (battery_x, battery_y, filled_width, battery_height))
                # # Border
                # pygame.draw.rect(render_surface, BLACK, (battery_x, battery_y, battery_width, battery_height), 1)

            # For rgb_array mode, convert surface to numpy array and return
            if self.render_mode == "rgb_array":
                return self._get_array_from_surface(surface)
            
            # --- Draw Info Text ---
            if self.font:
                info_y_start = maze_area_height + 5 # Start below maze area
                info_line_height = 20
                texts = [
                    f"Time: {self.time_elapsed:.2f}s",
                    f"Timestep: {self.current_timestep}",
                    f"Tasks Done: {self.num_tasks_completed}",
                    f"Total Dist: {self.total_ev_distance:.1f}m",
                    f"Total Energy: {self.total_energy_consumed:.1f}",
                ]
                for i, text in enumerate(texts):
                    text_surface = self.font.render(text, True, BLACK)
                    render_surface.blit(text_surface, (5, info_y_start + i * info_line_height))
            
            # For human mode, update the display
            pygame.display.flip()

            print("=== Environment State ===")
            print(f"Time Elapsed: {self.time_elapsed:.2f}s")
            print(f"Current Timestep: {self.current_timestep}")
            print(f"Tasks Completed: {self.num_tasks_completed}")
            print(f"Total Distance Traveled: {self.total_ev_distance:.1f}m")
            print(f"Total Energy Consumed: {self.total_energy_consumed:.1f}")
            print("UGV States:")
            for idx, ugv in enumerate(self.ugv_states):
                print(str(ugv))
            print("Traffic Information:")
            print(f"  Red Roads: {self.red_roads}")
            print(f"  Yellow Roads: {self.yellow_roads}")
            print(f"  Traffic Centroids: {self.traffic_centeroids}")
            print("==========================")
            
            # Wait until the Return key is pressed before ending render
            if self.num_tasks_completed < self.ff:
                self.clock.tick(self.metadata["render_fps"])
            else:
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.close()
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                key_pressed = 1
                            elif event.key == pygame.K_RIGHT:
                                key_pressed = 2
                            elif event.key == pygame.K_DOWN:
                                key_pressed = 3
                            elif event.key == pygame.K_LEFT:
                                key_pressed = 4
                            elif event.key == pygame.K_SPACE:
                                key_pressed = 5
                            elif event.key != pygame.K_RETURN:
                                continue
                            waiting = False
                    self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "print":
            print("=== Environment State ===")
            print(f"Time Elapsed: {self.time_elapsed:.2f}s")
            print(f"Current Timestep: {self.current_timestep}")
            print(f"Tasks Completed: {self.num_tasks_completed}")
            print(f"Total Distance Traveled: {self.total_ev_distance:.1f}m")
            print(f"Total Energy Consumed: {self.total_energy_consumed:.1f}")
            print("UGV States:")
            for idx, ugv in enumerate(self.ugv_states):
                print(str(ugv))
            print("Traffic Information:")
            print(f"  Red Roads: {self.red_roads}")
            print(f"  Yellow Roads: {self.yellow_roads}")
            print(f"  Traffic Centroids: {self.traffic_centeroids}")      
        
        return None  # Return None for all render modes except rgb_array
    
    def _cell_to_pygame_center(self, row, col):
        """Converts maze (row, col) to pygame pixel coordinates (center of cell)."""
        px, py = self._cell_to_pygame(row, col)
        px += self.cell_size_px // 2
        py += self.cell_size_px // 2
        return int(px), int(py)

    def distrubute_tasks(self):
        if self.task_list:
            # Convert task tuples to a NumPy array of shape (n_tasks, 2)
            tasks_array = np.array(self.task_list)
            # Cluster tasks into self.num_ugvs clusters
            kmeans = KMeans(n_clusters=self.num_ugvs)
            labels = kmeans.fit_predict(tasks_array)
            # Assign tasks to each UGV based on clustering labels
            ugv_task_list = [[] for _ in range(self.num_ugvs)]
            for task, label in zip(self.task_list, labels):
                ugv_task_list[label].append(task)
            for ugv in self.ugv_states:
                ugv.task_list = ugv_task_list[ugv.agent_id]
                ugv.active_task = ugv.task_list[0] if ugv.task_list else None

    def fill_traffic_centroids(self):
        # Reset previous roads and weights
        self.red_roads = []
        self.yellow_roads = []
        for u, v in self.G.edges():
            self.G[u][v]['traffic'] = 0

        if random.random() < self.traffic_prob:
            self.traffic_centeroids = random.sample(list(self.G.nodes), self.traffic_num_centroids)
            centroid_array = np.array(self.traffic_centeroids)

            for u, v in self.G.edges():
                midpoint = np.array([(u[0] + v[0])/2, (u[1] + v[1])/2])
                total_contrib = 0
                if centroid_array.size > 0: # Check if centroids exist
                    for centroid in centroid_array:
                        dist_sq = np.sum((midpoint - centroid)**2)
                        contrib = np.exp(-dist_sq / (2.0 * self.traffic_std_dev**2))
                        total_contrib += contrib

                if total_contrib < 0.33:
                    self.G[u][v]['traffic'] = 0
                elif total_contrib < 0.66:
                    self.G[u][v]['traffic'] = 1
                    self.yellow_roads.append((u, v))
                else:
                    self.G[u][v]['traffic'] = 2
                    self.red_roads.append((u, v))
        else:
            self.traffic_centeroids = []
            # Ensure weights are reset even if no traffic is generated
            for u, v in self.G.edges():
                 self.G[u][v]['traffic'] = 0

    def build_graph(self,):
        df_maze = generate_df_maze(
                self.max_row,
                self.max_col,
                self.maze_lp,
            )
        G = nx.Graph()
        for _, row in df_maze.iterrows():
            r, c = int(row['row']), int(row['col'])
            G.add_node((r, c))
            if row['E'] == 1:
                G.add_edge((r, c), (r, c+1), traffic=0)
            if row['W'] == 1:
                G.add_edge((r, c), (r, c - 1), traffic=0)
            if row['N'] == 1:
                G.add_edge((r, c), (r - 1, c), traffic=0)
            if row['S'] == 1:
                G.add_edge((r, c), (r + 1,c), traffic=0)

        all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for node in G.nodes:
            wall_distances = []
            for dr, dc in directions:
                steps = 0
                current = node
                while steps < 2:
                    next_node = (current[0] + dr, current[1] + dc)
                    if not G.has_edge(current, next_node):
                        break
                    steps += 1
                    current = next_node
                wall_distances.append(steps)
            G.nodes[node]['wall_distance'] = wall_distances

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

        # Maze State
        self.G, self.all_shortest_path_lengths = self.build_graph() # Rebuild graph with new maze
        for u, v in self.G.edges():
            self.G[u][v]['traffic'] = 0 # Reset traffic on graph edges
        self.red_roads = []
        self.yellow_roads = []
        self.traffic_centeroids = []
        if self.traffic_b:
            self.fill_traffic_centroids() # Generate initial traffic if enabled

        # UGV States
        self.ugv_states = [UGV(ugv_id=i, base_position=self.warehouse_pos, cell_dist=self.cell_size, max_range=self.max_ugv_range,drain_rate=self.drain_rate, 
                       max_speed=self.ugv_speed, traffic_delay_factor= self.traffic_delay_factor, G=self.G) for i in range(self.num_ugvs)]
        self.task_list = random.choices(list(self.G.nodes), k=self.num_tasks) # Randomly select tasks from graph nodes
        self.distrubute_tasks() # Distribute tasks to UGVs

        # Metrics
        self.num_tasks_completed = 0
        self.task_completion_times = []
        self.total_ev_distance = 0.0
        self.total_energy_consumed = 0.0

        obs = self._get_observation()
        info = self.analyzer()

        return obs, info
    
    def _get_nb_traffic(self, position):
        """
        Get the traffic density in the North, East, South, and West directions
        from the given UGV's position using the traffic_centeroids as Gaussian centers.
        Returns a list of four discrete traffic levels (0, 1, 2) corresponding to [N, E, S, W].
        """
        # Define neighbor cell positions relative to current position.
        neighbor_positions = (
            (position[0] - 1, position[1]),
            (position[0], position[1] + 1),
            (position[0] + 1, position[1]),
            (position[0], position[1] - 1)
        )
        
        traffic_levels = [self.G.edges.get((position, neighbor), {}).get('traffic', 0) for neighbor in neighbor_positions]
        return traffic_levels
    
    def _get_observation(self):
        steps2dest = []
        nb_traffic = []
        ugv_status = []

        for ugv_id, ugv in enumerate(self.ugv_states):
            ugv_pos = ugv.position
            task_pos = ugv.active_task

            # Steps to destination in each direction
            steps2dest_ = []
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1), (0,0)]:
                r,c = ugv_pos[0] + dr, ugv_pos[1] + dc
                # if r > 0 and r <= self.max_row and c > 0 and c <= self.max_col:
                if (dr,dc)==(0,0) or (ugv_pos,(r,c)) in self.G.edges:
                    steps2dest_.append(self.all_shortest_path_lengths[(r,c)][task_pos])
                else:
                    steps2dest_.append(self.max_row * self.max_col+1)
            steps2dest.append(steps2dest_)

            # Neighbor Traffic
            nb_traffic.append(self._get_nb_traffic(ugv_pos))

            ugv_status.append(ugv.status)

        steps2dest_flat = np.array(steps2dest, dtype=np.float32).flatten()
        # steps2dest_flat /= (self.max_row * self.max_col+1)  # Normalize steps to destination
        nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()
        ugv_status_flat = np.array(ugv_status, dtype=np.int32).flatten()
        obs_dict = {
            'steps2dest': steps2dest_flat,
            'nb_traffic': nb_traffic_flat,
            'ugv_status': ugv_status_flat,
        }

        return obs_dict
    
    def analyzer(self):
        """Analyzes the current state and returns an info dictionary."""
        self.total_ev_distance = sum(ugv.distance_traveled for ugv in self.ugv_states)
        self.total_energy_consumed = sum(ugv.energy_consumed for ugv in self.ugv_states)
        task_completion_time_diffs = [self.task_completion_times[i] - self.task_completion_times[i-1] for i in range(1, len(self.task_completion_times))]
        avg_task_completion_time = np.mean(task_completion_time_diffs) if task_completion_time_diffs else 0.0

        active_tasks = [ugv.active_task for ugv in self.ugv_states]
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
            if ugv.position == self.warehouse_pos:
                ugv.recharge()

    def task_handler(self):
        truncated = False
        for ugv_id, ugv in enumerate(self.ugv_states):
            if ugv.local_time == self.time_elapsed:
                ugv.status = 1
                if ugv.position == ugv.active_task:
                    ugv.task_completed = True
                    truncated = True
                    if ugv.active_task == ugv.task_list[0]:
                        ugv.task_list.pop(0)
                        self.num_tasks_completed += 1
                        self.task_completion_times.append(self.time_elapsed)
                    dist2task = self.all_shortest_path_lengths[ugv.position][ugv.task_list[0]]*self.cell_size
                    dist2wh = self.all_shortest_path_lengths[ugv.task_list[0]][self.warehouse_pos]*self.cell_size
                    if ugv.current_range >= dist2task+dist2wh:
                        ugv.active_task = ugv.task_list[0]
                        ugv.charging_status = 0
                    else:
                        ugv.active_task = self.warehouse_pos
                        ugv.charging_status = 1
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
                # --- Progress-Based Shaping Reward ---
                # Reward for moving closer to the nearest task
                bonus_factor = 5.0 
                current_distance = self.all_shortest_path_lengths[ugv.position][ugv.active_task]
                prev_distance = self.all_shortest_path_lengths[ugv.prev_position][ugv.active_task]
                reward += bonus_factor * (prev_distance - current_distance)
                
                # --- Task Completion Reward ---
                task_completion_reward = 50.0
                if ugv.task_completed:
                    reward += task_completion_reward

                # --- Time Penalty ---
                # Penalize based on the time taken for the action
                time_penalty_factor = 5.0 # Adjust this factor based on desired behavior
                reward -= time_penalty_factor * ugv.move_time/self.cell_size

                # --- Invalid Move Penalty ---
                invalid_move_penalty = 40.0
                if ugv.act_response == 0:
                    reward -= invalid_move_penalty

                # If UGV is out of range, apply a penalty
                out_of_range_penalty = 20.0
                if ugv.current_range < nx.shortest_path_length(self.G, ugv.position, self.warehouse_pos)*self.cell_size:
                    reward -= out_of_range_penalty

        return reward


    def _check_termination_condition(self):
        no_fuel_flag = True
        for ugv in self.ugv_states:
            if ugv.current_range >= self.cell_size:
                no_fuel_flag = False
                break
        if self.time_elapsed >= self.max_time or no_fuel_flag:
            self.terminated = True
        else:
            self.terminated = False

    def close(self):
        """Cleans up pygame resources."""
        if self.screen is not None:
            print("Closing Pygame window.")
            pygame.display.quit()
            pygame.quit()
            self.screen = None # Mark as closed

