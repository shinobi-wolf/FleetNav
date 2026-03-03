import numpy as np
import pandas as pd
import networkx as nx
import random
import math
from pyamaze import maze
import tkinter as tk
from world_1.planner import evrp_planner, mtsp_planner, mtsp_planner_recorded, evrp_planner_recorded

# Monkey-patch the 'zoomed' state to 'normal'
tk.Tk.state = lambda self, s=None: self.wm_state('normal' if s == 'zoomed' else s)
# Helper distance function
def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_df_maze(rows=20, cols=20, lp=80):
    # Create a Pyamaze maze
    random.seed(42)
    np.random.seed(42)
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
    return df

class World:
    def __init__(self, config):
        """
        Initializes the World with a maze loaded from CSV and basic parameters.
        This replicates the core EV (truck) logic from the old code, but omits drone logic.
        """

        self.df_maze = generate_df_maze(
            config["world"]["maze_size"],
            config["world"]["maze_size"],
            config["world"]["maze_loop_percentage"]
        )

        if '  cell  ' in self.df_maze.columns:
            self.df_maze[['row', 'col']] = (
                self.df_maze['  cell  '].astype(str)
                .str.replace(r'[()]', '', regex=True)
                .str.split(',', expand=True)
                .astype(int)
            )

        self.max_row = self.df_maze['row'].max()
        self.max_col = self.df_maze['col'].max()
        self.maze_loop_percentage = config["world"]["maze_loop_percentage"]
        self.maze_size = config["world"]["maze_size"]
        self.G = self._build_graph()
        self.algo = config["world"].get("algo", "mtsp")

        self.num_tasks = config["world"]["num_tasks"]
        self.num_patrols = config['world']['num_evs']
        self.R = config["ev"]["range"]
        self.R_D = [self.R] * self.num_patrols
        self.R_P = [i/self.R for i in self.R_D]

        self.num_uavs = config["world"]["num_uavs_per_bs"]
        self.num_base_stations = config["world"]["num_base_stations"]
        self.B = config["uav"]["range"]
        self.drone_speed = config["uav"]["speed"]

        bs_df = self.df_maze.sample(self.num_base_stations, random_state=47)[['row', 'col']]
        bs_list = []
        for t in bs_df.values.tolist():
            bs_list.append( (int(t[0]), int(t[1])) )
        self.base_stations = bs_list

        self.center_row = self.max_row // 2
        self.center_col = self.max_col // 2
        self.warehouse_pos = (int(self.center_row), int(self.center_col))

        self.default_colors = ["red", "blue", "green", "yellow", "magenta", "cyan", "orange", "black"]
        self.bms = config['world']['bms']

        # NEW: Track total distance traveled by EVs and drones
        self.total_ev_distance = 0.0
        self.total_drone_distance = 0.0

        self.world_state = {}

    def _build_graph(self):
        G = nx.Graph()
        for _, row in self.df_maze.iterrows():
            r, c = int(row['row']), int(row['col'])
            G.add_node((r, c))
            if row['E'] == 1:
                G.add_edge((r, c), (r, c + 1))
            if row['W'] == 1:
                G.add_edge((r, c), (r, c - 1))
            if row['N'] == 1:
                G.add_edge((r, c), (r - 1, c))
            if row['S'] == 1:
                G.add_edge((r, c), (r + 1, c))
        return G

    def _update_world_state(self):
        # This dictionary is updated each time so the console can read it
        self.world_state = {
            "num_patrols": self.num_patrols,
            "patrol_positions": self.patrol_positions,
            "patrol_colors": self.patrol_colors,
            "patrol_paths": self.patrol_paths,
            "R_P": self.R_P,
            "active_tasks": self.active_tasks,
            "drone_positions": self.drone_positions,
            "drone_status": self.drone_status,
            "completed_tasks": self.completed_tasks,
            "tasks_completed_flag": self.tasks_completed_flag,
            # NEW: Add the distance metrics to world_state
            "ev_distance_traveled": self.total_ev_distance,
            "drone_distance_traveled": self.total_drone_distance
        }

    def initialize(self, edge_weight):
        self.edge_weight = edge_weight
        task_candidates = []
        nodes = list(self.G.nodes())
        for n in nodes:
            if n == self.warehouse_pos:
                continue
            if nx.shortest_path_length(self.G, source=self.warehouse_pos, target=n) * edge_weight * 2 <= self.R:
                task_candidates.append(n)
        if len(task_candidates) < self.num_tasks:
            self.num_tasks = len(task_candidates)
        self.all_tasks = random.sample(task_candidates, self.num_tasks)

        self.completed_tasks = 0
        self.patrol_positions = [self.warehouse_pos for _ in range(self.num_patrols)]
        self.patrol_colors = [i for i in range(self.num_patrols)]
        self.R_D = [self.R] * self.num_patrols
        self.R_P = [i/self.R for i in self.R_D]

        # NEW: Reset distances on init
        self.total_ev_distance = 0.0
        self.total_drone_distance = 0.0

        # Task assignment & sequencing (delegated to planner)
        if self.algo == "mtsp":
            self.patrol_tasks, self.backcost, self.active_tasks, self.patrol_paths = mtsp_planner(
                    self.all_tasks, self.num_patrols, self.patrol_positions,
                    self.G, edge_weight, self.R
                )
        else:
            self.patrol_tasks, self.backcost, self.active_tasks, self.patrol_paths = evrp_planner(
                self.all_tasks, self.num_patrols, self.patrol_positions,
                self.G, edge_weight, self.R, self.warehouse_pos
            )
        # self.all_tasks, self.patrol_tasks, self.backcost, self.active_tasks, self.patrol_paths = mtsp_planner_recorded(
        #         self.num_patrols, self.num_tasks, self.R, self.maze_loop_percentage, self.maze_size, self.G, self.warehouse_pos,
        #         edge_weight)
        # self.all_tasks, self.patrol_tasks, self.backcost, self.active_tasks, self.patrol_paths = evrp_planner_recorded(
        #         self.num_patrols, self.num_tasks, self.R, self.maze_loop_percentage, self.maze_size, self.G, self.warehouse_pos,
        #         edge_weight)

        # If bms is enabled, remove warehouse from tasks
        if self.bms:
            for i in range(len(self.patrol_tasks)):
                if self.warehouse_pos in self.patrol_tasks[i]:
                    idx = self.patrol_tasks[i].index(self.warehouse_pos)
                    self.patrol_tasks[i].pop(idx)
                    self.backcost[i].pop(idx)      

        self.drone_positions = []
        self.drone_bases = []
        self.drone_status = []
        self.drone_battery = []
        self.drone_targets = []
        for i in range(self.num_base_stations):
            for _ in range(self.num_uavs):
                x,y = self.base_stations[i]
                self.drone_positions.append([x, y])
                self.drone_bases.append(i)
                self.drone_status.append(0)
                self.drone_battery.append(self.B)
                self.drone_targets.append(-1)

        self.recharge_request = [0 for _ in range(self.num_patrols)]
        self.tasks_completed_flag = False
        self.world_state["tasks_completed_flag"] = self.tasks_completed_flag

        # self._update_world_state()
        # print("-----------------------------------------\n\n")
        # print("Clusters", self.patrol_tasks)
        # self.print_world_state()
        # print("-----------------------------------------\n\n")

    def simulate(self, timesteps=1):
        for _ in range(timesteps):
            self.lmd_simulate()
            if self.bms:
                self.bms_simulate()  # If you want drones to move, uncomment
            if not self.tasks_completed_flag:
                check = True
                for i in self.patrol_paths:
                    if len(i) != 0:
                        check = False
                        break
                if check:
                    self.tasks_completed_flag = True

            self.world_state["tasks_completed_flag"] = self.tasks_completed_flag

            # print("-----------------------------------------\n\n")
            # print("Clusters", self.patrol_tasks)
            # self.print_world_state()
            # print("-----------------------------------------\n\n")

    def bms_simulate(self):
        DRONE_SPEED = self.drone_speed

        # 1) Assign idle drones to EVs that request recharge
        for ev_idx in range(self.num_patrols):
            if self.recharge_request[ev_idx] == 1 and (ev_idx not in self.drone_targets):
                ev_pos = self.patrol_positions[ev_idx]
                chosen_drone = None
                best_dist = float('inf')

                for d_idx in range(len(self.drone_positions)):
                    if self.drone_status[d_idx] == 0 and self.drone_battery[d_idx] > 0:
                        drone_pos = self.drone_positions[d_idx]
                        base_idx = self.drone_bases[d_idx]
                        base_pos = self.base_stations[base_idx]
                        dist_to_ev = _distance(drone_pos, ev_pos)
                        dist_ev_to_base = _distance(ev_pos, base_pos)
                        needed_battery = dist_to_ev + dist_ev_to_base
                        if self.drone_battery[d_idx] >= needed_battery:
                            if dist_to_ev < best_dist:
                                best_dist = dist_to_ev
                                chosen_drone = d_idx

                if chosen_drone is not None:
                    self.drone_status[chosen_drone] = 1
                    self.drone_targets[chosen_drone] = ev_idx

                self.recharge_request[ev_idx] = 0

        # 2) Move each drone according to its state
        for d_idx in range(len(self.drone_positions)):
            status = self.drone_status[d_idx]
            if status == 0:
                continue  # Idle at base

            elif status == 1:
                ev_idx = self.drone_targets[d_idx]
                if ev_idx < 0 or ev_idx >= len(self.patrol_positions):
                    self.drone_status[d_idx] = 0
                    self.drone_targets[d_idx] = -1
                    continue

                ev_pos = self.patrol_positions[ev_idx]
                drone_pos = self.drone_positions[d_idx]
                base_idx = self.drone_bases[d_idx]
                base_pos = self.base_stations[base_idx]

                dist_to_ev = _distance(drone_pos, ev_pos)
                dist_ev_to_base = _distance(ev_pos, base_pos)

                # Check battery feasibility
                if self.drone_battery[d_idx] < (dist_to_ev + dist_ev_to_base):
                    self.drone_status[d_idx] = 2
                    self.drone_targets[d_idx] = -1
                    continue

                if dist_to_ev <= 1:
                    # Arrived at EV
                    self.R_D[ev_idx] = self.R
                    self.R_P[ev_idx] = 1
                    self.drone_status[d_idx] = 2
                    self.drone_targets[d_idx] = -1
                    self.recharge_request[ev_idx] = 0
                else:
                    # Move drone toward EV
                    step = min(dist_to_ev, DRONE_SPEED)
                    dx = ev_pos[0] - drone_pos[0]
                    dy = ev_pos[1] - drone_pos[1]
                    norm = math.sqrt(dx*dx + dy*dy)
                    if norm > 0:
                        move_x = (dx / norm) * step
                        move_y = (dy / norm) * step

                        # NEW: Add distance traveled
                        self.total_drone_distance += step

                        self.drone_positions[d_idx][0] += move_x
                        self.drone_positions[d_idx][1] += move_y
                        self.drone_battery[d_idx] -= step
                        if self.drone_battery[d_idx] < 0:
                            self.drone_battery[d_idx] = 0
                            self.drone_status[d_idx] = 2
                            self.drone_targets[d_idx] = -1

            elif status == 2:
                # Returning to base
                base_idx = self.drone_bases[d_idx]
                base_pos = self.base_stations[base_idx]
                drone_pos = self.drone_positions[d_idx]
                dist_to_base = _distance(drone_pos, base_pos)

                if dist_to_base < 1e-3:
                    self.drone_positions[d_idx] = [base_pos[0], base_pos[1]]
                    self.drone_battery[d_idx] = self.B
                    self.drone_status[d_idx] = 0
                    self.drone_targets[d_idx] = -1
                else:
                    step = min(dist_to_base, DRONE_SPEED)
                    dx = base_pos[0] - drone_pos[0]
                    dy = base_pos[1] - drone_pos[1]
                    norm = math.sqrt(dx*dx + dy*dy)
                    if norm > 0:
                        move_x = (dx / norm) * step
                        move_y = (dy / norm) * step

                        # NEW: Add distance traveled
                        self.total_drone_distance += step

                        self.drone_positions[d_idx][0] += move_x
                        self.drone_positions[d_idx][1] += move_y
                        self.drone_battery[d_idx] -= step
                        if self.drone_battery[d_idx] < 0:
                            self.drone_battery[d_idx] = 0

        self._update_world_state()

    def lmd_simulate(self):
        edge_weight = self.edge_weight
        for i in range(self.num_patrols):
            if not self.patrol_paths[i] or self.R_D[i] <= 0 or self.recharge_request[i] == 1:
                continue

            # Keep old position for distance calc
            old_pos = self.patrol_positions[i]

            next_node = self.patrol_paths[i].pop(0)  # BFS next cell
            self.patrol_positions[i] = next_node

            # NEW: If you want each BFS hop to count as '1' distance:
            # self.total_ev_distance += 1
            #
            # If you want Euclidean distance between cells:
            distance_moved = _distance(old_pos, next_node)
            self.total_ev_distance += distance_moved

            # Battery usage
            self.R_D[i] -= edge_weight
            if self.R_D[i] < 0:
                self.R_D[i] = 0

            self.R_P[i] = self.R_D[i] / self.R

            # If the path is empty, arrived at target
            if len(self.patrol_paths[i]) == 0:
                final_task = self.active_tasks[i]
                if final_task is not None:
                    if final_task in self.patrol_tasks[i]:
                        self.completed_tasks += 1
                        final_task_idx = self.patrol_tasks[i].index(final_task)
                        self.patrol_tasks[i].remove(final_task)
                        self.backcost[i].pop(final_task_idx)
                        # if random.random() < 0.9:
                        #     self.recharge_request[i] = 1
                    elif final_task == self.warehouse_pos:
                        self.R_D[i] = self.R
                        self.R_P[i] = 1

                    self.active_tasks[i] = None

                    # Find new task
                    closest_dist = float('inf')
                    closest_task = None
                    closest_task_idx = None
                    current_pos = self.patrol_positions[i]
                    if self.patrol_tasks[i]:
                        task = self.patrol_tasks[i][0]
                        dist = nx.shortest_path_length(self.G, source=current_pos, target=(task[0], task[1]))
                        closest_dist = dist
                        closest_task = (task[0], task[1])
                        closest_task_idx = 0

                    if self.bms:
                        idle_drone_flag = False
                        for d_idx in range(len(self.drone_positions)):
                            if self.drone_status[d_idx] == 0 and self.drone_battery[d_idx] > 0:
                                self.recharge_request[i] = 1
                                idle_drone_flag = True
                                break
                        if closest_task is not None and ((closest_dist*edge_weight)+(self.backcost[i][closest_task_idx]*edge_weight) <= self.R_D[i] or idle_drone_flag):
                            self.patrol_paths[i] = nx.shortest_path(self.G, source=current_pos, target=closest_task)[1:]
                            self.active_tasks[i] = closest_task
                            continue
                    else:
                        if closest_task is not None and (closest_dist*edge_weight)+(self.backcost[i][closest_task_idx]*edge_weight) <= self.R_D[i]:
                            self.patrol_paths[i] = nx.shortest_path(self.G, source=current_pos, target=closest_task)[1:]
                            self.active_tasks[i] = closest_task
                            continue
                    
                    # Otherwise try warehouse
                    try:
                        dist_to_wh = nx.shortest_path_length(self.G, source=current_pos, target=self.warehouse_pos)
                        if dist_to_wh * edge_weight <= self.R_D[i]:
                            self.patrol_paths[i] = nx.shortest_path(self.G, source=current_pos, target=self.warehouse_pos)[1:]
                            self.active_tasks[i] = self.warehouse_pos
                        else:
                            self.patrol_paths[i] = []
                    except nx.NetworkXNoPath:
                        self.patrol_paths[i] = []

        self._update_world_state()

    def get_world_state(self):
        return self.world_state

    def print_world_state(self):
        print("\n===== WORLD STATE =====")
        print(f"Number of Patrols (EVs): {self.world_state['num_patrols']}")
        print(f"Tasks completed flag: {self.world_state['tasks_completed_flag']}")

        remaining_tasks = sum(len(tasks) for tasks in self.patrol_tasks)
        print(f"Total Tasks Completed: {self.completed_tasks}")
        print(f"Total Tasks Left: {remaining_tasks}")

        # Show total distances traveled
        print(f"EV Distance Traveled: {self.world_state['ev_distance_traveled']:.2f}")
        print(f"Drone Distance Traveled: {self.world_state['drone_distance_traveled']:.2f}")

        requesting_evs = [i + 1 for i, req in enumerate(self.recharge_request) if req == 1]
        if requesting_evs:
            print(f"Recharge Requests: EV(s) {requesting_evs}")
        else:
            print("No active recharge requests.")

        print("\n-- Patrol Positions and Status --")
        for i, pos in enumerate(self.world_state['patrol_positions']):
            battery_percentage = self.world_state['R_P'][i] * 100
            active_task = self.world_state['active_tasks'][i] if self.world_state['active_tasks'][i] else "None"
            is_requesting = " (Requesting Recharge)" if self.recharge_request[i] == 1 else ""
            print(
                f"  EV {i+1}: Position = {pos}, Battery = {battery_percentage:.2f}%, "
                f"Active Task = {active_task}{is_requesting}"
            )

        print("\n-- Patrol Paths --")
        for i, path in enumerate(self.world_state['patrol_paths']):
            path_str = " -> ".join([f"({x[0]},{x[1]})" for x in path]) if path else "No Path"
            print(f"  EV {i+1}: Path = {path_str}")

        print("\n-- Drone Positions and Status --")
        for d_idx, (pos, status, battery) in enumerate(zip(
            self.world_state['drone_positions'],
            self.world_state['drone_status'],
            self.drone_battery
        )):
            if status == 0:
                status_str = "Idle"
            elif status == 1:
                status_str = "Traveling to EV"
            else:
                status_str = "Returning to Base"

            assigned_ev_idx = self.drone_targets[d_idx]
            assigned_ev_str = "None"
            dist_str = ""
            if assigned_ev_idx != -1:
                assigned_ev_str = f"EV {assigned_ev_idx+1}"
                if status == 1:
                    ev_pos = self.world_state['patrol_positions'][assigned_ev_idx]
                    dist_to_ev = _distance(pos, ev_pos)
                    dist_str = f", Dist to EV: {dist_to_ev:.2f}"

            print(
                f"  Drone {d_idx+1}: Position = {pos}, Status = {status_str}, "
                f"Assigned EV = {assigned_ev_str}, Battery = {battery:.2f}/{self.B}{dist_str}"
            )

        print("======================\n")

