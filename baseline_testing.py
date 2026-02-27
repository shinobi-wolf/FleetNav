'''
We compare performance of mTSP vs EVRP
'''

import random
import time
from math import floor
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import networkx as nx
from world_1.planner import mtsp_planner, evrp_planner
from pyamaze import maze
import tkinter as tk
import random
import numpy as np

# Monkey-patch the 'zoomed' state to 'normal'
tk.Tk.state = lambda self, s=None: self.wm_state('normal' if s == 'zoomed' else s)

INF = 1000000

# ------------------ CONFIGURATION ------------------

NUM_EVS_RANGE       = list(range(3, 11))       # e.g., 4 to 20
NUM_TASKS_RANGE     = list(range(20, 31))        # e.g., 20 to 100
EV_RANGE_RANGE      = range(3000, 4501, 100)  # e.g., 500 to 3000 in steps of 10

# Maze parameters (randomized each epoch)
MAZE_SIZE_OPTIONS         = [35]#list(range(35, 51))
MAZE_LOOP_PERCENT_OPTIONS = list(range(50, 91, 5))

EPOCHS_PER_COMBO = 50

last_ending_params = [INF, INF]

# Fixed static window size
WINDOW_SIZE = 15

OUTPUT_FILE = "runs/iros_26/results_mtsp_v_evrp_reachable.csv"

# def generate_df_maze(rows=20, cols=20, lp=80):
#     """
#     Pure-Python headless maze generator using the same recursive-backtracker
#     algorithm as pyamaze (no Tkinter required).

#     Returns a DataFrame with columns ['  cell  ', 'E', 'W', 'N', 'S'].
#     Each value is 1 (open passage) or 0 (wall).
#     """
#     # Initialise all walls closed
#     maze_map = {}
#     grid = []
#     for r in range(1, rows + 1):
#         for c in range(1, cols + 1):
#             maze_map[(r, c)] = {'E': 0, 'W': 0, 'N': 0, 'S': 0}
#             grid.append((r, c))
#     grid_set = set(grid)

#     def open_east(r, c):
#         maze_map[(r, c)]['E'] = 1
#         if (r, c + 1) in grid_set:
#             maze_map[(r, c + 1)]['W'] = 1

#     def open_west(r, c):
#         maze_map[(r, c)]['W'] = 1
#         if (r, c - 1) in grid_set:
#             maze_map[(r, c - 1)]['E'] = 1

#     def open_north(r, c):
#         maze_map[(r, c)]['N'] = 1
#         if (r - 1, c) in grid_set:
#             maze_map[(r - 1, c)]['S'] = 1

#     def open_south(r, c):
#         maze_map[(r, c)]['S'] = 1
#         if (r + 1, c) in grid_set:
#             maze_map[(r + 1, c)]['N'] = 1

#     # Recursive backtracker (same as pyamaze's CreateMaze)
#     path = {}
#     _stack = [(1, 1)]
#     _closed = {(1, 1)}
#     x, y = 1, 1
#     while _stack:
#         cell_dirs = []
#         if (x, y + 1) not in _closed and (x, y + 1) in grid_set:
#             cell_dirs.append('E')
#         if (x, y - 1) not in _closed and (x, y - 1) in grid_set:
#             cell_dirs.append('W')
#         if (x + 1, y) not in _closed and (x + 1, y) in grid_set:
#             cell_dirs.append('S')
#         if (x - 1, y) not in _closed and (x - 1, y) in grid_set:
#             cell_dirs.append('N')
#         if cell_dirs:
#             direction = random.choice(cell_dirs)
#             if direction == 'E':
#                 open_east(x, y);  path[(x, y + 1)] = (x, y);  y += 1
#             elif direction == 'W':
#                 open_west(x, y);  path[(x, y - 1)] = (x, y);  y -= 1
#             elif direction == 'N':
#                 open_north(x, y); path[(x - 1, y)] = (x, y);  x -= 1
#             elif direction == 'S':
#                 open_south(x, y); path[(x + 1, y)] = (x, y);  x += 1
#             _closed.add((x, y))
#             _stack.append((x, y))
#         else:
#             x, y = _stack.pop()

#     # Add loops (same loopPercent logic as pyamaze)
#     if lp != 0:
#         def blocked_neighbours(cell):
#             n = []
#             for d, (dr, dc) in [('E', (0, 1)), ('W', (0, -1)), ('N', (-1, 0)), ('S', (1, 0))]:
#                 if maze_map[cell][d] == 0:
#                     nb = (cell[0] + dr, cell[1] + dc)
#                     if nb in grid_set:
#                         n.append(nb)
#             return n

#         def remove_wall_between(c1, c2):
#             if c1[0] == c2[0]:
#                 if c1[1] == c2[1] + 1:
#                     maze_map[c1]['W'] = 1; maze_map[c2]['E'] = 1
#                 else:
#                     maze_map[c1]['E'] = 1; maze_map[c2]['W'] = 1
#             else:
#                 if c1[0] == c2[0] + 1:
#                     maze_map[c1]['N'] = 1; maze_map[c2]['S'] = 1
#                 else:
#                     maze_map[c1]['S'] = 1; maze_map[c2]['N'] = 1

#         # Collect path cells from (rows,cols) back to start
#         pr, pc = rows, cols
#         path_cells = [(pr, pc)]
#         try:
#             while (pr, pc) != (1, 1):
#                 pr, pc = path[(pr, pc)]
#                 path_cells.append((pr, pc))
#         except KeyError:
#             pass
#         not_path_cells = [c for c in grid if c not in set(path_cells)]
#         random.shuffle(path_cells)
#         random.shuffle(not_path_cells)
#         target1 = int(len(path_cells) / 3 * lp / 100)
#         target2 = int(len(not_path_cells) / 3 * lp / 100)
#         for cells, target in [(path_cells, target1), (not_path_cells, target2)]:
#             count, i = 0, 0
#             while count < target and i < len(cells):
#                 nbs = blocked_neighbours(cells[i])
#                 if nbs:
#                     remove_wall_between(random.choice(nbs), cells[i])
#                     count += 1
#                 i += 1

#     # Build DataFrame in pyamaze's format
#     maze_data = []
#     for r in range(1, rows + 1):
#         for c in range(1, cols + 1):
#             walls = maze_map[(r, c)]
#             maze_data.append([f"({r},{c})", walls['E'], walls['W'], walls['N'], walls['S']])

#     return pd.DataFrame(maze_data, columns=["  cell  ", "E", "W", "N", "S"])

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


def build_graph(maze_size, loop_percentage):
    df_maze = generate_df_maze(maze_size, maze_size,loop_percentage)

    if '  cell  ' in df_maze.columns:
        df_maze[['row', 'col']] = (
            df_maze['  cell  '].astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.split(',', expand=True)
            .astype(int)
        )

    G = nx.Graph()
    for _, row in df_maze.iterrows():
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

EDGE_WEIGHT = 50


def evaluate_plan(patrol_tasks, backcost, num_evs, G, warehouse_pos, R, edge_weight):
    """
    Analytically evaluate a planner's output without step-by-step simulation.

    For each EV, walk through its assigned task list in order.  Before
    moving to the next task, check whether the EV has enough range to
    reach that task *and* return to the depot afterwards.  If not, the
    EV first returns to the depot to recharge (counted as an
    evrp_callback), then attempts the task again.

    Returns
    -------
    timesteps : int          – makespan (max total hops across all EVs)
    tasks_finished : int     – number of tasks successfully reached
    ev_distance : float      – total hops traveled by all EVs
    feasibility : bool       – False if any assigned task could not be reached
    evrp_callback : int      – unplanned depot-recharge detours
    """
    total_tasks    = sum(len(pt) for pt in patrol_tasks)
    tasks_finished = 0
    ev_distance    = 0.0
    evrp_callback  = 0
    ev_hops        = [0] * num_evs  # per-EV hop count (for makespan)
    feasible       = True
    task_order     = [[] for _ in range(num_evs)]  # for debugging

    for i in range(num_evs):
        cur     = warehouse_pos
        battery = float(R)

        for j, task in enumerate(patrol_tasks[i]):
            # Distance: current position → task
            try:
                d_to_task = nx.shortest_path_length(G, source=cur, target=task)
            except nx.NetworkXNoPath:
                feasible = False
                continue

            d_back = backcost[i][j]  # hops: task → warehouse

            # Can we reach the task AND return to depot on current battery?
            if (d_to_task + d_back) * edge_weight <= battery:
                # Go directly to the task
                battery     -= d_to_task * edge_weight
                ev_distance += d_to_task
                ev_hops[i]  += d_to_task
                cur          = task
                tasks_finished += 1
                task_order[i].append(task)
            else:
                # Not enough range – try returning to depot first
                try:
                    d_to_depot = nx.shortest_path_length(
                        G, source=cur, target=warehouse_pos
                    )
                except nx.NetworkXNoPath:
                    feasible = False
                    continue

                if d_to_depot * edge_weight <= battery:
                    # Return to depot and recharge
                    ev_distance += d_to_depot
                    ev_hops[i]  += d_to_depot
                    cur          = warehouse_pos
                    battery      = float(R)
                    evrp_callback += 1
                    task_order[i].append(warehouse_pos)

                    # Re-compute distance from depot to task
                    try:
                        d_from_wh = nx.shortest_path_length(
                            G, source=warehouse_pos, target=task
                        )
                    except nx.NetworkXNoPath:
                        feasible = False
                        continue

                    if (d_from_wh + d_back) * edge_weight <= R:
                        battery     -= d_from_wh * edge_weight
                        ev_distance += d_from_wh
                        ev_hops[i]  += d_from_wh
                        cur          = task
                        tasks_finished += 1
                        task_order[i].append(task)
                    else:
                        # Unreachable even from depot with full battery
                        feasible = False
                else:
                    # Stranded – cannot even reach depot; remaining tasks lost
                    feasible = False
                    break

    timesteps   = max(ev_hops) if ev_hops else 0
    feasibility = feasible and (tasks_finished >= total_tasks)
    return timesteps, tasks_finished, ev_distance, feasibility, evrp_callback, task_order


def run_simulation_job(params):
    '''
    1. Create a graph
    2. Populate tasks, EVs, charging stations (depot for now), depot.
    3. Get mTSP and EVRP paths for the same scenario.
    4. Run the simulation for each planner and log results.
    '''
    num_evs   = params['num_evs']
    num_tasks = params['num_tasks']
    ev_range  = params['ev_range']
    maze_size = params['maze_size']
    maze_lp   = params['maze_loop_percentage']
    edge_weight = EDGE_WEIGHT

    G = build_graph(maze_size, maze_lp)
    nodes = list(G.nodes())
    max_rc = max(n[0] for n in nodes)
    max_cc = max(n[1] for n in nodes)
    warehouse_pos = (max_rc // 2, max_cc // 2)

    # Ensure warehouse is actually a node in the graph
    if warehouse_pos not in G:
        warehouse_pos = min(nodes, key=lambda n: abs(n[0] - max_rc//2) + abs(n[1] - max_cc//2))

    # Sample task positions (exclude warehouse)
    task_candidates = []
    for n in nodes:
        if n == warehouse_pos:
            continue
        if nx.shortest_path_length(G, source=warehouse_pos, target=n) * edge_weight * 2 <= ev_range:
            task_candidates.append(n)
    if len(task_candidates) < num_tasks:
        num_tasks = len(task_candidates)
    all_tasks = random.sample(task_candidates, num_tasks)

    patrol_positions = [warehouse_pos] * num_evs
    R = float(ev_range)

    results = []

    # ---- mTSP planner ----
    pt_m, bc_m, at_m, pp_m = mtsp_planner(
        all_tasks, num_evs, list(patrol_positions), G, edge_weight, R
    )
    ts_m, tf_m, ed_m, feas_m, cb_m, task_order_m = evaluate_plan(
        pt_m, bc_m, num_evs, G, warehouse_pos, R, edge_weight
    )
    results.append(
        f"mtsp;{num_evs};{num_tasks};{ev_range};{maze_lp};{maze_size};"
        f"{ts_m};{tf_m};{ed_m:.2f};{feas_m};{cb_m};{task_order_m}\n"
    )

    # ---- EVRP planner ----
    pt_e, bc_e, at_e, pp_e = evrp_planner(
        all_tasks, num_evs, list(patrol_positions), G, edge_weight, R, warehouse_pos
    )
    ts_e, tf_e, ed_e, feas_e, cb_e, task_order_e = evaluate_plan(
        pt_e, bc_e, num_evs, G, warehouse_pos, R, edge_weight
    )
    results.append(
        f"evrp;{num_evs};{num_tasks};{ev_range};{maze_lp};{maze_size};"
        f"{ts_e};{tf_e};{ed_e:.2f};{feas_e};{cb_e};{task_order_e}\n"
    )

    # Write results (append, file-level lock via single write call)
    with open(OUTPUT_FILE, 'a') as f:
        f.writelines(results)

def main():
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write("planner;num_evs;num_tasks;ev_range;maze_loop_percentage;maze_size;"
                "timesteps;num_tasks_finished;ev_distance;feasibility;evrp_callback;task_order\n")

    jobs = []
    # for num_evs in NUM_EVS_RANGE:
    for loop_per in MAZE_LOOP_PERCENT_OPTIONS:
        for ev_range in EV_RANGE_RANGE:
            for num_tasks in NUM_TASKS_RANGE:
                job_params = {
                    'num_evs': 4,
                    'num_tasks': num_tasks,
                    'ev_range': ev_range,
                    'maze_size': 35,
                    'maze_loop_percentage': loop_per,
                }
                jobs.append(job_params)

    total_jobs = len(jobs)
    completed_jobs = 0
    start_time = time.time()

    # Start a ProcessPoolExecutor with a fixed number of workers
    executor = ProcessPoolExecutor(max_workers=WINDOW_SIZE)
    futures = {}

    # Submit an initial batch of tasks up to the fixed window size
    for _ in range(min(WINDOW_SIZE, total_jobs)):
        job_params = jobs.pop()
        fut = executor.submit(run_simulation_job, job_params)
        futures[fut] = job_params

    # Main loop: as tasks complete, submit new ones until all are processed
    while futures:
        for completed_future in as_completed(futures):
            completed_jobs += 1
            del futures[completed_future]

            # Calculate progress and ETA
            elapsed = time.time() - start_time
            progress = completed_jobs / total_jobs
            avg_time = elapsed / completed_jobs
            remaining = total_jobs - completed_jobs
            eta_sec = remaining * avg_time

            bar_len = 40
            filled_len = int(progress * bar_len)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)

            hrs = floor(eta_sec / 3600)
            mins = floor((eta_sec % 3600) / 60)
            secs = floor(eta_sec % 60)
            eta_str = f"{hrs}h {mins}m {secs}s"

            print(f"\r[{bar}] {progress * 100:.1f}%  ({completed_jobs}/{total_jobs})  ETA: {eta_str}", end="")

            # Submit a new job if available
            if jobs:
                next_job_params = jobs.pop()
                fut = executor.submit(run_simulation_job, next_job_params)
                futures[fut] = next_job_params

    total_time = time.time() - start_time
    print(f"\nAll {total_jobs} jobs done in {total_time:.2f} seconds.")
    executor.shutdown()

if __name__ == '__main__':
    main()
    # job_params = {
    #                 'num_evs': 5,
    #                 'num_tasks': 30,
    #                 'ev_range': 2000,
    #                 'maze_size': 35,
    #                 'maze_loop_percentage': 60,
    #             }
    # run_simulation_job(job_params)