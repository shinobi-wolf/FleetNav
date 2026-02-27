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


def evaluate_plan_with_bms(patrol_tasks, backcost, num_evs, G, warehouse_pos, R, edge_weight):
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

    ugv_curr_pos = [warehouse_pos] * num_evs  # start at warehouse
    ugv_curr_range = [float(R)] * num_evs
    ugv_active_task_idx = [0] * num_evs  # which task in their list they're currently on

    ugv_task_idx = 0
    while tasks_finished < total_tasks:
        
    
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

    # Create UAV charging stations at the warehouse (for BMS)


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
    ts_m, tf_m, ed_m, feas_m, cb_m, task_order_m = evaluate_plan_with_bms(
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
    ts_e, tf_e, ed_e, feas_e, cb_e, task_order_e = evaluate_plan_with_bms(
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
                    'uav':
                        {
                            'range': 3000,
                            'speed': 3
                        },
                    'num_uavs_per_bs': 2,
                    'num_base_stations': 4
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