import random
import time
from math import floor
from concurrent.futures import ProcessPoolExecutor, as_completed

from server import startup  # Your existing function that runs one simulation

INF = 1000000

# ------------------ CONFIGURATION ------------------

NUM_EVS_RANGE       = list(range(3, 11))       # e.g., 4 to 20
NUM_TASKS_RANGE     = list(range(20, 41))        # e.g., 20 to 100
EV_RANGE_RANGE      = range(1500, 3001, 100)  # e.g., 500 to 3000 in steps of 10

# Maze parameters (randomized each epoch)
MAZE_SIZE_OPTIONS         = list(range(35, 51))
MAZE_LOOP_PERCENT_OPTIONS = list(range(50, 91, 5))

EPOCHS_PER_COMBO = 50

last_ending_params = [INF, INF]

# Fixed static window size
WINDOW_SIZE = 7

OUTPUT_FILE = f"runs/scenario_2/results_ev_1.csv"
with open(OUTPUT_FILE, 'w') as f:
    f.write("num_evs,num_tasks,ev_range,maze_loop_percentage,maze_size,bms,timesteps,num_tasks_finished,ev_distance,drone_distance\n")

# ------------------ JOB WRAPPER ------------------

def run_simulation_job(params):
    """
    Each job's parameters include:
      - Fixed parameters: num_evs, num_tasks, ev_range
      - Randomized maze parameters: maze_size, maze_loop_percentage
      - And the 'bms' flag (True/False)
    Calls `startup(config, OUTPUT_FILE)` once.
    """
    config = {
        "world": {
            "maze_size": params['maze_size'],
            "maze_loop_percentage": params['maze_loop_percentage'],
            "num_evs": params['num_evs'],
            "num_tasks": params['num_tasks'],
            "num_uavs_per_bs": 2,
            "num_base_stations": 4,
            "bms": params['bms']
        },
        "ev": {
            "range": params['ev_range']
        },
        "uav": {
            "range": 2000,
            "speed": 3
        }
    }
    # Run one simulation; startup writes results to OUTPUT_FILE
    startup(config, OUTPUT_FILE)
    return True

# ------------------ MAIN SCRIPT ------------------

def main():
    # Build a list of jobs. For each combination of num_evs, num_tasks, ev_range,
    # create EPOCHS_PER_COMBO jobs with randomized maze parameters.
    # For each combo, include two jobs: one with bms False and one with bms True.
    '''SCENARIO 1'''
    # jobs = []
    # for evs in NUM_EVS_RANGE:
    #     if evs > last_ending_params[0]:
    #         continue
    #     for tasks in NUM_TASKS_RANGE:
    #         if evs == last_ending_params[0] and tasks > last_ending_params[1]:
    #             continue
    #         for ev_range in EV_RANGE_RANGE:
    #             if evs == last_ending_params[0] and tasks == last_ending_params[1] and ev_range>=last_ending_params[2]:
    #                 continue
    #             for _ in range(EPOCHS_PER_COMBO):
    #                 job_params = {
    #                     'num_evs': evs,
    #                     'num_tasks': tasks,
    #                     'ev_range': ev_range,
    #                     'maze_size': random.choice(MAZE_SIZE_OPTIONS),
    #                     'maze_loop_percentage': random.choice(MAZE_LOOP_PERCENT_OPTIONS),
    #                     'bms': False
    #                 }
    #                 jobs.append(job_params)
    #                 job_params = {
    #                     'num_evs': evs,
    #                     'num_tasks': tasks,
    #                     'ev_range': ev_range,
    #                     'maze_size': random.choice(MAZE_SIZE_OPTIONS),
    #                     'maze_loop_percentage': random.choice(MAZE_LOOP_PERCENT_OPTIONS),
    #                     'bms': True
    #                 }
    #                 jobs.append(job_params)

    '''SCENARIO 2'''
    jobs = []
    for num_evs in NUM_EVS_RANGE:
        if num_evs > last_ending_params[0]:
            continue
        for loop_per in MAZE_LOOP_PERCENT_OPTIONS:
            if num_evs == last_ending_params[0] and loop_per > last_ending_params[1]:
                continue
            for _ in range(EPOCHS_PER_COMBO):
                job_params = {
                    'num_evs': num_evs,
                    'num_tasks': random.choice(NUM_TASKS_RANGE),
                    'ev_range': random.choice(EV_RANGE_RANGE),
                    'maze_size': random.choice(MAZE_SIZE_OPTIONS),
                    'maze_loop_percentage': loop_per,
                    'bms': False
                }
                jobs.append(job_params)
                job_params = {
                    'num_evs': num_evs,
                    'num_tasks': random.choice(NUM_TASKS_RANGE),
                    'ev_range': random.choice(EV_RANGE_RANGE),
                    'maze_size': random.choice(MAZE_SIZE_OPTIONS),
                    'maze_loop_percentage': loop_per,
                    'bms': True
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

