import time
import csv
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import yaml

# Adjust the import path to match your project structure
from world_1.world import World

# Number of timesteps to iterate unless tasks complete earlier
ITERATIONS = 1000

def startup(config, output_file):
    """
    Runs a console-only simulation (no GUI) for performance logging.
    
    :param config: Dictionary containing all simulation parameters.
    """
    # Create the simulation world
    world = World(config)

    # If world.initialize requires an argument like cell_size, 
    # provide it or remove it if unneeded in your 'World' class
    world.initialize(50)

    # print("Starting simulation in console mode...")
    final_frame = ITERATIONS  # Default to the last iteration if tasks never complete
    num_tasks_completed = None
    
    for frame in range(ITERATIONS):
        # Step the simulation by 1 timestep (or more, if desired)
        world.simulate(timesteps=1)
        
        # Get the latest state; check if tasks have completed
        world_state = world.get_world_state()
        if world_state.get("tasks_completed_flag", False):
            final_frame = frame
            num_tasks_completed = world_state["completed_tasks"]
            ev_distance_traveled = world_state["ev_distance_traveled"]
            drone_distance_traveled = world_state["drone_distance_traveled"]
            break

    bms = 1 if config['world']['bms'] else 0

    with open(output_file, "a") as f:
        f.write(
            f"{config['world']['algo']},"
            f"{config['world']['num_evs']},"
            f"{config['world']['num_tasks']},"
            f"{config['ev']['range']},"
            f"{config['world']['maze_loop_percentage']},"
            f"{config['world']['maze_size']},"
            f"{bms},"
            f"{final_frame},"
            f"{num_tasks_completed},"
            f"{ev_distance_traveled},"
            f"{drone_distance_traveled}\n"
        )

