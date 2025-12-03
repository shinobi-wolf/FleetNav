import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import time # Added time for potential delay
import logging

import envs
from agents import get_agent

acts = ['up', 'right', 'down', 'left', 'stay']

def setup_logging(mode, log_file="simulation_log.txt"):
    """
    Configures logging based on the mode.
    :param mode: 'logging.info' for both console and file logging, 'quiet' for file logging only.
    :param log_file: Path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the base logging level

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    # File handler (always enabled)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (only if mode is 'logging.info')
    if mode != 'quiet':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Adjust console log level
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def extract_observations(observation, num_agents):
    ugv_status = observation['ugv_status'].reshape(num_agents, 1)
    steps2dest = observation['steps2dest'].reshape(num_agents, 4)
    target_node = observation['target_node'].reshape(num_agents, 1)
    graph_node = observation['graph_node'].reshape(num_agents, 1)
    nb_traffic = observation['nb_traffic'].reshape(num_agents, 4)

    all_obs = []
    for i in range(num_agents):
        obs = {
            'ugv_status': ugv_status[i],
            'steps2dest': steps2dest[i],
            'target_node': target_node[i],
            'graph_node': graph_node[i][0],
            'nb_traffic': nb_traffic[i][0],
        }
        all_obs.append(obs)
    return all_obs

def lmd_simulator(config):
    policy_name = config['agent']["policy"]
    policy_dir = f"runs/{policy_name}/"
    log_file = f"{policy_dir}/server_log.txt"
    render_mode = config['simulation']['render_mode']
    open(log_file, "w").close()
    setup_logging(render_mode, log_file=log_file)
    env = gym.make("SumoTRSPEnv-v0", config=config)

    agent = get_agent(env, config['agent'], 0)

    # Enhanced simulation startup logging
    logging.info(f"Config: {policy_dir}/config.yaml")
    
    obs, world_state = env.reset(seed=42)
    
    # Log initial state
    logging.info("Step 0:")
    logging.info(f"UGV: Position={world_state['ugv_position_idx']}, Task={world_state['ugv_target_idx']}, Task Junction={world_state['ugv_target_junction']}")
    
    env.render()

    terminated = False
    total_reward = 0.0
    step_idx = 0
    start_time = time.time()
    
    while not terminated:
        logging.info(f"Step {step_idx+1}:")
        
        action = agent.predict(obs)
        print(f"Agent selected action: {action}")

        # Execute step
        obs, reward, terminated, truncated, world_state = env.step(action)
        total_reward += reward
        print("Reward:", reward)
        
        logging.info(f"UGV: Position={world_state['ugv_position_idx']}, Task={world_state['ugv_target_idx']}, Task Junction={world_state['ugv_target_junction']}")
        
        env.render()
        if truncated:
            step_idx=0
        else:
            step_idx += 1
        
    # Calculate final metrics
    simulation_wall_time = time.time() - start_time
    total_time_taken = world_state.get("time_elapsed", 0)
    
    
    env.close() # Ensure pygame resources are cleaned up
    return {
        'time_taken': total_time_taken,
    }