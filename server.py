import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import time # Added time for potential delay
import logging

import envs
from agents import get_agent

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
    steps2dest = observation['steps2dest'].reshape(num_agents, 5)
    # nb_traffic = observation['nb_traffic'].reshape(num_agents, 4)
    state_info = observation['state_info'].reshape(num_agents, 2)

    all_obs = []
    for i in range(num_agents):
        obs = {
            'ugv_status': ugv_status[i],
            'steps2dest': steps2dest[i],
            # 'nb_traffic': nb_traffic[i],
            'state_info': state_info[i],
        }
        all_obs.append(obs)
    return all_obs

def lmd_simulator(config):
    policy_name = config["policy"]
    policy_dir = f"runs/{policy_name}/"
    log_file = f"{policy_dir}/server_log.txt"
    render_mode = config['simulation']['render_mode']
    open(log_file, "w").close()
    setup_logging(render_mode, log_file=log_file)
    env = gym.make("SumoLMDEnv-v0", config=config)

    num_agents = config['fleet']['ugv']['count']
    agents = get_agent(config, env)

    logging.info("Starting Simulation")
    observation, world_state = env.reset(seed=42)
    env.render()

    acts = ["Up", "Right", "Down", "Left", "Stay"]
    terminated = False
    total_reward = 0.0
    step_idx = 0
    while not terminated:
        action_list = []
        all_obs = extract_observations(observation, num_agents)
        for i, obs in enumerate(all_obs):
            agent = agents[i]
            if not obs['ugv_status']:
                logging.info(f"UGV {i} is not active. Skipping.")
                action_list.append(4)
                continue

            # Get action from policy

            if step_idx % 1==0:
                action = agent.predict(obs)
            else:
                steps2dest = obs['steps2dest']
                action = np.argmin(steps2dest)
            action_list.append(action)

        observation, reward, terminated, truncated, world_state = env.step(action_list)
        total_reward += reward
        step_idx += 1

        # Log action and results
        logging.info(f"Observation: {observation}") # Observation can be large, maybe omit from console
        logging.info(f"Reward: {reward}")
        logging.info(f"Terminated: {terminated}, Truncated: {truncated}")
        for ugv_id in range(num_agents):
            logging.info(f"UGV {ugv_id} Info...")
            action_str = acts[action_list[ugv_id]] # Format action list for logging
            logging.info(f"Action taken: {action_str}")
            
        env.render()

        logging.info("-------------------------------------------\n")

    # Log final metrics (use the last world_state)
    num_tasks_completed = world_state["num_tasks_completed"]
    ev_distance_traveled = world_state["total_ev_distance"]
    total_time_taken = world_state["time_elapsed"]
    total_energy_consumed = world_state["total_energy_consumed"]
    avg_task_completion_time = world_state["avg_task_completion_time"]

    logging.info("\nSimulation completed.")
    logging.info(f"Termination reason: {'Terminated' if terminated else 'Truncated' if truncated else 'Unknown'}")
    logging.info(f"Final frame: {step_idx}")
    logging.info(f"Total reward: {total_reward}")
    logging.info(f"Tasks completed: {num_tasks_completed}")
    logging.info(f"EV distance traveled: {ev_distance_traveled}")
    logging.info(f"Total time taken: {total_time_taken}")
    logging.info(f"Total energy consumed: {total_energy_consumed}")

    env.close() # Ensure pygame resources are cleaned up
    return {
        'frames': step_idx,
        'total_reward': total_reward,
        'tasks_completed': num_tasks_completed,
        'distance_traveled': ev_distance_traveled,
        'time_taken': total_time_taken,
        'energy_consumed': total_energy_consumed,
        'avg_task_completion_time': avg_task_completion_time,
    }