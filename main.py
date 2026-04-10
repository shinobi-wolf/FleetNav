import argparse
import yaml
import numpy as np
import time # Added time for potential delay
import logging

import gymnasium as gym

import envs
from agents import get_agent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.sarsa import SARSAAgent

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

def lmd_simulator(config):
    policy_name = config['agent']["policy"]
    policy_dir = f"runs/{policy_name}/"
    log_file = f"{policy_dir}/server_log.txt"
    render_mode = config['simulation']['render_mode']
    open(log_file, "w").close()
    setup_logging(render_mode, log_file=log_file)
    env = gym.make("SumoTRSPEnv-v0", config=config)

    if config['agent']['algo'] == 'dqn':
        agent = DQNAgent(config)
    elif config['agent']['algo'] == 'ppo':
        agent = PPOAgent(config)
    elif config['agent']['algo'] == 'sarsa':
        agent = SARSAAgent(config)
    else:
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
        print("Observation:", obs)
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

def main():
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")

    parser.add_argument('-p', '--policy', type=str,
                        help="Policy name (required for --eval and --run), e.g. '0d318b44'")

    args = parser.parse_args()

    # eval or run: policy is mandatory
    if not args.policy:
        parser.error('--policy (-p) is required for --eval and --run')

    policy = args.policy
    # config can override default run config path
    config_path = f"runs/{policy}/config.yaml"
    config = yaml.safe_load(open(config_path, 'r'))

    metrics = lmd_simulator(config)
    print("Metrics:", metrics)


if __name__ == '__main__':
    main()