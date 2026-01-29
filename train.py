import argparse
import yaml
import os
import numpy as np
import time # Added time for potential delay
import logging
import copy

import gymnasium as gym

import envs
# from agents import get_agent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

def lmd_trainer(config, policy_dir, resume=False):
    """
    Function to train the drone simulation model based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing simulation parameters.
        policy_dir (str): Directory where the policy and model will be saved.
    """
    
    agent = PPOAgent(config)

    agent.learn(policy_dir,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('-c', '--config', type=str,
                        help='Configuration file path (train: optional, eval/run: optional overrides)')
    parser.add_argument('-p', '--policy', type=str,
                        help="Policy name (required for --eval and --run), e.g. '0d318b44'")
    args = parser.parse_args()

    train_config_path = args.config if args.config else 'config/train.yaml'
    resume = False
    if not args.policy:
        train_config = yaml.safe_load(open(train_config_path, 'r'))

        policy_name = os.urandom(4).hex()
        policy_dir = os.path.join(f"runs/{policy_name}")
        os.makedirs(policy_dir, exist_ok=True)

        config_file = os.path.join(policy_dir, "config.yaml")
        inf_config = copy.deepcopy(train_config)
        inf_config['agent']['policy'] = policy_name
        inf_config['simulation']['render_mode'] = 'human'  # Set render mode for inference
        inf_config.pop('training', None)  # Remove training section for inference config
        with open(config_file, 'w') as f:
            yaml.dump(inf_config, f)
    else:
        policy_name = args.policy
        policy_dir = os.path.join(f"runs/{policy_name}")
        resume = True

    start_time = time.time()
    lmd_trainer(train_config, policy_dir, resume=resume)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")