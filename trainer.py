import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import time # Added time for potential delay
import logging

import envs
from agents import get_agent

def lmd_trainer(config, policy_dir):
    """
    Function to train the drone simulation model based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing simulation parameters.
        policy_dir (str): Directory where the policy and model will be saved.
    """
    env = gym.make('SumoTrainEnv-v0', config=config)
    
    agent = get_agent(env, config['agent'], 0)

    agent.learn(config['training'], policy_dir,
                   log_path=os.path.join(policy_dir, "train_log.txt"),)
    env.close()