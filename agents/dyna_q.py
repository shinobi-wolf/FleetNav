import numpy as np
import pickle
import os
import random
import networkx as nx
from scipy.stats import rankdata

from agents.agent import Agent
class DynaQ(Agent):
    """
    SARSA (State-Action-Reward-State-Action) Learning Agent.
    
    This on-policy TD control agent learns a Q-table via the SARSA update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      
    It uses the environment's reward function.
    """
    def __init__(self, env, config, policy_path=None, *args, **kwargs):
        self.env = env
        self.config = config
        self.policy_path = policy_path


    def predict(self, observation):
        raise NotImplementedError  
   

    def learn(self, training_config, policy_path=None, log_path=None):
        raise NotImplementedError