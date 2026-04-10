import pickle
import numpy as np


class Agent:
    def __init__(self, *args, **kwargs):

        self.state = None

        # Initialize policy as a dictionary (can be overwritten or loaded later)
        self.policy = {}  # key: state, value: action

        # Store action space for sampling random actions

    def predict(self, observation):
        """
        Choose an action based on the current policy using epsilon-greedy
        """
        raise NotImplementedError("Subclasses should implement the predict() method")

    def learn(self):
        """
        Placeholder for learning logic to be implemented by subclasses
        """
        raise NotImplementedError("Subclasses should implement the learn() method")

    def load_policy(self, filepath):
        """
        Load a policy from a pickle file
        """
        with open(filepath, 'rb') as f:
            self.policy = pickle.load(f)
        print(f"Policy loaded from {filepath}")

    def _observation_to_state(self, observation):
        """
        Convert observation to a hashable state representation (override if needed)
        """
        raise NotImplementedError("Subclasses should implement the _observation_to_state() method")
