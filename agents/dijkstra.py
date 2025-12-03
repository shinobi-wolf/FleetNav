import numpy as np
import networkx as nx
from agents.agent import Agent

class DijkstraAgent(Agent):
    """
    Dijkstra's Agent that finds shortest paths to tasks.
    Does not learn - simply uses Dijkstra's algorithm for pathfinding.
    """
    
    def __init__(self, agent_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        
    def predict(self, observation, *args, **kwargs):
        best_dir = observation['best_dir']
        return int(best_dir)
        
    def learn(self):
        """
        This agent doesn't learn - it just uses Dijkstra's algorithm
        """
        pass
        
    def _observation_to_state(self, observation):
        """
        Not used by this agent, but implemented to satisfy abstract method
        """
        pass