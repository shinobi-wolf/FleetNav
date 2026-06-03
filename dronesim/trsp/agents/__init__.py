from dronesim.trsp.agents.sarsa import SARSAAgent
from dronesim.trsp.agents.dijkstra import DijkstraAgent
from dronesim.trsp.agents.dqn import DQNAgent


def get_agent(config, agent_id, **kwargs):
    """Return the appropriate agent for the given config['agent'] block."""
    algo = config['algo']
    if algo == 'dijkstra':
        return DijkstraAgent(agent_id)
    if algo == 'sarsa':
        return SARSAAgent(config, **kwargs)
    if algo == 'dqn':
        return DQNAgent(config, **kwargs)
    raise ValueError(f"Unknown algo: {algo!r}")

