from dronesim.trsp.agents.sarsa import SARSAAgent
from dronesim.trsp.agents.dijkstra import DijkstraAgent
# from dronesim.trsp.agents.dqn import DQNAgent
from dronesim.trsp.agents.ppo import PPOAgent

def get_agent(env, config, agent_id, **kwargs):
    """
    Returns the appropriate agent based on the configuration.
    :param config: Configuration dictionary.
    :param env: The environment instance.
    :return: An agent instance or a list of agents for multi-agent environments.
    """
    algo = config['algo']
    if algo == 'dijkstra':
        return DijkstraAgent(agent_id)
    
    policy = config.get('policy', None)
    policy_path = None
    if policy:
        policy_path = f"runs/{policy}/policy.pkl"
    if algo == 'sarsa':
        return SARSAAgent(env, policy_path=policy_path)
    elif algo == 'dqn':
        return DQNAgent(env, policy_path=policy_path, **kwargs)
    elif algo == 'ppo':
        return PPOAgent(config)
        
