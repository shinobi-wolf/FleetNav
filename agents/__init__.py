from agents.sarsa import SARSAAgent
from agents.dijkstra import DijkstraAgent
# from agents.dqn import DQNAgent

def get_agent(env, config, agent_id):
    """
    Returns the appropriate agent based on the configuration.
    :param config: Configuration dictionary.
    :param env: The environment instance.
    :return: An agent instance or a list of agents for multi-agent environments.
    """
    algo = config['algo']
    if algo == 'dijkstra':
        return DijkstraAgent(agent_id)
    
    elif algo == 'sarsa':
        policy = config.get('policy', None)
        policy_path = None
        if policy:
            policy_path = f"runs/{policy}/policy.pkl"
        return SARSAAgent(env, policy_path=policy_path)
        
    # else:
    #     policy_path = None
    #     if not config['simulation']['train']:
    #         policy_name = config['policy']
    #         policy_path = f"runs/{policy_name}/dqn.cleanrl_model"
    #     if algo == 'dqn':
    #         return [DQNAgent(env, config=config, policy_path=policy_path) for _ in range(num_agents)]
        
