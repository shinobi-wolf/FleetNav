from dronesim.trsp.agents.sarsa_psl    import SARSAAgent
from dronesim.trsp.agents.dijkstra_psl import DijkstraAgent
from dronesim.trsp.agents.dsp          import DSPAgent
from dronesim.trsp.agents.dqn_psl      import DQNAgent
from dronesim.trsp.agents.ppo_psl      import PPOAgent
from dronesim.trsp.agents.a2c_psl      import A2CAgent
from dronesim.trsp.agents.dqn_svc      import DQNSVCAgent
from dronesim.trsp.agents.sarsa_svc    import SARSASVCAgent
from dronesim.trsp.agents.a2c_svc      import A2CSVCAgent
from dronesim.trsp.agents.ppo_svc      import PPOSVCAgent


def get_agent(config, agent_id, **kwargs):
    """Return the appropriate agent for the given config['agent'] block."""
    algo = config['algo']
    if algo == 'dijkstra':
        return DijkstraAgent(agent_id)
    if algo == 'dsp':
        return DSPAgent(agent_id, **kwargs)  # expects G_dir, node2idx, idx2node, ugv_speed
    if algo == 'sarsa':
        return SARSAAgent(config, **kwargs)
    if algo == 'sarsa_svc':
        return SARSASVCAgent(config, **kwargs)
    if algo == 'dqn':
        return DQNAgent(config, **kwargs)
    if algo == 'dqn_svc':
        return DQNSVCAgent(config, **kwargs)
    if algo == 'ppo':
        return PPOAgent(config, **kwargs)
    if algo == 'ppo_svc':
        return PPOSVCAgent(config, **kwargs)
    if algo == 'a2c':
        return A2CAgent(config, **kwargs)
    if algo == 'a2c_svc':
        return A2CSVCAgent(config, **kwargs)
    raise ValueError(f"Unknown algo: {algo!r}")

