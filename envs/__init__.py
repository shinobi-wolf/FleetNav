from gymnasium.envs.registration import register
from envs.lmd_sumo import SumoLMDEnv
from envs.trsp_sumo import SumoTRSPEnv

register(
    id='SumoLMDEnv-v0',
    entry_point=SumoLMDEnv,
    max_episode_steps=50000,  # or any suitable number
)

register(
    id='SumoTRSPEnv-v0',
    entry_point=SumoTRSPEnv,
    max_episode_steps=50000,  # or any suitable number
)
