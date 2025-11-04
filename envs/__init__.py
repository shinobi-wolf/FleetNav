from gymnasium.envs.registration import register
from envs.ral import LMDEnv
from envs.ral_sumo import SumoLMDEnv

register(
    id='LMDEnv-v0',
    entry_point=LMDEnv,
    max_episode_steps=50000,  # or any suitable number
)

register(
    id='SumoLMDEnv-v0',
    entry_point=SumoLMDEnv,
    max_episode_steps=50000,  # or any suitable number
)
