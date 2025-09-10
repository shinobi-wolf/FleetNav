from gymnasium.envs.registration import register
from envs.ral import LMDEnv
from envs.ral_pp import PPEnv

register(
    id='LMDEnv-v0',
    entry_point=LMDEnv,
    max_episode_steps=50000,  # or any suitable number
)

register(
    id='PPEnv-v0',
    entry_point=PPEnv,
    max_episode_steps=50000,  # or any suitable number
)
