from gymnasium.envs.registration import register
from envs.lmd_sumo import SumoLMDEnv
from envs.train_sumo import SumoTrainEnv

register(
    id='SumoLMDEnv-v0',
    entry_point=SumoLMDEnv,
    max_episode_steps=50000,  # or any suitable number
)

register(
    id='SumoTrainEnv-v0',
    entry_point=SumoTrainEnv,
    max_episode_steps=50000,  # or any suitable number
)
