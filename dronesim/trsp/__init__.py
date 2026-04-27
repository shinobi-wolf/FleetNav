from gymnasium.envs.registration import register

register(
    id='SumoTRSPEnv-v0',
    entry_point='dronesim.trsp.envs.sumo_trsp_env:SumoTRSPEnv',
)
