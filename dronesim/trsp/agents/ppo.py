import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration


class PPOAgent:
    def __init__(self, config):
        raise NotImplementedError("PPOAgent is not yet implemented")

    def predict(self, observation, **kwargs):
        raise NotImplementedError

    def learn(self, policy_dir):
        raise NotImplementedError
