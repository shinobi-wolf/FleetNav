import numpy as np
import pickle
import os
import random
import networkx as nx
from scipy.stats import rankdata

from agents.agent import Agent

class DynaQ(Agent):
    """
    Dyna-Q agent with per-UGV independent Q-tables and a simple model for planning.
    - Uses epsilon-greedy action selection per UGV
    - Q-learning updates on real transitions
    - Planning updates using stored model samples
    """
    def __init__(self, env, config, policy_path=None, *args, **kwargs):
        self.env = env
        self.config = config or {}
        # Hyperparameters (config overrides if provided)
        self.alpha = self.config.get('alpha', 0.1)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon', 0.1)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.9995)
        self.min_epsilon = self.config.get('min_epsilon', 0.01)
        self.planning_steps = int(self.config.get('planning_steps', 5))
        self.num_episodes = int(self.config.get('episodes', 100))
        self.max_steps = int(self.config.get('max_steps', 1000))

        # Environment-related
        # Expect environment to expose num_ugvs and a MultiDiscrete action_space
        try:
            self.num_ugvs = getattr(self.env, 'num_ugvs')
        except Exception:
            # Fallback: try to infer from action_space length
            if hasattr(self.env.action_space, 'nvec'):
                self.num_ugvs = len(self.env.action_space.nvec)
            else:
                self.num_ugvs = 1

        # Number of discrete actions per UGV (assume identical across UGVs)
        if hasattr(self.env.action_space, 'nvec'):
            self.num_actions = int(self.env.action_space.nvec[0])
        else:
            self.num_actions = int(self.env.action_space.n)

        # Q-tables: one dict per UGV mapping state_key -> np.array of action values
        self.Q = [dict() for _ in range(self.num_ugvs)]
        # Model: one dict per UGV mapping (state_key, action) -> (reward, next_state_key)
        self.model = [dict() for _ in range(self.num_ugvs)]

        # Policy file
        self.policy_path = policy_path
        if self.policy_path and os.path.exists(self.policy_path):
            try:
                self.load_policy(self.policy_path)
            except Exception:
                # Ignore load errors and start fresh
                pass

    def _state_key_for_ugv(self, observation, ugv_id):
        """Convert full observation dict to a compact, hashable state key for a single UGV."""
        # Observation structure expected: dict with keys 'steps2dest', 'nb_traffic', 'ugv_status'
        steps = np.array(observation['steps2dest'], dtype=float)
        nb = np.array(observation['nb_traffic'], dtype=int)
        status = np.array(observation['ugv_status'], dtype=int)

        # slices
        steps_per = 5
        nb_per = 4
        s_slice = steps[ugv_id * steps_per:(ugv_id + 1) * steps_per]
        nb_slice = nb[ugv_id * nb_per:(ugv_id + 1) * nb_per]
        st = int(status[ugv_id])

        # Quantize steps (they are integer-like but may be floats) and clamp
        s_q = tuple(int(np.clip(np.round(s), 0, 9999)) for s in s_slice)
        nb_q = tuple(int(x) for x in nb_slice)
        key = (s_q, nb_q, st)
        return key

    def _observe_to_state_keys(self, observation):
        return [self._state_key_for_ugv(observation, i) for i in range(self.num_ugvs)]

    def predict(self, observation):
        """Return an action array (one action per UGV) using epsilon-greedy policy."""
        state_keys = self._observe_to_state_keys(observation)
        actions = np.zeros(self.num_ugvs, dtype=int)
        for i, sk in enumerate(state_keys):
            # Ensure Q entry exists
            if sk not in self.Q[i]:
                self.Q[i][sk] = np.zeros(self.num_actions, dtype=float)

            if random.random() < self.epsilon:
                actions[i] = random.randrange(self.num_actions)
            else:
                # argmax with tie-breaking
                qvals = self.Q[i][sk]
                max_inds = np.flatnonzero(qvals == qvals.max())
                actions[i] = int(np.random.choice(max_inds))
        return actions

    def learn(self, training_config=None, policy_path=None, log_path=None):
        """Train the agent in the provided environment using Dyna-Q.
        training_config may override self.num_episodes, self.max_steps, etc.
        """
        cfg = training_config or {}
        episodes = int(cfg.get('episodes', self.num_episodes))
        max_steps = int(cfg.get('max_steps', self.max_steps))
        planning_steps = int(cfg.get('planning_steps', self.planning_steps))

        for ep in range(episodes):
            obs, info = self.env.reset()
            state_keys = self._observe_to_state_keys(obs)
            for t in range(max_steps):
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Normalize reward to float
                r = float(reward) if reward is not None else 0.0

                next_state_keys = self._observe_to_state_keys(next_obs)

                # Real experience Q-learning update and model storage
                for i in range(self.num_ugvs):
                    s = state_keys[i]
                    a = int(action[i])
                    s_next = next_state_keys[i]

                    qs = self.Q[i].setdefault(s, np.zeros(self.num_actions, dtype=float))
                    qnext = self.Q[i].setdefault(s_next, np.zeros(self.num_actions, dtype=float))

                    td_target = r + self.gamma * np.max(qnext)
                    qs[a] += self.alpha * (td_target - qs[a])

                    # Update model
                    self.model[i][(s, a)] = (r, s_next)

                # Planning updates
                for _ in range(planning_steps):
                    # Randomly pick a UGV that has model entries
                    ugv_candidates = [i for i in range(self.num_ugvs) if len(self.model[i]) > 0]
                    if not ugv_candidates:
                        break
                    i = random.choice(ugv_candidates)
                    (s_p, a_p), (r_p, s_p_next) = random.choice(list(self.model[i].items()))
                    qs = self.Q[i].setdefault(s_p, np.zeros(self.num_actions, dtype=float))
                    qnext = self.Q[i].setdefault(s_p_next, np.zeros(self.num_actions, dtype=float))
                    td_target = r_p + self.gamma * np.max(qnext)
                    qs[a_p] += self.alpha * (td_target - qs[a_p])

                obs = next_obs
                state_keys = next_state_keys

                if terminated or truncated:
                    break

            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Optionally save policy per-episode
            if policy_path or self.policy_path:
                out_path = policy_path or self.policy_path
                try:
                    self.save_policy(out_path)
                except Exception:
                    pass

        return self.Q

    def save_policy(self, path):
        data = {
            'Q': self.Q,
            'model': self.model,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'num_ugvs': self.num_ugvs,
            'num_actions': self.num_actions,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_policy(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.Q = data.get('Q', self.Q)
        self.model = data.get('model', self.model)
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.num_ugvs = data.get('num_ugvs', self.num_ugvs)
        self.num_actions = data.get('num_actions', self.num_actions)
