import numpy as np
import pickle
import os
import random
import networkx as nx
from scipy.stats import rankdata

from agents.agent import Agent

class DynaQ(Agent):
    """
    Dyna-Q Learning Agent.
    
    This off-policy TD control agent learns a Q-table via the Q-learning update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
      
    Additionally, it maintains a one-step environment model M(s,a) -> (s', r)
    and performs `planning_steps` simulated updates per real step by sampling
    from the learned model (Sutton & Barto, 2018).
    """
    def __init__(self, env, config, policy_path=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.cell_size = config['maze']['cell_size']
        self.max_range = config['fleet']['ugv']['range']
        self.max_cell_range = int(self.max_range/self.cell_size)
        self.max_rows = config['maze']['maze_size']
        self.max_cols = self.max_rows

        # Supported action spaces (Discrete or MultiDiscrete with one dimension)
        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        # Q-table and simple tabular model
        self.Q = {}
        # Model maps (state, action_idx) -> (next_state, reward)
        self.model = {}

        if policy_path is not None and os.path.exists(policy_path):
            with open(policy_path, 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Policy loaded from {policy_path}")

    # ---------- Helpers ----------
    def _observation_to_state(self, observation):
        """
        Match SARSA's state representation: a flat tuple of
        steps2dest (len=5) followed by nb_traffic (len=4).
        """
        steps2dest = observation['steps2dest'].tolist()
        nb_traffic = observation['nb_traffic'].tolist()
        obs = tuple(steps2dest + nb_traffic)
        return obs

    def _ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))

    def _action_index(self, action):
        # Convert env action (potentially [int]) into a plain int index
        if isinstance(action, (list, tuple, np.ndarray)):
            return int(action[0])
        return int(action)

    def choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection (returns a list to match env API,
        consistent with SARSA implementation).
        """
        self._ensure_state(state)
        if random.random() < epsilon:
            return [random.choice(self.action_list)]
        else:
            return [int(np.argmax(self.Q[state]))]

    # ---------- Public API ----------
    def predict(self, observation):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        """
        state = self._observation_to_state(observation)
        self._ensure_state(state)
        return int(np.argmax(self.Q[state]))

    def learn(self, training_config, policy_path=None, log_path=None):
        """
        Run Dyna-Q learning over multiple episodes. Each episode starts with env.reset()
        and runs until done or a max number of steps is reached. After every real step,
        perform `planning_steps` Q-learning updates from the learned model.
        """
        # Resolve where to save the policy
        if policy_path is None:
            policy_path_file = 'policy.pkl'
        else:
            # Mirror SARSA's convention of writing to a subfile named policy.pkl
            policy_path_file = os.path.join(policy_path, 'policy.pkl')

        # Pull hyperparameters
        cfg = training_config.get('dyna_q', training_config.get('dynaq', training_config))
        num_episodes = cfg.get('num_episodes')
        gamma = cfg.get('gamma')
        alpha = cfg.get('alpha')
        epsilon = cfg.get('epsilon')
        epsilon_decay = cfg.get('epsilon_decay')
        planning_steps = int(cfg.get('planning_steps', cfg.get('n_planning', 10)))

        results = []

        obs, info = self.env.reset()
        state = self._observation_to_state(obs)
        self._ensure_state(state)
        action = self.choose_action(state, epsilon)
        total_reward = 0.0
        episode_idx = 0

        while episode_idx < num_episodes:
            next_obs, reward, done, trunc, info = self.env.step(action)
            next_state = self._observation_to_state(next_obs)
            self._ensure_state(next_state)

            a_idx = self._action_index(action)

            # --- Real experience Q-learning update ---
            if not trunc:
                td_target = reward + gamma * np.max(self.Q[next_state])
            else:
                td_target = reward
                # Episode truncated by env; finalize logging/eps here
                print(f"Episode {episode_idx+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}")
                results.append((total_reward, epsilon))
                episode_idx += 1
                epsilon *= epsilon_decay
                total_reward = 0.0
                if episode_idx % 1000 == 0:
                    with open(policy_path_file, 'wb') as f:
                        pickle.dump(self.Q, f)

            td_error = td_target - self.Q[state][a_idx]
            self.Q[state][a_idx] += alpha * td_error

            # --- Update tabular model with the observed transition ---
            self.model[(state, a_idx)] = (next_state, reward)

            # --- Planning updates: sample from model ---
            if self.model and planning_steps > 0:
                for _ in range(planning_steps):
                    (s_p, a_p), (s_p_next, r_p) = random.choice(list(self.model.items()))
                    self._ensure_state(s_p)
                    self._ensure_state(s_p_next)
                    target_p = r_p + gamma * np.max(self.Q[s_p_next])
                    self.Q[s_p][a_p] += alpha * (target_p - self.Q[s_p][a_p])

            # --- Episode management ---
            if done:
                if not trunc:
                    print(f"Episode {episode_idx+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}")
                    results.append((total_reward, epsilon))
                    episode_idx += 1
                    epsilon *= epsilon_decay
                    if episode_idx % 1000 == 0:
                        with open(policy_path_file, 'wb') as f:
                            pickle.dump(self.Q, f)
                obs, info = self.env.reset()
                state = self._observation_to_state(obs)
                self._ensure_state(state)
                action = self.choose_action(state, epsilon)
                total_reward = 0.0
            else:
                state = next_state
                action = self.choose_action(state, epsilon)
                total_reward += reward

        # --- Logging & persistence ---
        if log_path is not None:
            with open(log_path, 'a') as f:
                f.write("Episode, Total Reward, Epsilon\n")
                i = 0
                for total_reward, eps_val in results:
                    f.write(f"{i+1},{total_reward:.2f}, {eps_val:.4f}\n")
                    i += 1
        with open(policy_path_file, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Policy saved to {policy_path_file}")
