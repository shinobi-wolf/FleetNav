import os
import pickle
import random
from collections import defaultdict

import numpy as np
import gymnasium as gym

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration


class SARSAPEDAgent:
    """SARSA agent for pedestrian (delivery-robot) mode.

    State: (graph_node, target_node, speed_bins) — the neighbour sidewalk speeds
    (density-coupled via the env's Weidmann override) are discretised into the state so
    the tabular policy can react to pedestrian congestion. Bins span [0, 1] because ped
    normalized speeds are capped at free-flow (unlike SVC's [0, 2]).
    Action space: Discrete(max_degree) — neighbor_index mode.
    """

    def __init__(self, config):
        self.config = config
        self.Q = {}
        self.env = gym.make("SumoTRSPEnv-v0", config=config)
        env = self.env

        sarsa_cfg = config.get('training', {}).get('sarsa', {})
        num_speed_bins = sarsa_cfg.get('num_speed_bins', 3)
        # Interior thresholds that evenly partition [0, 1] — ped Weidmann speeds ≤ v_free,
        # so normalized neighbor_speeds live in [~v_min/v_free, 1.0].
        self.speed_thresholds = np.linspace(0.0, 1.0, num_speed_bins + 1)[1:-1]

        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        policy_name = config['agent'].get('policy', None)
        if policy_name:
            policy_path = f"runs/trsp/{policy_name}/policy.pkl"
            with open(policy_path, 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Policy loaded from {policy_path}")

        self.exploration_steps = 0
        self.exploitation_steps = 0
        self.all_td_errors = []
        self.episode_summaries = []

    def _observation_to_state(self, obs):
        gn = int(obs['graph_node'])
        tn = int(obs['target_node'])
        speeds = np.asarray(obs['neighbor_speeds'])
        bins = tuple(np.digitize(speeds, self.speed_thresholds).tolist())
        return (gn, tn, bins)

    def predict(self, obs, G=None, idx2node=None, node2idx=None):
        if self.env:
            G = self.env.unwrapped.G
            idx2node = self.env.unwrapped.idx2node
            node2idx = self.env.unwrapped.node2idx
        state = self._observation_to_state(obs)
        sn = state[0]
        ugv_node = idx2node[int(sn)]
        num_neib = len(list(G.neighbors(ugv_node)))
        if state not in self.Q:
            return random.randint(0, num_neib - 1)
        q = self.Q[state][:num_neib]
        return int(np.argmax(q))

    def choose_action(self, state, epsilon):
        sn = state[0]
        ugv_node = self.idx2node[int(sn)]
        nb_nodes = list(self.G.neighbors(ugv_node))
        num_nbs = len(nb_nodes)
        if state not in self.Q:
            self.Q[state] = np.random.uniform(-1e-5, 1e-5, len(self.action_list))
        if random.random() < epsilon:
            return random.randint(0, num_nbs - 1)
        return int(np.argmax(self.Q[state][:num_nbs]))

    def learn(self, policy_dir):
        training_config = self.config['training']['sarsa']
        policy_path = policy_dir + '/policy.pkl'
        log_path = policy_dir + '/train_log.csv'

        results = []
        state_visits = defaultdict(int)
        sa_visits = defaultdict(int)
        new_states_explored = []
        new_sa_explored = []
        new_states_explored_val = 0
        new_sa_explored_val = 0

        num_episodes  = training_config.get('num_episodes')
        gamma         = training_config.get('gamma')
        alpha         = training_config.get('alpha')
        epsilon       = training_config.get('epsilon')
        epsilon_decay = training_config.get('epsilon_decay')

        obs, info = self.env.reset()
        self.G        = self.env.unwrapped.G
        self.node2idx = self.env.unwrapped.node2idx
        self.idx2node = self.env.unwrapped.idx2node

        state = self._observation_to_state(obs)
        if state not in self.Q:
            self.Q[state] = np.random.uniform(-1e-5, 1e-5, len(self.action_list))
        action = self.choose_action(state, epsilon)

        total_reward = 0.0
        episode_idx  = 0
        num_steps    = 0

        while episode_idx < num_episodes:
            next_obs, reward, done, trunc, info = self.env.step(action)
            num_steps += 1
            next_state = self._observation_to_state(next_obs)
            if next_state not in self.Q:
                self.Q[next_state] = np.random.uniform(-1e-5, 1e-5, len(self.action_list))
                new_states_explored_val += 1
            next_action = self.choose_action(next_state, epsilon)
            state_visits[state] += 1
            sa_visits[(state, action)] += 1
            if sa_visits[(state, action)] == 1:
                new_sa_explored_val += 1

            if not trunc:
                td_target = reward + gamma * self.Q[next_state][next_action]
                td_error  = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_error
            else:
                td_target = reward
                td_error  = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_error
                print(
                    f"Episode {episode_idx + 1}/{num_episodes}  "
                    f"Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}"
                )
                results.append((total_reward, epsilon))
                episode_idx  += 1
                epsilon      *= epsilon_decay
                total_reward  = 0.0
                reward        = 0.0
                new_states_explored.append(new_states_explored_val)
                new_sa_explored.append(new_sa_explored_val)
                new_states_explored_val = 0
                new_sa_explored_val     = 0
                if episode_idx % 100 == 0:
                    with open(policy_path, 'wb') as f:
                        pickle.dump(self.Q, f)
                    results_dict = {
                        'state_visits':        dict(state_visits),
                        'sa_visits':           dict(sa_visits),
                        'new_states_explored': new_states_explored,
                        'new_sa_explored':     new_sa_explored,
                    }
                    with open(f"{policy_dir}/training_results.pkl", 'wb') as f:
                        pickle.dump(results_dict, f)

            if done:
                if not trunc:
                    print(
                        f"Episode {episode_idx + 1}/{num_episodes}  "
                        f"Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}"
                    )
                    results.append((total_reward, epsilon))
                    episode_idx  += 1
                    epsilon      *= epsilon_decay
                    new_states_explored.append(new_states_explored_val)
                    new_sa_explored.append(new_sa_explored_val)
                    new_states_explored_val = 0
                    new_sa_explored_val     = 0
                    if episode_idx % 100 == 0:
                        with open(policy_path, 'wb') as f:
                            pickle.dump(self.Q, f)
                        results_dict = {
                            'state_visits':        dict(state_visits),
                            'sa_visits':           dict(sa_visits),
                            'new_states_explored': new_states_explored,
                            'new_sa_explored':     new_sa_explored,
                        }
                        with open(f"{policy_dir}/training_results.pkl", 'wb') as f:
                            pickle.dump(results_dict, f)
                self.env.close()
                obs, info = self.env.reset()
                state = self._observation_to_state(obs)
                if state not in self.Q:
                    self.Q[state] = np.random.uniform(-1e-5, 1e-5, len(self.action_list))
                action       = self.choose_action(state, epsilon)
                total_reward = 0.0
                reward       = 0.0
            else:
                state  = next_state
                action = next_action
                total_reward += reward

        with open(log_path, 'a') as f:
            f.write("Episode,Total Reward,Epsilon\n")
            for i, (r, e) in enumerate(results):
                f.write(f"{i + 1},{r:.2f},{e:.4f}\n")
        with open(policy_path, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Policy saved to {policy_path}")
