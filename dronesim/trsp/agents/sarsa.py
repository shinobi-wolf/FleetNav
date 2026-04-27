import numpy as np
import pickle
import os
import random
import networkx as nx
# from scipy.stats import rankdata
from collections import defaultdict

import gymnasium as gym

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration


class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) Learning Agent.
    
    This on-policy TD control agent learns a Q-table via the SARSA update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      
    It uses the environment's reward function.
    """
    def __init__(self,config):
        self.config = config
        self.Q = {}
        self.env = gym.make("SumoTRSPEnv-v0", config=config)
        env = self.env

        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        policy_name = config['agent'].get("policy", None)
        if policy_name:
            policy_path = os.path.join(f"runs/trsp/{policy_name}/policy.pkl")
            with open(policy_path, 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Policy loaded from {policy_path}")

        # Metrics / diagnostics (initialized/overwritten in learn())
        self.exploration_steps = 0
        self.exploitation_steps = 0
        self.all_td_errors = []
        self.episode_summaries = []
        self.last_action_was_exploratory = False

    def _observation_to_state(self, observation):
        graph_node = int(observation['graph_node'])
        target_node = int(observation['target_node'])
        time_slot = int(observation.get('time_slot', 0))
        return (graph_node, target_node, time_slot)

    def predict(self, observation, G=None, idx2node=None, node2idx=None):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        Also returns the number of neighbors at the current node.
        """
        if self.env:
            G = self.env.unwrapped.G
            idx2node = self.env.unwrapped.idx2node
            node2idx = self.env.unwrapped.node2idx
        state = self._observation_to_state(observation)
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
        else:
            # slice to valid actions only; don't mutate stored Q array
            return int(np.argmax(self.Q[state][:num_nbs]))

    def learn(self, policy_dir):
        """
        Run SARSA learning over multiple episodes. Each episode starts with an env.reset()
        and runs until done or a max number of steps is reached.
        """
        training_config = self.config['training']
        policy_path = policy_dir + '/policy.pkl'
        log_path = policy_dir + '/train_log.csv'
        training_config = training_config['sarsa']
        results = []
        state_visits = defaultdict(int)       # counts per state
        sa_visits = defaultdict(int)
        new_states_explored = []
        new_sa_explored = [] 
        new_states_explored_val = 0
        new_sa_explored_val = 0

        num_episodes = training_config.get('num_episodes')
        gamma = training_config.get('gamma')
        alpha = training_config.get('alpha')
        epsilon = training_config.get('epsilon')
        epsilon_decay = training_config.get('epsilon_decay')

        obs, info = self.env.reset()
        self.G = self.env.unwrapped.G
        self.node2idx = self.env.unwrapped.node2idx
        self.idx2node = self.env.unwrapped.idx2node
        state = self._observation_to_state(obs)
        if state not in self.Q:
            self.Q[state] = np.random.uniform(-1e-5, 1e-5, len(self.action_list))
        action = self.choose_action(state, epsilon)
        total_reward = 0.0
        episode_idx = 0
        num_steps = 0
        # self.env.render()
        while episode_idx < num_episodes:
            # print(f"Step {num_steps+1}  Action: {action}")
            next_obs, reward, done, trunc, info = self.env.step(action)
            # print(f"Reward: {reward:.2f}")
            # self.env.render()
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

            # SARSA update rule:
            if not trunc:
                td_target = reward + gamma * self.Q[next_state][next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_error
            else:
                td_target = reward
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_error
                print(f"Episode {episode_idx+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}")
                results.append((total_reward, epsilon))
                episode_idx += 1
                epsilon *= epsilon_decay
                total_reward = 0.0
                reward = 0.0
                new_states_explored.append(new_states_explored_val)
                new_sa_explored.append(new_sa_explored_val)
                new_states_explored_val = 0
                new_sa_explored_val = 0
                if episode_idx%100 == 0:
                    with open(policy_path, 'wb') as f:
                        pickle.dump(self.Q, f)
                    results_dict = {
                        'state_visits': dict(state_visits),
                        'sa_visits': dict(sa_visits),
                        'new_states_explored': new_states_explored,
                        'new_sa_explored': new_sa_explored
                    }
                    with open(f"{policy_dir}/training_results.pkl", 'wb') as f:
                        pickle.dump(results_dict, f)

            if done:
                if not trunc:
                    print(f"Episode {episode_idx+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}")
                    results.append((total_reward, epsilon))
                    episode_idx += 1
                    epsilon *= epsilon_decay
                    new_states_explored.append(new_states_explored_val)
                    new_sa_explored.append(new_sa_explored_val)
                    new_states_explored_val = 0
                    new_sa_explored_val = 0
                    if episode_idx%100 == 0:
                        with open(policy_path, 'wb') as f:
                            pickle.dump(self.Q, f)
                        results_dict = {
                            'state_visits': dict(state_visits),
                            'sa_visits': dict(sa_visits),
                            'new_states_explored': new_states_explored,
                            'new_sa_explored': new_sa_explored
                        }
                        with open(f"{policy_dir}/training_results.pkl", 'wb') as f:
                            pickle.dump(results_dict, f)
                self.env.close()
                obs, info = self.env.reset()
                state = self._observation_to_state(obs)
                if state not in self.Q:
                    self.Q[state] = np.random.uniform(-1e-5, 1e-5, len(self.action_list))
                action = self.choose_action(state, epsilon)
                total_reward = 0.0
                reward = 0.0
            else:
                state = next_state
                action = next_action
                total_reward += reward

        if log_path is not None:
            with open(log_path, 'a') as f:
                f.write("Episode, Total Reward, Epsilon\n")
                i=0
                for total_reward, epsilon in results:
                    f.write(f"{i+1},{total_reward:.2f}, {epsilon:.4f}\n")
                    i += 1
        with open(policy_path, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Policy saved to {policy_path}")