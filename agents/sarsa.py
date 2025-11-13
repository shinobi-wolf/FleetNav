import numpy as np
import pickle
import os
import random
import networkx as nx
from scipy.stats import rankdata

from agents.agent import Agent
class SARSAAgent(Agent):
    """
    SARSA (State-Action-Reward-State-Action) Learning Agent.
    
    This on-policy TD control agent learns a Q-table via the SARSA update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      
    It uses the environment's reward function.
    """
    def __init__(self, env, config, policy_path=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.cell_size = config['maze']['cell_size']
        self.max_range = config['fleet']['ugv']['range']
        self.max_cell_range = int(self.max_range/self.cell_size)

        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        self.Q = {}

        if policy_path is not None:
            with open(policy_path, 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Policy loaded from {policy_path}")

    def _observation_to_state(self, observation):
        # steps2dest = observation['steps2dest'].tolist()
        # nb_traffic = observation['nb_traffic'].tolist()
        # obs = tuple(steps2dest + nb_traffic)
        # return obs
        state_info = observation['state_info'].tolist()
        obs = tuple(state_info)
        return obs

    def predict(self, observation):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        """
        state = self._observation_to_state(observation)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
        return int(np.argmax(self.Q[state]))
    
    def choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        If the state is not in the Q-table, initialize its Q-values to zeros.
        """
        if random.random() < epsilon:
            return [random.choice(self.action_list)]
        else:
            return [int(np.argmax(self.Q[state]))]

    def learn(self, training_config, policy_path=None, log_path=None):
        """
        Run SARSA learning over multiple episodes. Each episode starts with an env.reset()
        and runs until done or a max number of steps is reached.
        """
        policy_path = policy_path + '/policy.pkl'
        training_config = training_config['sarsa']
        num_episodes = training_config.get('num_episodes')
        gamma = training_config.get('gamma')
        alpha = training_config.get('alpha')
        epsilon = training_config.get('epsilon')
        epsilon_decay = training_config.get('epsilon_decay')

        results = []
        obs, info = self.env.reset()
        state = self._observation_to_state(obs)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
        action = self.choose_action(state, epsilon)
        total_reward = 0.0
        episode_idx = 0
        while episode_idx < num_episodes:
            next_obs, reward, done, trunc, info = self.env.step(action)
            next_state = self._observation_to_state(next_obs)
            if next_state not in self.Q:
                self.Q[next_state] = np.zeros(len(self.action_list))
            next_action = self.choose_action(next_state, epsilon)

            # SARSA update rule:
            if not trunc:
                td_target = reward + gamma * self.Q[next_state][next_action]
            else:
                td_target = reward
                print(f"Episode {episode_idx+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}")
                results.append((total_reward, epsilon))
                episode_idx += 1
                epsilon *= epsilon_decay
                total_reward = 0.0
                if episode_idx%1000 == 0:
                    with open(policy_path, 'wb') as f:
                        pickle.dump(self.Q, f)
                
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += alpha * td_error

            if done:
                if not trunc:
                    print(f"Episode {episode_idx+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}")
                    results.append((total_reward, epsilon))
                    episode_idx += 1
                    epsilon *= epsilon_decay
                    if episode_idx%1000 == 0:
                        with open(policy_path, 'wb') as f:
                            pickle.dump(self.Q, f)
                self.env.close()
                obs, info = self.env.reset()
                state = self._observation_to_state(obs)
                if state not in self.Q:
                    self.Q[state] = np.zeros(len(self.action_list))
                action = self.choose_action(state, epsilon)
                total_reward = 0.0
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