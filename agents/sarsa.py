import numpy as np
import pickle
import os
import random
import networkx as nx
from scipy.stats import rankdata
from scipy.stats import rankdata
from collections import defaultdict

from agents.agent import Agent
class SARSAAgent(Agent):
    """
    SARSA (State-Action-Reward-State-Action) Learning Agent.
    
    This on-policy TD control agent learns a Q-table via the SARSA update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      
    It uses the environment's reward function.
    """
    def __init__(self, env, policy_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = {}
        self.env = env

        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        if policy_path is not None:
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
        best_dir = int(observation['best_dir'])
        return (graph_node, target_node)
        # return (graph_node, best_dir)
        # nb_traffic = observation['nb_traffic'].tolist()
        # return (graph_node, target_node, *nb_traffic)

    def predict(self, observation):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        Also returns the number of neighbors at the current node.
        """
        G = self.env.unwrapped.G
        state = self._observation_to_state(observation)
        # print("State:", state)
        if state not in self.Q:
            # print("State not in Q-table, selecting random action.")
            return random.randint(0, 3)
        curr_node = sorted(list(G.nodes))[int(state[0])]
        num_neib = len(list(G.neighbors(curr_node)))
        q = self.Q[state][:num_neib]
        return int(np.argmax(q))
    
    def choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        If the state is not in the Q-table, initialize its Q-values to zeros.
        """
        sn = state[0]
        num_nbs = len(list(self.G.neighbors(sorted(list(self.G.nodes))[int(sn)])))
        q = self.Q[state][:num_nbs]
        if random.random() < epsilon:
            return random.choice(self.action_list[:num_nbs])
        else:
            return int(np.argmax(q))

    def learn(self, training_config, policy_dir=None, log_path=None):
        """
        Run SARSA learning over multiple episodes. Each episode starts with an env.reset()
        and runs until done or a max number of steps is reached.
        """

        policy_path = policy_dir + '/policy.pkl'
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
        state = self._observation_to_state(obs)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
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
                self.Q[next_state] = np.zeros(len(self.action_list))
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
                    self.Q[state] = np.zeros(len(self.action_list))
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