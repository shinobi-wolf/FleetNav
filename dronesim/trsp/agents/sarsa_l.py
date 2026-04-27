import numpy as np
import pickle
import os
import random
from dronesim.trsp.agents.agent import Agent
import networkx as nx
from scipy.stats import rankdata

class SARSALambdaAgent(Agent):
    """
    SARSA(λ) Learning Agent with eligibility traces.
    
    This on-policy TD control agent extends SARSA with eligibility traces,
    allowing TD errors to propagate to previously visited state-action pairs.
    This enables faster learning, especially in environments with delayed rewards.
    """
    def __init__(self, env, config, gamma=0.99, alpha=0.1, epsilon=0.2, epsilon_decay=0.995,
                 lambda_=0.9, policy_path=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_  # Eligibility trace decay parameter

        self.cell_size = config['world']['cell_size']
        self.max_range = config['ugv']['range']
        self.max_cell_range = int(self.max_range/self.cell_size)
        self.max_rows = config['world']['maze_size']
        self.max_cols = self.max_rows

        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        self.Q = {}
        self.E = {}  # Eligibility traces

        if policy_path is not None and os.path.exists(policy_path):
            try:
                with open(policy_path, 'rb') as f:
                    self.Q = pickle.load(f)
                print(f"Policy loaded from {policy_path}")
            except Exception as e:
                print(f"Error loading policy: {e}")
                self.Q = {}

    def _observation_to_state(self, observation):
        wall_encoding = tuple(observation['wall_encoding'].tolist())
        task_direction = int(observation['task_direction'])
        step2dest = observation['steps2dest']
        step2dest_ranked = rankdata(step2dest, method='min')  # Subtract 1 to make ranks zero-based
        step2dest = tuple(step2dest_ranked.tolist())

        nb_traffic = tuple(observation['nb_traffic'].tolist())
        return (task_direction, step2dest,nb_traffic)

    def predict(self, observation):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        """
        state = self._observation_to_state(observation)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
        return int(np.argmax(self.Q[state]))
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        If the state is not in the Q-table, initialize its Q-values to zeros.
        """
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
            
        if random.random() < self.epsilon:
            return random.choice(self.action_list)
        else:
            return int(np.argmax(self.Q[state]))

    def learn(self, num_episodes=1000, max_steps_per_episode=100, policy_path=None, log_path=None):
        """
        Run SARSA(λ) learning over multiple episodes with eligibility traces.
        """
        results = []
        for episode in range(num_episodes):
            # Reset eligibility traces at the beginning of each episode
            self.E = {}
            
            obs, info = self.env.reset()
            state = self._observation_to_state(obs)
            if state not in self.Q:
                self.Q[state] = np.zeros(len(self.action_list))
                self.E[state] = np.zeros(len(self.action_list))
            else:
                self.E[state] = np.zeros(len(self.action_list))
                
            action = self.choose_action(state)
            total_reward = 0.0

            for step in range(max_steps_per_episode):
                next_obs, reward, done, trunc, info = self.env.step(action)
                next_state = self._observation_to_state(next_obs)
                
                if next_state not in self.Q:
                    self.Q[next_state] = np.zeros(len(self.action_list))
                    self.E[next_state] = np.zeros(len(self.action_list))
                elif next_state not in self.E:
                    self.E[next_state] = np.zeros(len(self.action_list))
                    
                next_action = self.choose_action(next_state)

                # SARSA TD error calculation
                td_target = reward + self.gamma * self.Q[next_state][next_action]
                td_error = td_target - self.Q[state][action]
                
                # Update eligibility trace for current state-action pair
                if state not in self.E:
                    self.E[state] = np.zeros(len(self.action_list))
                self.E[state][action] = 1.0  # Replacing traces
                
                # Update all state-action pairs according to their eligibility
                for s in self.E:
                    for a in range(len(self.action_list)):
                        if self.E[s][a] > 0:
                            self.Q[s][a] += self.alpha * td_error * self.E[s][a]
                            self.E[s][a] *= self.gamma * self.lambda_
                
                state = next_state
                action = next_action
                total_reward += reward

                if done or trunc:
                    break

            self.epsilon *= self.epsilon_decay
            print(f"Episode {episode+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {self.epsilon:.4f}  Steps: {step+1}")
            results.append((total_reward, self.epsilon))

            if policy_path is not None and (episode % 100 == 0 or episode == num_episodes - 1):
                try:
                    with open(policy_path, 'wb') as f:
                        pickle.dump(self.Q, f)
                except Exception as e:
                    print(f"Error saving policy: {e}")

        if log_path is not None:
            try:
                with open(log_path, 'a') as f:
                    # Write header if file is new/empty
                    if os.path.getsize(log_path) == 0:
                        f.write("Episode, Total Reward, Epsilon\n")
                    for i, (total_reward, epsilon) in enumerate(results):
                        f.write(f"{episode - num_episodes + i + 1},{total_reward:.2f},{epsilon:.4f}\n")
            except Exception as e:
                print(f"Error writing log: {e}")
                
        if policy_path is not None:
            with open(policy_path, 'wb') as f:
                pickle.dump(self.Q, f)
            print(f"Final policy saved to {policy_path}")
