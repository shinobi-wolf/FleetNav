import numpy as np
import pickle
import os

from dronesim.agents.agent import Agent

class DPAgent(Agent):
    """
    Dynamic Programming Agent for reinforcement learning.
    This agent uses a policy evaluation and improvement approach.
    """

    def __init__(self, env, gamma=0.99, theta=1e-6, model_dir="runs/dp", policy_name=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.G = None

        self.gamma = gamma
        self.theta = theta

        self.states = []
        self.value_function = {}
        self.policy = {}
        
        self.action_list = [0, 1, 2, 3, 4]

        

        self.model_dir = model_dir
        # Check if the model directory exists, if not create it
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(f"{self.model_dir}/policies"):
            os.makedirs(f"{self.model_dir}/policies")
        # if not os.path.exists(self.model_dir/'logs'):
        #     os.makedirs(self.model_dir/'logs')

        self.policy_name = policy_name
        if self.policy_name is not None:
            # Load the policy from the specified file
            with open(f"{self.model_dir}/policies/policy_{self.policy_name}.pkl", 'rb') as f:
                self.policy = pickle.load(f)
            print(f"Policy loaded from {self.policy_name}")
        

    def predict(self, observation):
        """
        Return the action for the (single) agent's current state using our policy.
        If the state is not in the policy, choose a random valid action.
        """
        state = self._observation_to_state(observation)
        # if state not in self.policy:
        #     return np.random.choice(self.action_list)
        return [self.policy[state]]

    def learn(self, iterations=1000):
        """
        Perform policy iteration (policy evaluation + policy improvement)
        until convergence or until max_iterations is reached.
        """
        _, info = self.env.reset()
        self.G = info['graph']
        ugv_poses = list(self.G.nodes())
        task_poses = list(self.G.nodes())
        for ugv in ugv_poses:
            for task in task_poses:
                self.states.append((ugv, task))
        
        # Initialize the value function and policy.
        self.value_function = {s: 0.0 for s in self.states}
        # Start with a random action for each state.
        self.policy = {s: np.random.choice(self.action_list) for s in self.states}

        print("Starting Planning with DP...")
        for i in range(iterations):
            print("Iteration: ", i)
            # print("Current Policy: ")
            # for row in range(1,6):
            #     tmp = ""
            #     for col in range(1,6):
            #         s = ((row, col), (1, 1))
            #         tmp += str(self.policy[s]) + " "
            #     print( tmp)
                    

            # 1) Policy Evaluation
            self._policy_evaluation()

            # print("Value Function: ")
            # for row in range(1,6):
            #     tmp = ""
            #     for col in range(1,6):
            #         s = ((row, col), (1, 1))
            #         tmp += str(self.value_function[s]) + " "
            #     print( tmp)

            # 2) Policy Improvement
            policy_stable = True
            for s in self.states:
                old_action = self.policy[s]
                self.policy[s] = self._best_action_for_state(s)
                if self.policy[s] != old_action:
                    policy_stable = False

            if policy_stable:
                break
        # Create a file name based on a random char string
        save_file_name = f"policy_{os.urandom(4).hex()}.pkl"
        with open(f'{self.model_dir}/policies/{save_file_name}', 'wb') as f:
            pickle.dump(self.policy, f)
        print(f"Policy saved to {self.model_dir}/policies/{save_file_name}")
        
    def _policy_evaluation(self):
        """
        Iteratively evaluate V(s) for the current policy until it converges.
        """
        while True:
            delta = 0.0
            for s in self.states:
                old_value = self.value_function[s]
                a = self.policy[s]
                new_value = self._compute_state_value(s, a)
                self.value_function[s] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < self.theta:
                break

    def _compute_state_value(self, state, action):
        """
        For a given state and action, compute the value:
        V(s) = R(s,a) + gamma * V(s')
        since transitions are deterministic. If the action is invalid,
        we remain in the same state with an extra penalty.
        """
        next_state, reward = self._transition(state, action)
        return reward + self.gamma * self.value_function[next_state]

    def _transition(self, state, action):
        """
        Return (next_state, reward) for taking `action` in `state`.
        - Step penalty is -1
        - Invalid move penalty is -20
        - If valid move, we move to that neighbor in self.G
        """
        ugv,task = state
        (r, c) = ugv
        reward = -1.0  # Base step penalty

        if action == 0:      # Up
            candidate = (r - 1, c)
        elif action == 1:    # Right
            candidate = (r, c + 1)
        elif action == 2:    # Down
            candidate = (r + 1, c)
        elif action == 3:    # Left
            candidate = (r, c - 1)
        else:                # Stay
            candidate = (r, c)

        # Check connectivity in self.G. If not connected, remain in place, add penalty.
        if candidate == ugv or (candidate in self.G.nodes and self.G.has_edge(ugv, candidate)):
            if candidate == task:
                reward += 10.0
            next_state = (candidate,task)
        else:
            next_state = state
            reward -= 10.0  # penalty for invalid move

        return next_state, reward

    def _best_action_for_state(self, state):
        """
        For policy improvement, choose the action that maximizes:
        R(s,a) + gamma * V(s')
        among all possible actions.
        """
        best_a = None
        best_value = float('-inf')
        for a in self.action_list:
            next_s, reward = self._transition(state, a)
            val = reward + self.gamma * self.value_function[next_s]
            if val > best_value:
                best_value = val
                best_a = a
        return best_a

    def _observation_to_state(self, observation):
        """
        Convert the environment observation to a single (row, col) state.
        For demonstration, let's assume we track the first UGV's position.
        """
        ugv = tuple(map(int, observation['ugv_positions']))
        task = tuple(map(int, observation['task_positions']))
        state = (ugv, task)
        return state