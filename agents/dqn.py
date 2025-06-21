import numpy as np
import random
import time
from scipy.stats import rankdata

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

from agents.agent import Agent

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.ReLU(),
            nn.Linear(9, env.single_action_space.n),
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQNWrapper(gym.Wrapper):
    """
    Wrapper for LMDEnv that forces single agent mode and flattens observations.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Define the new flattened observation space
        # For a single agent: 5 values from steps2dest + 4 values from nb_traffic = 9 values total
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(9,),  # 5 (steps2dest) + 4 (nb_traffic)
            dtype=np.float32
        )
        
        # If using MultiDiscrete action space, adapt it for a single agent
        if isinstance(self.env.action_space, spaces.MultiDiscrete):
            self.action_space = spaces.Discrete(self.env.action_space.nvec[0])
        else:
            self.action_space = self.env.action_space
    
    def _transform_obs(self, obs_dict):
        """Convert dictionary observation to flattened array."""
        steps2dest = obs_dict['steps2dest']
        # steps2dest = rankdata(steps2dest, method='ordinal')
        # Normalize steps2dest to a range of 0-1
        steps2dest = (steps2dest - np.min(steps2dest)) / (np.max(steps2dest) - np.min(steps2dest))
        nb_traffic = obs_dict['nb_traffic']
        return np.concatenate((steps2dest, nb_traffic), axis=0).astype(np.float32)
        # return steps_rank.astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info
    
    def step(self, action):
        # Convert single action to the format expected by the vectorized environment
        action_array = np.array([action])
        
        obs, reward, terminated, truncated, info = self.env.step(action_array)
        
        return self._transform_obs(obs), reward, terminated, truncated, info

class DQNAgent(Agent):
    def __init__(self, env, config=None, policy_path=None):
        super().__init__(env, config, policy_path)
        self.env = DQNWrapper(self.env)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.vector.SyncVectorEnv([lambda: self.env])
        self.epsilon = config['training']['dqn']['pred_epsilon']
        self.num_primitives = config['fleet']['ugv']['num_primitives']
        self.model = QNetwork(self.env).to('cuda')
        if policy_path:
            self.model.load_state_dict(torch.load(policy_path, map_location='cuda'))
            self.model.eval()
            print(self.model)

    def obs_converter(self, obs):
        assert isinstance(obs, dict), "Observation must be a dictionary"
        steps2dest = obs['steps2dest']
        # steps2dest = rankdata(steps2dest, method='ordinal')
        steps2dest = (steps2dest - np.min(steps2dest)) / (np.max(steps2dest) - np.min(steps2dest))
        nb_traffic = obs['nb_traffic']
        new_obs = np.concatenate((steps2dest, nb_traffic), axis=0)
        return new_obs

    def learn(self, config, policy_path, log_path):
        writer = SummaryWriter(policy_path)
        policy_path += "/dqn.cleanrl_model"
        device = 'cuda'
        
        random.seed(47)
        np.random.seed(47)
        torch.manual_seed(47)
        torch.backends.cudnn.deterministic = True

        assert isinstance(self.env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        q_network = self.model
        optimizer = optim.Adam(q_network.parameters(), lr=config['dqn']['learning_rate'])
        target_network = QNetwork(self.env).to('cuda')
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(
            config['dqn']['buffer_size'],
            self.env.single_observation_space,
            self.env.single_action_space,
            'cuda',
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for global_step in range(config['dqn']['total_timesteps']):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(config['dqn']['start_e'], config['dqn']['end_e'], config['dqn']['exploration_fraction'] * config['dqn']['total_timesteps'], global_step)
            if random.random() < epsilon:
                actions = np.array([self.env.single_action_space.sample() for _ in range(self.env.num_envs)])
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > config['dqn']['learning_starts']:
                if global_step % config['dqn']['train_frequency'] == 0:
                    data = rb.sample(config['dqn']['batch_size'])
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + config['dqn']['gamma'] * target_max * (1 - data.dones.flatten())
                    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % config['dqn']['target_network_frequency'] == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            config['dqn']['tau'] * q_network_param.data + (1.0 - config['dqn']['tau']) * target_network_param.data
                        )
                    torch.save(q_network.state_dict(), policy_path)

            if terminations.any():
                obs, _ = self.env.reset()

        torch.save(q_network.state_dict(), policy_path)
        print(f"model saved to {policy_path}")

        self.env.close()
        writer.close()

    def predict(self, obs):
        obs = self.obs_converter(obs)
        print("State:", obs)
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            action = int(action[0])
        else:
            q_values = self.model(torch.Tensor(obs).to('cuda'))
            q_values = q_values.cpu().detach().numpy().flatten()
            action = np.argmax(q_values[:self.num_primitives])
        return action