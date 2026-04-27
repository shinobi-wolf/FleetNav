import numpy as np
import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.num_actions = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
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
        self.max_deg = env.unwrapped.num_actions
        self.num_sg_pairs = env.unwrapped.num_sg_pairs
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(self.max_deg + self.num_sg_pairs,), 
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
        sg_pair_idx = obs_dict['sg_pair_idx']
        flat_obs = np.concatenate((steps2dest, np.eye(self.num_sg_pairs)[sg_pair_idx]), axis=0)
        return flat_obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info
    
    def step(self, action):
        # Convert single action to the format expected by the vectorized environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return self._transform_obs(obs), reward, terminated, truncated, info

def make_env(config,):
    def thunk():
        env = gym.make("SumoTRSPEnv-v0", config=config)
        env = DQNWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk

class DQNAgent():
    def __init__(self, config):
        self.config = config
        tmp = gym.make("SumoTRSPEnv-v0", config=config)
        self.num_sg_pairs = tmp.unwrapped.num_sg_pairs
        self.env = gym.vector.SyncVectorEnv(
            [make_env(config)]
        )
        self.model = QNetwork(self.env).to('cuda')
        policy_name = config['agent'].get("policy", None)
        if policy_name:
            policy_path = f"runs/{policy_name}/dqn.cleanrl_model"
            self.model.load_state_dict(torch.load(policy_path, map_location='cuda'))
            self.model.eval()
            print(self.model)

    def obs_converter(self, obs):
        print(obs)
        steps2dest = obs['steps2dest']
        sg_pair_idx = obs['sg_pair_idx']
        flat_obs = np.concatenate((steps2dest, np.eye(self.num_sg_pairs)[sg_pair_idx]), axis=0)
        return flat_obs

    def learn(self, policy_path):
        writer = SummaryWriter(policy_path)
        policy_path += "/dqn.cleanrl_model"
        device = 'cuda'
        config = self.config['training']
        
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

            # if terminations.any():
            #     self.env.close()
            #     obs, _ = self.env.reset()

        torch.save(q_network.state_dict(), policy_path)
        print(f"model saved to {policy_path}")

        self.env.close()
        writer.close()

    def predict(self, obs, G=None, idx2node=None, node2idx=None):
        obs = self.obs_converter(obs)
        if random.random() < 0:
            action = self.env.action_space.sample()
            action = int(action[0])
        else:
            q_values = self.model(torch.Tensor(obs).to('cuda'))
            q_values = q_values.cpu().detach().numpy().flatten()
            action = np.argmax(q_values)
        return action