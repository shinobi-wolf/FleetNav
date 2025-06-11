import numpy as np
import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

from agents.agent import Agent

class QNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(obs_shape).prod(), 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.ReLU(),
            nn.Linear(9, action_shape),
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQNAgent(Agent):
    def __init__(self, env, config=None, policy_path=None):
        super().__init__(env, config, policy_path)
        self.env = env
        self.obs_shape = config['training']['dqn']['obs_shape']
        self.action_shape = config['fleet']['ugv']['num_primitives']
        self.epsilon = config['training']['dqn']['pred_epsilon']
        self.model = QNetwork(self.obs_shape, self.action_shape).to('cuda')
        if policy_path:
            self.model.load_state_dict(torch.load(policy_path, map_location='cuda'))
            self.model.eval()
            print(self.model)

    def obs_converter(self, obs):
        assert isinstance(obs, dict), "Observation must be a dictionary"
        steps2dest = obs['steps2dest']
        nb_traffic = obs['nb_traffic']
        new_obs = np.concatenate((steps2dest, nb_traffic), axis=0)
        return new_obs

    def learn(self, config, policy_path, log_path):
        policy_path += "dqn.cleanrl_model"
        writer = SummaryWriter(policy_path)
        device = 'cuda'
        
        random.seed(47)
        np.random.seed(47)
        torch.manual_seed(47)
        torch.backends.cudnn.deterministic = True

        envs = gym.vector.SyncVectorEnv(
            [self.env]
        )

        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        q_network = self.model
        optimizer = optim.Adam(q_network.parameters(), lr=config['training']['learning_rate'])
        target_network = QNetwork(envs).to('cuda')
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(
            config['training']['buffer_size'],
            envs.single_observation_space,
            envs.single_action_space,
            'cuda',
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset()
        for global_step in range(config['training']['total_timesteps']):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(config['training']['start_e'], config['training']['end_e'], config['training']['exploration_fraction'] * config['training']['total_timesteps'], global_step)
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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
            if global_step > config['training']['learning_starts']:
                if global_step % config['training']['train_frequency'] == 0:
                    data = rb.sample(config['training']['batch_size'])
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + config['training']['gamma'] * target_max * (1 - data.dones.flatten())
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
                if global_step % config['training']['target_network_frequency'] == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            config['training']['tau'] * q_network_param.data + (1.0 - config['training']['tau']) * target_network_param.data
                        )

            if terminations.any():
                obs, _ = envs.reset()

        torch.save(q_network.state_dict(), policy_path)
        print(f"model saved to {policy_path}")

        envs.close()
        writer.close()

    def predict(self, obs):
        obs = self.obs_converter(obs)
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            action = int(action[0])
        else:
            q_values = self.model(torch.Tensor(obs).to('cuda'))
            action = torch.argmax(q_values, dim=0).cpu().numpy()
        return action