import numpy as np
import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPONN(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.num_actions = env.single_action_space.n
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 16)),
            nn.ReLU(),
            layer_init(nn.Linear(16, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class PPOWrapper(gym.Wrapper):
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
        env = PPOWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk

class PPOAgent():
    def __init__(self, config):
        self.config = config
        tmp = gym.make("SumoTRSPEnv-v0", config=config)
        self.num_sg_pairs = tmp.unwrapped.num_sg_pairs
        self.env = gym.vector.SyncVectorEnv([make_env(config)])
        policy_name = config['agent'].get("policy", None)
        if policy_name:
            policy_path = f"runs/{policy_name}/ppo.cleanrl_model"
            self.agent = PPONN(self.env).to('cuda')
            self.agent.load_state_dict(torch.load(policy_path, map_location='cuda'))
            self.agent.eval()
            print(self.agent)

    def obs_converter(self, obs):
        print(obs)
        steps2dest = obs['steps2dest']
        sg_pair_idx = obs['sg_pair_idx']
        flat_obs = np.concatenate((steps2dest, np.eye(self.num_sg_pairs)[sg_pair_idx]), axis=0)
        return flat_obs

    def learn(self, policy_path):
        writer = SummaryWriter(policy_path)
        policy_path += "/ppo.cleanrl_model"
        device = 'cuda'
        config = self.config['training']['ppo']
        
        random.seed(47)
        np.random.seed(47)
        torch.manual_seed(47)
        torch.backends.cudnn.deterministic = True

        assert isinstance(self.env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = PPONN(self.env).to('cuda')
        optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'],eps=1e-5)
        config['batch_size'] = int(config['num_steps'])
        config['minibatch_size'] = int(config['batch_size'] // config['num_minibatches'])
        config['num_iterations'] = config['total_timesteps'] // config['batch_size']

        # ALGO Logic: Storage setup
        obs = torch.zeros((config['num_steps'],) + self.env.single_observation_space.shape).to(device)
        actions = torch.zeros((config['num_steps'],) + self.env.single_action_space.shape).to(device)
        logprobs = torch.zeros((config['num_steps'],)).to(device)
        rewards = torch.zeros((config['num_steps'],)).to(device)
        dones = torch.zeros((config['num_steps'],)).to(device)
        values = torch.zeros((config['num_steps'],)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env.reset(seed=47)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(()).to(device)

        for iteration in range(1, config['num_iterations'] + 1):
            print(f"Starting iteration {iteration}/{config['num_iterations']}")
            # Annealing the rate if instructed to do so.
            if config['anneal_lr']:
                frac = 1.0 - (iteration - 1.0) / config['num_iterations']
                lrnow = frac * config['learning_rate']
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, config['num_steps']):
                global_step += 1
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config['num_steps'])):
                    if t == config['num_steps'] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(config['batch_size'])
            clipfracs = []
            for epoch in range(config['update_epochs']):
                np.random.shuffle(b_inds)
                for start in range(0, config['batch_size'], config['minibatch_size']):
                    end = start + config['minibatch_size']
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > config['clip_coef']).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if config['norm_adv']:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config['clip_vloss']:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -config['clip_coef'],
                            config['clip_coef'],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config['ent_coef'] * entropy_loss + v_loss * config['vf_coef']

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
                    optimizer.step()

                if config['target_kl'] is not None and approx_kl > config['target_kl']:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            torch.save(agent.state_dict(), policy_path)
            print(f"model saved to {policy_path}")

        self.env.close()
        writer.close()

    @torch.no_grad()
    def predict(self, obs, G=None, idx2node=None, node2idx=None, deterministic=True):
        """
        Predict one discrete action for a single observation.

        deterministic=True  -> greedy argmax over policy logits
        deterministic=False -> sample from the Categorical policy (stochastic)
        """
        flat = self.obs_converter(obs)

        # Ensure shape: (1, obs_dim)
        x = torch.tensor(flat, device="cuda", dtype=torch.float32).unsqueeze(0)

        logits = self.agent.actor(x)  # shape: (1, n_actions)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())

        return action
