import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.categorical import Categorical

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, num_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, mask: torch.Tensor, action=None):
        logits = self.actor(x)
        logits = logits.masked_fill(~mask, float('-inf'))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPOAgent:
    """PPO agent for TRSP with node-id action space and neighbour masking.

    Observation encoding: one-hot(graph_node) ++ one-hot(target_node) ++ one-hot(time_slot)
    Action space: Discrete(num_nodes) — masked to valid neighbours at each step.
    Training follows CleanRL single-env PPO with GAE advantage estimation.
    """

    def __init__(self, config):
        self.config = config
        ppo_cfg = config.get('training', {}).get('ppo', {})
        self.hidden_dim = ppo_cfg.get('hidden_dim', 64)

        self.env = gym.make('SumoTRSPEnv-v0', config=config)

        self.num_nodes = len(self.env.unwrapped.node2idx)
        self.num_slots = (
            self.env.unwrapped.traffic.K if self.env.unwrapped.traffic.enabled else 1
        )

        self.G = None
        self.node2idx = None
        self.idx2node = None

        self.obs_dim = 2 * self.num_nodes + self.num_slots

        policy_name = config['agent'].get('policy', None)
        saved_data = None
        if policy_name:
            policy_path = f"runs/trsp/{policy_name}/policy.pkl"
            with open(policy_path, 'rb') as f:
                saved_data = pickle.load(f)
            self.hidden_dim = saved_data.get('hidden_dim', self.hidden_dim)

        self.net = ActorCritic(self.obs_dim, self.num_nodes, self.hidden_dim)

        if saved_data is not None:
            self.net.actor.load_state_dict(saved_data['actor_state_dict'])
            self.net.critic.load_state_dict(saved_data['critic_state_dict'])
            print(f"Policy loaded from runs/trsp/{policy_name}/policy.pkl")

    # ------------------------------------------------------------------ helpers

    def _sync_graph(self):
        self.G = self.env.unwrapped.G
        self.node2idx = self.env.unwrapped.node2idx
        self.idx2node = self.env.unwrapped.idx2node

    def _encode_obs(self, obs) -> np.ndarray:
        gn = int(obs['graph_node'])
        tn = int(obs['target_node'])
        ts = int(obs.get('time_slot', 0))
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        vec[gn] = 1.0
        vec[self.num_nodes + tn] = 1.0
        vec[2 * self.num_nodes + ts] = 1.0
        return vec

    def _get_mask(self, node_idx: int) -> torch.Tensor:
        if self.G is None:
            self._sync_graph()
        node = self.idx2node[int(node_idx)]
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        for nb in self.G.neighbors(node):
            mask[self.node2idx[nb]] = True
        return mask

    # ------------------------------------------------------------------ inference

    def predict(self, obs, G=None, idx2node=None, node2idx=None):
        """Return the greedy (mode) node-id action, masked to valid neighbours."""
        if G is not None:
            self.G, self.idx2node, self.node2idx = G, idx2node, node2idx
        elif self.G is None:
            self._sync_graph()

        obs_t = torch.tensor(self._encode_obs(obs), dtype=torch.float32).unsqueeze(0)
        mask = self._get_mask(obs['graph_node']).unsqueeze(0)
        with torch.no_grad():
            logits = self.net.actor(obs_t)
            logits = logits.masked_fill(~mask, float('-inf'))
        return int(logits.argmax(dim=-1).item())

    # ------------------------------------------------------------------ training

    def learn(self, policy_dir: str):
        ppo_cfg = self.config['training']['ppo']
        num_iterations = ppo_cfg['num_iterations']
        num_steps = ppo_cfg['num_steps']
        update_epochs = ppo_cfg['update_epochs']
        num_minibatches = ppo_cfg['num_minibatches']
        minibatch_size = num_steps // num_minibatches
        gamma = ppo_cfg['gamma']
        gae_lambda = ppo_cfg['gae_lambda']
        clip_coef = ppo_cfg['clip_coef']
        ent_coef = ppo_cfg['ent_coef']
        vf_coef = ppo_cfg['vf_coef']
        max_grad_norm = ppo_cfg['max_grad_norm']
        lr = ppo_cfg['lr']
        anneal_lr = ppo_cfg.get('anneal_lr', True)
        save_interval = ppo_cfg.get('save_interval', 50)

        policy_path = os.path.join(policy_dir, 'policy.pkl')
        log_path = os.path.join(policy_dir, 'train_log.csv')

        optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # rollout storage
        obs_buf = np.zeros((num_steps, self.obs_dim), dtype=np.float32)
        actions_buf = np.zeros(num_steps, dtype=np.int64)
        logprobs_buf = np.zeros(num_steps, dtype=np.float32)
        rewards_buf = np.zeros(num_steps, dtype=np.float32)
        dones_buf = np.zeros(num_steps, dtype=np.float32)
        values_buf = np.zeros(num_steps, dtype=np.float32)
        masks_buf = np.zeros((num_steps, self.num_nodes), dtype=np.bool_)

        obs, _ = self.env.reset()
        self._sync_graph()
        next_done = 0.0
        episode_reward = 0.0
        episode_idx = 0
        results = []

        for iteration in range(1, num_iterations + 1):
            if anneal_lr:
                frac = 1.0 - (iteration - 1) / num_iterations
                optimizer.param_groups[0]['lr'] = frac * lr

            # ---- collect rollout ----
            for step in range(num_steps):
                obs_vec = self._encode_obs(obs)
                mask = self._get_mask(obs['graph_node'])

                # dones[step] = whether current state is a fresh episode start
                dones_buf[step] = next_done
                obs_buf[step] = obs_vec
                masks_buf[step] = mask.numpy()

                obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                mask_t = mask.unsqueeze(0)

                with torch.no_grad():
                    action, logprob, _, value = self.net.get_action_and_value(obs_t, mask_t)
                actions_buf[step] = action.item()
                logprobs_buf[step] = logprob.item()
                values_buf[step] = value.item()

                next_obs, reward, done, trunc, _ = self.env.step(action.item())
                reward = float(reward) if reward is not None else 0.0
                rewards_buf[step] = reward
                episode_reward += reward

                if done:
                    # wall-clock budget exhausted — full env restart (takes priority over trunc)
                    next_done = 1.0
                    self.env.close()
                    obs, _ = self.env.reset()
                    self._sync_graph()
                    episode_reward = 0.0
                elif trunc:
                    next_done = 1.0
                    print(
                        f"Iter {iteration}/{num_iterations}  "
                        f"Episode {episode_idx + 1}  "
                        f"Reward: {episode_reward:.2f}"
                    )
                    results.append((iteration, episode_reward))
                    episode_idx += 1
                    episode_reward = 0.0
                    obs = next_obs  # env auto-reset internally on trunc
                else:
                    next_done = 0.0
                    obs = next_obs

            # ---- bootstrap value ----
            with torch.no_grad():
                next_obs_vec = self._encode_obs(obs)
                next_val = self.net.get_value(
                    torch.tensor(next_obs_vec, dtype=torch.float32).unsqueeze(0)
                ).item()
                # if last step ended the episode, don't bootstrap
                if next_done:
                    next_val = 0.0

            # ---- GAE advantage computation (exact ppo_ref logic) ----
            advantages = np.zeros(num_steps, dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

            # ---- PPO update ----
            b_obs = torch.tensor(obs_buf, dtype=torch.float32)
            b_actions = torch.tensor(actions_buf, dtype=torch.long)
            b_logprobs = torch.tensor(logprobs_buf, dtype=torch.float32)
            b_advantages = torch.tensor(advantages, dtype=torch.float32)
            b_returns = torch.tensor(returns, dtype=torch.float32)
            b_values = torch.tensor(values_buf, dtype=torch.float32)
            b_masks = torch.tensor(masks_buf, dtype=torch.bool)

            b_inds = np.arange(num_steps)
            for _ in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.net.get_action_and_value(
                        b_obs[mb_inds], b_masks[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_adv = b_advantages[mb_inds]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -clip_coef, clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2
                    ).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                    optimizer.step()

            if iteration % save_interval == 0:
                self._save(policy_path)
                print(f"Checkpoint saved at iteration {iteration} → {policy_path}")

        self._save(policy_path)
        with open(log_path, 'w') as f:
            f.write("Iteration,Episode Reward\n")
            for iter_i, ep_r in results:
                f.write(f"{iter_i},{ep_r:.4f}\n")
        print(f"Policy saved to {policy_path}")

    def _save(self, policy_path: str):
        data = {
            'actor_state_dict': self.net.actor.state_dict(),
            'critic_state_dict': self.net.critic.state_dict(),
            'num_nodes': self.num_nodes,
            'num_slots': self.num_slots,
            'obs_dim': self.obs_dim,
            'hidden_dim': self.hidden_dim,
        }
        with open(policy_path, 'wb') as f:
            pickle.dump(data, f)
