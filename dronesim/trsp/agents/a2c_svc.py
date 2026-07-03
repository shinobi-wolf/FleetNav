import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.categorical import Categorical

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration
from dronesim.trsp.agents.device import get_device

DEVICE = get_device()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
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
        probs  = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class A2CSVCAgent:
    """A2C agent for SVC (Stochastic Vehicular Congestion) traffic.

    Observation encoding:
        one-hot(graph_node) ++ one-hot(target_node) ++ neighbor_speeds[max_degree]

    neighbor_speeds are live normalised TraCI edge speeds (see DQNSVCAgent for details).
    Action space: Discrete(num_nodes) — masked to valid neighbours at each step.
    Training: single-pass policy-gradient update per rollout (no clipping, no epochs).
    """

    def __init__(self, config):
        self.config     = config
        a2c_cfg         = config.get('training', {}).get('a2c', {})
        self.hidden_dim = a2c_cfg.get('hidden_dim', 128)
        self.max_degree = config['net']['max_degree']

        self.env = gym.make('SumoTRSPEnv-v0', config=config)

        self.num_nodes = len(self.env.unwrapped.node2idx)

        self.G        = None
        self.node2idx = None
        self.idx2node = None

        # obs: one-hot(graph_node) ++ one-hot(target_node) ++ neighbor_speeds
        self.obs_dim = 2 * self.num_nodes + self.max_degree

        policy_name = config['agent'].get('policy', None)
        saved_data  = None
        if policy_name:
            policy_path = f"runs/trsp/{policy_name}/policy.pkl"
            with open(policy_path, 'rb') as f:
                saved_data = pickle.load(f)
            self.hidden_dim = saved_data.get('hidden_dim', self.hidden_dim)

        self.device = DEVICE
        self.net    = ActorCritic(self.obs_dim, self.num_nodes, self.hidden_dim).to(self.device)

        if saved_data is not None:
            self.net.actor.load_state_dict(saved_data['actor_state_dict'])
            self.net.critic.load_state_dict(saved_data['critic_state_dict'])
            print(f"Policy loaded from runs/trsp/{policy_name}/policy.pkl  [device={self.device}]")
        else:
            print(f"A2CSVCAgent initialized  [device={self.device}]")

    # ------------------------------------------------------------------ helpers

    def _sync_graph(self):
        self.G        = self.env.unwrapped.G
        self.node2idx = self.env.unwrapped.node2idx
        self.idx2node = self.env.unwrapped.idx2node

    def _encode_obs(self, obs) -> np.ndarray:
        gn     = int(obs['graph_node'])
        tn     = int(obs['target_node'])
        speeds = np.asarray(obs['neighbor_speeds'], dtype=np.float32)
        vec    = np.zeros(self.obs_dim, dtype=np.float32)
        vec[gn] = 1.0
        vec[self.num_nodes + tn] = 1.0
        vec[2 * self.num_nodes:] = speeds
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

        obs_t = torch.tensor(self._encode_obs(obs), dtype=torch.float32).unsqueeze(0).to(self.device)
        mask  = self._get_mask(obs['graph_node']).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net.actor(obs_t)
            logits = logits.masked_fill(~mask, float('-inf'))
        return int(logits.argmax(dim=-1).item())

    # ------------------------------------------------------------------ training

    def learn(self, policy_dir: str):
        a2c_cfg        = self.config['training']['a2c']
        num_iterations = a2c_cfg['num_iterations']
        num_steps      = a2c_cfg['num_steps']
        gamma          = a2c_cfg['gamma']
        gae_lambda     = a2c_cfg['gae_lambda']
        ent_coef       = a2c_cfg['ent_coef']
        vf_coef        = a2c_cfg['vf_coef']
        max_grad_norm  = a2c_cfg['max_grad_norm']
        lr             = float(a2c_cfg['lr'])
        anneal_lr      = a2c_cfg.get('anneal_lr', False)
        save_interval  = a2c_cfg.get('save_interval', 50)

        policy_path = os.path.join(policy_dir, 'policy.pkl')
        log_path    = os.path.join(policy_dir, 'train_log.csv')

        optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        obs_buf     = np.zeros((num_steps, self.obs_dim), dtype=np.float32)
        actions_buf = np.zeros(num_steps, dtype=np.int64)
        rewards_buf = np.zeros(num_steps, dtype=np.float32)
        dones_buf   = np.zeros(num_steps, dtype=np.float32)
        values_buf  = np.zeros(num_steps, dtype=np.float32)
        masks_buf   = np.zeros((num_steps, self.num_nodes), dtype=np.bool_)

        obs, _ = self.env.reset()
        self._sync_graph()
        next_done      = 0.0
        episode_reward = 0.0
        episode_rewards = []
        results = []

        for iteration in range(1, num_iterations + 1):
            if anneal_lr:
                frac = 1.0 - (iteration - 1) / num_iterations
                optimizer.param_groups[0]['lr'] = frac * lr

            # ---- collect rollout ----
            for step in range(num_steps):
                obs_vec = self._encode_obs(obs)
                mask    = self._get_mask(obs['graph_node'])

                dones_buf[step] = next_done
                obs_buf[step]   = obs_vec
                masks_buf[step] = mask.numpy()

                obs_t  = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask_t = mask.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, _, _, value = self.net.get_action_and_value(obs_t, mask_t)
                actions_buf[step] = action.item()
                values_buf[step]  = value.item()

                next_obs, reward, done, trunc, _ = self.env.step(action.item())
                reward = float(reward) if reward is not None else 0.0
                rewards_buf[step] = reward
                episode_reward   += reward

                if done:
                    next_done = 1.0
                    self.env.close()
                    obs, _ = self.env.reset()
                    self._sync_graph()
                    episode_reward = 0.0
                elif trunc:
                    next_done = 1.0
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    obs = next_obs
                else:
                    next_done = 0.0
                    obs = next_obs

            # ---- bootstrap value ----
            with torch.no_grad():
                next_obs_vec = self._encode_obs(obs)
                next_val = self.net.get_value(
                    torch.tensor(next_obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                ).item()
                if next_done:
                    next_val = 0.0

            # ---- GAE advantage computation ----
            advantages = np.zeros(num_steps, dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues      = next_val
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues      = values_buf[t + 1]
                delta         = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

            # ---- A2C update — single pass, full rollout batch ----
            b_obs        = torch.tensor(obs_buf,    dtype=torch.float32).to(self.device)
            b_actions    = torch.tensor(actions_buf, dtype=torch.long).to(self.device)
            b_advantages = torch.tensor(advantages,  dtype=torch.float32).to(self.device)
            b_returns    = torch.tensor(returns,     dtype=torch.float32).to(self.device)
            b_masks      = torch.tensor(masks_buf,   dtype=torch.bool).to(self.device)

            _, newlogprob, entropy, newvalue = self.net.get_action_and_value(
                b_obs, b_masks, b_actions
            )

            adv = b_advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_loss    = -(newlogprob * adv).mean()
            v_loss     = 0.5 * ((newvalue.view(-1) - b_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss       = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
            optimizer.step()

            mean_ep_ret = np.mean(episode_rewards) if episode_rewards else float('nan')
            print(
                f"Iter {iteration}/{num_iterations}  "
                f"pg_loss={pg_loss.item():.4f}  "
                f"v_loss={v_loss.item():.4f}  "
                f"mean_ep_ret={mean_ep_ret:.2f}"
            )
            results.append((iteration, pg_loss.item(), v_loss.item(), mean_ep_ret))
            episode_rewards = []

            if iteration % save_interval == 0:
                self._save(policy_path)
                print(f"Checkpoint saved at iteration {iteration} → {policy_path}")

        self._save(policy_path)
        with open(log_path, 'w') as f:
            f.write("Iteration,PG Loss,Value Loss,Mean Episode Return\n")
            for iter_i, pg, vl, ret in results:
                f.write(f"{iter_i},{pg:.6f},{vl:.6f},{ret:.4f}\n")
        print(f"Policy saved to {policy_path}")

    def _save(self, policy_path: str):
        data = {
            'actor_state_dict':  self.net.actor.state_dict(),
            'critic_state_dict': self.net.critic.state_dict(),
            'num_nodes':         self.num_nodes,
            'max_degree':        self.max_degree,
            'obs_dim':           self.obs_dim,
            'hidden_dim':        self.hidden_dim,
        }
        with open(policy_path, 'wb') as f:
            pickle.dump(data, f)
