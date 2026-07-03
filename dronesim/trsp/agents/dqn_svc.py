import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration
from dronesim.trsp.agents.device import get_device

DEVICE = get_device()


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_nodes: int):
        self.cap  = capacity
        self.ptr  = 0
        self.size = 0
        self.obs        = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions    = np.zeros(capacity, dtype=np.int64)
        self.rewards    = np.zeros(capacity, dtype=np.float32)
        self.dones      = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, num_nodes), dtype=np.bool_)

    def push(self, obs, action, reward, next_obs, done, next_mask):
        self.obs[self.ptr]        = obs
        self.next_obs[self.ptr]   = next_obs
        self.actions[self.ptr]    = action
        self.rewards[self.ptr]    = reward
        self.dones[self.ptr]      = done
        self.next_masks[self.ptr] = next_mask
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.next_masks[idx],
        )

    def __len__(self):
        return self.size


class DQNSVCAgent:
    """DQN agent for SVC (Stochastic Vehicular Congestion) traffic model.

    Observation encoding:
        one-hot(graph_node) ++ one-hot(target_node) ++ neighbor_speeds[max_degree]

    neighbor_speeds[i] = traci.edge.getLastStepMeanSpeed(edge_to_neighbor_i) / ugv_speed,
    sorted by node2idx, padded with 1.0 (free-flow) for missing neighbor slots.

    Action space: Discrete(num_nodes) — masked to valid neighbours before argmax.
    """

    def __init__(self, config):
        self.config = config
        dqn_cfg = config.get('training', {}).get('dqn', {})
        self.hidden_dim  = dqn_cfg.get('hidden_dim', 128)
        self.max_degree  = config['net']['max_degree']

        self.env = gym.make('SumoTRSPEnv-v0', config=config)

        self.num_nodes = len(self.env.unwrapped.node2idx)

        # graph references — populated after first env.reset()
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

        self.device   = DEVICE
        self.q_net    = QNetwork(self.obs_dim, self.num_nodes, self.hidden_dim).to(self.device)
        self.q_target = QNetwork(self.obs_dim, self.num_nodes, self.hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        if saved_data is not None:
            self.q_net.load_state_dict(saved_data['state_dict'])
            self.q_target.load_state_dict(saved_data['state_dict'])
            print(f"Policy loaded from runs/trsp/{policy_name}/policy.pkl  [device={self.device}]")
        else:
            print(f"DQNSVCAgent initialized  [device={self.device}]")

    # ------------------------------------------------------------------ helpers

    def _sync_graph(self):
        self.G        = self.env.unwrapped.G
        self.node2idx = self.env.unwrapped.node2idx
        self.idx2node = self.env.unwrapped.idx2node

    def _encode_obs(self, obs) -> np.ndarray:
        gn     = int(obs['graph_node'])
        tn     = int(obs['target_node'])
        speeds = np.asarray(obs['neighbor_speeds'], dtype=np.float32)
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        vec[gn] = 1.0
        vec[self.num_nodes + tn] = 1.0
        vec[2 * self.num_nodes : 2 * self.num_nodes + self.max_degree] = speeds
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
        """Return the greedy node-id action, masked to valid neighbours."""
        if G is not None:
            self.G, self.idx2node, self.node2idx = G, idx2node, node2idx
        elif self.G is None:
            self._sync_graph()

        obs_t = torch.tensor(self._encode_obs(obs), dtype=torch.float32).unsqueeze(0).to(self.device)
        mask  = self._get_mask(obs['graph_node'])
        with torch.no_grad():
            q_vals = self.q_net(obs_t)[0].cpu()
        q_vals[~mask] = float('-inf')
        return int(q_vals.argmax().item())

    # ------------------------------------------------------------------ training

    def learn(self, policy_dir: str):
        dqn_cfg       = self.config['training']['dqn']
        num_episodes  = dqn_cfg['num_episodes']
        gamma         = dqn_cfg['gamma']
        lr            = dqn_cfg['lr']
        epsilon       = dqn_cfg['epsilon']
        epsilon_decay = dqn_cfg['epsilon_decay']
        epsilon_min   = dqn_cfg.get('epsilon_min', 0.05)
        batch_size    = dqn_cfg['batch_size']
        buf_capacity  = dqn_cfg['buffer_capacity']
        tgt_update    = dqn_cfg['target_update_freq']
        learn_start   = dqn_cfg.get('learn_start', batch_size * 4)

        policy_path = os.path.join(policy_dir, 'policy.pkl')
        log_path    = os.path.join(policy_dir, 'train_log.csv')

        optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        buffer    = ReplayBuffer(buf_capacity, self.obs_dim, self.num_nodes)

        obs, _ = self.env.reset()
        self._sync_graph()

        episode_idx  = 0
        total_steps  = 0
        total_reward = 0.0
        results      = []

        while episode_idx < num_episodes:
            obs_vec = self._encode_obs(obs)
            mask    = self._get_mask(obs['graph_node'])

            if random.random() < epsilon:
                valid  = mask.nonzero(as_tuple=True)[0].tolist()
                action = random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    q_vals = self.q_net(
                        torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                    )[0].cpu()
                q_vals[~mask] = float('-inf')
                action = int(q_vals.argmax().item())

            next_obs, reward, done, trunc, _ = self.env.step(action)
            total_reward += reward
            total_steps  += 1

            next_mask    = self._get_mask(next_obs['graph_node'])
            next_obs_vec = self._encode_obs(next_obs)
            episode_over = float(trunc or done)

            buffer.push(obs_vec, action, reward, next_obs_vec, episode_over, next_mask.numpy())

            if len(buffer) >= learn_start:
                self._update(buffer.sample(batch_size), gamma, optimizer)

            if total_steps % tgt_update == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())

            if trunc:
                print(
                    f"Episode {episode_idx + 1}/{num_episodes}  "
                    f"Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.4f}"
                )
                results.append((total_reward, epsilon))
                episode_idx  += 1
                total_reward  = 0.0
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                if episode_idx % 100 == 0:
                    self._save(policy_path)

            if done:
                self.env.close()
                obs, _ = self.env.reset()
                self._sync_graph()
            elif trunc:
                obs = next_obs
            else:
                obs = next_obs

        self._save(policy_path)
        with open(log_path, 'w') as f:
            f.write("Episode,Total Reward,Epsilon\n")
            for i, (r, e) in enumerate(results):
                f.write(f"{i + 1},{r:.2f},{e:.4f}\n")
        print(f"Policy saved to {policy_path}")

    def _update(self, batch, gamma: float, optimizer):
        obs_b, acts_b, rews_b, next_obs_b, dones_b, next_masks_b = batch

        obs_t        = torch.tensor(obs_b,        dtype=torch.float32).to(self.device)
        next_obs_t   = torch.tensor(next_obs_b,   dtype=torch.float32).to(self.device)
        acts_t       = torch.tensor(acts_b,       dtype=torch.long).to(self.device)
        rews_t       = torch.tensor(rews_b,       dtype=torch.float32).to(self.device)
        dones_t      = torch.tensor(dones_b,      dtype=torch.float32).to(self.device)
        next_masks_t = torch.tensor(next_masks_b, dtype=torch.bool).to(self.device)

        q_sa = self.q_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_target(next_obs_t)
            q_next[~next_masks_t] = float('-inf')
            q_next_max = q_next.max(dim=1).values
            q_next_max = torch.nan_to_num(q_next_max, neginf=0.0)
            td_target = rews_t + gamma * (1.0 - dones_t) * q_next_max

        loss = F.mse_loss(q_sa, td_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def _save(self, policy_path: str):
        data = {
            'state_dict': self.q_net.state_dict(),
            'num_nodes':  self.num_nodes,
            'max_degree': self.max_degree,
            'obs_dim':    self.obs_dim,
            'hidden_dim': self.hidden_dim,
        }
        with open(policy_path, 'wb') as f:
            pickle.dump(data, f)
