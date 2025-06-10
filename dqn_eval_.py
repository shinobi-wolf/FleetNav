import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset(seed=42)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

# # ALGO LOGIC: initialize agent here:
# class QNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(np.array(env.observation_space.shape).prod(), 16),
#             nn.ReLU(),
#             nn.Linear(16, 9),
#             nn.ReLU(),
#             nn.Linear(9, env.action_space.n),
#         )

#     def forward(self, x):
#         if x.dtype != torch.float32:
#             x = x.float()
#         return self.network(x)

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from dqn import make_env, QNetwork

    model_path = "runs/LMDEnv-v0__dqn__1__1749521016/dqn.cleanrl_model"
    evaluate(
        model_path,
        make_env,
        "LMDEnv-v0",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        device="cuda",
        capture_video=False,
    )
