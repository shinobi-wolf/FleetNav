import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import envs
import yaml


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.3,
    capture_video: bool = True,
):
    env_config = yaml.safe_load(open("config/train.yaml", 'r'))
    env = gym.make(env_id, config=env_config, render_mode="human")
    model = Model(env).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = env.reset()
    env.render()
    episodic_returns = []
    terminated = False
    while not terminated:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values, dim=0).cpu().numpy()
            print(f"Taking Model Action: action={action}, q_values={q_values}")
        obs, _, terminated, truncations, infos = env.step(action)
        print(f"action={action}, obs={obs}, terminated={terminated}, truncations={truncations}, infos={infos}")
        env.render()
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if "episode" not in info:
        #             continue
        #         print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
        #         episodic_returns += [info["episode"]["r"]]
        # obs = next_obs

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.ReLU(),
            nn.Linear(9, env.action_space.n),
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.network(x)

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from dqn import make_env

    model_path = "runs/LMDEnv-v0__dqn__1__1749577114/dqn.cleanrl_model"
    evaluate(
        model_path,
        make_env,
        "LMDEnv-v0",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        device="cuda",
        capture_video=True,
    )
