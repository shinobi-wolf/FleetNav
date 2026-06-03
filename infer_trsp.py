"""Run a saved TRSP policy for N inference episodes.

Usage:
    python infer_trsp.py -p <policy_id> [-n <num_episodes>] [-r <render_mode>]

Config is loaded from runs/trsp/<policy_id>/config.yaml (inference config saved
during training — no 'training' block required).
"""
import argparse
import json
import os
from pathlib import Path
import yaml
import numpy as np

import dronesim.trsp  # noqa: F401 — registers SumoTRSPEnv-v0
import gymnasium as gym
from dronesim.trsp.agents.sarsa import SARSAAgent
from dronesim.trsp.agents.dijkstra import DijkstraAgent
from dronesim.trsp.agents.dqn import DQNAgent


def run(config, num_episodes, policy_id=None):
    algo = config['agent']['algo']

    if algo == 'sarsa':
        agent = SARSAAgent(config)
        env = agent.env
    elif algo == 'dqn':
        agent = DQNAgent(config)
        env = agent.env
    elif algo == 'dijkstra':
        env = gym.make('SumoTRSPEnv-v0', config=config)
        agent = DijkstraAgent(agent_id=0)
    else:
        raise ValueError(f"Unsupported algo for sim runner: {algo!r}")

    rewards = []
    steps_list = []
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        ep_steps = 0
        ep_success = False
        done = False

        while not done:
            action = agent.predict(obs,
                                   G=env.unwrapped.G,
                                   idx2node=env.unwrapped.idx2node,
                                   node2idx=env.unwrapped.node2idx)
            obs, reward, terminated, truncated, info = env.step(int(action))
            env.render()
            total_reward += reward
            ep_steps += 1
            done = terminated# or truncated

            # task_handler resets the episode internally on truncation, so a
            # single env.reset() covers the whole training run. But for the
            # sim runner we want one episode per env.reset() call, so we break
            # on the first truncation or termination.

        # Check if the last truncation was due to task completion.
        # The info dict doesn't carry this directly after _reset_episode, so
        # we rely on the step count heuristic: if we didn't hit max_steps it
        # was a successful arrival.
        if ep_steps <= env.unwrapped.max_steps:
            ep_success = True
            successes += 1

        rewards.append(total_reward)
        steps_list.append(ep_steps)
        print(
            f"Episode {ep+1:4d}/{num_episodes}  "
            f"reward={total_reward:8.2f}  steps={ep_steps:4d}  "
            f"{'SUCCESS' if ep_success else 'timeout'}"
        )

    env.close()

    if policy_id:
        out_dir = Path("runs", "trsp", policy_id)
        profile_path = out_dir / "infer_traffic_profile.json"
        profile_path.write_text(
            json.dumps(env.unwrapped.traffic.profiles, indent=2)
        )
        print(f"Traffic profile → {profile_path}")
        log_path = out_dir / "infer_traffic_log.jsonl"
        with open(log_path, "w") as _f:
            for event in env.unwrapped.traffic_log:
                _f.write(json.dumps(event) + "\n")
        print(f"Traffic event log → {log_path}")

    print()
    print("=" * 60)
    print(f"  Episodes   : {num_episodes}")
    print(f"  Success    : {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean steps : {np.mean(steps_list):.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run a TRSP policy')
    parser.add_argument('-p', '--policy', required=True,
                        help="Policy ID (sub-directory name under runs/trsp/)")
    parser.add_argument('-n', '--num-episodes', type=int, default=1,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('-r', '--render', choices=['quiet', 'print', 'human'],
                        help='Override render mode from config')
    args = parser.parse_args()

    config_path = os.path.join('runs', 'trsp', args.policy, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config found at {config_path}. "
            "Run train_trsp.py first to create a policy."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.render:
        config['simulation']['render_mode'] = args.render

    run(config, args.num_episodes, policy_id=args.policy)


if __name__ == '__main__':
    main()
