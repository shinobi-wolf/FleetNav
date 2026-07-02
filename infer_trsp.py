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
from dronesim.trsp.agents.sarsa_psl    import SARSAAgent
from dronesim.trsp.agents.sarsa_svc    import SARSASVCAgent
from dronesim.trsp.agents.sarsa_ped    import SARSAPEDAgent
from dronesim.trsp.agents.dijkstra_psl import DijkstraAgent
from dronesim.trsp.agents.dqn_psl      import DQNAgent
from dronesim.trsp.agents.dqn_svc      import DQNSVCAgent
from dronesim.trsp.agents.dqn_ped      import DQNPEDAgent
from dronesim.trsp.agents.dsp          import DSPAgent
from dronesim.trsp.agents.dsp_ped      import DSPPEDAgent
from dronesim.trsp.agents.a2c_psl      import A2CAgent
from dronesim.trsp.agents.a2c_svc      import A2CSVCAgent
from dronesim.trsp.agents.ppo_psl      import PPOAgent
from dronesim.trsp.agents.ppo_svc      import PPOSVCAgent


def run(config, num_episodes, policy_id=None):
    algo = config['agent']['algo']

    if algo == 'sarsa':
        agent = SARSAAgent(config)
        env = agent.env
    elif algo == 'sarsa_svc':
        agent = SARSASVCAgent(config)
        env = agent.env
    elif algo == 'sarsa_ped':
        agent = SARSAPEDAgent(config)
        env = agent.env
    elif algo == 'dqn':
        agent = DQNAgent(config)
        env = agent.env
    elif algo == 'dqn_svc':
        agent = DQNSVCAgent(config)
        env = agent.env
    elif algo == 'dqn_ped':
        agent = DQNPEDAgent(config)
        env = agent.env
    elif algo == 'a2c':
        agent = A2CAgent(config)
        env = agent.env
    elif algo == 'a2c_svc':
        agent = A2CSVCAgent(config)
        env = agent.env
    elif algo == 'ppo':
        agent = PPOAgent(config)
        env = agent.env
    elif algo == 'ppo_svc':
        agent = PPOSVCAgent(config)
        env = agent.env
    elif algo == 'dijkstra':
        env = gym.make('SumoTRSPEnv-v0', config=config)
        agent = DijkstraAgent(agent_id=0)
    elif algo in ('dsp', 'dsp_ped'):
        env = gym.make('SumoTRSPEnv-v0', config=config)
        u = env.unwrapped
        if algo == 'dsp_ped':
            pso = config.get('traffic', {}).get('ped_speed_override', {})
            agent = DSPPEDAgent(0, u.G_dir, u.node2idx, u.idx2node, u.ugv_speed,
                                gamma=pso.get('gamma', 1.913),
                                density_jam=pso.get('density_jam', 5.4),
                                v_min=pso.get('v_min', 0.1))
        else:
            agent = DSPAgent(0, u.G_dir, u.node2idx, u.idx2node, u.ugv_speed)
    else:
        raise ValueError(f"Unsupported algo for sim runner: {algo!r}")

    def maybe_start_leg():
        """Leg-based planners (DSP) replan per OD leg; feed them the current src/dst."""
        if hasattr(agent, 'start_leg'):
            u = env.unwrapped
            src = u.idx2node[u.ugv.position]
            dst = u.idx2node[u.ugv.task_list[0]] if u.ugv.task_list else src
            agent.start_leg(src, dst)

    rewards = []
    steps_list = []
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        maybe_start_leg()
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
            if truncated and not done:   # env resampled a new OD leg — replan
                maybe_start_leg()

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
        traffic_model = env.unwrapped.traffic_model
        if traffic_model == 'psl':
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
    parser.add_argument('-p', '--policy',
                        help="Policy ID (sub-directory name under runs/trsp/)")
    parser.add_argument('-c', '--config',
                        help="Config YAML path — use instead of -p for agents that need no saved policy (e.g. dijkstra)")
    parser.add_argument('-n', '--num-episodes', type=int, default=1,
                        help='Number of episodes to run (default: 1)')
    parser.add_argument('-r', '--render', choices=['quiet', 'print', 'human'],
                        help='Override render mode from config')
    args = parser.parse_args()

    if args.policy:
        config_path = os.path.join('runs', 'trsp', args.policy, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config found at {config_path}. "
                "Run train_trsp.py first to create a policy."
            )
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        parser.error("Provide either -p <policy_id> or -c <config_path>")

    if args.render:
        config['simulation']['render_mode'] = args.render

    run(config, args.num_episodes, policy_id=args.policy)


if __name__ == '__main__':
    main()
