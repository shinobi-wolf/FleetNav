#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

import gymnasium as gym
import numpy as np
import yaml

# Ensure local package imports resolve even when launched via wrappers.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # Registers custom envs
from agents.dijkstra import DijkstraAgent

INF = 100000.0


class RandomAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def predict(self, _observation):
        return None


class HeuristicAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def predict(self, observation):
        steps = np.asarray(observation["steps2dest"], dtype=np.float32)
        traffic = np.asarray(observation["nb_traffic"], dtype=np.float32)

        valid = np.where(steps < INF)[0]
        if len(valid) == 0:
            return None

        # Prefer short paths and lighter traffic.
        score = steps[valid] + 0.5 * traffic[valid]
        return int(valid[int(np.argmin(score))])


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_policy_agent_config(config, policy_name):
    if not policy_name:
        return config

    policy_cfg = Path("runs") / policy_name / "config.yaml"
    if not policy_cfg.exists():
        print(f"[WARN] policy config not found at {policy_cfg}; using base config")
        return config

    run_cfg = load_yaml(policy_cfg)
    if "agent" in run_cfg:
        config["agent"] = run_cfg["agent"]
    return config


def unpack_observation(obs, num_ugvs, num_actions, time_slot):
    steps2dest = np.asarray(obs["steps2dest"], dtype=np.float32).reshape(num_ugvs, num_actions)
    nb_traffic = np.asarray(obs["nb_traffic"], dtype=np.int32).reshape(num_ugvs, num_actions)
    ugv_status = np.asarray(obs["ugv_status"], dtype=np.int32).reshape(num_ugvs)
    graph_node = np.asarray(obs["graph_node"], dtype=np.int32).reshape(num_ugvs)
    target_node = np.asarray(obs["target_node"], dtype=np.int32).reshape(num_ugvs)

    all_obs = []
    for i in range(num_ugvs):
        best_dir = int(np.argmin(steps2dest[i]))
        all_obs.append(
            {
                "ugv_status": int(ugv_status[i]),
                "steps2dest": steps2dest[i],
                "target_node": int(target_node[i]),
                "graph_node": int(graph_node[i]),
                "nb_traffic": nb_traffic[i],
                # Compatibility fields for existing agent classes.
                "best_dir": best_dir,
                "time_slot": int(time_slot),
            }
        )
    return all_obs


def build_agents(config, num_ugvs, algo):
    selected_algo = algo or config.get("agent", {}).get("algo", "dijkstra")
    selected_algo = selected_algo.lower()

    agents = []
    if selected_algo == "dijkstra":
        agents = [DijkstraAgent(agent_id=i) for i in range(num_ugvs)]
    elif selected_algo == "random":
        agents = [RandomAgent(agent_id=i) for i in range(num_ugvs)]
    else:
        # Default fallback for LMD multi-vehicle compatibility.
        selected_algo = "heuristic"
        agents = [HeuristicAgent(agent_id=i) for i in range(num_ugvs)]

    print(f"[INFO] Loaded {len(agents)} agents with mode: {selected_algo}")
    return agents, selected_algo


def fallback_action(local_obs, num_actions):
    if local_obs["ugv_status"] != 1:
        return num_actions

    steps = np.asarray(local_obs["steps2dest"], dtype=np.float32)
    valid = np.where(steps < INF)[0]
    if len(valid) == 0:
        return num_actions
    return int(valid[int(np.argmin(steps[valid]))])


def choose_action(agent, local_obs, algo, num_actions):
    if local_obs["ugv_status"] != 1:
        return num_actions

    if algo == "random":
        steps = np.asarray(local_obs["steps2dest"], dtype=np.float32)
        valid = np.where(steps < INF)[0]
        if len(valid) == 0:
            return num_actions
        return int(random.choice(valid.tolist()))

    try:
        action = agent.predict(local_obs)
    except Exception as exc:
        print(f"[WARN] agent predict failed ({exc}); using fallback")
        return fallback_action(local_obs, num_actions)

    if action is None:
        return fallback_action(local_obs, num_actions)

    action = int(action)
    if action < 0 or action >= num_actions:
        return fallback_action(local_obs, num_actions)

    if float(local_obs["steps2dest"][action]) >= INF:
        return fallback_action(local_obs, num_actions)

    return action


def run(config, algo=None, seed=42, max_steps=0, render_mode="human", policy=None):
    config = merge_policy_agent_config(config, policy)
    config.setdefault("simulation", {})
    config["simulation"]["render_mode"] = render_mode

    env = gym.make("SumoLMDEnv-v0", config=config)

    num_ugvs = int(config["fleet"]["size"])
    num_actions = int(config["net"]["max_degree"])
    agents, selected_algo = build_agents(config, num_ugvs, algo)

    obs, info = env.reset(seed=seed)
    env.render()

    total_reward = 0.0
    step_idx = 0
    terminated = False

    try:
        while not terminated:
            if max_steps > 0 and step_idx >= max_steps:
                break

            all_obs = unpack_observation(
                obs,
                num_ugvs=num_ugvs,
                num_actions=num_actions,
                time_slot=info.get("current_traffic_segment_idx", 0),
            )

            action_list = [
                choose_action(agents[i], all_obs[i], selected_algo, num_actions)
                for i in range(num_ugvs)
            ]

            obs, reward, terminated, truncated, info = env.step(action_list)
            total_reward += float(reward)
            step_idx += 1

            print(
                f"step={step_idx} "
                f"reward={float(reward):.3f} "
                f"total_reward={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} "
                f"tasks={info.get('num_tasks_completed')}/{info.get('total_tasks')} "
                f"seg={info.get('current_traffic_segment_idx')} "
                f"speed={info.get('traffic_speed_cmd_kmh'):.2f} km/h"
            )
            print(
                f"  ugv_pos_idx={info.get('ugv_pos_idx')} "
                f"targets_idx={info.get('active_tasks_idx')} "
                f"status={info.get('ugv_status')}"
            )

            env.render()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        env.close()

    print("\nRun complete")
    print(f"steps={step_idx}")
    print(f"total_reward={total_reward:.3f}")
    print(
        f"tasks_completed={info.get('num_tasks_completed')}/{info.get('total_tasks')} "
        f"time_elapsed={info.get('time_elapsed'):.2f}s "
        f"distance={info.get('total_ev_distance'):.2f}m"
    )


def main():
    parser = argparse.ArgumentParser(description="Run LMD multi-vehicle simulation")
    parser.add_argument("-c", "--config", default="config/lmd.yaml", help="Path to LMD config")
    parser.add_argument("-a", "--algo", default=None, help="Agent mode: dijkstra|heuristic|random")
    parser.add_argument("-p", "--policy", default=None, help="Optional policy name under runs/<policy>/config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--steps", type=int, default=0, help="Max steps (0 means run until termination)")
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "print", "quiet"],
        help="Render mode",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run(
        config=cfg,
        algo=args.algo,
        seed=args.seed,
        max_steps=args.steps,
        render_mode=args.render_mode,
        policy=args.policy,
    )


if __name__ == "__main__":
    main()
