#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import yaml

def _resolve_config_path(project_root, config_arg):
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    return config_path.resolve()


def main():
    parser = argparse.ArgumentParser(description="Smoke test for SumoLMDEnv")
    parser.add_argument("--config", default="config/lmd.yaml", help="Path to lmd config")
    parser.add_argument("--steps", type=int, default=8, help="Number of random steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--render-mode",
        default="quiet",
        choices=["quiet", "print", "human"],
        help="Render mode override for smoke test",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from dronesim.evrp.lmd_sumo import SumoLMDEnv

    config_path = _resolve_config_path(project_root, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["simulation"]["render_mode"] = args.render_mode

    env = SumoLMDEnv(config)

    try:
        obs, info = env.reset(seed=args.seed)
        print("reset ok")
        print(f"obs keys: {sorted(obs.keys())}")
        print(
            "initial info: "
            f"segment={info.get('current_traffic_segment_idx')} "
            f"cmd_speed_ms={info.get('traffic_speed_cmd_ms')} "
            f"num_ugvs={len(info.get('ugv_pos_idx', []))} "
            f"tasks={info.get('num_tasks_completed')}/{info.get('total_tasks')}"
        )

        for step_idx in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"step={step_idx + 1} reward={reward:.3f} "
                f"terminated={terminated} truncated={truncated} "
                f"segment={info.get('current_traffic_segment_idx')} "
                f"cmd_speed_kmh={info.get('traffic_speed_cmd_kmh'):.2f} "
                f"tasks={info.get('num_tasks_completed')}/{info.get('total_tasks')}"
            )

            if terminated:
                print("terminated early; ending smoke test loop")
                break

        print("smoke test finished")
        print(
            "final info: "
            f"time_elapsed={info.get('time_elapsed'):.2f}s "
            f"total_ev_distance={info.get('total_ev_distance'):.2f}m "
            f"speed_changed_this_step={info.get('speed_changed_this_step')}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
