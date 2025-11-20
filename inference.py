import argparse
import yaml
import os
import gymnasium as gym
import numpy as np
import time

import envs
from agents import get_agent

def main():
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()

    config_path = f"runs/{args.policy}/config.yaml"
    config = yaml.safe_load(open(config_path, 'r'))
    
    env = gym.make('SumoTrainEnv-v0', config=config)
    agent = get_agent(env, config['agent'], 0)

    obs, world_state = env.reset(seed=42)
    print("Step 0:")
    print(f"Position={world_state['ugv_position']}, Active Task={world_state['ugv_target']}")
    env.render()

    terminated = False
    total_reward = 0.0
    step_idx = 0
    start_time = time.time()

    while not terminated:
        step_start_time = time.time()
        print(f"Step {step_idx+1}:")
        
        action = agent.predict(obs)
        print(f"Agent selected action: {action}")
        # Execute step
        obs, reward, terminated, truncated, world_state = env.step(action)
        total_reward += reward
        print("Reward:", reward)
        print(f"Position={world_state['ugv_position']}, Active Task={world_state['ugv_target']}")

        env.render()
        if truncated:
            step_idx=0
        else:
            step_idx += 1
        
    # Calculate final metrics
    simulation_wall_time = time.time() - start_time
    total_time_taken = world_state.get("time_elapsed", 0)
    
    
    env.close() # Ensure pygame resources are cleaned up

if __name__ == '__main__':
    main()