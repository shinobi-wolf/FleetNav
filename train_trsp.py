import argparse
import yaml
import os
import time
import copy

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration
from dronesim.trsp.agents.sarsa import SARSAAgent
from dronesim.trsp.agents.dqn import DQNAgent
from dronesim.trsp.agents.ppo import PPOAgent


def lmd_trainer(config, policy_dir, resume=False):
    agent = SARSAAgent(config)
    agent.learn(policy_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRSP Training")
    parser.add_argument('-c', '--config', type=str,
                        help='Config file path (default: config/trsp/train.yaml)')
    parser.add_argument('-p', '--policy', type=str,
                        help="Policy name to resume (e.g. '0d318b44')")
    args = parser.parse_args()

    train_config_path = args.config if args.config else 'config/trsp/train.yaml'
    resume = False
    if not args.policy:
        train_config = yaml.safe_load(open(train_config_path, 'r'))

        policy_name = os.urandom(4).hex()
        policy_dir = os.path.join(f"runs/trsp/{policy_name}")
        os.makedirs(policy_dir, exist_ok=True)

        config_file = os.path.join(policy_dir, "config.yaml")
        inf_config = copy.deepcopy(train_config)
        inf_config['agent']['policy'] = policy_name
        inf_config['simulation']['render_mode'] = 'human'
        inf_config.pop('training', None)
        with open(config_file, 'w') as f:
            yaml.dump(inf_config, f)
    else:
        policy_name = args.policy
        policy_dir = os.path.join(f"runs/trsp/{policy_name}")
        resume = True
        train_config = yaml.safe_load(open(train_config_path, 'r'))

    start_time = time.time()
    lmd_trainer(train_config, policy_dir, resume=resume)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
