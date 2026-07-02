import argparse
import yaml
import os
import time
import copy

import dronesim.trsp  # noqa: F401 — triggers SumoTRSPEnv-v0 registration
from dronesim.trsp.agents.sarsa_psl import SARSAAgent
from dronesim.trsp.agents.dqn_psl   import DQNAgent
from dronesim.trsp.agents.dqn_svc   import DQNSVCAgent
from dronesim.trsp.agents.dqn_ped   import DQNPEDAgent
from dronesim.trsp.agents.ppo_psl   import PPOAgent
from dronesim.trsp.agents.a2c_psl   import A2CAgent
from dronesim.trsp.agents.sarsa_svc import SARSASVCAgent
from dronesim.trsp.agents.sarsa_ped import SARSAPEDAgent
from dronesim.trsp.agents.a2c_svc   import A2CSVCAgent
from dronesim.trsp.agents.ppo_svc   import PPOSVCAgent


def lmd_trainer(config, policy_dir, resume=False):
    algo = config['agent']['algo']
    if algo == 'dqn':
        agent = DQNAgent(config)
    elif algo == 'dqn_svc':
        agent = DQNSVCAgent(config)
    elif algo == 'dqn_ped':
        agent = DQNPEDAgent(config)
    elif algo == 'ppo':
        agent = PPOAgent(config)
    elif algo == 'ppo_svc':
        agent = PPOSVCAgent(config)
    elif algo == 'a2c':
        agent = A2CAgent(config)
    elif algo == 'a2c_svc':
        agent = A2CSVCAgent(config)
    elif algo == 'sarsa_svc':
        agent = SARSASVCAgent(config)
    elif algo == 'sarsa_ped':
        agent = SARSAPEDAgent(config)
    else:
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
        print(f"POLICY_ID={policy_name}")

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
