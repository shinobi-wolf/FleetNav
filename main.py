import argparse
from server import lmd_simulator
from trainer import lmd_trainer
import yaml
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-c', '--config', type=str, help="Configuration file path")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    parser.add_argument('-t', '--train', action='store_true', help="Run training mode")
    args = parser.parse_args()


    if args.train:
        if args.config:
            config_path = args.config
        else:
            config_path = f"config/train.yaml"
        config = yaml.safe_load(open(config_path, 'r'))
        
        policy_name = os.urandom(4).hex()
        policy_dir = os.path.join(f"runs/{policy_name}")
        os.makedirs(policy_dir, exist_ok=True)
        lmd_trainer(config, policy_dir)

        config_file = os.path.join(policy_dir, "config.yaml")
        config['agent']['policy'] = policy_name
        config['simulation']['train'] = False
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
            
    else:
        if args.config:
            config_path = args.config
        else:
            config_path = f"runs/{args.policy}/config.yaml"
        config = yaml.safe_load(open(config_path, 'r'))
     
        metrics = lmd_simulator(config)
        print("Metrics:", metrics)