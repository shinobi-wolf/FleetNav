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
            train_config_path = f"config/train.yaml"
            inf_config_path = f"config/sumo_lmd.yaml"
        train_config = yaml.safe_load(open(train_config_path, 'r'))
        inf_config = yaml.safe_load(open(inf_config_path, 'r'))
        
        policy_name = os.urandom(4).hex()
        policy_dir = os.path.join(f"runs/{policy_name}")
        os.makedirs(policy_dir, exist_ok=True)
        # train_config['agent']['policy'] = policy_name
        config_file = os.path.join(policy_dir, "config.yaml")
        inf_config['agent']['policy'] = policy_name
        inf_config['agent']['algo'] = train_config['agent']['algo']
        inf_config['net']['net_path'] = train_config['net']['net_path']
        with open(config_file, 'w') as f:
            yaml.dump(inf_config, f)

        lmd_trainer(train_config, policy_dir)

            
    else:
        if args.config:
            config_path = args.config
        else:
            config_path = f"runs/{args.policy}/config.yaml"
        config = yaml.safe_load(open(config_path, 'r'))
     
        metrics = lmd_simulator(config)
        print("Metrics:", metrics)