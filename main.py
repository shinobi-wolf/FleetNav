import argparse
from server_trsp import lmd_simulator
from trainer import lmd_trainer
import yaml
import os


def main():
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")

    # mutually exclusive top-level modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Run training mode')
    mode_group.add_argument('--eval', action='store_true', help='Run evaluation mode')
    mode_group.add_argument('--run', action='store_true', help='Run simulation/serve mode')

    # shared/conditional args
    parser.add_argument('-c', '--config', type=str,
                        help='Configuration file path (train: optional, eval/run: optional overrides)')
    parser.add_argument('-p', '--policy', type=str,
                        help="Policy name (required for --eval and --run), e.g. '0d318b44'")

    args = parser.parse_args()

    if args.train:
        # train: config optional (defaults to config/train.yaml)
        train_config_path = args.config if args.config else 'config/train.yaml'
        resume = False
        if not args.policy:
            inf_config_path = 'config/sumo_lmd.yaml'

            train_config = yaml.safe_load(open(train_config_path, 'r'))
            inf_config = yaml.safe_load(open(inf_config_path, 'r'))

            policy_name = os.urandom(4).hex()
            policy_dir = os.path.join(f"runs/{policy_name}")
            os.makedirs(policy_dir, exist_ok=True)

            config_file = os.path.join(policy_dir, "config.yaml")
            inf_config['agent']['policy'] = policy_name
            inf_config['agent']['algo'] = train_config['agent']['algo']
            inf_config['net']['net_path'] = train_config['net']['net_path']
            with open(config_file, 'w') as f:
                yaml.dump(inf_config, f)
        else:
            policy_name = args.policy
            policy_dir = os.path.join(f"runs/{policy_name}")
            resume = True

        lmd_trainer(train_config, policy_dir, resume=resume)

    elif args.run:
        # eval or run: policy is mandatory
        if not args.policy:
            parser.error('--policy (-p) is required for --eval and --run')

        policy = args.policy
        # config can override default run config path
        config_path = args.config if args.config else f"runs/{policy}/config.yaml"
        config = yaml.safe_load(open(config_path, 'r'))

        metrics = lmd_simulator(config)
        print("Metrics:", metrics)


if __name__ == '__main__':
    main()