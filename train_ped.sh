#!/usr/bin/env bash
python train_trsp.py -c config/trsp/ped/sarsa_rand_grid.yaml
python train_trsp.py -c config/trsp/ped/dqn_rand_grid.yaml
python train_trsp.py -c config/trsp/ped/sarsa_ames.yaml
python train_trsp.py -c config/trsp/ped/dqn_ames.yaml
