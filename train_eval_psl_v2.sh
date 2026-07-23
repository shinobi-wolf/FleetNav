# #!/usr/bin/env bash
# # Train + evaluate all PSL agents (sarsa, dqn, a2c, ppo) on the two lane-speed-fixed
# # networks (rand_grid, ames). PSL band {10,100} km/h. eval_trsp2.py also runs the
# # Dijkstra and DSP baselines automatically.
# # Override the interpreter with:  PYTHON=/path/to/python bash train_eval_psl_v2.sh
# set -euo pipefail

# PYTHON=${PYTHON:-/home/shinobi-wolf/anaconda3/envs/dronesim/bin/python}

# CONFIGS=(
#     config/trsp/psl_v2/sarsa_rand_grid.yaml
#     config/trsp/psl_v2/dqn_rand_grid.yaml
#     config/trsp/psl_v2/a2c_rand_grid.yaml
#     config/trsp/psl_v2/ppo_rand_grid.yaml
#     config/trsp/psl_v2/sarsa_ames.yaml
#     config/trsp/psl_v2/dqn_ames.yaml
#     config/trsp/psl_v2/a2c_ames.yaml
#     config/trsp/psl_v2/ppo_ames.yaml
# )

# mkdir -p logs/psl_v2
# POLICY_LOG=logs/psl_v2/policy_ids.txt
# : > "$POLICY_LOG"

# for cfg in "${CONFIGS[@]}"; do
#     tag=$(basename "$cfg" .yaml)
#     echo "======================================================================"
#     echo "TRAIN  $tag"
#     echo "======================================================================"
#     "$PYTHON" -u train_trsp.py -c "$cfg" 2>&1 | tee "logs/psl_v2/train_${tag}.log"

#     policy_id=$(grep '^POLICY_ID=' "logs/psl_v2/train_${tag}.log" | head -1 | cut -d= -f2)
#     echo "$tag=$policy_id" >> "$POLICY_LOG"

#     echo "----------------------------------------------------------------------"
#     echo "EVAL   $tag  (policy $policy_id)"
#     echo "----------------------------------------------------------------------"
#     "$PYTHON" -u eval_trsp2.py -p "$policy_id" 2>&1 | tee "logs/psl_v2/eval_${tag}.log"
# done

# echo "Done. Policy IDs: $POLICY_LOG"

# python train_trsp.py -c config/trsp/psl_v2/sarsa_rand_grid.yaml
# python train_trsp.py -c config/trsp/psl_v2/dqn_rand_grid.yaml
# python train_trsp.py -c config/trsp/psl_v2/a2c_rand_grid.yaml
# python train_trsp.py -c config/trsp/psl_v2/ppo_rand_grid.yaml

# python train_trsp.py -c config/trsp/psl_v2/sarsa_ames.yaml
# python train_trsp.py -c config/trsp/psl_v2/dqn_ames.yaml
# python train_trsp.py -c config/trsp/psl_v2/a2c_ames.yaml
# python train_trsp.py -c config/trsp/psl_v2/ppo_ames.yaml

python train_trsp.py -c config/trsp/psl_v2/sarsa_rand_grid_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/sarsa_rand_grid_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/sarsa_rand_grid_K4.yaml
python train_trsp.py -c config/trsp/psl_v2/dqn_rand_grid_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/dqn_rand_grid_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/dqn_rand_grid_K4.yaml
python train_trsp.py -c config/trsp/psl_v2/a2c_rand_grid_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/a2c_rand_grid_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/a2c_rand_grid_K4.yaml
python train_trsp.py -c config/trsp/psl_v2/ppo_rand_grid_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/ppo_rand_grid_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/ppo_rand_grid_K4.yaml

python train_trsp.py -c config/trsp/psl_v2/sarsa_ames_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/sarsa_ames_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/sarsa_ames_K4.yaml
python train_trsp.py -c config/trsp/psl_v2/dqn_ames_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/dqn_ames_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/dqn_ames_K4.yaml
python train_trsp.py -c config/trsp/psl_v2/a2c_ames_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/a2c_ames_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/a2c_ames_K4.yaml
python train_trsp.py -c config/trsp/psl_v2/ppo_ames_K2.yaml
python train_trsp.py -c config/trsp/psl_v2/ppo_ames_K3.yaml
python train_trsp.py -c config/trsp/psl_v2/ppo_ames_K4.yaml
