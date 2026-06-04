#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/shinobi-owl/anaconda3/envs/clean_rl_2/bin/python

CONFIGS=(
    config/trsp/a2c_grid_road_K1.yaml
    config/trsp/a2c_grid_road_K2.yaml
    config/trsp/a2c_rand_grid_K1.yaml
    config/trsp/a2c_rand_grid_K2.yaml
    config/trsp/a2c_ames_K1.yaml
    config/trsp/a2c_ames_K2.yaml
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "Training: $CONFIG"
    echo "============================================================"

    # snapshot existing policy dirs before training
    BEFORE=$(ls runs/trsp/ | sort)

    $PYTHON train_trsp.py -c "$CONFIG"

    # find the newly created policy dir
    AFTER=$(ls runs/trsp/ | sort)
    POLICY_ID=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1)

    if [ -z "$POLICY_ID" ]; then
        echo "ERROR: could not detect new policy directory after training $CONFIG"
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "Evaluating policy: $POLICY_ID  (config: $CONFIG)"
    echo "============================================================"

    $PYTHON eval_trsp2.py -p "$POLICY_ID" --dijkstra

    echo ""
done

echo "All A2C training and evaluation runs complete."
