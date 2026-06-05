#!/usr/bin/env bash
# train_eval_K34.sh
# Trains all agent × network × K combinations for K=3,4 (slot_duration=60s)
# and evaluates each policy against the Dijkstra baseline.
set -euo pipefail

PYTHON=/home/shinobi-owl/anaconda3/envs/clean_rl_2/bin/python
CONFIG_DIR=config/trsp/psl_K34

# Order: K3 configs first, then K4, grouped by agent for readability.
CONFIGS=(
    # # ── SARSA ──────────────────────────────────────────────────────────
    # $CONFIG_DIR/sarsa_grid_road_K3.yaml
    # $CONFIG_DIR/sarsa_rand_grid_K3.yaml
    # $CONFIG_DIR/sarsa_ames_K3.yaml
    # $CONFIG_DIR/sarsa_grid_road_K4.yaml
    # $CONFIG_DIR/sarsa_rand_grid_K4.yaml
    # $CONFIG_DIR/sarsa_ames_K4.yaml

    # ── DQN ────────────────────────────────────────────────────────────
    # $CONFIG_DIR/dqn_grid_road_K3.yaml
    # $CONFIG_DIR/dqn_rand_grid_K3.yaml
    # $CONFIG_DIR/dqn_ames_K3.yaml
    $CONFIG_DIR/dqn_grid_road_K4.yaml
    $CONFIG_DIR/dqn_rand_grid_K4.yaml
    $CONFIG_DIR/dqn_ames_K4.yaml

    # ── PPO ────────────────────────────────────────────────────────────
    $CONFIG_DIR/ppo_grid_road_K3.yaml
    $CONFIG_DIR/ppo_rand_grid_K3.yaml
    $CONFIG_DIR/ppo_ames_K3.yaml
    $CONFIG_DIR/ppo_grid_road_K4.yaml
    $CONFIG_DIR/ppo_rand_grid_K4.yaml
    $CONFIG_DIR/ppo_ames_K4.yaml

    # ── A2C ────────────────────────────────────────────────────────────
    $CONFIG_DIR/a2c_grid_road_K3.yaml
    $CONFIG_DIR/a2c_rand_grid_K3.yaml
    $CONFIG_DIR/a2c_ames_K3.yaml
    $CONFIG_DIR/a2c_grid_road_K4.yaml
    $CONFIG_DIR/a2c_rand_grid_K4.yaml
    $CONFIG_DIR/a2c_ames_K4.yaml
)

TOTAL=${#CONFIGS[@]}
IDX=0

for CONFIG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "============================================================"
    echo "[$IDX/$TOTAL] Training: $CONFIG"
    echo "============================================================"

    BEFORE=$(ls runs/trsp/ | sort)

    $PYTHON train_trsp.py -c "$CONFIG"

    AFTER=$(ls runs/trsp/ | sort)
    POLICY_ID=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1)

    if [ -z "$POLICY_ID" ]; then
        echo "ERROR: could not detect new policy directory after training $CONFIG"
        exit 1
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[$IDX/$TOTAL] Evaluating policy: $POLICY_ID  (config: $CONFIG)"
    echo "------------------------------------------------------------"

    $PYTHON eval_trsp2.py -p "$POLICY_ID" --dijkstra

    echo ""
done

echo "============================================================"
echo "All $TOTAL train+eval runs complete."
echo "============================================================"
