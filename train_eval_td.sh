#!/usr/bin/env bash
# Train and evaluate all TD traffic agents on all networks.
# Each config is in config/trsp/td/; results land in runs/trsp/<policy_id>/.
#
# Usage:
#   conda activate clean_rl_2
#   cd /path/to/DroneSim
#   bash train_eval_td.sh [--skip-train] [--skip-eval]
#
# Flags:
#   --skip-train   Evaluate only (reads policy IDs from logs/td/policy_ids.txt)
#   --skip-eval    Train only

set -euo pipefail

PYTHON=/home/shinobi-owl/anaconda3/envs/clean_rl_2/bin/python
LOGDIR=logs/td
POLICY_LOG="$LOGDIR/policy_ids.txt"
SKIP_TRAIN=false
SKIP_EVAL=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --skip-eval)  SKIP_EVAL=true  ;;
    esac
done

mkdir -p "$LOGDIR"
if [ "$SKIP_TRAIN" = false ]; then
    > "$POLICY_LOG"   # reset at the start of each full training run
fi

CONFIGS=(
    # config/trsp/td/sarsa_grid_road.yaml
    # config/trsp/td/sarsa_rand_grid.yaml
    # config/trsp/td/sarsa_ames.yaml
    # config/trsp/td/dqn_grid_road.yaml
    config/trsp/td/dqn_ames.yaml
    # config/trsp/td/a2c_grid_road.yaml
    config/trsp/td/a2c_rand_grid.yaml
    config/trsp/td/a2c_ames.yaml
    # config/trsp/td/ppo_grid_road.yaml
    config/trsp/td/ppo_rand_grid.yaml
    config/trsp/td/ppo_ames.yaml
)

POLICY_IDS=()

# ── TRAINING ──────────────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo "════════════════════════════════════════"
    echo " TD Training  ($(date '+%Y-%m-%d %H:%M'))"
    echo "════════════════════════════════════════"

    for cfg in "${CONFIGS[@]}"; do
        tag=$(basename "$cfg" .yaml)
        echo ""
        echo "── Training: $tag ──"
        train_log="$LOGDIR/train_${tag}.log"

        "$PYTHON" -u train_trsp.py -c "$cfg" 2>&1 | tee "$train_log"

        POLICY_ID=$(grep "^POLICY_ID=" "$train_log" | head -1 | cut -d= -f2)
        if [ -z "$POLICY_ID" ]; then
            echo "ERROR: could not parse POLICY_ID from $train_log" >&2
            exit 1
        fi

        echo "  → policy: $POLICY_ID  (log: $train_log)"
        POLICY_IDS+=("$POLICY_ID")
        echo "$tag=$POLICY_ID" >> "$POLICY_LOG"
    done

    echo ""
    echo "All training runs complete. Policy IDs saved to $POLICY_LOG"

else
    echo "Skipping training — reading policy IDs from $POLICY_LOG"
    if [ ! -f "$POLICY_LOG" ]; then
        echo "ERROR: $POLICY_LOG not found; run without --skip-train first." >&2
        exit 1
    fi
    while IFS='=' read -r _tag pid; do
        POLICY_IDS+=("$pid")
    done < "$POLICY_LOG"
fi

# ── EVALUATION ────────────────────────────────────────────────────────────────
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo " TD Evaluation  ($(date '+%Y-%m-%d %H:%M'))"
    echo "════════════════════════════════════════"

    for POLICY_ID in "${POLICY_IDS[@]}"; do
        echo ""
        echo "── Evaluating: $POLICY_ID ──"
        eval_log="$LOGDIR/eval_${POLICY_ID}.log"
        "$PYTHON" -u eval_trsp2.py -p "$POLICY_ID" 2>&1 | tee "$eval_log"
        echo "  → log: $eval_log"
    done

    echo ""
    echo "All evaluations complete."
fi

echo ""
echo "════════════════════════════════════════"
echo " Done  ($(date '+%Y-%m-%d %H:%M'))"
echo " Policy IDs: $POLICY_LOG"
echo "════════════════════════════════════════"
