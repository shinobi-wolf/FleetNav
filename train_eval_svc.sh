#!/usr/bin/env bash
# Train and evaluate all SVC agents (sarsa_svc, a2c_svc, ppo_svc) on
# all three networks (grid_road, rand_grid_road, ames).
# Each config is in config/trsp/svc/; results land in runs/trsp/<policy_id>/.
#
# Usage:
#   conda activate clean_rl_2
#   cd /path/to/DroneSim
#   bash train_eval_svc.sh [--skip-train] [--skip-eval]
#
# Flags:
#   --skip-train   Evaluate only (reads policy IDs from logs/svc_policy_ids.txt)
#   --skip-eval    Train only

set -euo pipefail

PYTHON=/home/shinobi-owl/anaconda3/envs/clean_rl_2/bin/python
LOGDIR=logs/svc
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
    config/trsp/svc/sarsa_grid_road.yaml
    config/trsp/svc/sarsa_rand_grid.yaml
    config/trsp/svc/sarsa_ames.yaml
    config/trsp/svc/a2c_grid_road.yaml
    config/trsp/svc/a2c_rand_grid.yaml
    config/trsp/svc/a2c_ames.yaml
    config/trsp/svc/ppo_grid_road.yaml
    config/trsp/svc/ppo_rand_grid.yaml
    config/trsp/svc/ppo_ames.yaml
)

POLICY_IDS=()

# ── TRAINING ──────────────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo "════════════════════════════════════════"
    echo " SVC Training  ($(date '+%Y-%m-%d %H:%M'))"
    echo "════════════════════════════════════════"

    for cfg in "${CONFIGS[@]}"; do
        tag=$(basename "$cfg" .yaml)
        echo ""
        echo "── Training: $tag ──"
        train_log="$LOGDIR/train_${tag}.log"

        # Run training; pipe through tee for live output but don't grep the pipe
        # (grep | head in a pipeline would send SIGPIPE and kill the training process)
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
    echo " SVC Evaluation  ($(date '+%Y-%m-%d %H:%M'))"
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
