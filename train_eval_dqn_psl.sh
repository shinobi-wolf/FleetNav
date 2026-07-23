#!/usr/bin/env bash
# Train + evaluate DQN on the redesigned PSL benchmarks (2 networks x K=1-4).
# Each eval_trsp2.py run also produces the Dijkstra and DSP baselines automatically.
# Override the interpreter with:  PYTHON=/path/to/python bash train_eval_dqn_psl.sh
set -euo pipefail

PYTHON=${PYTHON:-/home/shinobi-wolf/anaconda3/envs/dronesim/bin/python}

CONFIGS=(
    config/trsp/psl/dqn_grid_K1.yaml
    config/trsp/psl/dqn_grid_K2.yaml
    config/trsp/psl/dqn_grid_K3.yaml
    config/trsp/psl/dqn_grid_K4.yaml
    config/trsp/psl/dqn_rand_grid_K1.yaml
    config/trsp/psl/dqn_rand_grid_K2.yaml
    config/trsp/psl/dqn_rand_grid_K3.yaml
    config/trsp/psl/dqn_rand_grid_K4.yaml
)

mkdir -p logs/psl
POLICY_LOG=logs/psl/policy_ids.txt
: > "$POLICY_LOG"

for cfg in "${CONFIGS[@]}"; do
    tag=$(basename "$cfg" .yaml)
    echo "======================================================================"
    echo "TRAIN  $tag"
    echo "======================================================================"
    "$PYTHON" -u train_trsp.py -c "$cfg" 2>&1 | tee "logs/psl/train_${tag}.log"

    policy_id=$(grep '^POLICY_ID=' "logs/psl/train_${tag}.log" | head -1 | cut -d= -f2)
    echo "$tag=$policy_id" >> "$POLICY_LOG"

    echo "----------------------------------------------------------------------"
    echo "EVAL   $tag  (policy $policy_id)"
    echo "----------------------------------------------------------------------"
    "$PYTHON" -u eval_trsp2.py -p "$policy_id" 2>&1 | tee "logs/psl/eval_${tag}.log"
done

echo "Done. Policy IDs: $POLICY_LOG"
