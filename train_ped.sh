#!/usr/bin/env bash
# Train all four ped TRSP policies sequentially:
#   (rand_grid, sarsa) (rand_grid, dqn) (ames, sarsa) (ames, dqn)
# Each trains on its reduced 2000-ped TRAIN sumocfg (fast). This script does
# NOT evaluate — eval requires manually editing each saved policy config
# (runs/trsp/<POLICY_ID>/config.yaml): point net.net_path at the FULL sumocfg
# and restore traffic.ped_speed_override.gamma to gamma_eval, then run
# eval_trsp2.py.
#
# Usage (from an activated project env, e.g. `conda activate clean_rl_2`):
#   ./train_ped.sh
set -euo pipefail
cd "$(dirname "$0")"

CONFIGS=(
  config/trsp/ped/sarsa_rand_grid.yaml
  config/trsp/ped/dqn_rand_grid.yaml
  config/trsp/ped/sarsa_ames.yaml
  config/trsp/ped/dqn_ames.yaml
)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="train_ped_${STAMP}.log"
echo "Logging to ${LOG}"

for cfg in "${CONFIGS[@]}"; do
  echo "==================================================================" | tee -a "${LOG}"
  echo "Training: ${cfg}  ($(date))"                                        | tee -a "${LOG}"
  echo "==================================================================" | tee -a "${LOG}"
  python train_trsp.py -c "${cfg}" 2>&1 | tee -a "${LOG}"
done

echo ""
echo "All four trainings complete. Config -> POLICY_ID:"
paste -d' ' \
  <(printf '%s\n' "${CONFIGS[@]}") \
  <(grep -oE 'POLICY_ID=[0-9a-f]+' "${LOG}" | cut -d= -f2)
echo ""
echo "Next (manual): for each runs/trsp/<POLICY_ID>/config.yaml set net.net_path"
echo "to the FULL sumocfg and gamma to gamma_eval, then run eval_trsp2.py."
