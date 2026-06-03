#!/bin/bash
# Run eval_trsp2.py for every policy in runs/trsp/ (except dijkstra), with Dijkstra baseline.
set -euo pipefail

for policy_dir in runs/trsp/*/; do
    policy_id=$(basename "$policy_dir")
    [ "$policy_id" = "dijkstra" ] && continue
    if [ ! -f "${policy_dir}policy.pkl" ]; then
        echo "Skipping $policy_id (no policy.pkl)"
        continue
    fi
    echo "=== Evaluating $policy_id ==="
    python eval_trsp2.py -p "$policy_id" --dijkstra
done

echo "=== All evaluations complete ==="
