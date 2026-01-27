#!/bin/bash
# Run RESCue evaluation with N=16 hypotheses
# Optimized for quick experiments and debugging

echo "=== RESCue Evaluation: N=16 ==="
echo "Expected time per sample: ~8-12s"

python scripts/evaluate_pipeline.py \
    --max_n 16 \
    --parallel_requests 4 \
    --workers_planner 2 \
    --workers_executor 4 \
    --workers_verifier 4 \
    --mode comparative \
    --fraction 1.0 \
    "$@"
