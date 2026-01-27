#!/bin/bash
# Run RESCue evaluation with N=128 hypotheses
# Maximum diversity, highest accuracy, longest runtime

echo "=== RESCue Evaluation: N=128 ==="
echo "Expected time per sample: ~25-40s"

python scripts/evaluate_pipeline.py \
    --max_n 128 \
    --parallel_requests 8 \
    --workers_planner 2 \
    --workers_executor 6 \
    --workers_verifier 8 \
    --mode comparative \
    --fraction 1.0 \
    "$@"
