#!/bin/bash
# Run RESCue evaluation with N=32 hypotheses
# Good balance of diversity and speed

echo "=== RESCue Evaluation: N=32 ==="
echo "Expected time per sample: ~12-18s"

python scripts/evaluate_pipeline.py \
    --max_n 32 \
    --parallel_requests 6 \
    --workers_planner 2 \
    --workers_executor 4 \
    --workers_verifier 6 \
    --mode comparative \
    --fraction 1.0 \
    "$@"
