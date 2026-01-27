#!/bin/bash
# Run RESCue evaluation with N=32 hypotheses
# Good balance of diversity and speed

echo "=== RESCue Evaluation: N=32 ==="
echo "Expected time per sample: ~12-18s"

python scripts/evaluate_pipeline.py \
    --max_n 32 \
    --parallel_requests 6 \
    --pipeline_depth 12 \
    --mode comparative \
    --fraction 1.0 \
    "$@"
