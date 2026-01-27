#!/bin/bash
# Run RESCue evaluation with N=16 hypotheses
# Optimized for quick experiments and debugging

echo "=== RESCue Evaluation: N=16 ==="
echo "Expected time per sample: ~8-12s"

python scripts/evaluate_pipeline.py \
    --max_n 16 \
    --parallel_requests 4 \
    --pipeline_depth 8 \
    --mode comparative \
    --fraction 1.0 \
    "$@"
