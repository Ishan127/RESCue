#!/bin/bash
# Run RESCue evaluation with N=128 hypotheses
# Maximum diversity, highest accuracy, longest runtime

echo "=== RESCue Evaluation: N=128 ==="
echo "Expected time per sample: ~25-40s"

python scripts/evaluate_pipeline.py \
    --max_n 128 \
    --parallel_requests 8 \
    --pipeline_depth 16 \
    --mode comparative \
    --fraction 1.0 \
    "$@"
