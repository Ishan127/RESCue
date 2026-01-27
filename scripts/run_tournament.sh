#!/bin/bash
# Tournament-only evaluation using precomputed scores
#
# Usage:
#   bash scripts/run_tournament.sh 16                    # N=16, all strategies
#   bash scripts/run_tournament.sh 32 --strategy spatial # N=32, spatial only
#   bash scripts/run_tournament.sh 128                   # N=128
#   bash scripts/run_tournament.sh 16 --skip_tournament  # Skip tournament, use pointwise only

MAX_N=${1:-16}
WORKERS=${WORKERS:-64}

echo "=============================================="
echo "RESCUE TOURNAMENT EVALUATION"
echo "=============================================="
echo "Max N: $MAX_N"
echo "Workers: $WORKERS"
echo "=============================================="

python scripts/evaluate_tournament.py \
    --max_n $MAX_N \
    --cache_dir cache \
    --workers $WORKERS \
    --fraction 1.0 \
    "${@:2}"
