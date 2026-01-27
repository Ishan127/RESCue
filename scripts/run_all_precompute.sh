#!/bin/bash
# ============================================================
# MASTER PRECOMPUTE: Run all 4 phases sequentially
# Each phase deploys its own services and shuts them down when done
# ============================================================

set -e

echo "=============================================="
echo "MASTER PRECOMPUTE PIPELINE"
echo "=============================================="
echo "This will run all 4 phases:"
echo "  Phase 1: Plans (8 GPU Planner)"
echo "  Phase 2: Masks (SAM)"
echo "  Phase 3: CLIP Scores"
echo "  Phase 4: VLM Scores (8 GPU Verifier)"
echo ""
echo "Total estimated time: 2-4 hours (779 samples × 512 hypotheses)"
echo "=============================================="
read -p "Press Enter to continue or Ctrl+C to cancel..."

START_TIME=$(date +%s)

# Phase 1: Plans
echo ""
bash scripts/phase1_plans.sh

# Phase 2: Masks
echo ""
bash scripts/phase2_masks.sh

# Phase 3: CLIP
echo ""
bash scripts/phase3_clip.sh

# Phase 4: VLM
echo ""
bash scripts/phase4_vlm.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "ALL PHASES COMPLETE!"
echo "=============================================="
echo "Total time: $((ELAPSED / 60)) minutes"
echo ""
echo "Cache contents:"
echo "  cache/plans/plans.json     - 512 hypotheses per sample"
echo "  cache/masks/sample_*/      - Masks + CLIP + VLM scores"
echo ""
echo "Run experiments with:"
echo "  bash scripts/run_tournament.sh 16"
echo "  bash scripts/run_tournament.sh 32 --strategy spatial"
echo "=============================================="
