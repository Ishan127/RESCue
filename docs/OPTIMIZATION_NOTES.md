# Performance Optimization Summary

## GPU Utilization (4x 256GB MI325X)

| GPU | Component | Model | Port |
|-----|-----------|-------|------|
| 0,1 | Verifier (tensor-parallel=2) | Qwen3-VL-32B-Thinking | 8000 |
| 2   | Planner | Qwen3-VL-8B-Instruct | 8002 |
| 3   | SAM3 Executor | SAM-2.1 | 8001 |

## vLLM Configuration

### Verifier (deploy_llm.sh)
- `--max-num-seqs 64` - Process up to 64 concurrent requests
- `--tensor-parallel-size 2` - Split 32B model across 2 GPUs
- `--max-model-len 32768` - Support long context
- `--gpu-memory-utilization 0.9`

### Planner (deploy_planner_llm.sh)
- `--max-num-seqs 64` - Process up to 64 concurrent requests  
- `--max-model-len 32768` - Support long context
- `--gpu-memory-utilization 0.95`

## Tournament Verification (Full Ranking)

For N=64 candidates:
- **Rounds**: 6 (log2(64))
- **Comparisons**: 63 total (N-1)
- **Parallelism**: All comparisons in each round run simultaneously

### Round Breakdown:
| Round | Comparisons | Parallel Workers |
|-------|-------------|------------------|
| 1 | 32 | 32 |
| 2 | 16 | 16 |
| 3 | 8 | 8 |
| 4 | 4 | 4 |
| 5 | 2 | 2 |
| 6 | 1 | 1 |

### Expected Time:
- Each round: ~3-5s (batched on GPU)
- Total: ~18-30s for full 64-candidate ranking

## Hypothesis Generation (Parallel)

- 8 strategies × 8 variations = 64 hypotheses
- 32 parallel workers in ThreadPoolExecutor
- vLLM batches up to 64 concurrent requests
- Expected time: ~5-10s for 64 hypotheses

## Full Pipeline Time Estimate

| Stage | Expected Time |
|-------|--------------|
| Hypothesis Generation | 5-10s |
| SAM3 Mask Generation | 10-20s (parallel) |
| Tournament Verification | 18-30s |
| **Total** | **~45-60s per sample** |

## Evaluation (Multi-N)

The evaluation script:
1. Generates 64 candidates ONCE
2. Gets full ranking via tournament
3. For each N in [1,2,4,8,16,32,64]:
   - Takes best candidate among first N according to ranking
   - Computes IoU against ground truth

This gives proper evaluation for all N values without re-running the pipeline.

## Running Evaluation

```bash
# Start all services first
bash scripts/deploy_llm.sh      # Verifier on GPU 0,1
bash scripts/deploy_planner_llm.sh  # Planner on GPU 2
bash scripts/deploy_sam.sh     # SAM3 on GPU 3

# Run evaluation
cd /path/to/RESCue
python scripts/evaluate_multi_n.py --fraction 0.1 --mode comparative
```
