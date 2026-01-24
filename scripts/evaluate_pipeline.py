"""
Pipeline-Parallel Evaluation Script

Uses a queue-based system to process multiple samples concurrently:
- While Verifier processes sample N
- Executor processes sample N+1  
- Planner processes sample N+2

This maximizes GPU utilization across all 4 GPUs.
"""
import argparse
import sys
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Parse args FIRST
parser = argparse.ArgumentParser(description="Pipeline-Parallel Evaluation")
parser.add_argument("--fraction", type=float, default=0.1)
parser.add_argument("--max_n", type=int, default=64)
parser.add_argument("--planner_url", default="http://localhost:8002/v1")
parser.add_argument("--verifier_url", default="http://localhost:8000/v1")
parser.add_argument("--executor_url", default="http://localhost:8001")
parser.add_argument("--pipeline_depth", type=int, default=3, help="Number of samples to process concurrently")
parser.add_argument("--mode", choices=["comparative", "heuristic"], default="comparative")
args = parser.parse_args()

os.environ["PLANNER_API_BASE"] = args.planner_url
os.environ["VERIFIER_API_BASE"] = args.verifier_url

from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.planner import Planner
from src.executor import Executor
from src.verifier import Verifier
from src.utils import calculate_iou


@dataclass
class SampleTask:
    """A sample moving through the pipeline."""
    sample_idx: int
    image: Any
    query: str
    gt_mask: Optional[np.ndarray]
    temp_img_path: str
    
    # Filled by planner stage
    hypotheses: List[Dict] = field(default_factory=list)
    
    # Filled by executor stage
    candidates: List[Dict] = field(default_factory=list)
    
    # Filled by verifier stage
    ranking: List[int] = field(default_factory=list)
    
    # Timing
    t_start: float = 0
    t_plan_done: float = 0
    t_exec_done: float = 0
    t_verify_done: float = 0


class PipelineStage:
    """Base class for pipeline stages."""
    def __init__(self, name: str, input_queue: queue.Queue, output_queue: queue.Queue):
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.processed = 0
    
    def process(self, task: SampleTask) -> SampleTask:
        raise NotImplementedError
    
    def run(self):
        while self.running:
            try:
                task = self.input_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    self.output_queue.put(None)
                    break
                
                task = self.process(task)
                self.processed += 1
                self.output_queue.put(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                continue


class PlannerStage(PipelineStage):
    """Stage 1: Generate hypotheses from query."""
    
    def __init__(self, input_queue, output_queue, planner_url, max_n):
        super().__init__("Planner", input_queue, output_queue)
        self.planner = Planner(api_base=planner_url)
        self.max_n = max_n
    
    def process(self, task: SampleTask) -> SampleTask:
        task.t_start = time.time()
        
        try:
            hypotheses = self.planner.generate_hypotheses(task.temp_img_path, task.query, N=self.max_n)
            task.hypotheses = hypotheses if hypotheses else []
        except Exception as e:
            print(f"[Planner] Sample {task.sample_idx} error: {e}")
            task.hypotheses = [{"query": task.query, "bbox": None}]
        
        task.t_plan_done = time.time()
        return task


class ExecutorStage(PipelineStage):
    """Stage 2: Generate masks for each hypothesis."""
    
    def __init__(self, input_queue, output_queue, executor_url):
        super().__init__("Executor", input_queue, output_queue)
        self.executor = Executor(remote_url=executor_url)
    
    def process(self, task: SampleTask) -> SampleTask:
        if not task.hypotheses:
            print(f"[Executor] Sample {task.sample_idx}: No hypotheses to process")
            task.candidates = []
            task.t_exec_done = time.time()
            return task
        
        print(f"[Executor] Sample {task.sample_idx}: Processing {len(task.hypotheses)} hypotheses")
        
        candidates = []
        
        # Process hypotheses SEQUENTIALLY to avoid overwhelming SAM server
        for i, hyp in enumerate(task.hypotheses):
            try:
                # Load fresh image for each request
                image = Image.open(task.temp_img_path).copy()
                
                # Extract box and query from hypothesis
                box = hyp.get("box") or hyp.get("bbox")
                query = hyp.get("noun_phrase") or hyp.get("query") or task.query
                
                # Call executor
                masks = self.executor.execute(image, box, query)
                
                if masks and len(masks) > 0:
                    mask = masks[0]
                    result = {"hypothesis": hyp, "mask": mask}
                    if task.gt_mask is not None:
                        result["iou"] = calculate_iou(mask, task.gt_mask)
                    candidates.append(result)
                    
            except Exception as e:
                if i < 3:  # Only print first few errors
                    print(f"[Executor] Hypothesis {i} error: {e}")
        
        task.candidates = candidates
        print(f"[Executor] Sample {task.sample_idx}: Got {len(task.candidates)} masks")
        task.t_exec_done = time.time()
        return task


class VerifierStage(PipelineStage):
    """Stage 3: Rank candidates using tournament."""
    
    def __init__(self, input_queue, output_queue, verifier_url, mode):
        super().__init__("Verifier", input_queue, output_queue)
        self.verifier = Verifier()
        self.mode = mode
    
    def process(self, task: SampleTask) -> SampleTask:
        if not task.candidates:
            task.ranking = []
            task.t_verify_done = time.time()
            return task
        
        masks = [c['mask'] for c in task.candidates]
        
        try:
            if self.mode == "heuristic":
                # Quick heuristic ranking
                scores = [self._heuristic_score(m) for m in masks]
                task.ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            else:
                # Full VLM tournament
                results = self.verifier.verify_batch_comparative(task.image, masks, task.query)
                sorted_results = sorted(results, key=lambda r: r['rank'])
                task.ranking = [r['mask_idx'] for r in sorted_results]
        except Exception as e:
            print(f"[Verifier] Sample {task.sample_idx} error: {e}")
            task.ranking = list(range(len(task.candidates)))
        
        task.t_verify_done = time.time()
        return task
    
    def _heuristic_score(self, mask):
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        coverage = np.sum(mask_np) / mask_np.size
        if coverage < 0.01 or coverage > 0.8:
            return 0
        return 50 + (0.3 - abs(coverage - 0.3)) * 100


def run_pipeline_evaluation(fraction, max_n, planner_url, verifier_url, executor_url, 
                            pipeline_depth, mode):
    print(f"Loading ReasonSeg dataset...")
    try:
        ds = load_dataset("Ricky06662/ReasonSeg_val", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_samples = len(ds)
    num_samples = max(1, int(total_samples * fraction))
    print(f"Samples: {num_samples}/{total_samples} ({fraction*100:.0f}%)")
    
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    # Create queues
    q_input = queue.Queue(maxsize=pipeline_depth * 2)
    q_planned = queue.Queue(maxsize=pipeline_depth * 2)
    q_executed = queue.Queue(maxsize=pipeline_depth * 2)
    q_output = queue.Queue()
    
    # Create stages
    planner_stage = PlannerStage(q_input, q_planned, planner_url, max_n)
    executor_stage = ExecutorStage(q_planned, q_executed, executor_url)
    verifier_stage = VerifierStage(q_executed, q_output, verifier_url, mode)
    
    # Start stage threads
    threads = [
        threading.Thread(target=planner_stage.run, name="Planner"),
        threading.Thread(target=executor_stage.run, name="Executor"),
        threading.Thread(target=verifier_stage.run, name="Verifier"),
    ]
    
    for t in threads:
        t.start()
    
    print(f"\n{'='*70}")
    print(f"Pipeline Evaluation | Depth={pipeline_depth} | Mode={mode}")
    print(f"{'='*70}\n")
    
    # Feed samples into pipeline
    temp_files = []
    
    def feed_samples():
        for sample_idx, sample in enumerate(ds):
            image = sample.get('image')
            query = sample.get('text') or sample.get('query') or sample.get('sentence')
            gt_mask = sample.get('mask') or sample.get('label')
            
            if image is None or query is None:
                continue
            
            if gt_mask is not None:
                gt_mask = np.array(gt_mask) > 0
            
            temp_img_path = f"temp_pipeline_{sample_idx}.jpg"
            image.save(temp_img_path)
            temp_files.append(temp_img_path)
            
            task = SampleTask(
                sample_idx=sample_idx,
                image=image,
                query=query,
                gt_mask=gt_mask,
                temp_img_path=temp_img_path
            )
            
            q_input.put(task)
        
        # Send poison pills
        q_input.put(None)
    
    # Start feeder thread
    feeder = threading.Thread(target=feed_samples, name="Feeder")
    feeder.start()
    
    # Collect results
    N_VALUES = [1, 2, 4, 8, 16, 32, 64]
    results_by_n = {n: {'ious': [], 'oracle_ious': [], 'times': []} for n in N_VALUES if n <= max_n}
    
    completed = 0
    pbar = tqdm(total=num_samples, desc="Processing")
    
    while completed < num_samples:
        try:
            task = q_output.get(timeout=300)  # 5 min timeout
            if task is None:
                break
            
            completed += 1
            pbar.update(1)
            
            total_time = task.t_verify_done - task.t_start
            plan_time = task.t_plan_done - task.t_start
            exec_time = task.t_exec_done - task.t_plan_done
            verify_time = task.t_verify_done - task.t_exec_done
            
            pbar.set_postfix({
                'plan': f'{plan_time:.1f}s',
                'exec': f'{exec_time:.1f}s',
                'verify': f'{verify_time:.1f}s',
                'total': f'{total_time:.1f}s'
            })
            
            if not task.candidates or task.gt_mask is None:
                continue
            
            # Evaluate for each N
            for n in results_by_n.keys():
                if n > len(task.candidates):
                    continue
                
                # Find best among first N according to ranking
                if task.ranking:
                    best_idx = min((i for i in task.ranking if i < n), default=0)
                else:
                    best_idx = 0
                
                pred_mask = task.candidates[best_idx]['mask']
                iou = calculate_iou(pred_mask, task.gt_mask)
                oracle_iou = max(c.get('iou', 0) for c in task.candidates[:n])
                
                results_by_n[n]['ious'].append(iou)
                results_by_n[n]['oracle_ious'].append(oracle_iou)
                results_by_n[n]['times'].append(total_time)
                
        except queue.Empty:
            print("Timeout waiting for results")
            break
    
    pbar.close()
    
    # Wait for threads
    feeder.join()
    for t in threads:
        t.join(timeout=5)
    
    # Cleanup temp files
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Print results
    print_results(results_by_n, mode)
    
    # Save results
    output_file = f"results_pipeline_{mode}_{int(fraction*100)}pct.json"
    save_results(results_by_n, output_file)


def print_results(results_by_n, mode):
    print(f"\n{'='*70}")
    print(f"RESULTS ({mode.upper()} - Pipeline Parallel)")
    print(f"{'='*70}")
    
    print(f"\n{'N':>4} | {'gIoU':>8} | {'Oracle':>8} | {'%Oracle':>8} | {'Time(s)':>8} | {'Samples':>7}")
    print("-" * 60)
    
    for n in sorted(results_by_n.keys()):
        data = results_by_n[n]
        if not data['ious']:
            continue
        
        mean_iou = np.mean(data['ious'])
        mean_oracle = np.mean(data['oracle_ious'])
        pct_oracle = mean_iou / mean_oracle * 100 if mean_oracle > 0 else 0
        mean_time = np.mean(data['times'])
        
        print(f"{n:>4} | {mean_iou:>8.4f} | {mean_oracle:>8.4f} | {pct_oracle:>7.1f}% | {mean_time:>8.2f} | {len(data['ious']):>7}")


def save_results(results_by_n, output_file):
    output = {}
    for n, data in results_by_n.items():
        output[str(n)] = {
            'mean_iou': float(np.mean(data['ious'])) if data['ious'] else 0,
            'mean_oracle': float(np.mean(data['oracle_ious'])) if data['oracle_ious'] else 0,
            'mean_time': float(np.mean(data['times'])) if data['times'] else 0,
            'num_samples': len(data['ious'])
        }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_pipeline_evaluation(
        fraction=args.fraction,
        max_n=args.max_n,
        planner_url=args.planner_url,
        verifier_url=args.verifier_url,
        executor_url=args.executor_url,
        pipeline_depth=args.pipeline_depth,
        mode=args.mode
    )
