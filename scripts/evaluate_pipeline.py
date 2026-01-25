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
parser.add_argument("--executor_url", default="http://localhost:8001",
                    help="SAM load balancer URL (single endpoint, internally routes to multiple backends)")
parser.add_argument("--parallel_requests", type=int, default=8,
                    help="Number of parallel requests to SAM cluster")
parser.add_argument("--pipeline_depth", type=int, default=3, help="Number of samples to process concurrently")
parser.add_argument("--mode", choices=["comparative", "heuristic"], default="comparative")
parser.add_argument("--workers_planner", type=int, default=2)
parser.add_argument("--workers_executor", type=int, default=4)
parser.add_argument("--workers_verifier", type=int, default=4)
args = parser.parse_args()

print(f"Using SAM cluster at {args.executor_url} with {args.parallel_requests} parallel requests")

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
                    # Do NOT propagate None here. Coordination is handled by monitor.
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
        # Each worker needs its own planner instance for thread safety if not handled internally
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
    """Stage 2: Generate masks using SAM cluster (load-balanced)."""
    
    def __init__(self, input_queue, output_queue, executor_url, parallel_requests=16):
        super().__init__("Executor", input_queue, output_queue)
        # Single executor per worker (requests Session is thread-safe, but safer to have separate instances)
        self.executor = Executor(remote_url=executor_url, timeout=300)
        self.parallel_requests = parallel_requests
    
    def process(self, task: SampleTask) -> SampleTask:
        if not task.hypotheses:
            task.candidates = []
            task.t_exec_done = time.time()
            return task
        
        # Load fresh image copy
        image = Image.open(task.temp_img_path).copy()
        
        prompts_list = []
        for hyp in task.hypotheses:
            box = hyp.get("box") or hyp.get("bbox")
            prompts_list.append({
                "type": "box",
                "box": box,
                "label": True
            })

        try:
             masks = self.executor.segment(image, prompts_list=prompts_list)
             
             results = []
             for i, (hyp, mask) in enumerate(zip(task.hypotheses, masks)):
                 res = {"hypothesis": hyp, "mask": mask}
                 if task.gt_mask is not None:
                     res["iou"] = calculate_iou(mask, task.gt_mask)
                 results.append(res)
                 
             task.candidates = results
             
        except Exception as e:
            print(f"[Executor] Sample {task.sample_idx} Batch Error: {e}")
            task.candidates = []

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
                scores = [self._heuristic_score(m) for m in masks]
                task.ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            else:
                results = self.verifier.verify_batch_pointwise(task.image, masks, task.query)
                
                sorted_results = sorted(results, key=lambda r: r['rank'])
                task.ranking = [r['mask_idx'] for r in sorted_results]
                
                for res in sorted_results:
                     idx = res['mask_idx']
                     if idx < len(task.candidates):
                         task.candidates[idx]['verifier_score'] = res['score']
                         task.candidates[idx]['verifier_reasoning'] = res.get('reasoning')
                         if 'pointwise_details' in res:
                             task.candidates[idx]['pointwise_details'] = res['pointwise_details']
                
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
                            parallel_requests, pipeline_depth, mode):
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
    
    # Larger queues for higher throughput
    queue_size = pipeline_depth * 4
    q_input = queue.Queue(maxsize=queue_size)
    q_planned = queue.Queue(maxsize=queue_size)
    q_executed = queue.Queue(maxsize=queue_size)
    q_output = queue.Queue()
    
    # Init workers
    planner_workers = []
    for i in range(args.workers_planner):
        w = PlannerStage(q_input, q_planned, planner_url, max_n)
        planner_workers.append(threading.Thread(target=w.run, name=f"Planner-{i}"))

    executor_workers = []
    for i in range(args.workers_executor):
        w = ExecutorStage(q_planned, q_executed, executor_url, parallel_requests)
        executor_workers.append(threading.Thread(target=w.run, name=f"Executor-{i}"))

    verifier_workers = []
    for i in range(args.workers_verifier):
        w = VerifierStage(q_executed, q_output, verifier_url, mode)
        verifier_workers.append(threading.Thread(target=w.run, name=f"Verifier-{i}"))
    
    all_workers = planner_workers + executor_workers + verifier_workers
    for t in all_workers:
        t.start()
    
    print(f"\n{'='*70}")
    print(f"Pipeline Evaluation | Samples={num_samples}")
    print(f"Workers: Planner={args.workers_planner}, Exec={args.workers_executor}, Verify={args.workers_verifier}")
    print(f"{'='*70}\n")
    
    # Feed samples
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
            
            # Unique temp file for each sample
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
    
    def monitor_and_shutdown():
        # 1. Feed inputs
        feed_samples()
        
        # 2. Shutdown Planner (send N pills)
        for _ in range(args.workers_planner):
            q_input.put(None)
        
        # 3. Wait for planners
        for t in planner_workers:
            t.join()
            
        # 4. Shutdown Executor
        for _ in range(args.workers_executor):
            q_planned.put(None)
            
        # 5. Wait for executors
        for t in executor_workers:
            t.join()
            
        # 6. Shutdown Verifier
        for _ in range(args.workers_verifier):
            q_executed.put(None)
            
        # 7. Wait for verifiers
        for t in verifier_workers:
            t.join()
            
        # 8. Signal output loop
        q_output.put(None)

    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_and_shutdown, name="Monitor")
    monitor_thread.start()
    
    # Collect results
    N_VALUES = [1, 2, 4, 8, 16, 32, 64]
    results_by_n = {n: {'ious': [], 'oracle_ious': [], 'times': []} for n in N_VALUES if n <= max_n}
    detailed_samples = []
    
    completed = 0
    pbar = tqdm(total=num_samples, desc="Processing")
    
    while completed < num_samples:
        try:
            task = q_output.get(timeout=1200)
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

            # NEW: Collect detailed sample info
            sample_info = {
                'sample_idx': task.sample_idx,
                'query': task.query,
                'candidates': []
            }
            
            # Create a map from candidate index to rank
            rank_map = {}
            if task.ranking:
                for rank, candidate_idx in enumerate(task.ranking):
                    rank_map[candidate_idx] = rank
            
            for i, cand in enumerate(task.candidates):
                hyp = cand.get('hypothesis', {})
                iou_score = float(cand.get('iou', 0))
                
                box = hyp.get('box')
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                
                cand_info = {
                    'hypothesis': hyp.get('noun_phrase') or hyp.get('raw_text'),
                    'reasoning': hyp.get('reasoning'),
                    'box': box,
                    'iou': iou_score,
                    'verifier_rank': rank_map.get(i, -1),
                    'verifier_score': cand.get('verifier_score'),
                    'verifier_reasoning': cand.get('verifier_reasoning'),
                    # 'pointwise_breakdown': cand.get('pointwise_details', {}).get('breakdown'), # Save space
                }
                sample_info['candidates'].append(cand_info)
            
            detailed_samples.append(sample_info)

            # Evaluate for each N
            for n in results_by_n.keys():
                if n > len(task.candidates):
                    continue
                
                if n == 1:
                    best_idx = 0
                elif task.ranking:
                    best_idx = next((i for i in task.ranking if i < n), 0)
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
    monitor_thread.join()
    
    # Cleanup temp files
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except: 
                pass
    
    # Print results
    print_results(results_by_n, mode)
    
    # Save results
    output_file = f"results_pipeline_{mode}_{int(fraction*100)}pct.json"
    save_results(results_by_n, output_file, detailed_samples)


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


def save_results(results_by_n, output_file, detailed_samples=None):
    output = {
        'metrics': {},
        'samples': detailed_samples if detailed_samples else []
    }
    for n, data in results_by_n.items():
        output['metrics'][str(n)] = {
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
        parallel_requests=args.parallel_requests,
        pipeline_depth=args.pipeline_depth,
        mode=args.mode
    )
