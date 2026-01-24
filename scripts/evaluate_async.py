"""
Async Pipeline Evaluation using asyncio.

This version uses async/await for maximum concurrency:
- Multiple samples processed simultaneously
- Each stage can work on different samples concurrently
- Non-blocking I/O for API calls

Usage:
    python evaluate_async.py --fraction 0.1 --concurrency 4
"""
import argparse
import sys
import os
import asyncio
import aiohttp
import base64
import time
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
import io

# Parse args FIRST
parser = argparse.ArgumentParser(description="Async Pipeline Evaluation")
parser.add_argument("--fraction", type=float, default=0.1)
parser.add_argument("--max_n", type=int, default=64)
parser.add_argument("--planner_url", default="http://localhost:8002/v1")
parser.add_argument("--verifier_url", default="http://localhost:8000/v1")
parser.add_argument("--executor_url", default="http://localhost:8001")
parser.add_argument("--concurrency", type=int, default=4, help="Samples processed concurrently")
parser.add_argument("--mode", choices=["comparative", "heuristic"], default="comparative")
args = parser.parse_args()

os.environ["PLANNER_API_BASE"] = args.planner_url
os.environ["VERIFIER_API_BASE"] = args.verifier_url

from datasets import load_dataset
from tqdm.asyncio import tqdm
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import calculate_iou, apply_red_alpha_overlay


@dataclass
class SampleResult:
    sample_idx: int
    candidates: List[Dict] = field(default_factory=list)
    ranking: List[int] = field(default_factory=list)
    gt_mask: Optional[np.ndarray] = None
    total_time: float = 0


class AsyncPipeline:
    """Async pipeline for RESCue evaluation."""
    
    def __init__(self, planner_url, verifier_url, executor_url, max_n, mode):
        self.planner_url = planner_url
        self.verifier_url = verifier_url
        self.executor_url = executor_url
        self.max_n = max_n
        self.mode = mode
        self.planner_model = os.environ.get("PLANNER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.verifier_model = os.environ.get("VERIFIER_MODEL", "Qwen/Qwen3-VL-32B-Thinking")
    
    def _encode_image(self, image):
        """Encode PIL Image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def _call_llm(self, session, url, model, messages, temperature=0.7, max_tokens=1024):
        """Make async LLM API call."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with session.post(f"{url}/chat/completions", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                text = await resp.text()
                raise Exception(f"LLM API error: {resp.status} - {text[:200]}")
    
    async def _call_sam(self, session, image_b64, hypothesis):
        """Make async SAM API call."""
        payload = {
            "image": image_b64,
            "query": hypothesis.get("query", ""),
            "bbox": hypothesis.get("bbox"),
            "point": hypothesis.get("point")
        }
        
        async with session.post(f"{self.executor_url}/segment", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("mask"):
                    # Decode mask from base64
                    mask_bytes = base64.b64decode(data["mask"])
                    mask_img = Image.open(io.BytesIO(mask_bytes))
                    return np.array(mask_img) > 0
            return None
    
    async def plan(self, session, image, query):
        """Generate hypotheses using planner."""
        image_b64 = self._encode_image(image)
        
        prompt = f"""Analyze this image for the query: "{query}"

Generate {self.max_n} different object hypotheses that might match this query.
For each hypothesis, provide a specific description and optional bounding box.

Output JSON array:
[{{"query": "specific object description", "bbox": [x1,y1,x2,y2] or null}}, ...]

Focus on diverse interpretations: different objects, regions, scales."""

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }]
        
        try:
            response = await self._call_llm(
                session, self.planner_url, self.planner_model,
                messages, temperature=0.8, max_tokens=4096
            )
            
            # Parse JSON
            import json_repair
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                hypotheses = json_repair.loads(json_match.group())
                return hypotheses[:self.max_n]
        except Exception as e:
            print(f"Planner error: {e}")
        
        return [{"query": query, "bbox": None}]
    
    async def execute(self, session, image, hypotheses, gt_mask=None):
        """Generate masks for all hypotheses concurrently."""
        image_b64 = self._encode_image(image)
        
        async def exec_one(hyp):
            try:
                mask = await self._call_sam(session, image_b64, hyp)
                if mask is not None:
                    result = {"hypothesis": hyp, "mask": mask}
                    if gt_mask is not None:
                        result["iou"] = calculate_iou(mask, gt_mask)
                    return result
            except Exception as e:
                pass
            return None
        
        tasks = [exec_one(h) for h in hypotheses]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def verify(self, session, image, candidates, query):
        """Rank candidates using tournament."""
        if not candidates:
            return []
        
        if len(candidates) == 1:
            return [0]
        
        if self.mode == "heuristic":
            # Quick heuristic
            scores = [self._heuristic_score(c['mask']) for c in candidates]
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Tournament verification
        masks = [c['mask'] for c in candidates]
        return await self._async_tournament(session, image, masks, query)
    
    async def _async_tournament(self, session, image, masks, query):
        """Run tournament with async comparisons."""
        import random
        
        n = len(masks)
        indices = list(range(n))
        elimination_round = {}
        
        current_round = list(indices)
        random.shuffle(current_round)
        
        round_num = 0
        
        while len(current_round) > 1:
            round_num += 1
            
            pairs = []
            for i in range(0, len(current_round) - 1, 2):
                pairs.append((current_round[i], current_round[i + 1]))
            
            bye_idx = current_round[-1] if len(current_round) % 2 == 1 else None
            
            # Run all comparisons concurrently
            async def compare(idx_a, idx_b):
                try:
                    winner = await self._compare_pair(session, image, masks[idx_a], masks[idx_b], query)
                    return (idx_a, idx_b, idx_a if winner == "LEFT" else idx_b)
                except:
                    return (idx_a, idx_b, idx_a)
            
            tasks = [compare(a, b) for a, b in pairs]
            results = await asyncio.gather(*tasks)
            
            winners = []
            for idx_a, idx_b, winner in results:
                loser = idx_b if winner == idx_a else idx_a
                winners.append(winner)
                elimination_round[loser] = round_num
            
            current_round = winners
            if bye_idx is not None:
                current_round.append(bye_idx)
        
        if current_round:
            elimination_round[current_round[0]] = round_num + 1
        
        # Rank by elimination round
        ranking = sorted(indices, key=lambda i: elimination_round.get(i, 0), reverse=True)
        return ranking
    
    async def _compare_pair(self, session, image, mask_a, mask_b, query):
        """Compare two masks."""
        # Create side-by-side image
        overlay_a = apply_red_alpha_overlay(image, mask_a, alpha=0.5)
        overlay_b = apply_red_alpha_overlay(image, mask_b, alpha=0.5)
        
        w, h = overlay_a.size
        combined = Image.new('RGB', (w * 2 + 20, h + 50), (255, 255, 255))
        combined.paste(overlay_a, (0, 50))
        combined.paste(overlay_b, (w + 20, 50))
        
        image_b64 = self._encode_image(combined)
        
        prompt = f"""Compare two segmentation masks for: "{query}"

LEFT shows one mask (RED region).
RIGHT shows another mask (RED region).

Which better captures the object?
Answer: LEFT or RIGHT"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }]
        
        response = await self._call_llm(
            session, self.verifier_url, self.verifier_model,
            messages, temperature=0.1, max_tokens=100
        )
        
        text = response.upper()
        if "LEFT" in text and "RIGHT" not in text:
            return "LEFT"
        elif "RIGHT" in text:
            return "RIGHT"
        return "LEFT"
    
    def _heuristic_score(self, mask):
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        coverage = np.sum(mask_np) / mask_np.size
        if coverage < 0.01 or coverage > 0.8:
            return 0
        return 50 + (0.3 - abs(coverage - 0.3)) * 100
    
    async def process_sample(self, session, image, query, gt_mask, sample_idx):
        """Process one sample through full pipeline."""
        t0 = time.time()
        
        # Plan
        hypotheses = await self.plan(session, image, query)
        
        # Execute
        candidates = await self.execute(session, image, hypotheses, gt_mask)
        
        # Verify
        ranking = await self.verify(session, image, candidates, query)
        
        return SampleResult(
            sample_idx=sample_idx,
            candidates=candidates,
            ranking=ranking,
            gt_mask=gt_mask,
            total_time=time.time() - t0
        )


async def run_async_evaluation(fraction, max_n, planner_url, verifier_url, executor_url,
                               concurrency, mode):
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
    
    pipeline = AsyncPipeline(planner_url, verifier_url, executor_url, max_n, mode)
    
    N_VALUES = [1, 2, 4, 8, 16, 32, 64]
    results_by_n = {n: {'ious': [], 'oracle_ious': [], 'times': []} for n in N_VALUES if n <= max_n}
    
    print(f"\n{'='*70}")
    print(f"Async Evaluation | Concurrency={concurrency} | Mode={mode}")
    print(f"{'='*70}\n")
    
    # Prepare samples
    samples = []
    for sample_idx, sample in enumerate(ds):
        image = sample.get('image')
        query = sample.get('text') or sample.get('query') or sample.get('sentence')
        gt_mask = sample.get('mask') or sample.get('label')
        
        if image is None or query is None:
            continue
        
        if gt_mask is not None:
            gt_mask = np.array(gt_mask) > 0
        
        samples.append((sample_idx, image, query, gt_mask))
    
    # Process with semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_limit(session, sample_idx, image, query, gt_mask):
        async with semaphore:
            return await pipeline.process_sample(session, image, query, gt_mask, sample_idx)
    
    connector = aiohttp.TCPConnector(limit=concurrency * 10)
    timeout = aiohttp.ClientTimeout(total=600)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [process_with_limit(session, *s) for s in samples]
        
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            result = await coro
            results.append(result)
            
            # Update metrics
            if result.candidates and result.gt_mask is not None:
                for n in results_by_n.keys():
                    if n > len(result.candidates):
                        continue
                    
                    if result.ranking:
                        best_idx = min((i for i in result.ranking if i < n), default=0)
                    else:
                        best_idx = 0
                    
                    pred_mask = result.candidates[best_idx]['mask']
                    iou = calculate_iou(pred_mask, result.gt_mask)
                    oracle_iou = max(c.get('iou', 0) for c in result.candidates[:n])
                    
                    results_by_n[n]['ious'].append(iou)
                    results_by_n[n]['oracle_ious'].append(oracle_iou)
                    results_by_n[n]['times'].append(result.total_time)
    
    print_results(results_by_n, mode)
    
    output_file = f"results_async_{mode}_{int(fraction*100)}pct.json"
    save_results(results_by_n, output_file)


def print_results(results_by_n, mode):
    print(f"\n{'='*70}")
    print(f"RESULTS ({mode.upper()} - Async)")
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
    asyncio.run(run_async_evaluation(
        fraction=args.fraction,
        max_n=args.max_n,
        planner_url=args.planner_url,
        verifier_url=args.verifier_url,
        executor_url=args.executor_url,
        concurrency=args.concurrency,
        mode=args.mode
    ))
