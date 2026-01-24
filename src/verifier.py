import re
import json
import json_repair
import numpy as np
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from .utils import apply_red_alpha_overlay
from .api_utils import create_vision_message, get_verifier_client, VERIFIER_MODEL, VERIFIER_API_BASE

VERIFIER_VERBOSE = os.environ.get('VERIFIER_VERBOSE', '0') == '1'


class Verifier:
    """
    Tournament-based verification for handling many masks.
    Uses bracket elimination with side-by-side comparisons.
    """
    def __init__(self, client=None, model_path=None, 
                 api_base=None, verbose=None):
        self.model_path = model_path or VERIFIER_MODEL
        self.client = client if client else get_verifier_client()
        self.is_thinking_model = "thinking" in self.model_path.lower()
        self.verbose = verbose if verbose is not None else VERIFIER_VERBOSE

    def _create_side_by_side(self, image, mask_a, mask_b, label_a="A", label_b="B"):
        """Create side-by-side comparison of two masks."""
        overlay_a = apply_red_alpha_overlay(image, mask_a, alpha=0.5)
        overlay_b = apply_red_alpha_overlay(image, mask_b, alpha=0.5)
        
        w, h = overlay_a.size
        label_height = 50
        
        # Create combined image
        combined = Image.new('RGB', (w * 2 + 20, h + label_height), (255, 255, 255))
        
        # Add labels and images
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        # Left image (A)
        draw.text((w // 2 - 20, 5), label_a, fill=(0, 0, 255), font=font)
        combined.paste(overlay_a, (0, label_height))
        
        # Separator
        draw.rectangle([(w, label_height), (w + 20, h + label_height)], fill=(128, 128, 128))
        
        # Right image (B)
        draw.text((w + 20 + w // 2 - 20, 5), label_b, fill=(0, 0, 255), font=font)
        combined.paste(overlay_b, (w + 20, label_height))
        
        return combined

    def _compare_pair(self, image, mask_a, mask_b, query, idx_a, idx_b):
        """Compare two masks and return the winner index."""
        comparison_img = self._create_side_by_side(image, mask_a, mask_b, "LEFT", "RIGHT")
        
        prompt = f"""Compare these two segmentation masks for the query: "{query}"

LEFT image shows one mask (RED region).
RIGHT image shows another mask (RED region).

Which mask better captures the object described in the query?
Consider:
1. Does the RED region cover the CORRECT object?
2. Are the boundaries TIGHT and ACCURATE?
3. Does it avoid including EXCESS BACKGROUND?

Your final answer MUST end with exactly "ANSWER: LEFT" or "ANSWER: RIGHT"."""

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            comparison_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt, tmp_path)
            
            # For thinking models, allow more tokens for reasoning
            max_tok = 500 if self.is_thinking_model else 150
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.1,  # Low temp for decisive answer
                max_tokens=max_tok
            )
            
            text = completion.choices[0].message.content.strip().upper()
            
            if self.verbose:
                short_text = text[-200:] if len(text) > 200 else text
                print(f"[Compare {idx_a} vs {idx_b}]: ...{short_text}")
            
            # Parse response - look for explicit ANSWER: pattern first
            if "ANSWER: LEFT" in text or "ANSWER:LEFT" in text:
                return idx_a
            elif "ANSWER: RIGHT" in text or "ANSWER:RIGHT" in text:
                return idx_b
            
            # Fallback: look at last line or last words for the answer
            lines = text.strip().split('\n')
            last_line = lines[-1].strip()
            
            if last_line == "LEFT" or last_line.endswith("LEFT"):
                return idx_a
            elif last_line == "RIGHT" or last_line.endswith("RIGHT"):
                return idx_b
            
            # Final fallback: count occurrences in last 100 chars
            tail = text[-100:]
            left_count = tail.count("LEFT")
            right_count = tail.count("RIGHT")
            if left_count > right_count:
                return idx_a
            elif right_count > left_count:
                return idx_b
            
            # Truly ambiguous - default to first
            return idx_a
                    
        except Exception as e:
            if self.verbose:
                print(f"Comparison error: {e}")
            return idx_a  # Default to first on error
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _tournament_select(self, image, masks, query, indices):
        """Run tournament bracket to find best mask."""
        if len(indices) == 0:
            return None
        if len(indices) == 1:
            return indices[0]
        if len(indices) == 2:
            return self._compare_pair(image, masks[indices[0]], masks[indices[1]], 
                                      query, indices[0], indices[1])
        
        # Run bracket: compare pairs, winners advance
        import random
        current_round = list(indices)
        random.shuffle(current_round)  # Randomize bracket seeding
        
        round_num = 1
        while len(current_round) > 1:
            if self.verbose:
                print(f"[Tournament Round {round_num}]: {len(current_round)} candidates")
            
            next_round = []
            
            # Process pairs
            for i in range(0, len(current_round) - 1, 2):
                idx_a, idx_b = current_round[i], current_round[i + 1]
                winner = self._compare_pair(image, masks[idx_a], masks[idx_b], 
                                           query, idx_a, idx_b)
                next_round.append(winner)
            
            # Odd one out gets a bye
            if len(current_round) % 2 == 1:
                next_round.append(current_round[-1])
            
            current_round = next_round
            round_num += 1
        
        return current_round[0]

    def verify_batch_comparative(self, image_input, masks, query):
        """
        Full ranking using parallel tournament.
        Sends many comparisons to vLLM at once for batched GPU inference.
        """
        if len(masks) == 0:
            return []
        if len(masks) == 1:
            return [{"mask_idx": 0, "rank": 1, "score": 100, "reasoning": "Only candidate"}]
        
        n = len(masks)
        
        if n <= 6:
            return self._grid_compare(image_input, masks, query)
        else:
            return self._parallel_full_ranking(image_input, masks, query)

    def _parallel_full_ranking(self, image, masks, query):
        """
        Get full ranking using parallel tournament with elimination tracking.
        
        Strategy:
        1. Run single-elimination tournament (parallelized rounds)
        2. Track which round each candidate was eliminated
        3. Rank by elimination round (later = better)
        4. Within same round, use heuristic tie-breaker
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random
        
        n = len(masks)
        indices = list(range(n))
        
        # Track elimination: {idx: round_eliminated} (lower = eliminated earlier = worse)
        # Winner gets round = infinity
        elimination_round = {}
        
        # Compute heuristics for tie-breaking
        heuristics = {i: self._compute_mask_heuristic(masks[i]) for i in indices}
        
        # Shuffle for fair seeding
        current_round = list(indices)
        random.shuffle(current_round)
        
        round_num = 0
        
        while len(current_round) > 1:
            round_num += 1
            if self.verbose:
                print(f"[Tournament Round {round_num}]: {len(current_round)} candidates")
            
            # Create all pairs for this round
            pairs = []
            for i in range(0, len(current_round) - 1, 2):
                pairs.append((current_round[i], current_round[i + 1]))
            
            # Bye for odd one
            bye_idx = current_round[-1] if len(current_round) % 2 == 1 else None
            
            # Run ALL comparisons in parallel - vLLM will batch them!
            winners = []
            losers = []
            
            with ThreadPoolExecutor(max_workers=32) as executor:
                future_to_pair = {}
                for idx_a, idx_b in pairs:
                    future = executor.submit(
                        self._compare_pair, image, masks[idx_a], masks[idx_b],
                        query, idx_a, idx_b
                    )
                    future_to_pair[future] = (idx_a, idx_b)
                
                for future in as_completed(future_to_pair):
                    idx_a, idx_b = future_to_pair[future]
                    winner = future.result()
                    loser = idx_b if winner == idx_a else idx_a
                    winners.append(winner)
                    losers.append(loser)
            
            # Mark losers with their elimination round
            for loser in losers:
                elimination_round[loser] = round_num
            
            # Next round = winners + bye
            current_round = winners
            if bye_idx is not None:
                current_round.append(bye_idx)
        
        # Winner survives all rounds
        if current_round:
            winner = current_round[0]
            elimination_round[winner] = round_num + 1  # Survived longest
        
        # Build ranking: sort by elimination round (desc), then by heuristic (desc)
        ranking = sorted(
            indices,
            key=lambda i: (elimination_round.get(i, 0), heuristics.get(i, 0)),
            reverse=True
        )
        
        # Build results
        results = []
        for rank, idx in enumerate(ranking):
            score = max(10, 100 - rank * (90 // max(1, n - 1)))
            results.append({
                "mask_idx": idx,
                "rank": rank + 1,
                "score": score,
                "reasoning": f"Eliminated round {elimination_round.get(idx, 0)}"
            })
        
        return results

    def _compute_mask_heuristic(self, mask):
        """Quick heuristic score for tie-breaking."""
        import numpy as np
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        total_pixels = mask_np.size
        mask_pixels = np.sum(mask_np)
        
        if mask_pixels == 0:
            return 0
        
        # Coverage ratio (penalize too small or too large)
        coverage = mask_pixels / total_pixels
        if coverage < 0.01:
            coverage_score = coverage * 50
        elif coverage > 0.8:
            coverage_score = 50 - (coverage - 0.8) * 100
        else:
            coverage_score = 50 + (0.3 - abs(coverage - 0.3)) * 100
        
        return coverage_score

    def _parallel_tournament(self, image, masks, query, indices):
        """Run tournament with parallel comparisons."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random
        
        if len(indices) == 0:
            return None
        if len(indices) == 1:
            return indices[0]
        
        current_round = list(indices)
        random.shuffle(current_round)
        
        round_num = 1
        while len(current_round) > 1:
            if self.verbose:
                print(f"[Tournament Round {round_num}]: {len(current_round)} candidates")
            
            # Create pairs
            pairs = []
            for i in range(0, len(current_round) - 1, 2):
                pairs.append((current_round[i], current_round[i + 1]))
            
            # Run comparisons in parallel (4 at a time to not overload)
            next_round = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for idx_a, idx_b in pairs:
                    future = executor.submit(
                        self._compare_pair, image, masks[idx_a], masks[idx_b],
                        query, idx_a, idx_b
                    )
                    futures[future] = (idx_a, idx_b)
                
                for future in as_completed(futures):
                    winner = future.result()
                    next_round.append(winner)
            
            # Odd one out gets a bye
            if len(current_round) % 2 == 1:
                next_round.append(current_round[-1])
            
            current_round = next_round
            round_num += 1
        
        return current_round[0]

    def _tournament_rank(self, image, masks, query):
        """Get full ranking using tournament elimination."""
        n = len(masks)
        indices = list(range(n))
        
        # Single tournament to find winner, then remove and repeat
        # This gives us a full ranking
        ranking = []
        remaining = list(indices)
        
        while remaining:
            if len(remaining) == 1:
                ranking.append(remaining[0])
                break
            
            winner = self._tournament_select(image, masks, query, remaining)
            if winner is not None:
                ranking.append(winner)
                remaining.remove(winner)
            else:
                # Shouldn't happen, but add remaining in order
                ranking.extend(remaining)
                break
        
        # Build results
        results = []
        for rank, idx in enumerate(ranking):
            score = max(10, 100 - rank * (90 // max(1, n - 1)))
            results.append({
                "mask_idx": idx,
                "rank": rank + 1,
                "score": score,
                "reasoning": ""
            })
        
        return results

    def _grid_compare(self, image, masks, query):
        """Original grid comparison for small N."""
        labels = [chr(65 + i) for i in range(len(masks))]
        grid_img = self._create_comparison_grid(image, masks, labels)
        
        label_list = ", ".join(labels)
        
        prompt = f"""This image shows {len(masks)} segmentation masks (labeled {label_list}) for: "{query}"

Each panel shows the SAME image with a different RED highlighted region.

Look carefully at each mask:
1. Which RED region covers the CORRECT object for the query?
2. Which has the TIGHTEST boundaries?
3. Which avoids BACKGROUND noise?

Rank ALL masks from BEST to WORST.

Output ONLY: {{"ranking": ["X", "Y", ...], "best_reason": "why X is best"}}"""

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            grid_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        results = []
        
        try:
            messages = create_vision_message(prompt, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            text = completion.choices[0].message.content.strip()
            if self.verbose:
                print(f"[Grid Compare]: {text[:300]}...")
            
            ranking = self._parse_ranking(text, labels)
            
            for rank_idx, label in enumerate(ranking):
                mask_idx = ord(label) - 65
                if mask_idx < len(masks):
                    score = max(10, 100 - rank_idx * 15)
                    results.append({
                        "mask_idx": mask_idx,
                        "rank": rank_idx + 1,
                        "score": score,
                        "label": label,
                        "reasoning": ""
                    })
            
        except Exception as e:
            if self.verbose:
                print(f"Grid compare error: {e}")
            for i in range(len(masks)):
                results.append({
                    "mask_idx": i,
                    "rank": i + 1,
                    "score": 100 - i * 10,
                    "label": labels[i] if i < len(labels) else str(i),
                    "reasoning": f"Error: {str(e)}"
                })
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return results

    def _create_comparison_grid(self, image, masks, labels):
        """Create grid of mask overlays."""
        overlays = []
        for mask in masks:
            overlay = apply_red_alpha_overlay(image, mask, alpha=0.5)
            overlays.append(overlay)
        
        w, h = overlays[0].size
        n = len(overlays)
        
        if n <= 2:
            cols, rows = n, 1
        elif n <= 4:
            cols, rows = 2, 2
        else:
            cols, rows = 3, 2
        
        label_height = 40
        cell_h = h + label_height
        grid = Image.new('RGB', (cols * w, rows * cell_h), (255, 255, 255))
        
        for idx, (overlay, label) in enumerate(zip(overlays, labels)):
            if idx >= cols * rows:
                break
            row, col = idx // cols, idx % cols
            x, y = col * w, row * cell_h
            
            labeled = Image.new('RGB', (w, cell_h), (255, 255, 255))
            labeled.paste(overlay, (0, label_height))
            draw = ImageDraw.Draw(labeled)
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except:
                font = ImageFont.load_default()
            draw.text((w // 2 - 15, 5), label, fill=(0, 0, 0), font=font)
            
            grid.paste(labeled, (x, y))
        
        return grid

    def _parse_ranking(self, text, valid_labels):
        """Extract ranking from VLM response."""
        ranking = []
        
        # Try JSON parse
        json_match = re.search(r'"ranking"\s*:\s*\[([^\]]+)\]', text)
        if json_match:
            letters = re.findall(r'[A-Z]', json_match.group(1))
            ranking = [l for l in letters if l in valid_labels]
        
        # Fallback: find letters in order
        if not ranking:
            letters = re.findall(r'\b([A-Z])\b', text)
            seen = []
            for l in letters:
                if l in valid_labels and l not in seen:
                    seen.append(l)
            ranking = seen
        
        # Ensure all labels present
        for label in valid_labels:
            if label not in ranking:
                ranking.append(label)
        
        return ranking

    def verify(self, image_input, mask, query):
        """Single mask verification."""
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.5)
        
        prompt = f"""This image shows a RED highlighted region for the query: "{query}"

Rate the segmentation quality from 0-100:
- 90-100: Perfect - correct object, tight boundaries
- 70-89: Good - correct object, minor issues
- 50-69: Okay - mostly correct but problems
- 30-49: Poor - significant issues
- 0-29: Wrong - incorrect object

Output JSON: {{"score": X, "reason": "brief explanation"}}"""

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,
                max_tokens=200
            )
            
            text = completion.choices[0].message.content.strip()
            
            # Parse score
            score_match = re.search(r'"score"\s*:\s*(\d+)', text)
            if score_match:
                score = int(score_match.group(1))
            else:
                score_match = re.search(r'\b(\d{1,3})\b', text)
                score = int(score_match.group(1)) if score_match else 50
            
            score = max(0, min(100, score))
            return {"score": score, "total": score}
            
        except Exception as e:
            if self.verbose:
                print(f"Verify error: {e}")
            return {"score": 50, "total": 50}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    def verify_batch_hybrid(self, image_input, masks, query, top_k=8):
        """
        Hybrid "Filter & Fight" Strategy:
        1. Fast Filter: Pointwise score all masks using detailed composite metrics.
        2. Elite Tournament: Run pairwise tournament on Top-K pointwise winners.
        3. Merge: Rank 1-K from tournament, K+1-N from pointwise scores.
        """
        n = len(masks)
        if n == 0: return []
        if n == 1: return [{"mask_idx": 0, "rank": 1, "score": 100, "reasoning": "Only candidate"}]
        
        # --- Step 1: Fast Pointwise Scoring (Parallel) ---
        if self.verbose:
             print(f"[Hybrid] Scoring {n} masks pointwise...")
        
        pointwise_results = self._pointwise_score_batch(image_input, masks, query)
        
        # Sort by (Score DESC, GeometricHeuristic DESC, Index ASC)
        # Compute geometric heuristic for tie-breaking
        heuristics = {r['mask_idx']: self._compute_mask_heuristic(masks[r['mask_idx']]) for r in pointwise_results}
        
        sorted_candidates = sorted(
            pointwise_results,
            key=lambda r: (r['total_score'], heuristics[r['mask_idx']], -r['mask_idx']),
            reverse=True
        )
        
        # --- Step 2: Elite Tournament ---
        # Take Top-K (or all if N <= K)
        k = min(n, top_k)
        elite_candidates = sorted_candidates[:k]
        elite_indices = [c['mask_idx'] for c in elite_candidates]
        
        if self.verbose:
            print(f"[Hybrid] Running tournament on top {k} candidates: {elite_indices}")
            
        # Run tournament on just these indices
        # We reuse _tournament_rank but need to map indices specifically
        tournament_results = self._parallel_full_ranking_subset(image_input, masks, query, elite_indices)
        
        # --- Step 3: Merge Results ---
        final_ranking = []
        
        # 1. Add Tournament Winners (Ranks 1..K)
        # tournament_results returns objects with {mask_idx, rank, score}
        # Re-assign scores to be > max_pointwise to ensure consistency
        max_pt_score = sorted_candidates[k]['total_score'] if k < n else 0
        
        for res in tournament_results:
            # Map tournament rank 1..K to score 100..90 approx, but keep above max_pt_score
            # Or just accept the tournament score if it's high enough.
            # Let's enforce strictly that tournament winners > losers.
            res['rank'] = res['rank'] # pure rank 1..K
            
            # Augment with the detailed reasoning from pointwise if available
            pt_res = next((p for p in pointwise_results if p['mask_idx'] == res['mask_idx']), {})
            res['pointwise_details'] = pt_res
            final_ranking.append(res)
            
        # 2. Add The Rest (Ranks K+1..N)
        for i, pt_res in enumerate(sorted_candidates[k:]):
            rank = k + 1 + i
            final_ranking.append({
                "mask_idx": pt_res['mask_idx'],
                "rank": rank,
                "score": pt_res['total_score'],
                "reasoning": "Pointwise ranking (did not qualify for tournament)",
                "pointwise_details": pt_res
            })
            
        return final_ranking

    def _parallel_full_ranking_subset(self, image, masks, query, subset_indices):
        """Run tournament only on a subset of indices."""
        # This is a modified version of _parallel_full_ranking that accepts specific indices
        # Logic is identical to _parallel_full_ranking but initializes with subset_indices
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random
        
        indices = list(subset_indices)
        n = len(indices)
        
        elimination_round = {}
        heuristics = {i: self._compute_mask_heuristic(masks[i]) for i in indices}
        
        current_round = list(indices)
        random.shuffle(current_round)
        
        round_num = 0
        while len(current_round) > 1:
            round_num += 1
            pairs = []
            for i in range(0, len(current_round) - 1, 2):
                pairs.append((current_round[i], current_round[i + 1]))
            
            bye_idx = current_round[-1] if len(current_round) % 2 == 1 else None
            
            winners = []
            losers = []
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_pair = {}
                for idx_a, idx_b in pairs:
                    future = executor.submit(
                        self._compare_pair, image, masks[idx_a], masks[idx_b],
                        query, idx_a, idx_b
                    )
                    future_to_pair[future] = (idx_a, idx_b)
                
                for future in as_completed(future_to_pair):
                    idx_a, idx_b = future_to_pair[future]
                    winner = future.result()
                    loser = idx_b if winner == idx_a else idx_a
                    winners.append(winner)
                    losers.append(loser)
            
            for loser in losers:
                elimination_round[loser] = round_num
            
            current_round = winners
            if bye_idx is not None:
                current_round.append(bye_idx)
                
        if current_round:
            winner = current_round[0]
            elimination_round[winner] = round_num + 1

        ranking = sorted(
            indices,
            key=lambda i: (elimination_round.get(i, 0), heuristics.get(i, 0)),
            reverse=True
        )
        
        results = []
        for rank, idx in enumerate(ranking):
            score = max(90, 100 - rank * 2) # Elite scores are high
            results.append({
                "mask_idx": idx,
                "rank": rank + 1,
                "score": score,
                "reasoning": f"Tournament survivor (Round {elimination_round.get(idx, 0)})"
            })
        return results

    def _pointwise_score_batch(self, image, masks, query):
        """
        Run parallel pointwise scoring for all masks.
        Returns list of dicts with scores and breakdown.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=32) as executor: # Batch efficiently
            future_to_idx = {
                executor.submit(self._score_single_composite, image, mask, query): i 
                for i, mask in enumerate(masks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    res['mask_idx'] = idx
                    results.append(res)
                except Exception as e:
                    if self.verbose: print(f"Pointwise error on {idx}: {e}")
                    results.append({'mask_idx': idx, 'total_score': 0, 'error': str(e)})
                    
        return results

    def _score_single_composite(self, image, mask, query):
        """
        Score a single mask using the 5-part composite metric (0-100).
        1. Geometric (10%)
        2. Rating (20%)
        3. IoU (30%)
        4. Boundary (20%)
        5. Semantic (20% - 4 sub-metrics)
        """
        # 1. Geometric (10%) - purely local calculation
        geo_score_raw = self._compute_mask_heuristic(mask) # 0-100
        geo_score = geo_score_raw * 0.10
        
        # Prepare VLM Input
        overlay = apply_red_alpha_overlay(image, mask, alpha=0.5)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay.save(tmp.name, quality=95)
            tmp_path = tmp.name
            
        try:
            prompt = f"""Evaluate this segmentation mask (red region) for the query: "{query}"

Score 0-5 for these specific metrics:

1. RATING.CLASS: One of [PERFECT, GOOD, AVERAGE, BAD, WRONG]
2. PREDICTED.IOU: Estimated IoU 0-100%
3. BOUNDARY: Edge quality 0-100%
4. SEMANTIC.CATEGORY: Correct object class? (0-5)
5. SEMANTIC.ATTRIBUTE: Color/shape match? (0-5)
6. SEMANTIC.CONTEXT: Action/context match? (0-5)
7. SEMANTIC.COUNT: Correct instance count? (0-5)

Output ONLY valid JSON like this:
{{
  "rating_class": "GOOD",
  "predicted_iou": 75,
  "boundary_score": 80,
  "semantic_scores": {{
    "category": 5,
    "attribute": 4,
    "context": 5,
    "count": 5
  }}
}}"""
            messages = create_vision_message(prompt, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.1,
                max_tokens=300
            )
            text = completion.choices[0].message.content.strip()
            
            # --- Parse JSON ---
            # Robust parsing strategy:
            # 1. Try json_repair (best for LLM output)
            # 2. Key-based regex fallback if json_repair fails/missing
            data = {}
            try:
                try:
                    import json_repair
                    data = json_repair.loads(text)
                except ImportError:
                    if self.verbose: print("json_repair not found, using regex fallback")
                    # Fallback to regex extraction + json
                    json_match = re.search(r'\{[\s\S]*\}', text)
                    if json_match:
                        data = json.loads(json_match.group(0))
            except Exception as e:
                if self.verbose: print(f"JSON Parsing fully failed: {e}")
                
            # Final fallback: manual key extraction
            if not data:
                 if "PERFECT" in text: data["rating_class"] = "PERFECT"
                 elif "GOOD" in text: data["rating_class"] = "GOOD"
                 # Extract numbers
                 iou_m = re.search(r'IOU.*?(\d+)', text, re.IGNORECASE)
                 if iou_m: data["predicted_iou"] = int(iou_m.group(1))

            # --- Calculate Scores ---
            
            # 2. Rating (20%)
            rating_map = {"PERFECT": 100, "GOOD": 75, "AVERAGE": 50, "BAD": 25, "WRONG": 0}
            r_class = data.get("rating_class", "BAD").upper()
            # fuzzy match
            if "PERFECT" in r_class: r_val = 100
            elif "GOOD" in r_class: r_val = 75
            elif "AVERAGE" in r_class: r_val = 50
            elif "WRONG" in r_class: r_val = 0
            elif "BAD" in r_class: r_val = 25
            else: r_val = 25
            score_rating = r_val * 0.20
            
            # 3. IoU (30%)
            pred_iou = float(data.get("predicted_iou", 0))
            score_iou = min(100, max(0, pred_iou)) * 0.30
            
            # 4. Boundary (20%)
            b_qual = float(data.get("boundary_score", 0))
            score_boundary = min(100, max(0, b_qual)) * 0.20
            
            # 5. Semantic (20%) - sum of 4 * 5pts = 20pts max
            sem = data.get("semantic_scores", {})
            s1 = sem.get("category", 0)
            s2 = sem.get("attribute", 0)
            s3 = sem.get("context", 0)
            s4 = sem.get("count", 0)
            # Ensure they are 0-5
            s_sum = min(5, s1) + min(5, s2) + min(5, s3) + min(5, s4) # max 20
            # score is direct since total is 20
            score_semantic = s_sum 
            
            # Total
            total_score = geo_score + score_rating + score_iou + score_boundary + score_semantic
            
            return {
                "total_score": round(total_score, 2),
                "breakdown": {
                    "geo": round(geo_score, 2),
                    "rating": round(score_rating, 2),
                    "iou": round(score_iou, 2),
                    "boundary": round(score_boundary, 2),
                    "semantic": s_sum
                },
                "raw_response": data
            }
            
        except Exception as e:
            if self.verbose: print(f"Composite score error: {e}")
            return {"total_score": 0, "error": str(e)}
        finally:
             if os.path.exists(tmp_path):
                 os.remove(tmp_path)
