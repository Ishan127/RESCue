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
