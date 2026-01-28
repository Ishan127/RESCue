import re
import json
import json_repair
import numpy as np
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from .utils import apply_red_alpha_overlay, calculate_iou
from .api_utils import create_vision_message, get_verifier_client, VERIFIER_MODEL, VERIFIER_API_BASE

VERIFIER_VERBOSE = os.environ.get('VERIFIER_VERBOSE', '0') == '1'


class Verifier:
    """
    Pointwise verification for ranking segmentation masks.
    Each mask is scored independently on 5 metrics for deterministic ranking.
    """
    def __init__(self, client=None, model_path=None, 
                 api_base=None, verbose=None):
        self.model_path = model_path or VERIFIER_MODEL
        if client:
            self.client = client
        else:
            from .api_utils import get_openai_client
            # Use api_base if provided, else use default VERIFIER_API_BASE
            url = api_base if api_base else None
            if url:
                 self.client = get_openai_client(base_url=url)
            else:
                 self.client = get_verifier_client()
                 
        self.is_thinking_model = "thinking" in self.model_path.lower()
        self.verbose = verbose if verbose is not None else VERIFIER_VERBOSE

    def _compute_mask_heuristic(self, mask):
        """Quick heuristic score for tie-breaking based on mask coverage."""
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

    def verify(self, image_input, mask, query):
        """Single mask verification - returns a simple score."""
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.5)
        
        prompt = f"""This image shows a RED highlighted region for the query: "{query}"

Rate the segmentation quality from 0-100:
- 90-100: Perfect - correct object, tight boundaries
- 70-89: Good - correct object, minor issues
- 50-69: Okay - mostly correct but problems
- 30-49: Poor - significant issues
- 0-29: Wrong - incorrect object

Output JSON: {{"score": X, "reason": "brief explanation"}}"""

        # DIRECT IN-MEMORY PASSING
        try:
            messages = create_vision_message(prompt, image=overlay_img)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.1,
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
            
            return {"score": min(100, max(0, score)), "total": score}
            
        except Exception as e:
            if self.verbose:
                print(f"Verify error: {e}")
            return {"score": 50, "total": 50}

    def verify_batch_pointwise(self, image_input, masks, query, skip_clip=False, skip_consistency=False, max_workers=None):
        """
        Pointwise-only ranking: Score each mask independently, rank by total score.
        
        This is simpler and more deterministic than tournament-based ranking:
        - No tournament randomness (bracket seeding)
        - No left/right comparison bias
        - Each mask scored on its own merits
        - Full breakdown available for debugging
        
        Returns list of dicts with mask_idx, rank, score, and pointwise_details.
        """
        n = len(masks)
        if n == 0:
            return []
        if n == 1:
            # Still score the single mask for consistency
            from .api_utils import encode_pil_image, encode_image
            if isinstance(image_input, str):
                base64_img = encode_image(image_input)
                from PIL import Image as PILImage
                with PILImage.open(image_input) as img:
                    w, h = img.size
            else:
                base64_img = encode_pil_image(image_input)
                w, h = image_input.size
            single_result = self._score_single_box(base64_img, w, h, masks[0], query)
            return [{
                "mask_idx": 0,
                "rank": 1,
                "score": single_result.get('total_score', 50),
                "reasoning": "Only candidate",
                "pointwise_details": single_result
            }]
        
        if self.verbose:
            print(f"[Pointwise] Scoring {n} masks independently...")
        
        # 1. Run VLM Scoring (Parallel)
        # Increase workers to match N for high throughput async servers
        if max_workers is None:
             max_workers = min(128, n) 
        
        # Pre-encode image for Prefix Caching efficiency
        from .api_utils import encode_pil_image, encode_image
        base64_img = None
        w, h = 0, 0
        if isinstance(image_input, str):
            base64_img = encode_image(image_input)
            with Image.open(image_input) as img:
                w, h = img.size
        else:
            base64_img = encode_pil_image(image_input)
            w, h = image_input.size
            
        vlm_results = self._pointwise_score_batch(base64_img, w, h, masks, query, max_workers=max_workers)
        
        # 2. Run CLIP Scoring in PARALLEL with VLM (they use different GPUs)
        clip_scores = [0.0] * n
        if not skip_clip:
            if not hasattr(self, 'clip_verifier'):
                from .clip_verifier import ClipVerifier
                self.clip_verifier = ClipVerifier()
            
            try:
                clip_scores = self.clip_verifier.verify_batch(image_input, masks, query)
            except Exception as e:
                if self.verbose: print(f"CLIP Error: {e}")
                clip_scores = [0.0] * n

        # 3. Compute Consistency Scores (15%)
        # OPTIMIZATION: Skip for small N (overhead not worth it)
        consistency_scores = []
        if not skip_consistency and n >= 8:
            try:
                # Vectorized IoU Calculation (O(N^2) -> Matrix Op)
                # Stack masks: (N, H, W) -> flattened (N, HW)
                mask_stack = []
                for m in masks:
                    m_arr = np.array(m).astype(bool)
                    if m_arr.ndim == 3: m_arr = m_arr[:,:,0] if m_arr.shape[2]==1 else m_arr[0]
                    mask_stack.append(m_arr.reshape(-1))
                
                # Use float32 to prevent potential overflow in very large images/batches during dot product, though unlikely for pixels
                # But bool matmul is not standard, convert to float or int
                flat_masks = np.stack(mask_stack).astype(np.float32) 
                
                # Intersection: Dot product
                intersections = flat_masks @ flat_masks.T # (N, N)
                
                # Areas
                areas = flat_masks.sum(axis=1) # (N,)
                
                # Union = A + B - Intersection
                # Broadcast areas: (N, 1) + (1, N)
                unions = areas[:, None] + areas[None, :] - intersections
                
                # Avoid division by zero
                unions[unions == 0] = 1.0
                
                iou_matrix = intersections / unions
                
                # Mean IoU for each mask with others (exclude self-diagonal)
                # Subtract 1.0 (self-iou) and divide by N-1
                consistency_scores = (iou_matrix.sum(axis=1) - 1.0) / (n - 1)
                consistency_scores = consistency_scores.tolist()
                
            except Exception as e:
                print(f"[Verifier] Vectorized IoU failed: {e}. Falling back to loop.")
                # Fallback old loop
                for i in range(n):
                    total_iou = 0
                    count = 0
                    for j in range(n):
                        if i == j: continue
                        total_iou += calculate_iou(masks[i], masks[j])
                        count += 1
                    avg_iou = total_iou / count if count > 0 else 0
                    consistency_scores.append(avg_iou)
        else:
             consistency_scores = [1.0] * n

        # 4. Combine Scores (50% VLM + 35% CLIP + 15% Consistency)
        final_results = []
        heuristics = {}
        
        # MinMax Scale CLIP scores to 0-35 range per batch (Was 40, now 35)
        if clip_scores:
            c_min = min(clip_scores)
            c_max = max(clip_scores)
            if c_max > c_min:
                scaled_clip = [((s - c_min) / (c_max - c_min)) * 35.0 for s in clip_scores]
            else:
                scaled_clip = [17.5] * n 
        else:
            scaled_clip = [0.0] * n

        for i, res in enumerate(vlm_results):
            vlm_score = res.get('total_score', 0)
            
            # VLM: 50% weight (Scale 0-100 -> 0-50)
            vlm_val = vlm_score * 0.5
            
            # CLIP: 35% weight (Already scaled to 0-35)
            clip_val = scaled_clip[i]
            
            # Consistency: 15% weight (Scale 0-1 -> 0-15)
            cons_val = consistency_scores[i] * 15.0
            
            hybrid_score = vlm_val + clip_val + cons_val
            
            res['vlm_score'] = vlm_score
            res['vlm_contrib'] = round(vlm_val, 2)
            res['clip_score'] = round(clip_scores[i], 2)
            res['clip_contrib'] = round(clip_val, 2)
            res['cons_score'] = round(consistency_scores[i], 2)
            res['cons_contrib'] = round(cons_val, 2)
            res['total_score'] = round(hybrid_score, 2)
            
            final_results.append(res)
            heuristics[res['mask_idx']] = self._compute_mask_heuristic(masks[res['mask_idx']])
        
        # Sort by: (hybrid_score DESC, heuristic DESC, index ASC)
        sorted_results = sorted(
            final_results,
            key=lambda r: (
                r.get('total_score', 0),
                heuristics.get(r['mask_idx'], 0),
                -r['mask_idx']
            ),
            reverse=True
        )
        
        # Build final ranking
        final_ranking = []
        for rank, res in enumerate(sorted_results):
            final_ranking.append({
                "mask_idx": res['mask_idx'],
                "rank": rank + 1,
                "score": res.get('total_score', 0),
                "reasoning": f"Hybrid: {res['total_score']} (VLM:{res['vlm_score']} CLIP:{res['clip_score']} Cons:{res['cons_score']})",
                "pointwise_details": res
            })
            
        # --- Pyramid Tournament Refinement ---
        # "N=1 we have direct pick. N=2 we ask LLM... N=4 we compare winner with top scorer of remaining 2..."
        # Strategy: Compare Current Winner vs Ranked Candidate at indices 1, 2, 4, 8, 16, 32...
        if n > 1:
            try:
                final_ranking = self._pyramid_tournament(image_input, masks, query, final_ranking)
            except Exception as e:
                print(f"Tournament failed: {e}")

        if self.verbose:
            top3 = [(r['mask_idx'], r['score']) for r in final_ranking[:3]]
            print(f"[Tournament] Final Top 3: {top3}")
        
        return final_ranking

    def _pyramid_tournament(self, image, masks, query, ranking):
        """
        Iteratively challenge the top candidate against the leader of the next block.
        Indices to challenge: 1, 2, 4, 8, 16...
        """
        current_winner_idx = 0  # Index in the 'ranking' list
        winner_res = ranking[0]
        
        challenges = []
        step = 1
        while step < len(ranking):
            challenges.append(step)
            step *= 2
            
        if self.verbose:
            print(f"[Tournament] Challenges at indices: {challenges}")

        for challenger_rank_idx in challenges:
            if challenger_rank_idx >= len(ranking):
                break
                
            challenger_res = ranking[challenger_rank_idx]
            
            # Perform Duel: Winner vs Challenger
            # Mask indices are in res['mask_idx']
            wa_idx = winner_res['mask_idx']
            wb_idx = challenger_res['mask_idx']
            
            if self.verbose:
                print(f"[Duel] {wa_idx} (Score {winner_res['score']}) vs {wb_idx} (Score {challenger_res['score']})")

            winner_is_a = self._compare_pair(image, masks[wa_idx], masks[wb_idx], winner_res, challenger_res, query)
            
            if winner_is_a:
                # Winner stays, challenger loses
                pass
            else:
                # Challenger becomes new winner
                if self.verbose:
                    print(f"-> Challenger {wb_idx} WON!")
                winner_res = challenger_res
                current_winner_idx = challenger_rank_idx
        
        # Move the final winner to the top of the list
        if current_winner_idx != 0:
            ranking.pop(current_winner_idx)
            ranking.insert(0, winner_res)
            # Re-assign ranks
            for i, r in enumerate(ranking):
                r['rank'] = i + 1
                
        return ranking

    def _compare_pair(self, image, mask_a, mask_b, res_a, res_b, query):
        """
        Pairwise VLM comparison. Returns True if A is better, False if B is better.
        Includes pointwise metrics in the prompt context.
        """
        try:
            # Prepare composite images
            img_a = apply_red_alpha_overlay(image, mask_a, alpha=0.5, black_background=True)
            img_b = apply_red_alpha_overlay(image, mask_b, alpha=0.5, black_background=True)
            
            import io
            import base64
            
            def to_b64(img):
                buf = io.BytesIO()
                img.convert('RGB').save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            b64_a = to_b64(img_a)
            b64_b = to_b64(img_b)
            
            # Format Metrics for Context
            def format_metrics(res):
                details = res.get('pointwise_details', {}).get('breakdown', {})
                return (
                    f"Total: {res.get('total_score', 0)} "
                    f"(VLM: {res.get('vlm_score', 0)}, CLIP: {res.get('clip_score', 0)}, Cons: {res.get('cons_score', 0)})\n"
                    f"   VLM Detail: IoU={details.get('iou', 0)}, Boundary={details.get('boundary', 0)}, Semantic={details.get('semantic', 0)}"
                )

            metrics_a = format_metrics(res_a)
            metrics_b = format_metrics(res_b)
            
            prompt = f"""Compare these two segmentation masks for the query: "{query}"

Image 1 (Candidate A) Metrics:
{metrics_a}

Image 2 (Candidate B) Metrics:
{metrics_b}

Which mask is better?
Think step-by-step:
1. Analyze Candidate A: Does it cover the object fully? Is it too tight/loose?
2. Analyze Candidate B: Compare coverage and boundary precision.
3. Compare Metrics: Does the visual evidence match the high/low scores?
4. Decision: Which one is closer to the ground truth intent?

Output JSON: {{ "winner": "A" or "B", "reason": "short explanation" }}"""

            # Construct message with 2 images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}
                        }
                    ]
                }
            ]
            
            # Schema
            schema = {
                "type": "object",
                "properties": {
                    "winner": {"type": "string", "enum": ["A", "B"]},
                    "reason": {"type": "string"}
                },
                "required": ["winner"]
            }
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.0,
                max_tokens=1024, # Increased for Thinking
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True}, # ENABLE SYSTEM 2 THINKING
                    "guided_json": schema,
                    "guided_decoding_backend": "outlines"
                }
            )
            
            text = completion.choices[0].message.content.strip()
            if '"winner": "B"' in text or "'winner': 'B'" in text:
                return False
            return True # Default to A (current winner) if unclear
            
        except Exception as e:
            if self.verbose:
                print(f"Duel Error: {e}")
            return True # Conservative: keep current winner

    def _pointwise_score_batch(self, base64_img, w, h, masks, query, max_workers=32):
        """
        Run parallel pointwise scoring for all masks using Cached Box Prompting.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._score_single_box, base64_img, w, h, mask, query): i 
                for i, mask in enumerate(masks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    res['mask_idx'] = idx
                    results.append(res)
                except Exception as e:
                    if self.verbose:
                        print(f"Pointwise error on {idx}: {e}")
                    results.append({'mask_idx': idx, 'total_score': 0, 'error': str(e)})
                    
        return results



    def _score_single_box(self, base64_img, w, h, mask, query):
        """
        Score using Bounding Box Prompting on the original image.
        Target Metric: Evaluate if the object inside the box matches the query description + segmentation quality inference.
        """
        # 1. Geometric (10%)
        geo_score_raw = self._compute_mask_heuristic(mask)
        geo_score = geo_score_raw * 0.10
        
        # Calculate Box
        mask_np = np.array(mask) > 0
        if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
        
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return {"total_score": 0, "error": "Empty mask"}
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Normalize to 1000 for Qwen3-VL
        x1_n = int(xmin / w * 1000)
        y1_n = int(ymin / h * 1000)
        x2_n = int(xmax / w * 1000)
        y2_n = int(ymax / h * 1000)
        
        box_str = f"[{x1_n}, {y1_n}, {x2_n}, {y2_n}]"
        
        # JSON Schema
        scoring_schema = {
            "type": "object",
            "properties": {
                "rating_class": {"type": "string", "enum": ["PERFECT", "GOOD", "AVERAGE", "BAD", "WRONG"]},
                "predicted_iou": {"type": "integer", "minimum": 0, "maximum": 100},
                "boundary_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "semantic_category": {"type": "integer", "minimum": 0, "maximum": 5},
                "semantic_attribute": {"type": "integer", "minimum": 0, "maximum": 5},
                "semantic_context": {"type": "integer", "minimum": 0, "maximum": 5},
                "semantic_count": {"type": "integer", "minimum": 0, "maximum": 5}
            },
            "required": ["rating_class", "predicted_iou", "boundary_score", "semantic_category", "semantic_attribute", "semantic_context", "semantic_count"]
        }
            
        try:
            prompt = f"""/no_think
Evaluate the object located at {box_str}.
Query: "{query}"

Is this object the correct one? rate the segmentation implied by this box.

Output ONLY a JSON object with these scores:
=== RATING CLASS ===
- "PERFECT": Exact match to query
- "GOOD": Correct object
- "AVERAGE": Right type, maybe wrong instance
- "BAD": Wrong object

=== NUMERIC SCORES (0-100/0-5) ===
predicted_iou: Estimate overlap with ground truth
boundary_score: Estimate if box/mask fits object well
semantic_category: Correct object type? (0-5)
semantic_attribute: Attributes match? (0-5)
semantic_context: Context matches? (0-5)
semantic_count: Count matches? (0-5)

Output ONLY the JSON."""

            messages = create_vision_message(prompt, base64_image=base64_img, image_first=True)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "guided_json": scoring_schema,
                    "guided_decoding_backend": "outlines"
                }
            )
            text = completion.choices[0].message.content.strip()
            
            # --- Parse JSON (Logic reused) ---
            json_text = text
            if '</think>' in text:
                json_text = text.split('</think>')[-1].strip()
            elif '</reasoning>' in text.lower():
                json_text = text.split('</reasoning>')[-1].strip()
            
            data = {}
            try:
                parsed = json_repair.loads(json_text)
                if isinstance(parsed, dict): data = parsed
            except Exception:
                match = re.search(r'\{[^{}]*\}', json_text)
                if match: data = json.loads(match.group(0))
            
            # Fallbacks
            if not data:
                if "PERFECT" in text: data["rating_class"] = "PERFECT"
                elif "GOOD" in text: data["rating_class"] = "GOOD"
            
            # Calculate Scores (Reused weighting)
            r_class = data.get("rating_class", "BAD").upper() if isinstance(data.get("rating_class"), str) else "BAD"
            r_vals = {"PERFECT": 100, "GOOD": 75, "AVERAGE": 50, "BAD": 25, "WRONG": 0}
            r_val = r_vals.get(r_class, 25)
            score_rating = r_val * 0.20
            
            score_iou = min(100, max(0, float(data.get("predicted_iou", 0)))) * 0.30
            score_boundary = min(100, max(0, float(data.get("boundary_score", 0)))) * 0.20
            
            s_sum = sum([min(5, int(data.get(k, 0))) for k in ["semantic_category", "semantic_attribute", "semantic_context", "semantic_count"]])
            score_semantic = s_sum 
            
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
            if self.verbose:
                print(f"Composite box score error: {e}")
            return {"total_score": 0, "error": str(e)}

