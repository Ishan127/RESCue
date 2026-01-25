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
        self.client = client if client else get_verifier_client()
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

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt, tmp_path)
            
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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def verify_batch_pointwise(self, image_input, masks, query):
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
            single_result = self._score_single_composite(image_input, masks[0], query)
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
        vlm_results = self._pointwise_score_batch(image_input, masks, query)
        
        # 2. Run CLIP Scoring (Batch)
        try:
            if not hasattr(self, 'clip_verifier'):
                from .clip_verifier import ClipVerifier
                self.clip_verifier = ClipVerifier()
            
            clip_scores = self.clip_verifier.verify_batch(image_input, masks, query)
        except Exception as e:
            if self.verbose: print(f"CLIP Error: {e}")
            clip_scores = [0.0] * n


        # 3. Compute Consistency Scores (15%)
        # "Cross-Reasoning Path Agreement": Mean IoU with all other masks
        consistency_scores = []
        if n > 1:
            # We can use a simpler loop for N=64 (4096 checks is instant)
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
            consistency_scores = [1.0] * n # Single mask is consistent with itself

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

            winner_is_a = self._compare_pair(image, masks[wa_idx], masks[wb_idx], query)
            
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

    def _compare_pair(self, image, mask_a, mask_b, query):
        """
        Pairwise VLM comparison. Returns True if A is better, False if B is better.
        """
        try:
            # Prepare composite images
            # Red overlay for both? Or maybe Red vs Blue?
            # User said "put them in a single prompt". 
            # Safest is showing two separate images to VLM.
            
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
            
            prompt = f"""Compare these two segmentation masks for the query: "{query}"

Image 1: Candidate A
Image 2: Candidate B

Which mask is better?
- Check object identity (is it the right object?)
- Check boundaries (is it tight?)
- Check attributes/color

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
                max_tokens=256,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
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

    def _pointwise_score_batch(self, image, masks, query):
        """
        Run parallel pointwise scoring for all masks.
        Returns list of dicts with scores and breakdown.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
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
                    if self.verbose:
                        print(f"Pointwise error on {idx}: {e}")
                    results.append({'mask_idx': idx, 'total_score': 0, 'error': str(e)})
                    
        return results

    def _score_single_composite(self, image, mask, query):
        """
        Score a single mask using the 5-part composite metric (0-100).
        
        Uses vLLM guided JSON decoding to guarantee all fields are returned.
        
        Components:
        1. Geometric (10%) - mask size/coverage heuristic
        2. Rating (20%) - LLM rates PERFECT/GOOD/AVERAGE/BAD/WRONG  
        3. IoU (30%) - LLM predicts estimated IoU
        4. Boundary (20%) - LLM rates edge quality
        5. Semantic (20%) - 4 sub-metrics for category/attribute/context/count
        """
        # 1. Geometric (10%) - purely local calculation
        geo_score_raw = self._compute_mask_heuristic(mask)  # 0-100
        geo_score = geo_score_raw * 0.10
        
        # Prepare VLM Input
        import io
        import base64
        overlay = apply_red_alpha_overlay(image, mask, alpha=0.5, black_background=True)
        
        # Optimize: Save to memory instead of disk
        buffer = io.BytesIO()
        overlay.convert('RGB').save(buffer, format="JPEG", quality=85) # Quality 85 is sufficient/faster
        b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # JSON Schema for guided decoding - enforces all 7 fields
        scoring_schema = {
            "type": "object",
            "properties": {
                "rating_class": {
                    "type": "string",
                    "enum": ["PERFECT", "GOOD", "AVERAGE", "BAD", "WRONG"]
                },
                "predicted_iou": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100
                },
                "boundary_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100
                },
                "semantic_category": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5
                },
                "semantic_attribute": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5
                },
                "semantic_context": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5
                },
                "semantic_count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5
                }
            },
            "required": [
                "rating_class", "predicted_iou", "boundary_score",
                "semantic_category", "semantic_attribute", "semantic_context", "semantic_count"
            ]
        }
            
        try:
            # Balanced prompt for accurate scoring with good variation
            prompt = f"""/no_think
You are evaluating a segmentation mask. The RED region in the image is the proposed mask for: "{query}"

Output ONLY a JSON object with these 7 scores:

=== RATING CLASS ===
- "PERFECT": Exact object, pixel-perfect boundaries (rare, <10% of masks)
- "GOOD": Correct object, minor boundary imperfections (most correct masks: 60%)
- "AVERAGE": Right object but visible issues - boundary bleeds, extra parts (20%)
- "BAD": Partially wrong - includes incorrect regions or misses key parts
- "WRONG": Completely wrong object (only if mask is on unrelated object)

=== NUMERIC SCORES ===
predicted_iou (0-100): Overlap with ideal mask
- 80-95 typical for GOOD masks, 60-79 for AVERAGE, <60 for BAD

boundary_score (0-100): Edge accuracy
- 80-90 for clean edges, 60-79 for minor issues, <60 for poor edges

semantic_category (0-5): Correct object type? (5=exact, 3-4=related, 0=wrong type)
semantic_attribute (0-5): Attributes match? (color/size/shape - 5=perfect, 3-4=mostly, 1-2=some)
semantic_context (0-5): Context/action matches query? (5=perfect, 3-4=mostly)
semantic_count (0-5): Correct instance count? (5=exact count)

=== SCORING GUIDANCE ===
- If the mask covers the RIGHT object type, semantic_category should be 4-5
- Only use rating "WRONG" if the mask is on a completely unrelated object
- Use the full 0-5 range for semantic scores based on specific issues
- Most masks of the correct object are "GOOD" or "AVERAGE", not "PERFECT"

=== EXAMPLES ===
Good mask with minor boundary issues:
{{"rating_class": "GOOD", "predicted_iou": 82, "boundary_score": 75, "semantic_category": 5, "semantic_attribute": 4, "semantic_context": 5, "semantic_count": 5}}

Average mask with some problems:
{{"rating_class": "AVERAGE", "predicted_iou": 68, "boundary_score": 60, "semantic_category": 5, "semantic_attribute": 3, "semantic_context": 4, "semantic_count": 5}}

Wrong object entirely:
{{"rating_class": "WRONG", "predicted_iou": 5, "boundary_score": 20, "semantic_category": 0, "semantic_attribute": 0, "semantic_context": 0, "semantic_count": 0}}

Respond with ONLY the JSON, no explanation."""

            messages = create_vision_message(prompt, base64_image=b64_img)
            
            # For vLLM with Qwen3: disable thinking via chat_template_kwargs
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=16192,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "guided_json": scoring_schema,
                    "guided_decoding_backend": "outlines"
                }
            )
            text = completion.choices[0].message.content.strip()
            
            # Log raw VLM response for debugging
            import datetime
            log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "verifier_vlm_log.txt")
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"\n{'='*60}\n")
                log_f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                log_f.write(f"Query: {query}\n")
                log_f.write(f"Raw VLM Response:\n{text}\n")
            
            # --- Parse JSON ---
            # Strip Qwen3's <think>...</think> block if present
            json_text = text
            if '</think>' in text:
                json_text = text.split('</think>')[-1].strip()
            elif '</reasoning>' in text.lower():
                json_text = text.split('</reasoning>')[-1].strip()
            
            data = {}
            try:
                try:
                    parsed = json_repair.loads(json_text)
                    if isinstance(parsed, dict):
                        data = parsed
                    elif isinstance(parsed, str):
                        if self.verbose:
                            print(f"json_repair returned string, using regex fallback")
                        json_match = re.search(r'\{[^{}]*\}', parsed)
                        if json_match:
                            data = json.loads(json_match.group(0))
                except ImportError:
                    if self.verbose:
                        print("json_repair not found, using regex fallback")
                    json_match = re.search(r'\{[^{}]*\}', json_text)
                    if json_match:
                        data = json.loads(json_match.group(0))
            except Exception as e:
                if self.verbose:
                    print(f"JSON Parsing fully failed: {e}")
                
            # Final fallback: manual key extraction
            if not data:
                if "PERFECT" in text:
                    data["rating_class"] = "PERFECT"
                elif "GOOD" in text:
                    data["rating_class"] = "GOOD"
                iou_m = re.search(r'IOU.*?(\d+)', text, re.IGNORECASE)
                if iou_m:
                    data["predicted_iou"] = int(iou_m.group(1))

            # --- Calculate Scores ---
            
            # 2. Rating (20%)
            r_class = data.get("rating_class", "BAD")
            if isinstance(r_class, str):
                r_class = r_class.upper()
            else:
                r_class = "BAD"
            
            if "PERFECT" in r_class:
                r_val = 100
            elif "GOOD" in r_class:
                r_val = 75
            elif "AVERAGE" in r_class:
                r_val = 50
            elif "WRONG" in r_class:
                r_val = 0
            elif "BAD" in r_class:
                r_val = 25
            else:
                r_val = 25
            score_rating = r_val * 0.20
            
            # 3. IoU (30%)
            pred_iou = float(data.get("predicted_iou", 0))
            score_iou = min(100, max(0, pred_iou)) * 0.30
            
            # 4. Boundary (20%)
            b_qual = float(data.get("boundary_score", 0))
            score_boundary = min(100, max(0, b_qual)) * 0.20
            
            # 5. Semantic (20%) - sum of 4 * 5pts = 20pts max
            # Now using flat field names from guided JSON schema
            s1 = int(data.get("semantic_category", 0))
            s2 = int(data.get("semantic_attribute", 0))
            s3 = int(data.get("semantic_context", 0))
            s4 = int(data.get("semantic_count", 0))
            s_sum = min(5, s1) + min(5, s2) + min(5, s3) + min(5, s4)  # max 20
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
            if self.verbose:
                print(f"Composite score error: {e}")
            return {"total_score": 0, "error": str(e)}

