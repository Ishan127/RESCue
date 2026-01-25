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
        
        # Score all masks in parallel
        pointwise_results = self._pointwise_score_batch(image_input, masks, query)
        
        # Also compute geometric heuristic for tie-breaking
        heuristics = {}
        for r in pointwise_results:
            idx = r['mask_idx']
            heuristics[idx] = self._compute_mask_heuristic(masks[idx])
        
        # Sort by: (total_score DESC, heuristic DESC, index ASC for stability)
        sorted_results = sorted(
            pointwise_results,
            key=lambda r: (
                r.get('total_score', 0),
                heuristics.get(r['mask_idx'], 0),
                -r['mask_idx']  # Lower index wins ties
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
                "reasoning": f"Pointwise score: {res.get('total_score', 0):.1f}/100",
                "pointwise_details": res
            })
        
        if self.verbose:
            top3 = [(r['mask_idx'], r['score']) for r in final_ranking[:3]]
            print(f"[Pointwise] Top 3: {top3}")
        
        return final_ranking

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
        overlay = apply_red_alpha_overlay(image, mask, alpha=0.5)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
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
            # Comprehensive prompt with JSON example and detailed scoring rubrics
            prompt = f"""/no_think
You are a segmentation quality evaluator. The RED highlighted region in the image is a proposed segmentation mask for the query: "{query}"

Your task: Evaluate how well this RED mask matches the query requirements. Output ONLY a JSON object.

=== SCORING METRICS ===

1. rating_class (string): Overall quality assessment
   - "PERFECT": Mask covers exactly the correct object with pixel-accurate boundaries. Very rare - only for exceptional masks.
   - "GOOD": Correct object selected, boundaries are mostly accurate with minor imperfections. Most good masks fall here.
   - "AVERAGE": Right general area/object but noticeable issues (boundary bleeds, includes wrong parts, slightly off).
   - "BAD": Significant problems - wrong parts included, major boundary errors, or partially incorrect object.
   - "WRONG": Completely incorrect object selected, does not match the query at all.

2. predicted_iou (integer 0-100): Estimated Intersection-over-Union with the ideal ground truth mask
   - 90-100: Near-perfect overlap (extremely rare)
   - 75-89: Good overlap with minor differences
   - 50-74: Moderate overlap, noticeable discrepancies
   - 25-49: Poor overlap, significant mismatch
   - 0-24: Very poor or no meaningful overlap

3. boundary_score (integer 0-100): How well the mask edges follow the object contours
   - 90-100: Pixel-perfect edges following exact object boundaries (rare)
   - 70-89: Clean edges with minor imperfections
   - 50-69: Visible boundary issues - jagged edges, bleeding into background
   - 25-49: Poor boundaries - significant edge errors
   - 0-24: Very poor boundaries - mask edges don't follow object at all

4. semantic_category (integer 0-5): Does the mask cover the correct object TYPE/CLASS?
   - 5: Exactly the right object category (e.g., query asks for "dog", mask covers a dog)
   - 3-4: Closely related category (e.g., query asks for "vehicle", mask covers a car)
   - 1-2: Loosely related (e.g., query asks for "animal", mask covers part of an animal)
   - 0: Wrong category entirely

5. semantic_attribute (integer 0-5): Do the object's visual attributes match the query description?
   - 5: Color, shape, size, texture all match the query description perfectly
   - 3-4: Most attributes match, minor discrepancies
   - 1-2: Some attributes match but obvious mismatches exist
   - 0: Attributes don't match at all (e.g., query says "red car" but mask is on blue car)

6. semantic_context (integer 0-5): Does the mask match the ACTION or CONTEXT in the query?
   - 5: Perfect contextual match (e.g., "person running" - mask is on a running person)
   - 3-4: Context mostly matches
   - 1-2: Weak contextual alignment
   - 0: Context mismatch (e.g., "person sitting" but mask is on standing person)

7. semantic_count (integer 0-5): Is the correct NUMBER of instances masked?
   - 5: Exactly the right number of instances (e.g., query asks for "a bird", mask covers exactly one bird)
   - 3-4: Close but not exact (e.g., covers 2 when 1 was requested)
   - 1-2: Significant count mismatch
   - 0: Completely wrong count

=== EXAMPLE OUTPUT ===
{{"rating_class": "GOOD", "predicted_iou": 78, "boundary_score": 72, "semantic_category": 5, "semantic_attribute": 4, "semantic_context": 5, "semantic_count": 5}}

=== IMPORTANT ===
Be CRITICAL and use the full range of scores. Most masks have flaws - look for them!
Respond with ONLY the JSON object, no explanation."""

            messages = create_vision_message(prompt, tmp_path)
            
            # For vLLM with Qwen3: disable thinking via chat_template_kwargs
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,  # Slight randomness for variation
                max_tokens=2048,  # Increased for comprehensive output
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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
