import re
import numpy as np
import os
import tempfile
from PIL import Image
from .utils import apply_red_alpha_overlay
from .api_utils import create_vision_message, get_openai_client


class Verifier:
    def __init__(self, client=None, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", 
                 api_base="http://localhost:8000/v1", mode="detailed"):
        self.model_path = model_path
        self.mode = mode
        if client:
            self.client = client
        else:
            self.client = get_openai_client(base_url=api_base)
        
        self._batch_scores = []
        self._batch_mode = False

    def start_batch(self):
        self._batch_scores = []
        self._batch_mode = True
    
    def end_batch(self):
        self._batch_mode = False
        if not self._batch_scores:
            return []
        
        scores = np.array(self._batch_scores)
        if scores.max() - scores.min() > 0:
            normalized = (scores - scores.min()) / (scores.max() - scores.min()) * 100
        else:
            normalized = scores
        
        self._batch_scores = []
        return normalized.tolist()

    def verify(self, image_input, mask, query, return_details=False):
        if self.mode == "detailed":
            result = self._verify_detailed(image_input, mask, query)
        else:
            result = self._verify_simple(image_input, mask, query)
        
        score = result["total"] if isinstance(result, dict) else result
        
        if self._batch_mode:
            self._batch_scores.append(score)
        
        if return_details and isinstance(result, dict):
            return result
        return score

    def _create_enhanced_overlay(self, image_input, mask):
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.4)
        
        try:
            import cv2
            mask_np = np.array(mask).astype(np.uint8)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            
            contours, _ = cv2.findContours(
                (mask_np * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            overlay_np = np.array(overlay_img)
            cv2.drawContours(overlay_np, contours, -1, (255, 255, 0), 2)  # Yellow boundary
            overlay_img = Image.fromarray(overlay_np)
        except Exception as e:
            print(f"Boundary highlighting failed: {e}")
        
        return overlay_img

    def _verify_detailed(self, image_input, mask, query):
        overlay_img = self._create_enhanced_overlay(image_input, mask)
        
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        mask_coverage = mask_np.sum() / mask_np.size * 100
        
        prompt_text = f"""You are evaluating a segmentation mask for the query: "{query}"

The RED highlighted region shows the predicted segmentation. Yellow outline shows the boundary.
Mask covers {mask_coverage:.1f}% of the image.

Rate EACH criterion separately from 0-25 points:

1. **OBJECT IDENTITY (0-25)**: Is the highlighted region the correct type of object described in "{query}"?
   - 25: Exactly the right object
   - 15-24: Correct object type but with minor issues
   - 5-14: Partially correct or ambiguous
   - 0-4: Wrong object entirely

2. **SPATIAL ACCURACY (0-25)**: Is the object in the correct location as implied by "{query}"?
   - 25: Perfect location match
   - 15-24: Correct general area
   - 5-14: Somewhat displaced
   - 0-4: Wrong location

3. **COMPLETENESS (0-25)**: Does the mask cover the ENTIRE target object? This should be a direct measure of the IoU.
   - 25: Full coverage, nothing missing
   - 15-24: Minor parts missing
   - 5-14: Significant parts missing
   - 0-4: Most of object not covered

4. **BOUNDARY PRECISION (0-25)**: Are the mask edges accurate to the object's true boundaries?
   - 25: Pixel-perfect boundaries
   - 15-24: Minor boundary errors
   - 5-14: Noticeable over/under-segmentation
   - 0-4: Very poor boundaries, lots of background included

Think step by step, then provide scores in EXACTLY this format:
IDENTITY: <score>
SPATIAL: <score>
COMPLETENESS: <score>
BOUNDARY: <score>
TOTAL: <sum>"""

        scores = {"identity": 0, "spatial": 0, "completeness": 0, "boundary": 0, "total": 0}
        
        if not self.client:
            return scores
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt_text, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent scoring
                max_tokens=512
            )
            
            text = completion.choices[0].message.content
            print(f"[Verifier Detailed]: {text[:200]}...")
            
            # Parse individual scores
            patterns = {
                "identity": r"IDENTITY[:\s]*(\d+(?:\.\d+)?)",
                "spatial": r"SPATIAL[:\s]*(\d+(?:\.\d+)?)",
                "completeness": r"COMPLETENESS[:\s]*(\d+(?:\.\d+)?)",
                "boundary": r"BOUNDARY[:\s]*(\d+(?:\.\d+)?)",
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    scores[key] = min(25, max(0, float(match.group(1))))
            
            # Calculate total
            scores["total"] = sum(scores[k] for k in ["identity", "spatial", "completeness", "boundary"])
            
            # If parsing failed, try to get total directly
            if scores["total"] == 0:
                total_match = re.search(r"TOTAL[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
                if total_match:
                    scores["total"] = min(100, max(0, float(total_match.group(1))))
                else:
                    # Last resort: find any number near the end
                    numbers = re.findall(r"\b(\d{1,3})\b", text[-100:])
                    if numbers:
                        scores["total"] = min(100, max(0, float(numbers[-1])))
            
        except Exception as e:
            print(f"Verifier Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return scores

    def _verify_simple(self, image_input, mask, query):
        """Original simple verification with improved prompting."""
        overlay_img = self._create_enhanced_overlay(image_input, mask)
        
        prompt_text = f"""Query: "{query}"

Evaluate the RED highlighted segmentation region.

Consider:
- Is this the correct object described in the query?
- Does the mask fully cover the target object?
- Are the boundaries accurate (no extra background, no missing parts)?

Be CRITICAL and DISCRIMINATIVE. Most segmentations have flaws.
- Score 90-100: Nearly perfect match
- Score 70-89: Good but with minor issues
- Score 50-69: Acceptable but noticeable problems
- Score 30-49: Poor, significant errors
- Score 0-29: Wrong object or very bad segmentation

Respond with ONLY: Score: <number>"""

        if not self.client:
            return 0.0
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt_text, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,  # Lower for consistency
                max_tokens=128
            )
            
            text = completion.choices[0].message.content
            print(f"[Verifier Simple]: {text}")
            
            # Parse score
            match = re.search(r"Score[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if match:
                return min(100, max(0, float(match.group(1))))
            
            # Fallback
            numbers = re.findall(r"\b(\d{1,3})\b", text)
            if numbers:
                return min(100, max(0, float(numbers[-1])))
            
            return 0.0
            
        except Exception as e:
            print(f"Verifier Error: {e}")
            return 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def verify_comparative(self, image_input, masks, query):
        if len(masks) <= 1:
            return [self.verify(image_input, masks[0], query)] if masks else []
        
        n_masks = min(len(masks), 6)  # Limit to 6 for visual clarity
        
        comparison_images = []
        for i, mask in enumerate(masks[:n_masks]):
            overlay = self._create_enhanced_overlay(image_input, mask)
            # Add label
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(overlay)
            draw.text((10, 10), f"Candidate {i+1}", fill=(255, 255, 255))
            comparison_images.append(overlay)
        
        # Create grid
        w, h = comparison_images[0].size
        cols = min(3, n_masks)
        rows = (n_masks + cols - 1) // cols
        grid = Image.new('RGB', (w * cols, h * rows))
        
        for i, img in enumerate(comparison_images):
            x = (i % cols) * w
            y = (i // cols) * h
            grid.paste(img, (x, y))
        
        prompt_text = f"""Query: "{query}"

You see {n_masks} candidate segmentations labeled "Candidate 1" through "Candidate {n_masks}".

RANK them from BEST to WORST based on:
1. Correct object identification
2. Complete coverage of the target
3. Accurate boundaries

Format your response as:
RANKING: <best_number>, <second_number>, ..., <worst_number>
SCORES: <score1>, <score2>, ..., <scoreN>

Where scores are 0-100 for each candidate in ORDER (Candidate 1's score, Candidate 2's score, etc.)
Ensure scores reflect the ranking - best should have highest score."""

        scores = [50.0] * len(masks)  # Default
        
        if not self.client:
            return scores
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            grid.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt_text, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.3,
                max_tokens=256
            )
            
            text = completion.choices[0].message.content
            print(f"[Verifier Comparative]: {text}")
            
            # Parse scores
            scores_match = re.search(r"SCORES[:\s]*([\d,.\s]+)", text, re.IGNORECASE)
            if scores_match:
                score_strs = re.findall(r"(\d+(?:\.\d+)?)", scores_match.group(1))
                for i, s in enumerate(score_strs[:n_masks]):
                    scores[i] = min(100, max(0, float(s)))
            
            # If scores parsing failed, use ranking to assign scores
            if all(s == 50.0 for s in scores[:n_masks]):
                rank_match = re.search(r"RANKING[:\s]*([\d,\s]+)", text, re.IGNORECASE)
                if rank_match:
                    ranks = re.findall(r"(\d+)", rank_match.group(1))
                    # Assign scores based on rank position
                    base_scores = list(range(100, 100 - n_masks * 15, -15))
                    for pos, rank in enumerate(ranks[:n_masks]):
                        idx = int(rank) - 1
                        if 0 <= idx < n_masks:
                            scores[idx] = base_scores[pos] if pos < len(base_scores) else 30
            
        except Exception as e:
            print(f"Verifier Comparative Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        while len(scores) < len(masks):
            scores.append(30.0)
        
        return scores
