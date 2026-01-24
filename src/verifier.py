import re
import json
import json_repair
import numpy as np
import os
import tempfile
from PIL import Image
from .utils import apply_red_alpha_overlay
from .api_utils import create_vision_message, get_openai_client


class Verifier:
    """
    Comparative verification: Uses pairwise ranking instead of absolute scoring.
    Recommended model: Qwen3-VL-32B-Thinking (chain-of-thought reasoning)
    """
    def __init__(self, client=None, model_path="Qwen/Qwen3-VL-32B-Thinking", 
                 api_base="http://localhost:8000/v1"):
        self.model_path = model_path
        self.client = client if client else get_openai_client(base_url=api_base)
        self.is_thinking_model = "thinking" in model_path.lower()

    def _create_comparison_grid(self, image, masks, labels):
        overlays = []
        for i, mask in enumerate(masks):
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
            row, col = idx // cols, idx % cols
            x, y = col * w, row * cell_h
            
            from PIL import ImageDraw, ImageFont
            labeled = Image.new('RGB', (w, cell_h), (255, 255, 255))
            labeled.paste(overlay, (0, label_height))
            draw = ImageDraw.Draw(labeled)
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except:
                font = ImageFont.load_default()
            draw.text((w//2 - 30, 5), label, fill=(0, 0, 0), font=font)
            
            grid.paste(labeled, (x, y))
        
        return grid

    def verify_batch_comparative(self, image_input, masks, query):
        if len(masks) == 0:
            return []
        if len(masks) == 1:
            return [{"rank": 1, "score": 100, "reasoning": "Only candidate"}]
        
        labels = [chr(65 + i) for i in range(len(masks))]
        grid_img = self._create_comparison_grid(image_input, masks, labels)
        
        label_list = ", ".join(labels)
        
        if self.is_thinking_model:
            prompt = f"""This image shows {len(masks)} segmentation masks (labeled {label_list}) for the query: "{query}"

            Each panel shows the same image with a different RED mask overlay.

            Think carefully and analyze each mask:
            1. Which RED region best captures the EXACT object described in the query?
            2. Which mask has the tightest boundaries without cutting off parts of the object?
            3. Which mask includes the least background noise?

            After your analysis, rank ALL masks from best to worst.

            Output your final answer as JSON:
            {{"ranking": ["best_label", "second_best", ...], "reasoning": "brief explanation of why the winner is best"}}"""
        else:
            prompt = f"""This image shows {len(masks)} different segmentation masks (labeled {label_list}) for: "{query}"

Each panel shows the SAME image with a different RED highlighted region.

Evaluate each mask:
- Does the RED region cover the correct object?
- Is the boundary tight and accurate?
- Does it avoid excess background?

Rank ALL masks from best to worst.

Output ONLY JSON: {{"ranking": ["{labels[0]}", "{labels[1]}", ...], "reasoning": "why"}}"""

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            grid_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        results = []
        
        try:
            messages = create_vision_message(prompt, tmp_path)
            
            max_tokens = 1024 if self.is_thinking_model else 200
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            text = completion.choices[0].message.content.strip()
            print(f"[Verifier Comparative]: {text[:500]}...")
            
            json_match = re.search(r'\{[^{}]*"ranking"[^{}]*\}', text, re.DOTALL)
            if json_match:
                parsed = json_repair.loads(json_match.group())
            else:
                parsed = json_repair.loads(text)
            
            ranking = parsed.get("ranking", labels)
            reasoning = parsed.get("reasoning", "")
            
            ranking = [r.upper().strip() for r in ranking if r.upper().strip() in labels]
            
            for label in labels:
                if label not in ranking:
                    ranking.append(label)
            
            n = len(ranking)
            for rank_idx, label in enumerate(ranking):
                mask_idx = ord(label) - 65
                score = max(10, 100 - (rank_idx * (90 // max(1, n - 1))))
                results.append({
                    "mask_idx": mask_idx,
                    "rank": rank_idx + 1,
                    "score": score,
                    "label": label,
                    "reasoning": reasoning if rank_idx == 0 else ""
                })
            
            results.sort(key=lambda x: x["mask_idx"])
            
        except Exception as e:
            print(f"Verifier Comparison Error: {e}")
            for i in range(len(masks)):
                results.append({
                    "mask_idx": i,
                    "rank": i + 1,
                    "score": 100 - i * 10,
                    "label": labels[i],
                    "reasoning": f"Error: {str(e)}"
                })
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return results

    def verify(self, image_input, mask, query):
        """Single mask verification (fallback / compatibility)"""
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.5)
        
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        if self.is_thinking_model:
            prompt_text = f"""Look at this image with a RED highlighted region. The task was to segment: "{query}"

Think step by step:
1. What object is the RED region highlighting?
2. Is this the correct object for the query?
3. How well do the boundaries fit - is it too loose, too tight, or just right?
4. What percentage of the target object is covered?

After thinking, output JSON:
{{"score": 0-100, "issues": "list any problems"}}"""
        else:
            prompt_text = f"""Image shows RED highlighted region for query: "{query}"

Rate the segmentation quality from 0-100:
- 90-100: Perfect fit, correct object
- 70-89: Correct object, minor boundary issues
- 50-69: Correct object but significant issues
- 30-49: Partially correct or major issues
- 0-29: Wrong object

Output ONLY JSON: {{"score": X, "issues": "brief description"}}"""

        scores = {"score": 50, "issues": "", "total": 50}
        
        if not self.client:
            return scores
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt_text, tmp_path)
            
            max_tokens = 512 if self.is_thinking_model else 100
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            text = completion.choices[0].message.content.strip()
            print(f"[Verifier]: {text[:300]}...")
            
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
            if json_match:
                parsed = json_repair.loads(json_match.group())
            else:
                parsed = json_repair.loads(text)
            
            score = int(parsed.get("score", 50))
            score = max(0, min(100, score))
            
            scores["score"] = score
            scores["issues"] = str(parsed.get("issues", ""))
            scores["total"] = score
            
        except Exception as e:
            print(f"Verifier Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return scores
        
        return scores
