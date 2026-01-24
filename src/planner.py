import re
import numpy as np
import torch
import logging
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import defaultdict

from .utils import get_device
from .api_utils import get_planner_client, create_vision_message, PLANNER_MODEL, PLANNER_API_BASE

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("RESCue.Planner")
logger.setLevel(logging.WARNING)

@dataclass(frozen=True)
class PlannerConfig:
    model_path: str = PLANNER_MODEL  # Fast 7B model for planning
    api_base: Optional[str] = PLANNER_API_BASE  # Port 8002 by default
    device: Optional[str] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    
    use_stratified_sampling: bool = True
    base_temperature: float = 0.7
    diversity_factor: float = 1.0  
    iou_threshold: float = 0.65   
    
    min_box_area_ratio: float = 0.001  
    max_retries: int = 3

@dataclass
class Hypothesis:
    box: List[int]
    reasoning: str
    target_concept: str
    confidence: float
    source_strategy: str
    raw_text: str
    
    @property
    def area(self) -> int:
        w = max(0, self.box[2] - self.box[0])
        h = max(0, self.box[3] - self.box[1])
        return w * h
    
    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        return (cx, cy)
    
    def iou(self, other: 'Hypothesis') -> float:
        x1_a, y1_a, x2_a, y2_a = self.box
        x1_b, y1_b, x2_b, y2_b = other.box
        
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)
        
        w_i = max(0, x2_i - x1_i)
        h_i = max(0, y2_i - y1_i)
        inter = w_i * h_i
        
        if inter == 0:
            return 0.0
            
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box,
            "reasoning": self.reasoning,
            "noun_phrase": self.target_concept,
            "confidence": self.confidence,
            "strategy": self.source_strategy,
            "raw_text": self.raw_text
        }

class Planner:
    def __init__(self, config: PlannerConfig = None, **kwargs):
        if config is None:
            self.config = PlannerConfig(**kwargs)
        else:
            self.config = config
            
        self.device = self.config.device or get_device()
        self.client = None
        
        if self.config.api_base:
            logger.info(f"Initializing Planner with API: {self.config.api_base}")
            try:
                self.client = get_planner_client()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise e
        else:
             logger.error("API base URL not provided. Local inference is removed in this refactor.")
             raise ValueError("API base URL is required.")

    def generate_hypotheses(self, image_path: str, query: str, N: int = 1, temperature: float = 0.7, parallel: bool = True) -> List[Dict]:
        if N < 1:
            return []
        
        base_temp = temperature if temperature is not None else self.config.base_temperature
        
        query_configs = self._generate_query_configs(query, N, base_temp)
        
        candidates: List[Hypothesis] = []
        
        from PIL import Image
        with Image.open(image_path) as img:
            real_w, real_h = img.size
            img_area = real_w * real_h
        
        if parallel and len(query_configs) > 1:
            candidates = self._generate_hypotheses_parallel(
                image_path, query_configs, real_w, real_h, img_area
            )
        else:
            for i, config in enumerate(query_configs):
                varied_query = config["query"]
                strategy = config["strategy"]
                temp = config["temperature"]
                
                try:
                    hyp = self._generate_single_hypothesis(
                        image_path, 
                        varied_query,
                        strategy,
                        temp,
                        real_w, real_h, img_area
                    )
                    if hyp:
                        candidates.append(hyp)
                except Exception as e:
                    pass
        
        final_hypotheses = self._select_diverse_subset(candidates, N, query)
        return [h.to_dict() for h in final_hypotheses]
    
    def _generate_hypotheses_parallel(self, image_path: str, query_configs: List[Dict], 
                                       w: int, h: int, img_area: int) -> List[Hypothesis]:
        """Generate hypotheses in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        candidates = []
        max_workers = min(8, len(query_configs))
        
        def generate_one(config):
            try:
                return self._generate_single_hypothesis(
                    image_path,
                    config["query"],
                    config["strategy"],
                    config["temperature"],
                    w, h, img_area
                )
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_one, cfg) for cfg in query_configs]
            
            for future in as_completed(futures):
                try:
                    hyp = future.result()
                    if hyp:
                        candidates.append(hyp)
                except:
                    pass
        
        return candidates
    
    def _generate_query_configs(self, original_query: str, N: int, base_temp: float) -> List[Dict]:
        configs = []
        
        configs.append({
            "query": original_query,
            "strategy": "original",
            "temperature": base_temp
        })
        
        if N == 1:
            return configs
        
        variations = self._generate_query_variations(original_query, N - 1)
        
        for i, varied_query in enumerate(variations):
            if i < 2: 
                strategy = "conservative"
                temp = max(0.3, base_temp - 0.2)
            elif i < 4: 
                strategy = "exploratory"
                temp = min(1.0, base_temp + 0.2)
            else:  
                strategy = "spatial"
                temp = base_temp + 0.1
            
            configs.append({
                "query": varied_query,
                "strategy": strategy,
                "temperature": temp
            })
        
        return configs[:N]  
    
    def _generate_query_variations(self, original_query: str, num_variations: int) -> List[str]:
        prompt = f"""Given this image segmentation query: "{original_query}"

        Generate {num_variations} DIFFERENT ways to interpret or rephrase this query. Each variation should:
        1. Focus on a DIFFERENT aspect or interpretation of what's being asked
        2. Be specific and actionable for locating an object in an image
        3. Consider literal, functional, visual, or contextual interpretations

        Output as JSON array of strings:
        {{"variations": ["variation 1", "variation 2", ...]}}

        Examples of good variations for "What could hold water?":
        - "a container or vessel that can store liquid" (literal/functional)
        - "a cup, bowl, or glass visible in the scene" (specific objects)
        - "something with a concave shape that could collect water" (visual property)
        - "a sink, bathtub, or plumbing fixture" (contextual/environmental)"""

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            text = completion.choices[0].message.content
            
            import json
            json_match = re.search(r'\{[^{}]*"variations"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                variations = data.get("variations", [])
                if variations:
                    return variations[:num_variations]
            
            quoted = re.findall(r'"([^"]{10,})"', text)
            if quoted:
                return quoted[:num_variations]
                
        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
        
        fallbacks = [
            f"Find the most obvious {original_query.split()[-1] if original_query.split() else 'object'}",
            f"Look for something that matches: {original_query}",
            f"Identify the object described by: {original_query}",
        ]
        return fallbacks[:num_variations]
    
    def _generate_single_hypothesis(self, image_path: str, query: str, strategy: str, 
                                     temperature: float, w: int, h: int, img_area: int) -> Optional[Hypothesis]:
        prompt_text = self._construct_prompt(query, strategy)
        messages = create_vision_message(prompt_text, image_path)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=messages,
                temperature=temperature,
                max_tokens=400,
                logprobs=True,
                top_logprobs=1
            )
        except Exception as e:
            logger.error(f"API Call Failed: {e}")
            return None
        
        choice = completion.choices[0]
        text = choice.message.content
        
        confidence = 0.5
        if choice.logprobs and choice.logprobs.content:
            try:
                token_logprobs = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
                if token_logprobs:
                    avg_logprob = sum(token_logprobs) / len(token_logprobs)
                    confidence = math.exp(avg_logprob)
            except Exception:
                pass
        
        return self._parse_completion(text, w, h, confidence, strategy, img_area)

    def _construct_prompt(self, query: str, strategy: str) -> str:
        """Construct prompt based on strategy type."""
        base_prompt = (
            f"Query: {query}\n"
            "Analyze the image and locate the object described by the query.\n"
        )
        
        if strategy == "original":
            instruction = (
                "Step 1 (See): Identify the object that DIRECTLY matches the query.\n"
                "Step 2 (Think): Verify this is the most relevant match.\n"
                "Step 3 (Propose): Output the precise bounding box.\n"
            )
        elif strategy == "conservative":
            instruction = (
                "Step 1 (See): List ALL visible objects that could relate to the query.\n"
                "Step 2 (Think): Select the MOST LITERAL and OBVIOUS match.\n"
                "Step 3 (Propose): Output the precise bounding box.\n"
            )
        elif strategy == "exploratory":
            instruction = (
                "Step 1 (Brainstorm): Consider ALTERNATIVE or LESS OBVIOUS interpretations.\n"
                "Step 2 (Select): Choose an object that matches INDIRECTLY or FUNCTIONALLY.\n"
                "Step 3 (Propose): Output the bounding box for this alternative.\n"
            )
        elif strategy == "spatial":
            instruction = (
                "Step 1 (Scan): Look at DIFFERENT REGIONS - edges, corners, background.\n"
                "Step 2 (Locate): Find a matching object NOT in the obvious center area.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        else:
            instruction = (
                "Step 1 (See): Identify objects relevant to the query.\n"
                "Step 2 (Think): Analyze and select the best match.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )

        format_instr = (
            "\nCoordinates are normalized 0-1000. Box format: [x1, y1, x2, y2] where (x1,y1) is top-left.\n"
            "Format your answer EXACTLY as:\n"
            "Reasoning: <your reasoning>\n"
            "Target Concept: <DESCRIPTIVE phrase with color/size/position, e.g., 'large black tripod on the left'>\n"
            "Box: [x1, y1, x2, y2]"
        )
        
        return f"{base_prompt}{instruction}{format_instr}"

    def _concepts_similar(self, concept1: str, concept2: str) -> bool:
        if concept1 == concept2:
            return True
        
        if concept1 in concept2 or concept2 in concept1:
            return True
        
        words1 = set(concept1.split())
        words2 = set(concept2.split())
        
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'with'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union > 0.5 if union > 0 else False

    def _parse_completion(self, text: str, w: int, h: int, conf: float, strategy: str, img_area: int) -> Optional[Hypothesis]:
        
        box_match = re.search(r"Box:\s*\[\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*\]", text)
        
        if not box_match:
            box_match = re.search(r"\[\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*\]", text)
            
        if not box_match:
            return None
            
        try:
            coords = [int(c) for c in box_match.groups()]
            x1, y1, x2, y2 = coords
            
            px1 = int(x1 / 1000 * w)
            py1 = int(y1 / 1000 * h)
            px2 = int(x2 / 1000 * w)
            py2 = int(y2 / 1000 * h)
            
            px1 = max(0, min(px1, w))
            py1 = max(0, min(py1, h))
            px2 = max(0, min(px2, w))
            py2 = max(0, min(py2, h))
            
            if px2 <= px1 or py2 <= py1:
                return None
                
            box_area = (px2 - px1) * (py2 - py1)
            if box_area < img_area * self.config.min_box_area_ratio:
                return None 
                
            concept_match = re.search(r"Target Concept:\s*(.*)", text)
            reasoning_match = re.search(r"Reasoning:\s*(.*)", text, re.DOTALL)
            
            concept = concept_match.group(1).strip() if concept_match else "object"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            if "Target Concept:" in reasoning:
                reasoning = reasoning.split("Target Concept:")[0].strip()
            
            return Hypothesis(
                box=[px1, py1, px2, py2],
                reasoning=reasoning,
                target_concept=concept,
                confidence=conf,
                source_strategy=strategy,
                raw_text=text
            )
            
        except Exception as e:
            logger.warning(f"Error parsing box coords: {e}")
            return None

    def _select_diverse_subset(self, candidates: List[Hypothesis], N: int, query: str) -> List[Hypothesis]:
        if not candidates:
            return []
        
        concept_groups: Dict[str, List[Hypothesis]] = defaultdict(list)
        for c in candidates:
            concept_key = c.target_concept.lower().strip()
            concept_groups[concept_key].append(c)
        
        unique_candidates = []
        for concept, group in concept_groups.items():
            best = max(group, key=lambda h: h.confidence)
            unique_candidates.append(best)
        
        logger.info(f"Deduplicated {len(candidates)} -> {len(unique_candidates)} unique concepts")
        
        if not unique_candidates:
            return []
            
        max_len = max(len(c.reasoning) for c in unique_candidates) if unique_candidates else 1
        
        def score_fn(h: Hypothesis):
            len_score = len(h.reasoning) / max_len if max_len > 0 else 0
            return 0.6 * h.confidence + 0.4 * len_score

        ranked = sorted(unique_candidates, key=score_fn, reverse=True)
        
        selected: List[Hypothesis] = []
        
        for cand in ranked:
            if len(selected) >= N:
                break
                
            is_diverse = True
            for existing in selected:
                iou = cand.iou(existing)
                if iou > self.config.iou_threshold:
                    is_diverse = False
                    break
                
                if self._concepts_similar(cand.target_concept.lower(), existing.target_concept.lower()):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(cand)
                logger.info(f"  Selected: {cand.target_concept} (conf={cand.confidence:.2f})")
                
        if len(selected) < N:
            remaining_needed = N - len(selected)
            others = [c for c in ranked if c not in selected]
            
            def diversity_score(h):
                if not selected:
                    return 1.0
                max_iou = max(h.iou(s) for s in selected)
                concept_penalty = 0.5 if any(self._concepts_similar(h.target_concept.lower(), s.target_concept.lower()) for s in selected) else 0
                return 1.0 - max_iou - concept_penalty
                
            others_sorted = sorted(others, key=diversity_score, reverse=True)
            
            for h in others_sorted[:remaining_needed]:
                selected.append(h)
                logger.info(f"  Added (fill): {h.target_concept} (conf={h.confidence:.2f})")
            
        return selected

def get_default_planner(model_path="Qwen/Qwen3-VL-30B-A3B-Instruct"):
    config = PlannerConfig(model_path=model_path)
    return Planner(config)
