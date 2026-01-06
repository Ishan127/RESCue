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
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RESCue.Planner")

@dataclass(frozen=True)
class PlannerConfig:
    model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct"
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
        
        if LLM is not None:
            logger.info(f"Loading Planner (vLLM): {self.config.model_path}")
            logger.info(f"Config: dtype={self.config.dtype}, quant={self.config.quantization}")
            logger.info(f"Detected Device: {self.device}")
            
            try:
                self.llm = LLM(
                    model=self.config.model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    dtype=self.config.dtype,
                    quantization=self.config.quantization
                )
            except Exception as e:
                logger.error(f"Failed to initialize vLLM: {e}")
                raise e
        else:
            logger.error("vLLM library not found.")
            raise ImportError("vLLM is not installed.")

    def generate_hypotheses(self, image_path: str, query: str, N: int = 1, temperature: float = 0.7) -> List[Dict]:
        if N < 1:
            return []
            
        logger.info(f"Generating {N} hypotheses for query: '{query}'")
        
        base_temp = temperature if temperature is not None else self.config.base_temperature
        
        candidates: List[Hypothesis] = []
        
        if self.config.use_stratified_sampling and N > 1:
            n_conservative = math.ceil(N * 0.4)
            n_exploratory = math.ceil(N * 0.4)
            n_spatial = N - n_conservative - n_exploratory
            if n_spatial < 0: n_spatial = 0
            
            strategies = [
                ("conservative", n_conservative, max(0.1, base_temp - 0.2)),
                ("exploratory", n_exploratory, min(1.0, base_temp + 0.2)),
                ("spatial", n_spatial, base_temp)
            ]
        else:
            strategies = [("standard", N, base_temp)]
            
        for strategy_name, count, temp in strategies:
            if count <= 0: continue
            
            logger.info(f"Strategy '{strategy_name}': generating {count} samples (T={temp:.2f})")
            
            try:
                new_hyps = self._sample_batch(
                    image_path, 
                    query, 
                    count, 
                    temperature=temp,
                    strategy=strategy_name
                )
                candidates.extend(new_hyps)
            except Exception as e:
                logger.error(f"Strategy '{strategy_name}' failed: {e}")

        final_hypotheses = self._select_diverse_subset(candidates, N, query)
        
        logger.info(f"Final selection: {len(final_hypotheses)} hypotheses")
        return [h.to_dict() for h in final_hypotheses]

    def _construct_prompt(self, query: str, strategy: str) -> str:
        base_prompt = (
            f"Query: {query}\n"
            "Analyze the image and locate the object described by the query.\n"
        )
        
        if strategy == "conservative":
            instruction = (
                "Step 1 (See): List visible objects strictly related to the query.\n"
                "Step 2 (Think): perform a logical deduction to identify the correct object.\n"
                "Step 3 (Propose): Output the precise bounding box.\n"
            )
        elif strategy == "exploratory":
            instruction = (
                "Step 1 (Brainstorm): Consider figurative or indirect interpretations of the query.\n"
                "Step 2 (Select): Choose the most likely candidate even if ambiguous.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        elif strategy == "spatial":
            instruction = (
                "Step 1 (Scan): Scan the image distinct regions (foreground, background).\n"
                "Step 2 (Locate): Identify the target based on its spatial context.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        else:
            instruction = (
                "Step 1 (See): List all objects visible in the image that are relevant to the query.\n"
                "Step 2 (Think): Analyze the spatial and causal relationships. Resolve ambiguities.\n"
                "Step 3 (Propose): Output the bounding box coordinates [x_1, y_1, x_2, y_2] for the target object.\n"
            )

        format_instr = (
            "Format your answer as:\n"
            "Reasoning: <your reasoning>\n"
            "Target Concept: <noun phrase>\n"
            "Box: [x1, y1, x2, y2]"
        )
        
        return f"{base_prompt}{instruction}{format_instr}"

    def _sample_batch(self, image_path: str, query: str, n: int, temperature: float, strategy: str) -> List[Hypothesis]:
        prompt_text = self._construct_prompt(query, strategy)
        
        inputs = [
            {
                "prompt": f"<|image_pad|>{prompt_text}",
                "multi_modal_data": {"image": image_path},
            }
        ]
        sampling_params = SamplingParams(
            temperature=temperature, 
            n=n, 
            max_tokens=600,
            logprobs=1  
        )
        
        outputs = self.llm.generate(
            [inputs[0]["prompt"]],
            sampling_params=sampling_params,
        )
        
        batch_hypotheses = []
        from PIL import Image
        
        with Image.open(image_path) as img:
            real_w, real_h = img.size
            img_area = real_w * real_h

        for completion in outputs[0].outputs:
            text = completion.text
            
            if completion.logprobs:
                try:
                    token_logprobs = [list(t.values())[0].logprob for t in completion.logprobs if t]
                    avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else -99.0
                    confidence = math.exp(avg_logprob) # Rough probability [0, 1]
                except Exception:
                    confidence = 0.5
            else:
                confidence = 0.5

            parsed = self._parse_completion(
                text, 
                real_w, 
                real_h, 
                confidence, 
                strategy,
                img_area
            )
            
            if parsed:
                batch_hypotheses.append(parsed)
            else:
                logger.debug(f"Failed to parse completion: {text[:50]}...")

        return batch_hypotheses

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
            
        max_len = max(len(c.reasoning) for c in candidates) if candidates else 1
        
        def score_fn(h: Hypothesis):
            len_score = len(h.reasoning) / max_len
            return 0.7 * h.confidence + 0.3 * len_score

        ranked = sorted(candidates, key=score_fn, reverse=True)
        
        selected: List[Hypothesis] = []
        
        for cand in ranked:
            if len(selected) >= N:
                break
                
            is_new = True
            for existing in selected:
                iou = cand.iou(existing)
                if iou > self.config.iou_threshold:
                    is_new = False
                    break
            
            if is_new:
                selected.append(cand)
                
        if len(selected) < N:
            remaining_needed = N - len(selected)
            others = [c for c in ranked if c not in selected]
            
            def max_overlap(h):
                if not selected: return 0.0
                return max(h.iou(s) for s in selected)
                
            others_sorted = sorted(others, key=max_overlap)
            
            selected.extend(others_sorted[:remaining_needed])
            
        return selected

def get_default_planner(model_path="Qwen/Qwen2.5-VL-72B-Instruct"):
    config = PlannerConfig(model_path=model_path)
    return Planner(config)
