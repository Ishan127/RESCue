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
from .api_utils import get_openai_client, create_vision_message
# from vllm import LLM, SamplingParams # Removed vLLM dependency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RESCue.Planner")

@dataclass(frozen=True)
class PlannerConfig:
    model_path: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    api_base: Optional[str] = "http://localhost:8000/v1"
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
                self.client = get_openai_client(base_url=self.config.api_base)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise e
        else:
             logger.error("API base URL not provided. Local inference is removed in this refactor.")
             raise ValueError("API base URL is required.")

    def generate_hypotheses(self, image_path: str, query: str, N: int = 1, temperature: float = 0.7) -> List[Dict]:
        if N < 1:
            return []
            
        logger.info(f"Generating {N} hypotheses for query: '{query}'")
        
        base_temp = temperature if temperature is not None else self.config.base_temperature
        
        candidates: List[Hypothesis] = []
        
        # Track all concepts across strategies to avoid duplicates
        all_concepts: List[str] = []
        
        if self.config.use_stratified_sampling and N > 1:
            # Distribute more evenly and ensure spatial gets at least 1
            n_conservative = max(1, N // 3)
            n_exploratory = max(1, N // 3)
            n_spatial = max(1, N - n_conservative - n_exploratory)
            
            strategies = [
                ("conservative", n_conservative, max(0.3, base_temp - 0.2)),
                ("exploratory", n_exploratory, min(1.0, base_temp + 0.3)),
                ("spatial", n_spatial, base_temp + 0.1)
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
                    strategy=strategy_name,
                    existing_concepts=all_concepts  # Pass existing concepts to avoid duplicates
                )
                candidates.extend(new_hyps)
                all_concepts.extend([h.target_concept for h in new_hyps])
            except Exception as e:
                logger.error(f"Strategy '{strategy_name}' failed: {e}")

        final_hypotheses = self._select_diverse_subset(candidates, N, query)
        
        logger.info(f"Final selection: {len(final_hypotheses)} hypotheses")
        return [h.to_dict() for h in final_hypotheses]

    def _construct_prompt(self, query: str, strategy: str, attempt: int = 0) -> str:
        base_prompt = (
            f"Query: {query}\n"
            "Analyze the image and locate the object described by the query.\n"
        )
        
        # Add diversity hints based on attempt number
        diversity_hints = ""
        if attempt > 0:
            diversity_options = [
                "Consider a DIFFERENT object than the most obvious answer. Look for alternative interpretations.\n",
                "Focus on a DIFFERENT REGION of the image than the center. Look at edges, corners, background.\n",
                "Think about PARTS of objects rather than whole objects. What components match the query?\n",
                "Consider SMALLER or PARTIALLY VISIBLE objects that might also match the description.\n",
                "Look for objects that match the query INDIRECTLY or METAPHORICALLY.\n",
            ]
            diversity_hints = diversity_options[attempt % len(diversity_options)]
        
        if strategy == "conservative":
            instruction = (
                "Step 1 (See): List ALL visible objects that could possibly relate to the query (at least 3 candidates).\n"
                "Step 2 (Think): For EACH candidate, explain why it might or might not match. Select the MOST LITERAL match.\n"
                "Step 3 (Propose): Output the precise bounding box for YOUR CHOSEN object.\n"
                f"{diversity_hints}"
            )
        elif strategy == "exploratory":
            instruction = (
                "Step 1 (Brainstorm): List at least 3 DIFFERENT possible interpretations of the query, including figurative or indirect ones.\n"
                "Step 2 (Select): Choose the LEAST OBVIOUS but still valid interpretation. Avoid the most common answer.\n"
                "Step 3 (Propose): Output the bounding box for this alternative object.\n"
                f"{diversity_hints}"
            )
        elif strategy == "spatial":
            instruction = (
                "Step 1 (Scan): Divide the image into regions (top-left, top-right, bottom-left, bottom-right, center). List objects in EACH region.\n"
                "Step 2 (Locate): Find matching objects in DIFFERENT REGIONS than the obvious center. Prefer objects at the edges or in the background.\n"
                "Step 3 (Propose): Output the bounding box.\n"
                f"{diversity_hints}"
            )
        else:
            instruction = (
                "Step 1 (See): List ALL objects visible in the image that could match the query (minimum 3).\n"
                "Step 2 (Think): Analyze spatial and causal relationships. Consider multiple valid answers.\n"
                "Step 3 (Propose): Output the bounding box for ONE specific object.\n"
                f"{diversity_hints}"
            )

        format_instr = (
            "\nIMPORTANT: Coordinates are normalized 0-1000. Box format: [x1, y1, x2, y2] where (x1,y1) is top-left.\n"
            "Format your answer EXACTLY as:\n"
            "Reasoning: <your step-by-step reasoning>\n"
            "Target Concept: <specific noun phrase for the object, be precise>\n"
            "Box: [x1, y1, x2, y2]"
        )
        
        return f"{base_prompt}{instruction}{format_instr}"

    def _sample_batch(self, image_path: str, query: str, n: int, temperature: float, strategy: str, existing_concepts: List[str] = None) -> List[Hypothesis]:
        """Sample a batch of hypotheses with retry logic for diversity."""
        from PIL import Image
        
        with Image.open(image_path) as img:
            real_w, real_h = img.size
            img_area = real_w * real_h
        
        batch_hypotheses = []
        existing_concepts = existing_concepts or []
        max_attempts = n + self.config.max_retries  # Extra attempts to ensure we get n diverse results
        
        for attempt in range(max_attempts):
            if len(batch_hypotheses) >= n:
                break
            
            # Construct prompt with diversity hint based on attempt
            prompt_text = self._construct_prompt(query, strategy, attempt=attempt)
            
            # Add exclusion list if we have existing concepts
            all_concepts = existing_concepts + [h.target_concept for h in batch_hypotheses]
            if all_concepts:
                exclude_text = f"\nIMPORTANT: Do NOT select these objects (already chosen): {', '.join(set(all_concepts))}. Choose a DIFFERENT object.\n"
                prompt_text = prompt_text.replace("Format your answer EXACTLY as:", exclude_text + "Format your answer EXACTLY as:")
            
            messages = create_vision_message(prompt_text, image_path)
            
            # Request more samples than needed to increase diversity
            samples_to_request = min(n - len(batch_hypotheses) + 1, 3)
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model_path,
                    messages=messages,
                    temperature=min(1.0, temperature + 0.1 * attempt),  # Increase temp on retries
                    n=samples_to_request,
                    max_tokens=800,
                    logprobs=True,
                    top_logprobs=1
                )
            except Exception as e:
                logger.error(f"API Call Failed: {e}")
                continue
            
            for choice in completion.choices:
                if len(batch_hypotheses) >= n:
                    break
                    
                text = choice.message.content
                
                # Estimate confidence from logprobs if available
                confidence = 0.5
                if choice.logprobs and choice.logprobs.content:
                    try:
                        token_logprobs = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
                        if token_logprobs:
                            avg_logprob = sum(token_logprobs) / len(token_logprobs)
                            confidence = math.exp(avg_logprob)
                    except Exception:
                        pass

                parsed = self._parse_completion(
                    text, 
                    real_w, 
                    real_h, 
                    confidence, 
                    strategy,
                    img_area
                )
                
                if parsed:
                    # Check if this concept is too similar to existing ones
                    concept_lower = parsed.target_concept.lower().strip()
                    is_duplicate = False
                    for existing in all_concepts + [h.target_concept for h in batch_hypotheses]:
                        if self._concepts_similar(concept_lower, existing.lower().strip()):
                            is_duplicate = True
                            logger.debug(f"Skipping duplicate concept: {parsed.target_concept} (similar to {existing})")
                            break
                    
                    if not is_duplicate:
                        batch_hypotheses.append(parsed)
                        logger.info(f"  Added hypothesis: {parsed.target_concept} (attempt {attempt+1})")
                else:
                    logger.debug(f"Failed to parse completion: {text[:100]}...")
        
        if len(batch_hypotheses) < n:
            logger.warning(f"Only generated {len(batch_hypotheses)}/{n} diverse hypotheses after {max_attempts} attempts")
        
        return batch_hypotheses
    
    def _concepts_similar(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts are semantically similar."""
        # Exact match
        if concept1 == concept2:
            return True
        
        # One contains the other
        if concept1 in concept2 or concept2 in concept1:
            return True
        
        # Word overlap check
        words1 = set(concept1.split())
        words2 = set(concept2.split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'with'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return False
        
        # High overlap = similar
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
        
        # First, deduplicate by concept
        concept_groups: Dict[str, List[Hypothesis]] = defaultdict(list)
        for c in candidates:
            concept_key = c.target_concept.lower().strip()
            concept_groups[concept_key].append(c)
        
        # Take best from each concept group
        unique_candidates = []
        for concept, group in concept_groups.items():
            # Sort by confidence and take best
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
        
        # Select diverse hypotheses based on BOTH box IoU AND concept similarity
        for cand in ranked:
            if len(selected) >= N:
                break
                
            is_diverse = True
            for existing in selected:
                # Check box overlap
                iou = cand.iou(existing)
                if iou > self.config.iou_threshold:
                    is_diverse = False
                    break
                
                # Check concept similarity
                if self._concepts_similar(cand.target_concept.lower(), existing.target_concept.lower()):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(cand)
                logger.info(f"  Selected: {cand.target_concept} (conf={cand.confidence:.2f})")
                
        # If we still need more, add remaining with lowest overlap
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
