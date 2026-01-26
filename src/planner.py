import re
import numpy as np
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
            from .api_utils import get_openai_client
            # Create client explicitly with the configured API base
            self.client = get_openai_client(base_url=self.config.api_base)
        else:
             logger.error("API base URL not provided. Local inference is removed in this refactor.")
             raise ValueError("API base URL is required.")

    def generate_hypotheses(self, image_input: Union[str, Any], query: str, N: int = 1, temperature: float = 0.7, parallel: bool = True) -> List[Dict]:
        if N < 1:
            return []
        
        base_temp = temperature if temperature is not None else self.config.base_temperature
        
        # Over-generate by 25% to allow for selection filtering
        target_count = int(N * 1.25)
        if target_count < N + 2: target_count = N + 2 # Ensure at least some selection buffer
        
        query_configs = self._generate_query_configs(query, target_count, base_temp)
        
        candidates: List[Hypothesis] = []
        
        from PIL import Image
        if isinstance(image_input, str):
            with Image.open(image_input) as img:
                real_w, real_h = img.size
                img_area = real_w * real_h
                # Keep path for parallel case if needed, or load if using threads
        else:
            # Assume PIL Image
            real_w, real_h = image_input.size
            img_area = real_w * real_h
        
        if parallel and len(query_configs) > 1:
            candidates = self._generate_hypotheses_parallel(
                image_input, query_configs, real_w, real_h, img_area
            )
        else:
            for i, config in enumerate(query_configs):
                varied_query = config["query"]
                strategy = config["strategy"]
                temp = config["temperature"]
                
                try:
                    hyp = self._generate_single_hypothesis(
                        image_input, 
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
        
        # --- PAD WITH RANDOM HYPOTHESES IF < N ---
        if len(final_hypotheses) < N:
            needed = N - len(final_hypotheses)
            logger.warning(f"Generated {len(final_hypotheses)} hypotheses. Padding with {needed} random hypotheses.")
            import random
            
            for i in range(needed):
                # Random box within image coords - with bounds checking
                max_x1 = max(1, int(real_w * 0.8))
                max_y1 = max(1, int(real_h * 0.8))
                x1 = random.randint(0, max_x1)
                y1 = random.randint(0, max_y1)
                
                # Ensure valid width/height ranges
                min_w = max(1, int(real_w * 0.05))
                max_w = max(min_w + 1, int(real_w - x1))
                min_h = max(1, int(real_h * 0.05))
                max_h = max(min_h + 1, int(real_h - y1))
                
                w = random.randint(min_w, max_w)
                h = random.randint(min_h, max_h)
                
                box = [x1, y1, x1 + w, y1 + h]
                
                final_hypotheses.append(Hypothesis(
                    box=box,
                    reasoning="Random fill to meet N requirement",
                    target_concept="random fill",
                    confidence=0.1,
                    source_strategy="random",
                    raw_text="Randomly generated"
                ))
        # ----------------------------------------
        
        return [h.to_dict() for h in final_hypotheses]
    
    def _generate_hypotheses_parallel(self, image_input: Union[str, Any], query_configs: List[Dict], 
                                       w: int, h: int, img_area: int) -> List[Hypothesis]:
        """Generate hypotheses in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        candidates = []
        # Use more workers since planner has max-num-seqs=8192 now on MI325X
        max_workers = min(128, len(query_configs))
        
        def generate_one(config):
            try:
                return self._generate_single_hypothesis(
                    image_input,
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
        """Generate N hypothesis configs with diverse strategies."""
        configs = []
        
        # Define strategy distribution for N hypotheses
        # Each strategy gets roughly equal share, with different temperatures
        strategies = [
            ("original", base_temp),           # Direct match
            ("conservative", base_temp - 0.2), # Most literal interpretation
            ("exploratory", base_temp + 0.2),  # Alternative interpretations
            ("spatial", base_temp + 0.1),      # Different regions
            ("functional", base_temp),         # By function/purpose
            ("visual", base_temp + 0.1),       # By visual attributes
            ("contextual", base_temp),         # By scene context
            ("part_whole", base_temp),         # Part of larger object or container
        ]
        
        # First: one hypothesis with original query for each strategy
        for strategy, temp in strategies:
            if len(configs) >= N:
                break
            configs.append({
                "query": original_query,
                "strategy": strategy,
                "temperature": max(0.3, min(1.0, temp))
            })
        
        if len(configs) >= N:
            return configs[:N]
        
        # Generate query variations for remaining slots
        remaining = N - len(configs)
        variations = self._generate_query_variations(original_query, remaining)
        
        # Distribute variations across strategies (cycling)
        for i, varied_query in enumerate(variations):
            strategy, temp = strategies[i % len(strategies)]
            # Add some randomness to temperature, but ensure it stays positive before clamping
            temp_offset = (i % 3 - 1) * 0.1  # -0.1 to +0.1 (smaller range)
            adjusted_temp = max(0.3, min(1.0, temp + temp_offset))
            configs.append({
                "query": varied_query,
                "strategy": strategy,
                "temperature": adjusted_temp
            })
        
        return configs[:N]  
    
    def _generate_query_variations(self, original_query: str, num_variations: int) -> List[str]:
        """Generate query variations in batches for reliability."""
        all_variations = []
        batch_size = 50  # Increased from 10 to 50 for N=256 scaling
        
        while len(all_variations) < num_variations:
            needed = min(batch_size, num_variations - len(all_variations))
            batch = self._generate_variation_batch(original_query, needed, all_variations)
            if not batch:
                break
            all_variations.extend(batch)
        
        # If we didn't get enough, add synthetic variations
        if len(all_variations) < num_variations:
            all_variations.extend(self._generate_synthetic_variations(
                original_query, num_variations - len(all_variations)
            ))
        
        return all_variations[:num_variations]
    
    def _generate_variation_batch(self, original_query: str, count: int, existing: List[str]) -> List[str]:
        """Generate a batch of query variations using guided JSON."""
        existing_note = ""
        if existing:
            existing_note = f"\n\nAlready generated (DO NOT repeat): {existing[:5]}..."
        
        prompt = f"""You are a query variation generator.
Original Query: "{original_query}"

Task: Generate {count} distinct rephrasings for object localization.
- Variations must be short phrases, not questions.
- Types: Literal, Functional ("object used for..."), Visual ("red object..."), Spatial ("object in corner...").
{existing_note}

Output ONLY a JSON object with this exact format:
{{
  "variations": [
    "variation 1",
    "variation 2",
    ...
  ]
}}"""

        # JSON schema for a list of strings
        schema = {
            "type": "object",
            "properties": {
                "variations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": count,
                    "maxItems": count + 2
                }
            },
            "required": ["variations"]
        }

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2048*12,
                extra_body={
                    "guided_json": schema,
                    "guided_decoding_backend": "outlines"
                }
            )
            
            text = completion.choices[0].message.content
            
            import json
            data = {}
            try:
                # 1. Try standard JSON
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    data = parsed
            except:
                try:
                    # 2. Try json_repair
                    import json_repair
                    parsed = json_repair.loads(text)
                    if isinstance(parsed, dict):
                        data = parsed
                    elif isinstance(parsed, list): # Model might just return the list array directly
                         data = {"variations": parsed}
                except:
                    # 3. Fallback regex for {"variations": [...]}
                    match = re.search(r'\{.*"variations".*\}', text, re.DOTALL)
                    if match:
                        try:
                            parsed = json.loads(match.group())
                            if isinstance(parsed, dict):
                                data = parsed
                        except:
                            pass
                    else:
                        # 4. Fallback regex for just the list [...]
                        match_list = re.search(r'\[.*\]', text, re.DOTALL)
                        if match_list:
                            try:
                                parsed = json.loads(match_list.group())
                                if isinstance(parsed, list):
                                    data = {"variations": parsed}
                            except:
                                pass
            
            if not isinstance(data, dict):
                 data = {}

            if not data and "variations" not in data:
                 logger.warning(f"Failed to parse JSON from Planner.\nRaw text:\n{text}\n{'='*20}")
            
            variations = data.get("variations", [])
            return [v for v in variations if isinstance(v, str) and v not in existing]
                
        except Exception as e:
            logger.warning(f"Failed to generate variation batch: {e}")
            return []
    
    def _generate_synthetic_variations(self, original_query: str, count: int) -> List[str]:
        """Generate synthetic variations without VLM."""
        words = original_query.split()
        target = words[-1] if words else "object"
        
        templates = [
            f"the {target} in the image",
            f"something that looks like {target}",
            f"the main {target} visible",
            f"a {target} in the scene",
            f"the most prominent {target}",
            f"any {target} that stands out",
            f"the {target} in the foreground",
            f"the {target} in the center",
            f"the largest {target}",
            f"the closest {target}",
            f"find {original_query}",
            f"locate {original_query}",
            f"identify {original_query}",
            f"the area containing {target}",
            f"region with {target}",
            f"where is {target}?",
            f"show me {target}",
            f"detect {target}",
            f"segment {target}",
            f"outline {target}",
            f"the visible {target}",
            f"a specific {target}",
            f"that {target}",
            f"look for {target}",
            f"search for {target}",
            f"spot the {target}",
            f"point out {target}",
            f"highlight {target}",
            f"isolate {target}",
            f"focus on {target}",
            f"the left {target}",
            f"the right {target}",
            f"the top {target}",
            f"the bottom {target}",
            f"a small {target}",
            f"a big {target}",
            f"a dark {target}",
            f"a bright {target}",
            f"the {target} nearby",
            f"the {target} far away",
            f"an object matching '{original_query}'",
            f"the thing described as '{original_query}'",
            f"candidate for '{original_query}'",
            f"possible '{original_query}'",
            f"instance of {target}",
            f"example of {target}",
            f"rendering of {target}",
            f"view of {target}",
            f"part of {target}",
            f"whole {target}",
            f"container of {target}",
            f"surface of {target}",
            f"side of {target}",
            f"edge of {target}",
            f"corner of {target}",
            f"a red {target}", f"a blue {target}", f"a green {target}", f"a black {target}", f"a white {target}",
            f"a large {target}", f"a small {target}", f"a medium {target}", f"a tiny {target}", f"a huge {target}",
            f"the {target} on top", f"the {target} below", f"the {target} next to something",
            f"a distinct {target}", f"a clear {target}", f"a blurry {target}",
            f"the {target} in focus", f"the {target} out of focus",
            f"a horizontal {target}", f"a vertical {target}",
            f"the {target} facing left", f"the {target} facing right",
            f"the {target} facing forward", f"the {target} facing back",
            f"a {target} near the edge", f"a {target} in the middle",
            f"a {target} filling the frame", f"a {target} partially hidden",
            f"the best view of {target}", f"the worst view of {target}",
            f"a candidate {target}", f"a hypothesis for {target}",
            f"region likely containing {target}", f"area with {target}",
            f"bounding box for {target}", f"location of {target}",
            f"position of {target}", f"coordinates of {target}",
            f"the {target} you are looking for", f"the asked {target}",
            f"target: {target}", f"query match: {target}",
            f"visual match for {target}", f"semantic match for {target}",
            f"spatial match for {target}", f"contextual match for {target}",
            f"the primary {target}", f"the secondary {target}",
            f"another {target}", f"different {target}",
            f"similar to {target}", f"related to {target}",
            f"associated with {target}", f"connected to {target}",
            f"the {target} itself", f"just the {target}",
            f"only the {target}", f"all of the {target}",
             ] + [f"variant {i} of {target}" for i in range(50)] + [
             f"hypothesis {i} for {original_query}" for i in range(50)
             ]
        
        import random
        random.shuffle(templates)
        return templates[:count]
    
    def _generate_single_hypothesis(self, image_input: Union[str, Any], query: str, strategy: str, 
                                     temperature: float, w: int, h: int, img_area: int) -> Optional[Hypothesis]:
        prompt_text = self._construct_prompt(query, strategy)
        
        # Handle input type for create_vision_message
        image_path = image_input if isinstance(image_input, str) else None
        image_obj = image_input if not isinstance(image_input, str) else None
        
        messages = create_vision_message(prompt_text, image_path=image_path, image=image_obj)
        
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
                "Step 2 (Select): Choose an object that matches INDIRECTLY or METAPHORICALLY.\n"
                "Step 3 (Propose): Output the bounding box for this alternative.\n"
            )
        elif strategy == "spatial":
            instruction = (
                "Step 1 (Scan): Systematically scan DIFFERENT REGIONS - edges, corners, background, foreground.\n"
                "Step 2 (Locate): Find a matching object that might be PARTIALLY VISIBLE or in an UNEXPECTED location.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        elif strategy == "functional":
            instruction = (
                "Step 1 (Function): Think about what FUNCTION or PURPOSE the query implies.\n"
                "Step 2 (Find): Find an object that SERVES that function, even if it looks different.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        elif strategy == "visual":
            instruction = (
                "Step 1 (Attributes): Focus on VISUAL ATTRIBUTES - color, shape, texture, size.\n"
                "Step 2 (Match): Find objects with SIMILAR visual properties to what the query describes.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        elif strategy == "contextual":
            instruction = (
                "Step 1 (Scene): Understand the SCENE TYPE and CONTEXT (indoor, outdoor, kitchen, etc.).\n"
                "Step 2 (Expect): Find objects that TYPICALLY appear in this context matching the query.\n"
                "Step 3 (Propose): Output the bounding box.\n"
            )
        elif strategy == "part_whole":
            instruction = (
                "Step 1 (Decompose): Consider if the query refers to a PART of something larger.\n"
                "Step 2 (Compose): Or if the query is a CONTAINER/WHOLE that includes smaller parts.\n"
                "Step 3 (Propose): Output the bounding box for the part or whole.\n"
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
        
        # Initial scoring
        max_len = max(len(c.reasoning) for c in candidates) if candidates else 1
        
        def score_fn(h: Hypothesis):
            len_score = len(h.reasoning) / max_len if max_len > 0 else 0
            return 0.6 * h.confidence + 0.4 * len_score

        # Start with all candidates ranked by score
        remaining = sorted(candidates, key=score_fn, reverse=True)
        selected: List[Hypothesis] = []
        
        # Pass 1: Select high-diversity subset
        for cand in remaining:
            if len(selected) >= N:
                break
                
            is_diverse = True
            for existing in selected:
                iou = cand.iou(existing)
                # Stricter IoU threshold for diversity pass
                if iou > self.config.iou_threshold:
                    is_diverse = False
                    break
                
                # Use similarity check instead of exact match to allow variations
                # like "left tripod" vs "right tripod" (different position modifiers)
                if self._concepts_similar(cand.target_concept, existing.target_concept):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(cand)
                
        # Pass 2: Fill remaining slots with next best candidates (ignoring diversity if needed)
        if len(selected) < N:
            needed = N - len(selected)
            # Filter out already selected
            pool = [c for c in remaining if c not in selected]
            
            # Sort remaining by a mix of confidence and dissimilarity to current set
            def diversity_score(h):
                if not selected:
                    return 1.0
                max_iou = max((h.iou(s) for s in selected), default=0)
                return 0.5 * h.confidence + 0.5 * (1.0 - max_iou)
                
            pool_sorted = sorted(pool, key=diversity_score, reverse=True)
            selected.extend(pool_sorted[:needed])
            
        return selected


