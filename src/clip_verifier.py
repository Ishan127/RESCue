
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import numpy as np

class ClipVerifier:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[ClipVerifier] Loading {model_name} on {self.device}...")
        
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            print("[ClipVerifier] Model loaded successfully.")
        except Exception as e:
            print(f"[ClipVerifier] Error loading model: {e}")
            self.model = None

    def verify_batch(self, image_input, masks, query):
        """
        Verify a batch of masks against a query using CLIP/SigLIP.
        Returns a list of scores (0-100).
        """
        if self.model is None or not masks:
            return [0.0] * len(masks)

        # Prepare crops
        crops = []
        valid_indices = []
        
        # Ensure image is PIL
        if not isinstance(image_input, Image.Image):
            image_input = Image.fromarray(image_input)

        for i, mask in enumerate(masks):
            try:
                # Get bbox from mask
                mask_np = np.array(mask) > 0
                if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
                
                rows = np.any(mask_np, axis=1)
                cols = np.any(mask_np, axis=0)
                
                if not np.any(rows) or not np.any(cols):
                    continue
                    
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                
                # Expand slightly (context)
                h, w = mask_np.shape
                pad = 10
                ymin = max(0, ymin - pad)
                ymax = min(h, ymax + pad)
                xmin = max(0, xmin - pad)
                xmax = min(w, xmax + pad)
                
                crop = image_input.crop((xmin, ymin, xmax, ymax))
                crops.append(crop)
                valid_indices.append(i)
            except Exception:
                continue

        if not crops:
            return [0.0] * len(masks)

        # Encode text
        # SigLIP expects specific prompt template usually, but raw text works for simple grounding
        texts = [query] 
        
        with torch.no_grad():
            inputs = self.processor(text=texts, images=crops, padding="max_length", return_tensors="pt").to(self.device)
            
            # Get logits
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image # [N_crops, 1_text]
            probs = torch.sigmoid(logits_per_image).cpu().numpy() # SigLIP uses sigmoid, CLIP uses softmax
            
            # If standard CLIP, might need softmax. SigLIP is trained with Sigmoid Loss.
            # But logits_per_image in Hugging Face implementation might be raw. 
            # SigLIP logits are bias + scale * dot_product. 
            # Sigmoid is appropriate.
            
        scores = [0.0] * len(masks)
        for idx, prob in zip(valid_indices, probs):
            scores[idx] = float(prob[0]) * 100.0
            
        return scores
