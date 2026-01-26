
import io
import base64
import requests
import numpy as np
import os
from PIL import Image

class ClipVerifier:
    def __init__(self, server_url=None):
        # Default to localhost:8003 if not set
        self.server_url = server_url or os.environ.get("CLIP_SERVER_URL", "http://localhost:8003/verify")
        print(f"[ClipVerifier] Initialized Client pointing to {self.server_url}")

    def verify_batch(self, image_input, masks, query):
        """
        Verify a batch of masks against a query using remote CLIP/SigLIP Server.
        Returns a list of scores (0-100).
        """
        if not masks:
            return []

        # Prepare crops locally to minimize data transfer
        crops_b64 = []
        valid_indices = []
        
        # Work with numpy for efficient masking
        if isinstance(image_input, Image.Image):
            image_np = np.array(image_input)
        else:
            image_np = np.array(image_input)

        try:
        from concurrent.futures import ThreadPoolExecutor
        
        def process_crop(idx, mask):
            try:
                # Ensure mask is boolean numpy
                mask_np = np.array(mask) > 0
                if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
                
                rows = np.any(mask_np, axis=1)
                cols = np.any(mask_np, axis=0)
                
                if not np.any(rows) or not np.any(cols):
                    return None
                    
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                
                # Expand slightly (context)
                h, w = mask_np.shape
                pad = 10
                ymin = max(0, ymin - pad)
                ymax = min(h, ymax + pad)
                xmin = max(0, xmin - pad)
                xmax = min(w, xmax + pad)
                
                # Get crops
                img_crop = image_np[ymin:ymax, xmin:xmax]
                mask_crop = mask_np[ymin:ymax, xmin:xmax]
                
                # Apply mask to image (Black out background)
                # Create a black background canvas
                masked_crop = np.zeros_like(img_crop)
                
                # Handle shape mismatch if any
                if mask_crop.shape != img_crop.shape[:2]:
                    # Resize mask crop to match img crop (nearest neighbor)
                    import cv2
                    mask_crop = cv2.resize(mask_crop.astype(np.uint8), (img_crop.shape[1], img_crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

                masked_crop[mask_crop] = img_crop[mask_crop]
                
                # Convert to PIL
                crop_pil = Image.fromarray(masked_crop)
                
                # In-memory save to base64
                buf = io.BytesIO()
                crop_pil.convert("RGB").save(buf, format="JPEG", quality=90)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                return idx, b64
            except Exception as e:
                print(f"Crop Error {idx}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=min(32, len(masks))) as executor:
            futures = [executor.submit(process_crop, i, m) for i, m in enumerate(masks)]
            for f in futures:
                res = f.result()
                if res:
                    valid_indices.append(res[0])
                    crops_b64.append(res[1])
                
            if not crops_b64:
                return [0.0] * len(masks)
                
            # Send to Server
            payload = {
                "crops": crops_b64,
                "query": query
            }
            
            response = requests.post(self.server_url, json=payload, timeout=30)
            if response.status_code == 200:
                server_scores = response.json()['scores']
                
                # Map back to original indices
                final_scores = [0.0] * len(masks)
                for valid_idx, score in zip(valid_indices, server_scores):
                    final_scores[valid_idx] = score
                return final_scores
            else:
                print(f"[ClipVerifier] Server Error: {response.text}")
                return [0.0] * len(masks)
                
        except Exception as e:
            print(f"[ClipVerifier] Client Error: {e}")
            return [0.0] * len(masks)
