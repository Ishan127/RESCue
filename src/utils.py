import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import os
import requests
from io import BytesIO

def get_device():
    if torch.cuda.is_available():
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"Device: ROCm (HIP) detected. Using 'cuda' device alias mapping to {torch.cuda.get_device_name(0)}.")
            return "cuda" 
        else:
            print(f"Device: CUDA detected. Using {torch.cuda.get_device_name(0)}.")
            return "cuda"
    return "cpu"

def load_image(image_path):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        try:
            response = requests.get(image_path)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            raise
    return Image.open(image_path).convert("RGB")

def apply_red_alpha_overlay(image, mask, alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    mask = mask > 0
    overlay = image.copy()
    
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    overlay[:, :, 0] = np.where(mask, 
                               (1 - alpha) * red_channel + alpha * 255, 
                               red_channel).astype(np.uint8)
    
    overlay[:, :, 1] = np.where(mask, 
                               (1 - alpha) * green_channel + alpha * 0, 
                               green_channel).astype(np.uint8)
    
    overlay[:, :, 2] = np.where(mask, 
                               (1 - alpha) * blue_channel + alpha * 0, 
                               blue_channel).astype(np.uint8)
                               
    return Image.fromarray(overlay)

def plot_results(image, boxes, masks, scores, output_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    for i, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"Score: {score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
    
    plt.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_mask(mask, output_path):
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask.save(output_path)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
        
    return intersection / union
