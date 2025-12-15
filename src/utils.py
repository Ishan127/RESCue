import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

def load_image(image_path):
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
