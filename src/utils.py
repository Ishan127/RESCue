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
    
    print("Overlay - Image shape:", image.shape, "Mask shape:", mask.shape)
    
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[2] == 3:
            mask = mask[:, :, 0]
        elif mask.shape[0] == 1:
            mask = mask[0]

    mask = mask > 0
    
    if image.shape[:2] != mask.shape[:2]:
        print(f"Warning: Image {image.shape} and Mask {mask.shape} mismatch. Resizing mask.")
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    overlay = image.copy()
    
    overlay[:, :, 0] = np.where(mask, 
                               (1 - alpha) * image[:, :, 0] + alpha * 255, 
                               image[:, :, 0]).astype(np.uint8)
    
    overlay[:, :, 1] = np.where(mask, 
                               (1 - alpha) * image[:, :, 1], 
                               image[:, :, 1]).astype(np.uint8)
    
    overlay[:, :, 2] = np.where(mask, 
                               (1 - alpha) * image[:, :, 2], 
                               image[:, :, 2]).astype(np.uint8)
                               
    return Image.fromarray(overlay)

def calculate_iou(mask1, mask2):
    mask1 = np.asarray(mask1)
    mask2 = np.asarray(mask2)
    
    if mask1.ndim == 3:
        if mask1.shape[2] == 1:
            mask1 = mask1[:, :, 0]
        elif mask1.shape[0] == 1:
            mask1 = mask1[0]
        else:
            mask1 = mask1[:, :, 0]
    
    if mask2.ndim == 3:
        if mask2.shape[2] == 1:
            mask2 = mask2[:, :, 0]
        elif mask2.shape[0] == 1:
            mask2 = mask2[0]
        else:
            mask2 = mask2[:, :, 0]
    
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    if mask1.shape != mask2.shape:
        print(f"Resizing mask for IoU: {mask1.shape} -> {mask2.shape}")
        mask1 = cv2.resize(mask1.astype(np.uint8), (mask2.shape[1], mask2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
        
    return intersection / union
