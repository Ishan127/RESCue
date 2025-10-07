import torch
from src.models.hires_model import RESCUE_Model
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def denorm_image(tensor_img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_img.device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor_img.device).view(3,1,1)
    img = tensor_img * std + mean
    img = img.clamp(0,1).cpu().permute(1,2,0).numpy()
    return img

def evaluate_model(model, val_loader, device, tokenizer):
    model.eval()
    with torch.no_grad():
        for images, texts, gt_masks_list in val_loader:
            images = images.to(device)
            text_inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=77)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            out = model(images, texts, text_inputs)
            preds = out["pred_masks"]
            B, Q, H, W = preds.shape
            for b in range(min(2, B)):
                pred_logits = preds[b]
                pred_best = torch.sigmoid(pred_logits).max(dim=0).values.cpu().numpy()
                img_np = denorm_image(images[b])
                gt_mask = gt_masks_list[b]
                if isinstance(gt_mask, torch.Tensor) and gt_mask.shape[0] > 0:
                    gt_overlay = gt_mask[0].numpy()
                else:
                    gt_overlay = np.zeros((H, W))
                fig, axs = plt.subplots(1,3,figsize=(12,4))
                axs[0].imshow(img_np); axs[0].axis('off'); axs[0].set_title("Image")
                axs[1].imshow(img_np); axs[1].imshow(gt_overlay, alpha=1, cmap='Reds'); axs[1].axis('off'); axs[1].set_title("GT")
                axs[2].imshow(img_np); axs[2].imshow(pred_best, alpha=1, cmap='Blues'); axs[2].axis('off'); axs[2].set_title("Pred")
                plt.show()
            break
