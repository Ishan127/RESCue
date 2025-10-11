import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import the model and the necessary helper function from the model file
from src.models.hires_model import RESCUE_Model, detect_granular_cue 
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
# Import your specific loss function
from src.utils.losses import hungarian_loss_for_sample

from transformers import BertTokenizer

# ======================================================================================
# VISUALIZATION FUNCTION (Adapted for the new indices format)
# ======================================================================================
def visualize_predictions(images, texts, true_masks, pred_logits, indices, num_samples=4):
    """Shows a comparison of ground truth and best-matched predicted masks."""
    print("\n--- Visualizing Predictions ---")
    num_samples = min(num_samples, len(images))
    if num_samples == 0: return

    pred_masks_prob = pred_logits.sigmoid()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    if num_samples == 1: axes = [axes]

    for i in range(num_samples):
        # Un-normalize the image for correct display
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean; img = np.clip(img, 0, 1)

        true_mask_np = true_masks[i][0].cpu().numpy().squeeze()
        
        pred_idx, pred_mask_np = "N/A", np.zeros_like(true_mask_np)
        
        # The indices format is now (row_ind, col_ind) from scipy
        matched_pred_indices = indices[i][0]
        if len(matched_pred_indices) > 0:
            pred_idx = matched_pred_indices[0] # Get the first matched prediction index
            pred_mask_np = (pred_masks_prob[i, pred_idx] > 0.5).cpu().numpy().squeeze()

        # Plot Ground Truth
        ax = axes[i][0]; ax.imshow(img); ax.imshow(np.ma.masked_where(true_mask_np == 0, true_mask_np), cmap='cool', alpha=0.6); ax.set_title("Ground Truth"); ax.axis('off')
        # Plot Prediction
        ax = axes[i][1]; ax.imshow(img); ax.imshow(np.ma.masked_where(pred_mask_np == 0, pred_mask_np), cmap='autumn', alpha=0.6); ax.set_title(f"Prediction (Query #{pred_idx})"); ax.axis('off')
        fig.text(0.5, 0.95 - i/num_samples, f'Prompt: "{texts[i]}"', ha='center', fontsize=12, wrap=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

# ======================================================================================
# MODIFIED TRAINING LOOP
# ======================================================================================
def train_model(model, train_loader, optimizer, device, num_epochs=1):
    """
    Main training loop for the RESCUE model, using the per-sample Hungarian loss.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        last_batch_for_viz = None

        for images, texts, gt_masks_list in loop:
            images = images.to(device)
            
            # Text processing
            text_inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            run_stage3_mask = detect_granular_cue(texts).to(device)
            
            # Forward pass
            try:
                out = model(images, text_inputs['input_ids'], text_inputs['attention_mask'], run_stage3_mask)
            except Exception as e:
                print("Error in model forward:", e); raise
                
            pred_masks = out["pred_masks"]
            B, Q, H, W = pred_masks.shape
            
            # --- FIX APPLIED HERE: Per-sample loss calculation ---
            total_loss = torch.tensor(0.0, device=device)
            batch_indices = [] # To store indices for the whole batch
            
            # 1. Iterate through each sample in the batch
            for b in range(B):
                pred_logits_q_hw = pred_masks[b].view(Q, -1)
                gt_masks = gt_masks_list[b].to(device)
                
                try:
                    gt_flat = gt_masks.view(gt_masks.shape[0], -1)
                    # 2. Call the per-sample loss function
                    loss_dict = hungarian_loss_for_sample(pred_logits_q_hw, gt_flat)
                except Exception as e:
                    print(f"Error in hungarian_loss_for_sample for sample {b}: {e}"); raise
                
                # 3. Accumulate the loss and store the indices
                total_loss += loss_dict['loss']
                batch_indices.append(loss_dict['indices'])
                
            total_loss = total_loss / B # Average loss over the batch
            
            # --- Standard backward pass ---
            optimizer.zero_grad()
            try:
                total_loss.backward()
            except Exception as e:
                print("Error in backward:", e); raise
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
            # Store data for visualization, now including the correct indices
            last_batch_for_viz = (images, texts, gt_masks_list, pred_masks.detach(), batch_indices)
            
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

        if last_batch_for_viz:
            visualize_predictions(*last_batch_for_viz)

# Your main execution block...

