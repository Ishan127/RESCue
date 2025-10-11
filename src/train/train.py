import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Import the model and the necessary helper function
from src.models.hires_model import RESCUE_Model, detect_granular_cue
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from src.utils.losses import hungarian_loss_for_sample

# Import the tokenizer
from transformers import BertTokenizer

# Placeholder for your actual dataset loading
# from datasets import load_dataset
# grefcoco = load_dataset("qixiangbupt/grefcoco")
# train_split = grefcoco["train"]
# train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)
train_loader = None  # TODO: Replace with actual DataLoader

def visualize_predictions(original_images, texts, true_masks, pred_logits, indices, num_samples=4):
    """
    Shows a comparison of ground truth and best-matched predicted masks.
    MODIFIED to accept original PIL images for cleaner visualization.
    """
    print("\n--- Visualizing Predictions ---")
    num_samples = min(num_samples, len(original_images))
    if num_samples == 0: return

    pred_masks_prob = pred_logits.sigmoid()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    if num_samples == 1: axes = [axes]

    for i in range(num_samples):
        # Use the original, un-normalized image
        img = original_images[i] 
        
        true_mask_np = true_masks[i][0].cpu().numpy().squeeze()
        
        pred_idx, pred_mask_np = "N/A", np.zeros_like(true_mask_np)
        
        if indices and i < len(indices) and len(indices[i][0]) > 0:
            pred_idx = indices[i][0][0].item()
            pred_mask_np = (pred_masks_prob[i, pred_idx] > 0.5).cpu().numpy().squeeze()

        # Plot Ground Truth
        ax = axes[i][0]
        ax.imshow(img)
        ax.imshow(np.ma.masked_where(true_mask_np == 0, true_mask_np), cmap='cool', alpha=0.6)
        ax.set_title("Ground Truth")
        ax.axis('off')

        # Plot Prediction
        ax = axes[i][1]
        ax.imshow(img)
        ax.imshow(np.ma.masked_where(pred_mask_np == 0, pred_mask_np), cmap='autumn', alpha=0.6)
        ax.set_title(f"Prediction (Query #{pred_idx})")
        ax.axis('off')
        
        fig.text(0.5, 0.95 - (i / num_samples) * (0.95/num_samples if num_samples > 1 else 0.95) , f'Prompt: "{texts[i]}"', ha='center', fontsize=12, wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def train_model(model, train_loader, optimizer, device, num_epochs=1):
    """
    Main training loop for the RESCUE model, with periodic visualization.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        # <<< CHANGE APPLIED HERE: Step counter for periodic visualization >>>
        step_counter = 0

        # <<< CHANGE APPLIED HERE: The loop now unpacks original_images >>>
        for images, texts, gt_masks_list, original_images in loop:
            step_counter += 1
            images = images.to(device)
            
            # --- Text processing ---
            text_inputs = tokenizer(
                texts, padding='max_length', return_tensors='pt',
                max_length=512, truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            run_stage3_mask = detect_granular_cue(texts).to(device)
            
            # --- Forward pass ---
            try:
                out = model(images, text_inputs['input_ids'], text_inputs['attention_mask'], run_stage3_mask)
            except Exception as e:
                print("Error in model forward:", e); raise
                
            pred_masks = out["pred_masks"]
            
            # --- Loss Calculation ---
            B, Q, H, W = pred_masks.shape
            total_loss = torch.tensor(0.0, device=device)
            
            # <<< CHANGE APPLIED HERE: Collect indices for the batch >>>
            batch_indices = []
            
            for b in range(B):
                pred_logits_q_hw = pred_masks[b].view(Q, -1)
                gt_masks = gt_masks_list[b].to(device)
                
                try:
                    gt_flat = gt_masks.view(gt_masks.shape[0], -1)
                    loss_dict = hungarian_loss_for_sample(pred_logits_q_hw, gt_flat)
                except Exception as e:
                    print(f"Error in hungarian_loss_for_sample for sample {b}: {e}"); raise
                    
                total_loss += loss_dict['loss']
                batch_indices.append(loss_dict['indices'])
            
            total_loss = total_loss / B
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            try:
                total_loss.backward()
            except Exception as e:
                print("Error in backward:", e); raise
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

            # <<< CHANGE APPLIED HERE: Periodic Visualization >>>
            # Visualize every 100 steps, or on the very last step of the epoch
            if step_counter % 100 == 0 or step_counter == len(train_loader):
                # We need to switch to eval mode for visualization to disable things like dropout
                model.eval()
                with torch.no_grad():
                    # It's better to re-run inference on the same batch for a clean visualization
                    viz_out = model(images, text_inputs['input_ids'], text_inputs['attention_mask'], run_stage3_mask)
                    visualize_predictions(original_images, texts, gt_masks_list, viz_out['pred_masks'], batch_indices)
                # Switch back to train mode
                model.train()
            
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

# Your main execution block...

