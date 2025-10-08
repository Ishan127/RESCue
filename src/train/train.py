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

# Import the tokenizer, which is now needed in the training script
from transformers import CLIPTokenizer
from transformers import BertTokenizer

# Placeholder for your actual dataset loading
# from datasets import load_dataset
# grefcoco = load_dataset("qixiangbupt/grefcoco")
# train_split = grefcoco["train"]
# train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)
train_loader = None  # TODO: Replace with actual DataLoader

def visualize_predictions(images, texts, true_masks, pred_logits, indices, num_samples=4):
    """Shows a comparison of ground truth and best-matched predicted masks."""
    print("\n--- Visualizing Predictions ---")
    num_samples = min(num_samples, len(images))
    if num_samples == 0: return

    pred_masks_prob = pred_logits.sigmoid()

    # Create a figure to display the results
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    if num_samples == 1: axes = [axes]

    for i in range(num_samples):
        # We need to un-normalize the image for correct display
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        true_mask_np = true_masks[i][0].cpu().numpy().squeeze()
        
        pred_idx, pred_mask_np = "N/A", np.zeros_like(true_mask_np)
        
        # Check if the Hungarian algorithm found any matches for this sample
        if len(indices[i][0]) > 0:
            # Get the index of the best matching prediction
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
        
        # Add the text prompt as a title for the row
        fig.text(0.5, 0.95 - i/num_samples, f'Prompt: "{texts[i]}"', ha='center', fontsize=12, wrap=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def train_model(model, train_loader, optimizer, device, num_epochs=1):
    """
    Main training loop for the RESCUE model.
    """
    # The tokenizer is now managed by the training script, not passed in.
    # Ensure this model name matches the one used inside your Stage1_FusionModule.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0

        last_batch_for_viz = None

        for images, texts, gt_masks_list in loop:
            images = images.to(device)
            
            # --- FIX APPLIED HERE ---
            # 1. Tokenize the raw text in the training loop before calling the model.
            text_inputs = tokenizer(
                texts, 
                padding='max_length', 
                return_tensors='pt',
                #max_length=77, # Max length for CLIP's text encoder
                truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # 2. Calculate the conditional mask in the training loop.
            run_stage3_mask = detect_granular_cue(texts).to(device)
            # This tensor can be correctly scattered by DataParallel.
            
            # 3. Call the model with the corrected, tensor-only inputs.
            # The raw 'texts' list is no longer passed to the model's forward pass.
            try:
                out = model(
                    images=images,
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    run_stage3_mask=run_stage3_mask
                )

            except Exception as e:
                print("Error in model forward:", e)
                raise
                
            pred_masks = out["pred_masks"].to(device)
            indices = out.get("indices")
            B, Q, H, W = pred_masks.shape
            total_loss = torch.tensor(0.0, device=device)
            
            for b in range(B):
                pred_logits_q_hw = pred_masks[b].view(Q, -1)
                gt_masks = gt_masks_list[b].to(device)
                
                try:
                    if gt_masks.shape[0] == 0:
                        loss_b = hungarian_loss_for_sample(pred_logits_q_hw, torch.zeros((0, H*W), device=device))
                    else:
                        gt_flat = gt_masks.view(gt_masks.shape[0], -1)
                        loss_b = hungarian_loss_for_sample(pred_logits_q_hw, gt_flat)
                except Exception as e:
                    print(f"Error in hungarian_loss_for_sample for sample {b}: {e}")
                    raise
                    
                total_loss = total_loss + loss_b
                
            total_loss = total_loss / B
            optimizer.zero_grad()
            
            try:
                total_loss.backward()
            except Exception as e:
                print("Error in backward:", e)
                raise
                
            optimizer.step()
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

            last_batch_for_viz = (images, texts, gt_masks_list, pred_masks.detach(), indices)
            
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

        if last_batch_for_viz:
            visualize_predictions(*last_batch_for_viz)

# Your main execution block remains the same. The tokenizer is no longer passed to train_model.
"""if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RESCUE_Model(image_size=224, patch_size=16, hidden_dim=256, num_queries=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
    # train_model(model, train_loader, optimizer, device, num_epochs=1) # Uncomment when train_loader is ready"""
