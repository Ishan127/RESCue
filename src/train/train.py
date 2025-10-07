import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Import the model and the necessary helper function
from src.models.hires_model import RESCUE_Model, detect_granular_cue
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from src.utils.losses import hungarian_loss_for_sample

# Import the tokenizer, which is now needed in the training script
from transformers import CLIPTokenizer

# Placeholder for your actual dataset loading
# from datasets import load_dataset
# grefcoco = load_dataset("qixiangbupt/grefcoco")
# train_split = grefcoco["train"]
# train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)
train_loader = None  # TODO: Replace with actual DataLoader

def train_model(model, train_loader, optimizer, device, num_epochs=1):
    """
    Main training loop for the RESCUE model.
    """
    # The tokenizer is now managed by the training script, not passed in.
    # Ensure this model name matches the one used inside your Stage1_FusionModule.
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        for images, texts, gt_masks_list in loop:
            images = images.to(device)
            
            # --- FIX APPLIED HERE ---
            # 1. Tokenize the raw text in the training loop before calling the model.
            text_inputs = tokenizer(
                texts, 
                padding='max_length', 
                return_tensors='pt',
                max_length=77, # Max length for CLIP's text encoder
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
            
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

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
