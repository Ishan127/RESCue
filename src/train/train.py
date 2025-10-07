import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from src.models.hires_model import RESCUE_Model
from src.utils.losses import hungarian_loss_for_sample

# Placeholder for loading your dataset
# from datasets import load_dataset
# grefcoco = load_dataset("qixiangbupt/grefcoco")
# train_split = grefcoco["train"]
# train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)

# Replace the above with your actual dataset loading
train_loader = None  # TODO: Replace with actual DataLoader

def train_model(model, train_loader, optimizer, device, num_epochs=1, tokenizer=None):
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for images, texts, gt_masks_list in loop:
            print("Batch start")
            print(f"images.shape: {images.shape}")
            print(f"texts: {texts}")
            print(f"gt_masks_list: {[gt.shape if isinstance(gt, torch.Tensor) else None for gt in gt_masks_list]}")
            images = images.to(device)
            text_inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=77)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            print(f"text_inputs shapes: {{k: v.shape for k, v in text_inputs.items()}}")
            try:
                out = model(images, texts, text_inputs)
            except Exception as e:
                print("Error in model forward:", e)
                raise
            pred_masks = out["pred_masks"].to(device)
            print(f"pred_masks.shape: {pred_masks.shape}")
            B, Q, H, W = pred_masks.shape
            total_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                print(f"Sample {b}: pred_masks[{b}].shape: {pred_masks[b].shape}")
                pred_logits_q_hw = pred_masks[b].view(Q, -1)
                gt_masks = gt_masks_list[b].to(device)
                print(f"Sample {b}: gt_masks.shape: {gt_masks.shape}")
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
            print("Batch end\n")
        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

"""if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RESCUE_Model(image_size=224, patch_size=16, hidden_dim=256, num_queries=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
    # train_model(model, train_loader, optimizer, device, num_epochs=1)  # Uncomment when train_loader is ready"""
