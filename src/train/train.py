import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from src.models.hires_model import HiRes_Full_Model
from src.utils.losses import hungarian_loss_for_sample
from src.utils.metrics import batch_miou_ap_from_logits
from src.utils.visualize import show_or_save
import matplotlib.pyplot as plt
import sys

# Placeholder for loading your dataset
# from datasets import load_dataset
# grefcoco = load_dataset("qixiangbupt/grefcoco")
# train_split = grefcoco["train"]
# train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)

# Replace the above with your actual dataset loading
train_loader = None  # TODO: Replace with actual DataLoader

def _denorm_image(t):
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return x


def _show_train_preview(images, pred_logits_b_q_h_w, gt_masks_list, step_idx, max_show=2, viz_mode="save", viz_dir="viz_train"):
    probs = torch.sigmoid(pred_logits_b_q_h_w)
    best = probs.max(dim=1).values  # [B,H,W]
    B = best.shape[0]
    show_n = min(max_show, B)
    for b in range(show_n):
        img_np = _denorm_image(images[b])
        pred_np = best[b].detach().cpu().numpy()
        gt = gt_masks_list[b]
        if isinstance(gt, torch.Tensor) and gt.ndim == 3 and gt.shape[0] > 0:
            gt_np = gt.max(dim=0).values.cpu().numpy()
        else:
            gt_np = torch.zeros_like(best[b]).cpu().numpy()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np); axs[0].axis('off'); axs[0].set_title(f"Step {step_idx} Image")
        axs[1].imshow(img_np); axs[1].imshow(gt_np, alpha=0.6, cmap='Reds'); axs[1].axis('off'); axs[1].set_title("GT")
        axs[2].imshow(img_np); axs[2].imshow(pred_np, alpha=0.6, cmap='Blues'); axs[2].axis('off'); axs[2].set_title("Pred")
        fname = f"step_{step_idx:06d}_b{b}.png"
        show_or_save(fig, viz_dir, fname, mode=viz_mode)


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    num_epochs=1,
    viz_every=50,
    viz_mode="save",
    viz_dir="viz_train",
    progress_mode: str = "auto",  # 'auto'|'tqdm'|'plain'
    log_every: int = 50,
):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        use_tqdm = (progress_mode == 'tqdm') or (progress_mode == 'auto' and sys.stdout.isatty())
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True) if use_tqdm else train_loader
        epoch_loss = 0.0
        miou_running = 0.0
        ap_running = 0.0
        ap_count = 0
        for step_idx, (images, texts, gt_masks_list) in enumerate(loop if use_tqdm else enumerate(loop, start=1), start=1 if use_tqdm else 0):
            if not use_tqdm:
                # when not using tqdm, ensure we have correct step_idx and batch
                if isinstance(images, int):
                    # adjust unpack from enumerate(loop, start=1)
                    step_idx, batch = images, texts
                    images, texts, gt_masks_list = batch
            images = images.to(device)
            text_inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=77, truncation=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            out = model(images, text_inputs)
            pred_masks = out["pred_masks"].to(device)
            B, Q, H, W = pred_masks.shape
            # Loss via Hungarian
            total_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                pred_logits_q_hw = pred_masks[b].view(Q, -1)
                gt_masks = gt_masks_list[b].to(device)
                if gt_masks.shape[0] == 0:
                    loss_b = hungarian_loss_for_sample(pred_logits_q_hw, torch.zeros((0, H * W), device=device))
                else:
                    gt_flat = gt_masks.view(gt_masks.shape[0], -1)
                    loss_b = hungarian_loss_for_sample(pred_logits_q_hw, gt_flat)
                total_loss = total_loss + loss_b
            total_loss = total_loss / B
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Metrics (mIoU, AP) on the fly
            miou_b, ap_b, ap_valid, _ = batch_miou_ap_from_logits(pred_masks.detach(), gt_masks_list, threshold=0.5)
            miou_running += miou_b
            if not (ap_b != ap_b):  # check for NaN
                ap_running += ap_b
                ap_count += 1

            epoch_loss += total_loss.item()
            global_step += 1
            # Visualize every viz_every steps for Kaggle notebook
            if viz_every and (global_step % viz_every == 0):
                _show_train_preview(
                    images.detach(), pred_masks.detach(), gt_masks_list,
                    global_step, viz_mode=viz_mode, viz_dir=viz_dir
                )

            if use_tqdm:
                loop.set_postfix(loss=total_loss.item(), mIoU=f"{miou_b:.3f}", AP=f"{ap_b:.3f}" if ap_b == ap_b else 'nan')
            else:
                if (step_idx % max(1, log_every)) == 0:
                    ap_str = f"{ap_b:.3f}" if ap_b == ap_b else 'nan'
                    print(f"Epoch {epoch+1}/{num_epochs} Step {step_idx}/{len(train_loader)} | loss={total_loss.item():.3f} mIoU={miou_b:.3f} AP={ap_str}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        avg_miou = miou_running / max(1, len(train_loader))
        avg_ap = (ap_running / ap_count) if ap_count > 0 else float('nan')
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f} | avg mIoU: {avg_miou:.4f} | avg AP: {avg_ap:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiRes_Full_Model(image_size=224, patch_size=16, hidden_dim=256, num_queries=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
    # train_model(model, train_loader, optimizer, device, num_epochs=1)  # Uncomment when train_loader is ready
