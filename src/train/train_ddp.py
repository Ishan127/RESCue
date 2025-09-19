
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import CLIPTokenizer
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from src.models.hires_model import HiRes_Full_Model
from src.utils.losses import hungarian_loss_for_sample
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, num_epochs=1):
    setup(rank, world_size)
    # Load dataset and create distributed sampler
    grefcoco = load_dataset("qixiangbupt/grefcoco")
    train_split = grefcoco["train"]
    val_split = train_split.select(range(0, 1000))
    train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=train_sampler, collate_fn=grefcoco_collate_fn, num_workers=2)
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    model = HiRes_Full_Model(image_size=224, patch_size=16, hidden_dim=256, num_queries=10).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch+1}/{num_epochs}") if rank == 0 else train_loader
        for images, texts, gt_masks_list in loop:
            images = images.to(rank, non_blocking=True)
            text_inputs = tokenizer(texts, padding='max_length', return_tensors='pt', max_length=77)
            text_inputs = {k: v.to(rank) for k, v in text_inputs.items()}
            out = model(images, text_inputs)
            pred_masks = out["pred_masks"].to(rank)
            B, Q, H, W = pred_masks.shape
            total_loss = torch.tensor(0.0, device=rank)
            for b in range(B):
                pred_logits_q_hw = pred_masks[b].view(Q, -1)
                gt_masks = gt_masks_list[b].to(rank)
                if gt_masks.shape[0] == 0:
                    loss_b = hungarian_loss_for_sample(pred_logits_q_hw, torch.zeros((0, H*W), device=rank))
                else:
                    gt_flat = gt_masks.view(gt_masks.shape[0], -1)
                    loss_b = hungarian_loss_for_sample(pred_logits_q_hw, gt_flat)
                total_loss = total_loss + loss_b
            total_loss = total_loss / B
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            if rank == 0:
                loop.set_postfix(loss=total_loss.item())
        if rank == 0:
            print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
