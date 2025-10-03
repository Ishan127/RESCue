import argparse
import torch
from src.models.hires_model import RESCUE_Model
from src.train.train import train_model
from src.eval.evaluate import evaluate_model
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer

grefcoco = load_dataset("qixiangbupt/grefcoco")
train_split = grefcoco["train"]
val_split = grefcoco["train"]
train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
val_ds = GRefCocoTorchDataset(val_split, image_size=224, train=False)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=grefcoco_collate_fn, num_workers=2)

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate HiRes model.")
    parser.add_argument('--mode', choices=['train', 'eval'], required=True, help='Mode: train or eval')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RESCUE_Model(image_size=224, hidden_dim=256, num_object_queries=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    if args.mode == 'train':
        from torch.optim import AdamW
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
        train_model(model, train_loader, optimizer, device, num_epochs=1, tokenizer=tokenizer)
    elif args.mode == 'eval':
        evaluate_model(model, val_loader, device, tokenizer)

if __name__ == "__main__":
    main()
