import argparse
import torch
from src.models.hires_model import HiRes_Full_Model
from src.train.train import train_model
from src.eval.evaluate import evaluate_model
from src.data.datasets import GRefCocoTorchDataset, grefcoco_collate_fn
from torch.utils.data import DataLoader
from datasets import load_dataset

grefcoco = load_dataset("qixiangbupt/grefcoco")
train_split = grefcoco["train"]
val_split = grefcoco["train"]
train_ds = GRefCocoTorchDataset(train_split, image_size=224, train=True)
val_ds = GRefCocoTorchDataset(val_split, image_size=224, train=False)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=grefcoco_collate_fn, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=grefcoco_collate_fn, num_workers=2)

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate HiRes model.")
    parser.add_argument('--mode', choices=['train', 'eval'], required=True, help='Mode: train or eval')
    parser.add_argument('--viz-every', type=int, default=50, help='Show matplotlib previews every N steps during training')
    parser.add_argument('--viz-mode', choices=['save','display','off'], default='save', help='Visualization mode for previews when running as a script')
    parser.add_argument('--viz-train-dir', type=str, default='viz_train', help='Directory to save training previews (if viz-mode=save)')
    parser.add_argument('--viz-eval-dir', type=str, default='viz_eval', help='Directory to save evaluation previews (if viz-mode=save)')
    parser.add_argument('--eval-vis-batches', type=int, default=1, help='Number of batches to visualize during evaluation')
    parser.add_argument('--progress', choices=['auto','tqdm','plain'], default='auto', help='Progress display mode: auto (tqdm if TTY), tqdm, or plain (print lines)')
    parser.add_argument('--log-every', type=int, default=50, help='When progress=plain, print metrics every N steps')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiRes_Full_Model(image_size=224, patch_size=16, hidden_dim=256, num_queries=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)
    if args.mode == 'train':
        from torch.optim import AdamW
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
        train_model(
            model, train_loader, optimizer, device,
            num_epochs=4, viz_every=args.viz_every,
            viz_mode=args.viz_mode, viz_dir=args.viz_train_dir,
            progress_mode=args.progress, log_every=args.log_every
        )
    elif args.mode == 'eval':
        evaluate_model(
            model, val_loader, device,
            max_vis_batches=args.eval_vis_batches,
            viz_mode=args.viz_mode, viz_dir=args.viz_eval_dir
        )

if __name__ == "__main__":
    main()
