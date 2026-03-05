"""
train.py — Training loop for U-Net aerial image segmentation.

Usage examples:
    # Basic run with defaults
    python -m src.train

    # Custom hyperparams
    python -m src.train --epochs 30 --batch-size 8 --lr 5e-5

    # Resume from checkpoint
    python -m src.train --resume results/checkpoints/best.pth

    # 5-fold cross-validation
    python -m src.train --n-folds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # graceful fallback

from src.config import (
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    CHECKPOINTS_DIR,
    SEED,
)
from src.dataset import get_dataloaders, get_dataloaders_kfold
from src.model import UNet
from src.utils import get_device, save_checkpoint, load_checkpoint, set_seed


def _wrap_tqdm(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in _wrap_tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)               # (B, C, H, W)
        loss   = criterion(logits, masks)    # masks: (B, H, W) int64
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for images, masks in _wrap_tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def run_training(
    args: argparse.Namespace,
    train_loader,
    val_loader,
    model: nn.Module,
    checkpoint_path: Path,
) -> float:
    """Run one training loop. Returns best validation loss."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 1
    best_val_loss = float("inf")
    last_val_loss = 0.0

    if args.resume and checkpoint_path == CHECKPOINTS_DIR / "best.pth":
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", best_val_loss)
            print(f"[train] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        last_val_loss = val_loss

        tag = "✓ best" if val_loss < best_val_loss else ""
        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  {tag}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                checkpoint_path,
            )

    last_path = checkpoint_path.parent / "last.pth"
    if checkpoint_path.name.startswith("best_fold_"):
        last_path = checkpoint_path.parent / f"last_fold_{checkpoint_path.stem.split('_')[-1]}.pth"
    save_checkpoint(
        model, optimizer, args.epochs, last_val_loss,
        last_path,
    )
    return best_val_loss


def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = get_device()

    if args.n_folds is not None:
        # K-fold cross-validation
        fold_losses = []
        for fold in range(args.n_folds):
            print(f"\n{'='*55}")
            print(f"  Fold {fold + 1}/{args.n_folds}")
            print(f"{'='*55}")

            train_loader, val_loader = get_dataloaders_kfold(
                fold=fold,
                n_splits=args.n_folds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
            ckpt_path = CHECKPOINTS_DIR / f"best_fold_{fold + 1}.pth"

            best_loss = run_training(
                args, train_loader, val_loader, model, ckpt_path
            )
            fold_losses.append(best_loss)
            print(f"[train] Fold {fold + 1} best val loss: {best_loss:.4f}")

        mean_loss = float(np.mean(fold_losses))
        std_loss = float(np.std(fold_losses))
        print(f"\n[train] K-fold done. Mean val loss: {mean_loss:.4f} ± {std_loss:.4f}")
        return

    # Standard single split
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    print(f"\n{'='*55}")
    print(f"  U-Net training  |  {args.epochs} epochs  |  device: {device}")
    print(f"{'='*55}\n")

    best_val_loss = run_training(
        args, train_loader, val_loader, model, CHECKPOINTS_DIR / "best.pth"
    )
    print("\n[train] Done. Best val loss:", round(best_val_loss, 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for aerial segmentation")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS,    help="Number of epochs")
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,    help="Batch size")
    p.add_argument("--lr",          type=float, default=LEARNING_RATE, help="Learning rate")
    p.add_argument("--num-workers", type=int,   default=4,             help="DataLoader workers")
    p.add_argument("--resume",      type=str,   default=None,          help="Path to checkpoint to resume")
    p.add_argument("--n-folds",     type=int,   default=None,          help="Enable K-fold CV (e.g. 5 for 5-fold)")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
