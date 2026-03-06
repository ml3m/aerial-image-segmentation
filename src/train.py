"""
train.py — Training loop for U-Net aerial image segmentation.

Usage examples:
    # Basic run with defaults
    python -m src.train

    # Custom hyperparams
    python -m src.train --epochs 30 --batch-size 8 --lr 5e-5

    # Resume from checkpoint
    python -m src.train --resume results/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

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
from src.dataset import get_dataloaders
from src.model import UNet
from src.utils import get_device, save_checkpoint, load_checkpoint, set_seed


def _wrap_tqdm(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable

def _get_amp_context(device: torch.device, enabled: bool):
    """Return autocast context. Uses cuda.amp for ROCm/CUDA compatibility."""
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    if device.type == "mps":
        try:
            return torch.amp.autocast(device_type="mps", enabled=True)
        except (AttributeError, RuntimeError):
            return nullcontext()
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    amp_ctx = _get_amp_context(device, use_amp)

    for images, masks in _wrap_tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp_ctx:
            logits = model(images)               # (B, C, H, W)
            loss   = criterion(logits, masks)    # masks: (B, H, W) int64

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    use_amp: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    amp_ctx = _get_amp_context(device, use_amp)

    for images, masks in _wrap_tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = get_device()

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except RuntimeError as e:
            if "Dynamo" in str(e) or "3.12" in str(e):
                print(f"[train] torch.compile skipped (not supported on Python 3.12+): {e}")
                args.compile = False
            else:
                raise

    # Loss: CrossEntropy handles multi-class pixel classification
    criterion = nn.CrossEntropyLoss()

    use_amp = args.amp and device.type in ("cuda", "mps")
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    # Optional: resume from checkpoint
    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", best_val_loss)
        print(f"[train] Resuming from epoch {start_epoch}")

    speedups = []
    if use_amp:
        speedups.append("AMP")
    if args.compile:
        speedups.append("compile")
    speedup_str = f"  [{', '.join(speedups)}]" if speedups else ""

    print(f"\n{'='*55}")
    print(f"  U-Net training  |  {args.epochs} epochs  |  device: {device}{speedup_str}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss   = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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
                CHECKPOINTS_DIR / "best.pth",
            )

    save_checkpoint(
        model, optimizer, args.epochs, val_loss, CHECKPOINTS_DIR / "last.pth"
    )
    print("\n[train] Done. Best val loss:", round(best_val_loss, 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for aerial segmentation")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS,    help="Number of epochs")
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,    help="Batch size")
    p.add_argument("--lr",          type=float, default=LEARNING_RATE, help="Learning rate")
    p.add_argument("--num-workers", type=int,   default=4,             help="DataLoader workers")
    p.add_argument("--resume",      type=str,   default=None,          help="Path to checkpoint to resume")
    p.add_argument("--amp",         action="store_true",               help="Use mixed precision (FP16) for faster training")
    p.add_argument("--compile",     action="store_true",               help="Use torch.compile for faster training")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
