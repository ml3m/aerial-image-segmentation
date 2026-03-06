"""
train.py — Training loop for U-Net aerial image segmentation.
Optimised for AMD RX 6700S (ROCm) on Arch Linux / Python 3.12.

Usage examples:
    # Recommended fast run
    python -m src.train --amp --batch-size 16 --num-workers 6

    # With torch.compile (PyTorch 2.3+)
    python -m src.train --amp --compile --batch-size 16 --num-workers 6

    # Resume from checkpoint
    python -m src.train --amp --resume results/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import time
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_tqdm(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _get_amp_context(device: torch.device, enabled: bool):
    """
    Return the correct autocast context for the device.

    On ROCm/HIP, PyTorch exposes AMD GPUs as device type 'cuda', so the
    standard torch.amp.autocast('cuda') path is used.  The device-agnostic
    torch.amp.autocast API (PyTorch ≥ 2.1) is preferred over the deprecated
    torch.cuda.amp.autocast.
    """
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        # Works for both CUDA and ROCm (HIP) backends
        try:
            return torch.amp.autocast(device_type="cuda")
        except AttributeError:
            # Fallback for PyTorch < 2.1
            return torch.cuda.amp.autocast()
    if device.type == "mps":
        try:
            return torch.amp.autocast(device_type="mps", enabled=True)
        except (AttributeError, RuntimeError):
            return nullcontext()
    return nullcontext()


def _make_scaler(use_amp: bool, device: torch.device):
    """
    Return a GradScaler when AMP is active on a CUDA/ROCm device.

    Uses the device-agnostic torch.amp.GradScaler (PyTorch ≥ 2.3) and falls
    back to the legacy torch.cuda.amp.GradScaler for older versions.
    """
    if not (use_amp and device.type == "cuda"):
        return None
    try:
        # Preferred: device-agnostic API (PyTorch ≥ 2.3)
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        # PyTorch < 2.3 — torch.amp.GradScaler doesn't exist yet
        return torch.cuda.amp.GradScaler()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

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
        # non_blocking=True overlaps H→D transfer with GPU compute
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = get_device()

    # ------------------------------------------------------------------
    # Sanity-check: confirm ROCm/HIP is active when on AMD hardware
    # ------------------------------------------------------------------
    if device.type == "cuda":
        hip_ver = getattr(torch.version, "hip", None)
        backend = f"ROCm/HIP {hip_ver}" if hip_ver else "CUDA"
        print(f"[train] GPU backend : {backend}")
        print(f"[train] GPU         : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Data  — pin_memory + prefetch_factor give free throughput on ROCm.
    # We try to forward them to get_dataloaders; if the function signature
    # doesn't accept them we fall back to patching the loaders directly.
    # ------------------------------------------------------------------
    _dl_extra = dict(
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    try:
        train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            **_dl_extra,
        )
    except TypeError:
        # get_dataloaders doesn't accept pin_memory / prefetch_factor yet —
        # create loaders with defaults and patch the flags in-place.
        train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        for _loader in (train_loader, val_loader):
            _loader.pin_memory        = _dl_extra["pin_memory"]
            if _dl_extra["prefetch_factor"] is not None:
                _loader.prefetch_factor = _dl_extra["prefetch_factor"]
        print("[train] pin_memory / prefetch_factor patched onto existing loaders")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    # torch.compile works on Python 3.12 with PyTorch ≥ 2.3.
    # "reduce-overhead" is best for fixed-size inputs (segmentation).
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[train] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"[train] torch.compile skipped: {e}")
            args.compile = False

    # ------------------------------------------------------------------
    # Loss, AMP, optimiser, scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    use_amp = args.amp and device.type in ("cuda", "mps")
    scaler  = _make_scaler(use_amp, device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ------------------------------------------------------------------
    # Optional checkpoint resume
    # ------------------------------------------------------------------
    start_epoch   = 1
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": []}

    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt      = load_checkpoint(ckpt_path, model, optimizer, device)
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", best_val_loss)
        print(f"[train] Resuming from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    speedups = []
    if use_amp:
        speedups.append("AMP/FP16")
    if args.compile:
        speedups.append("compile")
    speedup_str = f"  [{', '.join(speedups)}]" if speedups else ""

    print(f"\n{'='*60}")
    print(f"  U-Net  |  {args.epochs} epochs  |  bs={args.batch_size}"
          f"  |  device: {device}{speedup_str}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(val_loss)

        epoch_secs = time.perf_counter() - epoch_start
        epoch_times.append(epoch_secs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        tag = "✓ best" if val_loss < best_val_loss else ""
        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
            f"time: {epoch_secs:.1f}s  {tag}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                CHECKPOINTS_DIR / "best.pth",
            )

    total_secs = time.perf_counter() - total_start

    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        CHECKPOINTS_DIR / "last.pth",
    )

    # ------------------------------------------------------------------
    # Final stats summary
    # ------------------------------------------------------------------
    n_epochs     = len(epoch_times)
    avg_epoch    = sum(epoch_times) / n_epochs
    fastest      = min(epoch_times)
    slowest      = max(epoch_times)
    best_epoch   = history["val_loss"].index(min(history["val_loss"])) + 1
    total_mins, total_secs_rem = divmod(int(total_secs), 60)

    # Throughput: images processed per second during training
    train_samples  = len(train_loader.dataset)
    val_samples    = len(val_loader.dataset)
    total_samples  = (train_samples + val_samples) * n_epochs
    throughput     = total_samples / total_secs

    # Peak GPU memory (ROCm reports this the same way as CUDA)
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        mem_str = f"{peak_mem_mb:,.0f} MB"
    else:
        mem_str = "n/a"

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"{'='*60}")
    print(f"  Epochs trained     : {n_epochs}")
    print(f"  Total time         : {total_mins}m {total_secs_rem:02d}s")
    print(f"  Avg time / epoch   : {avg_epoch:.1f}s")
    print(f"  Fastest epoch      : {fastest:.1f}s")
    print(f"  Slowest epoch      : {slowest:.1f}s")
    print(f"  Throughput         : {throughput:,.0f} samples/s")
    print(f"  Peak GPU memory    : {mem_str}")
    print(f"  Best val loss      : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"  Final val loss     : {history['val_loss'][-1]:.4f}")
    print(f"  Final train loss   : {history['train_loss'][-1]:.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for aerial segmentation")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS,    help="Number of epochs")
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,    help="Batch size (try 16–32)")
    p.add_argument("--lr",          type=float, default=LEARNING_RATE, help="Learning rate")
    p.add_argument("--num-workers", type=int,   default=6,             help="DataLoader workers (6 is a good default on Ryzen laptops)")
    p.add_argument("--resume",      type=str,   default=None,          help="Path to checkpoint to resume from")
    p.add_argument("--amp",         action="store_true",               help="Mixed precision FP16 (recommended on RX 6700S)")
    p.add_argument("--compile",     action="store_true",               help="torch.compile reduce-overhead mode (PyTorch ≥ 2.3)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
