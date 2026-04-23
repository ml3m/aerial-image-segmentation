"""
U-Net training driver.

Run:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m train.train_unet
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m train.train_unet --epochs 30 --batch-size 8
    python -m train.train_unet --resume results/unet/checkpoints/best.pth
"""

from __future__ import annotations

# The HSA override MUST be applied before torch is imported on ROCm/RX 6700S.
from utils.device import apply_hsa_override  # noqa: E402

apply_hsa_override()

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
    tqdm = None

from data.potsdam_dataset import get_dataloaders
from models.unet import UNet
from utils.cfg import load_config, resolve_path
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device
from utils.seed import set_seed


def _wrap_tqdm(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


def _amp_context(device: torch.device, enabled: bool):
    """Return a torch.amp autocast context, or nullcontext if disabled."""
    if not enabled or device.type not in ("cuda", "mps"):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except (AttributeError, RuntimeError):
        if device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()


def _build_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return None
    # Prefer the non-deprecated API (PyTorch 2.4+).
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


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
    ctx = _amp_context(device, use_amp)

    for images, masks in _wrap_tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(images)
            loss = criterion(logits, masks)

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
    ctx = _amp_context(device, use_amp)

    for images, masks in _wrap_tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with ctx:
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.device.seed)
    device = get_device()

    epochs = args.epochs or cfg.unet.epochs
    batch_size = args.batch_size or cfg.unet.batch_size
    lr = args.lr or cfg.unet.lr
    num_workers = args.num_workers if args.num_workers is not None else cfg.device.num_workers

    if args.image_size is not None:
        h, w = args.image_size
        cfg.unet.image_size = [int(h), int(w)]

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        cfg=cfg,
    )

    model = UNet(in_channels=3, num_classes=cfg.unet.num_classes).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except (RuntimeError, AttributeError) as e:
            print(f"[train] torch.compile disabled: {e}")
            args.compile = False

    criterion = nn.CrossEntropyLoss()

    use_amp = (args.amp or (args.amp is None and cfg.device.amp)) and device.type in ("cuda", "mps")
    scaler = _build_scaler(device, use_amp)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=cfg.unet.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", best_val))
        print(f"[train] resumed at epoch {start_epoch}")

    ckpt_dir = resolve_path(cfg.paths.unet_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tags = []
    if use_amp:
        tags.append("AMP")
    if args.compile:
        tags.append("compile")
    tag_str = f"  [{', '.join(tags)}]" if tags else ""

    print(f"\n{'=' * 55}")
    print(f"  U-Net | {epochs} epochs | device: {device}{tag_str}")
    print(f"{'=' * 55}\n")

    val_loss = float("inf")
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(val_loss)

        flag = "✓ best" if val_loss < best_val else ""
        print(
            f"Epoch [{epoch:03d}/{epochs}]  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  {flag}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_dir / "best.pth")

    save_checkpoint(model, optimizer, epochs, val_loss, ckpt_dir / "last.pth")
    print(f"\n[train] Done. Best val loss: {round(best_val, 4)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for aerial segmentation")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable mixed precision (default: follow config.device.amp)")
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Override U-Net train/val resize (default: config unet.image_size). "
        "Use smaller values on mobile AMD GPUs if training hangs (e.g. 384 384).",
    )
    p.add_argument("--compile", action="store_true", help="Use torch.compile (may fail on 3.12)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
