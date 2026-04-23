"""Checkpoint save/load helpers with forward-compatible `weights_only` handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a checkpoint. Uses `weights_only=False` because we need optimizer state."""
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[ckpt] loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt
