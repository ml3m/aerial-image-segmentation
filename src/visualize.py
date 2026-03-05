"""
visualize.py — Visualize model predictions alongside ground truth masks.

Usage:
    # Save 4 random validation samples to results/figures/
    python -m src.visualize --checkpoint results/checkpoints/best.pth --num-samples 4

    # Show interactively (no save)
    python -m src.visualize --checkpoint results/checkpoints/best.pth --show
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from src.config import (
    CLASS_NAMES,
    CLASS_COLORS,
    NUM_CLASSES,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
)
from src.dataset import get_dataloaders
from src.model import UNet
from src.utils import get_device, load_checkpoint

def decode_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert H×W integer class mask → H×W×3 uint8 RGB image using class colours.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_idx] = color
    return rgb

def _unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation and convert to H×W×3 uint8."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = tensor.cpu().permute(1, 2, 0).numpy()   # (H, W, 3)
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)

def plot_sample(
    image: torch.Tensor,
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    title: str = "",
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Three-panel figure: aerial image | ground truth | prediction."""
    img_np   = _unnormalize(image)
    true_rgb = decode_mask(true_mask.cpu().numpy())
    pred_rgb = decode_mask(pred_mask.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np);     axes[0].set_title("Aerial Image",   fontsize=13)
    axes[1].imshow(true_rgb);   axes[1].set_title("Ground Truth",   fontsize=13)
    axes[2].imshow(pred_rgb);   axes[2].set_title("U-Net Prediction", fontsize=13)
    for ax in axes:
        ax.axis("off")

    # Legend
    patches = [
        mpatches.Patch(facecolor=tuple(c / 255 for c in col), label=name)
        for name, col in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=9, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[visualize] Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)

@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = get_device()

    _, val_loader = get_dataloaders(batch_size=1, num_workers=0)

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    if args.checkpoint:
        load_checkpoint(Path(args.checkpoint), model, device=device)

    # Collect all validation samples
    samples = [(img, msk) for img, msk in val_loader]
    chosen  = random.sample(samples, min(args.num_samples, len(samples)))

    model.eval()
    for i, (image, mask) in enumerate(chosen):
        image  = image.to(device)
        logits = model(image)
        pred   = logits.argmax(dim=1).squeeze(0)  # (H, W)

        save_path = FIGURES_DIR / f"sample_{i + 1:03d}.png" if not args.show else None
        plot_sample(
            image=image.squeeze(0),
            true_mask=mask.squeeze(0),
            pred_mask=pred,
            title=f"Validation sample {i + 1}",
            save_path=save_path,
            show=args.show,
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize U-Net predictions")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best.pth"),
    )
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--show", action="store_true", help="Display figures instead of saving")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
