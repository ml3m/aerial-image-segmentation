"""
evaluate file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import (
    NUM_CLASSES,
    CLASS_NAMES,
    BATCH_SIZE,
    CHECKPOINTS_DIR,
)
from src.dataset import get_dataloaders
from src.model import UNet
from src.utils import get_device, load_checkpoint

def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    accumulate a confusion matrix from a batch.

    arguments:
        preds:   (N,) int64 predicted    class indices
        targets: (N,) int64 ground-truth class indices
    returns:
        confusion matrix
    """
    mask = (targets >= 0) & (targets < num_classes)
    combined = num_classes * targets[mask] + preds[mask]
    cm = torch.bincount(combined, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)

def iou_from_confusion(cm: torch.Tensor) -> torch.Tensor:
    intersection = torch.diag(cm).float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou = torch.where(union > 0, intersection / union, torch.tensor(float("nan")))
    return iou

def pixel_accuracy_from_confusion(cm: torch.Tensor) -> float:
    correct = torch.diag(cm).sum().item()
    total   = cm.sum().item()
    return correct / total if total > 0 else 0.0

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> dict:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)                            # (B, C, H, W)
        preds  = logits.argmax(dim=1).cpu()               # (B, H, W)

        flat_preds   = preds.view(-1)
        flat_targets = masks.view(-1)
        cm += compute_confusion_matrix(flat_preds, flat_targets, num_classes)

    per_class_iou = iou_from_confusion(cm)
    mean_iou      = per_class_iou[~per_class_iou.isnan()].mean().item()
    pix_acc       = pixel_accuracy_from_confusion(cm)

    return {
        "per_class_iou": per_class_iou.numpy(),
        "mean_iou":      mean_iou,
        "pixel_accuracy": pix_acc,
    }

def print_results(results: dict, class_names: list[str] = CLASS_NAMES) -> None:
    print("\n" + "=" * 50)
    print("  Segmentation Evaluation Results")
    print("=" * 50)
    for name, iou in zip(class_names, results["per_class_iou"]):
        iou_str = f"{iou * 100:.2f} %" if not np.isnan(iou) else "  N/A  "
        print(f"  {name:<25s}  IoU: {iou_str}")
    print("-" * 50)
    print(f"  Mean IoU (mIoU) :  {results['mean_iou'] * 100:.2f} %")
    print(f"  Pixel Accuracy  :  {results['pixel_accuracy'] * 100:.2f} %")
    print("=" * 50 + "\n")

def main(args: argparse.Namespace) -> None:
    device = get_device()

    _, val_loader = get_dataloaders(batch_size=args.batch_size)

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    load_checkpoint(Path(args.checkpoint), model, device=device)

    results = evaluate(model, val_loader, device)
    print_results(results)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate U-Net segmentation quality")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best.pth"),
        help="Path to model checkpoint",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
