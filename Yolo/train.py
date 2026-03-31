"""
YOLO Segmentation Training Script
Supports:
  1. Normal training (single train/val split)
  2. 10-fold cross-validation

Usage:
  Normal training:
    python train.py --mode normal --data dataset.yaml --epochs 100

  10-fold cross-validation:
    python train.py --mode kfold --data_root dataset --epochs 100

  See --help for all options.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml
import numpy as np
from sklearn.model_selection import KFold
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
#  Normal Training
# ──────────────────────────────────────────────────────────────────────────────

def train_normal(args):
    """Standard single-split training."""
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or "normal_train",
        patience=args.patience,
        save=True,
        plots=True,
    )
    print("\n=== Training complete ===")
    print(f"Results saved to: {results.save_dir}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  K-Fold Cross-Validation
# ──────────────────────────────────────────────────────────────────────────────

def get_image_label_pairs(data_root: str):
    """
    Discover all image/label pairs under data_root.

    Expected layout:
      data_root/
        images/
          train/   (and/or val/ — all images merged for k-fold)
            *.png / *.jpg / *.tif
        labels/
          train/   (and/or val/ — all labels merged for k-fold)
            *.txt

    Returns list of (image_path, label_path) tuples.
    """
    data_root = Path(data_root)
    img_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    image_dirs = []
    for subdir in ["images/train", "images/val", "images"]:
        candidate = data_root / subdir
        if candidate.is_dir():
            image_dirs.append(candidate)

    all_images = []
    for img_dir in image_dirs:
        for f in img_dir.iterdir():
            if f.suffix.lower() in img_extensions:
                all_images.append(f)

    all_images = sorted(set(all_images))

    pairs = []
    for img_path in all_images:
        rel = img_path.relative_to(data_root)
        label_rel = Path(str(rel).replace("images", "labels", 1)).with_suffix(".txt")
        label_path = data_root / label_rel

        if not label_path.exists():
            label_path_alt = data_root / "labels" / img_path.stem
            label_path_alt = label_path_alt.with_suffix(".txt")
            if label_path_alt.exists():
                label_path = label_path_alt

        pairs.append((img_path, label_path))

    return pairs


def create_fold_yaml(data_root: str, fold_dir: Path, fold_idx: int, class_names: dict) -> str:
    """Create a dataset YAML for a single fold."""
    yaml_content = {
        "path": str(fold_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    yaml_path = fold_dir / f"fold_{fold_idx}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return str(yaml_path)


def setup_fold_directories(
    fold_dir: Path,
    pairs: list,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
):
    """Symlink (or copy) images/labels into fold-specific train/val directories."""
    for split, indices in [("train", train_indices), ("val", val_indices)]:
        img_dir = fold_dir / "images" / split
        lbl_dir = fold_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            img_src, lbl_src = pairs[idx]

            img_dst = img_dir / img_src.name
            lbl_dst = lbl_dir / lbl_src.name

            if not img_dst.exists():
                try:
                    img_dst.symlink_to(img_src.resolve())
                except OSError:
                    shutil.copy2(img_src, img_dst)

            if lbl_src.exists() and not lbl_dst.exists():
                try:
                    lbl_dst.symlink_to(lbl_src.resolve())
                except OSError:
                    shutil.copy2(lbl_src, lbl_dst)


def train_kfold(args):
    """10-fold cross-validation training."""
    k = args.k_folds
    data_root = Path(args.data_root)

    with open(args.data, "r") as f:
        base_cfg = yaml.safe_load(f)
    class_names = base_cfg.get("names", {})

    pairs = get_image_label_pairs(str(data_root))
    if not pairs:
        print(f"[ERROR] No image/label pairs found in '{data_root}'.")
        print("Expected structure: data_root/images/(train|val)/ + data_root/labels/(train|val)/")
        sys.exit(1)

    print(f"Found {len(pairs)} image/label pairs for {k}-fold CV.\n")

    kfold = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(pairs))

    fold_results = []
    folds_base = Path(args.project) / (args.name or "kfold")
    folds_base.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1} / {k}")
        print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
        print(f"{'='*60}\n")

        fold_dir = folds_base / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        setup_fold_directories(fold_dir, pairs, train_idx, val_idx)

        fold_yaml = create_fold_yaml(str(data_root), fold_dir, fold_idx + 1, class_names)

        model = YOLO(args.model)
        results = model.train(
            data=fold_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(folds_base),
            name=f"fold_{fold_idx + 1}_train",
            patience=args.patience,
            save=True,
            plots=True,
        )
        fold_results.append(results)

    print_kfold_summary(fold_results, k)
    return fold_results


def print_kfold_summary(fold_results, k):
    """Print aggregated metrics across all folds."""
    print(f"\n{'='*60}")
    print(f"  {k}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")

    metrics_keys = []
    fold_metrics = []

    for i, result in enumerate(fold_results):
        try:
            r = result.results_dict
            fold_metrics.append(r)
            if not metrics_keys:
                metrics_keys = list(r.keys())
        except Exception:
            print(f"  Fold {i+1}: Could not retrieve metrics.")

    if not fold_metrics:
        print("  No metrics could be collected.")
        return

    print(f"  {'Metric':<40s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*60}")

    for key in metrics_keys:
        vals = []
        for fm in fold_metrics:
            v = fm.get(key)
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            arr = np.array(vals)
            print(f"  {key:<40s} {arr.mean():>10.4f} {arr.std():>10.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
#  Argument Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Segmentation Training (Normal + K-Fold)."
    )

    parser.add_argument(
        "--mode", type=str, choices=["normal", "kfold"], default="normal",
        help="Training mode: 'normal' or 'kfold'. Default: normal.",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="yolov8n-seg.pt",
        help="YOLO model variant. Default: yolov8n-seg.pt",
    )

    # Data
    parser.add_argument(
        "--data", type=str, default="dataset.yaml",
        help="Path to dataset YAML (used for normal training and to read class names for kfold).",
    )
    parser.add_argument(
        "--data_root", type=str, default="../dataset",
        help="Root directory of the dataset (for kfold mode). Default: ../dataset/",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs. Default: 100.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size. Default: 640.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Default: 8.")
    parser.add_argument(
        "--early_stop", action=argparse.BooleanOptionalAction, default=True,
        help="Enable early stopping. Use --no-early_stop to disable. Default: enabled.",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs with no improvement). Default: 20.")

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: '0' for GPU 0, 'cpu' for CPU, 'auto' to detect. Default: auto.",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers. Default: 4.")

    # Output
    parser.add_argument("--project", type=str, default="runs/segment", help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name.")

    # K-fold specific
    parser.add_argument("--k_folds", type=int, default=10, help="Number of folds for kfold mode. Default: 10.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold splits. Default: 42.")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    effective_patience = args.patience if args.early_stop else 0
    args.patience = effective_patience

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            backend = "ROCm" if hasattr(torch.version, "hip") and torch.version.hip else "CUDA"
            print(f"{backend} available — using GPU: {gpu_name}")
        else:
            args.device = "cpu"
            print("No GPU detected — falling back to CPU")

    print(f"Mode       : {args.mode}")
    print(f"Model      : {args.model}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image sz   : {args.imgsz}")
    print(f"Batch      : {args.batch}")
    print(f"Device     : {args.device}")
    if args.early_stop:
        print(f"Early stop : ON (patience={args.patience} epochs)")
    else:
        print(f"Early stop : OFF")

    if args.mode == "normal":
        print(f"Data YAML: {args.data}\n")
        train_normal(args)
    elif args.mode == "kfold":
        print(f"Data root: {args.data_root}")
        print(f"Folds    : {args.k_folds}\n")
        train_kfold(args)


if __name__ == "__main__":
    main()
