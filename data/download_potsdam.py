"""
Download the Potsdam dataset from Kaggle via kagglehub.

kagglehub caches the dataset under `~/.cache/kagglehub`, so repeat calls are
idempotent and free.

Usage:
    python -m data.download_potsdam
"""

from __future__ import annotations

from pathlib import Path

from utils.cfg import load_config


def download_potsdam() -> Path:
    import kagglehub

    cfg = load_config()
    handle = cfg.unet.dataset.kagglehub_handle
    print(f"[potsdam] resolving kagglehub dataset: {handle}")
    cache_path = Path(kagglehub.dataset_download(handle))
    print(f"[potsdam] cache → {cache_path}")

    images_dir = cache_path / cfg.unet.dataset.images_subdir
    masks_dir = cache_path / cfg.unet.dataset.masks_subdir

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            "Expected Potsdam layout not found:\n"
            f"  images: {images_dir}\n"
            f"  masks : {masks_dir}\n"
            "Check the kagglehub handle and the `images_subdir`/`masks_subdir` in config.yaml."
        )

    n_img = sum(1 for _ in images_dir.iterdir())
    n_msk = sum(1 for _ in masks_dir.iterdir())
    print(f"[potsdam] {n_img} images / {n_msk} masks located")
    return cache_path


if __name__ == "__main__":
    download_potsdam()
