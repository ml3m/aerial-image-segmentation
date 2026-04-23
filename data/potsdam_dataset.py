"""
ISPRS Potsdam `torch.utils.data.Dataset` with Albumentations.

Expected layout (under the kagglehub cache root):
    <cache>/patches/Images/   RGB aerial images
    <cache>/patches/Labels/   RGB annotation masks (palette below)

Class palette (from config.yaml → unet.class_info):
    0 impervious_surface  (255,255,255)
    1 building            (  0,  0,255)
    2 low_vegetation      (  0,255,255)
    3 tree                (  0,255,  0)
    4 car                 (255,255,  0)
    5 clutter             (255,  0,  0)

Pairs are matched by the trailing numeric id in each filename
(e.g. `Image_336.tif` ↔ `Label_336.tif`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from data.augmentations import build_train_transform, build_val_transform
from data.download_potsdam import download_potsdam
from utils.cfg import load_config

_NUM_RE = re.compile(r"\d+")
_IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _file_id(path: Path) -> str | None:
    m = _NUM_RE.search(path.stem)
    return m.group() if m else None


def build_color_to_class(class_info: Sequence[Sequence]) -> dict[Tuple[int, int, int], int]:
    """Convert config-style [(id, name, r, g, b), ...] → {(r,g,b): id}."""
    out: dict[Tuple[int, int, int], int] = {}
    for entry in class_info:
        cls_id = int(entry[0])
        r, g, b = int(entry[2]), int(entry[3]), int(entry[4])
        out[(r, g, b)] = cls_id
    return out


def rgb_mask_to_class(
    mask_rgb: np.ndarray,
    color_to_class: dict[Tuple[int, int, int], int],
    num_classes: int,
) -> np.ndarray:
    """
    Convert an HxWx3 uint8 RGB mask to an HxW int64 class-index mask.
    Any RGB triplet not in `color_to_class` maps to the last class (treated
    as clutter/background).
    """
    h, w = mask_rgb.shape[:2]
    out = np.full((h, w), fill_value=num_classes - 1, dtype=np.int64)
    for (r, g, b), cls_idx in color_to_class.items():
        match = (
            (mask_rgb[..., 0] == r)
            & (mask_rgb[..., 1] == g)
            & (mask_rgb[..., 2] == b)
        )
        out[match] = cls_idx
    return out


class PotsdamDataset(Dataset):
    """Potsdam semantic segmentation dataset."""

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        color_to_class: dict[Tuple[int, int, int], int],
        num_classes: int,
        transform: A.Compose | Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.color_to_class = color_to_class
        self.num_classes = num_classes
        self.transform = transform

        image_by_id: dict[str, Path] = {}
        for p in self.images_dir.iterdir():
            if p.suffix.lower() in _IMG_EXT:
                fid = _file_id(p)
                if fid is not None:
                    image_by_id[fid] = p

        mask_by_id: dict[str, Path] = {}
        for p in self.masks_dir.iterdir():
            if p.suffix.lower() in _IMG_EXT:
                fid = _file_id(p)
                if fid is not None:
                    mask_by_id[fid] = p

        common = sorted(set(image_by_id) & set(mask_by_id), key=int)
        if not common:
            raise FileNotFoundError(
                "No matching image/mask pairs found.\n"
                f"  images_dir: {self.images_dir}\n"
                f"  masks_dir : {self.masks_dir}"
            )

        self.image_paths = [image_by_id[i] for i in common]
        self.mask_paths = [mask_by_id[i] for i in common]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_rgb = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))
        mask_cls = rgb_mask_to_class(mask_rgb, self.color_to_class, self.num_classes)

        if self.transform is not None:
            result = self.transform(image=image, mask=mask_cls)
            image_t = result["image"]
            mask_t = result["mask"]
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_cls)

        # Albumentations may leave the mask as float or int32 depending on dtype.
        if mask_t.dtype != torch.long:
            mask_t = mask_t.long()

        return image_t, mask_t


def get_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
    persistent_workers: bool = False,
    cfg=None,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders driven entirely by config.yaml.

    Downloads the dataset on first run (idempotent via kagglehub cache).

    Args:
        cfg: Optional pre-loaded config (e.g. after CLI overrides on ``unet.image_size``).
        pin_memory: Passed to DataLoader. On ROCm (HIP), ``False`` avoids rare host/driver issues.
    """
    cfg = cfg if cfg is not None else load_config()
    batch_size = batch_size if batch_size is not None else cfg.unet.batch_size
    num_workers = num_workers if num_workers is not None else cfg.device.num_workers
    if pin_memory is None:
        # ROCm reports as ``cuda`` but sets torch.version.hip; pinned CPU memory is less beneficial
        # and has been implicated in sporadic GPU hangs on some mobile AMD stacks.
        pin_memory = getattr(torch.version, "hip", None) is None

    cache_root = download_potsdam()
    images_dir = cache_root / cfg.unet.dataset.images_subdir
    masks_dir = cache_root / cfg.unet.dataset.masks_subdir
    color_to_class = build_color_to_class(cfg.unet.class_info)

    train_tf = build_train_transform(cfg.unet.image_size)
    val_tf = build_val_transform(cfg.unet.image_size)

    # We construct two independent datasets so each split gets its own transform.
    train_full = PotsdamDataset(
        images_dir, masks_dir, color_to_class, cfg.unet.num_classes, transform=train_tf
    )
    val_full = PotsdamDataset(
        images_dir, masks_dir, color_to_class, cfg.unet.num_classes, transform=val_tf
    )

    n_total = len(train_full)
    n_train = int(n_total * cfg.unet.train_split)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(cfg.device.seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)
    train_ds = torch.utils.data.Subset(train_full, list(train_idx))
    val_ds = torch.utils.data.Subset(val_full, list(val_idx))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    print(f"[dataset] train={n_train}  val={n_val}  total={n_total}")
    return train_loader, val_loader
