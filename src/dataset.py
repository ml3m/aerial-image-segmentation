"""
data/
    images/  (RGB aerial photos)
    masks/   (RGB annotation masks)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.config import (
    IMAGES_DIR,
    MASKS_DIR,
    COLOR_TO_CLASS,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    SEED,
)

def _rgb_mask_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an H×W×3 uint8 RGB mask to an H×W int64 class-index mask.
    Unknown colours are mapped to the last class (Clutter/Background).
    """
    h, w = mask_rgb.shape[:2]
    out = np.full((h, w), fill_value=NUM_CLASSES - 1, dtype=np.int64)
    for color, cls_idx in COLOR_TO_CLASS.items():
        match = (
            (mask_rgb[:, :, 0] == color[0])
            & (mask_rgb[:, :, 1] == color[1])
            & (mask_rgb[:, :, 2] == color[2])
        )
        out[match] = cls_idx
    return out


class PotsdamDataset(Dataset):
    """
    Potsdam dataset.

    arguments:
        images_dir: Path to the dir containing aerial images
        masks_dir:  Path to the dir containing RGB label masks.
        image_size: (H, W) to resize inputs.
    """

    # even though our data is in .tiff only
    _IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        images_dir: Path = IMAGES_DIR,
        masks_dir: Path = MASKS_DIR,
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.image_size = image_size

        import re
        _NUM_RE = re.compile(r"\d+")

        def _id(path: Path) -> str | None:
            """Extract the trailing numeric ID from a filename, e.g. 'Image_336' → '336'."""
            m = _NUM_RE.search(path.stem)
            return m.group() if m else None

        # Build dicts: numeric_id → full path
        image_by_id: dict[str, Path] = {}
        for p in self.images_dir.iterdir():
            if p.suffix.lower() in self._IMG_EXT:
                fid = _id(p)
                if fid is not None:
                    image_by_id[fid] = p

        mask_by_id: dict[str, Path] = {}
        for p in self.masks_dir.iterdir():
            if p.suffix.lower() in self._IMG_EXT:
                fid = _id(p)
                if fid is not None:
                    mask_by_id[fid] = p

        common_ids = sorted(set(image_by_id.keys()) & set(mask_by_id.keys()), key=int)
        if not common_ids:
            raise FileNotFoundError(
                f"No matching image/mask pairs found.\n"
                f"  images_dir: {self.images_dir}\n"
                f"  masks_dir:  {self.masks_dir}\n"
                "Run download_dataset.py first."
            )

        # Resolve full paths using the pre-built dicts
        self.image_paths: list[Path] = [image_by_id[fid] for fid in common_ids]
        self.mask_paths:  list[Path] = [mask_by_id[fid]  for fid in common_ids]

        # Fixed transforms
        self._resize_img  = transforms.Resize(image_size, interpolation=Image.BILINEAR)
        self._resize_mask = transforms.Resize(image_size, interpolation=Image.NEAREST)
        self._to_tensor   = transforms.ToTensor()
        self._normalize   = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats — good starting point
            std =[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("RGB")

        # Resize
        image = self._resize_img(image)
        mask  = self._resize_mask(mask)

        # Convert image → normalised float tensor [3, H, W]
        image_tensor = self._normalize(self._to_tensor(image))

        # Convert RGB mask → class-index tensor [H, W]
        mask_np    = np.array(mask)
        mask_cls   = _rgb_mask_to_class(mask_np)
        mask_tensor = torch.from_numpy(mask_cls)   # dtype int64

        return image_tensor, mask_tensor

def get_dataloaders(
    images_dir: Path = IMAGES_DIR,
    masks_dir:  Path = MASKS_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    seed: int = SEED,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from the Potsdam dataset.

    Returns:
        (train_loader, val_loader)
    """
    full_dataset = PotsdamDataset(images_dir, masks_dir, image_size)

    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val   = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"[dataset] train={n_train}  val={n_val}  total={n_total}")
    return train_loader, val_loader
