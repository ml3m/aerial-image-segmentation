"""
Albumentations transform pipelines for the U-Net.

The training pipeline uses geometric + photometric augmentations that keep
(image, mask) aligned. The validation pipeline only resizes + normalizes.

Both return torch tensors via `ToTensorV2`; the caller then gets:
  image : float32  [3, H, W]   ImageNet-normalized
  mask  : int64    [H, W]      class indices
"""

from __future__ import annotations

from typing import Iterable, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet stats — reasonable defaults for any natural-image backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: Iterable[int]) -> A.Compose:
    h, w = tuple(image_size)
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transform(image_size: Iterable[int]) -> A.Compose:
    h, w = tuple(image_size)
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_inference_transform(image_size: Iterable[int]) -> A.Compose:
    """Same as val but exposed separately for clarity at call sites."""
    return build_val_transform(image_size)
