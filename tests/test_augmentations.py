"""Albumentations pipeline correctness."""

import numpy as np
import torch

from data.augmentations import build_train_transform, build_val_transform


def test_train_transform_shapes_and_dtypes():
    tf = build_train_transform((64, 64))
    image = np.random.randint(0, 255, (128, 96, 3), dtype=np.uint8)
    mask = np.random.randint(0, 6, (128, 96), dtype=np.int64)

    out = tf(image=image, mask=mask)
    img_t, msk_t = out["image"], out["mask"]
    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape == (3, 64, 64)
    assert img_t.dtype == torch.float32
    assert msk_t.shape == (64, 64)
    # Mask class values must remain integer class ids (no normalization).
    vals = set(np.unique(msk_t.cpu().numpy()).tolist())
    assert vals.issubset({0, 1, 2, 3, 4, 5})


def test_val_transform_deterministic_shape():
    tf = build_val_transform((32, 32))
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.int64)
    out = tf(image=image, mask=mask)
    assert out["image"].shape == (3, 32, 32)
    assert out["mask"].shape == (32, 32)


def test_rotation_preserves_class_set():
    """After a geometric transform, the unique class ids should still be a subset of the originals."""
    tf = build_train_transform((64, 64))
    # Two easily distinguishable regions.
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[:32] = 1
    mask[32:] = 3
    image = np.stack([mask.astype(np.uint8) * 80] * 3, axis=-1)

    for _ in range(5):
        out = tf(image=image, mask=mask)
        uniq = set(np.unique(out["mask"].cpu().numpy()).tolist())
        assert uniq.issubset({0, 1, 3})
