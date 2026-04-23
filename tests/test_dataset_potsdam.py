"""Dataset correctness + RGB-mask-to-class conversion."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.augmentations import build_train_transform, build_val_transform
from data.potsdam_dataset import (
    PotsdamDataset,
    build_color_to_class,
    rgb_mask_to_class,
)
from utils.cfg import load_config


CLASS_INFO = [
    (0, "impervious_surface", 255, 255, 255),
    (1, "building", 0, 0, 255),
    (2, "low_vegetation", 0, 255, 255),
    (3, "tree", 0, 255, 0),
    (4, "car", 255, 255, 0),
    (5, "clutter", 255, 0, 0),
]
COLOR_TO_CLASS = build_color_to_class(CLASS_INFO)


def test_build_color_to_class_roundtrip():
    assert COLOR_TO_CLASS[(255, 255, 255)] == 0
    assert COLOR_TO_CLASS[(0, 0, 255)] == 1
    assert COLOR_TO_CLASS[(255, 0, 0)] == 5


def test_rgb_mask_to_class_all_classes():
    # Build a 6x1 RGB mask where each pixel is a distinct palette color.
    palette = np.array([c[2:5] for c in CLASS_INFO], dtype=np.uint8)  # (6, 3)
    mask_rgb = palette.reshape(6, 1, 3)
    out = rgb_mask_to_class(mask_rgb, COLOR_TO_CLASS, num_classes=6)
    assert out.shape == (6, 1)
    assert list(out.squeeze().tolist()) == [0, 1, 2, 3, 4, 5]


def test_rgb_mask_to_class_unknown_maps_to_last():
    mask_rgb = np.full((2, 2, 3), fill_value=42, dtype=np.uint8)  # not in palette
    out = rgb_mask_to_class(mask_rgb, COLOR_TO_CLASS, num_classes=6)
    assert (out == 5).all()


def _write_synthetic_pair(tmp_path: Path, idx: int):
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir(exist_ok=True)
    msk_dir.mkdir(exist_ok=True)

    rng = np.random.RandomState(idx)
    img = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    # Build a mask using only palette colors
    palette = np.array([c[2:5] for c in CLASS_INFO], dtype=np.uint8)
    cls = rng.randint(0, len(palette), (32, 32))
    mask_rgb = palette[cls]

    Image.fromarray(img).save(img_dir / f"Image_{idx}.png")
    Image.fromarray(mask_rgb).save(msk_dir / f"Label_{idx}.png")


def test_dataset_getitem_shapes(tmp_path):
    for i in range(3):
        _write_synthetic_pair(tmp_path, i)

    tf = build_val_transform((16, 16))
    ds = PotsdamDataset(
        images_dir=tmp_path / "images",
        masks_dir=tmp_path / "masks",
        color_to_class=COLOR_TO_CLASS,
        num_classes=6,
        transform=tf,
    )
    assert len(ds) == 3
    img, msk = ds[0]
    assert img.shape == (3, 16, 16)
    assert msk.shape == (16, 16)
    assert msk.dtype.is_floating_point is False
    assert int(msk.min()) >= 0 and int(msk.max()) <= 5


def test_dataset_empty_raises(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    with pytest.raises(FileNotFoundError):
        PotsdamDataset(
            images_dir=tmp_path / "images",
            masks_dir=tmp_path / "masks",
            color_to_class=COLOR_TO_CLASS,
            num_classes=6,
        )


def test_build_color_to_class_from_real_cfg():
    cfg = load_config()
    mapping = build_color_to_class(cfg.unet.class_info)
    assert len(mapping) == cfg.unet.num_classes
