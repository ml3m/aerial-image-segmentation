"""Palette helpers shared by inference + dataset debug tools."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def palette_from_class_info(
    class_info: Sequence[Sequence],
) -> np.ndarray:
    """
    Convert config-style `[(id, name, r, g, b), ...]` into a (num_classes, 3)
    uint8 numpy array of RGB colors indexed by class id.
    """
    n = len(class_info)
    palette = np.zeros((n, 3), dtype=np.uint8)
    for entry in class_info:
        cls_id = int(entry[0])
        palette[cls_id] = (int(entry[2]), int(entry[3]), int(entry[4]))
    return palette


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Turn an HxW int mask into an HxWx3 uint8 RGB image via palette lookup."""
    mask = mask.astype(np.int64)
    # Guard against mask values outside the palette.
    np.clip(mask, 0, palette.shape[0] - 1, out=mask)
    return palette[mask]


def hash_color(cls_id: int) -> Tuple[int, int, int]:
    """Deterministic per-class color for YOLO detections (BGR for OpenCV)."""
    seed = (cls_id + 1) * 2654435761 & 0xFFFFFFFF
    r = (seed >> 0) & 0xFF
    g = (seed >> 8) & 0xFF
    b = (seed >> 16) & 0xFF
    return int(b), int(g), int(r)
