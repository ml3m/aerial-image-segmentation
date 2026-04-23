"""
Combine U-Net mask + YOLO OBB detections into a single BGR visualization.

Inputs:
  image_bgr : HxWx3 uint8 (OpenCV BGR)
  mask      : HxW int class-index mask at the *original* image resolution
  detections: list of dicts:
        {"class_id": int, "class_name": str, "conf": float,
         "corners": np.ndarray of shape (4, 2) in pixel coordinates}

Output:
  HxWx3 uint8 BGR composite with:
    - colorized semantic mask alpha-blended over the image
    - each OBB drawn as a closed polygon with a class-colored label
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from inference.visualization import colorize_mask, hash_color


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    corners: np.ndarray  # shape (4, 2), float or int, pixel coords


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    palette_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend a colorized mask over the image (both HxW[x3], uint8)."""
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.int32),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask_rgb = colorize_mask(mask, palette_rgb)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, mask_bgr, alpha, 0.0)


def draw_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    thickness: int = 2,
) -> np.ndarray:
    """Draw OBB polygons + labels on the image. Returns a new array."""
    out = image_bgr.copy()
    for det in detections:
        color = hash_color(det.class_id)
        pts = np.asarray(det.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

        label = f"{det.class_name} {det.conf:.2f}"
        # Anchor the label at the top-left corner of the polygon.
        x = int(np.min(det.corners[:, 0]))
        y = int(np.min(det.corners[:, 1])) - 4
        y = max(y, 12)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            out,
            (x, y - th - baseline),
            (x + tw + 4, y + baseline),
            color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            out, label, (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def combine(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    detections: Sequence[Detection],
    palette_rgb: np.ndarray,
    mask_alpha: float = 0.45,
    box_thickness: int = 2,
) -> np.ndarray:
    """Full pipeline: mask overlay → boxes → return BGR image."""
    composite = overlay_mask(image_bgr, mask, palette_rgb, alpha=mask_alpha)
    composite = draw_detections(composite, detections, thickness=box_thickness)
    return composite
