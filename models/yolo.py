"""
Thin wrapper around `ultralytics.YOLO` so the rest of the code does not
import ultralytics directly. This keeps the dependency lazy and easier to mock.
"""

from __future__ import annotations

from pathlib import Path


def load_yolo(weights: str | Path):
    """Load a YOLO model from a weights file (e.g. `yolov8n-obb.pt`)."""
    from ultralytics import YOLO

    return YOLO(str(weights))
