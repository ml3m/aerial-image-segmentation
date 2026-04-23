"""
Run combined inference behind a process-wide GPU lock.

Imports ``run_inference`` only inside the locked section so this module can be
imported after ``apply_hsa_override()`` without pulling torch at import time.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

_GPU_LOCK = threading.Lock()

ALLOWED_UPLOAD_EXT = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def _suffix_allowed(filename: str) -> str | None:
    suf = Path(filename).suffix.lower()
    return suf if suf in ALLOWED_UPLOAD_EXT else None


def create_job_dir(upload_root: Path) -> tuple[str, Path]:
    job_id = str(uuid.uuid4())
    job_path = upload_root / job_id
    job_path.mkdir(parents=True, exist_ok=False)
    (job_path / "out").mkdir(exist_ok=True)
    return job_id, job_path


def save_upload(file_storage: FileStorage, job_path: Path) -> Path:
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided.")
    raw_name = file_storage.filename
    suf = _suffix_allowed(raw_name)
    if not suf:
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXT))}"
        )
    safe = secure_filename(Path(raw_name).name)
    if not safe:
        safe = "upload"
    dest = job_path / f"input{suf}"
    file_storage.save(dest)
    return dest


def apply_inference_overrides(cfg, overrides: dict[str, Any]) -> None:
    """Mutate cfg.inference in place for optional form fields."""
    inf = cfg.inference
    if "conf_threshold" in overrides and overrides["conf_threshold"] is not None:
        inf.conf_threshold = float(overrides["conf_threshold"])
    if "iou_threshold" in overrides and overrides["iou_threshold"] is not None:
        inf.iou_threshold = float(overrides["iou_threshold"])
    if "mask_alpha" in overrides and overrides["mask_alpha"] is not None:
        inf.mask_alpha = float(overrides["mask_alpha"])
    if "box_thickness" in overrides and overrides["box_thickness"] is not None:
        inf.box_thickness = int(overrides["box_thickness"])


def run_inference_job(
    image_path: Path,
    output_dir: Path,
    overrides: dict[str, Any] | None = None,
    unet_weights: str | Path | None = None,
    yolo_weights: str | Path | None = None,
) -> dict[str, str]:
    """
    Load config, apply overrides, run ``run_inference`` under GPU lock.

    Returns the string path dict from ``run_inference``.
    """
    from utils.cfg import load_config

    cfg = load_config()
    if overrides:
        apply_inference_overrides(cfg, overrides)

    with _GPU_LOCK:
        from inference.pipeline import run_inference

        return run_inference(
            image_path=image_path,
            cfg=cfg,
            unet_weights=unet_weights,
            yolo_weights=yolo_weights,
            output_dir=output_dir,
        )
