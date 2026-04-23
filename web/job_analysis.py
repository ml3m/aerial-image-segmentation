"""Per-job plots and detection crops for the web UI (post-inference)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from data.potsdam_dataset import build_color_to_class, rgb_mask_to_class
from utils.cfg import Config

log = logging.getLogger(__name__)


def rgb_mask_to_class_ids(mask_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    color_to_class = build_color_to_class(cfg.unet.class_info)
    return rgb_mask_to_class(mask_rgb, color_to_class, cfg.unet.num_classes)


def _class_names_ordered(cfg: Config) -> list[str]:
    rows = sorted(cfg.unet.class_info, key=lambda e: int(e[0]))
    return [str(r[1]) for r in rows]


def write_class_mix_plot(
    class_ids: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    n = len(class_names)
    counts = np.bincount(class_ids.ravel().astype(np.int64), minlength=n)
    total = max(int(counts.sum()), 1)
    pct = 100.0 * counts / total

    fig, ax = plt.subplots(figsize=(max(6.0, n * 0.55), 4.0))
    x = np.arange(n)
    ax.bar(x, counts, color="#0d5c5c", edgecolor="#094040")
    ax.set_xticks(x, labels=class_names, rotation=35, ha="right")
    ax.set_ylabel("Pixels")
    ax.set_title("U-Net class mix (decoded mask)")
    for i in range(n):
        if counts[i] > 0:
            ax.text(
                i,
                counts[i],
                f"{pct[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_yolo_analytics(
    detections: list,
    image_shape: tuple[int, int],
    out_path: Path,
) -> None:
    h, w = int(image_shape[0]), int(image_shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    if not detections:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No detections",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_axis_off()
        fig.suptitle("YOLO analytics", y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    by_name = Counter(str(d["class_name"]) for d in detections)
    labels = list(by_name.keys())
    axes[0].bar(range(len(labels)), [by_name[k] for k in labels], color="#2a6f6f")
    axes[0].set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Detections per class")

    confs = [float(d["conf"]) for d in detections]
    axes[1].hist(confs, bins=min(20, max(5, len(confs))), color="#6f4a2a", edgecolor="#4a3018")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Confidence distribution")

    xs: list[float] = []
    ys: list[float] = []
    for d in detections:
        arr = np.asarray(d["corners"], dtype=np.float64)
        xs.append(float(arr[:, 0].mean()))
        ys.append(float(arr[:, 1].mean()))
    axes[2].scatter(xs, ys, alpha=0.65, s=28, c="#1c4d6e", edgecolors="none")
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)
    axes[2].set_xlabel("x (px)")
    axes[2].set_ylabel("y (px)")
    axes[2].set_title("Detection centers")
    axes[2].set_aspect("equal", adjustable="box")

    fig.suptitle("YOLO analytics", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_detection_crops(
    image_bgr: np.ndarray,
    detections: list,
    crops_dir: Path,
) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("crop_*.png"):
        old.unlink(missing_ok=True)

    ih, iw = image_bgr.shape[:2]
    for i, d in enumerate(detections):
        arr = np.asarray(d["corners"], dtype=np.float64)
        x0 = int(np.clip(np.floor(arr[:, 0].min()), 0, max(iw - 1, 0)))
        x1 = int(np.clip(np.ceil(arr[:, 0].max()), 0, iw))
        y0 = int(np.clip(np.floor(arr[:, 1].min()), 0, max(ih - 1, 0)))
        y1 = int(np.clip(np.ceil(arr[:, 1].max()), 0, ih))
        if x1 <= x0:
            x1 = min(x0 + 1, iw) if iw else x0 + 1
        if y1 <= y0:
            y1 = min(y0 + 1, ih) if ih else y0 + 1
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        cv2.imwrite(str(crops_dir / f"crop_{i:02d}.png"), crop)


def run_job_analysis(job_root: Path, cfg: Config) -> None:
    """Read mask, detections, and input from a job directory; write plots and crops."""
    job_root = Path(job_root)
    out = job_root / "out"

    inputs = sorted(job_root.glob("input.*"))
    if not inputs:
        log.warning("run_job_analysis: no input.* under %s", job_root)
        return
    inp_path = inputs[0]

    try:
        img = cv2.imread(str(inp_path), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("run_job_analysis: could not read %s", inp_path)
        else:
            cv2.imwrite(str(out / "input_preview.png"), img)
    except Exception:  # noqa: BLE001
        log.exception("run_job_analysis: input_preview failed")

    mask_path = out / "mask.png"
    if mask_path.is_file():
        try:
            mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask_bgr is None:
                raise ValueError("mask.png unreadable")
            cls = rgb_mask_to_class_ids(mask_bgr, cfg)
            write_class_mix_plot(cls, _class_names_ordered(cfg), out / "class_mix.png")
        except Exception:  # noqa: BLE001
            log.exception("run_job_analysis: class_mix failed")

    det_path = out / "detections.json"
    detections: list = []
    if det_path.is_file():
        try:
            detections = json.loads(det_path.read_text())
            if not isinstance(detections, list):
                detections = []
        except Exception:  # noqa: BLE001
            log.exception("run_job_analysis: detections.json parse failed")
            detections = []

    try:
        img2 = cv2.imread(str(inp_path), cv2.IMREAD_COLOR)
        if img2 is not None:
            h, w = img2.shape[:2]
        else:
            h, w = 256, 256
        write_yolo_analytics(detections, (h, w), out / "yolo_analytics.png")
    except Exception:  # noqa: BLE001
        log.exception("run_job_analysis: yolo_analytics failed")

    try:
        img3 = cv2.imread(str(inp_path), cv2.IMREAD_COLOR)
        if img3 is not None and detections:
            write_detection_crops(img3, detections, out / "crops")
    except Exception:  # noqa: BLE001
        log.exception("run_job_analysis: detection crops failed")
