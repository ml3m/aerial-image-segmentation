"""
End-to-end inference pipeline:

    load weights → YOLO detect (OBB) → U-Net segment → combine → save artifacts

Artifacts written under `cfg.paths.inference_out_dir` (default `results/inference/`):
    detections.json   list of OBB detections
    mask.png          colorized semantic mask (full-resolution)
    uncertainty.png   U-Net entropy heatmap (colormap BGR)
    result.png        composite (mask overlay + boxes)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
import torch

from data.augmentations import build_inference_transform
from inference.combine import Detection, combine
from inference.visualization import palette_from_class_info
from models.unet import UNet
from utils.cfg import Config, load_config, resolve_path
from utils.checkpoint import load_checkpoint
from utils.device import get_device


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _run_unet(
    image_bgr: np.ndarray,
    unet: UNet,
    device: torch.device,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (class-index mask HxW, uncertainty visualization BGR HxWx3).

    Uncertainty is Shannon entropy of the per-pixel class distribution,
    normalized by log(C), colorized with INFERNO for display.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tf = build_inference_transform(image_size)
    tensor = tf(image=image_rgb)["image"].unsqueeze(0).to(device)

    unet.eval()
    with torch.no_grad():
        logits = unet(tensor)
        probs = torch.softmax(logits, dim=1)
        ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1)[0]
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int32)
    ent_np = ent.cpu().numpy().astype(np.float32)

    h, w = image_bgr.shape[:2]
    num_classes = int(logits.shape[1])
    ent_max = float(np.log(max(num_classes, 2)))

    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        ent_np = cv2.resize(ent_np, (w, h), interpolation=cv2.INTER_LINEAR)

    ent_norm = np.clip(ent_np / ent_max, 0.0, 1.0)
    ent_u8 = (ent_norm * 255.0).astype(np.uint8)
    uncertainty_bgr = cv2.applyColorMap(ent_u8, cv2.COLORMAP_INFERNO)
    return pred, uncertainty_bgr


def _run_yolo(
    image_path: Path,
    yolo_weights: Optional[Path],
    conf: float,
    iou: float,
) -> List[Detection]:
    """
    Run YOLO OBB on the image. Returns an empty list if:
      - no weights file is found (soft-fail: still generate mask-only result), or
      - the model produces zero detections.
    """
    if yolo_weights is None or not Path(yolo_weights).is_file():
        print(f"[infer] no YOLO weights at {yolo_weights} -- returning empty detections")
        return []

    from models.yolo import load_yolo

    model = load_yolo(yolo_weights)
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    if not results:
        return []

    res = results[0]
    names: dict = getattr(res, "names", {}) or {}
    obb = getattr(res, "obb", None)
    if obb is None or getattr(obb, "xyxyxyxy", None) is None:
        return []

    corners_tensor = obb.xyxyxyxy
    if hasattr(corners_tensor, "cpu"):
        corners_tensor = corners_tensor.cpu().numpy()
    else:
        corners_tensor = np.asarray(corners_tensor)

    confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else np.asarray(obb.conf)
    cls_ids = obb.cls.cpu().numpy().astype(int) if hasattr(obb.cls, "cpu") else np.asarray(obb.cls).astype(int)

    detections: List[Detection] = []
    for corners, conf_val, cid in zip(corners_tensor, confs, cls_ids):
        corners = np.asarray(corners).reshape(4, 2)
        detections.append(
            Detection(
                class_id=int(cid),
                class_name=str(names.get(int(cid), f"class_{int(cid)}")),
                conf=float(conf_val),
                corners=corners.astype(float),
            )
        )
    return detections


def _serialize_detections(detections: List[Detection]) -> List[dict]:
    out = []
    for d in detections:
        dct = asdict(d)
        dct["corners"] = d.corners.tolist()
        out.append(dct)
    return out


def run_inference(
    image_path: str | Path,
    cfg: Optional[Config] = None,
    unet_weights: Optional[str | Path] = None,
    yolo_weights: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    unet_model: Optional[UNet] = None,  # for tests / advanced use
) -> dict:
    """
    Run YOLO + U-Net on a single image and write mask, composite, uncertainty,
    and detections JSON.

    Returns a dict of absolute paths to the written files.
    """
    cfg = cfg or load_config()
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    out_dir = Path(output_dir) if output_dir else resolve_path(cfg.paths.inference_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # --- U-Net -------------------------------------------------------------
    if unet_model is None:
        unet_model = UNet(in_channels=3, num_classes=cfg.unet.num_classes).to(device)
        weights = Path(unet_weights) if unet_weights else resolve_path(cfg.inference.unet_weights)
        if weights.is_file():
            load_checkpoint(weights, unet_model, optimizer=None, device=device)
        else:
            print(f"[infer] no U-Net weights at {weights} -- using random init")
    else:
        unet_model = unet_model.to(device)

    image_bgr = _read_image_bgr(image_path)
    mask, uncertainty_bgr = _run_unet(
        image_bgr,
        unet_model,
        device,
        image_size=tuple(cfg.unet.image_size),
    )

    mask_raw = mask.copy()

    # --- Morphological Post-Processing ---
    cleaned_mask = np.zeros_like(mask)
    morph_stats: dict[str, int] = {
        "shape_filter_blobs": 0,
        "shape_filter_pixels": 0,
        "car_overlap_blobs": 0,
        "car_overlap_pixels": 0,
    }

    # Pre-compute car mask to identify roads misclassified as buildings
    car_mask = (mask == 4).astype(np.uint8)

    # 1: building
    class_bin = (mask == 1).astype(np.uint8)
    kernel_build = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened = cv2.morphologyEx(class_bin, cv2.MORPH_OPEN, kernel_build)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_build)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            cv2.drawContours(cleaned_mask, [cnt], -1, 1, -1)
            continue

        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        # Check car overlap: create a filled mask of this blob's outer boundary
        blob_mask = np.zeros_like(closed)
        cv2.drawContours(blob_mask, [cnt], -1, 1, -1)
        overlaps_car = bool((blob_mask & car_mask).any())
        blob_pixel_count = int(blob_mask.sum())

        # Heuristic for roads: highly elongated, low circularity, OR has a car on it
        if overlaps_car:
            morph_stats["car_overlap_blobs"] += 1
            morph_stats["car_overlap_pixels"] += blob_pixel_count
        elif aspect_ratio > 7.0 or circularity < 0.07:
            morph_stats["shape_filter_blobs"] += 1
            morph_stats["shape_filter_pixels"] += blob_pixel_count
        else:
            cv2.drawContours(cleaned_mask, [cnt], -1, 1, -1)

    # 2: low_vegetation, 3: tree
    kernel_veg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for c in [2, 3]:
        class_bin = (mask == c).astype(np.uint8)
        opened = cv2.morphologyEx(class_bin, cv2.MORPH_OPEN, kernel_veg)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_veg)
        cleaned_mask[closed == 1] = c

    # 4: car
    class_bin = (mask == 4).astype(np.uint8)
    kernel_car = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(class_bin, cv2.MORPH_OPEN, kernel_car)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_car)
    cleaned_mask[closed == 1] = 4

    # 5: clutter
    class_bin = (mask == 5).astype(np.uint8)
    kernel_clutter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(class_bin, cv2.MORPH_OPEN, kernel_clutter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_clutter)
    cleaned_mask[closed == 1] = 5

    mask = cleaned_mask

    # --- YOLO --------------------------------------------------------------
    weights_yolo = Path(yolo_weights) if yolo_weights else resolve_path(cfg.inference.yolo_weights)
    detections = _run_yolo(
        image_path,
        weights_yolo,
        conf=cfg.inference.conf_threshold,
        iou=cfg.inference.iou_threshold,
    )

    # --- Compose + save ---------------------------------------------------
    palette_rgb = palette_from_class_info(cfg.unet.class_info)
    composite_bgr = combine(
        image_bgr,
        mask,
        detections,
        palette_rgb=palette_rgb,
        mask_alpha=cfg.inference.mask_alpha,
        box_thickness=cfg.inference.box_thickness,
    )

    mask_raw_rgb = palette_rgb[np.clip(mask_raw, 0, palette_rgb.shape[0] - 1)]
    mask_raw_bgr = cv2.cvtColor(mask_raw_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    mask_rgb = palette_rgb[np.clip(mask, 0, palette_rgb.shape[0] - 1)]
    mask_bgr = cv2.cvtColor(mask_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # --- Diff heatmap (pixels changed by post-processing) ---
    diff = (mask_raw != mask).astype(np.uint8) * 255
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
    # Black-out unchanged pixels so only changes pop
    diff_color[diff == 0] = 0

    paths = {
        "mask_raw": out_dir / "mask_raw.png",
        "mask": out_dir / "mask.png",
        "mask_diff": out_dir / "mask_diff.png",
        "result": out_dir / "result.png",
        "detections": out_dir / "detections.json",
        "uncertainty": out_dir / "uncertainty.png",
        "morph_stats": out_dir / "morph_stats.json",
    }
    cv2.imwrite(str(paths["mask_raw"]), mask_raw_bgr)
    cv2.imwrite(str(paths["mask"]), mask_bgr)
    cv2.imwrite(str(paths["mask_diff"]), diff_color)
    cv2.imwrite(str(paths["result"]), composite_bgr)
    cv2.imwrite(str(paths["uncertainty"]), uncertainty_bgr)
    with paths["detections"].open("w") as f:
        json.dump(_serialize_detections(detections), f, indent=2)
    with paths["morph_stats"].open("w") as f:
        json.dump(morph_stats, f, indent=2)

    total_changed = int((mask_raw != mask).sum())
    print(
        f"[infer] done:\n"
        f"  detections: {len(detections):>4d} -> {paths['detections']}\n"
        f"  mask                  -> {paths['mask']}\n"
        f"  mask_diff ({total_changed} px changed) -> {paths['mask_diff']}\n"
        f"  uncertainty           -> {paths['uncertainty']}\n"
        f"  composite             -> {paths['result']}\n"
        f"  morph_stats           -> {paths['morph_stats']}"
    )
    return {k: str(v) for k, v in paths.items()}
