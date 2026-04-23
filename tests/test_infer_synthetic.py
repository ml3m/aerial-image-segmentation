"""
End-to-end inference pipeline test with an untrained U-Net and no YOLO weights.

This never touches the network (no kagglehub download) and never requires a
GPU. We pass an in-memory UNet and a fake image on disk; the pipeline should
produce valid JSON + PNGs even when YOLO weights are missing.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from inference.pipeline import run_inference
from models.unet import UNet


def test_run_inference_without_yolo_weights(tmp_path):
    # 1. Write a synthetic 256x256 BGR image to disk.
    img_path = tmp_path / "aerial.png"
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    # 2. Small untrained UNet (speeds up the test considerably).
    unet = UNet(in_channels=3, num_classes=6, base_filters=8)

    out_dir = tmp_path / "out"
    paths = run_inference(
        image_path=img_path,
        yolo_weights=tmp_path / "does_not_exist.pt",  # force empty detections
        output_dir=out_dir,
        unet_model=unet,
    )

    # 3. All artifacts must exist and be non-empty / valid.
    for key in ("mask", "result", "detections", "uncertainty"):
        p = Path(paths[key])
        assert p.is_file(), f"missing artifact: {key} at {p}"
        assert p.stat().st_size > 0

    dets = json.loads(Path(paths["detections"]).read_text())
    assert isinstance(dets, list)
    assert dets == []  # no YOLO → empty list

    mask_img = cv2.imread(paths["mask"], cv2.IMREAD_COLOR)
    assert mask_img is not None
    assert mask_img.shape == (256, 256, 3)

    result_img = cv2.imread(paths["result"], cv2.IMREAD_COLOR)
    assert result_img is not None
    assert result_img.shape == (256, 256, 3)

    unc_img = cv2.imread(paths["uncertainty"], cv2.IMREAD_COLOR)
    assert unc_img is not None
    assert unc_img.shape == (256, 256, 3)
