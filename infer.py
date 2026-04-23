"""
Command-line inference entrypoint.

Usage:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python infer.py --image path/to/aerial.jpg
    python infer.py --image sample.png --output my_out/ --unet-weights ckpt.pth
"""

from __future__ import annotations

# HSA override MUST land in os.environ before torch is imported.
from utils.device import apply_hsa_override  # noqa: E402

apply_hsa_override()

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO+U-Net combined inference")
    p.add_argument("--image", required=True, help="Path to input aerial image")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--unet-weights", default=None, help="Override U-Net weights path")
    p.add_argument("--yolo-weights", default=None, help="Override YOLO weights path")
    p.add_argument("--output", default=None, help="Directory for output artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import lazily so apply_hsa_override() runs before torch/ultralytics load.
    from inference.pipeline import run_inference

    run_inference(
        image_path=args.image,
        cfg=None if args.config is None else _load_cfg(args.config),
        unet_weights=args.unet_weights,
        yolo_weights=args.yolo_weights,
        output_dir=args.output,
    )


def _load_cfg(path: str):
    from utils.cfg import load_config
    return load_config(path)


if __name__ == "__main__":
    main()
