"""
YOLOv8 OBB training on the VSAI dataset.

The HSA override for the RX 6700S (gfx1032) MUST be set before torch or
ultralytics is imported.

Run:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m train.train_yolo
    python -m train.train_yolo --epochs 100 --batch 8 --imgsz 1024

This script:
  1. Ensures the VSAI kagglehub dataset is downloaded and `vsai_dataset.yaml`
     is up to date (regenerated each run so `path:` is always correct).
  2. Calls `ultralytics.YOLO(...).train(...)` with an OBB model.
"""

from __future__ import annotations

# Env var MUST land in os.environ before torch is imported.
from utils.device import apply_hsa_override  # noqa: E402

apply_hsa_override()

import argparse
from pathlib import Path

from data.download_vsai import download_vsai
from utils.cfg import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 OBB on VSAI")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    p.add_argument("--model", type=str, default=None, help="YOLO weights (e.g. yolov8n-obb.pt)")
    p.add_argument("--data", type=str, default=None, help="Override dataset YAML path")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", help="'auto', '0', 'cpu', ...")
    p.add_argument("--project", type=str, default=None, help="Run output directory")
    p.add_argument("--name", type=str, default="train", help="Run name within project dir")
    p.add_argument("--skip-download", action="store_true",
                   help="Assume vsai_dataset.yaml already exists and is valid")
    return p.parse_args()


def _resolve_device(flag: str) -> str:
    if flag != "auto":
        return flag
    import torch

    if torch.cuda.is_available():
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        print(f"[yolo] {backend} detected — using GPU 0")
        return "0"
    print("[yolo] no GPU detected — falling back to CPU")
    return "cpu"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not args.skip_download:
        download_vsai()  # idempotent; always regenerates vsai_dataset.yaml with absolute path

    data_yaml = Path(args.data) if args.data else resolve_path(cfg.yolo.dataset.generated_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML missing: {data_yaml}. Run `python -m data.download_vsai` first."
        )

    model_weights = args.model or cfg.yolo.model
    epochs = args.epochs or cfg.yolo.epochs
    imgsz = args.imgsz or cfg.yolo.imgsz
    batch = args.batch or cfg.yolo.batch
    patience = args.patience if args.patience is not None else cfg.yolo.patience
    workers = args.workers if args.workers is not None else cfg.yolo.workers
    project = args.project or str(resolve_path(cfg.paths.yolo_runs_dir))

    device = _resolve_device(args.device)

    from models.yolo import load_yolo

    print(
        "\n"
        f"[yolo] model    : {model_weights}\n"
        f"[yolo] data     : {data_yaml}\n"
        f"[yolo] epochs   : {epochs}\n"
        f"[yolo] imgsz    : {imgsz}\n"
        f"[yolo] batch    : {batch}\n"
        f"[yolo] patience : {patience}\n"
        f"[yolo] device   : {device}\n"
        f"[yolo] project  : {project}\n"
    )

    model = load_yolo(model_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=args.name,
        patience=patience,
        save=True,
        plots=True,
    )
    print(f"\n[yolo] done. results → {getattr(results, 'save_dir', project)}")


if __name__ == "__main__":
    main()
