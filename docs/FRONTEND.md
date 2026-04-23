# Frontend integration guide

This document is for developers building a web UI (e.g. React, Vue, Svelte, or a server-rendered stack) that talks to a backend wrapping this aerial-image pipeline. It describes **what the pipeline produces**, **how to call it**, **request/response shapes**, and **pitfalls** so the frontend can be built in parallel.

---

## 1. Product summary

The system runs **two models** on one image:

| Model | Task | Output |
|--------|------|--------|
| **U-Net** | Semantic segmentation (6 classes) | Per-pixel class mask → colorized `mask.png` |
| **YOLOv8-OBB** | Oriented vehicle detection (2 classes) | Polygons + scores → `detections.json` |

The **composite** `result.png` overlays the colored mask (alpha blend) and draws each detection as a rotated box with a label.

The frontend typically:

1. Lets the user pick or drag-drop an image.
2. Sends it to **your** backend (Flask/FastAPI/etc.) — *this repo does not ship a REST API yet*.
3. Displays `result.png` (and optionally `mask.png`) and/or renders overlays from `detections.json` on a canvas.

---

## 2. Repository layout (relevant paths)

| Path | Role |
|------|------|
| `config.yaml` | Default weights paths, inference thresholds, mask alpha, U-Net input size |
| `infer.py` | CLI entry: `python infer.py --image <path> [--output dir] [--unet-weights ...] [--yolo-weights ...]` |
| `inference/pipeline.py` | **`run_inference()`** — the function your backend should wrap |
| `utils/device.py` | **`apply_hsa_override()`** — must run before `import torch` on AMD ROCm |
| `utils/cfg.py` | `load_config()`, `resolve_path()` — config loading |
| `results/unet/checkpoints/best.pth` | Default U-Net weights (after training) |
| `results/yolo/<run_name>/weights/best.pt` | YOLO weights (path depends on training `--name`; default config uses `train`) |

---

## 3. Backend contract (what you should ask the API team to implement)

There is **no HTTP server in this repo**. The frontend should assume a backend that exposes something like:

### Suggested: `POST /api/infer`

- **Request:** `multipart/form-data` with field `image` (file).
- **Response options** (pick one pattern):
  - **A)** JSON only: `{ "detections": [...], "result_base64": "...", "mask_base64": "..." }`
  - **B)** JSON + URLs: `{ "detections": [...], "result_url": "/files/job-uuid/result.png", ... }`
  - **C)** `multipart/mixed` or separate `GET` for images after `POST` returns a `job_id`.

The pipeline **writes files to disk**; the backend must copy or stream them to the client and use **unique output directories per request** (see §7).

### Suggested: `GET /api/health`

- Returns `200` when the process is up; optionally `gpu_available: true/false`.

### Optional query/body parameters for inference

| Parameter | Type | Default (from `config.yaml`) | Notes |
|-----------|------|-------------------------------|--------|
| `conf_threshold` | float | `0.25` | YOLO confidence |
| `iou_threshold` | float | `0.45` | YOLO NMS IoU |
| `mask_alpha` | float | `0.45` | Mask overlay strength on composite |

*Today these live only in config; exposing them requires backend to load config, mutate values, or pass extended kwargs if the pipeline is refactored.*

---

## 4. Core Python API (for backend implementers)

```python
from utils.device import apply_hsa_override
apply_hsa_override()  # BEFORE importing torch or inference.pipeline

from inference.pipeline import run_inference
from utils.cfg import load_config

paths = run_inference(
    image_path="/abs/path/to/upload.png",
    cfg=load_config(),                    # or load_config("/path/to/config.yaml")
    unet_weights=None,                   # optional override; None → config
    yolo_weights=None,
    output_dir="/abs/path/to/job-output", # use a unique dir per request
)
# paths = {"mask": str, "result": str, "detections": str}
```

Reference: `run_inference` in `inference/pipeline.py` — returns **absolute paths** to three files.

CLI equivalent:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python infer.py --image /path/to/image.png --output /path/to/out
```

---

## 5. Output files

| File | Content |
|------|---------|
| `result.png` | BGR image: original + semantically colored mask + OBB polygons and labels |
| `mask.png` | BGR image: mask only (colors from `config.yaml` → `unet.class_info`) |
| `detections.json` | JSON array of detections (see §6) |

Default output directory if `output_dir` omitted: `results/inference/` (from `config.yaml` → `paths.inference_out_dir`).

---

## 6. `detections.json` schema

Each element is one oriented box:

```json
{
  "class_id": 0,
  "class_name": "small-vehicle",
  "conf": 0.87,
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

- **`corners`:** four points in **pixel coordinates** in the **same coordinate system as the input image** (width × height).
- **Classes (VSAI training):** `small-vehicle`, `large-vehicle` (IDs 0 and 1).
- Empty array `[]` is valid if there are no detections or YOLO weights are missing.

Frontend ideas:

- Draw polygons with `<canvas>` or SVG `polygon`.
- Or simply show `result.png` if you do not need interactive boxes.

---

## 7. Concurrency and file names

`run_inference` always writes **`mask.png`**, **`result.png`**, **`detections.json`** inside the given `output_dir`. **Two concurrent requests with the same `output_dir` will overwrite each other.**

**Backend must use a unique directory per job**, e.g. `uploads/<uuid>/`.

---

## 8. Input image requirements

- The pipeline reads the file with **OpenCV** (`cv2.imread`). Common formats: PNG, JPEG, TIFF, etc.
- **Any resolution** is allowed; U-Net internally resizes using `config.yaml` → `unet.image_size`, then upsamples the predicted class map back to the original size with nearest-neighbor interpolation.

Frontend validation (optional):

- Max file size (set with product requirements).
- Allowed MIME types: `image/png`, `image/jpeg`, `image/tiff`, etc.

---

## 9. Configuration highlights (`config.yaml`)

### Paths

- `paths.inference_out_dir` — default folder for inference outputs.
- `inference.unet_weights` — e.g. `results/unet/checkpoints/best.pth`
- `inference.yolo_weights` — e.g. `results/yolo/train/weights/best.pt`

**Important:** If training used `python -m train.train_yolo --name safe`, weights may be at `results/yolo/safe/weights/best.pt`. The config file must match reality, or the backend must pass `yolo_weights=` explicitly.

### Inference tuning

- `inference.conf_threshold`, `inference.iou_threshold`
- `inference.mask_alpha` — opacity of the semantic overlay on `result.png`
- `inference.box_thickness` — OBB line thickness in pixels

### U-Net (affects segmentation only)

- `unet.image_size` — e.g. `[512, 512]`
- `unet.num_classes` — `6`
- `unet.class_info` — defines mask colors and class names for impervious surface, building, vegetation, tree, car, clutter

---

## 10. Semantic class colors (for legends / secondary UI)

From `config.yaml` (`unet.class_info`), typical mapping:

| ID | Name | RGB |
|----|------|-----|
| 0 | impervious_surface | 255, 255, 255 |
| 1 | building | 0, 0, 255 |
| 2 | low_vegetation | 0, 255, 255 |
| 3 | tree | 0, 255, 0 |
| 4 | car | 255, 255, 0 |
| 5 | clutter | 255, 0, 0 |

Use this for a **legend** next to `mask.png` / `result.png`. Detection class names come from YOLO (`small-vehicle`, `large-vehicle`), not from this table.

---

## 11. Environment and deployment notes

### AMD ROCm (e.g. Radeon RX 6700S)

- Set **`HSA_OVERRIDE_GFX_VERSION=10.3.0`** before importing PyTorch in the server process (see `utils/device.py`).
- Docker: project includes `Dockerfile` and `docker-run.sh` with GPU device flags (`/dev/kfd`, `/dev/dri`), shared memory, and optional `KAGGLE_*` for dataset download (not required for inference-only if weights exist).

### GPU vs CPU

- **GPU:** reasonable latency for single images.
- **CPU:** possible but very slow; health check should surface capability.

### Kaggle / datasets

- **Not required** for serving inference if `best.pth` and `best.pt` are already on the server.

---

## 12. Latency and UX

- First request after process start can be **slow** on ROCm (kernel compilation / warmup).
- Show a **spinner** or progress state; consider a **warmup** call on server boot.
- Large megapixel images increase U-Net and YOLO time; optional backend downscaling is a product decision (not implemented in the pipeline today).

---

## 13. Error cases the UI should handle

| Condition | Suggested HTTP | User-facing message |
|-----------|----------------|---------------------|
| Missing file / bad upload | 400 | “Invalid or missing image.” |
| Corrupt image (OpenCV cannot read) | 400 / 422 | “Could not read image.” |
| Missing U-Net weights | 503 (recommended) | Pipeline may run with **random** U-Net weights — mask is meaningless; backend should check files exist before inferring. |
| Missing YOLO weights | 503 or partial success | Detections empty; mask-only composite still produced. |
| GPU OOM / HIP error | 500 | “Inference failed; try a smaller image or retry later.” |

---

## 14. Domain mismatch (set expectations in the UI)

- **U-Net** was trained on **ISPRS Potsdam** (urban orthophoto style).
- **YOLO** was trained on **VSAI** (drone vehicle patches).

On **cross-domain** images, **segmentation may look poor** while **vehicle boxes** may still be reasonable (or vice versa). A short disclaimer in the UI avoids false expectations.

---

## 15. Local development without the ML stack

Frontend developers can:

1. **Mock the API** using static `result.png`, `mask.png`, and a sample `detections.json` checked into a fixture folder.
2. Run the **CLI** on sample images to generate fixtures:
   ```bash
   python infer.py --image sample.png --output fixtures/job1
   ```

---

## 16. Security checklist (backend + frontend)

- Validate upload size and type.
- Do not expose arbitrary server paths in JSON responses.
- Use short-lived signed URLs or opaque job IDs for downloaded images.
- Rate-limit `POST /api/infer` if the service is public.

---

## 17. Quick reference — CLI smoke test

```bash
# From repo root, with GPU env set as in docker-run.sh / README
python infer.py --image /path/to/aerial.jpg --output /tmp/infer-test
ls /tmp/infer-test
# mask.png  result.png  detections.json
```

---

## 18. Contact points in code

| Need | File |
|------|------|
| Inference orchestration | `inference/pipeline.py` |
| Overlay logic | `inference/combine.py`, `inference/visualization.py` |
| CLI flags | `infer.py` |
| Config | `config.yaml`, `utils/cfg.py` |
| ROCm workaround | `utils/device.py` |

---

*Generated for parallel frontend/backend development. Backend implementers should read §3–§4 and §7; UI implementers should focus on §5–§6, §12–§15.*
