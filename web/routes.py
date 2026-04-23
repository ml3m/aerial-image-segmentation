"""HTTP routes for the inference UI."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from utils.cfg import load_config
from web.inference_service import (
    create_job_dir,
    run_inference_job,
    save_upload,
)
from web.job_analysis import run_job_analysis

bp = Blueprint("main", __name__)

log = logging.getLogger(__name__)

_JOB_FILES = frozenset(
    {
        "result.png",
        "mask.png",
        "detections.json",
        "uncertainty.png",
        "class_mix.png",
        "yolo_analytics.png",
        "input_preview.png",
    }
)
_CROP_RE = re.compile(r"^crop_\d+\.png$")
_TRAINING_FIGURES = frozenset(
    {"training_yolo.png", "training_unet.png", "inference_composite.png"}
)


@bp.after_request
def _no_store_when_needed(response):
    ep = request.endpoint
    if ep in ("main.job_file", "main.job_crop"):
        response.headers["Cache-Control"] = "no-store, private"
        return response
    if ep == "main.index" and (
        request.args.get("job") or request.args.get("error")
    ):
        response.headers["Cache-Control"] = "no-store, private"
    return response


def _parse_optional_float(name: str) -> float | None:
    raw = request.form.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid value for {name.replace('_', ' ')}.") from e


def _parse_optional_int(name: str) -> int | None:
    raw = request.form.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid value for {name.replace('_', ' ')}.") from e


@bp.route("/")
def index():
    cfg = load_config()
    job_id = request.args.get("job", "").strip()
    job_out: Path | None = None
    detections: list | None = None
    error: str | None = request.args.get("error")

    if job_id:
        try:
            uuid.UUID(job_id)
        except ValueError:
            abort(404)
        upload_root: Path = current_app.config["UPLOAD_ROOT"]
        job_out = upload_root / job_id / "out"
        if not job_out.is_dir():
            abort(404)
        det_path = job_out / "detections.json"
        if det_path.is_file():
            try:
                detections = json.loads(det_path.read_text())
            except json.JSONDecodeError:
                detections = []

    artifacts: dict[str, bool] = {}
    crop_urls: list[str] = []
    if job_out is not None and job_id:
        for name in (
            "result.png",
            "mask.png",
            "uncertainty.png",
            "input_preview.png",
            "class_mix.png",
            "yolo_analytics.png",
        ):
            artifacts[name] = (job_out / name).is_file()
        crops_dir = job_out / "crops"
        if crops_dir.is_dir():
            for p in sorted(crops_dir.glob("crop_*.png")):
                crop_urls.append(
                    url_for("main.job_crop", job_id=job_id, filename=p.name)
                )

    return render_template(
        "index.html",
        cfg=cfg,
        defaults={
            "conf": cfg.inference.conf_threshold,
            "iou": cfg.inference.iou_threshold,
            "mask_alpha": cfg.inference.mask_alpha,
            "box_thickness": cfg.inference.box_thickness,
        },
        job_id=job_id if job_out else None,
        detections=detections,
        error=error,
        artifacts=artifacts,
        crop_urls=crop_urls,
    )


@bp.post("/infer")
def infer():
    upload_root: Path = current_app.config["UPLOAD_ROOT"]

    try:
        job_id, job_path = create_job_dir(upload_root)
        input_path = save_upload(request.files.get("image"), job_path)
        out_dir = job_path / "out"

        overrides = {
            "conf_threshold": _parse_optional_float("conf_threshold"),
            "iou_threshold": _parse_optional_float("iou_threshold"),
            "mask_alpha": _parse_optional_float("mask_alpha"),
            "box_thickness": _parse_optional_int("box_thickness"),
        }

        run_inference_job(
            image_path=input_path,
            output_dir=out_dir,
            overrides={k: v for k, v in overrides.items() if v is not None},
        )
    except ValueError as e:
        return redirect(url_for("main.index", error=str(e)), code=303)
    except Exception as e:  # noqa: BLE001 — surface to user
        return redirect(
            url_for("main.index", error=f"Inference failed: {e}"),
            code=303,
        )

    try:
        run_job_analysis(job_path, load_config())
    except Exception:  # noqa: BLE001
        log.exception("post-inference job analysis failed")

    return redirect(url_for("main.index", job=job_id), code=303)


@bp.route("/jobs/<job_id>/<filename>")
def job_file(job_id: str, filename: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        abort(404)
    if filename not in _JOB_FILES:
        abort(404)
    if Path(filename).name != filename:
        abort(404)

    upload_root: Path = current_app.config["UPLOAD_ROOT"]
    out_dir = upload_root / job_id / "out"
    if not out_dir.is_dir():
        abort(404)
    return send_from_directory(out_dir, secure_filename(filename), max_age=0)


@bp.route("/jobs/<job_id>/crops/<filename>")
def job_crop(job_id: str, filename: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        abort(404)
    if not _CROP_RE.match(filename) or Path(filename).name != filename:
        abort(404)

    upload_root: Path = current_app.config["UPLOAD_ROOT"]
    crops_dir = upload_root / job_id / "out" / "crops"
    if not crops_dir.is_dir():
        abort(404)
    return send_from_directory(crops_dir, secure_filename(filename), max_age=0)


@bp.route("/training-figures/<filename>")
def training_figure(filename: str):
    if filename not in _TRAINING_FIGURES or Path(filename).name != filename:
        abort(404)
    root: Path = current_app.config["PROJECT_ROOT"] / "figures"
    if not (root / filename).is_file():
        abort(404)
    return send_from_directory(root, filename, max_age=3600)


@bp.route("/training")
def training():
    root: Path = current_app.config["PROJECT_ROOT"] / "figures"
    available = []
    labels = {
        "training_yolo.png": "YOLO training curves",
        "training_unet.png": "U-Net training",
        "inference_composite.png": "Example inference composite",
    }
    for fname in sorted(_TRAINING_FIGURES):
        if (root / fname).is_file():
            available.append(
                {
                    "filename": fname,
                    "label": labels.get(fname, fname),
                    "url": url_for("main.training_figure", filename=fname),
                }
            )
    return render_template("training.html", figures=available)


@bp.route("/health")
def health():
    try:
        import torch

        gpu = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu = False
    return jsonify(ok=True, gpu_available=gpu)
