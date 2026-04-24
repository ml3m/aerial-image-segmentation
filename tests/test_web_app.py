"""Flask web UI: routing and upload flow with inference mocked."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest

# apply_hsa_override runs in create_app; keep imports after repo root on path (conftest)


@pytest.fixture
def app(tmp_path):
    from web.app import create_app

    upload = tmp_path / "uploads"
    return create_app(
        {
            "TESTING": True,
            "UPLOAD_ROOT": upload,
            "MAX_CONTENT_LENGTH": 5 * 1024 * 1024,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Run inference" in r.data


def test_health_returns_json(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.is_json
    body = r.get_json()
    assert body["ok"] is True
    assert "gpu_available" in body


def test_infer_mocked_redirects_with_job(client, monkeypatch):
    def fake_run_inference_job(
        *,
        image_path: Path,
        output_dir: Path,
        overrides=None,
        unet_weights=None,
        yolo_weights=None,
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.png").write_bytes(b"x")
        (out / "mask.png").write_bytes(b"y")
        (out / "uncertainty.png").write_bytes(b"z")
        (out / "detections.json").write_text("[]")
        return {
            "result": str(out / "result.png"),
            "mask": str(out / "mask.png"),
            "uncertainty": str(out / "uncertainty.png"),
            "detections": str(out / "detections.json"),
        }

    def noop_job_analysis(job_path, cfg):
        return None

    # Patch the name bound in routes (import-time binding), not only the service module.
    monkeypatch.setattr(
        "web.routes.run_inference_job",
        fake_run_inference_job,
    )
    monkeypatch.setattr("web.routes.run_job_analysis", noop_job_analysis)

    data = {
        "image": (io.BytesIO(b"\x89PNG\r\n\x1a\n"), "tiny.png"),
    }
    r = client.post(
        "/infer",
        data=data,
        content_type="multipart/form-data",
    )
    assert r.status_code == 303
    assert "job=" in r.headers["Location"]


def test_job_file_404_bad_uuid(client):
    r = client.get("/jobs/not-a-uuid/result.png")
    assert r.status_code == 404


def test_job_file_allowlist(app, client):
    job_id = str(uuid.uuid4())
    upload_root: Path = app.config["UPLOAD_ROOT"]
    out = upload_root / job_id / "out"
    out.mkdir(parents=True)
    (out / "result.png").write_bytes(b"ok")

    r = client.get(f"/jobs/{job_id}/result.png")
    assert r.status_code == 200

    r2 = client.get(f"/jobs/{job_id}/evil.png")
    assert r2.status_code == 404


def test_job_file_analytics_allowlist(app, client):
    job_id = str(uuid.uuid4())
    upload_root: Path = app.config["UPLOAD_ROOT"]
    out = upload_root / job_id / "out"
    out.mkdir(parents=True)
    (out / "class_mix.png").write_bytes(b"mix")

    ok = client.get(f"/jobs/{job_id}/class_mix.png")
    assert ok.status_code == 200
    assert "no-store" in ok.headers.get("Cache-Control", "").lower()

    missing = client.get(f"/jobs/{job_id}/yolo_analytics.png")
    assert missing.status_code == 404


def test_job_crop_route(app, client):
    job_id = str(uuid.uuid4())
    upload_root: Path = app.config["UPLOAD_ROOT"]
    crops = upload_root / job_id / "out" / "crops"
    crops.mkdir(parents=True)
    (crops / "crop_00.png").write_bytes(b"c")

    r = client.get(f"/jobs/{job_id}/crops/crop_00.png")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("Cache-Control", "").lower()

    assert client.get(f"/jobs/{job_id}/crops/crop_ab.png").status_code == 404
    assert client.get(f"/jobs/{job_id}/crops/../mask.png").status_code == 404


def test_index_no_store_when_job_in_query(app, client):
    job_id = str(uuid.uuid4())
    upload_root: Path = app.config["UPLOAD_ROOT"]
    (upload_root / job_id / "out").mkdir(parents=True)
    (upload_root / job_id / "out" / "detections.json").write_text("[]")

    r = client.get(f"/?job={job_id}")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("Cache-Control", "").lower()


def test_clears_uploads_on_start_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("WEB_KEEP_UPLOADS_ON_START", raising=False)
    upload = tmp_path / "up"
    upload.mkdir()
    stale = upload / "stale"
    stale.mkdir()
    (stale / "x.txt").write_text("old")

    from web.app import create_app

    create_app({"TESTING": True, "UPLOAD_ROOT": upload})
    assert not stale.exists()


def test_keeps_uploads_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_KEEP_UPLOADS_ON_START", "1")
    upload = tmp_path / "up"
    upload.mkdir()
    stale = upload / "stale"
    stale.mkdir()
    (stale / "x.txt").write_text("old")

    from web.app import create_app

    create_app({"TESTING": True, "UPLOAD_ROOT": upload})
    assert stale.is_dir()
    assert (stale / "x.txt").read_text() == "old"
    monkeypatch.delenv("WEB_KEEP_UPLOADS_ON_START", raising=False)


def test_training_page_ok(client):
    r = client.get("/training")
    assert r.status_code == 200
