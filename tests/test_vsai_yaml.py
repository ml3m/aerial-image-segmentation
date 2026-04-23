"""
VSAI dataset-yaml generation logic, without hitting the network.

We mock `kagglehub.dataset_download` to return a fake VSAI layout under tmp_path
and then run `download_vsai()` — the output YAML must be valid and point to the
expected paths and class names.
"""

from pathlib import Path

import yaml

from data import download_vsai as dv


def _make_fake_vsai(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    # One image + one label per split (content is irrelevant here).
    (root / "images" / "train" / "img1.png").write_bytes(b"\x89PNG\r\n")
    (root / "images" / "val" / "img2.png").write_bytes(b"\x89PNG\r\n")
    (root / "labels" / "train" / "img1.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4\n")
    (root / "labels" / "val" / "img2.txt").write_text("1 0.5 0.5 0.6 0.6 0.7 0.7 0.8 0.8\n")

    (root / "classes.txt").write_text("small_vehicle\nlarge_vehicle\n")
    return root


def test_download_vsai_generates_yaml(tmp_path, monkeypatch):
    fake_root = _make_fake_vsai(tmp_path / "vsai_cache")

    # Redirect kagglehub.dataset_download to return our fake root.
    import kagglehub

    def fake_download(handle: str) -> str:
        return str(fake_root)

    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)

    # Redirect the generated YAML target to tmp_path so we don't touch the repo.
    yaml_path = tmp_path / "vsai_dataset.yaml"

    original_resolve = dv.resolve_path

    def fake_resolve(value):
        if str(value).endswith("vsai_dataset.yaml"):
            return yaml_path
        return original_resolve(value)

    monkeypatch.setattr(dv, "resolve_path", fake_resolve)

    produced = dv.download_vsai()
    assert Path(produced) == yaml_path
    assert yaml_path.is_file()

    data = yaml.safe_load(yaml_path.read_text())
    assert Path(data["path"]).resolve() == fake_root.resolve()
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"] == {0: "small_vehicle", 1: "large_vehicle"}


def test_classes_fallback_from_labels(tmp_path, monkeypatch):
    """Without a classes.txt we should infer class count from label files."""
    root = tmp_path / "vsai_cache"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "images" / "train" / "a.png").write_bytes(b"\x89PNG\r\n")
    (root / "images" / "val" / "b.png").write_bytes(b"\x89PNG\r\n")
    (root / "labels" / "train" / "a.txt").write_text("2 0 0 0 0 0 0 0 0\n")
    (root / "labels" / "val" / "b.txt").write_text("0 0 0 0 0 0 0 0 0\n")

    import kagglehub

    monkeypatch.setattr(kagglehub, "dataset_download", lambda _h: str(root))

    yaml_path = tmp_path / "vsai_dataset.yaml"
    original_resolve = dv.resolve_path
    monkeypatch.setattr(
        dv,
        "resolve_path",
        lambda v: yaml_path if str(v).endswith("vsai_dataset.yaml") else original_resolve(v),
    )

    dv.download_vsai()
    data = yaml.safe_load(yaml_path.read_text())
    # Max class id = 2 → three classes generated.
    assert data["names"] == {0: "class_0", 1: "class_1", 2: "class_2"}
