"""
Download the VSAI OBB dataset via kagglehub and emit a YOLO-ready dataset YAML.

Usage:
    python -m data.download_vsai

Result:
    - Dataset cached by kagglehub (idempotent).
    - data/vsai_dataset.yaml written with absolute `path:` + `train:`/`val:` + `names:`.

VSAI is distributed in the YOLOv8/11 OBB layout. We probe the cache directory
to discover the actual layout (some re-uploads nest the data differently) and
fail with an actionable error message if the expected directories are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from utils.cfg import load_config, resolve_path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _find_dir(root: Path, name: str) -> Optional[Path]:
    """Return the first directory named `name` under `root` (BFS)."""
    if root.name == name and root.is_dir():
        return root
    if not root.is_dir():
        return None
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child in sorted(current.iterdir()):
            if child.is_dir():
                if child.name == name:
                    return child
                queue.append(child)
    return None


def _classes_from_classes_txt(root: Path) -> Optional[Dict[int, str]]:
    """If the cache contains a `classes.txt` or `obj.names`, parse it."""
    for candidate_name in ("classes.txt", "obj.names", "names.txt"):
        hits = list(root.rglob(candidate_name))
        if hits:
            classes = [
                line.strip()
                for line in hits[0].read_text().splitlines()
                if line.strip()
            ]
            return {i: n for i, n in enumerate(classes)}
    return None


def _classes_from_labels(labels_dir: Path) -> Dict[int, str]:
    """Fallback: infer class count from the max class-id across all label files."""
    max_id = -1
    for txt in labels_dir.rglob("*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid > max_id:
                max_id = cid
    if max_id < 0:
        # Empty or unparseable labels: fall back to a single-class dataset.
        return {0: "object"}
    return {i: f"class_{i}" for i in range(max_id + 1)}


def _find_split_root(cache_root: Path) -> Path:
    """
    Locate the directory containing `train/images`, `val/images`, etc.

    We prefer the split-style layout (train/val/test each with images+labels).
    Searches up to 3 levels deep inside the kagglehub cache.
    """
    for candidate in [cache_root, *[p for p in cache_root.rglob("*") if p.is_dir()][:500]]:
        if (candidate / "train" / "images").is_dir() and (
            (candidate / "val" / "images").is_dir()
            or (candidate / "valid" / "images").is_dir()
        ):
            return candidate
    return cache_root


def _resolve_splits(dataset_root: Path) -> tuple[str, str, Path]:
    """
    Pick relative paths (from dataset_root) for train/val images
    and return the labels directory used to infer class IDs.
    """
    candidates = [
        ("train/images", "val/images"),
        ("train/images", "valid/images"),
        ("images/train", "images/val"),
        ("images/training", "images/validation"),
        ("train", "val"),
    ]
    for train, val in candidates:
        tdir = dataset_root / train
        vdir = dataset_root / val
        if tdir.is_dir() and vdir.is_dir():
            labels_dir = (
                (dataset_root / train.replace("images", "labels"))
                if "images" in train
                else tdir.parent / "labels"
            )
            return train, val, labels_dir

    flat = dataset_root / "images"
    if flat.is_dir():
        print("[vsai] WARNING: no train/val split found — using `images/` for both.")
        return "images", "images", dataset_root / "labels"

    raise FileNotFoundError(
        "Could not find a train/val split under the VSAI cache.\n"
        f"Searched under: {dataset_root}\n"
        "Expected one of: train/images + val/images, images/train + images/val, or train/ + val/."
    )


def _load_bundled_yaml(cache_root: Path) -> Optional[tuple[Path, dict]]:
    """If the dataset ships a `data.yaml`, return (path, parsed)."""
    hits = sorted(cache_root.rglob("data.yaml"))
    if not hits:
        return None
    bundled = hits[0]
    try:
        data = yaml.safe_load(bundled.read_text()) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict) or "names" not in data:
        return None
    return bundled, data


def download_vsai() -> Path:
    import kagglehub

    cfg = load_config()
    handle = cfg.yolo.dataset.kagglehub_handle
    print(f"[vsai] resolving kagglehub dataset: {handle}")
    cache_root = Path(kagglehub.dataset_download(handle))
    print(f"[vsai] cache → {cache_root}")

    yaml_path = resolve_path(cfg.yolo.dataset.generated_yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    bundled = _load_bundled_yaml(cache_root)
    if bundled is not None:
        bundled_path, bundled_data = bundled
        dataset_root = bundled_path.parent
        print(f"[vsai] using bundled data.yaml → {bundled_path}")

        train_rel = bundled_data.get("train", "train/images")
        val_rel = bundled_data.get("val", "val/images")
        names = bundled_data["names"]
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}

        if not (dataset_root / train_rel).is_dir():
            print(
                f"[vsai] WARNING: bundled train path {train_rel!r} missing — "
                "falling back to directory discovery."
            )
            train_rel, val_rel, _ = _resolve_splits(dataset_root)

        yaml_out = {
            "path": str(dataset_root.resolve()),
            "train": train_rel,
            "val": val_rel,
            "names": names,
        }
    else:
        split_root = _find_split_root(cache_root)
        print(f"[vsai] split root → {split_root}")
        train_rel, val_rel, labels_dir = _resolve_splits(split_root)
        names = _classes_from_classes_txt(cache_root) or _classes_from_labels(labels_dir)
        yaml_out = {
            "path": str(split_root.resolve()),
            "train": train_rel,
            "val": val_rel,
            "names": names,
        }

    print(f"[vsai] classes: {names}")
    with yaml_path.open("w") as f:
        yaml.safe_dump(yaml_out, f, default_flow_style=False, sort_keys=False)
    print(f"[vsai] wrote dataset YAML → {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    download_vsai()
