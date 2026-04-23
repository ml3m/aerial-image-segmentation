"""Device selection + HSA override env var."""

import os

import torch

from utils.device import apply_hsa_override, get_device


def test_apply_hsa_override_sets_env(monkeypatch):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    apply_hsa_override("10.3.0")
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"


def test_apply_hsa_override_is_idempotent(monkeypatch):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")  # user-provided
    apply_hsa_override("10.3.0")
    # setdefault() must NOT clobber an existing user value.
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)
    assert d.type in {"cuda", "mps", "cpu"}
