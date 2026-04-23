"""
Device selection and ROCm compatibility.

The RX 6700S (gfx1032) is not officially supported by ROCm. The workaround is
to set `HSA_OVERRIDE_GFX_VERSION=10.3.0` *before* `import torch`. This module
exposes a helper that applies the override into `os.environ` if it hasn't been
set yet — safe to call from training/inference entrypoints.
"""

from __future__ import annotations

import os


def apply_hsa_override(value: str = "10.3.0") -> None:
    """
    Ensure HSA_OVERRIDE_GFX_VERSION is set before torch is imported.

    Must be called from the top of an entrypoint script *before* `import torch`.
    Idempotent: if the env var is already set we keep the user's value.
    """
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", value)


def get_device():
    """
    Return the best available torch device.

    On AMD ROCm builds of PyTorch, ROCm surfaces as `torch.cuda` — the device
    type is still `"cuda"`. We also support Apple MPS for local dev.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        gpu = torch.cuda.get_device_name(0)
        print(f"[device] {backend} GPU detected: {gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Apple MPS detected")
    else:
        device = torch.device("cpu")
        print("[device] No accelerator — falling back to CPU")
    return device
