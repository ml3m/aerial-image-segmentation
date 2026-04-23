#!/usr/bin/env bash
# ROCm hardware validation for the RX 6700S (gfx1032).
#
# Run inside the project Docker container (preferred) or on the host after
# installing the ROCm build of PyTorch.
#
#   ./tests/rocm_check.sh
#   ./tests/rocm_check.sh --with-training   # also run 1-epoch U-Net + YOLO smoke
#
# The HSA override is exported for the whole shell up front.
set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION=10.3.0

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

WITH_TRAINING=0
for arg in "$@"; do
    case "$arg" in
        --with-training) WITH_TRAINING=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

echo "=== [1/4] torch.cuda.is_available() on ROCm ==="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "ROCm device not detected by torch.cuda"
print("  device name :", torch.cuda.get_device_name(0))
print("  torch       :", torch.__version__)
print("  hip version :", getattr(torch.version, "hip", None))
assert getattr(torch.version, "hip", None), "This PyTorch build is not ROCm/HIP"
PY
echo

echo "=== [2/4] U-Net GPU smoke test ==="
python -m models.unet
echo

echo "=== [3/4] Single forward+backward step on GPU ==="
python - <<'PY'
import torch, torch.nn as nn
from models.unet import UNet
from utils.device import get_device

device = get_device()
model = UNet(in_channels=3, num_classes=6, base_filters=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(2, 3, 256, 256, device=device)
y = torch.randint(0, 6, (2, 256, 256), device=device)
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
opt.step()
print(f"  loss after 1 step: {loss.item():.4f}  (no hipError → pass)")
PY
echo

if [ "$WITH_TRAINING" -eq 1 ]; then
    echo "=== [4/4] Mini training runs (U-Net + YOLO, 1 epoch) ==="
    echo "--- U-Net 1 epoch ---"
    python -m train.train_unet --epochs 1 --batch-size 2 --num-workers 0 --no-amp
    echo "--- YOLO 1 epoch ---"
    python -m train.train_yolo --epochs 1 --batch 2 --imgsz 640
else
    echo "=== [4/4] Skipping mini training (pass --with-training to run it) ==="
fi

echo
echo "ROCm validation complete."
