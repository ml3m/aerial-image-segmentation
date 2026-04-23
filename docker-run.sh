#!/usr/bin/env bash
# Launch the project container with AMD GPU passthrough.
#
# Required flags:
#   --device=/dev/kfd + /dev/dri   expose the GPU to ROCm
#   --ipc=host                     shared memory for PyTorch DataLoaders
#   --shm-size=8G                  generous /dev/shm for DataLoader workers
#   HSA_OVERRIDE_GFX_VERSION=10.3.0  gfx1032 (RX 6700S) -> gfx1030 workaround
#
# Usage:
#   ./docker-run.sh                     # interactive shell
#   ./docker-run.sh python infer.py --image sample.png

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-aerial-seg:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[docker] Building $IMAGE_NAME (first run)..."
    # --network=host: BuildKit's default bridge network sometimes can't reach
    # DNS (common on Arch with iptables/nftables). Host networking always
    # works where the shell itself has network.
    docker build --network=host -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

docker run --rm -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size=8G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -e HOME=/workspace/aerial \
    -e KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-}" \
    -e KAGGLE_USERNAME="${KAGGLE_USERNAME:-}" \
    -e KAGGLE_KEY="${KAGGLE_KEY:-}" \
    -v "$PROJECT_DIR":/workspace/aerial \
    -w /workspace/aerial \
    "$IMAGE_NAME" \
    "$@"
