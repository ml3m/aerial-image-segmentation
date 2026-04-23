# syntax=docker/dockerfile:1
#
# Base image bundles a ROCm-aware PyTorch build and the HIP runtime. It already
# surfaces AMD GPUs as `torch.cuda.is_available()` == True, so no CUDA-specific
# APIs are required.
FROM rocm/pytorch:latest

ENV HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/aerial

# System dependency needed by opencv-python-headless for video/image codecs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-web.txt ./
# `torch`/`torchvision` are already provided by rocm/pytorch; we skip them here
# so pip never tries to reinstall with a potentially-wrong index URL. Every
# other dependency is pulled from PyPI. If BuildKit's default bridge can't
# reach DNS on your host, invoke `docker build --network=host ...` (docker-run.sh
# does this automatically).
RUN pip install --no-cache-dir \
        "ultralytics>=8.2.0" \
        "albumentations>=1.4.0" \
        "opencv-python-headless>=4.8.0" \
        "Pillow>=10.0.0" \
        "numpy>=1.24.0" \
        "scikit-learn>=1.3.0" \
        "PyYAML>=6.0" \
        "kagglehub>=0.3.0" \
        "tqdm>=4.65.0" \
        "matplotlib>=3.7.0" \
        "pytest>=7.4.0" \
 && pip install --no-cache-dir -r requirements-web.txt

COPY . .

EXPOSE 5000
# Default: interactive shell. Web UI (single GPU worker): \
#   gunicorn -w 1 -b 0.0.0.0:5000 'web.app:create_app()'

CMD ["bash"]
