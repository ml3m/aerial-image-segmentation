# YOLO Segmentation - ISPRS Potsdam Dataset

Train a YOLOv8 segmentation model on the ISPRS Potsdam aerial imagery dataset, with support for **normal training**, **10-fold cross-validation**, and **early stopping**.

## Classes

| ID | Class              | Mask Color (RGB)      |
|----|--------------------|-----------------------|
| 0  | Impervious surface | (255, 255, 255) white |
| 1  | Building           | (0, 0, 255) blue      |
| 2  | Low vegetation     | (0, 255, 255) cyan    |
| 3  | Tree               | (0, 255, 0) green     |
| 4  | Car                | (255, 255, 0) yellow  |
| 5  | Clutter            | (255, 0, 0) red       |

## Setup

```bash
pip install -r Yolo_requirements.txt
```

By default this installs CPU PyTorch. For GPU acceleration, install the appropriate PyTorch build **before** the other requirements:

**NVIDIA (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**AMD (ROCm):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```

The training script auto-detects CUDA or ROCm at startup and falls back to CPU if neither is available.

## Step 1 — Download the Dataset

Download from Kaggle:  
https://www.kaggle.com/datasets/deasadiqbal/private-data-1/data

Extract so you have two directories: one with **images** and one with **RGB masks**.

## Step 2 — Prepare the Dataset

This converts the RGB masks to YOLO polygon labels and splits into train/val:

```bash
python prepare_dataset.py \
    --image_dir path/to/raw/images \
    --mask_dir  path/to/raw/masks \
    --output_dir dataset \
    --val_split 0.2 \
    --detect_colors
```

Use `--detect_colors` on the first run to verify the RGB color map matches your masks. If the colors differ from the defaults, edit `CLASS_COLOR_MAP` in `convert_masks_to_yolo.py`.

After this, you should have:

```
dataset/
  images/
    train/    (80% of images)
    val/      (20% of images)
  labels/
    train/    (YOLO .txt polygon labels)
    val/      (YOLO .txt polygon labels)
```

Then update the `path` field in `dataset.yaml` to point to the `dataset/` folder.

## Step 3 — Train

### Normal Training

```bash
python train.py --mode normal --data dataset.yaml --epochs 100
```

### 10-Fold Cross-Validation

```bash
python train.py --mode kfold --data dataset.yaml --data_root dataset --epochs 100
```

### Early Stopping

Early stopping is **enabled by default**. Training halts automatically when validation metrics stop improving, preventing overfitting and saving time.

| Flag              | Default | Description                                         |
|-------------------|---------|-----------------------------------------------------|
| `--early_stop`    | `ON`    | Enabled by default                                  |
| `--no-early_stop` | —       | Disables early stopping; trains for all epochs      |
| `--patience`      | `20`    | Epochs to wait without improvement before stopping  |

```bash
# Default: early stopping ON, patience=20
python train.py --mode normal --data dataset.yaml --epochs 100

# Custom patience (stop after 50 epochs without improvement)
python train.py --mode normal --data dataset.yaml --epochs 200 --patience 50

# Disable early stopping (always run all epochs)
python train.py --mode normal --data dataset.yaml --epochs 100 --no-early_stop
```

Early stopping applies to both normal and kfold modes.

### All Options

| Flag              | Default          | Description                              |
|-------------------|------------------|------------------------------------------|
| `--mode`          | `normal`         | `normal` or `kfold`                      |
| `--model`         | `yolov8n-seg.pt` | YOLO model (n/s/m/l/x variants)         |
| `--data`          | `dataset.yaml`   | Dataset config YAML                      |
| `--data_root`     | `dataset`        | Root data dir (kfold mode)               |
| `--epochs`        | `100`            | Training epochs                          |
| `--imgsz`         | `640`            | Input image size                         |
| `--batch`         | `8`              | Batch size                               |
| `--early_stop`    | `ON`             | Enable early stopping                    |
| `--no-early_stop` | —                | Disable early stopping                   |
| `--patience`      | `20`             | Early stopping patience (epochs)         |
| `--device`        | `auto`           | `auto`, `0` (GPU), or `cpu`              |
| `--workers`       | `4`              | DataLoader workers                       |
| `--k_folds`       | `10`             | Number of folds (kfold mode)             |
| `--seed`          | `42`             | Random seed                              |
| `--project`       | `runs/segment`   | Output directory                         |
| `--name`          | auto-generated   | Experiment name                          |

### Model Variants

Pick a model size based on your GPU memory:

| Model            | Params | Speed  | Accuracy |
|------------------|--------|--------|----------|
| `yolov8n-seg.pt` | 3.4M   | Fast   | Lower    |
| `yolov8s-seg.pt` | 11.8M  | Fast   | Good     |
| `yolov8m-seg.pt` | 27.3M  | Medium | Better   |
| `yolov8l-seg.pt` | 46.0M  | Slow   | High     |
| `yolov8x-seg.pt` | 71.8M  | Slow   | Highest  |

## Project Structure

```
IDL_proj/
├── requirements.txt          # Python dependencies
├── dataset.yaml              # YOLO dataset configuration
├── convert_masks_to_yolo.py  # RGB mask → YOLO polygon converter
├── prepare_dataset.py        # Full dataset preparation pipeline
├── train.py                  # Training (normal + 10-fold CV + early stopping)
└── README.md
```

## Output

Training results (weights, metrics, plots) are saved under `runs/segment/`.

For k-fold, each fold gets its own subdirectory with a summary printed at the end showing mean and standard deviation across all folds.
