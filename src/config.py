"""
config file
"""

from pathlib import Path

import kagglehub

ROOT_DIR = Path(__file__).resolve().parent.parent   # repo root
KAGGLE_DATASET = "deasadiqbal/private-data-1"
_CACHE_PATH = Path(kagglehub.dataset_download(KAGGLE_DATASET)) # indepotent

# we have: 
    # <cache>/patches/Images/
    # <cache>/patches/Labels/
IMAGES_DIR = _CACHE_PATH / "patches" / "Images"
MASKS_DIR  = _CACHE_PATH / "patches" / "Labels"

RESULTS_DIR     = ROOT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR     = RESULTS_DIR / "figures"

CLASS_INFO = [
#   id         name                 color
    (0, "Impervious Surface",  (255, 255, 255)),
    (1, "Building",            (0,   0,   255)),
    (2, "Low Vegetation",      (0,   255, 255)),
    (3, "Tree",                (0,   255, 0  )),
    (4, "Car",                 (255, 255, 0  )),
    (5, "Clutter/Background",  (255, 0,   0  )),
]

NUM_CLASSES  = len(CLASS_INFO)
CLASS_NAMES  = [c[1] for c in CLASS_INFO]
CLASS_COLORS = [c[2] for c in CLASS_INFO]
COLOR_TO_CLASS = {color: idx for idx, _, color in CLASS_INFO}

IMAGE_SIZE  = (256, 256)   # default is 300×300
BATCH_SIZE  = 8
NUM_EPOCHS  = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8          # 80 % train, 20 % val
SEED = 42
