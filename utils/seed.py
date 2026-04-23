"""Deterministic seeding across random/numpy/torch."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Trade a bit of speed for reproducibility during development.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
