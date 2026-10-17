"""Configuration module for OOD detection study.

Single source of truth for all paths, hyperparameters, seeds, and class assignments.
All paths are derived dynamically from repository root.
"""

from pathlib import Path
import random
import numpy as np
import torch

# Random seed for full reproducibility across libraries
SEED = 1337

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 Class Partition
# Known classes (In-distribution): IDs 0-4
CIFAR_KNOWN_NAMES = ["airplane", "automobile", "bird", "cat", "deer"]
CIFAR_KNOWN_IDS = [0, 1, 2, 3, 4]

# Held-out classes (Near-OOD): IDs 5-9
# Deliberate semantic pairs: dog <-> cat, frog <-> bird, horse <-> deer, ship <-> airplane, truck <-> automobile
CIFAR_HELDOUT_NAMES = ["dog", "frog", "horse", "ship", "truck"]
CIFAR_HELDOUT_IDS = [5, 6, 7, 8, 9]

# Dataset Dimensions and Transformations
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0  # 0 is required for reliable Windows multiprocessing

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Feature Extraction and Probe Hyperparameters
FEAT_DIM = 512
VAL_FRACTION = 0.2
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RIDGE_LAMBDA = 1e-3

# Evaluation & Calibration Hyperparameters
N_ECE_BINS = 15
TEMP_BOUNDS = (0.05, 100.0)
TARGET_TPR = 0.95

# Repository Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PATHS = {
    "root": ROOT_DIR,
    "data": ROOT_DIR / "data",
    "cache": ROOT_DIR / "cache",
    "results": ROOT_DIR / "results",
    "figures": ROOT_DIR / "figures",
    "split_indices": ROOT_DIR / "cache" / "split_indices.npz",
    "scaler": ROOT_DIR / "cache" / "scaler.npz",
    "metrics_json": ROOT_DIR / "results" / "metrics.json",
    "summary_md": ROOT_DIR / "results" / "summary.md",
}

# Ensure essential output and cache directories exist
for p in [PATHS["data"], PATHS["cache"], PATHS["results"], PATHS["figures"]]:
    p.mkdir(parents=True, exist_ok=True)
