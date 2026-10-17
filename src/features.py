"""Feature extraction and caching module.

Extracts penultimate 512-d representations from frozen ImageNet ResNet-18.
Caches representations to cache/features_<split>.npz with exact original dataset indices.
Fits and applies StandardScaler statistics computed strictly on known_train.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from src.config import DEVICE, FEAT_DIM, PATHS, set_seed
from src.data import make_loaders


def build_backbone(device: str = DEVICE) -> nn.Module:
    """Instantiates a frozen pretrained ResNet-18 feature extractor."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # 512-d penultimate features
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model


@torch.no_grad()
def extract_features(loader, backbone: nn.Module, device: str = DEVICE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts features, labels, and dataset indices deterministically.

    Returns:
        feats: (N, 512) float32
        labels: (N,) int64
        dataset_indices: (N,) int64
    """
    backbone.eval()
    feat_list = []
    label_list = []
    idx_list = []

    start_time = time.time()
    total_samples = len(loader.dataset)
    processed = 0

    for i, (images, labels, indices) in enumerate(loader):
        images = images.to(device)
        feats = backbone(images)
        feat_list.append(feats.cpu().numpy().astype(np.float32))
        label_list.append(labels.numpy().astype(np.int64))
        idx_list.append(indices.numpy().astype(np.int64))
        processed += images.size(0)

        if i == 0:
            elapsed = time.time() - start_time
            throughput = processed / max(elapsed, 1e-4)
            eta_sec = (total_samples - processed) / max(throughput, 1e-4)
            print(f"    Initial batch: {throughput:.1f} img/s | Estimated split time: {eta_sec:.1f}s")

    all_feats = np.concatenate(feat_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    all_indices = np.concatenate(idx_list, axis=0)

    total_time = time.time() - start_time
    print(f"    Extracted {len(all_feats)} samples in {total_time:.1f}s ({len(all_feats)/max(total_time, 1e-4):.1f} img/s)")
    return all_feats, all_labels, all_indices


def get_features(split: str, backbone: nn.Module, device: str = DEVICE, force: bool = False) -> dict[str, np.ndarray]:
    """Retrieves cached features or extracts them from the dataset loader."""
    cache_path = PATHS["cache"] / f"features_{split}.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path)
        return {"f": data["f"], "y": data["y"], "idx": data["idx"]}

    print(f"[FEATURES] Extracting split: {split} ...")
    loaders = make_loaders()
    loader = loaders[split]
    feats, labels, indices = extract_features(loader, backbone, device)

    np.savez_compressed(cache_path, f=feats, y=labels, idx=indices)
    return {"f": feats, "y": labels, "idx": indices}


def extract_and_scale_all_splits(backbone: nn.Module, force: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """Extracts features for all 5 splits and standardizes them using known_train statistics only.

    Hard rule: Scaler mean and std are computed ONLY on known_train to prevent test leakage.
    """
    splits = ["known_train", "known_val", "known_test", "near_ood", "far_ood"]
    raw_data = {}

    for s in splits:
        raw_data[s] = get_features(s, backbone, DEVICE, force=force)

    scaler_path = PATHS["scaler"]
    if scaler_path.exists() and not force:
        scaler = np.load(scaler_path)
        mean = scaler["mean"]
        std = scaler["std"]
    else:
        # Fit scaler ONLY on known_train
        train_f = raw_data["known_train"]["f"]
        mean = np.mean(train_f, axis=0)
        std = np.std(train_f, axis=0) + 1e-8
        np.savez_compressed(scaler_path, mean=mean, std=std)

    scaled_data = {}
    for s in splits:
        scaled_f = (raw_data[s]["f"] - mean) / std
        scaled_data[s] = {
            "f": scaled_f.astype(np.float32),
            "raw_f": raw_data[s]["f"],
            "y": raw_data[s]["y"],
            "idx": raw_data[s]["idx"],
        }

    return scaled_data
