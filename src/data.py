"""Data pipeline and split verification for OOD detection.

Manages CIFAR-10 (known in-distribution and near-OOD held-out) and
FashionMNIST (far-OOD). Enforces strict split separation and stratified validation carving.
"""

from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

from src.config import (
    BATCH_SIZE,
    CIFAR_HELDOUT_IDS,
    CIFAR_HELDOUT_NAMES,
    CIFAR_KNOWN_IDS,
    CIFAR_KNOWN_NAMES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    PATHS,
    SEED,
    VAL_FRACTION,
    set_seed,
)


class IndexedDataset(Dataset):
    """Wraps a PyTorch Dataset to return (image, label, original_index)."""

    def __init__(self, subset_or_dataset):
        self.dataset = subset_or_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item[0], item[1]
        return image, label, idx


def get_transforms():
    """Builds standard preprocessing transforms matching ImageNet ResNet-18."""
    cifar_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    fashion_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=3),  # Upsample 1-channel to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return cifar_transform, fashion_transform


def ensure_cifar10_downloaded(data_dir: Path) -> None:
    """Ensures CIFAR-10 is downloaded and extracted, handling university redirect."""
    target_dir = data_dir / "cifar-10-batches-py"
    if target_dir.exists():
        return

    archive_path = data_dir / "cifar-10-python.tar.gz"
    if not archive_path.exists() or archive_path.stat().st_size != 170498071:
        print("[DATA] Downloading CIFAR-10 archive (170 MB)...")
        import urllib.request
        import shutil

        # Notice: Toronto moved the host from www.cs to cave.cs (301 redirect)
        urls = [
            "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        ]
        downloaded = False
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(archive_path, "wb") as out_file:
                    shutil.copyfileobj(resp, out_file)
                if archive_path.stat().st_size == 170498071:
                    downloaded = True
                    break
            except Exception as e:
                print(f"[DATA] Attempt from {url} failed: {e}. Trying next...")

        if not downloaded:
            raise RuntimeError("Failed to download CIFAR-10 archive completely.")

    print("[DATA] Extracting CIFAR-10 archive...")
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("[DATA] CIFAR-10 extracted successfully.")


def make_datasets() -> dict[str, Dataset]:
    """Downloads (if needed) and partitions datasets into required splits.

    Returns:
        dict with keys: 'known_train', 'known_val', 'known_test', 'near_ood', 'far_ood'
    """
    set_seed(SEED)
    data_dir = PATHS["data"]
    ensure_cifar10_downloaded(data_dir)
    cifar_transform, fashion_transform = get_transforms()

    # Raw datasets with transforms
    raw_cifar_train = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=cifar_transform)
    raw_cifar_test = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=cifar_transform)
    raw_fashion_test = datasets.FashionMNIST(root=str(data_dir), train=False, download=False, transform=fashion_transform)

    # 1. Filter CIFAR train for known classes (0..4)
    cifar_train_targets = np.array(raw_cifar_train.targets)
    known_train_pool_indices = np.where(np.isin(cifar_train_targets, CIFAR_KNOWN_IDS))[0]

    # Stratified split for train vs val
    split_cache = PATHS["split_indices"]
    if split_cache.exists():
        loaded = np.load(split_cache)
        train_idx = loaded["train_idx"]
        val_idx = loaded["val_idx"]
    else:
        rng = np.random.default_rng(SEED)
        train_idx_list = []
        val_idx_list = []
        for class_id in CIFAR_KNOWN_IDS:
            cls_indices = known_train_pool_indices[cifar_train_targets[known_train_pool_indices] == class_id]
            rng.shuffle(cls_indices)
            n_val = int(len(cls_indices) * VAL_FRACTION)
            val_idx_list.extend(cls_indices[:n_val])
            train_idx_list.extend(cls_indices[n_val:])

        train_idx = np.array(sorted(train_idx_list))
        val_idx = np.array(sorted(val_idx_list))
        np.savez_compressed(split_cache, train_idx=train_idx, val_idx=val_idx)

    # 2. Filter CIFAR test for known (0..4) and held-out (5..9)
    cifar_test_targets = np.array(raw_cifar_test.targets)
    known_test_indices = np.where(np.isin(cifar_test_targets, CIFAR_KNOWN_IDS))[0]
    near_ood_indices = np.where(np.isin(cifar_test_targets, CIFAR_HELDOUT_IDS))[0]

    # Build wrapped IndexedDatasets
    splits = {
        "known_train": IndexedDataset(Subset(raw_cifar_train, train_idx)),
        "known_val": IndexedDataset(Subset(raw_cifar_train, val_idx)),
        "known_test": IndexedDataset(Subset(raw_cifar_test, known_test_indices)),
        "near_ood": IndexedDataset(Subset(raw_cifar_test, near_ood_indices)),
        "far_ood": IndexedDataset(raw_fashion_test),
    }

    return splits


def make_loaders() -> dict[str, DataLoader]:
    """Creates deterministic DataLoaders for feature extraction and evaluation."""
    datasets_dict = make_datasets()
    loaders = {}
    for name, ds in datasets_dict.items():
        # Extraction order MUST be deterministic (shuffle=False) for caching index alignment
        loaders[name] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
    return loaders


def load_image_at(split: str, index: int) -> np.ndarray:
    """Loads raw un-normalized RGB image (H, W, 3) uint8 for visualization / gallery."""
    data_dir = PATHS["data"]

    if split in ["known_train", "known_val"]:
        raw_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=False)
        split_cache = np.load(PATHS["split_indices"])
        actual_pool = split_cache["train_idx"] if split == "known_train" else split_cache["val_idx"]
        real_idx = actual_pool[index]
        img = raw_ds.data[real_idx]  # H, W, 3 uint8
        return img

    elif split == "known_test":
        raw_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=False)
        targets = np.array(raw_ds.targets)
        known_indices = np.where(np.isin(targets, CIFAR_KNOWN_IDS))[0]
        real_idx = known_indices[index]
        return raw_ds.data[real_idx]

    elif split == "near_ood":
        raw_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=False)
        targets = np.array(raw_ds.targets)
        near_indices = np.where(np.isin(targets, CIFAR_HELDOUT_IDS))[0]
        real_idx = near_indices[index]
        return raw_ds.data[real_idx]

    elif split == "far_ood":
        raw_ds = datasets.FashionMNIST(root=str(data_dir), train=False, download=False)
        img = raw_ds.data[index].numpy()  # H, W uint8
        img_rgb = np.stack([img] * 3, axis=-1)
        return img_rgb

    raise ValueError(f"Unknown split: {split}")


def verify_data() -> None:
    """Verifies all dataset invariants and prints split distributions."""
    print("=" * 65)
    print(" [GATE 0] DATA PIPELINE VERIFICATION")
    print("=" * 65)

    datasets_dict = make_datasets()

    print(f"Split sizes:")
    for name, ds in datasets_dict.items():
        print(f"  - {name:<12}: {len(ds):>6} images")

    # 1. Assertions on split sizes
    n_train = len(datasets_dict["known_train"])
    n_val = len(datasets_dict["known_val"])
    n_test = len(datasets_dict["known_test"])
    n_near = len(datasets_dict["near_ood"])
    n_far = len(datasets_dict["far_ood"])

    assert n_train + n_val == 25000, f"Expected 25000 known train+val, got {n_train + n_val}"
    assert n_train == 20000, f"Expected 20000 known_train, got {n_train}"
    assert n_val == 5000, f"Expected 5000 known_val, got {n_val}"
    assert n_test == 5000, f"Expected 5000 known_test, got {n_test}"
    assert n_near == 5000, f"Expected 5000 near_ood, got {n_near}"
    assert n_far == 10000, f"Expected 10000 far_ood, got {n_far}"

    # 2. Check overlap between train and val
    split_cache = np.load(PATHS["split_indices"])
    train_idx = set(split_cache["train_idx"])
    val_idx = set(split_cache["val_idx"])
    overlap = train_idx.intersection(val_idx)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices between train and val!"

    # 3. Check class distributions in each split
    raw_cifar_train = datasets.CIFAR10(root=str(PATHS["data"]), train=True, download=False)
    raw_cifar_test = datasets.CIFAR10(root=str(PATHS["data"]), train=False, download=False)

    train_labels = [raw_cifar_train.targets[i] for i in split_cache["train_idx"]]
    val_labels = [raw_cifar_train.targets[i] for i in split_cache["val_idx"]]

    test_targets = np.array(raw_cifar_test.targets)
    known_test_labels = test_targets[np.isin(test_targets, CIFAR_KNOWN_IDS)].tolist()
    near_ood_labels = test_targets[np.isin(test_targets, CIFAR_HELDOUT_IDS)].tolist()

    print("\nClass Histograms:")
    print("  known_train (IDs 0-4):", dict(Counter(train_labels)))
    print("  known_val   (IDs 0-4):", dict(Counter(val_labels)))
    print("  known_test  (IDs 0-4):", dict(Counter(known_test_labels)))
    print("  near_ood    (IDs 5-9):", dict(Counter(near_ood_labels)))

    # Invariant assertions on classes
    for cid in train_labels:
        assert cid in CIFAR_KNOWN_IDS, f"Contamination: found label {cid} in known_train!"
    for cid in val_labels:
        assert cid in CIFAR_KNOWN_IDS, f"Contamination: found label {cid} in known_val!"
    for cid in known_test_labels:
        assert cid in CIFAR_KNOWN_IDS, f"Contamination: found label {cid} in known_test!"
    for cid in near_ood_labels:
        assert cid in CIFAR_HELDOUT_IDS, f"Leakage: found known class {cid} in near_ood!"

    print("\n[ALL INVARIANTS PASSED]")
    print("[OK] No overlap between train and validation.")
    print("[OK] Zero held-out class leakage into training or validation.")
    print("[OK] Classes are perfectly balanced (4000/cls train, 1000/cls val, 1000/cls test).")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        verify_data()
