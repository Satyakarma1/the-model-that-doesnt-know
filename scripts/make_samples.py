"""Generates sample gallery images and manifest.json for the Streamlit dashboard.

Ensures that every sample image is correctly indexed, scaled 4x nearest-neighbor,
and annotated with its ground truth class, dataset split, and source index.
"""

import json
from pathlib import Path
import random
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from torchvision import datasets

from src.config import (
    CIFAR_HELDOUT_IDS,
    CIFAR_HELDOUT_NAMES,
    CIFAR_KNOWN_IDS,
    CIFAR_KNOWN_NAMES,
    PATHS,
)
from src.data import make_datasets


def make_samples():
    samples_dir = PATHS["figures"] / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Delete every pre-existing sample png and old manifest
    for old_file in samples_dir.glob("*.png"):
        old_file.unlink()
    if (samples_dir / "manifest.json").exists():
        (samples_dir / "manifest.json").unlink()

    print("[SAMPLES] Cleared old sample files.")

    # 2. Load splits and raw datasets without transformation
    splits = make_datasets()
    data_dir = PATHS["data"]
    raw_cifar_test = datasets.CIFAR10(root=str(data_dir), train=False, download=False)
    raw_fashion_test = datasets.FashionMNIST(root=str(data_dir), train=False, download=False)

    rng = random.Random(1337)
    manifest = []

    # Map class IDs to names
    cifar_all_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    fashion_classes = {
        0: "t-shirt",
        9: "ankle boot",
    }

    # 3. Known classes (from known_test)
    known_test_ds = splits["known_test"].dataset  # Subset
    for cid in CIFAR_KNOWN_IDS:
        cname = cifar_all_names[cid]
        # Find all indices in known_test matching this class
        matching_split_indices = [
            i for i, src_idx in enumerate(known_test_ds.indices)
            if raw_cifar_test.targets[src_idx] == cid
        ]
        # Deterministically sample 3 indices
        chosen = rng.sample(matching_split_indices, 3)
        for k, split_idx in enumerate(chosen):
            src_idx = int(known_test_ds.indices[split_idx])
            raw_arr = raw_cifar_test.data[src_idx]  # 32x32x3 uint8
            img = Image.fromarray(raw_arr)
            img_upscaled = img.resize((128, 128), Image.NEAREST)
            fname = f"sample_known_{cname}_{k}.png"
            img_upscaled.save(samples_dir / fname)

            manifest.append({
                "file": fname,
                "true_class": cname,
                "true_id": cid,
                "role": "known",
                "split": "known_test",
                "split_index": int(split_idx),
                "source_dataset_index": src_idx,
            })

    # 4. Near-OOD classes (from near_ood)
    near_ood_ds = splits["near_ood"].dataset  # Subset
    for cid in CIFAR_HELDOUT_IDS:
        cname = cifar_all_names[cid]
        matching_split_indices = [
            i for i, src_idx in enumerate(near_ood_ds.indices)
            if raw_cifar_test.targets[src_idx] == cid
        ]
        chosen = rng.sample(matching_split_indices, 3)
        for k, split_idx in enumerate(chosen):
            src_idx = int(near_ood_ds.indices[split_idx])
            raw_arr = raw_cifar_test.data[src_idx]  # 32x32x3 uint8
            img = Image.fromarray(raw_arr)
            img_upscaled = img.resize((128, 128), Image.NEAREST)
            fname = f"sample_near_{cname}_{k}.png"
            img_upscaled.save(samples_dir / fname)

            manifest.append({
                "file": fname,
                "true_class": cname,
                "true_id": cid,
                "role": "near",
                "split": "near_ood",
                "split_index": int(split_idx),
                "source_dataset_index": src_idx,
            })

    # 5. Far-OOD classes (from FashionMNIST test)
    # raw_fashion_test.data is (10000, 28, 28), targets is tensor
    fashion_targets = np.array(raw_fashion_test.targets)
    for fid, fname_label in fashion_classes.items():
        matching_indices = np.where(fashion_targets == fid)[0].tolist()
        chosen = rng.sample(matching_indices, 3)
        clean_name = fname_label.replace(" ", "_").replace("-", "_")
        for k, src_idx in enumerate(chosen):
            raw_arr = raw_fashion_test.data[src_idx].numpy()  # 28x28 uint8
            img_rgb = np.stack([raw_arr] * 3, axis=-1)
            img = Image.fromarray(img_rgb)
            img_upscaled = img.resize((112, 112), Image.NEAREST)
            fname = f"sample_far_{clean_name}_{k}.png"
            img_upscaled.save(samples_dir / fname)

            manifest.append({
                "file": fname,
                "true_class": fname_label,
                "true_id": fid,
                "role": "far",
                "split": "far_ood",
                "split_index": int(src_idx),
                "source_dataset_index": int(src_idx),
            })

    # 6. Save manifest.json
    manifest_path = samples_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SAMPLES] Generated {len(manifest)} verified samples in {samples_dir}")
    print(f"{'Filename':<32} | {'Role':<6} | {'True Class':<12} | {'Split':<10} | {'Src Idx':<8}")
    print("-" * 78)
    for item in manifest:
        print(f"{item['file']:<32} | {item['role']:<6} | {item['true_class']:<12} | {item['split']:<10} | {item['source_dataset_index']:<8}")

    return manifest


if __name__ == "__main__":
    make_samples()
