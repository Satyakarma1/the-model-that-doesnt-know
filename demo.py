"""Interactive inference and OOD verification demo.

Usage:
    python demo.py --test-sample
    python demo.py path/to/image.png
"""

from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import CIFAR_KNOWN_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, PATHS
from src.features import build_backbone
from src.probe import fit_probe, fit_temperature, get_logits, get_proba
from src.scores import energy_score, fit_gaussian_stats, mahalanobis_score, msp_score


def load_model_and_stats():
    """Loads cached training features and fits probe, temperature, and gaussian stats."""
    train_cache = PATHS["cache"] / "features_known_train.npz"
    val_cache = PATHS["cache"] / "features_known_val.npz"
    scaler_cache = PATHS["cache"] / "scaler.npz"

    if not (train_cache.exists() and val_cache.exists() and scaler_cache.exists()):
        raise RuntimeError("Please run 'python -m src.run_all' first to generate cached features.")

    train_data = np.load(train_cache)
    val_data = np.load(val_cache)
    scaler = np.load(scaler_cache)

    mean, std = scaler["mean"], scaler["std"]
    X_train = (train_data["f"] - mean) / std
    y_train = train_data["y"]
    X_val = (val_data["f"] - mean) / std
    y_val = val_data["y"]

    # Fit probe with best C=0.01
    clf = fit_probe(X_train, y_train, C=0.01)

    # Fit temperature
    val_logits = get_logits(clf, X_val)
    T, _, _ = fit_temperature(val_logits, y_val)

    # Fit Mahalanobis
    means, cov = fit_gaussian_stats(X_train, y_train, n_classes=len(CIFAR_KNOWN_NAMES))

    # Threshold for Energy detector at 95% TPR
    test_cache = PATHS["cache"] / "features_known_test.npz"
    near_cache = PATHS["cache"] / "features_near_ood.npz"
    test_f = (np.load(test_cache)["f"] - mean) / std
    near_f = (np.load(near_cache)["f"] - mean) / std
    from src.evaluate import fpr_at_tpr
    score_id = energy_score(get_logits(clf, test_f), T=T)
    score_ood = energy_score(get_logits(clf, near_f), T=T)
    energy_thr, _ = fpr_at_tpr(score_id, score_ood, target_tpr=0.95)

    return clf, T, means, cov, mean, std, energy_thr


def predict_image(image_path_or_array, backbone, clf, T, means, cov, mean, std, energy_thr):
    """Predicts class and uncertainty verdict for a given image."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if isinstance(image_path_or_array, (str, Path)):
        img = Image.open(image_path_or_array).convert("RGB")
    else:
        img = Image.fromarray(image_path_or_array).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw_feat = backbone(tensor).cpu().numpy().astype(np.float32)

    feat = (raw_feat - mean) / std
    logits = get_logits(clf, feat)
    probs = get_proba(logits, T=T)[0]

    # Scores
    msp = float(msp_score(logits)[0])
    energy = float(energy_score(logits, T=T)[0])
    maha = float(mahalanobis_score(feat, means, cov)[0])

    top_idx = int(np.argmax(probs))
    top_pred = CIFAR_KNOWN_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    # Verdict based on Energy 95% TPR threshold
    is_ood = energy >= energy_thr
    verdict = "REJECT (Uncertain / OOD)" if is_ood else "ACCEPT (Familiar / Known)"

    print("\n" + "=" * 55)
    print("              OOD INFERENCE RESULT")
    print("=" * 55)
    print(f"Top Prediction    : {top_pred.upper()} (Confidence: {top_conf*100:.2f}%)")
    print(f"Energy OOD Score  : {energy:.4f} (Threshold @ 95% TPR: {energy_thr:.4f})")
    print(f"MSP OOD Score     : {msp:.4f}")
    print(f"Mahalanobis Dist  : {maha:.4f}")
    print("-" * 55)
    print(f"Decision Verdict  : {verdict}")
    print("=" * 55)

    return {
        "prediction": top_pred,
        "confidence": top_conf,
        "energy": energy,
        "is_ood": is_ood,
        "verdict": verdict,
    }


if __name__ == "__main__":
    backbone = build_backbone(DEVICE)
    clf, T, means, cov, mean, std, energy_thr = load_model_and_stats()

    if len(sys.argv) > 1 and sys.argv[1] != "--test-sample":
        img_path = Path(sys.argv[1])
        predict_image(img_path, backbone, clf, T, means, cov, mean, std, energy_thr)
    else:
        print("[DEMO] Running self-test on random sample from near_ood (CIFAR Dog)...")
        from src.data import load_image_at
        test_img = load_image_at("near_ood", 0)
        predict_image(test_img, backbone, clf, T, means, cov, mean, std, energy_thr)
