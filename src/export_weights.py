"""Exports deployment artifacts: results/model_weights.npz and results/scores.npz.

Decouples online inference and threshold exploration on Streamlit Cloud from local
cache/ and data/ directories.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import CIFAR_KNOWN_NAMES, PATHS, TARGET_TPR
from src.evaluate import fpr_at_tpr
from src.probe import fit_probe, fit_temperature, get_logits
from src.scores import (
    energy_score,
    fit_gaussian_stats,
    mahalanobis_score,
    msp_score,
)


def export():
    cache_dir = PATHS["cache"]
    results_dir = PATHS["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[EXPORT] Loading cached features...")
    scaler = np.load(cache_dir / "scaler.npz")
    mean, std = scaler["mean"], scaler["std"]

    train_data = np.load(cache_dir / "features_known_train.npz")
    val_data = np.load(cache_dir / "features_known_val.npz")
    test_data = np.load(cache_dir / "features_known_test.npz")
    near_data = np.load(cache_dir / "features_near_ood.npz")
    far_data = np.load(cache_dir / "features_far_ood.npz")

    X_train = (train_data["f"] - mean) / std
    y_train = train_data["y"]
    X_val = (val_data["f"] - mean) / std
    y_val = val_data["y"]
    X_test = (test_data["f"] - mean) / std
    X_near = (near_data["f"] - mean) / std
    X_far = (far_data["f"] - mean) / std

    print("[EXPORT] Fitting linear probe and temperature scaling...")
    clf = fit_probe(X_train, y_train, C=0.01)
    val_logits = get_logits(clf, X_val)
    T, _, _ = fit_temperature(val_logits, y_val)
    means, cov = fit_gaussian_stats(X_train, y_train, n_classes=len(CIFAR_KNOWN_NAMES))

    print("[EXPORT] Computing detector scores across all splits...")
    logits_test = get_logits(clf, X_test)
    logits_near = get_logits(clf, X_near)
    logits_far = get_logits(clf, X_far)

    # 1. MSP scores
    msp_known = msp_score(logits_test).astype(np.float32)
    msp_near = msp_score(logits_near).astype(np.float32)
    msp_far = msp_score(logits_far).astype(np.float32)

    # 2. Energy scores
    energy_known = energy_score(logits_test, T=T).astype(np.float32)
    energy_near = energy_score(logits_near, T=T).astype(np.float32)
    energy_far = energy_score(logits_far, T=T).astype(np.float32)

    # 3. Mahalanobis scores
    maha_known = mahalanobis_score(X_test, means, cov).astype(np.float32)
    maha_near = mahalanobis_score(X_near, means, cov).astype(np.float32)
    maha_far = mahalanobis_score(X_far, means, cov).astype(np.float32)

    # Thresholds for Energy detector at 95% TPR
    thr_near_95, near_fpr = fpr_at_tpr(energy_known, energy_near, target_tpr=TARGET_TPR)
    thr_far_95, far_fpr = fpr_at_tpr(energy_known, energy_far, target_tpr=TARGET_TPR)

    # Section 1.3: Derive deployment threshold
    # Largest Energy threshold whose false-rejection rate on known_test is <= 10%
    sorted_known = np.sort(energy_known)
    thr_deploy = float(sorted_known[int(0.90 * len(sorted_known))])

    # Compute achieved detection rates (TPR) at thr_deploy
    # Since higher score = more OOD, OOD sample is caught if score >= thr
    tpr_near_at_deploy = float(np.mean(energy_near >= thr_deploy))
    tpr_far_at_deploy = float(np.mean(energy_far >= thr_deploy))
    frr_known_at_deploy = float(np.mean(energy_known >= thr_deploy))

    print(f"\n--- DERIVED DEPLOYMENT THRESHOLD ---")
    print(f"thr_deploy: {thr_deploy:.4f}")
    print(f"Known false rejection rate: {frr_known_at_deploy*100:.2f}% (<= 10.00%)")
    print(f"Near-OOD detected: {tpr_near_at_deploy*100:.2f}%")
    print(f"Far-OOD detected:  {tpr_far_at_deploy*100:.2f}%")
    print(f"Academic 95% TPR threshold (thr_near_95): {thr_near_95:.4f} (Known rejection: {near_fpr*100:.2f}%)")

    # Save results/model_weights.npz
    weights_path = results_dir / "model_weights.npz"
    np.savez_compressed(
        weights_path,
        coef=clf.coef_,
        intercept=clf.intercept_,
        scaler_mean=mean,
        scaler_std=std,
        temperature=np.array([T]),
        gaussian_means=means,
        gaussian_cov=cov,
        thr_near_95=np.array([thr_near_95]),
        thr_far_95=np.array([thr_far_95]),
        thr_deploy=np.array([thr_deploy]),
        thr_balanced=np.array([thr_deploy]),  # Alias for backwards compatibility
    )
    print(f"[EXPORT] Saved {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")

    # Save results/scores.npz for the interactive Threshold Explorer
    scores_path = results_dir / "scores.npz"
    np.savez_compressed(
        scores_path,
        MSP_known=msp_known,
        MSP_near=msp_near,
        MSP_far=msp_far,
        Energy_known=energy_known,
        Energy_near=energy_near,
        Energy_far=energy_far,
        Mahalanobis_known=maha_known,
        Mahalanobis_near=maha_near,
        Mahalanobis_far=maha_far,
    )
    print(f"[EXPORT] Saved {scores_path} ({scores_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    export()
