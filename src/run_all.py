"""Full pipeline orchestration module for OOD detection study.

Runs end-to-end evaluation:
- Loads cached features (or extracts deterministically)
- Fits linear probe and temperature scaling on known_val
- Computes MSP, Energy, and Mahalanobis detectors on Near-OOD and Far-OOD
- Generates metrics.json and summary.md
- Renders all 4 publication figures
"""

import json
from pathlib import Path
import sys
import time
import numpy as np

from src.config import (
    CIFAR_HELDOUT_NAMES,
    CIFAR_KNOWN_NAMES,
    DEVICE,
    PATHS,
    TARGET_TPR,
    set_seed,
)
from src.evaluate import (
    auroc,
    ece,
    fpr_at_tpr,
    ood_to_known_confusion,
    per_class_auroc,
    risk_coverage,
)
from src.features import build_backbone, extract_and_scale_all_splits
from src.figures import (
    plot_failure_gallery,
    plot_per_class_auroc,
    plot_risk_coverage,
    plot_score_distributions,
)
from src.probe import (
    evaluate_closed_set,
    fit_probe,
    fit_temperature,
    get_logits,
    get_proba,
    select_C,
)
from src.scores import (
    energy_score,
    fit_gaussian_stats,
    mahalanobis_score,
    msp_score,
)


def run(force_recompute: bool = False) -> dict:
    """Executes the full experimental pipeline end-to-end."""
    set_seed()
    start_time = time.time()

    print("=" * 75)
    print(" THE MODEL THAT DOESN'T KNOW: OUT-OF-DISTRIBUTION DETECTION STUDY")
    print(f" Compute Device: {DEVICE}")
    print("=" * 75)

    # 1. Feature Extraction & Scaling
    backbone = build_backbone(DEVICE)
    print("\n[STEP 1/5] Loading & Scaling Penultimate Features (512-d)...")
    scaled_data = extract_and_scale_all_splits(backbone, force=force_recompute)

    X_train = scaled_data["known_train"]["f"]
    y_train = scaled_data["known_train"]["y"]
    X_val = scaled_data["known_val"]["f"]
    y_val = scaled_data["known_val"]["y"]
    X_test = scaled_data["known_test"]["f"]
    y_test = scaled_data["known_test"]["y"]
    X_near = scaled_data["near_ood"]["f"]
    y_near = scaled_data["near_ood"]["y"]
    X_far = scaled_data["far_ood"]["f"]
    y_far = scaled_data["far_ood"]["y"]

    # 2. Linear Probe & Closed-Set Evaluation
    print("\n[STEP 2/5] Training Linear Classifier & Temperature Calibration...")
    best_c = select_C(X_train, y_train)
    clf = fit_probe(X_train, y_train, best_c)

    test_acc, cm = evaluate_closed_set(clf, X_test, y_test)
    val_preds = clf.predict(X_val)
    val_acc = float(np.mean(val_preds == y_val))
    print(f"  Closed-set Known Val Accuracy : {val_acc*100:.2f}%")
    print(f"  Closed-set Known Test Accuracy: {test_acc*100:.2f}%")

    # Fit temperature ONLY on known_val
    val_logits = get_logits(clf, X_val)
    optimal_T, nll_val_pre, nll_val_post = fit_temperature(val_logits, y_val)

    # 3. Logits and Probabilities across splits
    test_logits = get_logits(clf, X_test)
    near_logits = get_logits(clf, X_near)
    far_logits = get_logits(clf, X_far)

    # Uncalibrated vs Calibrated Probabilities on test
    test_probs_uncal = get_proba(test_logits, T=1.0)
    test_probs_cal = get_proba(test_logits, T=optimal_T)

    ece_pre = ece(y_test, test_probs_uncal, n_bins=15)
    ece_post = ece(y_test, test_probs_cal, n_bins=15)
    print(f"  Expected Calibration Error (ECE) [15 bins]: {ece_pre*100:.2f}% -> {ece_post*100:.2f}% (T={optimal_T:.3f})")

    # 4. Detector Scoring (MSP, Energy, Mahalanobis)
    print("\n[STEP 3/5] Computing Detector Scores (MSP, Energy, Mahalanobis)...")
    scores = {
        "MSP": {
            "known_test": msp_score(test_logits),
            "near_ood": msp_score(near_logits),
            "far_ood": msp_score(far_logits),
        },
        "Energy": {
            "known_test": energy_score(test_logits, T=optimal_T),
            "near_ood": energy_score(near_logits, T=optimal_T),
            "far_ood": energy_score(far_logits, T=optimal_T),
        },
    }

    # Mahalanobis fitted strictly on known_train
    means, cov = fit_gaussian_stats(X_train, y_train, n_classes=len(CIFAR_KNOWN_NAMES))
    scores["Mahalanobis"] = {
        "known_test": mahalanobis_score(X_test, means, cov),
        "near_ood": mahalanobis_score(X_near, means, cov),
        "far_ood": mahalanobis_score(X_far, means, cov),
    }

    # 5. Compute Metrics Table
    print("\n[STEP 4/5] Evaluating AUROC, FPR@95%TPR, Risk-Coverage, and Attractor Confusion...")
    metrics_list = []
    methods = ["MSP", "Energy", "Mahalanobis"]
    ood_sets = [("near_ood", "Near-OOD (CIFAR Held-out)"), ("far_ood", "Far-OOD (FashionMNIST)")]

    for method in methods:
        score_id = scores[method]["known_test"]
        for ood_key, ood_label in ood_sets:
            score_ood = scores[method][ood_key]
            auc = auroc(score_id, score_ood)
            thr, fpr = fpr_at_tpr(score_id, score_ood, target_tpr=TARGET_TPR)
            metrics_list.append({
                "method": method,
                "ood_set": ood_key,
                "ood_label": ood_label,
                "auroc": round(auc, 4),
                "fpr_at_95tpr": round(fpr, 4),
                "threshold": round(thr, 4),
            })

    # Risk-Coverage for selective prediction on known_test
    # Confidence defined as negative OOD score (higher confidence = more familiar)
    test_correct = np.argmax(test_logits, axis=1) == y_test
    risk_cov_data = {}
    for method in methods:
        # Higher score = more OOD, so confidence = -score
        conf = -scores[method]["known_test"]
        risk_cov_data[method] = risk_coverage(conf, test_correct)

    # Per-class AUROC (attractor analysis using Energy detector)
    per_class_near = per_class_auroc(scores["Energy"]["known_test"], y_test, scores["Energy"]["near_ood"], CIFAR_KNOWN_NAMES)
    per_class_far = per_class_auroc(scores["Energy"]["known_test"], y_test, scores["Energy"]["far_ood"], CIFAR_KNOWN_NAMES)
    per_class_data = {"near_ood": per_class_near, "far_ood": per_class_far}

    # OOD-to-known confusion (histogram of accepted predictions at 95% TPR)
    near_preds = np.argmax(near_logits, axis=1)
    confusion_hist = ood_to_known_confusion(
        scores["Energy"]["near_ood"],
        near_preds,
        scores["Energy"]["known_test"],
        CIFAR_KNOWN_NAMES,
        target_tpr=TARGET_TPR,
    )

    # Falsely accepted samples for the 5x5 gallery
    # Candidate pool: near-OOD and far-OOD samples whose score < 95% TPR threshold
    thr_energy_near, _ = fpr_at_tpr(scores["Energy"]["known_test"], scores["Energy"]["near_ood"], TARGET_TPR)
    near_accepted_mask = scores["Energy"]["near_ood"] < thr_energy_near
    far_accepted_mask = scores["Energy"]["far_ood"] < thr_energy_near

    accepted_samples = []
    # Add near-OOD accepted
    near_probs = get_proba(near_logits, T=optimal_T)
    for idx in np.where(near_accepted_mask)[0]:
        accepted_samples.append({
            "split": "near_ood",
            "dataset_idx": int(scaled_data["near_ood"]["idx"][idx]),
            "score": float(scores["Energy"]["near_ood"][idx]),
            "conf": float(np.max(near_probs[idx])),
            "pred_idx": int(near_preds[idx]),
            "pred_name": CIFAR_KNOWN_NAMES[near_preds[idx]],
        })

    # Add far-OOD accepted
    far_preds = np.argmax(far_logits, axis=1)
    far_probs = get_proba(far_logits, T=optimal_T)
    for idx in np.where(far_accepted_mask)[0]:
        accepted_samples.append({
            "split": "far_ood",
            "dataset_idx": int(scaled_data["far_ood"]["idx"][idx]),
            "score": float(scores["Energy"]["far_ood"][idx]),
            "conf": float(np.max(far_probs[idx])),
            "pred_idx": int(far_preds[idx]),
            "pred_name": CIFAR_KNOWN_NAMES[far_preds[idx]],
        })

    # Sort accepted samples descending by confidence (most confident errors first)
    accepted_samples.sort(key=lambda s: s["conf"], reverse=True)

    # 6. Generate Figures
    print("\n[STEP 5/5] Generating Visualizations...")
    plot_score_distributions(scores)
    plot_risk_coverage(risk_cov_data, baseline_acc=test_acc)
    plot_per_class_auroc(per_class_data)
    plot_failure_gallery(accepted_samples)

    # 7. Compile Final Results Object
    results = {
        "closed_set": {
            "validation_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "chosen_C": best_c,
            "confusion_matrix": cm.tolist(),
        },
        "calibration": {
            "optimal_temperature": round(optimal_T, 4),
            "ece_before_temp_scaling": round(ece_pre, 4),
            "ece_after_temp_scaling": round(ece_post, 4),
            "nll_val_pre": round(nll_val_pre, 4),
            "nll_val_post": round(nll_val_post, 4),
        },
        "metrics_table": metrics_list,
        "per_class_auroc": per_class_data,
        "ood_to_known_confusion": confusion_hist,
        "risk_coverage": {
            m: {
                "aurc": round(risk_cov_data[m]["aurc"], 4),
                "coverage_at_90acc": round(risk_cov_data[m]["coverage_at_90acc"], 4),
            }
            for m in methods
        },
        "timing": {
            "total_seconds": round(time.time() - start_time, 2),
        },
    }

    # Save metrics.json
    with open(PATHS["metrics_json"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured metrics -> {PATHS['metrics_json']}")

    # Save summary.md
    summary_text = build_summary_markdown(results)
    with open(PATHS["summary_md"], "w") as f:
        f.write(summary_text)
    print(f"Saved human-readable summary -> {PATHS['summary_md']}")

    # Print ASCII Table to stdout
    print_ascii_report(results)

    return results


def build_summary_markdown(results: dict) -> str:
    """Builds clean Markdown report with GitHub-flavored markdown tables."""
    lines = [
        "# OOD Detection Study: Quantitative Results Summary",
        "",
        "## 1. Closed-Set Classification Performance",
        f"- **Known Test Accuracy**: {results['closed_set']['test_accuracy']*100:.2f}%",
        f"- **Validation Accuracy**: {results['closed_set']['validation_accuracy']*100:.2f}%",
        f"- **Best Regularization Parameter (C)**: {results['closed_set']['chosen_C']}",
        f"- **Expected Calibration Error (ECE)**: {results['calibration']['ece_before_temp_scaling']*100:.2f}% -> {results['calibration']['ece_after_temp_scaling']*100:.2f}% (T = {results['calibration']['optimal_temperature']:.3f})",
        "",
        "## 2. Out-of-Distribution Detection Benchmark",
        "| Detector Method | Evaluation Split | AUROC (Higher is Better) | FPR @ 95% TPR (Lower is Better) | Threshold |",
        "|---|---|:---:|:---:|:---:|",
    ]

    for item in results["metrics_table"]:
        lines.append(
            f"| **{item['method']}** | {item['ood_label']} | **{item['auroc']:.4f}** | **{item['fpr_at_95tpr']*100:.2f}%** | {item['threshold']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Selective Classification (Risk-Coverage & AURC)",
        "| Detector | AURC (Area Under Risk-Coverage) | Coverage at 90% Accuracy |",
        "|---|:---:|:---:|",
    ])
    for m, d in results["risk_coverage"].items():
        lines.append(f"| **{m}** | {d['aurc']:.4f} | {d['coverage_at_90acc']*100:.1f}% |")

    lines.extend([
        "",
        "## 4. Attractor Analysis: Per-Class AUROC (Energy Detector)",
        "| Known Class | Near-OOD AUROC | Far-OOD AUROC |",
        "|---|:---:|:---:|",
    ])
    for cls_name in CIFAR_KNOWN_NAMES:
        near_auc = results["per_class_auroc"]["near_ood"][cls_name]
        far_auc = results["per_class_auroc"]["far_ood"][cls_name]
        lines.append(f"| {cls_name.capitalize()} | {near_auc:.4f} | {far_auc:.4f} |")

    lines.extend([
        "",
        "## 5. Failure Analysis: OOD Misclassification at 95% TPR",
        "Distribution of predicted classes when unknown inputs slipped past the detector:",
    ])
    for cls_name, count in results["ood_to_known_confusion"].items():
        lines.append(f"- **{cls_name.capitalize()}**: {count} inputs")

    return "\n".join(lines)


def print_ascii_report(results: dict) -> None:
    """Prints clean terminal summary table."""
    print("\n" + "=" * 75)
    print("                      EXPERIMENTAL RESULTS REPORT")
    print("=" * 75)
    print(f"Closed-Set Test Accuracy : {results['closed_set']['test_accuracy']*100:.2f}%")
    print(f"ECE (Pre / Post Temp)    : {results['calibration']['ece_before_temp_scaling']*100:.2f}% -> {results['calibration']['ece_after_temp_scaling']*100:.2f}% (T={results['calibration']['optimal_temperature']:.3f})")
    print("-" * 75)
    print(f"{'Method':<14} | {'Evaluation Split':<26} | {'AUROC':<8} | {'FPR@95%':<8}")
    print("-" * 75)
    for r in results["metrics_table"]:
        print(f"{r['method']:<14} | {r['ood_label']:<26} | {r['auroc']:<8.4f} | {r['fpr_at_95tpr']*100:>6.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    force = "--recompute" in sys.argv
    run(force_recompute=force)
