"""Evaluation metrics module for Out-of-Distribution detection and calibration.

Calculates:
- AUROC (Area under ROC curve with OOD as positive class)
- FPR@95%TPR (False Positive Rate at 95% True Positive Rate threshold)
- ECE (Expected Calibration Error with 15 equal-width bins)
- Risk-Coverage curve and AURC
- Per-class AUROC (attractor analysis)
- OOD-to-known confusion (falsely accepted failure breakdown)
"""

import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(score_in: np.ndarray, score_ood: np.ndarray) -> float:
    """Computes Area Under the ROC Curve with OOD as the positive class.

    Hard rule: score_in and score_ood must follow HIGHER = MORE OOD.
    """
    assert score_in.ndim == 1 and len(score_in) > 0, "score_in must be non-empty 1D array"
    assert score_ood.ndim == 1 and len(score_ood) > 0, "score_ood must be non-empty 1D array"

    y_true = np.concatenate([np.zeros(len(score_in), dtype=int), np.ones(len(score_ood), dtype=int)])
    y_scores = np.concatenate([score_in, score_ood])

    return float(roc_auc_score(y_true, y_scores))


def fpr_at_tpr(score_in: np.ndarray, score_ood: np.ndarray, target_tpr: float = 0.95) -> tuple[float, float]:
    """Calculates False Positive Rate at specified True Positive Rate (e.g. 95%).

    Returns:
        threshold: score threshold achieving >= target_tpr on OOD data
        fpr: fraction of in-distribution inputs falsely flagged as OOD (score >= threshold)
    """
    y_true = np.concatenate([np.zeros(len(score_in), dtype=int), np.ones(len(score_ood), dtype=int)])
    y_scores = np.concatenate([score_in, score_ood])

    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    # First index where TPR reaches or exceeds target_tpr
    idx = np.where(tpr_curve >= target_tpr)[0][0]
    threshold = float(thresholds[idx])
    fpr = float(fpr_curve[idx])
    actual_tpr = float(tpr_curve[idx])

    assert actual_tpr >= (target_tpr - 0.005), f"Resulting TPR {actual_tpr} is below target {target_tpr}"
    return threshold, fpr


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error with equal-width probability bins.

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    assert len(y_true) == len(probs), "Labels and probabilities must have same length"
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accuracies = (preds == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confs > low) & (confs <= high) if i > 0 else (confs >= low) & (confs <= high)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confs[mask])
            ece_val += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return float(ece_val)


def risk_coverage(
    confidence: np.ndarray,
    correct: np.ndarray,
    grid: np.ndarray = np.arange(0.01, 1.001, 0.01),
) -> dict:
    """Computes selective classification accuracy across descending confidence coverage steps.

    Args:
        confidence: 1D array of prediction confidence scores
        correct: 1D boolean array indicating whether prediction matched ground truth
        grid: coverage fractions (e.g. 0.01 to 1.00)

    Returns:
        dict with coverage, accuracy, risk, aurc, and coverage_at_90acc
    """
    sort_idx = np.argsort(-confidence)
    sorted_correct = correct[sort_idx]
    n_total = len(confidence)

    coverages = []
    accuracies = []
    risks = []

    for cov in grid:
        k = max(1, int(np.round(cov * n_total)))
        acc = float(np.mean(sorted_correct[:k]))
        risk = 1.0 - acc
        coverages.append(float(cov))
        accuracies.append(acc)
        risks.append(risk)

    coverages = np.array(coverages)
    accuracies = np.array(accuracies)
    risks = np.array(risks)

    # Area Under Risk-Coverage Curve (AURC) via trapezoidal integration
    aurc = float(trapezoid(risks, coverages))

    # Largest coverage with accuracy >= 0.90
    valid_covs = coverages[accuracies >= 0.90]
    coverage_at_90acc = float(np.max(valid_covs)) if len(valid_covs) > 0 else 0.0

    return {
        "coverage": coverages.tolist(),
        "accuracy": accuracies.tolist(),
        "risk": risks.tolist(),
        "aurc": aurc,
        "coverage_at_90acc": coverage_at_90acc,
    }


def per_class_auroc(
    score_id: np.ndarray,
    y_id: np.ndarray,
    score_ood: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Computes AUROC for each known class individually against the OOD dataset.

    Identifies which known class is most vulnerable to confusing OOD inputs.
    """
    results = {}
    unique_classes = sorted(list(set(y_id)))
    for c_idx, c_name in enumerate(class_names):
        c_id = unique_classes[c_idx]
        mask_c = y_id == c_id
        score_c = score_id[mask_c]
        auc_c = auroc(score_c, score_ood)
        results[c_name] = round(auc_c, 4)
    return results


def ood_to_known_confusion(
    score_ood: np.ndarray,
    pred_ood: np.ndarray,
    score_in: np.ndarray,
    class_names: list[str],
    target_tpr: float = 0.95,
) -> dict[str, int]:
    """Histograms predicted classes for OOD inputs that were falsely accepted at target TPR."""
    threshold, _ = fpr_at_tpr(score_in, score_ood, target_tpr=target_tpr)
    # Falsely accepted OOD inputs have score < threshold (i.e. deemed familiar)
    accepted_mask = score_ood < threshold
    accepted_preds = pred_ood[accepted_mask]

    hist = {name: 0 for name in class_names}
    for p in accepted_preds:
        hist[class_names[p]] += 1

    return hist
