"""Linear probe classifier and temperature scaling calibration module.

Trains a multinomial logistic regression head on frozen known_train features.
Performs cross-validated C selection and temperature fitting on known_val only.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.config import C_GRID, CIFAR_KNOWN_NAMES, SEED, TEMP_BOUNDS, set_seed


def select_C(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Selects best C parameter via 3-fold stratified cross-validation on training set."""
    set_seed(SEED)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_c = C_GRID[0]
    best_score = -1.0

    print("  [C Selection] Running 3-fold CV over C grid:", C_GRID)
    for c in C_GRID:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            clf = LogisticRegression(
                C=c,
                max_iter=2000,
                solver="lbfgs",
                random_state=SEED,
            )
            clf.fit(X_train[train_idx], y_train[train_idx])
            pred = clf.predict(X_train[val_idx])
            scores.append(accuracy_score(y_train[val_idx], pred))
        mean_score = float(np.mean(scores))
        print(f"    C = {c:<6} -> CV Mean Accuracy: {mean_score*100:.2f}%")
        if mean_score > best_score:
            best_score = mean_score
            best_c = c

    print(f"  Selected best C = {best_c} (CV Accuracy: {best_score*100:.2f}%)")
    return best_c


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, C: float) -> LogisticRegression:
    """Fits multinomial logistic regression classifier on standardized features."""
    set_seed(SEED)
    clf = LogisticRegression(
        C=C,
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def get_logits(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Computes raw pre-softmax logits from linear weights and intercepts.

    Returns:
        logits: (N, K) float32
    """
    logits = X @ clf.coef_.T + clf.intercept_
    return logits.astype(np.float32)


def get_proba(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Computes softmax probabilities under temperature T."""
    scaled = logits / max(T, 1e-4)
    return softmax(scaled, axis=1)


def nll_criterion(T_arr: np.ndarray, logits: np.ndarray, targets: np.ndarray) -> float:
    """Computes mean negative log-likelihood (cross entropy) at temperature T."""
    T = T_arr[0]
    scaled_logits = logits / T
    # Log-softmax: z_i - logsumexp(z)
    log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
    nll = -log_probs[np.arange(len(targets)), targets]
    return float(np.mean(nll))


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float]:
    """Fits temperature scalar T by minimizing NLL on known_val logits strictly.

    Returns:
        (best_T, nll_before, nll_after)
    """
    nll_before = nll_criterion(np.array([1.0]), logits_val, y_val)

    res = minimize(
        nll_criterion,
        x0=[1.0],
        args=(logits_val, y_val),
        method="L-BFGS-B",
        bounds=[TEMP_BOUNDS],
    )
    best_T = float(res.x[0])
    nll_after = float(res.fun)

    print(f"  [TEMPERATURE FIT] Optimal T = {best_T:.4f} (NLL: {nll_before:.4f} -> {nll_after:.4f})")
    return best_T, nll_before, nll_after


def evaluate_closed_set(clf: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluates probe accuracy and returns accuracy and confusion matrix on known_test."""
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=sorted(list(set(y_test))))
    return acc, cm


if __name__ == "__main__":
    from src.features import build_backbone, extract_and_scale_all_splits

    print("=" * 65)
    print(" [GATE 1] FEATURE EXTRACTION & LINEAR PROBE EVALUATION")
    print("=" * 65)

    backbone = build_backbone()
    scaled_data = extract_and_scale_all_splits(backbone)

    X_train = scaled_data["known_train"]["f"]
    y_train = scaled_data["known_train"]["y"]
    X_val = scaled_data["known_val"]["f"]
    y_val = scaled_data["known_val"]["y"]
    X_test = scaled_data["known_test"]["f"]
    y_test = scaled_data["known_test"]["y"]

    # Select C
    best_c = select_C(X_train, y_train)

    # Fit probe
    print(f"\nFitting final probe on known_train (N={len(X_train)}) with C={best_c}...")
    clf = fit_probe(X_train, y_train, best_c)

    # Evaluate validation and test accuracy
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    test_acc, cm = evaluate_closed_set(clf, X_test, y_test)

    print("\n" + "=" * 65)
    print(f"Known Validation Accuracy : {val_acc*100:.2f}%")
    print(f"Known Test Accuracy       : {test_acc*100:.2f}%")
    print("=" * 65)

    print("\n5x5 Confusion Matrix on known_test:")
    header = "       " + "".join([f"{name[:6]:>8}" for name in CIFAR_KNOWN_NAMES])
    print(header)
    for i, name in enumerate(CIFAR_KNOWN_NAMES):
        row_str = f"{name[:6]:<6} " + "".join([f"{cm[i, j]:>8}" for j in range(len(CIFAR_KNOWN_NAMES))])
        print(row_str)

    # Sanity checks
    assert 0.88 <= test_acc <= 0.98, f"Accuracy {test_acc} outside expected sanity band [0.88, 0.98]!"
    print("\n[OK] Closed-set test accuracy lands firmly within sanity band (88% - 98%).")
    print("[OK] Gate 1 passed successfully.")
    print("=" * 65)

