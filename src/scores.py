"""Out-of-Distribution scoring functions for MSP, Energy, and Mahalanobis detectors.

Standard Convention Enforced Everywhere:
    HIGHER SCORE = MORE OUT-OF-DISTRIBUTION (LESS CERTAIN).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, softmax

from src.config import RIDGE_LAMBDA


def msp_score(logits: np.ndarray) -> np.ndarray:
    """Computes Maximum Softmax Probability (MSP) OOD score.

    Mathematical definition:
        score(x) = - max_c [ softmax(logits)_c ]

    Sign convention:
        A familiar in-distribution input has high maximum probability (e.g. 0.98),
        yielding score = -0.98. An unfamiliar OOD input has low maximum probability
        (e.g. 0.25), yielding score = -0.25.
        Therefore, HIGHER SCORE = MORE OOD.
    """
    probs = softmax(logits, axis=1)
    max_p = np.max(probs, axis=1)
    return -max_p.astype(np.float64)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Computes Energy-based OOD score (Liu et al., 2020).

    Mathematical definition:
        E(x; T) = -T * logsumexp(logits / T, axis=1)

    Sign convention:
        In-distribution inputs produce large positive logits in known classes,
        resulting in high logsumexp, which produces a large negative energy (e.g. -12.5).
        OOD inputs produce smaller, flatter logits, resulting in lower logsumexp,
        producing a higher (less negative) energy (e.g. -3.2).
        Therefore, HIGHER SCORE = MORE OOD.
    """
    scaled = logits / max(T, 1e-4)
    lse = logsumexp(scaled, axis=1)
    energy = -T * lse
    return energy.astype(np.float64)


def fit_gaussian_stats(X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Fits class-conditional means and tied pooled within-class covariance.

    Strict invariant: Fitted strictly on known_train standardized features.
    Covariance regularized with RIDGE_LAMBDA * I to guarantee strict positive definiteness.

    Returns:
        means: (n_classes, D) float64
        cov: (D, D) float64
    """
    n_samples, feat_dim = X_train.shape
    unique_classes = sorted(list(set(y_train)))
    assert len(unique_classes) == n_classes, f"Expected {n_classes} classes, found {len(unique_classes)}"

    means = np.zeros((n_classes, feat_dim), dtype=np.float64)
    cov = np.zeros((feat_dim, feat_dim), dtype=np.float64)

    # 1. Compute empirical class means
    for c_idx, c in enumerate(unique_classes):
        X_c = X_train[y_train == c].astype(np.float64)
        means[c_idx] = np.mean(X_c, axis=0)

    # 2. Compute pooled within-class covariance
    for c_idx, c in enumerate(unique_classes):
        X_c = X_train[y_train == c].astype(np.float64)
        diff = X_c - means[c_idx]
        cov += diff.T @ diff

    cov = cov / max(n_samples - n_classes, 1)

    # 3. Add ridge regularization to diagonal
    cov += RIDGE_LAMBDA * np.eye(feat_dim, dtype=np.float64)

    # 4. Verify Positive Definiteness via Cholesky decomposition
    try:
        _ = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as err:
        raise ValueError("Covariance matrix is not positive definite after ridge!") from err

    return means, cov


def mahalanobis_score(X: np.ndarray, means: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Computes Mahalanobis distance to nearest class centroid (Lee et al., 2018).

    Mathematical definition:
        score(x) = min_c sqrt( (x - mu_c)^T Sigma^-1 (x - mu_c) )

    Numerical method:
        Sigma^-1 (x - mu_c) is computed via Cholesky factor / solve (cho_factor/cho_solve).
        Direct matrix inversion (np.linalg.inv) is strictly avoided.

    Sign convention:
        In-distribution inputs lie close to their class centroid, yielding small distance.
        OOD inputs lie far from all known class centroids, yielding large distance.
        Therefore, HIGHER SCORE = MORE OOD.
    """
    n_samples = X.shape[0]
    n_classes = means.shape[0]
    c_factor = cho_factor(cov, lower=True)

    class_distances = np.zeros((n_samples, n_classes), dtype=np.float64)

    for c in range(n_classes):
        diff = X.astype(np.float64) - means[c]  # (N, D)
        # Solve Sigma * z = diff.T  -> z has shape (D, N)
        solved = cho_solve(c_factor, diff.T)  # (D, N)
        # Quadratic form: sum_j diff[i, j] * solved[j, i]
        sq_dist = np.sum(diff * solved.T, axis=1)
        # Numerical guard: clip negative rounding artifacts to zero
        sq_dist = np.clip(sq_dist, 0.0, None)
        class_distances[:, c] = np.sqrt(sq_dist)

    # OOD score is distance to closest known class
    min_dist = np.min(class_distances, axis=1)
    return min_dist
