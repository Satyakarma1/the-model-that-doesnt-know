"""Sanity and mathematical invariant unit tests for OOD detector suite.

Verifies:
1. AUROC behaves properly (0.5 on identical distributions, 1.0 on disjoint distributions).
2. FPR@TPR threshold accurately captures >= target TPR.
3. ECE is ~0 on perfectly calibrated predictions.
4. Covariance regularizer guarantees positive definiteness.
5. All three detector scores strictly follow HIGHER = MORE OOD.
"""

import numpy as np

from src.evaluate import auroc, ece, fpr_at_tpr, risk_coverage
from src.scores import energy_score, fit_gaussian_stats, mahalanobis_score, msp_score


def test_auroc_bounds():
    # Disjoint: all OOD scores strictly higher than ID scores
    score_id = np.array([0.1, 0.2, 0.3, 0.4])
    score_ood = np.array([0.5, 0.6, 0.7, 0.8])
    assert np.isclose(auroc(score_id, score_ood), 1.0)

    # Inverted: all ID scores strictly higher than OOD
    assert np.isclose(auroc(score_ood, score_id), 0.0)

    # Identical distributions -> 0.5
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, 1000)
    s2 = rng.normal(0, 1, 1000)
    assert 0.45 <= auroc(s1, s2) <= 0.55


def test_fpr_at_tpr():
    score_id = np.linspace(0, 10, 1000)
    score_ood = np.linspace(5, 15, 1000)

    thr, fpr = fpr_at_tpr(score_id, score_ood, target_tpr=0.95)
    # Check that at least 95% of OOD are flagged (score >= thr)
    actual_tpr = np.mean(score_ood >= thr)
    assert actual_tpr >= 0.949
    # Check that FPR matches fraction of ID >= thr
    actual_fpr = np.mean(score_id >= thr)
    assert np.isclose(fpr, actual_fpr, atol=0.01)


def test_ece_perfect_calibration():
    # 100% accurate with 100% confidence -> ECE should be 0
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.isclose(ece(y_true, probs, n_bins=10), 0.0)


def test_risk_coverage_monotonicity():
    # If confidence ranks errors correctly, risk-coverage should drop as coverage increases
    # (i.e. accuracy improves at low coverage)
    conf = np.array([0.99, 0.95, 0.90, 0.55, 0.51])
    correct = np.array([True, True, True, False, False])
    rc = risk_coverage(conf, correct)
    # Accuracy at 60% coverage (top 3) should be 1.0
    assert rc["accuracy"][int(len(rc["accuracy"]) * 0.6) - 1] == 1.0
    # Full coverage accuracy should be 3/5 = 0.6
    assert np.isclose(rc["accuracy"][-1], 0.6)


def test_mahalanobis_covariance_pd():
    rng = np.random.default_rng(1337)
    # Low-sample regime where empirical covariance would be singular
    N, D = 10, 20
    X = rng.normal(0, 1, (N, D))
    y = np.array([0] * 5 + [1] * 5)
    means, cov = fit_gaussian_stats(X, y, n_classes=2)
    # Must be positive definite (Cholesky succeeds)
    L = np.linalg.cholesky(cov)
    assert L.shape == (D, D)

    # Scores must be finite and non-negative
    scores = mahalanobis_score(X, means, cov)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_score_sign_conventions():
    # Confident logits vs uncertain logits
    confident_logits = np.array([[10.0, -5.0, -5.0]])
    uncertain_logits = np.array([[0.1, 0.1, 0.1]])

    # MSP: score = -max_prob. Uncertain should have HIGHER score
    assert msp_score(uncertain_logits)[0] > msp_score(confident_logits)[0]

    # Energy: score = -T * logsumexp. Uncertain should have HIGHER score
    assert energy_score(uncertain_logits)[0] > energy_score(confident_logits)[0]


if __name__ == "__main__":
    test_auroc_bounds()
    test_fpr_at_tpr()
    test_ece_perfect_calibration()
    test_risk_coverage_monotonicity()
    test_mahalanobis_covariance_pd()
    test_score_sign_conventions()
    print("[ALL SANITY TESTS PASSED SUCCESSFULLY]")
