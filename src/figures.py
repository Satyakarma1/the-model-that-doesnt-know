"""Figure generation module for OOD study.

Generates 4 publication-quality figures:
1. figures/score_distributions.png (1x3 panels for MSP, Energy, Mahalanobis)
2. figures/risk_coverage.png (Selective classification curves)
3. figures/per_class_auroc.png (Attractor analysis per known class)
4. figures/failure_gallery.png (5x5 grid of falsely accepted OOD images)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless rendering for headless/server/Windows scripts
import matplotlib.pyplot as plt
import numpy as np

from src.config import CIFAR_KNOWN_NAMES, PATHS
from src.data import load_image_at
from src.evaluate import fpr_at_tpr, risk_coverage


def plot_score_distributions(
    scores_dict: dict[str, dict[str, np.ndarray]],
    save_path: Path = PATHS["figures"] / "score_distributions.png",
) -> None:
    """Plots 1x3 panel of score distributions for MSP, Energy, and Mahalanobis."""
    methods = ["MSP", "Energy", "Mahalanobis"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, method in zip(axes, methods):
        score_id = scores_dict[method]["known_test"]
        score_near = scores_dict[method]["near_ood"]
        score_far = scores_dict[method]["far_ood"]

        # Combined min and max for shared bins
        all_s = np.concatenate([score_id, score_near, score_far])
        bins = np.linspace(np.percentile(all_s, 0.5), np.percentile(all_s, 99.5), 50)

        # Plot density histograms
        ax.hist(score_id, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="Known Test (ID)")
        ax.hist(score_near, bins=bins, density=True, alpha=0.55, color="#ff7f0e", label="Near-OOD (CIFAR Held-out)")
        ax.hist(score_far, bins=bins, density=True, alpha=0.55, color="#9467bd", label="Far-OOD (FashionMNIST)")

        # Vertical dashed line at near-OOD 95% TPR threshold
        thr_near, _ = fpr_at_tpr(score_id, score_near, target_tpr=0.95)
        ax.axvline(thr_near, color="red", linestyle="--", linewidth=1.8, label=f"Near-OOD 95% TPR Thr ({thr_near:.2f})")

        ax.set_title(f"Score Distribution: {method}", fontsize=13, fontweight="bold")
        ax.set_xlabel("OOD Score (Higher = More OOD)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size / 1024:.1f} KB)")


def plot_risk_coverage(
    risk_cov_data: dict[str, dict],
    baseline_acc: float,
    save_path: Path = PATHS["figures"] / "risk_coverage.png",
) -> None:
    """Plots selective classification accuracy vs coverage for all methods."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"MSP": "#1f77b4", "Energy": "#2ca02c", "Mahalanobis": "#d62728"}
    for method, data in risk_cov_data.items():
        cov = data["coverage"]
        acc = [a * 100 for a in data["accuracy"]]
        aurc = data["aurc"]
        cov90 = data["coverage_at_90acc"] * 100
        ax.plot(
            cov,
            acc,
            label=f"{method} (AURC: {aurc:.3f}, Cov@90%Acc: {cov90:.1f}%)",
            color=colors.get(method, "blue"),
            linewidth=2.2,
        )

    # Baseline line: accept everything
    ax.axhline(baseline_acc * 100, color="gray", linestyle=":", linewidth=1.8, label=f"Baseline (All Accepted): {baseline_acc*100:.1f}%")

    ax.set_title("Risk-Coverage Curves: Selective Classification", fontsize=14, fontweight="bold")
    ax.set_xlabel("Coverage (Fraction of Accepted Predictions)", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%) on Accepted Inputs", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(min(baseline_acc * 100 - 5, 80), 101)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size / 1024:.1f} KB)")


def plot_per_class_auroc(
    per_class_data: dict[str, dict[str, float]],
    save_path: Path = PATHS["figures"] / "per_class_auroc.png",
) -> None:
    """Grouped bar chart showing AUROC per known class for Near vs Far OOD."""
    classes = CIFAR_KNOWN_NAMES
    near_scores = [per_class_data["near_ood"][c] for c in classes]
    far_scores = [per_class_data["far_ood"][c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width / 2, near_scores, width, label="Near-OOD (CIFAR Held-out)", color="#ff7f0e", alpha=0.85)
    rects2 = ax.bar(x + width / 2, far_scores, width, label="Far-OOD (FashionMNIST)", color="#9467bd", alpha=0.85)

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Per-Class AUROC: Attractor Analysis across Known Classes", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Label heights
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size / 1024:.1f} KB)")


def plot_failure_gallery(
    accepted_samples: list[dict],
    save_path: Path = PATHS["figures"] / "failure_gallery.png",
) -> None:
    """Plots a 5x5 grid of OOD inputs falsely accepted by the detector at 95% TPR.

    Border color: Orange for Near-OOD, Purple for Far-OOD.
    """
    assert len(accepted_samples) >= 25, f"Need at least 25 accepted samples for 5x5 gallery, got {len(accepted_samples)}"
    samples_to_show = accepted_samples[:25]

    fig, axes = plt.subplots(5, 5, figsize=(12, 13))

    for idx, (ax, sample) in enumerate(zip(axes.flat, samples_to_show)):
        img = load_image_at(sample["split"], sample["dataset_idx"])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        pred_name = sample["pred_name"]
        conf = sample["conf"]
        split_type = sample["split"]

        color = "#ff7f0e" if split_type == "near_ood" else "#9467bd"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)

        ax.set_title(f"pred: {pred_name}\nconf: {conf:.2f}", fontsize=9, fontweight="bold", pad=2)

    # Custom legend for border colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="#ff7f0e", linewidth=2.5, label="Near-OOD (CIFAR Held-out)"),
        Patch(facecolor="white", edgecolor="#9467bd", linewidth=2.5, label="Far-OOD (FashionMNIST)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=11)

    plt.subplots_adjust(top=0.93, bottom=0.03, left=0.05, right=0.95, hspace=0.35, wspace=0.15)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size / 1024:.1f} KB)")
