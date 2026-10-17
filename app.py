"""The Model That Doesn't Know: Uncertainty-Aware Vision Classification.

Interactive Streamlit Web Dashboard:
- Curated sample gallery driven by ground-truth manifest
- Single-image inference playground & interactive calibrated probabilities
- "Where the threshold hurts" trade-off physical explorer with live ROC curve
- Diagnostic visualizations (score distributions, log-scale error-rate vs coverage)
- Quantitative benchmark tables with live metric loading from results/metrics.json
"""

import json
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import percentileofscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import streamlit as st
import torch
from torchvision import transforms

from src.config import (
    CIFAR_HELDOUT_NAMES,
    CIFAR_KNOWN_NAMES,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    PATHS,
    TARGET_TPR,
)
from src.features import build_backbone
from src.probe import get_logits, get_proba
from src.scores import energy_score, mahalanobis_score, msp_score

# -----------------------------------------------------------------------------
# Page Configuration & Modern Theme Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="The model that doesn't know",
    page_icon=":material/psychology_alt:",
    layout="wide",
)


@st.cache_data
def load_metrics(path_str: str) -> dict:
    """Loads benchmark metrics from results/metrics.json."""
    with open(path_str) as f:
        return json.load(f)


@st.cache_data
def load_scores(path_str: str) -> dict[str, np.ndarray]:
    """Loads precomputed detector score arrays from results/scores.npz."""
    data = np.load(path_str)
    return {k: data[k] for k in data.files}


@st.cache_data
def load_manifest(path_str: str) -> list[dict]:
    """Loads verified sample gallery manifest."""
    with open(path_str) as f:
        return json.load(f)


@st.cache_resource
def load_system():
    """Initializes backbone, classifier probe, and detector statistics."""
    backbone = build_backbone(DEVICE)
    weights_path = PATHS["results"] / "model_weights.npz"

    if weights_path.exists():
        weights = np.load(weights_path)
        clf = LogisticRegression()
        clf.classes_ = np.arange(len(CIFAR_KNOWN_NAMES))
        clf.coef_ = weights["coef"]
        clf.intercept_ = weights["intercept"]
        mean = weights["scaler_mean"]
        std = weights["scaler_std"]
        T = float(weights["temperature"][0])
        means = weights["gaussian_means"]
        cov = weights["gaussian_cov"]
        thr_near_95 = float(weights["thr_near_95"][0])
        thr_far_95 = float(weights["thr_far_95"][0])
        thr_deploy = float(weights["thr_deploy"][0]) if "thr_deploy" in weights else float(weights["thr_balanced"][0])
        return backbone, clf, T, means, cov, mean, std, thr_near_95, thr_far_95, thr_deploy

    # Fallback for local dev if weights not yet exported
    train_data = np.load(PATHS["cache"] / "features_known_train.npz")
    val_data = np.load(PATHS["cache"] / "features_known_val.npz")
    scaler = np.load(PATHS["cache"] / "scaler.npz")

    mean, std = scaler["mean"], scaler["std"]
    X_train = (train_data["f"] - mean) / std
    y_train = train_data["y"]
    X_val = (val_data["f"] - mean) / std
    y_val = val_data["y"]

    from src.probe import fit_probe, fit_temperature
    from src.scores import fit_gaussian_stats
    clf = fit_probe(X_train, y_train, C=0.01)
    val_logits = get_logits(clf, X_val)
    T, _, _ = fit_temperature(val_logits, y_val)
    means, cov = fit_gaussian_stats(X_train, y_train, n_classes=len(CIFAR_KNOWN_NAMES))

    test_f = (np.load(PATHS["cache"] / "features_known_test.npz")["f"] - mean) / std
    near_f = (np.load(PATHS["cache"] / "features_near_ood.npz")["f"] - mean) / std
    far_f = (np.load(PATHS["cache"] / "features_far_ood.npz")["f"] - mean) / std
    from src.evaluate import fpr_at_tpr
    score_id = energy_score(get_logits(clf, test_f), T=T)
    score_near = energy_score(get_logits(clf, near_f), T=T)
    score_far = energy_score(get_logits(clf, far_f), T=T)

    thr_near_95, _ = fpr_at_tpr(score_id, score_near, target_tpr=TARGET_TPR)
    thr_far_95, _ = fpr_at_tpr(score_id, score_far, target_tpr=TARGET_TPR)
    sorted_known = np.sort(score_id)
    thr_deploy = float(sorted_known[int(0.90 * len(sorted_known))])

    return backbone, clf, T, means, cov, mean, std, thr_near_95, thr_far_95, thr_deploy


def classify_and_filter(img: Image.Image, backbone, clf, T, means, cov, mean, std, active_thr):
    """Runs a single PIL image through feature extractor, probe, and uncertainty detectors."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    rgb_img = img.convert("RGB")
    tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw_feat = backbone(tensor).cpu().numpy().astype(np.float32)

    feat = (raw_feat - mean) / std
    logits = get_logits(clf, feat)
    probs = get_proba(logits, T=T)[0]

    msp = float(msp_score(logits)[0])
    energy = float(energy_score(logits, T=T)[0])
    maha = float(mahalanobis_score(feat, means, cov)[0])

    top_idx = int(np.argmax(probs))
    top_pred = CIFAR_KNOWN_NAMES[top_idx]
    top_conf = float(probs[top_idx])
    is_ood = energy >= active_thr

    return {
        "pred": top_pred,
        "conf": top_conf,
        "probs": probs,
        "msp": msp,
        "energy": energy,
        "maha": maha,
        "is_ood": is_ood,
    }


# -----------------------------------------------------------------------------
# Initialize System & Load Data
# -----------------------------------------------------------------------------
backbone, clf, T, means, cov, mean, std, thr_near_95, thr_far_95, thr_deploy = load_system()
metrics = load_metrics(str(PATHS["metrics_json"]))
scores = load_scores(str(PATHS["results"] / "scores.npz"))

# -----------------------------------------------------------------------------
# Sidebar Metadata (No Results)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### The model that doesn't know")
    st.caption("AI & Machine Learning Track · Promotional Recruitment")

    st.markdown("#### Known classes (5)")
    st.markdown("- Airplane\n- Automobile\n- Bird\n- Cat\n- Deer")

    st.markdown("#### Near-OOD held out (5)")
    st.caption("Semantic neighbors excluded from training:")
    st.markdown("- Dog *(neighbor of Cat)*\n- Frog *(neighbor of Bird)*\n- Horse *(neighbor of Deer)*\n- Ship *(neighbor of Airplane)*\n- Truck *(neighbor of Automobile)*")

    st.markdown("#### Far-OOD domain")
    st.caption("FashionMNIST test images upsampled to 3 channels (visually & semantically disjoint).")

# -----------------------------------------------------------------------------
# Hero Block & KPI Row
# -----------------------------------------------------------------------------
st.badge("Uncertainty-aware classification", icon=":material/radar:", color="violet")
st.title("The model that doesn't know")
st.markdown("### Five classes. Everything else is a maybe.")
st.caption(
    "A frozen ResNet-18 with a linear probe, scored by three independent uncertainty "
    "measures and evaluated on a deliberately hard near-OOD split and an easy far-OOD one."
)

# Extract headline numbers strictly from metrics.json
known_acc = metrics["closed_set"]["test_accuracy"] * 100

# Find best near-OOD AUROC and detector
near_entries = [m for m in metrics["metrics_table"] if m["ood_set"] == "near_ood"]
best_near = max(near_entries, key=lambda x: x["auroc"])
best_near_auroc = best_near["auroc"]
best_near_method = best_near["method"]

# Find best far-OOD AUROC and detector
far_entries = [m for m in metrics["metrics_table"] if m["ood_set"] == "far_ood"]
best_far = max(far_entries, key=lambda x: x["auroc"])
best_far_auroc = best_far["auroc"]
best_far_method = best_far["method"]

ece_val = metrics["calibration"]["ece_before_temp_scaling"] * 100

k1, k2, k3, k4 = st.columns(4)
with k1.container(border=True):
    st.metric("Known-set accuracy", f"{known_acc:.1f}%")
    st.caption("5 classes · 5,000 held-out CIFAR images")
with k2.container(border=True):
    st.metric("Near-OOD AUROC", f"{best_near_auroc:.4f}")
    st.caption(f"{best_near_method} · the hard regime")
with k3.container(border=True):
    st.metric("Far-OOD AUROC", f"{best_far_auroc:.4f}")
    st.caption(f"{best_far_method} · the easy regime")
with k4.container(border=True):
    st.metric("Calibration ECE", f"{ece_val:.2f}%")
    st.caption(f"T = {T:.3f} — already calibrated, scaling changes little")

st.markdown("")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Probe an image",
    "Where the threshold hurts",
    "Diagnostics",
    "Benchmark",
])

# -----------------------------------------------------------------------------
# TAB 1: Probe an image
# -----------------------------------------------------------------------------
with tab1:
    # Operating Threshold Controls
    with st.container(border=True):
        st.markdown("#### Operating threshold mode")
        t_mode_col, t_slider_col = st.columns([1.6, 1.4])

        with t_mode_col:
            operating_preset = st.radio(
                "Select operating threshold preset:",
                [
                    "Permissive Mode (-2.50) — allows familiar known images (e.g. Cat #0, Airplane #0) through",
                    f"Balanced Deployment Mode ({thr_deploy:.2f}) — derived guarantee of ≤ 10% known false rejections",
                    f"Strict Academic Near-OOD Benchmark ({thr_near_95:.2f} @ 95% TPR) — catches 95% of impostors",
                    "Custom Threshold Slider",
                ],
                index=0,
                key="tab1_threshold_preset",
            )

        with t_slider_col:
            if "Permissive" in operating_preset:
                active_thr = -2.50
                st.caption("Permissive mode sets threshold to **-2.50**, allowing clean familiar known-class images to pass.")
            elif "Balanced" in operating_preset:
                active_thr = float(thr_deploy)
                st.caption(f"Balanced mode sets threshold to **{thr_deploy:.2f}**, matching the derived 90% in-distribution acceptance guarantee.")
            elif "Strict" in operating_preset:
                active_thr = float(thr_near_95)
                st.caption(f"Strict mode sets threshold to **{thr_near_95:.2f}**, tuned to catch 95% of subtle Near-OOD impostors (e.g. dogs/horses).")
            else:
                active_thr = st.slider(
                    "Energy threshold (values above threshold are rejected):",
                    min_value=-12.0,
                    max_value=0.0,
                    value=float(thr_deploy),
                    step=0.25,
                    key="tab1_custom_slider",
                )
                st.caption(f"Active threshold: **{active_thr:.2f}**. Move slider right for more acceptance, left for stricter rejection.")

    source_choice = st.segmented_control(
        "Image source",
        options=["Curated samples", "Your own image"],
        default="Curated samples",
        label_visibility="collapsed",
        key="tab1_source_choice",
    )

    image_to_test = None
    sample_item = None
    manifest_items = []
    manifest_path = PATHS["figures"] / "samples" / "manifest.json"
    if manifest_path.exists():
        manifest_items = load_manifest(str(manifest_path))

    if source_choice == "Curated samples" and manifest_items:
        role_labels = {
            "known": "known class",
            "near": "held out (near-OOD)",
            "far": "held out (far-OOD)",
        }
        # Format options as TrueClass · role
        sample_labels = [
            f"{item['true_class'].title()} · {role_labels.get(item['role'], item['role'])} (Sample #{item['file'].split('_')[-1].split('.')[0]})"
            for item in manifest_items
        ]
        chosen_label = st.selectbox(
            "Select sample image",
            options=sample_labels,
            label_visibility="collapsed",
            key="tab1_sample_select",
        )
        chosen_idx = sample_labels.index(chosen_label)
        sample_item = manifest_items[chosen_idx]
        sample_img_file = PATHS["figures"] / "samples" / sample_item["file"]
        if sample_img_file.exists():
            image_to_test = Image.open(sample_img_file).convert("RGB")
    else:
        uploaded_file = st.file_uploader(
            "Upload image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="tab1_file_uploader",
        )
        if uploaded_file is not None:
            image_to_test = Image.open(uploaded_file).convert("RGB")

    if image_to_test is not None:
        c_left, c_right = st.columns([1, 1.6])

        with c_left:
            with st.container(border=True):
                st.image(image_to_test, width="stretch", alt="Input image under test")
                if sample_item:
                    st.caption(f"Ground truth: **{sample_item['true_class'].title()}** ({sample_item['role']} regime)")
                else:
                    st.caption("Custom image upload (Ground truth unlabelled)")

        with c_right:
            res = classify_and_filter(
                image_to_test, backbone, clf, T, means, cov, mean, std, active_thr
            )

            with st.container(border=True):
                if res["is_ood"]:
                    st.error("Rejected — the model does not know this", icon=":material/cancel:")
                    st.caption(f"Energy score {res['energy']:.2f} ≥ threshold {active_thr:.2f}")
                else:
                    st.success("Accepted as familiar", icon=":material/check_circle:")
                    st.caption(f"Energy score {res['energy']:.2f} < threshold {active_thr:.2f}")

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Prediction", res["pred"].capitalize())
                with m_col2:
                    st.metric("Calibrated confidence", f"{res['conf']*100:.1f}%")

                # Verdict diagnostic caption based on ground truth
                if sample_item:
                    role = sample_item["role"]
                    if role == "known" and not res["is_ood"]:
                        st.caption("Correct: known class, accepted.")
                    elif role == "known" and res["is_ood"]:
                        st.caption("False rejection: the model is familiar with this class but the threshold rejected it.")
                    elif role in ["near", "far"] and res["is_ood"]:
                        st.caption("Correct: outside the five known classes, rejected.")
                    else:
                        st.caption(
                            f"**Failure:** an unknown class was accepted as {res['pred'].capitalize()}. "
                            "This is the failure mode the study is about."
                        )
                else:
                    st.caption("Ground truth is unknown. Evaluated against active threshold.")

                # Multi-detector metric breakdown
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Energy OOD score", f"{res['energy']:.2f}", f"Threshold: {active_thr:.2f}")
                with m2:
                    st.metric("Softmax confidence", f"{res['conf']*100:.1f}%")
                with m3:
                    st.metric("Mahalanobis distance", f"{res['maha']:.1f}")

            # Technical Rationale Expander
            with st.expander("🔬 Understand this decision (Why did the model accept or reject this?)", expanded=False):
                st.markdown(
                    f"- **Classification Prediction**: The linear probe predicted **{res['pred'].upper()}** with **{res['conf']*100:.1f}% calibrated confidence**.\n"
                    f"- **Uncertainty Decision**: The energy score is **{res['energy']:.2f}** vs active threshold **{active_thr:.2f}**.\n"
                    f"- **Why can a high-confidence prediction be rejected?**\n"
                    f"  Softmax probabilities measure relative logit ratios (exp(z_i) / sum(exp(z))). Even for an unfamiliar input, if one logit is slightly higher, softmax can output 80–95% confidence. The Energy detector computes the absolute free energy: E = -T * log(sum(exp(z/T))). If overall activations are moderate (energy higher than {active_thr:.2f}), the detector intercepts and flags the sample as unfamiliar.\n"
                    f"- **Why can an impostor (e.g. horse or boot) produce low energy?**\n"
                    f"  Deep features occasionally project unfamiliar edge patterns into known feature clusters. Under permissive thresholds, some impostors slip past. Switching to the **Strict Academic Benchmark ({thr_near_95:.2f})** catches 95% of them, but at the cost of rejecting more legitimate in-distribution inputs."
                )

            # Altair Probability Bars
            prob_df = pd.DataFrame({
                "Class": [c.capitalize() for c in CIFAR_KNOWN_NAMES],
                "Probability": [float(p) for p in res["probs"]],
            })
            bar_chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#7C5CFF")
                .encode(
                    x=alt.X("Probability:Q", axis=alt.Axis(format="%", title="Calibrated probability"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Class:N", sort="-x", title=None),
                    tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=180, title="Calibrated class probabilities")
            )
            st.altair_chart(bar_chart, width="stretch")

            # Altair Score Ruler with clean formatting to prevent label overlap
            s_known = scores["Energy_known"]
            s_near = scores["Energy_near"]
            s_far = scores["Energy_far"]

            p_known = (float(np.percentile(s_known, 5)), float(np.percentile(s_known, 95)))
            p_near = (float(np.percentile(s_near, 5)), float(np.percentile(s_near, 95)))
            p_far = (float(np.percentile(s_far, 5)), float(np.percentile(s_far, 95)))

            bands_df = pd.DataFrame([
                {"Split": "Known In-Distribution", "x1": p_known[0], "x2": p_known[1], "color": "#3DDC97"},
                {"Split": "Near-OOD (CIFAR held-out)", "x1": p_near[0], "x2": p_near[1], "color": "#FFB86B"},
                {"Split": "Far-OOD (FashionMNIST)", "x1": p_far[0], "x2": p_far[1], "color": "#A78BFA"},
            ])

            ruler_chart = (
                alt.Chart(bands_df)
                .mark_bar(opacity=0.4, height=18)
                .encode(
                    x=alt.X("x1:Q", title="Energy score (values to the right are more unfamiliar / OOD)"),
                    x2="x2:Q",
                    y=alt.Y("Split:N", title=None, axis=alt.Axis(labelColor="#E8ECF5", labelFontSize=12, labelPadding=8)),
                    color=alt.Color("color:N", scale=None),
                )
            )

            marker_df = pd.DataFrame([{"score": res["energy"]}])
            verdict_color = "#FF6B6B" if res["is_ood"] else "#3DDC97"
            rule = (
                alt.Chart(marker_df)
                .mark_rule(color=verdict_color, strokeWidth=3)
                .encode(x="score:Q")
            )

            thr_line_df = pd.DataFrame([{"thr": active_thr}])
            thr_rule = (
                alt.Chart(thr_line_df)
                .mark_rule(color="#E8ECF5", strokeDash=[4, 4], strokeWidth=2)
                .encode(x="thr:Q")
            )

            full_ruler = (ruler_chart + rule + thr_rule).properties(
                height=150, title="Score distribution bands (5th–95th percentile) with active threshold"
            )
            st.altair_chart(full_ruler, width="stretch")

            pct = percentileofscore(s_known, res["energy"])
            st.caption(f"This input's energy score falls at the **{pct:.1f}th percentile** among known-test samples. Dashed line indicates active threshold ({active_thr:.2f}).")

# -----------------------------------------------------------------------------
# TAB 2: Where the threshold hurts
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### The fundamental trade-off of out-of-distribution detection")
    st.caption("Dragging the threshold physically demonstrates why near-OOD detection requires sacrificing legitimate inputs.")

    ctrl_col, btn_col = st.columns([1, 1.5])
    with ctrl_col:
        method = st.segmented_control(
            "Scoring method",
            options=["Energy", "MSP", "Mahalanobis"],
            default="Energy",
            key="tab2_method",
        )

    # Load arrays for method
    s_known = scores[f"{method}_known"]
    s_near = scores[f"{method}_near"]
    s_far = scores[f"{method}_far"]

    # Compute 95% TPR threshold for near-OOD
    from src.evaluate import fpr_at_tpr
    m_thr_95, m_near_fpr = fpr_at_tpr(s_known, s_near, target_tpr=TARGET_TPR)

    # Compute deployment threshold (known false rejection <= 10%)
    sorted_k = np.sort(s_known)
    m_thr_deploy = float(sorted_k[int(0.90 * len(sorted_k))])

    min_val = float(min(np.min(s_known), np.min(s_near), np.min(s_far)))
    max_val = float(max(np.max(s_known), np.max(s_near), np.max(s_far)))

    # Session state for threshold slider
    slider_key = f"tab2_slider_{method}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = round(float(m_thr_deploy), 2)

    with btn_col:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Catch 95% of near-OOD", icon=":material/target:", key=f"btn_95_{method}"):
                st.session_state[slider_key] = round(float(m_thr_95), 2)
                st.rerun()
        with b2:
            if st.button("Keep false rejections under 10%", icon=":material/balance:", key=f"btn_dep_{method}"):
                st.session_state[slider_key] = round(float(m_thr_deploy), 2)
                st.rerun()

    active_slider_val = st.slider(
        f"Decision threshold ({method})",
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        step=0.05,
        key=slider_key,
    )

    # Compute readouts
    known_rej = float(np.mean(s_known >= active_slider_val))
    near_caught = float(np.mean(s_near >= active_slider_val))
    far_caught = float(np.mean(s_far >= active_slider_val))

    r1, r2, r3 = st.columns(3)
    with r1.container(border=True):
        st.metric("Known inputs rejected", f"{known_rej*100:.1f}%")
        st.caption("false rejection rate")
    with r2.container(border=True):
        st.metric("Near-OOD caught", f"{near_caught*100:.1f}%")
        st.caption("the hard regime")
    with r3.container(border=True):
        st.metric("Far-OOD caught", f"{far_caught*100:.1f}%")
        st.caption("the easy regime")

    if near_caught >= 0.94:
        st.warning(
            f"To catch 95% of near-OOD inputs this threshold rejects {known_rej*100:.0f}% of legitimate ones.",
            icon=":material/warning:",
        )

    # Altair ROC Curve with current operating point
    y_near = np.concatenate([np.zeros_like(s_known), np.ones_like(s_near)])
    s_near_all = np.concatenate([s_known, s_near])
    fpr_n, tpr_n, _ = roc_curve(y_near, s_near_all)

    y_far = np.concatenate([np.zeros_like(s_known), np.ones_like(s_far)])
    s_far_all = np.concatenate([s_known, s_far])
    fpr_f, tpr_f, _ = roc_curve(y_far, s_far_all)

    step_n = max(1, len(fpr_n) // 150)
    df_n = pd.DataFrame({"FPR": fpr_n[::step_n], "TPR": tpr_n[::step_n], "Regime": "Near-OOD (CIFAR held-out)"})
    step_f = max(1, len(fpr_f) // 150)
    df_f = pd.DataFrame({"FPR": fpr_f[::step_f], "TPR": tpr_f[::step_f], "Regime": "Far-OOD (FashionMNIST)"})
    df_roc = pd.concat([df_n, df_f])

    roc_lines = (
        alt.Chart(df_roc)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("FPR:Q", title="False Positive Rate (Known Rejection)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="True Positive Rate (OOD Detection)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Regime:N",
                scale=alt.Scale(domain=["Near-OOD (CIFAR held-out)", "Far-OOD (FashionMNIST)"], range=["#FFB86B", "#A78BFA"]),
                title=None,
            ),
        )
    )

    point_df = pd.DataFrame([{"FPR": known_rej, "TPR": near_caught}])
    operating_point = (
        alt.Chart(point_df)
        .mark_circle(size=140, color="#FF6B6B")
        .encode(x="FPR:Q", y="TPR:Q")
    )

    full_roc = (roc_lines + operating_point).properties(
        height=320, title=f"Receiver Operating Characteristic (ROC) — {method}"
    )
    st.altair_chart(full_roc, width="stretch")

    # Method AUROC caption
    curr_entry = next((m for m in metrics["metrics_table"] if m["method"] == method and m["ood_set"] == "near_ood"), None)
    if curr_entry:
        st.caption(f"Near-OOD AUROC for {method}: **{curr_entry['auroc']:.4f}** (FPR@95: {curr_entry['fpr_at_95tpr']*100:.2f}%)")

# -----------------------------------------------------------------------------
# TAB 3: Diagnostics
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Published diagnostic visualizations")

    # 1. Score distributions
    with st.container(border=True):
        st.markdown("#### 1. Score distributions: Known vs. Near-OOD vs. Far-OOD")
        st.image(str(PATHS["figures"] / "score_distributions.png"), width="stretch", alt="OOD Score Distributions")
        st.caption("Near-OOD overlaps known; far-OOD separates cleanly.")

    # 2. Risk-coverage curves + log-scale error rate
    with st.container(border=True):
        st.markdown("#### 2. Risk-coverage: Selective classification accuracy & error rate")
        st.image(str(PATHS["figures"] / "risk_coverage.png"), width="stretch", alt="Risk-Coverage Curves")

        # Second chart: error rate on log scale
        risk_cov_json = PATHS["results"] / "risk_coverage.json"
        if risk_cov_json.exists():
            with open(risk_cov_json) as f:
                rc_data = json.load(f)
            err_rows = []
            for m_name, m_dict in rc_data.items():
                for c_val, r_val in zip(m_dict["coverage"], m_dict["risk"]):
                    err_rows.append({
                        "Method": m_name,
                        "Coverage": float(c_val),
                        "ErrorRate": max(float(r_val), 0.001),
                    })
            err_df = pd.DataFrame(err_rows)
            err_chart = (
                alt.Chart(err_df)
                .mark_line(strokeWidth=2.2)
                .encode(
                    x=alt.X("Coverage:Q", title="Coverage (Proportion of inputs accepted)", scale=alt.Scale(domain=[0.1, 1.0])),
                    y=alt.Y("ErrorRate:Q", scale=alt.Scale(type="log"), title="Error Rate (log scale)", axis=alt.Axis(format="%")),
                    color=alt.Color("Method:N", scale=alt.Scale(domain=["MSP", "Energy", "Mahalanobis"], range=["#7C5CFF", "#3DDC97", "#A78BFA"]), title=None),
                )
                .properties(height=260, title="Selective classification error rate vs coverage (log scale)")
            )
            st.altair_chart(err_chart, width="stretch")
            st.caption("As coverage decreases from 1.0 to 0.5, error rate drops by approximately 9× (from 8% to under 1%).")

    # 3. Per-class AUROC
    with st.container(border=True):
        st.markdown("#### 3. Attractor analysis: Per-class AUROC")
        st.image(str(PATHS["figures"] / "per_class_auroc.png"), width="stretch", alt="Per-Class AUROC Attractor Analysis")
        near_per_class = metrics["per_class_auroc"]["near_ood"]
        weakest = sorted(near_per_class.items(), key=lambda x: x[1])[:2]
        st.caption(
            f"Classes {weakest[0][0].capitalize()} ({weakest[0][1]:.4f}) and {weakest[1][0].capitalize()} ({weakest[1][1]:.4f}) "
            "are the weakest known classes, acting as attractors for dogs and horses."
        )

    # 4. Failure gallery
    with st.container(border=True):
        st.markdown("#### 4. Failure gallery: Falsely accepted OOD inputs at 95% TPR")
        st.image(str(PATHS["figures"] / "failure_gallery.png"), width="stretch", alt="Failure Gallery of Falsely Accepted OOD")
        st.caption("Orange border = Near-OOD, violet border = Far-OOD. These samples were falsely accepted at the 95% TPR threshold.")

# -----------------------------------------------------------------------------
# TAB 4: Benchmark
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Quantitative benchmark summary")

    bench_df = pd.DataFrame(metrics["metrics_table"])
    st.dataframe(
        bench_df[["method", "ood_label", "auroc", "fpr_at_95tpr", "threshold"]].rename(
            columns={
                "method": "Detector Method",
                "ood_label": "Evaluation Dataset",
                "auroc": "AUROC",
                "fpr_at_95tpr": "FPR @ 95% TPR",
                "threshold": "Decision Threshold",
            }
        ),
        column_config={
            "AUROC": st.column_config.NumberColumn("AUROC", format="%.4f"),
            "FPR @ 95% TPR": st.column_config.ProgressColumn("FPR @ 95% TPR", format="%.1f%%", min_value=0.0, max_value=1.0),
            "Decision Threshold": st.column_config.NumberColumn("Decision Threshold", format="%.4f"),
        },
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Selective classification coverage at 95% accuracy")
    cov_rows = []
    for m in ["MSP", "Energy", "Mahalanobis"]:
        rc_info = metrics["risk_coverage"].get(m, {})
        cov_rows.append({
            "Detector": m,
            "AURC": rc_info.get("aurc", 0.0),
            "Coverage at 95% Accuracy": f"{rc_info.get('coverage_at_95acc', 0.0)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(cov_rows), hide_index=True, width="stretch")

    st.markdown("#### Attractor breakdown (Energy detector)")
    attractor_rows = []
    for c in CIFAR_KNOWN_NAMES:
        attractor_rows.append({
            "Known Class": c.capitalize(),
            "Near-OOD AUROC": metrics["per_class_auroc"]["near_ood"][c],
            "Far-OOD AUROC": metrics["per_class_auroc"]["far_ood"][c],
            "Falsely Accepted Impostors": metrics["ood_to_known_confusion"].get(c, 0),
        })
    st.dataframe(pd.DataFrame(attractor_rows), hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown("#### Model calibration")
        cal_pre = metrics["calibration"]["ece_before_temp_scaling"] * 100
        cal_post = metrics["calibration"]["ece_after_temp_scaling"] * 100
        cal_T = metrics["calibration"]["optimal_temperature"]
        st.write(
            f"ECE = {cal_pre:.2f}%. Temperature scaling (T = {cal_T:.3f}) does not improve it (ECE = {cal_post:.2f}%); "
            "a regularised logistic probe on frozen features is already well calibrated."
        )

    with st.container(border=True):
        st.markdown("#### Rebuild benchmark")
        st.caption("Re-executes the full benchmark pipeline. Requires local cache/ and is unavailable on Streamlit Cloud.")
        if st.button("Rebuild from cached features", icon=":material/refresh:"):
            cache_train = PATHS["cache"] / "features_known_train.npz"
            if not cache_train.exists():
                st.warning("Rebuild requires local cached features (~200MB, excluded from repository).", icon=":material/info:")
            else:
                with st.spinner("Rebuilding benchmark suite..."):
                    from src.run_all import run
                    res = run()
                st.success(f"Benchmark rebuilt in {res['timing']['total_seconds']}s. Refresh to view updated figures.")
