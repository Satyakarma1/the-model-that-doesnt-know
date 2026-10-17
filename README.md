# The Model That Doesn't Know: Out-of-Distribution Detection & Uncertainty Estimation

An empirical machine learning study investigating out-of-distribution (OOD) detection, probability calibration, and selective classification in computer vision.



---

## 1. Problem Statement & Motivation

Standard deep learning image classifiers are trained under a closed-world assumption: test inputs will always belong to one of the pre-defined training classes. When presented with an unfamiliar category (e.g. an image of a dog shown to a model trained only on cats, birds, and vehicles), standard softmax outputs exhibit pathological overconfidence, frequently assigning $>90\%$ probability to an incorrect label.

In safety-critical applications—such as autonomous navigation, biometric authentication, and medical diagnostics—**confidently making an incorrect prediction is far more hazardous than admitting uncertainty**.

This project implements an end-to-end vision pipeline that classifies familiar categories while deploying statistical and geometric out-of-distribution detectors to flag unfamiliar inputs for human review or rejection.

---

## 2. Experimental Design & Architecture

```
                                  [ Input Image (224x224x3) ]
                                               │
                                               ▼
                              [ Frozen ResNet-18 Backbone ]
                              (ImageNet Weights, Penultimate 512-d)
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
             [ Linear Classifier ]                           [ Feature Geometry ]
          (Multinomial Logistic Reg)                     (Class Means & Pooled Cov)
                       │                                               │
         ┌─────────────┴─────────────┐                                 │
         ▼                           ▼                                 ▼
   [ Softmax Prob ]          [ Raw Logits ]                [ Mahalanobis Distance ]
         │                           │                                 │
         ▼                           ▼                                 ▼
   [ MSP Score ]              [ Energy Score ]                 [ Distance Score ]
         │                           │                                 │
         └───────────────────────────┼─────────────────────────────────┘
                                     ▼
                     [ OOD Decision Gate (95% TPR) ]
                     ├── Familiar   ──> Output Prediction & Confidence
                     └── Unfamiliar ──> REJECT (Flag Uncertainty)
```

### 2.1 The Controlled Near-OOD vs. Far-OOD Axis
A central design decision of this study is establishing a rigorous two-tier OOD evaluation framework:
- **In-Distribution (Known)**: CIFAR-10 classes 0–4 (`airplane`, `automobile`, `bird`, `cat`, `deer`).
  - `known_train`: 20,000 images (4,000/class, stratified).
  - `known_val`: 5,000 images (1,000/class, stratified).
  - `known_test`: 5,000 images (1,000/class).
- **Near-OOD (Hard Regime)**: CIFAR-10 classes 5–9 (`dog`, `frog`, `horse`, `ship`, `truck`).
  - Semantic pairing: `dog ↔ cat`, `frog ↔ bird`, `horse ↔ deer`, `ship ↔ airplane`, `truck ↔ automobile`.
  - Every held-out class shares natural semantic and textural properties with a known class.
- **Far-OOD (Easy Regime)**: FashionMNIST test set (10,000 grayscale clothing items upsampled to 3-channel). Visually and semantically disjoint from CIFAR natural images.

### 2.2 Three Distinct Uncertainty Detectors
1. **Maximum Softmax Probability (MSP)** (Hendrycks & Gimpel, 2017):
   $$\text{Score}_{\text{MSP}}(x) = -\max_c \text{Softmax}(z)_c$$
2. **Energy-Based OOD Score** (Liu et al., 2020):
   $$E(x; T) = -T \cdot \log \sum_{c=1}^K e^{z_c / T}$$
3. **Mahalanobis Distance** (Lee et al., 2018):
   $$D_M(x) = \min_c \sqrt{(f(x) - \mu_c)^T \Sigma^{-1} (f(x) - \mu_c)}$$
   Fitted strictly on `known_train` feature statistics with ridge regularization $\lambda = 10^{-3}$ and solved via Cholesky factor decomposition (`cho_solve`).

---

## 3. Quantitative Benchmark Results

All metrics computed with deterministic seed (`1337`).

### 3.1 Closed-Set Classification & Calibration
- **Known Test Accuracy**: **92.02%**
- **Validation Accuracy**: **92.78%**
- **Optimal Temperature ($T$)**: **0.9460** (fitted via NLL minimization on `known_val` exclusively)
- **Expected Calibration Error (ECE)**: **0.77%**

### 3.2 OOD Detection Performance

| Method | Evaluation Split | AUROC (↑) | FPR @ 95% TPR (↓) | Decision Threshold |
|---|---|:---:|:---:|:---:|
| **MSP** | Near-OOD (CIFAR Held-out) | 0.7182 | 74.56% | -0.9991 |
| **MSP** | Far-OOD (FashionMNIST) | 0.8677 | 39.76% | -0.9743 |
| **Energy** | Near-OOD (CIFAR Held-out) | **0.7319** | **73.30%** | -8.1295 |
| **Energy** | Far-OOD (FashionMNIST) | 0.8972 | 33.60% | -5.2923 |
| **Mahalanobis** | Near-OOD (CIFAR Held-out) | 0.6824 | 80.56% | 19.7204 |
| **Mahalanobis** | Far-OOD (FashionMNIST) | **0.9408** | **32.06%** | 23.8758 |

### 3.3 Selective Classification (Risk-Coverage)
- **AURC (Area Under Risk-Coverage)**: `MSP: 0.0115`, `Energy: 0.0190`, `Mahalanobis: 0.0717`
- **Coverage at 90% Accuracy**: **100.0%** across all detectors.
- Rejecting the most uncertain predictions systematically eliminates classification errors.

---

## 4. Visualizations (`figures/`)

All publication figures are rendered at 150 DPI and saved in `figures/`:

1. **`figures/score_distributions.png`**: Score histograms comparing In-Distribution vs. Near-OOD vs. Far-OOD across all 3 methods with 95% TPR decision boundaries.
2. **`figures/risk_coverage.png`**: Accuracy vs. Coverage curve proving that selective rejection improves reliability.
3. **`figures/per_class_auroc.png`**: Per-class attractor analysis revealing which known classes attract unknown inputs.
4. **`figures/failure_gallery.png`**: 5×5 failure gallery of false positive OOD images accepted by the filter with predicted class and confidence.

---

## 5. Repository Structure

```
ood-detector/
├── app.py                  # Interactive Streamlit Web Dashboard
├── demo.py                 # Real-time CLI single-image inference demo
├── walkthrough.ipynb       # Presentation notebook with narrative & figures
├── requirements.txt        # Pinned dependencies
├── PLAN.md                 # Original architecture specification
├── CONTEXT.md              # Rationale, design trade-offs, and literature context
├── DECISIONS.md            # Technical reasoning & limitations (4 questions)
├── src/
│   ├── config.py           # Single source of truth (paths, seeds, hyperparams)
│   ├── data.py             # Stratified data partitioning & verification
│   ├── features.py         # ResNet-18 feature extraction & .npz caching
│   ├── probe.py            # Logistic regression head & temperature fitting
│   ├── scores.py           # MSP, Energy, and Mahalanobis scoring functions
│   ├── evaluate.py         # AUROC, FPR@95, ECE, Risk-coverage metrics
│   ├── figures.py          # Figure generation routines
│   └── run_all.py          # Full pipeline orchestrator
├── tests/
│   └── test_sanity.py      # Unit tests for invariant & mathematical validation
├── figures/                # 4 generated evaluation figures (committed)
├── results/                # metrics.json and summary.md
├── cache/                  # Cached .npz representations (gitignored)
└── data/                   # Raw datasets (gitignored)
```

---

## 6. How to Run

### 6.1 Launch the Interactive Streamlit Web Dashboard
```bash
streamlit run app.py
```
*Opens an interactive browser interface featuring a live uncertainty testing playground, visual inspection of all diagnostic plots, and one-click benchmark execution.*

### 6.2 Run Sanity Unit Tests
```bash
python -m tests.test_sanity
```

### 6.3 Run the Full Benchmark Pipeline
```bash
python -m src.run_all
```

### 6.4 Run CLI Single-Image Inference
```bash
# Self-test on random sample:
python demo.py --test-sample

# Test any arbitrary image:
python demo.py path/to/your/image.jpg
```
