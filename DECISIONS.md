# DECISIONS.md — Technical Reasoning, Architecture Trade-Offs, and Failure Analysis

This document provides honest answers and rationale for the key architectural decisions, challenges, and limitations of this project, following the CodeChef-VIT Recruitment Task submission guidelines.

---

### 01. What was the most technically challenging part of your implementation, and how did you approach it?

The most technically challenging aspect was **preventing subtle data leakage while designing a fair, realistic evaluation protocol for the Near-OOD versus Far-OOD regimes**.

In standard computer vision experiments, out-of-distribution detection is often tested on synthetic artifacts (e.g. Gaussian white noise or heavily mismatched textures), yielding superficially perfect AUROCs above 0.99 that conceal real-world model fragility. To build a genuinely informative study, I deliberately created a hard **Near-OOD split within CIFAR-10**—training exclusively on five classes (`airplane`, `automobile`, `bird`, `cat`, `deer`) while holding out the remaining five (`dog`, `frog`, `horse`, `ship`, `truck`). Because every held-out class has a close semantic twin in the training set (such as `dog ↔ cat` and `horse ↔ deer`), the representations of unknown objects reside immediately adjacent to or inside the known feature manifold.

The engineering challenge was ensuring strict separation at every phase:
1. **Stratification & Validation Carving**: The 80/20 train/validation split was stratified strictly within the known pool (producing 20,000 train and 5,000 val samples) and their exact indices were cached to disk (`split_indices.npz`).
2. **Scaler Statistics**: Feature standardisation parameters ($\mu, \sigma$) were calculated strictly on `known_train`. Applying test-set statistics to feature standardisation is a widespread subtle leakage that artificially improves distance metrics.
3. **Temperature Calibration Isolation**: The temperature parameter ($T = 0.9460$) was fitted by minimizing negative log-likelihood strictly over `known_val` logits. Held-out classes were never touched until final detector evaluation.

I guarded this pipeline by writing automated sanity invariant tests (`tests/test_sanity.py`) that strictly asserted zero overlap between indices and verified class distributions before running any detector scoring.

---

### 02. Describe one important design decision you made, and why did you choose that approach over alternatives?

I chose to employ a **frozen ImageNet-pretrained ResNet-18 backbone as a feature extractor with a multinomial logistic regression probe**, rather than fine-tuning the entire neural network end-to-end.

#### Alternatives Considered:
1. **End-to-End Fine-Tuning**: Unfreezing all residual blocks and training with cross-entropy on the 5 known classes.
2. **Training from Scratch**: Training a custom CNN or ResNet on CIFAR from random initialization.

#### Why Frozen Backbone + Linear Probe?
- **Isolating the Variable Under Test**: The goal of this task is to benchmark and analyze out-of-distribution *detection mechanisms* (MSP, Energy, Mahalanobis distance), not to optimize classification feature learning. End-to-end fine-tuning introduces confounding hyperparameters—learning rate schedules, weight decay, dropout rates, and batch size choices—making it ambiguous whether a failure stems from poor representations or from the detector math.
- **Computational Efficiency & Deterministic Reproducibility**: Fine-tuning across thousands of epochs would consume several hours on CPU hardware. By extracting the 512-dimensional penultimate features once and caching them into compressed `.npz` files, the entire downstream benchmark (classifier training, 3-fold cross-validation grid search for $C$, temperature optimization, scoring, and metric generation) runs in under 15 seconds.
- **Representational Quality**: Deep ImageNet features provide rich, general-purpose semantic embeddings that transfer exceptionally well to CIFAR (achieving 92.02% closed-set test accuracy).

#### The Trade-Off:
ImageNet models are trained on $224 \times 224$ natural images. Upscaling $32 \times 32$ CIFAR images introduces bilinear interpolation smoothing artifacts, which caps maximum closed-set accuracy at ~92% compared to ~96% achievable with specialized CIFAR backbones. I accepted this trade-off because 92.02% is well above the required threshold to demonstrate strong classification while giving us instant iteration speed.

---

### 03. What is one limitation or failure case your current implementation does not handle well?

The current implementation struggles significantly with **fine-grained semantic overlap in geometric feature space**, specifically between quadruped mammals.

In the quantitative benchmark, the Mahalanobis detector performed remarkably well on Far-OOD data (**AUROC: 0.9408**, FPR@95: 32.06%), but degraded severely on Near-OOD data (**AUROC: 0.6824**, FPR@95: 80.56%). 

#### Root Cause of the Failure:
1. **Gaussian Distribution Assumption**: The Mahalanobis detector assumes that each class forms a multivariate Gaussian distribution in 512-dimensional feature space with a single, shared pooled covariance matrix $\Sigma$. In practice, deep feature manifolds of complex natural categories are multi-modal and non-Gaussian.
2. **The "Magnet" Class Phenomenon**: Our per-class attractor analysis revealed that `cat` and `deer` had the lowest AUROCs (0.6789 and 0.6825, respectively). In the failure gallery of falsely accepted OOD samples at the 95% TPR threshold:
   - **65 unknown dogs were classified as cats** (many with $>50\%$ confidence).
   - **35 unknown horses were classified as deer**.
   - **127 unknown trucks/ships were classified as automobiles**.

Because the frozen ImageNet representations for a dog (fur, snout, ears, paws) geometrically project within the convex hull of the `cat` cluster, distance-based and logit-based scores cannot determine whether the sample is an unusual cat or an unknown animal.

---

### 04. If you had another week to work on the project, what would you improve first and why?

If given an additional week, I would prioritize implementing **ODIN (Out-of-Distribution detector for Neural Networks) with input gradient perturbations**, followed by a **Contrastive Representation Learning objective (Supervised Contrastive Loss)**.

#### Why ODIN with Input Preprocessing?
Currently, detectors evaluate raw forward-pass activations. ODIN applies a small backward pass perturbation to the input image before computing the score:
$$x' = x - \epsilon \cdot \text{sign}\left( -\nabla_x \log \text{Softmax}_{y^*}(f(x) / T) \right)$$
In-distribution images lie near sharp local optima where this small gradient shift dramatically amplifies the maximum softmax probability. In contrast, out-of-distribution inputs lie in flatter regions of the loss landscape, meaning their probability increases by a significantly smaller margin. This creates a sharper separation margin between knowns and near-OOD inputs without requiring any architectural modification.

#### Why Contrastive Learning?
Cross-entropy loss only penalizes the classifier until the correct logit exceeds incorrect logits, creating dispersed feature clusters with arbitrary internal structure. In contrast, Supervised Contrastive Loss (SupCon) explicitly pulls in-distribution intra-class embeddings tightly together while pushing different classes maximally apart onto a unit hypersphere. This produces compact, spherical clusters where Mahalanobis and cosine-similarity distance boundaries become vastly more sensitive to outlier detection.
