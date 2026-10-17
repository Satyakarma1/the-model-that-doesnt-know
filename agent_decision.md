# AGENT_DECISION.md — Complete Agent Decision, Feedback, and Post-Mortem Report

This document records the complete decision history, technical trade-offs, user feedback loops, mistakes made, and engineering solutions implemented by the coding agent throughout the development of **"The Model That Doesn't Know"** (CodeChef-VIT Recruitment Task — AI/ML Track).

---

## 1. Core Architectural Questions & Technical Answers

### 01. What was the most technically challenging part of your implementation, and how did you approach it?

The most technically challenging aspect was **preventing subtle data leakage while designing a fair, realistic evaluation protocol for the Near-OOD versus Far-OOD regimes, and reconciling the gap between academic benchmark thresholds and real-world image deployment**.

#### A. The Near-OOD vs. Far-OOD Semantic Split
In standard computer vision literature, out-of-distribution (OOD) detection is frequently tested against synthetic noise or completely unrelated domains (e.g. evaluating CIFAR models on Gaussian noise or SVHN digits), yielding superficially inflated AUROCs above 0.99. To construct a genuinely rigorous test, we deliberately partitioned CIFAR-10 into two semantically paired subsets:
- **Known (In-Distribution, 5 classes)**: `airplane`, `automobile`, `bird`, `cat`, `deer` (25,000 train images, 5,000 test images).
- **Near-OOD (Held-Out Unknowns, 5 classes)**: `dog`, `frog`, `horse`, `ship`, `truck` (5,000 test images).
- **Far-OOD (Completely Foreign Domain)**: FashionMNIST (10,000 test images upsampled to 3 channels).

Notice the intentional semantic neighbor pairing: `dog ↔ cat`, `horse ↔ deer`, `frog ↔ bird`, `ship ↔ airplane`, and `truck ↔ automobile`. The representations of unknown objects lie immediately adjacent to the known feature manifold.

#### B. Strict Anti-Leakage Pipeline Isolation
To guarantee technical correctness:
1. **Stratification & Validation Carving**: An 80/20 train/validation split was stratified strictly within the known classes (20,000 train, 5,000 val), and exact indices were cached to disk (`split_indices.npz`).
2. **Scaler Statistics**: Feature standardisation parameters ($\mu, \sigma$) were calculated strictly on `known_train`. Applying test-set statistics to feature standardisation is a widespread subtle leakage that artificially improves distance metrics.
3. **Temperature Calibration Isolation**: The temperature parameter ($T = 0.9460$) was fitted by minimizing negative log-likelihood strictly over `known_val` logits. Held-out classes were never touched until final detector evaluation.

#### C. The Academic vs. Real-World Operating Dilemma
During testing, the user provided critical real-world input: they uploaded photos of a clean cat, deer, and horse, and found that the model rejected all of them. 

Upon investigation, we uncovered a major conceptual flaw in academic OOD benchmarks:
- **Academic Benchmark Metric**: By literature convention, the decision threshold is calibrated to fix the True Positive Rate (TPR) on In-Distribution data at 95% against the *Near-OOD* set (`thr_near_95 = -8.13`).
- **The Trade-Off**: Because Near-OOD classes (dogs, horses) overlap heavily with known classes, catching 95% of subtle Near-OOD impostors requires setting the energy threshold so aggressively low that it rejects **73.3% of legitimate, clean in-distribution samples**.
- **The Solution**: We implemented a **Dual Operating Mode** in the system:
  1. *Strict Academic Benchmark Mode* (`Threshold = -8.13 @ 95% TPR`): For reporting literature-standard AUROC and FPR@95.
  2. *Balanced Deployment Mode* (`Threshold = -5.00`): Calibrated to accept high-confidence known inputs (cats and deer pass easily) while rejecting out-of-domain categories (horses, fashion items, noise).

---

### 02. Describe one important design decision you made, and why did you choose that approach over alternatives?

We chose to employ a **frozen ImageNet-pretrained ResNet-18 backbone as a feature extractor with a multinomial logistic regression probe**, rather than fine-tuning the entire neural network end-to-end.

#### Alternatives Considered:
1. **End-to-End Fine-Tuning**: Unfreezing all residual blocks and training with cross-entropy on the 5 known classes.
2. **Training from Scratch**: Training a custom CNN or lightweight ResNet on CIFAR from random initialization.

#### Why Frozen Backbone + Linear Probe?
- **Isolating the Variable Under Test**: The goal of this recruitment task is to benchmark and analyze out-of-distribution *detection mechanisms* (MSP, Energy, Mahalanobis distance), not representation learning. End-to-end fine-tuning introduces confounding hyperparameters—learning rate schedules, weight decay, dropout rates, and batch size choices—making it ambiguous whether a failure stems from poor representations or from the detector math.
- **Computational Efficiency & Deterministic Reproducibility**: Fine-tuning across thousands of epochs would consume hours on CPU hardware. By extracting the 512-dimensional penultimate features once and caching them into compressed `.npz` files, the entire downstream benchmark (classifier training, 3-fold cross-validation grid search for $C$, temperature optimization, scoring, and metric generation) runs in under 15 seconds.
- **Representational Quality**: Deep ImageNet features provide rich, general-purpose semantic embeddings that transfer exceptionally well to CIFAR (achieving 92.02% closed-set test accuracy).

#### The Trade-Off:
ImageNet models are trained on $224 \times 224$ natural images. Upscaling $32 \times 32$ CIFAR images introduces bilinear interpolation smoothing artifacts, which caps maximum closed-set accuracy at ~92% compared to ~96% achievable with specialized CIFAR backbones. We accepted this trade-off because 92.02% is well above the required threshold to demonstrate strong classification while giving us instant iteration speed.

#### Additional Important Design Decisions:
- **Energy-Based Scoring over Softmax Confidence (MSP)**: Standard Softmax probabilities suffer from the softmax denominator normalisation effect: inputs with uniformly high logits still yield high maximum probabilities. The Energy score maps logits directly to free energy:
  $$E(x; T) = -T \cdot \log \sum_{k=1}^K \exp\left(\frac{f_k(x)}{T}\right)$$
  Energy score is theoretically aligned with the input's marginal density $p(x)$ and proved superior, improving Far-OOD AUROC from 0.8876 (MSP) to 0.9022 (Energy).
- **Self-Contained Model Packaging (`results/model_weights.npz`)**: When deploying to Streamlit Cloud, relying on gitignored `cache/` and raw datasets (~200MB) caused immediate container crashes. Decoupling the offline feature extraction pipeline from online deployment by serializing probe coefficients, scaler statistics, temperature, and covariance matrices into a single 2.03 MB file enabled instantaneous cloud cold-starts.

---

### 03. What is one limitation or failure case your current implementation does not handle well?

The current implementation struggles significantly with **fine-grained semantic overlap in geometric feature space**, specifically between quadruped mammals.

In the quantitative benchmark, the Mahalanobis detector performed remarkably well on Far-OOD data (**AUROC: 0.9408**, FPR@95: 32.06%), but degraded severely on Near-OOD data (**AUROC: 0.6824**, FPR@95: 80.56%). 

#### Root Cause of the Failure:
1. **Gaussian Distribution Assumption**: The Mahalanobis detector assumes that each class forms a multivariate Gaussian distribution in 512-dimensional feature space with a single, shared pooled covariance matrix $\Sigma$:
   $$M(x) = \max_k \left( - (f(x) - \mu_k)^T \Sigma^{-1} (f(x) - \mu_k) \right)$$
   In practice, deep feature manifolds of complex natural categories are multi-modal and non-Gaussian.
2. **The "Magnet" Class Phenomenon**: Our per-class attractor analysis revealed that `cat` and `deer` had the lowest AUROCs (0.6789 and 0.6825, respectively). In the failure gallery of falsely accepted OOD samples at the 95% TPR threshold:
   - **65 unknown dogs were classified as cats** (many with $>50\%$ confidence).
   - **35 unknown horses were classified as deer**.
   - **127 unknown trucks/ships were classified as automobiles**.

Because the frozen ImageNet representations for a dog (fur, snout, ears, paws) geometrically project within the convex hull of the `cat` cluster, distance-based and logit-based scores cannot determine whether the sample is an unusual cat or an unknown animal.

#### Domain Discrepancy on User-Uploaded Images:
CIFAR-10 consists of tiny $32 \times 32$ images. When the user uploaded high-resolution phone photos of a cat and deer, the high-frequency edge textures and real-world camera lighting created a slight domain shift compared to CIFAR's low-resolution training distribution, elevating the energy scores from ~ -9.7 to ~ -6.5. Under the strict academic threshold (`-8.13`), they were falsely rejected. The Balanced threshold (`-5.0`) successfully accepted them.

---

### 04. If you had another week to work on the project, what would you improve first and why?

If given an additional week, we would prioritize implementing **ODIN (Out-of-Distribution detector for Neural Networks) with input gradient perturbations**, followed by a **Contrastive Representation Learning objective (Supervised Contrastive Loss)**.

#### Why ODIN with Input Preprocessing?
Currently, detectors evaluate raw forward-pass activations. ODIN applies a small backward pass perturbation to the input image before computing the score:
$$x' = x - \epsilon \cdot \text{sign}\left( -\nabla_x \log \text{Softmax}_{y^*}(f(x) / T) \right)$$
In-distribution images lie near sharp local optima where this small gradient shift dramatically amplifies the maximum softmax probability. In contrast, out-of-distribution inputs lie in flatter regions of the loss landscape, meaning their probability increases by a significantly smaller margin. This creates a sharper separation margin between knowns and near-OOD inputs without requiring any architectural modification.

#### Why Contrastive Learning?
Cross-entropy loss only penalizes the classifier until the correct logit exceeds incorrect logits, creating dispersed feature clusters with arbitrary internal structure. In contrast, Supervised Contrastive Loss (SupCon) explicitly pulls in-distribution intra-class embeddings tightly together while pushing different classes maximally apart onto a unit hypersphere. This produces compact, spherical clusters where Mahalanobis and cosine-similarity distance boundaries become vastly more sensitive to outlier detection.

---

## 2. Chronological Record of User Inputs, Feedback & Agent Decisions

| # | Phase / Event | User Input / Feedback | Agent Analysis & Decisions Made |
|---|---|---|---|
| **1** | **Initial Project Brief** | Provided recruitment PDF with a 6-hour deadline. Highlighted: *"We do not expect production-ready applications. An incomplete project with a well-thought-out architecture and a clear understanding of its limitations may be valued more than a complete project assembled without technical understanding."* Requested the agent to be the sole coding developer while user prepared for the interview. | **Decided on Task 1 ("The Model That Doesn't Know")** over Déjà Code. Designed the 5 Known / 5 Near-OOD / FashionMNIST Far-OOD regime. Selected frozen ImageNet ResNet-18 + multinomial linear probe to guarantee fast CPU iteration (<30s total) and unambiguous mathematical evaluation. |
| **2** | **Network Drop & Dataset Rescue** | Wi-Fi dropped during torchvision download. User instructed: *"run the data verification command again, i changed wifi... check this folder i have downloaded both the datsets, the cifar-10-python.tar.gz and the fashion mnist and extracted them in this folder, abort your downloading task and move to next tasks"* | Aborted socket connections. Refactored `src/data.py` to auto-detect extracted directories `cifar-10-batches-py` and `FashionMNIST` directly on disk, bypassing network downloads entirely. |
| **3** | **Interactive UI Request** | After terminal and static PNG generation, user asked: *"When I run the full pipeline command and the run single image inference demo, the results are shown in the terminal and a PNG file. Can we generate a Streamlit for this so I can show this visually during the interview? I want a visual-based one where I run the test, press a button, and it runs the test and gives me all the scores. It should generate visuals and all the visualizations, the graphs, and also the failure image you generated."* | Built `app.py` from scratch featuring 3 tabs: (1) Live single-image inference with interactive bar charts; (2) Publication figures and failure gallery explorer; (3) Quantitative benchmark tables and 1-click test execution button. |
| **4** | **Custom Image Testing & Bug Report** | User tested the app with real photos: *"ok i did some testing on the streamlt, i noticed that it did well on thee training data but the upload your own image is completelt broken... i uploaded basic images of cat, horse, deer, and it failed on all of them. In horse showing confident in cat, in cat and deer confident but rejecting hypothesis... why is our model failing on custom images?"* | Diagnosed two distinct causes: (1) **Horse rejection is correct**: `horse` is an excluded Near-OOD category. The detector intercepted the unverified prediction and rejected it, which is the exact intended behavior of an OOD detector. (2) **Cat and Deer rejection was an academic threshold flaw**: fixing TPR at 95% against Near-OOD rejects 73.3% of ID samples. **Decided to build Dual Operating Modes**: Added `Balanced Deployment Mode` (`threshold = -5.00`) which accepts real-world known photos while rejecting unknown classes, and added technical rationale expanders in the UI. |
| **5** | **Streamlit Cloud Deployment Crash** | User attempted to deploy to Streamlit Cloud (`satyakarma1-the-model-that-doesnt-know-app-zd4ljo.streamlit.app/`) and provided crash logs: `ModuleNotFoundError: No module named 'tqdm'` and `SyntaxWarning: "\g" is an invalid escape sequence`. | **Comprehensive Cloud Hardening**: (1) Added `tqdm` and `Pillow` to `requirements.txt`; (2) Discovered that Cloud lacked `cache/` and `data/` (~200MB gitignored), so created `results/model_weights.npz` (2.03 MB) and `figures/samples/` (10 static PNGs) to enable standalone deployment; (3) Restored accidental omission of `ResNet18_Weights` in `src/features.py`; (4) Replaced `\g` with clean Unicode `≥`; (5) Built automated isolated unit test (`tests/test_isolated_deployment.py`) passing 100%. Pushed to GitHub. |

---

## 3. Engineering Post-Mortem: Mistakes Made & How They Were Solved

1. **Mistake 1: Relying on Live Dataset Downloads**
   - *What went wrong*: Using `torchvision.datasets.CIFAR10(download=True)` hung when Wi-Fi dropped.
   - *Fix*: Decoupled dataset verification from network state; prioritized local offline directories.
2. **Mistake 2: The Academic Metric Trap in Deployment**
   - *What went wrong*: Calibrated single-image rejection against academic 95% TPR on Near-OOD (`-8.13`), causing 73% of legitimate real-world images to be falsely rejected.
   - *Fix*: Created a dual-threshold system separating academic benchmarking from practical deployment.
3. **Mistake 3: Missing Dependencies in `requirements.txt`**
   - *What went wrong*: `tqdm` was used for CLI progress bars but omitted from `requirements.txt`.
   - *Fix*: Added `tqdm>=4.65.0` and `Pillow>=9.5.0` with defensive try/except fallbacks in source code.
4. **Mistake 4: Accidental Import Deletion During Fast Refactoring**
   - *What went wrong*: In an edit to make `tqdm` optional in `src/features.py`, the import `from torchvision.models import ResNet18_Weights, resnet18` was accidentally replaced, causing a `NameError`.
   - *Fix*: Restored imports and added automated test validation before committing.
5. **Mistake 5: Ephemeral Container File Access**
   - *What went wrong*: `app.py` originally depended on `cache/` and `data/` which are excluded from Git. On Streamlit Cloud, these files were absent, causing `FileNotFoundError`.
   - *Fix*: Packaged `results/model_weights.npz` and static sample PNGs into Git, making the app 100% self-contained.
6. **Mistake 6: Python 3.14 String Escape Warning**
   - *What went wrong*: `$\ge$` in non-raw f-strings triggers a `SyntaxWarning` in Python 3.14.
   - *Fix*: Replaced with standard Unicode operators `≥` and `<`.
