# PLAN.md — Agent Build Specification

> **Read CONTEXT.md first.** It explains *why* every choice below was made.
> This file is the *what* and the *in what order*.
>
> **You are building code only.** Do not write `README.md`, `DECISIONS.md`, or any
> prose deliverable. The human writes those. You write `src/`, `tests/`, a short
> `PLAN.md`-adjacent `NOTES_AGENT.md` if you must record assumptions, and result
> artifacts. Nothing else.

---

## 0. Mission

Build a small, rigorous **out-of-distribution (OOD) detection study** on CIFAR-10 +
FashionMNIST using a frozen ImageNet ResNet-18 backbone, a linear probe over 5 known
classes, and three OOD detectors (MSP, Energy, Mahalanobis). Produce a metrics table,
four figures, and a failure gallery.

The deliverable is a **measurement**, not a product. If the numbers are honest and
the protocol is clean, the project succeeds even if AUROC is mediocre.

---

## 1. Hard invariants (violating any of these invalidates the whole study)

| # | Invariant |
|---|---|
| 1 | The 5 held-out CIFAR classes (`dog, frog, horse, ship, truck`) appear **only** as near-OOD test data. They are never seen by the probe, never used for `C` selection, never used for temperature fitting, never used for any statistic. |
| 2 | Temperature `T` is fit on a **validation split carved from the known-class training set**. Never on test data of any kind. |
| 3 | Every score follows one convention: **higher = more out-of-distribution**. State the sign in each function's docstring. |
| 4 | AUROC is computed with **OOD as the positive class**. |
| 5 | Features are extracted **once** and cached to `cache/features_<split>.npz`. Re-running evaluation never re-extracts. |
| 6 | Seeds are fixed for `torch`, `numpy`, `random`, and sklearn. DataLoader order during extraction is deterministic (`shuffle=False`). |
| 7 | No absolute paths anywhere. All paths derive from a single `PATHS` object in `config.py`. |
| 8 | `cache/` and `data/` are gitignored. |

---

## 2. Environment

```
python 3.10+
torch, torchvision (CPU is fine; use CUDA if available)
numpy, scipy, scikit-learn, matplotlib, pandas
```

Pin versions in `requirements.txt`. Windows: set `num_workers=0` or `2` in DataLoaders;
scripts must be guarded with `if __name__ == "__main__":`. Set
`matplotlib.use("Agg")` before importing pyplot so figures save headlessly.

Print the compute device at startup. If CUDA is unavailable, say so and continue on CPU.

---

## 3. Repo layout (create exactly this)

```
ood-detector/
  PLAN.md  CONTEXT.md
  requirements.txt
  .gitignore
  src/
    __init__.py
    config.py
    data.py
    features.py
    probe.py
    scores.py
    evaluate.py
    figures.py
    run_all.py
  tests/
    test_sanity.py
  results/       # metrics.json, summary.md  (gitignored except a committed sample)
  figures/       # 4 PNGs (committed)
  cache/         # gitignored
  data/          # gitignored
```

`.gitignore` must contain at minimum: `data/`, `cache/`, `__pycache__/`, `*.pyc`,
`.venv/`, `.ipynb_checkpoints/`.

Each module 50–150 lines. No monolith. No notebook as the primary artifact.

---

## 4. Data protocol (exact)

CIFAR-10 class ids:

| id | class | role |
|---|---|---|
| 0 | airplane | known |
| 1 | automobile | known |
| 2 | bird | known |
| 3 | cat | known |
| 4 | deer | known |
| 5 | dog | **near-OOD** |
| 6 | frog | **near-OOD** |
| 7 | horse | **near-OOD** |
| 8 | ship | **near-OOD** |
| 9 | truck | **near-OOD** |

Splits and expected sizes:

| split | source | count | use |
|---|---|---|---|
| `known_train` | CIFAR train, ids 0–4, 80% of each class | 20,000 | fit probe, fit Mahalanobis means/cov, fit feature scaler |
| `known_val` | CIFAR train, ids 0–4, 20% of each class | 5,000 | fit temperature `T`; select `C` (or use internal CV — see §6.3) |
| `known_test` | CIFAR test, ids 0–4 | 5,000 | in-distribution evaluation |
| `near_ood` | CIFAR test, ids 5–9 | 5,000 | near-OOD evaluation |
| `far_ood` | FashionMNIST test | 10,000 | far-OOD evaluation |

Split `known_train` / `known_val` **stratified by class**, seeded, indices written to
`cache/split_indices.npz` so it never drifts between runs.

**Preprocessing**

```python
# CIFAR-10
Compose([Resize(224, interpolation=BILINEAR), ToTensor(),
         Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# FashionMNIST
Compose([Resize(224, interpolation=BILINEAR), Grayscale(num_output_channels=3),
         ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

`IMAGENET_MEAN = (0.485, 0.456, 0.406)`, `IMAGENET_STD = (0.229, 0.224, 0.225)`.

**Do not invert FashionMNIST colours.** Record this as a deliberate choice (a comment is
enough). It is a stretch ablation, not a default.

---

## 5. Module specifications

### 5.1 `config.py`

Single source of truth. Export:

```python
SEED = 1337
CIFAR_KNOWN_NAMES  = ["airplane","automobile","bird","cat","deer"]
CIFAR_KNOWN_IDS    = [0, 1, 2, 3, 4]
CIFAR_HELDOUT_NAMES= ["dog","frog","horse","ship","truck"]
CIFAR_HELDOUT_IDS  = [5, 6, 7, 8, 9]
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0            # bump to 2 only if it works on this machine
FEAT_DIM = 512
VAL_FRACTION = 0.2
RIDGE_LAMBDA = 1e-3
N_ECE_BINS = 15
TEMP_BOUNDS = (0.05, 100.0)
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
PATHS = {...}  # root, data, cache, results, figures — all derived, no absolutes
```

### 5.2 `data.py`

```python
def make_datasets() -> dict[str, Dataset]      # keys: known_train, known_val, known_test, near_ood, far_ood
def make_loaders() -> dict[str, DataLoader]    # shuffle only for known_train
def verify_data() -> None                      # prints split sizes + per-class counts; asserts invariants
def load_image_at(split: str, index: int) -> np.ndarray  # un-normalised HWC uint8, for the failure gallery
```

`verify_data()` must assert:
- `known_train` + `known_val` == 25,000, exactly 5,000/class pre-split
- no overlap between `known_train` indices and `known_val` indices
- `known_test` has zero images with label in 5–9
- `near_ood` has zero images with label in 0–4
- class histograms match expectations

Called via `python -m src.data --verify`.

**Why `load_image_at` exists:** the failure gallery needs the *original* image to
display. Re-loading through the dataset is cheap and avoids caching 45,000 images.

### 5.3 `features.py`

```python
def build_backbone(device) -> nn.Module
    # torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # replace .fc with nn.Identity()
    # .eval(), requires_grad_(False), .to(device)

def extract_features(loader, backbone, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # returns (feats[float32, N, 512], labels[int64, N], dataset_indices[int64, N])
    # @torch.no_grad(), backbone.eval(), no shuffle

def get_features(split: str, backbone, device, force: bool = False) -> dict
    # caches to cache/features_<split>.npz with keys: f, y, idx
    # prints "loading cache" or "extracting N images, ETA ~X min"
    # reuses cache unless force=True or shape mismatch
```

Extraction order must be stable — the cached `idx` array is what makes the gallery
work. Print throughput (images/sec) after the first batch so the ETA is honest.

**Feature scaling:** after extraction, z-score each dimension using `known_train` mean/std
only, and apply the same transform to every split. Save the scaler stats to
`cache/scaler.npz`. This is used by the probe and by all detectors. Document the choice.

### 5.4 `probe.py`

```python
def select_C(X_train, y_train) -> float
    # StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    # grid = config.C_GRID, score = mean validation accuracy
    # return best C; log the full grid result to stdout

def fit_probe(X_train, y_train, C) -> LogisticRegression
    # LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=SEED)
    # (no class_weight — classes are balanced by construction)

def get_logits(clf, X) -> np.ndarray        # X @ coef_.T + intercept_
def get_proba(clf, X) -> np.ndarray         # softmax(get_logits(...)) — do not call clf.predict_proba
    # so that temperature scaling composes cleanly with the same code path

def fit_temperature(logits_val, y_val) -> float
    # minimise mean NLL of softmax(logits / T) over T in config.TEMP_BOUNDS
    # scipy.optimize.minimize(method="L-BFGS-B", bounds=..., x0=[1.0])
    # return float(T); print T and NLL before/after

def apply_temperature(logits, T) -> np.ndarray
```

`fit_temperature` must receive **`known_val` logits only**.

### 5.5 `scores.py`

Every function: docstring states the score, the sign, and *why* the sign is that way.

```python
def msp_score(logits) -> np.ndarray
    """OOD score = -max_c softmax(logits)_c.  Higher = more OOD."""

def energy_score(logits, T=1.0) -> np.ndarray
    """OOD score = -T * logsumexp(logits / T, axis=1).  Higher = more OOD."""

def fit_gaussian_stats(X_train, y_train, n_classes) -> tuple[np.ndarray, np.ndarray]
    # means: (n_classes, D), class-conditional mean over known_train only
    # cov: pooled within-class covariance, ddof=1 per class, then averaged by count
    #      + RIDGE_LAMBDA * I
    # assert covariance is positive definite (np.linalg.cholesky must succeed)
    # return means, cov

def mahalanobis_score(X, means, cov) -> np.ndarray
    """OOD score = min_c sqrt((x - mu_c)^T Sigma^-1 (x - mu_c)).  Higher = more OOD.
       Larger minimum distance => no known class explains the input."""
    # For each class c: diff = X - mu_c ; solve(cov, diff.T) — use
    # scipy.linalg.cho_factor/cho_solve or np.linalg.solve. NEVER np.linalg.inv.
    # d_c = sqrt(sum(diff * solved.T, axis=1))
    # return min over c
```

Numerical notes to obey:
- Use `np.linalg.solve` or `cho_solve`. `inv` is a code smell and will be rejected.
- Add the ridge **before** the Cholesky factorisation.
- Clip squared distances at 0 before `sqrt` to survive float error.

### 5.6 `evaluate.py`

```python
def auroc(score_in, score_ood) -> float
    # roc_auc_score(y_true=[0]*len(in) + [1]*len(ood), y_score=concat)
    # higher score = more OOD. Assert both arrays are 1-D and non-empty.

def fpr_at_tpr(score_in, score_ood, target_tpr=0.95) -> tuple[float, float]
    # Returns (threshold, fpr). Derive from the ROC curve, not a bare percentile:
    #   fpr, tpr, thr = roc_curve(y_true, y_score)
    #   i = first index where tpr >= target_tpr
    #   return thr[i], fpr[i]
    # ⚠ SIGN TRAP: np.percentile(score_ood, 95) gives the 5% TPR point, not 95%.
    #   If you use the percentile form it must be percentile(score_ood, 5).
    #   Assert the resulting TPR is in [0.949, 1.0] before returning.

def ece(y_true, probs, n_bins=15) -> float
    # bin by max prob (equal width, 0..1); |acc - conf| weighted by bin mass

def risk_coverage(confidence, correct, grid=np.arange(0, 1.001, 0.01)) -> dict
    # sort by confidence DESCENDING; cumulative accuracy at each coverage step
    # ALSO return: aurc (trapezoid area under risk = 1 - accuracy),
    #              coverage_at_90acc (largest coverage with accuracy >= 0.90, else 0.0)

def per_class_auroc(score_id, y_id, score_ood, class_names) -> dict[str, float]

def ood_to_known_confusion(score_ood, pred_ood, score_in, class_names, target_tpr=0.95)
    # threshold from fpr_at_tpr; accepted = score_ood >= threshold
    # return predicted-class histogram over accepted OOD inputs, with class names
```

Confidence for `risk_coverage` = mean max-probability **after** temperature scaling.

### 5.7 `figures.py`

All four saved to `figures/` at 150 dpi, with axis labels, legends, and a readable title.

1. `score_distributions.png` — 1×3 panels (MSP, Energy, Mahalanobis). Each panel overlays
   known_test / near_ood / far_ood histograms with `density=True`, `alpha=0.55`.
   Vertical dashed line at the near-OOD 95%-TPR threshold. Shared bins per panel.
2. `risk_coverage.png` — coverage (x) vs accuracy (y) for each detector's *rejection*
   behaviour, plus a horizontal line at top-1 accuracy (the "accept everything" baseline).
3. `per_class_auroc.png` — grouped bars, 5 known classes × {near, far}.
4. `failure_gallery.png` — 5×5 grid of OOD images that were **accepted** at the 95%-TPR
   threshold, sorted by confidence descending. Title each: `"pred: dog  0.91"`.
   Border colour: near-OOD orange, far-OOD purple. Legend for the border colours.

Use `load_image_at(split, idx)` for panel 4 — the cached `idx` array gives the mapping.

### 5.8 `run_all.py`

```bash
python -m src.run_all               # full pipeline, uses cache if present
python -m src.run_all --recompute   # force re-extraction
python -m src.data --verify         # data check only
```

Writes:
- `results/metrics.json` — list of `{method, ood_set, auroc, fpr_at_95tpr, threshold}`
  for the 3 methods × 2 OOD sets, plus `{calibration, risk_coverage, per_class_auroc,
  ood_to_known_confusion, closed_set}` blocks.
- `results/summary.md` — human-readable tables generated from the same dict.
- `figures/*.png`

Prints the metrics table to stdout as a fixed-width ASCII table.

---

## 6. Execution order — STOP at each gate

After each gate: `git add -A && git commit -m "<gate name>"`, then **stop and report**.
Report format: what was built, the number(s) produced, anything that looked off, and the
next gate you propose. Do not proceed until the human says go.

### Gate 0 — skeleton + data
Files: `config.py`, `data.py`, `requirements.txt`, `.gitignore`, `src/__init__.py`, tests skeleton.
Run `python -m src.data --verify`.
Report: all split sizes, per-class histograms, and confirmation that no held-out class
appears in `known_*`.

**This must download CIFAR-10 (~170MB) and FashionMNIST (~30MB).** Report the download
time. If a download fails, stop immediately and report — do not substitute a dataset.

### Gate 1 — features + probe
Files: `features.py`, `probe.py`.
Report: extraction time and throughput, chosen `C` with the grid, closed-set accuracy on
`known_test`, and a 5×5 confusion matrix printed as text.

**Sanity band: closed-set accuracy 0.88–0.97.** 5-class CIFAR with ImageNet features.
- `> 0.99` → suspected leakage (held-out classes in training, or train/test overlap). Stop.
- `< 0.85` → suspected bug (normalisation, label remap, scaler fitted on the wrong split). Stop.
- Large train-vs-val gap → overfitting; report it, do not "fix" it by changing the split.

### Gate 2 — detectors + metrics
Files: `scores.py`, `evaluate.py`, `tests/test_sanity.py`.
Report the full 3×2 AUROC/FPR@95 table and these checks:
- `len(score_known_test) == 5000`, `len(score_near_ood) == 5000`, `len(score_far_ood) == 10000`
- For **every** method, `auroc(near) < auroc(far)`. If not, the setup is broken, not impressive.
- No AUROC is exactly 1.0000. If one is, you are almost certainly comparing a split against itself.
- `tests/test_sanity.py` passes: identical distributions → 0.5; perfectly separated → 1.0;
  ECE of a one-hot "perfect" predictor on its own labels → ~0; Mahalanobis covariance is PD.

Write the tests *before* trusting the table. They are the reason the table is believable.

### Gate 3 — calibration + risk-coverage
Add temperature scaling, ECE before/after, risk-coverage + AURC + coverage@90% to the pipeline.
Report: `T`, ECE before, ECE after, AURC per method, coverage@90% per method.
Expect `T > 1` (overconfident model) and ECE dropping. If `T < 0.9`, report it — it means
the validation split is behaving unusually and is worth a comment.

### Gate 4 — figures + failure gallery
Files: `figures.py`.
Report: the four file paths, their sizes, and a one-line description of what each shows.
Rendering failures (blank axes, identical panels, off-screen legends) are common — check
the file sizes are non-trivial and describe what you see in each.

### Gate 5 — integration
`run_all.py` end to end from a clean state. Then verify a **fresh clone** works:
`rm -rf cache/` and re-run. Report total wall-clock for the cold run.

---

## 7. Expected numbers (sanity bands, **not targets**)

| metric | near-OOD (CIFAR held-out) | far-OOD (FashionMNIST) |
|---|---|---|
| AUROC (any detector) | 0.75 – 0.90 | 0.93 – 0.999 |
| FPR@95TPR | 0.45 – 0.80 | 0.05 – 0.30 |

The high near-OOD FPR@95 is the **headline finding**, not a failure. At the operating
point that catches 95% of unknown classes, roughly half to three-quarters of legitimate
inputs get rejected. That number is the project's argument.

Expected ordering: **Energy ≥ MSP** on near-OOD. If Mahalanobis beats both by a wide
margin (e.g. +0.15 AUROC), suspect an under-regularised covariance or a mis-specified
sign — check before reporting.

---

## 8. Things that will get the submission rejected

1. Any use of the held-out classes outside OOD scoring.
2. Temperature fit on test data.
3. `np.linalg.inv`.
4. A single aggregate AUROC across both OOD sets.
5. Reporting numbers without the split sizes and counts that produced them.
6. A notebook as the only artifact, or code duplicated between notebook and `src/`.
7. Writing `README.md` / `DECISIONS.md` (the human owns those).
8. Adding a Streamlit/Gradio demo before Gate 4 is signed off.
9. Hard-coded absolute paths, `cache/` or `data/` committed.
10. Silently changing `CIFAR_KNOWN_IDS`, `IMG_SIZE`, `RIDGE_LAMBDA`, or the val split.
    If you believe a constant is wrong, stop and ask.

---

## 9. Stretch goals — strictly in this priority order, only after Gate 5

1. **`demo.py`** — `python demo.py <image.jpg>` prints top-3 classes with probabilities,
   the three OOD scores, and an ACCEPT/REJECT verdict. ~10 minutes. Highest value per minute.
2. **Fine-tune the backbone** for 2–3 epochs on `known_train` (GPU only) and re-run the full
   detector comparison. This directly addresses "train or fine-tune an image classification
   model" in the task sheet and turns the frozen-vs-fine-tuned contrast into a second finding.
   Do **not** attempt this on CPU.
3. **FashionMNIST inversion ablation** — invert pixel values and re-run far-OOD. Tests
   whether the far-OOD gap is driven by the black-background mismatch rather than semantics.
4. **ODIN-style input preprocessing** — temperature + small input perturbation. Skip unless
   everything above is done.
5. **Thin presentation notebook** — imports from `src/`, displays figures with markdown
   commentary. No logic. Only if time remains.

---

## 10. Definition of done

- [ ] `python -m src.run_all` runs cold and warm, both succeed
- [ ] `results/metrics.json` contains 3 methods × 2 OOD sets + calibration + per-class + confusion
- [ ] Four figures exist and are legible
- [ ] `tests/test_sanity.py` passes
- [ ] `git log --oneline` shows incremental commits, one per gate
- [ ] Fresh clone runs end to end
- [ ] No prose docs written by the agent
