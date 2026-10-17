# CONTEXT.md — Why This Project Exists, and Why It Is Built This Way

> Read this before `PLAN.md`. `PLAN.md` tells you *what* to build. This file tells you
> *why*, so that when a decision isn't covered by the spec you can make the right call
> yourself instead of guessing.
>
> The human writes all prose deliverables (`README.md`, `DECISIONS.md`). You write code.

---

## 1. The setting

CodeChef-VIT recruitment task sheet, Promo Cycle 2026-27, Section 02 — Technical
Challenges, domain **AI and Machine Learning**. Applicants pick **one of two tasks**.
Deadline is roughly six hours from the start of work.

Evaluation criteria, verbatim from the sheet:

- Problem-solving and technical reasoning
- Implementation quality and correctness
- Understanding of technical decisions
- Creativity and independent thinking
- Documentation and explanation

And the closing note, which governs how much to build:

> "We do not expect production-ready applications. An incomplete project with a
> well-thought-out architecture and a clear understanding of its limitations may be
> valued more than a complete project assembled without technical understanding."

That sentence is the strongest signal in the document. A small, correct, well-understood
core beats a large, fragile one. Depth over breadth. Every hour spent on the code should
buy a number or a check that can be explained out loud.

A separate `DECISIONS.md` is optional but encouraged, in the applicant's own words, with
four questions: hardest part, one design decision, one limitation, one week of improvement.
That file is written by the human — but this project is structured so the answers are
*true* and *specific*, which is why each design choice below is recorded with its
alternatives and its cost.

---

## 2. The chosen task: Task 1, "The Model That Doesn't Know"

Plain statement of the problem:

> A classifier trained on a fixed set of categories will be handed inputs from categories
> it has never seen. It should say "I don't recognise this" instead of confidently
> mislabelling. Build that, and show that it works.

Required functionality from the sheet (01–08):

1. At least five known categories.
2. A dataset with documented preprocessing.
3. Train or fine-tune an image classifier.
4. Evaluate on unseen images from known categories.
5. Introduce images from excluded categories.
6. A method for detecting uncertain/unfamiliar inputs.
7. Compare behaviour with and without uncertainty handling.
8. Analyse incorrect predictions and cases where detection fails.

Bonus items, of which this project targets four: OOD detection, confidence calibration,
comparing multiple uncertainty methods, visualising confidence distributions. (A
real-time demo is a stretch goal; robustness to corrupted images is out of scope.)

### Why Task 1 and not Task 2 ("Déjà Code")

**Dependency risk.** Task 2 requires the CodeChef Competitive Programming Dataset, which
lives behind Kaggle authentication. If the download stalls, the project stalls. Task 1
needs CIFAR-10 (~170 MB) and FashionMNIST (~30 MB), both pulled by `torchvision` with no
credentials.

**Verifiability.** Task 2's central claim — "these problems are semantically similar" —
requires manual annotation to defend, and its metrics (silhouette score, cluster purity)
are easy to produce and hard to trust. Task 1 produces AUROC, FPR@95TPR, and ECE, which
have known-good ranges in the literature. A reviewer can tell within a minute whether the
numbers are plausible or invented.

**Differentiation.** Task 2 collapses into TF-IDF + Sentence-BERT + KMeans, which is the
modal submission. Task 1 with a near/far OOD axis has an actual finding to report.

---

## 3. The core idea

Train a 5-class classifier on half of CIFAR-10. Test it on the other half of CIFAR-10 —
the classes it has never seen. Measure how well it can tell "I know this" from "I don't."

The setup is deliberately **hard**: the held-out classes are chosen to be near neighbours
of the known classes.

| known | semantic neighbour held out |
|---|---|
| airplane | ship |
| automobile | truck |
| bird | frog |
| cat | dog |
| deer | horse |

Every unknown class has a known class that looks like it. This is **near-OOD** — the
regime where detection genuinely fails, and therefore the regime worth measuring. A
detector on random noise would hit AUROC 1.0 trivially and tell nobody anything.

Then add **FashionMNIST as far-OOD**: grayscale clothing, visually nothing like CIFAR.
The gap between near-OOD and far-OOD performance is the headline result, and the reason
the two sets both exist.

---

## 4. Why each design decision was made

Each entry: the decision, the alternative that was rejected, and what it costs. This is
raw material for `DECISIONS.md`.

### 4.1 Frozen backbone, linear probe

**Decision.** ResNet-18 with ImageNet weights, `fc` removed, `eval()`, no gradients.
512-d penultimate features. A scikit-learn logistic regression on top.

**Alternative.** Fine-tune the whole network on the 5 known classes.

**Why.** With ~20k training images and a 6-hour budget, fine-tuning makes *representation
quality* the dominant variable, which confounds what is being measured. The question is
"how well does this feature space support OOD detection," not "how good can CIFAR-10
features get." Freezing also makes the whole pipeline run in ~10 minutes on CPU, and
makes every number reproducible from a cached `.npz`.

**Cost.** ImageNet features are not optimal for 32×32 CIFAR images upscaled to 224 —
closed-set accuracy lands around 0.92 rather than 0.96. Accepted. Doing better on the
known classes is not the point.

**Note.** The sheet says "train or fine-tune." A multinomial logistic regression fitted
on cached features *is* training a classifier; the backbone is a frozen feature extractor.
State this plainly in the README rather than hiding it. It is a defensible reading, and
the alternative is honestly scoped as a stretch goal.

### 4.2 Three detectors, one feature space

**MSP** (maximum softmax probability) — the baseline every paper compares against. Cheap,
weak, honest. OOD score is `-max_prob`.

**Energy** — `-T·logsumexp(logits/T)`. Free (needs only logits), usually beats MSP because
it uses the full logit vector rather than the winner alone. Theoretically motivated: it
corresponds to the log partition function.

**Mahalanobis** — distance to the nearest class-conditional Gaussian under a pooled
within-class covariance. Uses feature geometry rather than the classifier's output, so it
fails in a *different* way from the two logit-based scores. That contrast is what makes
the three-method comparison informative rather than three copies of one answer.

**Rejected:** MC dropout (needs the head to be a neural network; marginal payoff over a
linear head), deep ensembles (same reason plus compute), ODIN (stretch goal), ReAct /
ViM / kNN (interesting, but each is another half hour that the analysis deserves more).

### 4.3 Temperature fitted on known-class validation data

Overconfident softmax outputs make ECE and any probability threshold look wrong. A single
scalar `T` fitted by minimising NLL on `known_val` fixes most of it in 20 lines.

The critical constraint: `known_val` is carved from **known-class training data**. If the
temperature were fitted on held-out-class data, the calibration numbers would be
contaminated by the very inputs being evaluated. This is the kind of subtle leak that
looks harmless and invalidates a result. It is called out in `PLAN.md` as an invariant.

### 4.4 Metrics chosen for what they prove

- **Closed-set accuracy** — proves the classifier works at all. A sanity gate, not a result.
- **AUROC** — threshold-free ranking quality. The standard in the OOD literature, so the
  numbers are comparable to published results.
- **FPR@95TPR** — the operating-point number. "If I insist on catching 95% of unknowns,
  what fraction of legitimate inputs do I wrongly reject?" This is the number a product
  manager would ask for, and it is much harsher than AUROC.
- **ECE (15 bins)** — calibration, before and after temperature scaling. Proves the
  probability numbers mean something.
- **Risk-coverage curve + AURC** — accuracy as a function of the fraction of inputs
  accepted. This is the direct expression of the task's premise: *"confidently making an
  incorrect prediction can be worse than admitting uncertainty."* A model that rejects the
  bottom 20% of its inputs and is 99% accurate on the rest is more useful than one that is
  93% accurate everywhere — and only this curve shows that.
- **Per-class AUROC** — which known class acts as a magnet for unknown inputs. This is
  the analysis that separates the submission from a generic benchmark run.
- **OOD→known confusion** — for the unknowns that slip through, *what* do they get called?
  If every held-out dog is called a cat with 0.9 confidence, that is a specific, explainable
  failure, not a number.

### 4.5 Scripts, not a notebook

The sheet's suggested stack lists both "training code or notebook." Scripts win here:

- Every number is reproducible with one command.
- `git diff` shows what changed; a notebook's diff is unreadable.
- No hidden kernel state — the most common cause of numbers that cannot be reproduced.
- Modular pieces can be reviewed independently.

A thin notebook that *imports* from `src/` and displays the figures is worth 15 minutes at
the very end, for a reviewer who prefers to read in a notebook. It must contain zero logic.
If time is short, skip the notebook; do not skip the tests.

### 4.6 Cache the features

45,000 images through ResNet-18 at 224×224 is ~10 minutes on CPU. Re-running evaluation
must not pay that cost again, and iteration speed determines how much analysis fits in the
window. `cache/features_<split>.npz` holds `f`, `y`, and `idx`.

The `idx` array is not optional: the failure gallery needs the original pixel data for a
specific cached row, and `idx` is the mapping back to the dataset.

---

## 5. Vocabulary the human needs to be able to explain out loud

If you cannot explain each of these in one sentence, the project will not survive the
interview.

| term | one-sentence meaning |
|---|---|
| **in-distribution (ID)** | Inputs drawn from the same classes the model was trained on. |
| **OOD** | Inputs from classes the model never saw. |
| **near-OOD / far-OOD** | OOD inputs that do / do not resemble the known classes. |
| **AUROC** | Probability that a random OOD input scores higher than a random ID input; 0.5 = coin flip, 1.0 = perfect. |
| **FPR@95TPR** | At the threshold that catches 95% of OOD inputs, what fraction of ID inputs is wrongly rejected. |
| **TPR / FPR** | Fraction of OOD flagged / fraction of ID wrongly flagged. |
| **MSP** | Max softmax probability; a weak but standard OOD baseline. |
| **Energy score** | `-T·logsumexp(logits/T)`; a better-behaved score than MSP from the same logits. |
| **Mahalanobis distance** | Distance from a point to a distribution's centre, corrected for covariance. |
| **ECE** | Expected calibration error: average gap between predicted confidence and actual accuracy. |
| **Temperature scaling** | Divide logits by a scalar fitted on validation data to reduce overconfidence. |
| **Risk-coverage curve** | Accuracy among the inputs the model chose to accept, as a function of how many it accepted. |
| **AURC** | Area under that curve; lower is better. |
| **Logit** | Pre-softmax score for a class. |

---

## 6. What the human owns

The agent must not do these — they carry the marks, and the sheet explicitly asks for the
applicant's own words:

- `README.md` — setup, methodology, results tables, limitations.
- `DECISIONS.md` — the four questions, in the applicant's own voice.
- Every claim about *why* a number looks the way it does.
- Reading the generated figures and deciding whether the story holds.
- Final git history review and submission.

---

## 7. The honest framing

What this project is: a small, reproducible measurement of how three standard OOD
detectors behave on a deliberately constructed near/far OOD split, with the failure cases
shown rather than hidden.

What it is not: a novel method, a production system, or a state-of-the-art result. The
architecture is borrowed from the literature on purpose — the contribution is the
controlled comparison and the analysis of where it breaks.

Saying that clearly in the README is worth more than pretending otherwise. The sheet asks
for an understanding of limitations; it does not ask for novelty it can't get in six hours.
