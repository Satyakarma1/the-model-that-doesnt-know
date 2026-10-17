# DASHBOARD_UPDATE.md — Brief for the coding agent

> **Scope:** rewrite `app.py` and add supporting artifacts. Do not touch `src/` analysis
> logic, `results/metrics.json`, or the figures unless a step below explicitly says so.
> Do not write `README.md` or `DECISIONS.md`.
>
> Read `CONTEXT.md` (the *why*) and `PLAN.md` (the protocol) before starting. The audit
> that motivated this brief found four blocking defects; fix them before any visual work.

---

## 0. Why this update exists

The audit found that the dashboard's bundled sample images are mislabelled
(`sample_near_dog.png` is a ship, `sample_near_horse.png` is a frog,
`sample_known_deer.png` is a truck, `sample_near_truck.png` is a frog). Labels are
derived from dictionary keys rather than from the images themselves, so the UI is
confidently wrong. Two headline metrics are hardcoded instead of read from
`metrics.json`, the "balanced" threshold is an undefined magic constant, and the app
ships with no theme.

The goal of the rewrite is a dashboard that is **visibly honest** about what the model
does and does not know, and **engaging** — the core interaction should let a viewer
feel the near-OOD failure in about five seconds.

---

## 1. Blocking data fixes (do these first, commit, stop)

### 1.1 Regenerate the sample gallery — `scripts/make_samples.py`

Write a script that produces `figures/samples/*.png` **and** `figures/samples/manifest.json`.

Requirements:

- Use `torchvision.datasets.CIFAR10` / `FashionMNIST` from `PATHS["data"]` (already
  present; no downloads).
- Build the split indices by calling `src.data.make_datasets()` and reading the
  `IndexedDataset` wrapper — never by assuming class layout from a flat index.
- For each of these, pick **3 deterministic images** (`random.Random(1337).choice`
  over the class's indices within the split), so the gallery has variety:
  5 known classes (from `known_test`), 5 held-out classes (from `near_ood`),
  2 FashionMNIST classes (from `far_ood`: ankle boot = id 9, t-shirt = id 0).
- Save each at 4× nearest-neighbour upscale (32×32 → 128×128 for CIFAR; 28×28 → 112×112
  for FashionMNIST) with `PIL.Image.resize(..., Image.NEAREST)`.
- Write `manifest.json` as a list of records:

```json
[
  {
    "file": "sample_known_airplane_0.png",
    "true_class": "airplane",
    "true_id": 0,
    "role": "known",          // "known" | "near" | "far"
    "split": "known_test",
    "split_index": 1234,
    "source_dataset_index": 41
  }
]
```

- Delete every pre-existing `figures/samples/*.png` before writing. Do not leave the
  mislabelled files behind.
- Print a table of file → true class so the human can eyeball it.
- **Run it, then stop and report the table.** The human must visually confirm the
  images match their labels before anything else proceeds. This is the checkpoint.

### 1.2 Export the deployment artifacts — `src/export_weights.py`

A single script that writes `results/model_weights.npz` **and** `results/scores.npz`,
replacing whatever ad-hoc process produced the current weights file.

`model_weights.npz` keys (keep existing names so nothing else breaks):
`coef, intercept, scaler_mean, scaler_std, temperature, gaussian_means, gaussian_cov,
thr_near_95, thr_far_95, thr_deploy`.

`results/scores.npz` — float32 arrays, compressed, for the threshold explorer:

```
MSP_known, MSP_near, MSP_far
Energy_known, Energy_near, Energy_far
Mahalanobis_known, Mahalanobis_near, Mahalanobis_far
```

Expected size ~0.5 MB. Commit it. `.gitignore` already needs the
`!results/model_weights.npz` unignore rule; add `!results/scores.npz`.

### 1.3 Derive the deployment threshold

Replace `thr_balanced = -5.0` with a computed value:

```python
# Largest Energy threshold whose false-rejection rate on known_test is <= 10%.
sorted_known = np.sort(energy_known)
thr_deploy = float(sorted_known[int(0.90 * len(sorted_known))])
```

Store it in `model_weights.npz` as `thr_deploy`. Report the achieved TPR on near-OOD and
far-OOD at that threshold. It will be unflattering for near-OOD — that is the point.
Update `tests/test_isolated_deployment.py` to assert `thr_deploy` against the value
printed by the export script, not against `-5.0`.

### 1.4 Remove every hardcoded metric from `app.py`

Grep for numeric literals matching the known results (`92.02`, `0.7319`, `0.9408`,
`0.8720`, `0.6789`, `73.3`, `0.77`). All of them must come from
`results/metrics.json` via one cached loader:

```python
@st.cache_data
def load_metrics(path_str: str) -> dict:
    with open(path_str) as f:
        return json.load(f)
```

Also add a `results/risk_coverage.json` written by `run_all.py`, containing the
`coverage` and `accuracy` arrays per method (they are currently computed and discarded
after plotting). Extend `evaluate.risk_coverage` to also return
`coverage_at_95acc` and report that **instead of** `coverage_at_90acc` in the UI —
`coverage_at_90acc` is 100% for every detector because baseline accuracy is 92%, so it
carries no information.

**Stop and report after 1.1–1.4.**

---

## 2. Theme — `.streamlit/config.toml`

Create the file. Locked dark; this is a demo, not a product with an appearance toggle.
Do not add CSS.

```toml
[theme]
base = "dark"
primaryColor = "#7C5CFF"
backgroundColor = "#0A0E17"
secondaryBackgroundColor = "#141926"
textColor = "#E8ECF5"
linkColor = "#A78BFA"
borderColor = "#232A3B"
codeBackgroundColor = "#141926"
codeTextColor = "#9BE8C4"
font = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"
headingFont = "'Space Grotesk':https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap"
codeFont = "'JetBrains Mono':https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap"
baseRadius = "0.75rem"
buttonRadius = "0.6rem"
showWidgetBorder = true
showSidebarBorder = true
linkUnderline = false
metricValueFontSize = "2rem"
metricValueFontWeight = 600
paddingTop = "1.25rem"
paddingBottom = "3rem"
chartCategoricalColors = ["#7C5CFF", "#3DDC97", "#FFB86B", "#A78BFA", "#FF6B6B", "#4F8EF7", "#8B93A7"]

[theme.sidebar]
backgroundColor = "#0D121C"
secondaryBackgroundColor = "#141926"
textColor = "#E8ECF5"
borderColor = "#232A3B"
primaryColor = "#7C5CFF"
linkColor = "#A78BFA"
```

**Semantic colour contract** — use these consistently everywhere, in charts and text:

| meaning | colour |
|---|---|
| in-distribution / accepted | `#3DDC97` green |
| near-OOD (hard regime) | `#FFB86B` amber |
| far-OOD (easy regime) | `#A78BFA` violet |
| rejected / error | `#FF6B6B` red |
| accent / brand | `#7C5CFF` |

---

## 3. Page structure

Replace the emoji page icon and title block. No CSS, no HTML injection.

```python
st.set_page_config(
    page_title="The model that doesn't know",
    page_icon=":material/psychology_alt:",
    layout="wide",
)
```

### Hero block

```python
st.badge("Uncertainty-aware classification", icon=":material/radar:", color="violet")
st.title("The model that doesn't know")
st.markdown("### Five classes. Everything else is a maybe.")
st.caption(
    "A frozen ResNet-18 with a linear probe, scored by three independent uncertainty "
    "measures and evaluated on a deliberately hard near-OOD split and an easy far-OOD one."
)
```

Then a KPI row of four bordered cards. **No `delta` arguments** — every value is a
static research result and a delta arrow would imply a trend that does not exist. Put
the qualifier in a caption inside each card:

```python
k1, k2, k3, k4 = st.columns(4)
with k1.container(border=True):
    st.metric("Known-set accuracy", f"{acc:.1f}%")
    st.caption("5 classes · 5,000 held-out CIFAR images")
...
```

Card contents, all read from `metrics.json`:

1. Known-set accuracy
2. Near-OOD AUROC — best method named, caption "the hard regime"
3. Far-OOD AUROC — best method named, caption "the easy regime"
4. ECE — caption `f"T = {T:.3f} — already calibrated, scaling changes little"`

That fourth caption is load-bearing: it states the honest calibration result.

---

## 4. Navigation

Use four tabs. Keep the content of all four cheap — do not load scores or compute inside
a tab unless it is trivial; the score arrays are small so loading them up front is fine.

```python
t1, t2, t3, t4 = st.tabs([
    "Probe an image",
    "Where the threshold hurts",
    "Diagnostics",
    "Benchmark",
])
```

---

## 5. Tab 1 — "Probe an image"

Two columns: image on the left (1 part), verdict on the right (1.6 parts).

**Input source:** `st.segmented_control` with two options, `["Curated samples", "Your own image"]`.
Not a radio, not a dropdown.

**Curated samples:** read `figures/samples/manifest.json`. Present the *true* class name
and role as the selectbox label, e.g. `"Horse — held out (near-OOD)"`. Group with
`st.selectbox` over a sorted list of `f"{true_class} · {role_label}"` strings. Load the
file named in the manifest. **Never construct a path or label from a list position.**

**Own image:** `st.file_uploader`, same as today.

**Verdict card** — a bordered container containing:

- `st.success(..., title="Accepted as familiar")` or
  `st.error(..., title="Rejected — the model does not know this")`, plus a caption
  stating the energy score against the active threshold.
- The top predicted class and calibrated confidence as an `st.metric`.
- A caption, driven by **measured values and the true label**, not string matching:

| condition | caption |
|---|---|
| true role is `known` and accepted | "Correct: known class, accepted." |
| true role is `known` and rejected | "False rejection: the model is familiar with this class but the threshold rejected it." |
| true role is `near` or `far` and rejected | "Correct: outside the five known classes, rejected." |
| true role is `near` or `far` and accepted | "**Failure:** an unknown class was accepted as `{pred}`. This is the failure mode the study is about." |

For uploaded images there is no true label: state plainly that ground truth is unknown
and which verdict was produced, with no interpretive claim.

**Probability bars:** horizontal Altair bars over the five known classes, coloured
`#7C5CFF`, sorted descending, with `alt="Calibrated class probabilities"`. Do not use
`st.bar_chart` here — Altair gives you control of sort order and axis.

**Score ruler** — the "where does this land" visual. Build a native Altair chart (no
custom component, no matplotlib):

- x-axis: OOD score.
- Three translucent bands drawn as `mark_rect`, one per split, spanning the 5th–95th
  percentile of the committed `scores.npz` arrays, coloured green / amber / violet with
  `opacity=0.25`.
- A vertical `mark_rule` at the sample's own score, coloured by verdict.
- A caption naming the percentile the sample falls in among known-test inputs.

This is the moment that makes the demo land: the user sees their uploaded photo sit
outside the green band.

---

## 6. Tab 2 — "Where the threshold hurts" (new, the centrepiece)

Goal: make the FPR@95 tradeoff physically obvious in one drag.

**Controls:** `st.segmented_control` for method (`MSP`, `Energy`, `Mahalanobis`), then a
slider over the observed score range with step `0.05`. Two preset buttons in a
horizontal container:

- `st.button("Catch 95% of near-OOD", icon=":material/target:")` — sets the slider to the
  95%-TPR threshold (reuse `evaluate.fpr_at_tpr` logic on the arrays).
- `st.button("Keep false rejections under 10%", icon=":material/balance:")` — sets it to
  `thr_deploy`.

**Readouts,** three bordered columns updating on every drag:

| card | value | caption |
|---|---|---|
| Known inputs rejected | `mean(s_known >= thr)` | "false rejection rate" |
| Near-OOD caught | `mean(s_near >= thr)` | "the hard regime" |
| Far-OOD caught | `mean(s_far >= thr)` | "the easy regime" |

When `catch_near >= 0.94`, render directly beneath: `st.warning("To catch 95% of near-OOD
inputs this threshold rejects {x:.0f}% of legitimate ones.", icon=":material/warning:")`.
That sentence, with the live number, is the entire project in one line.

**Chart:** the ROC curve for the selected method (TPR near-OOD on y, FPR on x), drawn
with Altair, with a point at the current threshold. Below it, a caption naming the
near-OOD AUROC. Add a second faint line for far-OOD in violet so the two regimes are
visible at once.

Percentiles and thresholds must be computed from `results/scores.npz`. Do not recompute
from `cache/` — it is absent on Streamlit Cloud.

---

## 7. Tab 3 — "Diagnostics"

Four bordered containers, each with a heading, the image, and a **caption** (not
`st.info` — captions are lighter and read as annotation rather than alert):

1. **Score distributions** — `figures/score_distributions.png`.
   Caption: near-OOD overlaps known; far-OOD separates cleanly.
2. **Risk–coverage** — `figures/risk_coverage.png`. Add a second chart below it built
   from `results/risk_coverage.json`: **error rate** (`1 − accuracy`) against coverage on
   a log y-axis. The current absolute-accuracy plot is visually flat at 98–100%; the
   error-rate view shows the 9× drop from 8% to under 1%, which is the actual result.
3. **Per-class AUROC** — `figures/per_class_auroc.png`. Caption naming `cat` and `deer`
   as the weakest known classes, pulled from `metrics.json`, not typed.
4. **Failure gallery** — `figures/failure_gallery.png`. Caption explaining orange =
   near-OOD, violet = far-OOD, and that these were accepted at the 95%-TPR threshold.

Every `st.image` gets `width="stretch"` and a short `alt=` naming what the figure shows.

---

## 8. Tab 4 — "Benchmark"

- `st.dataframe` of `metrics_table` with `column_config`:
  `auroc` as `st.column_config.NumberColumn(format="%.4f")`,
  `fpr_at_95tpr` as `st.column_config.ProgressColumn(format="%.1f%%", max_value=1.0)`,
  human column names, `hide_index=True`.
- A second table of per-class AUROC with the falsely-accepted counts merged in.
- A **calibration panel** stating plainly:
  "ECE = 0.77%. Temperature scaling (T = 0.946) does not improve it; a regularised
  logistic probe on frozen features is already well calibrated." Read both numbers from
  `metrics.json` and do not imply an improvement that is not there.
- Keep the re-run button but relabel it "Rebuild from cached features" and state in the
  caption that it requires `cache/` and is unavailable in the deployed app.

---

## 9. Micro-polish checklist (all mandatory)

- [ ] No `use_container_width` anywhere. Use `width="stretch"` or `"content"`.
- [ ] No emoji in labels; Material Symbols via `icon=":material/...:"`.
- [ ] Sentence casing for every heading, tab, and widget label.
- [ ] No empty widget labels; use `label_visibility="collapsed"` where the surrounding UI
      supplies context.
- [ ] No `st.divider()` runs; use spacing or bordered containers.
- [ ] `alt=` on every image and chart.
- [ ] `st.info` only where the text is genuinely an instruction. Everything explanatory
      is `st.caption`.
- [ ] Sidebar contains only project metadata and the class legend — no results.
- [ ] All widget labels that appear more than once have a stable `key`.

---

## 10. Acceptance criteria — verify before reporting done

1. `streamlit run app.py` starts with no warnings in the terminal.
2. Every label in the curated-samples picker matches the image shown. Spot-check all 12.
3. Changing the method in tab 2 changes all three readouts and the chart.
4. Moving the slider to the "95% of near-OOD" preset reproduces the published FPR
   within ±1 percentage point for the Energy detector (≈73.3%).
5. No numeric literal from `metrics.json` appears in `app.py` source. Grep for `0.7319`,
   `0.9408`, `92.02`, `0.77` — zero hits.
6. `python -m tests.test_isolated_deployment` passes with `cache/` and `data/` absent.
7. The dashboard renders correctly with `results/scores.npz` present but `cache/` empty.
8. `git log --oneline` shows one commit per numbered section of this brief.

Report: the sample manifest table, the three thresholds under each preset, and a
screenshot or description of tabs 1 and 2.
