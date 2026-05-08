# Surrogate vs True Gradient — Empirical Extension

Empirical extension of **Gygax & Zenke (2024), Section 4**. We replace the
Heaviside spiking unit with a smooth, steep sigmoid (`true_activation = σ(β_f x)`,
`β_f = 50`) so the *true* gradient is computable analytically, then compare it
against a surrogate gradient (`SG = σ'(β_sg x)`) at every step of training.
The whole project is a single empirical question:

> **When (and why) does the surrogate-gradient direction stop agreeing with the
> true-gradient direction during training?**

This README is a **reading guide and write-up in one**: it walks through every
script in the order it was written, explains *why* each design choice was made,
shows how each script's output becomes the next script's input, and reports the
headline numerical result of each analysis.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# bare-bones replication: just the original ablation grid
python main.py
```

Force device:

```bash
SG_DEVICE=cpu python main.py
SG_DEVICE=cuda:0 python main.py
```

Figures land in `figures/`, per-seed scalars in `results/`.
The full pipeline (16 scripts) is described in
[Pipeline](#pipeline-the-16-scripts-in-the-order-they-were-written) below;
the recommended run order is at the bottom.

---

## Repository layout

```
sg_experiment/         core library — model, data, training loop, metrics
  ├── models.py        MLPTrue, MLPSurrogate, custom autograd Function
  ├── data.py          make_moons / make_circles loaders
  ├── metrics.py       cosine sim, sign agreement, relative magnitude
  ├── experiment.py    per-epoch training loop recording both gradients
  ├── plots.py         figure helpers (heatmaps, line plots)
  └── device.py        SG_DEVICE env var → cuda/cpu

main.py                       Phase 1: original ablation grid (depth × β_sg × width × dataset)
regression_experiment.py      Phase 1: ruggedness regression

controlled_depth_experiment.py Phase 2: capacity-controlled depth sweep
collapse_detection.py          Phase 2: detect late-training cos-sim drops
collapse_threshold_sweep.py    Phase 2: threshold robustness
hessian_utils.py               (library) shared Hutchinson + Lanczos primitives
hessian_analysis.py            Phase 2: GPU Hessian probing
hessian_analysis_cpu.py        Phase 2: CPU-deterministic rerun
perturbation_sensitivity.py    Phase 2: paired probed-vs-unprobed test, N=30

saturation_analysis.py         Phase 3: hypothesis 1 — saturation         (REJECTED)
layerwise_alignment.py         Phase 3: hypothesis 2 — per-layer breakdown (REVERSED)
landscape_visualization.py     Phase 3: hypothesis 3 — landscape geometry (NULL)

improvement_5_betasg_collapse.py  Phase 4: β_sg sweep robustness
improvement_6_perturbation_n60.py Phase 4: N=60 rerun of perturbation test
improvement_34_landscape.py       Phase 4: multi-scale + Hessian-eigvec landscape
improvement_1_saturation_v2.py    Phase 4: stricter derivative-magnitude saturation rule
improvement_2_layerwise_v2.py     Phase 4: per-layer detect_collapse
improvement_7_bootstrap.py        Phase 4: nonparametric bootstrap CIs

results/                       per-seed JSON outputs
figures/                       all plots
```

---

## Pipeline: the 16 scripts in the order they were written

The project went through **three phases**:

1. **Replicate** the static depth result (Gygax & Zenke 2024).
2. **Sharpen** it: confirm depth (not capacity), find a *late-training*
   collapse event, falsify the obvious mechanistic hypotheses (curvature,
   instrumentation, hardware).
3. **Diagnose** what the collapse actually is (saturation? layer-wise?
   landscape?), then **stress-test** every conclusion with seven follow-up
   improvements.

Each later phase exists because the previous phase raised a question it could
not answer. The bullets below say what was asked, what was chosen, and what
the next question was.

### Phase 0 — Library

#### `sg_experiment/models.py`
- `MLPTrue` and `MLPSurrogate` **share weights**. Each step we copy the true
  net's parameters into the surrogate, run a backward pass with the same
  data, and read both gradients. This guarantees the only difference between
  the two gradients is the activation derivative used in backprop, not
  parameter drift.
- The surrogate is implemented via a `torch.autograd.Function` whose forward
  is `σ(β_f x)` (so loss is identical) and whose backward is
  `β_sg σ(β_sg x) (1 − σ(β_sg x))`. This mirrors the SNN setup of the base
  paper exactly: forward is the "true" function, backward is the "surrogate"
  derivative.
- Defaults: `β_f = 50` (steep sigmoid → close to a Heaviside but
  differentiable), `β_sg = 5` (the default surrogate steepness in Gygax &
  Zenke), hidden = 64, BCE-with-logits.

#### `sg_experiment/metrics.py`
- We compare gradients with cosine similarity (direction) and sign agreement
  (per-coordinate). Both are scale-invariant so we can compare across
  layers, depths, widths, datasets, and seeds without rescaling.

#### `sg_experiment/experiment.py`
- Full-batch SGD, lr = 0.01, 300 epochs. Full batch removes mini-batch noise
  as a confound — every per-epoch gradient is exactly the gradient of the
  loss on the full dataset, so the only noise source between runs is the
  weight initialisation (controlled by `torch.manual_seed`).

---

### Phase 1 — Replication

#### (1) `main.py` — depth × β_sg × width × dataset grid
- *Question:* Does the static "depth dominates" finding of Gygax & Zenke
  (2024) replicate when training is run for many epochs (not just one
  step)?
- *Choice:* one factor at a time on top of the (depth=4, β_sg=5, width=64,
  moons) baseline. 20 seeds per cell. Heatmap output.
- *Result:* yes — depth is the dominant factor. β_sg matters little inside
  [2, 10]. Width matters mildly. Dataset hardly matters.
- *Next question:* Is the depth effect a *capacity* effect (deeper → more
  parameters) or a *chain-length* effect (deeper → more multiplications of
  small derivatives)?

#### (2) `regression_experiment.py` — ruggedness regression
- *Question:* Same as (1), framed as a regression coefficient instead of a
  heatmap.
- *Choice:* OLS of cosine-sim ↦ {depth, log(β_sg), log(width), dataset
  one-hots}, all scaled. Lets us read off effect sizes with confidence
  intervals.

---

### Phase 2 — Sharpening

#### (3) `controlled_depth_experiment.py` — capacity-controlled depth sweep
- *Question (from 1):* Is the depth effect just a parameter-count effect?
- *Choice:* Hold parameter count fixed at the depth-4 / width-64 reference
  (12,737 params) and binary-search the hidden width per depth that lands
  closest. Search range `[8, 4000]`.
- *Result:* at fixed capacity, cosine sim drops from **0.99 (depth 1, width
  3184)** to **0.49 (depth 4, width 64)**. **Depth effect is chain-length,
  not capacity.**
- *Next question:* The seed-to-seed variability is huge — is something
  *qualitatively* different happening in some seeds?

#### (4) `collapse_detection.py` — late-training collapse
- *Question:* Are the seeds bimodal (some collapse, some don't)?
- *Choices made and why:*
  - `window=20`, `threshold=0.15`: a "collapse" is a sustained drop of >0.15
    versus the trailing-20 mean; eyeballed on 100 example trajectories.
  - **Causal trailing cumsum smoothing** instead of `np.convolve(mode='same')`:
    the latter has a known boundary artifact that put a fake "collapse" at
    every run's tail. Causal smoothing fixes this.
  - `min_epoch=50`: ignore early instability when the network is still
    learning the easy structure.
- *Result:* yes — late-training collapse is a real, sustained event in 30–60%
  of 4-layer moons runs around epoch 240.
- *Next question:* Is the threshold choice load-bearing?

#### (5) `collapse_threshold_sweep.py` — threshold robustness
- *Choice:* sweep `threshold ∈ [0.05, 0.30]`, plot detection count.
- *Result:* `0.15` sits at the elbow of the curve — robust.
- *Next question:* What *causes* the collapse?

#### `hessian_utils.py` — shared probing primitives
- Hutchinson trace estimator with **m = 5 Rademacher probes** (variance vs
  cost trade-off). Lanczos with **n_iter = 30, k = 5**, full
  re-orthogonalisation (numerical stability matters more than speed at
  m = 5). All probe randomness reads from an **isolated `torch.Generator`**
  so it never advances the model RNG. This isolation is what makes (8)
  possible.

#### (6) `hessian_analysis.py` — GPU Hessian curvature
- *Question:* Is the collapse a curvature event (sharpness spike →
  surrogate fails to track)?
- *Choice:* probe every 5 epochs from epoch 0 to 300; record both the
  Hutchinson trace and `λ_max`.
- *Result:* on the collapsing seed, `λ_max` actually **decreases**
  post-collapse (t = −2.04, p ≈ 0.094). Across collapsed-vs-non-collapsed
  averages, the trajectories are indistinguishable. **Curvature hypothesis
  refuted.**
- *Next question:* Did the *act of probing* alter the trajectory?

#### (7) `hessian_analysis_cpu.py` — CPU-deterministic rerun
- *Question:* Is GPU non-determinism producing the per-seed variability?
- *Choice:* `torch.use_deterministic_algorithms(True)` on CPU, same seeds.
- *Result:* same seed (9) collapses at the same epoch (106), `λ_max`
  trajectories agree to 3 decimal places. **Hardware non-determinism is not
  the cause.**

#### (8) `perturbation_sensitivity.py` — paired probed-vs-unprobed test, N = 30
- *Question:* Could the Hessian probing in (6) itself be inducing
  collapses?
- *Choice:* paired design with the *same* `torch.manual_seed` for both arms
  (eliminates between-seed variance entirely). McNemar's exact test on the
  2×2 paired contingency table is the right test for paired binary
  outcomes; we also report Wilson and Newcombe CIs and Cohen's h.
- *Result:* probing **never creates** a collapse (0 probed-only); it
  appears to **suppress** collapse in 5/30 seeds. McNemar p = 0.0625.
  **Borderline at α = 0.05, N = 30 — needs (13).**

---

### Phase 3 — Mechanism

We had three ex-ante hypotheses for the collapse mechanism. Each got its
own script.

#### (9) `saturation_analysis.py` — Hypothesis 1: saturation
- *Hypothesis:* late in training, hidden units saturate (sigmoid output
  → 0 or → 1), the true derivative vanishes, and the surrogate (which is a
  shallower sigmoid) keeps producing non-zero gradient → divergence.
- *Choices:* output-magnitude rule, `σ(β_f x) > 0.99 OR < 0.01`. Per-layer
  and global tracking; pre/post-collapse comparison.
- *Result:* saturation fraction is **flat at 0.87–0.88 across all layers,
  all seeds, all 300 epochs**. Pearson r(sat, cos) = 0.008. **REJECTED.**
- *Caveat that motivated #15:* this rule uses *output* magnitude, but
  the SG-vs-true comparison cares about *derivative* magnitude. Could a
  stricter derivative-based rule rescue the hypothesis?

#### (10) `layerwise_alignment.py` — Hypothesis 2: per-layer breakdown
- *Hypothesis:* the input-side layers (where many derivatives multiply
  together) drop alignment first, and the output layer drops last.
- *Choices:* `_per_layer_grads` concatenates each Linear layer's weight + bias
  gradient; per-layer cosine sim; threshold-0.6 drop time per layer.
- *Result:* the **opposite**. Input-side layers (L0, L1) sit at cos ≈ 0.10
  *throughout training*; only the **output-side layer** (L3) carries
  high-alignment gradient (≈ 0.55), and L3 is the layer that drops at
  collapse. The global cos sim ≈ output-side cos sim because the output-
  side layer dominates gradient mass.
- *This is the key qualitative finding of the project.*
- *Caveat that motivated #16:* the 0.6 threshold is meaningless for layers
  that never reach 0.6 (it just reports "epoch 50"). Need to use the same
  detector as (4) per layer.

#### (11) `landscape_visualization.py` — Hypothesis 3: landscape
- *Hypothesis:* there's a geometric event in weight space at collapse
  (cliff, ridge, saddle).
- *Choices:* Li et al. (2018) **filter normalisation** for the random
  visualisation directions (so the per-filter scale matches the true scale
  the optimiser sees). Scout 10 seeds, pick one with a clean collapse,
  retrain *that* seed with parameter-state checkpoints saved at every
  epoch in [180, 300]. Visualise at collapse_ep ± 20.
- *Result:* the three plots are **almost identical** smooth bowls. Loss
  *decreases* across collapse (0.120 → 0.110 → 0.102) and `λ_max`
  decreases (221 → 218 → 145). **Null geometric event.** This is the
  evidence that the SG remains a useful descent direction even after
  losing alignment — the central message of the project.
- *Caveats that motivated #14:* (i) the visualisation range was a single
  ±1.0 — far larger than the per-step parameter delta (~10⁻²); (ii) the
  Hessian-eigenvector arrows in the overlay were tiny because random
  projections of high-dim unit vectors are O(1/√n).

---

### Phase 4 — Stress-tests (seven improvements)

After Phase 3 reached its synthesis ("the collapse is a transition in
*which layers contribute the dominant gradient mass*, not a phase change
in the loss surface, the curvature spectrum, or the activity regime"), I
asked: what could still be wrong? Each weakness got its own script.

#### (12) `improvement_5_betasg_collapse.py` — β_sg sweep
- *Why:* (1) showed β_sg matters little for the *static* metric; does it
  matter for the *late-training collapse event*?
- *Choices:* `β_sg ∈ {2, 3, 5, 7, 10}`, **40 seeds each** (a 10-seed pilot
  produced an apparent monotone drift in mean final cos that did not
  survive the larger sample), output collapse rate + median epoch +
  final cos sim.
- *Result:* collapse rate sits in $[0.50, 0.58]$ with heavily overlapping
  Clopper–Pearson CIs, median collapse epoch fluctuates non-monotonically
  in $[154, 189]$, and mean final cos is in a tight band $[0.490, 0.516]$.
  Late-training collapse is **insensitive to β_sg** over this range — not
  a surrogate-slope-mismatch phenomenon.

#### (13) `improvement_6_perturbation_n60.py` — N = 60 rerun
- *Why:* (8) was just-marginal at N = 30; double N for power.
- *Choices:* `import perturbation_sensitivity as base; base.N_SEEDS = 60`
  to *guarantee* identical statistics code path — no copy-paste drift.
- *Result:* paired diff `+0.117`, bootstrap 95 % CI `[+0.017, +0.217]`
  (excludes 0). McNemar p = 0.0654. Effect size halved (0.167 → 0.117) but
  direction preserved. Probing weakly suppresses collapse.

#### (14) `improvement_34_landscape.py` — multi-scale + Hessian-eigvec landscape
- *Why:* fix the two caveats from (11). Reuses the same scout-then-retrain
  pipeline (`import landscape_visualization as base`) so the seed selection
  and checkpoints are the *same* runs.
- *Improvements:*
  - `#3`: 4 ranges `{0.05, 0.1, 0.5, 1.0}` × 3 epochs (collapse ± 20). At
    every zoom the iterate sits in a smooth quadratic basin.
  - `#4`: top-2 Hessian eigenvectors *as* the visualisation plane (filter-
    normalised). `λ_1` drops 221 → 218 → 145 across collapse, basin walls
    soften visibly. Cleanest single piece of evidence that collapse is *not*
    a sharpening event.

#### (15) `improvement_1_saturation_v2.py` — stricter derivative-magnitude rule
- *Why:* fix the caveat from (9).
- *Choice:* "saturated" ≡ true derivative `< ε · β_f / 4` for ε ∈ {0.01,
  0.05, 0.10}; also track the per-layer mean true / surrogate derivative
  ratio.
- *Result:* sat fraction still flat at 0.88 across the board. Pearson r(sat,
  cos) ∈ {−0.030, 0.014, 0.034}. **The (9) rejection survives.**

#### (16) `improvement_2_layerwise_v2.py` — per-layer detect_collapse
- *Why:* fix the caveat from (10).
- *Choice:* run `detect_collapse(per_layer_cos)` on each layer's trajectory
  with the *same* parameters (window 20, threshold 0.15, min_epoch 50).
  Also report per-layer mean gradient L₂ norm.
- *Result:* of 10 seeds × 4 hidden layers = 40 trajectories, only 3 drop;
  none align with the global collapse epoch. Mean per-layer ‖g‖₂ goes 0.08
  (L0) → 0.17, 0.15, 0.17 → **0.27 (readout)**: gradient mass is
  output-side. Same conclusion as (10), now with the right detector.

#### (17) `improvement_7_bootstrap.py` — nonparametric bootstrap CIs
- *Why:* the Welch t-tests in (6) and (8) assume normal sampling
  distributions — questionable at N = 10. Replace with B = 10 000 bootstrap.
- *Choice:* loads existing per-seed JSONs from `results/` (no recomputation)
  and reports percentile CIs on every key scalar. Paired diff CI for the
  perturbation test uses **paired** resampling (resample seed indices, take
  the difference *within* a pair).
- *Result:* GPU and CPU `λ_max` CIs **indistinguishable** (formal version of
  the (7) hardware-determinism check). Both perturbation diff CIs strictly
  exclude 0.

---

## Recommended run order

```bash
# Phase 1 — replication
python main.py
python regression_experiment.py

# Phase 2 — sharpening
python controlled_depth_experiment.py
python collapse_detection.py
python collapse_threshold_sweep.py
python hessian_analysis.py
python hessian_analysis_cpu.py
python perturbation_sensitivity.py

# Phase 3 — mechanism
python saturation_analysis.py
python layerwise_alignment.py
python landscape_visualization.py

# Phase 4 — stress-tests
python improvement_5_betasg_collapse.py
python improvement_6_perturbation_n60.py
python improvement_34_landscape.py
python improvement_1_saturation_v2.py
python improvement_2_layerwise_v2.py
python improvement_7_bootstrap.py   # last — depends on JSONs from earlier scripts
```

Total runtime on an RTX 4080 Laptop is on the order of an hour;
on CPU expect several hours.

---

## How the scripts feed each other

```
main.py ──────────┐
                  ▼
            (depth dominates?)
                  │
                  ▼
controlled_depth_experiment.py    ─→  yes, chain-length not capacity
                  │
                  ▼
            (per-seed bimodality?)
                  │
                  ▼
collapse_detection.py  ◀── collapse_threshold_sweep.py
                  │
                  ▼
            (what causes it?)
        ┌─────────┼──────────────┐
        ▼         ▼              ▼
   saturation? layer-wise?   geometry?
        │         │              │
        ▼         ▼              ▼
 saturation_   layerwise_   landscape_
 analysis.py   alignment.py visualization.py
        │         │              │
   REJECTED   REVERSED        NULL
        │         │              │
        ▼         ▼              ▼
  imp_1_v2.py imp_2_v2.py imp_34_landscape.py
   (stricter)  (per-layer  (multi-scale +
                detector)   Hessian plane)

In parallel for the rigor side:

hessian_analysis.py ──→ perturbation_sensitivity.py ──→ imp_6 (N=60)
        │                       │                            │
        ▼                       ▼                            ▼
hessian_analysis_cpu.py   imp_5_betasg_collapse.py     imp_7_bootstrap.py
                                                       (consumes all JSONs)
```

---

## Synthesis: what the collapse actually is

Putting all sixteen scripts together:

- The collapse is **not** a saturation phenomenon — (9) and (15).
- The collapse is **not** a curvature / sharpness phenomenon — (6), (11),
  (14): `λ_max` actually **decreases** through collapse (221 → 218 → 145).
- The collapse is **not** an instrumentation artifact — (8), (13): probing
  never *creates* a collapse, it weakly *suppresses* it.
- The collapse is **not** a hardware-non-determinism artifact — (7), (17):
  GPU and CPU bootstrap CIs are indistinguishable.
- The collapse is **not** a geometric event in weight space — (11), (14):
  smooth quadratic basin at every zoom, monotonically softening.
- The collapse **is** a transition in *which layers contribute the
  dominant gradient mass* — (10), (16): the output-side layer carries
  cos ≈ 0.55 throughout, and is the only layer whose alignment falls at
  the collapse epoch. Input-side layers sit at cos ≈ 0.10 the whole run,
  but their gradient norm is too small to move the global cosine.
- Crucially, training loss keeps **decreasing** across the collapse
  (0.120 → 0.110 → 0.102) — (11). The SG remains a valid descent
  direction even when nearly orthogonal to the true gradient. This
  *strengthens* the original Gygax & Zenke result rather than weakening
  it: useful descent does not require directional alignment with the true
  gradient.

**Reframed mechanism.** In deep sigmoid MLPs, the surrogate gradient and
the true gradient agree on what the *output-side* weights should do, and
disagree about what the *input-side* weights should do. The gradient mass
concentrates near the output, so the global cosine similarity reflects the
output-side agreement. Late in training, the output-side agreement also
breaks (the network has reduced loss enough that the output-side gradient
is now of comparable magnitude to the perpetually misaligned input-side
gradients). Once that happens, the global cosine drops, but the SG
direction is still useful enough that loss continues to fall.

---

## What the seven improvements still do *not* cover

Even with these seventeen scripts, the project's external validity
remains bounded. Future work would add:

1. **Multiple datasets.** All sigmoid-MLP results live on `make_moons`
   (and partially `make_circles`). No image / tabular / sequence
   benchmark; no real-world data.
2. **Multiple optimisers.** Only full-batch SGD with lr = 0.01.
   Adam, momentum, mini-batch noise, and learning-rate schedules
   are all known to interact non-trivially with curvature.
3. **Multiple architectures.** Only sigmoid-activation MLPs at
   width 64 and depths 2–6. No ReLU, no normalisation, no
   residual connections, no attention.
4. **Slow-collapse low-lr regime.** A run at lr ≪ 0.01 would test
   whether collapse is an effective-time phenomenon or an
   epoch-count phenomenon.
5. **Theoretical chain-rule sketch.** A short analytic argument
   linking output-layer SG/true-gradient agreement to a closed-form
   condition on the post-activation distribution would convert the
   mechanism above from empirical to predictive.

---

## Reproducibility

- Pinned `requirements.txt`. CUDA 13.0 + torch 2.11.0+cu130, Python 3.12.
- `torch.manual_seed(seed)` per run; isolated `torch.Generator` for all
  probe-vector randomness; `torch.use_deterministic_algorithms(True)` on
  CPU for the determinism baseline.
- All figures saved to `figures/`, all per-seed scalars to `results/`.
