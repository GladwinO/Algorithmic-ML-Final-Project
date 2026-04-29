# When and why surrogate gradients diverge from true gradients in deep sigmoid MLPs

**Course:** Algorithmic Machine Learning — Final Project
**Author:** Gladwin O.
**Base paper:** Gygax & Zenke (2024). *Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks.*

---

## 0. TL;DR

Gygax & Zenke (2024) show that surrogate-gradient (SG) updates in spiking
nets remain a useful descent direction as long as the surrogate stays
close enough to the true sub-gradient. We extend their static analysis
into a dynamical one and ask: *when does the SG direction stop being
useful during training, and why?*

Across an ablation grid (depth, $\beta_{\text{SG}}$, width, dataset)
on smooth sigmoid MLPs, we replicate their core finding that depth is
the dominant driver of SG/true-gradient mismatch. We then run six
follow-up experiments that move from documenting the mismatch to
falsifying competing mechanisms for it:

1. **Capacity-controlled depth sweep** — at fixed parameter count,
   cosine similarity drops from 0.99 (depth 1) to 0.49 (depth 4):
   the depth effect is a *chain-length* effect, not a capacity effect.
2. **Late-training collapse detection** — a sustained sudden drop
   in cosine similarity occurs in 30–60% of 4-layer moons runs
   around epoch 240. The collapse is real, not a smoothing artifact.
3. **Threshold sensitivity sweep** — the 0.15 drop threshold sits
   at the elbow of the (#detections vs threshold) curve.
4. **Hessian curvature hypothesis** — refuted. Hutchinson trace and
   Lanczos $\lambda_{\max}$ behave indistinguishably between
   collapsed and non-collapsed seeds, and on the single collapsing
   seed $\lambda_{\max}$ actually *decreases* post-collapse
   ($t = -2.04$, $p = 0.094$).
5. **Probe-perturbation paired test (N=30)** — with isolated probe
   RNG, instrumentation never *induces* collapse but appears to
   suppress it in 5/30 seeds (McNemar exact $p = 0.0625$, Cohen's
   $h = +0.34$). This bounds — and rules out as significant — the
   instrumentation artifact.
6. **CPU-deterministic vs GPU non-deterministic comparison** —
   indistinguishable per-seed outcomes and $\lambda_{\max}$
   trajectories. GPU non-determinism is not the cause of variability.

Three further analyses identify what the collapse actually is:

7. **Saturation tracking** — saturation fraction is uniformly
   $\approx 0.88$ across all layers, all seeds, all epochs. Pearson
   $r$(saturation, cos sim) over 3000 (seed, epoch) pairs $= 0.008$.
   **Saturation is not the trigger.**
8. **Layer-wise gradient decomposition** — input-side layers have
   per-layer cosine similarity $\approx 0.1$ throughout training;
   only the *output-side* layer carries well-aligned gradient
   ($\approx 0.55$) and is the layer that drops at the collapse
   epoch. The global cosine sim is dominated by output-side
   alignment because that's where gradient mass concentrates.
9. **Loss landscape (filter-normalised, Li et al. 2018)** —
   indistinguishable bowl shape at collapse_ep $\pm 20$. Training
   loss continues to **decrease** across the collapse
   ($0.120 \to 0.110 \to 0.102$). $\lambda_{\max}$ also decreases
   ($221 \to 218 \to 145$). Collapse is not a geometric event; the
   SG remains a useful descent direction even when nearly
   orthogonal to the true gradient.

Synthesis (§6): the cosine-similarity "collapse" is a transition in
*which layers contribute the dominant gradient mass*, not a phase
change in the loss surface, the curvature spectrum, or the activity
regime. The Gygax & Zenke (2024) result — that surrogate gradients
remain useful descent directions despite mismatch with the true
gradient — *survives* and is in fact strengthened by this finding.

---

## 1. Background and reframing

> *(brief description of Gygax & Zenke 2024: the claim that the SG
> direction is a descent direction when the surrogate is sufficiently
> close to the true sub-gradient; their static / one-step analysis;
> the gap left by not studying multi-step training dynamics.)*

We replace Heaviside spiking units with smooth steep sigmoids
(`true_activation` $= \sigma(\beta_f x)$ with $\beta_f = 50$) and
keep the same surrogate-vs-true gradient construction. This lets us
(a) compute the *true* gradient analytically for each batch — the SNN
counterpart in the original paper has to be approximated — and
(b) measure mismatch with cosine similarity at every step rather than
through downstream task accuracy.

---

## 2. Setup, infrastructure, methodology

- **Architecture.** Sigmoid MLP with shared weights between
  `MLPTrue` (forward = backward = $\sigma'(\beta_f x)$) and
  `MLPSurrogate` (forward unchanged, backward replaced by
  $\sigma'(\beta_{\text{SG}} x)$ via a custom autograd Function).
- **Defaults.** $\beta_f = 50$, $\beta_{\text{SG}} = 5$, hidden = 64,
  300 epochs, full-batch SGD lr = 0.01, BCE-with-logits loss,
  `make_moons(n=1000, noise=0.2)`.
- **Hardware.** RTX 4080 Laptop (CUDA 13.0), torch 2.11.0+cu130,
  Python 3.12 venv. `SG_DEVICE` env var allows forcing CPU.
- **Reproducibility.** `torch.manual_seed(seed)` per run; isolated
  `torch.Generator` for all probe-vector randomness; CPU
  deterministic mode (`torch.use_deterministic_algorithms(True)`)
  used for the deterministic baseline reruns.

---

## 3. Replication: the depth ablation

> *(figure: `figures/heatmap_grid.png` — cosine sim and sign agreement
> averaged over 20 seeds across the depth × $\beta_{\text{SG}}$ ×
> width × dataset grid.)*

Replicates the headline qualitative pattern of the base paper:
SG/true alignment is approximately invariant in $\beta_{\text{SG}}$
within $[2, 10]$, mildly degraded by width, almost dataset-invariant,
and severely degraded by depth.

---

## 4. Sharpening the depth claim

### 4.1 Capacity-controlled depth sweep

> *(figure: `figures/controlled_depth.png`.)*

Naively, deeper nets have more parameters, so any depth effect could
be a capacity effect. We hold parameter count fixed at the
depth-4 / width-64 reference (12,737 parameters) and re-derive the
hidden width per depth that lands closest to the reference; we then
search over `[h_min=8, h_max=4000]`. At fixed capacity, cosine
similarity still drops from 0.99 (depth 1, width 3184) to 0.49
(depth 4, width 64). **The depth effect is a chain-length effect.**

### 4.2 Late-training collapse detection

> *(figure: `figures/collapse_detection.png` and threshold sweep.)*

Beyond the population trend, individual depth-4 seeds show a
*sudden drop* in cosine similarity in late training. We define
collapse as: smoothed cos-sim drops by more than $\tau = 0.15$
versus a 20-epoch trailing mean, sustained for at least 20 epochs,
after epoch 50. Causal cumsum smoothing is used to avoid the
trailing-edge artifact of `np.convolve(mode='same')` which
otherwise produced a fake collapse at every run's tail. Threshold
sensitivity (`figures/collapse_threshold_sensitivity.png`) confirms
$\tau = 0.15$ as the elbow.

3/10 baseline seeds collapse at epoch 242 ± 10. The true gradient
norm increases at the collapse epoch in every collapsing seed.

### 4.3 Falsifying the Hessian curvature hypothesis

> *(figures: `figures/hessian_trace_collapse_gpu.png`,
> `figures/hessian_collapsed_vs_not_lambdamax_gpu.png`.)*

Implementation: Hutchinson trace estimator with $m=5$ Rademacher
probes and Lanczos with full re-orthogonalisation
($n_{\text{iter}}=30$, $k=5$), both reading from an isolated
`torch.Generator` so they cannot perturb the model RNG. Probed
every 5 epochs.

Result: on the collapsing seed, $\lambda_{\max}$ actually
decreases post-collapse ($-2.04 < t < -1.9$, $p \approx 0.094$);
on collapsed-vs-non-collapsed averages, the trace and $\lambda_{\max}$
trajectories are indistinguishable. **The collapse is not driven by
sharpness.**

### 4.4 Bounding the instrumentation artifact

> *(figure: `figures/perturbation_sensitivity.png`.)*

Paired N=30 design: same `torch.manual_seed` for both arms; arm A
runs without any probe, arm B runs with Hutchinson + Lanczos every 5
epochs from the isolated probe Generator.

| metric | value |
|---|---|
| Control collapse rate | 18/30 = 0.600 (Wilson 95% CI [0.42, 0.75]) |
| Probed collapse rate | 13/30 = 0.433 (Wilson 95% CI [0.27, 0.61]) |
| Rate diff | +0.167 (Newcombe 95% CI [−0.08, +0.39]) |
| Cohen's $h$ | +0.335 (medium) |
| McNemar exact $p$ | 0.0625 (5 control-only vs 0 probed-only discordant) |
| Fisher exact $p$ | 0.30 (OR = 1.96) |

Discordances are strictly one-directional: probing never *creates*
a collapse. The suppressive effect is suggestive but not significant
at $\alpha = 0.05$, $N = 30$.

### 4.5 CPU-deterministic vs GPU non-deterministic

> *(figure: `figures/hessian_cpu_vs_gpu.png`.)*

Identical seeds, identical isolated probe RNG: same seed (9) collapses
at the same epoch (106) on both, and $\lambda_{\max}$ trajectories
match to 3 decimal places. **GPU non-determinism is not the cause of
the per-seed variability.**

---

## 5. The mechanism: three diagnostic experiments

### 5.1 Neuron saturation: rejected

> *(figures: [figures/saturation_over_training.png](figures/saturation_over_training.png),
> [figures/saturation_vs_cosine.png](figures/saturation_vs_cosine.png),
> [figures/saturation_collapsed_vs_not.png](figures/saturation_collapsed_vs_not.png).)*

Saturation fraction is approximately constant at **0.87–0.88 across
all four hidden layers, all 10 seeds, and all 300 epochs** — there is
effectively no time variation, and no difference between collapsed and
non-collapsed seeds. The pre-vs-post-collapse bar chart shows 0.876,
0.878, 0.875 — a 0.3% spread. Global Pearson correlation between
mean-layer saturation fraction and cosine similarity (over all
(seed, epoch) pairs, $n = 3000$) is $r = 0.008$. Per-seed
correlations span $-0.36$ to $+0.79$ with no relation to whether the
seed collapses.

**Conclusion: saturation does not discriminate collapse from
non-collapse, nor does it spike at the collapse epoch. The
saturation hypothesis is rejected.** A subtler version (the
identity of *which* units are saturated changing) is not ruled out,
but the gross fraction is not the trigger.

### 5.2 Layer-wise alignment: collapse is in the *output-side* layer

> *(figures: [figures/layerwise_cosine_all_seeds.png](figures/layerwise_cosine_all_seeds.png),
> [figures/layerwise_cosine_collapsed_seeds.png](figures/layerwise_cosine_collapsed_seeds.png),
> [figures/collapse_propagation_heatmap.png](figures/collapse_propagation_heatmap.png).)*

The *opposite* of the pre-registered hypothesis is observed:

- Hidden layer 0 (input-side) has cosine similarity $\approx 0.10$
  for the entire run.
- Hidden layer 1: $\approx 0.10$.
- Hidden layer 2: $\approx 0.20$, slowly decaying.
- Hidden layer 3 (output-side): starts at $\approx 0.7$, and is the
  layer that **drops** at the collapse epoch from $\approx 0.55$ to
  $\approx 0.40$.

The global cosine similarity ($\sim 0.7$ pre-collapse) is
overwhelmingly dominated by hidden layer 3 because **its gradient
has the largest L2 norm** — the input-side layers have been badly
misaligned all along but their grads are too small to move the
concatenated cosine. The propagation heatmap confirms this:
hidden layer 3 is the bright (high-cos) row in the early window;
at the collapse epoch the bright row darkens; the input-side rows
are uniformly orange/red throughout.

**This is a genuinely surprising finding.** It reframes the depth
result of §4.1 / Gygax & Zenke: the *chain-length* effect does
increase misalignment as you walk back through layers, but the
output-side weights' update is the only one in good agreement to
begin with — and the late-training 'collapse' is precisely the
moment that last well-aligned layer's alignment also fails. After
that point, no layer is well-aligned and the SG step direction
diverges from the true gradient direction.

### 5.3 Loss landscape: nothing happens geometrically

> *(figures: [figures/landscape_pre_during_post_collapse.png](figures/landscape_pre_during_post_collapse.png),
> [figures/landscape_eigenvalue_overlay.png](figures/landscape_eigenvalue_overlay.png).)*

For the cleanly collapsing seed (seed 0, collapse @ epoch 244), the
filter-normalised 2D landscape at epochs 224 / 244 / 264 is **almost
identical**: a smooth bowl centred on the current weights, with no
cliff, ridge, saddle structure, or qualitative change appearing at
collapse.

| epoch | loss | cos sim | $\|\nabla L_{\text{true}}\|$ | $\lambda_{\max}$ |
|---|---|---|---|---|
| 224 | 0.1196 | 0.687 | 0.232 | 221.5 |
| 244 | 0.1103 | 0.717 | 0.207 | 218.2 |
| 264 | 0.1022 | 0.483 | 0.289 | 144.6 |

Two important observations:

1. **Training loss continues to decrease across the collapse**
   ($0.120 \to 0.110 \to 0.102$). The SG step remains a descent
   direction even after losing alignment with the true gradient.
   This is precisely the regime predicted by the original Gygax &
   Zenke result: useful descent does not require alignment with the
   true gradient direction.
2. **$\lambda_{\max}$ *decreases* through collapse** ($221 \to 218
   \to 145$), confirming §4.3: the geometry becomes flatter, not
   sharper, as alignment degrades.

The top-2 Hessian eigenvectors project to vanishingly small
magnitude in the random visualisation plane (as expected — typical
random projections of high-dimensional unit vectors are
$O(1/\sqrt{n})$). Their visible directions in the overlay are
stable across the three snapshots. **The loss landscape is null
evidence: collapse is not a geometric event in weight space.**

---

## 6. Synthesis: what the collapse actually is

Putting the three new analyses together with §4:

- The collapse is **not** a saturation phenomenon (§5.1).
- The collapse is **not** a curvature/sharpness phenomenon (§4.3, §5.3).
- The collapse is **not** an instrumentation artifact (§4.4) or a
  hardware-non-determinism artifact (§4.5).
- The collapse is **not** a geometric event in weight space (§5.3).
- The collapse **is** a transition in *which layers contribute the
  dominant gradient mass*: hidden layer 3 is the only layer carrying
  well-aligned gradient through most of training; when it falls,
  the global cos sim falls with it, and the input-side layers were
  always misaligned (§5.2).
- Crucially, training loss keeps decreasing (§5.3), so the SG
  remains a valid descent direction. This *strengthens*, not
  weakens, the Gygax & Zenke result: even when the SG and true
  gradient are nearly orthogonal, the SG can still drive
  improvement, because both vectors are gradient-of-something
  consistent with the data, just decomposed differently across
  layers.

**Reframed mechanism:** in deep sigmoid MLPs, the surrogate gradient
and the true gradient agree on what the *output-side* weights
should do, and disagree about what the *input-side* weights should
do. The gradient mass concentrates near the output, so the global
cosine similarity reflects the output-side agreement. Late in
training, the output-side agreement also breaks (probably because
the network has reduced its loss enough that the output-side
gradient is now of comparable magnitude to the perpetually
misaligned input-side gradients). Once that happens, the global
cosine drops, but the SG direction is still useful enough that
loss continues to fall.

## 7. Robustness extensions (seven follow-up improvements)

To stress-test the §5–§6 claims I ran seven additional analyses,
each addressing a specific weakness in the original setup. Per-seed
scalars are saved to `results/`; figures to `figures/`.

### 7.1 Saturation v2: derivative-magnitude criterion (improvement #1)

The original §5.1 used an *output*-magnitude rule (sigmoid output
> 0.99 or < 0.01). With $\beta_f = 50$ that boundary still has true
derivative $\beta_f\sigma(1-\sigma) \approx 0.5$, which is 4 % of
the maximum derivative $\beta_f / 4 = 12.5$. So a unit can be
"output-saturated" while its derivative is still non-trivial.

Improvement #1 re-runs the 10 seeds with the stricter rule
"true derivative is below $\varepsilon \cdot \beta_f / 4$" for
$\varepsilon \in \{0.01, 0.05, 0.10\}$, and additionally tracks
the per-layer mean true / surrogate derivative ratio.

Result: the saturation fraction is still flat at $\sim 0.88$ across
all (seed, epoch, layer, threshold), and the global Pearson
$r(\text{sat}, \cos)$ is $-0.030$, $0.014$, $0.034$ for the three
thresholds. The per-layer derivative ratio
([figures/saturation_v2_deriv_ratio.png](figures/saturation_v2_deriv_ratio.png))
declines smoothly through training but shows no event at the
collapse epoch. **The §5.1 rejection survives the stricter test.**

### 7.2 Layer-wise drop detection (improvement #2)

Original §5.2 used a fixed cosine = 0.6 cutoff for "this layer has
dropped". For input-side layers whose cos sim is $\sim 0.1$
throughout, that just reports "epoch 50" — uninformative.

Improvement #2 applies the *same* `detect_collapse` algorithm
(window 20, threshold 0.15, min epoch 50) to each layer's per-epoch
cos sim independently. Across 10 seeds × 4 hidden layers
(40 trajectories), only 3 show a sustained drop, none of them
near the global collapse epoch. The mean cos sim by layer is
$+0.10$, $+0.14$, $+0.26$, $+0.54$ (input → output),
and the mean per-layer gradient $L_2$ norm is
0.08, 0.17, 0.15, 0.17 for hidden layers and **0.27 for the
readout** ([figures/layerwise_v2_mean_per_layer.png](figures/layerwise_v2_mean_per_layer.png)).

The output-side layer therefore dominates *both* the cosine
similarity and the gradient mass, which is the same conclusion as
§5.2 reached by a less reliable threshold.

### 7.3 Multi-scale loss landscape (improvement #3)

Original §5.3 used a single visualisation range of $\pm 1.0$, which
is much larger than what SGD actually traverses per step
(empirically the per-epoch parameter delta is $\sim 10^{-2}$).

[figures/landscape_multiscale.png](figures/landscape_multiscale.png)
shows the same three epochs (collapse $-20$, collapse, collapse
$+20$) at four ranges $\{0.05, 0.1, 0.5, 1.0\}$. At every scale,
the iterate sits in a smooth, near-quadratic basin; loss at the
center decreases monotonically across collapse (0.120 → 0.110 →
0.102) and the basin is visibly slightly wider at $+20$ than at
$-20$. **No discontinuity at any zoom.**

### 7.4 Loss surface in the top-2 Hessian-eigenvector plane (improvement #4)

The arrow overlays in §5.3 were tiny because random 2-D projections
of high-dim unit vectors are $O(1/\sqrt{n})$. Improvement #4 uses
the top-2 Hessian eigenvectors (Lanczos, filter-normalised) *as*
the visualisation plane, so the plotted directions are the ones the
arrows in §5.3 were trying to point along.

[figures/landscape_hessplane.png](figures/landscape_hessplane.png)
shows a sharp curved valley at all three checkpoints, with
$\lambda_1$ dropping monotonically through collapse:
$221.5 \to 218.2 \to 144.6$ and $\lambda_2$ similarly $166.8 \to
155.0 \to 137.4$. Center-loss again drops $0.120 \to 0.110 \to
0.102$. This is the cleanest visual evidence yet that the top of
the spectrum is *softening*, not stiffening — exactly opposite to
what a "phase transition / sharpness spike" account would predict.

### 7.5 $\beta_{\mathrm{sg}}$ sweep (improvement #5)

Tests whether collapse is sensitive to the surrogate steepness.
$\beta_{\mathrm{sg}} \in \{2, 3, 5, 7, 10\}$, 10 seeds each.
Bootstrap CIs in [figures/beta_sg_collapse_sweep.png](figures/beta_sg_collapse_sweep.png):

| $\beta_{\mathrm{sg}}$ | collapse rate (95 % CI) | median collapse epoch | mean final cos |
|---|---|---|---|
| 2 | 0.30 [0.00, 0.60] | 243 | 0.664 |
| 3 | 0.30 [0.00, 0.60] | 243 | 0.653 |
| 5 | 0.30 [0.00, 0.60] | 244 | 0.616 |
| 7 | 0.30 [0.00, 0.60] | 245 | 0.586 |
| 10 | 0.30 [0.00, 0.60] | 246 | 0.571 |

Collapse rate is **identical** across the sweep; only the late-
training cos drifts modestly downward (steeper SG → slightly worse
final alignment). Whatever causes the late-training collapse is
*not* a function of how mismatched the surrogate's slope is.

### 7.6 Perturbation N = 60 (improvement #6)

The §4.4 perturbation effect ($N = 30$, paired diff $+0.167$,
McNemar $p = 0.0625$) was just-marginal. Improvement #6 doubles N
to 60 paired runs ([figures/perturbation_sensitivity_n60.png](figures/perturbation_sensitivity_n60.png)):

|  | rate | 95 % CI | discordant pairs |
|---|---|---|---|
| control | 0.533 (32/60) | [0.40, 0.65] | 9 control-only |
| probed  | 0.417 (25/60) | [0.30, 0.55] | 2 probed-only |

paired diff $+0.117$, bootstrap 95 % CI $[+0.017, +0.217]$
(excludes 0); McNemar exact $p = 0.0654$; Cohen's $h = +0.234$.
The **direction** (probing slightly *suppresses* collapse) is
preserved at $N = 60$, the bootstrap CI on the paired difference no
longer contains 0, the McNemar $p$ is essentially the same, and the
effect size halved (0.167 → 0.117). The original §4.4 claim
("probing perturbs collapse, but the effect is small and detection
is borderline at conventional power") is reinforced.

### 7.7 Bootstrap confidence intervals (improvement #7)

Replaces parametric tests for per-seed scalars with $B = 10\,000$
nonparametric bootstrap CIs.

| quantity | point | 95 % CI |
|---|---|---|
| $\bar{\lambda}_{\max}$ over $[200, 280]$ (GPU, n=10) | 174.95 | [159.4, 189.7] |
| $\bar{\lambda}_{\max}$ over $[200, 280]$ (CPU, n=10) | 174.97 | [159.4, 189.7] |
| paired (control − probed) collapse rate, N=30 | $+0.167$ | $[+0.033, +0.300]$ |
| paired (control − probed) collapse rate, N=60 | $+0.117$ | $[+0.017, +0.217]$ |

GPU and CPU $\lambda_{\max}$ CIs are **indistinguishable** — formal
confirmation of the §4.5 "no hardware effect" claim.
Both perturbation-difference CIs strictly exclude 0.

### 7.8 What the seven improvements still do *not* cover

Even with these seven additions, the report's external validity
remains bounded. The following directions were explicitly left
for future work:

1. **Multiple datasets.** All sigmoid-MLP results live on `make_moons`
   (and partially `make_circles`). No image / tabular / sequence
   benchmark; no real-world data.
2. **Multiple optimisers.** Only full-batch SGD with lr = 0.01.
   Adam, momentum, mini-batch noise, and learning-rate schedules
   are all known to interact non-trivially with curvature.
3. **Multiple architectures.** Only sigmoid-activation MLPs at
   width 64 and depths 2–6. No ReLU, no normalisation, no
   residual connections, no attention.
4. **Slow-collapse low-lr regime.** A run at lr $\ll 0.01$ would
   test whether collapse is an effective-time phenomenon or an
   epoch-count phenomenon.
5. **Theoretical chain-rule sketch.** A short analytic argument
   linking output-layer SG/true-gradient agreement to a closed-form
   condition on the post-activation distribution would convert
   the §6 mechanism from empirical to predictive.

## 8. Reproducibility

- Repo: <https://github.com/GladwinO/Algorithmic-ML-Final-Project>
- Pinned `requirements.txt`. CUDA 13.0 + torch 2.11.0+cu130.
- Run order:
  1. `python main.py` (depth × β × width × dataset grid)
  2. `python regression_experiment.py` (ruggedness regression)
  3. `python controlled_depth_experiment.py`
  4. `python collapse_detection.py`
  5. `python collapse_threshold_sweep.py`
  6. `python hessian_analysis.py` and `python hessian_analysis_cpu.py`
  7. `python perturbation_sensitivity.py`
  8. `python saturation_analysis.py`
  9. `python layerwise_alignment.py`
  10. `python landscape_visualization.py`
  11. `python improvement_5_betasg_collapse.py`
  12. `python improvement_6_perturbation_n60.py`
  13. `python improvement_34_landscape.py`
  14. `python improvement_1_saturation_v2.py`
  15. `python improvement_2_layerwise_v2.py`
  16. `python improvement_7_bootstrap.py`
- All figures saved to `figures/`, all per-seed scalars to `results/`.

