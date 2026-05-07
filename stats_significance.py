"""Statistical significance analysis for the SG-vs-true study.

Reads saved per-experiment JSON in ``results/`` and writes ``stats_results.txt``.

Where the required per-seed arrays were not persisted to disk (e.g. per-seed
final cosine for the depth ablation, per-layer per-seed cosines, raw
saturation/cosine pairs), the relevant section is emitted as a clearly marked
``[SKIPPED]`` block rather than fabricating numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "stats_results.txt"

LINES: list[str] = []


def emit(s: str = "") -> None:
    LINES.append(s)


def header(title: str) -> None:
    emit("")
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def trailing_mean(xs: Iterable[float], k: int = 50) -> float:
    arr = np.asarray(list(xs), dtype=float)
    return float(arr[-k:].mean())


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
all_hist = json.loads((RESULTS / "all_histories.json").read_text())
beta_sweep = json.loads((RESULTS / "beta_sg_collapse_sweep.json").read_text())
pert_n60 = json.loads((RESULTS / "perturbation_sensitivity_n60.json").read_text())
hess_gpu = json.loads((RESULTS / "hessian_gpu.json").read_text())
hess_cpu = json.loads((RESULTS / "hessian_cpu.json").read_text())
PER_SEED_PATH = RESULTS / "per_seed_data.json"
per_seed = json.loads(PER_SEED_PATH.read_text()) if PER_SEED_PATH.exists() else None


def bonferroni(pvals):
    m = len(pvals)
    return [min(1.0, p * m) for p in pvals]


def welch_t(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = stats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite dof
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    dof = num / den if den > 0 else float("nan")
    return float(res.statistic), float(dof), float(res.pvalue)


# ---------------------------------------------------------------------------
# 1. Chain-length effect (depth)  --  uses results/per_seed_data.json
# ---------------------------------------------------------------------------
header("1. CHAIN-LENGTH EFFECT (depth ablation, per-seed)")

if per_seed is None:
    emit("[SKIPPED] results/per_seed_data.json missing.")
    emit("Run `python collect_per_seed_data.py` first.")
    depth_finals = {}
else:
    depth_finals = {int(k): list(v) for k, v in per_seed["depth_uncontrolled"].items()}
    emit(f"N seeds = {per_seed['meta']['n_seeds']}, window = last "
         f"{per_seed['meta']['window']} epochs of {per_seed['meta']['n_epochs']}.")
    emit("Per-depth final cosine (mean of last 50 epochs):")
    for d in sorted(depth_finals):
        v = depth_finals[d]
        m = float(np.mean(v))
        s = float(np.std(v, ddof=1))
        emit(f"  depth={d}: n={len(v)}, mean={m:.4f}, sd={s:.4f}")

    groups = [depth_finals[d] for d in sorted(depth_finals)]
    f, p = stats.f_oneway(*groups)
    emit(f"\nOne-way ANOVA across depths: F={f:.4f}, p={fmt_p(p)}")

    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    raw_p = []
    raw_t = []
    raw_df = []
    for a, b in pairs:
        t, dof, p = welch_t(depth_finals[a], depth_finals[b])
        raw_t.append(t)
        raw_df.append(dof)
        raw_p.append(p)
    adj = bonferroni(raw_p)
    emit("\nPairwise Welch t-tests with Bonferroni correction (m=6):")
    emit(f"  {'pair':<10}{'t':>10}{'dof':>10}{'raw p':>14}{'adj p':>14}")
    for (a, b), t, dof, rp, ap in zip(pairs, raw_t, raw_df, raw_p, adj):
        emit(f"  d{a} vs d{b:<5}{t:>10.3f}{dof:>10.2f}{fmt_p(rp):>14}{fmt_p(ap):>14}")

# Combined OLS regression across all axes
emit("")
emit("OLS regression (final cosine ~ depth + log(beta_sg) + log(width) + dataset)")
emit("using all_histories.json (1 seed per cell, original ablation grid):")

rows = []
for axis in ("depth", "beta", "width", "dataset"):
    block = all_hist[axis]
    for cfg, hist in zip(block["configs"], block["histories"]):
        rows.append(
            dict(
                final=trailing_mean(hist["cosine_sim"], 50),
                depth=int(cfg["num_layers"]),
                log_beta=math.log(float(cfg["beta_sg"])),
                log_width=math.log(float(cfg["hidden_dim"])),
                dataset=1.0 if cfg["dataset"] == "moons" else 0.0,
            )
        )

y = np.array([r["final"] for r in rows])
X = np.column_stack(
    [
        np.ones(len(rows)),
        [r["depth"] for r in rows],
        [r["log_beta"] for r in rows],
        [r["log_width"] for r in rows],
        [r["dataset"] for r in rows],
    ]
)
names = ["intercept", "depth", "log(beta_sg)", "log(width)", "dataset[moons]"]

beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta_hat
n, k = X.shape
dof = max(n - k, 1)
sigma2 = float(resid @ resid / dof)
cov = sigma2 * np.linalg.pinv(X.T @ X)
se = np.sqrt(np.diag(cov))
tcrit = stats.t.ppf(0.975, dof)

emit(f"  n={n}, residual dof={dof}, sigma^2={sigma2:.4g}")
emit(f"  {'name':<18}{'coef':>10}{'SE':>10}{'95% CI':>26}{'t':>8}{'p':>10}")
for name, b, s in zip(names, beta_hat, se):
    lo, hi = b - tcrit * s, b + tcrit * s
    t_stat = b / s if s > 0 else float("nan")
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if s > 0 else float("nan")
    emit(f"  {name:<18}{b:>10.4f}{s:>10.4f}{f'[{lo:+.3f}, {hi:+.3f}]':>26}{t_stat:>8.2f}{fmt_p(p_val):>10}")

emit("")
emit("Note: configs are one-factor-at-a-time around a baseline, not a full grid;")
emit("the OLS coefficients quantify direction/magnitude but the design is unbalanced.")


# ---------------------------------------------------------------------------
# 2. Controlled depth experiment
# ---------------------------------------------------------------------------
header("2. CONTROLLED-DEPTH EXPERIMENT (capacity held fixed, per-seed)")
if per_seed is None:
    emit("[SKIPPED] results/per_seed_data.json missing.")
else:
    ctrl = per_seed["depth_controlled"]
    emit(f"Target params = {ctrl['target_params']}; widths per depth: {ctrl['widths']}")
    emit("Per-depth final cosine (mean of last 50 epochs):")
    for d in sorted(ctrl["finals"], key=int):
        vals = ctrl["finals"][d]
        emit(f"  depth={d} (width={ctrl['widths'][d]}): n={len(vals)}, "
             f"mean={np.mean(vals):.4f}, sd={np.std(vals, ddof=1):.4f}")
    pairs = [(2, 3), (3, 4), (2, 4)]
    raw_p, raw_t, raw_df = [], [], []
    for a, b in pairs:
        t, dof, p = welch_t(ctrl["finals"][str(a)], ctrl["finals"][str(b)])
        raw_t.append(t); raw_df.append(dof); raw_p.append(p)
    adj = bonferroni(raw_p)
    emit("\nPairwise Welch t-tests with Bonferroni correction (m=3):")
    emit(f"  {'pair':<10}{'t':>10}{'dof':>10}{'raw p':>14}{'adj p':>14}")
    for (a, b), t, dof, rp, ap in zip(pairs, raw_t, raw_df, raw_p, adj):
        emit(f"  d{a} vs d{b:<5}{t:>10.3f}{dof:>10.2f}{fmt_p(rp):>14}{fmt_p(ap):>14}")


# ---------------------------------------------------------------------------
# 3. Collapse rate (4-layer moons baseline)
# ---------------------------------------------------------------------------
header("3. COLLAPSE RATE (binomial test)")

# beta_sg = 5 cell of the sweep matches the baseline 4-layer moons setup.
baseline = beta_sweep["5"]
n_baseline = len(baseline["finals"])
k_baseline = baseline["n_collapsed"]
emit(f"Baseline (depth=4, moons, beta_sg=5): {k_baseline}/{n_baseline} seeds collapsed.")

bt0 = stats.binomtest(k_baseline, n=n_baseline, p=1e-12, alternative="greater")
emit(f"Exact binomial test, H0: p=0 (vs greater): p-value = {fmt_p(bt0.pvalue)}")
ci0 = stats.binomtest(k_baseline, n=n_baseline).proportion_ci(method="exact")
emit(f"Observed proportion = {k_baseline / n_baseline:.3f}, 95% CI (Clopper-Pearson) = [{ci0.low:.3f}, {ci0.high:.3f}]")
emit("(H0: p=0 is degenerate; with k>=1 the p-value is exactly 0 in the limit.")
emit(" The CI is the more informative quantity here.)")


# ---------------------------------------------------------------------------
# 4. Collapse timing
# ---------------------------------------------------------------------------
header("4. COLLAPSE TIMING (n=3 detections, t-distribution CI)")

epochs = [e for e in baseline["epochs"] if e is not None]
emit(f"Detected collapse epochs: {epochs}")
arr = np.asarray(epochs, dtype=float)
m = float(arr.mean())
sd = float(arr.std(ddof=1))
n_e = len(arr)
tcrit_e = stats.t.ppf(0.975, df=n_e - 1)
half = tcrit_e * sd / math.sqrt(n_e)
emit(f"Mean = {m:.2f}, SD = {sd:.2f}, n = {n_e}")
emit(f"95% CI (t, df={n_e - 1}): [{m - half:.2f}, {m + half:.2f}]")
emit("NOTE: small-sample CI; df=2 yields a wide t-critical (~4.30) and the CI is")
emit("highly sensitive to the three observations.")


# ---------------------------------------------------------------------------
# 5. Per-layer cosine similarity
# ---------------------------------------------------------------------------
header("5. PER-LAYER COSINE SIMILARITY (pairwise t-tests, Bonferroni)")
if per_seed is None or "layerwise" not in per_seed:
    emit("[SKIPPED] results/per_seed_data.json missing per-layer block.")
else:
    arr = np.asarray(per_seed["layerwise"]["per_layer_finals"])  # (seeds, n_layers)
    n_layers = arr.shape[1]
    emit(f"Per-seed final cosine (mean of last 50 epochs), {arr.shape[0]} seeds x "
         f"{n_layers} Linear layers (index 0 = input-side, last = readout):")
    for li in range(n_layers):
        col = arr[:, li]
        emit(f"  layer {li}: mean={np.nanmean(col):.4f}, sd={np.nanstd(col, ddof=1):.4f}")
    emit("")
    emit("NOTE: the readout layer (highest index) is identical for true vs surrogate")
    emit("by construction (no surrogate activation between it and the loss), so its")
    emit("cosine is 1.0. Statistical comparisons are between hidden layers only.")

    # Hidden layers are 0 .. n_layers-2
    hidden_pairs = [(3, 2), (2, 1), (1, 0)]
    raw_p, raw_t, raw_df = [], [], []
    for a, b in hidden_pairs:
        # paired t-test (same seed, different layer)
        diffs = arr[:, a] - arr[:, b]
        diffs = diffs[~np.isnan(diffs)]
        res = stats.ttest_rel(arr[:, a], arr[:, b], nan_policy="omit")
        raw_t.append(float(res.statistic))
        raw_df.append(int(np.sum(~np.isnan(arr[:, a] - arr[:, b]))) - 1)
        raw_p.append(float(res.pvalue))
    adj = bonferroni(raw_p)
    emit("Pairwise paired t-tests (m=3, Bonferroni):")
    emit(f"  {'pair':<14}{'t':>10}{'dof':>8}{'raw p':>14}{'adj p':>14}")
    for (a, b), t, dof, rp, ap in zip(hidden_pairs, raw_t, raw_df, raw_p, adj):
        emit(f"  L{a} vs L{b:<10}{t:>10.3f}{dof:>8}{fmt_p(rp):>14}{fmt_p(ap):>14}")


# ---------------------------------------------------------------------------
# 6. Saturation correlation
# ---------------------------------------------------------------------------
header("6. SATURATION vs COSINE CORRELATION")
if per_seed is None or "saturation" not in per_seed:
    emit("[SKIPPED] results/per_seed_data.json missing saturation block.")
else:
    pairs = np.asarray(per_seed["saturation"]["pairs"])
    sat = pairs[:, 0]
    cos = pairs[:, 1]
    emit(f"N pairs (epoch x seed) = {len(pairs)}")
    r, p = stats.pearsonr(sat, cos)
    rho, p_sp = stats.spearmanr(sat, cos)
    emit(f"Pearson r  = {r:+.4f}, two-sided p = {fmt_p(p)}")
    emit(f"Spearman rho = {rho:+.4f}, two-sided p = {fmt_p(p_sp)}")


# ---------------------------------------------------------------------------
# 7. beta_sg sweep
# ---------------------------------------------------------------------------
header("7. BETA_SG SWEEP")

betas = sorted(int(k) for k in beta_sweep)
n_collapse = [beta_sweep[str(b)]["n_collapsed"] for b in betas]
n_total = [len(beta_sweep[str(b)]["finals"]) for b in betas]
final_cos = [beta_sweep[str(b)]["mean_final_cos"] for b in betas]

emit("Per-beta_sg summary:")
emit(f"  {'beta_sg':>8}{'collapse':>12}{'mean_final_cos':>20}")
for b, k_c, n_t, fc in zip(betas, n_collapse, n_total, final_cos):
    emit(f"  {b:>8}{f'{k_c}/{n_t}':>12}{fc:>20.4f}")

# Fisher's exact via 2 x 5 chi^2 monte carlo (Fisher exact in scipy is 2x2).
table = np.array([n_collapse, [t - c for t, c in zip(n_total, n_collapse)]])
try:
    res = stats.fisher_exact(table) if table.shape == (2, 2) else None
except Exception:
    res = None
if res is None:
    chi2, p_chi, dof_chi, _ = stats.chi2_contingency(table)
    emit(f"\n2x{len(betas)} contingency (collapse vs no-collapse across beta_sg):")
    emit(f"  chi-square = {chi2:.4f}, dof = {dof_chi}, p = {fmt_p(p_chi)}")
    # also a Monte-Carlo Fisher-style p-value using stats.fisher_exact extension
    try:
        # scipy >=1.13 supports method='exact' for general Fisher via fisher_exact only on 2x2;
        # use stats.permutation_test for the row-margin-preserving permutation p-value.
        rng = np.random.default_rng(0)
        observed = chi2
        # Build the column-label vector and row-label vector, permute columns.
        col_labels = []
        row_labels = []
        for j, b in enumerate(betas):
            col_labels += [j] * n_total[j]
            row_labels += [1] * n_collapse[j] + [0] * (n_total[j] - n_collapse[j])
        col_labels = np.array(col_labels)
        row_labels = np.array(row_labels)
        B = 10000
        count = 0
        for _ in range(B):
            perm = rng.permutation(row_labels)
            tab = np.zeros_like(table)
            for j in range(len(betas)):
                mask = col_labels == j
                tab[1, j] = perm[mask].sum()
                tab[0, j] = mask.sum() - tab[1, j]
            chi_b, _, _, _ = stats.chi2_contingency(tab)
            if chi_b >= observed:
                count += 1
        emit(f"  Permutation p-value (B={B}): {fmt_p((count + 1) / (B + 1))}")
    except Exception as exc:
        emit(f"  (permutation p-value failed: {exc})")

rho, _ = stats.spearmanr(betas, final_cos)
# Exact two-sided p-value for Spearman rho with small n via permutation of ranks.
from itertools import permutations as _perms

n_b = len(betas)
ref_ranks = stats.rankdata(final_cos)
base_ranks = np.arange(1, n_b + 1)
count_ge = 0
total = 0
for perm in _perms(ref_ranks):
    r, _ = stats.spearmanr(base_ranks, np.array(perm))
    if abs(r) >= abs(rho) - 1e-12:
        count_ge += 1
    total += 1
p_exact = count_ge / total
emit(f"\nSpearman correlation (beta_sg, mean_final_cos): rho = {rho:.4f}")
emit(f"  Exact two-sided permutation p (n={n_b}, {total} perms): {fmt_p(p_exact)}")


# ---------------------------------------------------------------------------
# 8. Perturbation sensitivity (paired)
# ---------------------------------------------------------------------------
header("8. PERTURBATION SENSITIVITY (paired, N=60)")

tab = pert_n60["table"]
b_disc = tab["cn"]  # control collapse, probed none
c_disc = tab["nc"]  # control none, probed collapse
emit(f"Discordant pairs: control-only={b_disc}, probed-only={c_disc}")
total_disc = b_disc + c_disc
bt = stats.binomtest(b_disc, n=total_disc, p=0.5, alternative="two-sided")
emit(f"Exact binomial test on discordant pairs ({b_disc}/{total_disc}, H0: p=0.5):")
emit(f"  p-value = {fmt_p(bt.pvalue)}")
emit(f"  McNemar p (from JSON) = {pert_n60['mcnemar_p']:.4f}")
emit(f"  Newcombe 95% CI on rate diff = [{pert_n60['newcombe_ci'][0]:+.4f}, {pert_n60['newcombe_ci'][1]:+.4f}]")


# ---------------------------------------------------------------------------
# Write out
# ---------------------------------------------------------------------------
OUT.write_text("\n".join(LINES) + "\n")
print(f"Wrote {OUT}")
