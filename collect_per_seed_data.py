"""Collect per-seed scalars needed for the statistical tests.

Produces ``results/per_seed_data.json`` with three sections:

* ``depth_uncontrolled``: 4 depths x N seeds, fixed width=64, on moons.
  Each entry is the mean cosine over the last 50 epochs.

* ``depth_controlled``: 4 depths x N seeds, hidden width chosen per depth so
  that total parameter count tracks the depth=4, width=64 reference.

* ``layerwise``: depth=4 x N seeds, baseline config. Saves the mean per-layer
  cosine over the last 50 epochs (one entry per Linear layer, including the
  readout) and a flat list of all (mean-saturation-fraction, cosine-sim)
  pairs across every epoch and seed.

This script is intentionally self-contained -- it pulls only from the
``sg_experiment`` library and does not modify the existing analysis scripts.

Run with ``python collect_per_seed_data.py`` (CPU is fine; ~5 minutes).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sg_experiment.data import get_dataset
from sg_experiment.device import DEVICE
from sg_experiment.experiment import run_experiment
from sg_experiment.metrics import cosine_similarity, get_flat_gradients
from sg_experiment.models import MLPSurrogate, MLPTrue

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "per_seed_data.json"

N_SEEDS = 10
N_EPOCHS = 300
WINDOW = 50
SAT_HIGH = 0.99
SAT_LOW = 0.01


# --------------------------------------------------------------------------
# Capacity-controlled width search (mirrors controlled_depth_experiment.py)
# --------------------------------------------------------------------------
def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _params_for(num_layers: int, h: int, in_dim: int = 2, out_dim: int = 1) -> int:
    p = in_dim * h + h
    p += (num_layers - 1) * (h * h + h)
    p += h * out_dim + out_dim
    return p


def controlled_width(num_layers: int, target: int, h_min: int = 8, h_max: int = 4000) -> int:
    best_h, best_diff = h_min, float("inf")
    for h in range(h_min, h_max + 1):
        diff = abs(_params_for(num_layers, h) - target)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


# --------------------------------------------------------------------------
# Instrumented run: per-layer cosines + saturation/cosine pairs
# --------------------------------------------------------------------------
def _per_layer_grads(net):
    out = []
    for layer in net.layers:
        parts = []
        if layer.weight.grad is not None:
            parts.append(layer.weight.grad.detach().cpu().flatten())
        if layer.bias is not None and layer.bias.grad is not None:
            parts.append(layer.bias.grad.detach().cpu().flatten())
        out.append(torch.cat(parts) if parts else torch.zeros(0))
    return out


def _saturation_fraction(true_net, X, beta_f):
    """Mean fraction of saturated hidden units (across all hidden layers)."""
    fracs = []
    with torch.no_grad():
        h = X
        for layer in true_net.layers[:-1]:
            pre = layer(h)
            sig = torch.sigmoid(beta_f * pre)
            mask = (sig > SAT_HIGH) | (sig < SAT_LOW)
            fracs.append(float(mask.float().mean().item()))
            h = sig
    return float(np.mean(fracs)) if fracs else 0.0


def instrumented_run(seed: int):
    """Returns dict with cosine_sim/epoch lists, per-layer cosines list of lists,
    and a flat list of saturation fractions per epoch."""
    cfg = dict(hidden_dim=64, num_layers=4, beta_f=50, beta_sg=5,
               dataset="moons", n_epochs=N_EPOCHS, lr=0.01, seed=seed)
    torch.manual_seed(cfg["seed"])
    X, y = get_dataset(cfg["dataset"])

    true_net = MLPTrue(2, cfg["hidden_dim"], cfg["num_layers"], cfg["beta_f"]).to(DEVICE)
    surr_net = MLPSurrogate(2, cfg["hidden_dim"], cfg["num_layers"],
                            cfg["beta_f"], cfg["beta_sg"]).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    opt_true = torch.optim.SGD(true_net.parameters(), lr=cfg["lr"])
    opt_surr = torch.optim.SGD(surr_net.parameters(), lr=cfg["lr"])

    cos_total = []
    cos_per_layer = []  # list of length n_layers (5 for depth=4)
    sat_per_epoch = []

    for epoch in range(cfg["n_epochs"]):
        opt_true.zero_grad()
        loss_true = loss_fn(true_net(X), y)
        loss_true.backward()
        gt_layers = _per_layer_grads(true_net)

        surr_net.load_state_dict(true_net.state_dict())
        opt_surr.zero_grad()
        loss_surr = loss_fn(surr_net(X), y)
        loss_surr.backward()
        gs_layers = _per_layer_grads(surr_net)

        per_layer = []
        for gt, gs in zip(gt_layers, gs_layers):
            if gt.numel() == 0:
                per_layer.append(float("nan"))
            else:
                per_layer.append(cosine_similarity(gt, gs))
        cos_per_layer.append(per_layer)
        flat_t = torch.cat(gt_layers)
        flat_s = torch.cat(gs_layers)
        cos_total.append(cosine_similarity(flat_t, flat_s))
        sat_per_epoch.append(_saturation_fraction(true_net, X, cfg["beta_f"]))

        opt_true.step()

    return dict(cosine_sim=cos_total,
                cos_per_layer=cos_per_layer,
                saturation=sat_per_epoch)


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------
def trailing_mean(xs, k=WINDOW):
    return float(np.mean(np.asarray(xs)[-k:]))


def main():
    print(f"Device: {DEVICE}")
    out: dict = {"meta": dict(n_seeds=N_SEEDS, n_epochs=N_EPOCHS, window=WINDOW)}

    # 1) Uncontrolled depth ---------------------------------------------------
    out["depth_uncontrolled"] = {}
    print("\n[1/3] Uncontrolled depth (width=64, moons)")
    for d in (1, 2, 3, 4):
        finals = []
        t0 = time.time()
        for seed in range(N_SEEDS):
            cfg = dict(hidden_dim=64, num_layers=d, beta_f=50, beta_sg=5,
                       dataset="moons", n_epochs=N_EPOCHS, lr=0.01, seed=seed)
            h = run_experiment(cfg)
            finals.append(trailing_mean(h["cosine_sim"]))
        out["depth_uncontrolled"][str(d)] = finals
        print(f"  depth={d}: {[round(v,3) for v in finals]}  ({time.time()-t0:.1f}s)")

    # 2) Capacity-controlled depth -------------------------------------------
    ref_params = count_params(MLPTrue(2, 64, 4, beta_f=50))
    out["depth_controlled"] = {"target_params": ref_params, "widths": {}, "finals": {}}
    print(f"\n[2/3] Capacity-controlled depth (target params = {ref_params})")
    for d in (1, 2, 3, 4):
        w = controlled_width(d, ref_params)
        out["depth_controlled"]["widths"][str(d)] = w
        finals = []
        t0 = time.time()
        for seed in range(N_SEEDS):
            cfg = dict(hidden_dim=w, num_layers=d, beta_f=50, beta_sg=5,
                       dataset="moons", n_epochs=N_EPOCHS, lr=0.01, seed=seed)
            h = run_experiment(cfg)
            finals.append(trailing_mean(h["cosine_sim"]))
        out["depth_controlled"]["finals"][str(d)] = finals
        print(f"  depth={d}, width={w}: {[round(v,3) for v in finals]}  ({time.time()-t0:.1f}s)")

    # 3) Per-layer + saturation pairs (depth=4 baseline) ---------------------
    print(f"\n[3/3] Instrumented depth=4 baseline ({N_SEEDS} seeds)")
    per_layer_finals = []  # one [n_layers] list per seed (mean of last 50 epochs)
    sat_cos_pairs = []      # flat list of (sat, cos) pairs across all seeds and epochs
    for seed in range(N_SEEDS):
        t0 = time.time()
        h = instrumented_run(seed)
        # per-layer final cosines (mean of last 50 epochs, ignoring NaNs)
        arr = np.asarray(h["cos_per_layer"])  # shape (epochs, n_layers)
        per_layer_finals.append([float(np.nanmean(arr[-WINDOW:, li])) for li in range(arr.shape[1])])
        for s, c in zip(h["saturation"], h["cosine_sim"]):
            sat_cos_pairs.append([s, c])
        print(f"  seed={seed}: per-layer final cos = "
              f"{[round(x,3) for x in per_layer_finals[-1]]}  ({time.time()-t0:.1f}s)")

    out["layerwise"] = dict(
        per_layer_finals=per_layer_finals,
        n_layers=len(per_layer_finals[0]),
    )
    out["saturation"] = dict(pairs=sat_cos_pairs)

    OUT.write_text(json.dumps(out))
    print(f"\nSaved {OUT}  ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
