"""Collect per-seed per-epoch traces needed to add error bands to:

  * Figure 1: depth_controlled_vs_uncontrolled_comparison.png
  * Figure 2: collapse_detection_4layer.png
  * Figure 6: layerwise_cosine_all_seeds.png

Writes the resulting traces to ``results/per_seed_traces.json``. This is a
one-shot data-collection step (~5-10 minutes on CPU); the regen script
``regenerate_figures_with_errorbars.py`` then reads from this file.

The depth=4, width=64 instrumented run is reused for all three uses
(uncontrolled[4], controlled[4], and layerwise) since the controlled-width
search returns width=64 for depth=4 by construction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

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
OUT = RESULTS / "per_seed_traces.json"

N_SEEDS = 40
N_EPOCHS = 300


# ---------------------------------------------------------------------------
# helpers (mirroring controlled_depth_experiment.py / collect_per_seed_data.py)
# ---------------------------------------------------------------------------
def _params_for(num_layers: int, h: int, in_dim: int = 2, out_dim: int = 1) -> int:
    p = in_dim * h + h
    p += (num_layers - 1) * (h * h + h)
    p += h * out_dim + out_dim
    return p


def controlled_width(num_layers: int, target: int,
                     h_min: int = 8, h_max: int = 4000) -> int:
    best_h, best_diff = h_min, float("inf")
    for h in range(h_min, h_max + 1):
        diff = abs(_params_for(num_layers, h) - target)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


def _per_layer_grads(net) -> List[torch.Tensor]:
    out = []
    for layer in net.layers:
        parts = []
        if layer.weight.grad is not None:
            parts.append(layer.weight.grad.detach().cpu().flatten())
        if layer.bias is not None and layer.bias.grad is not None:
            parts.append(layer.bias.grad.detach().cpu().flatten())
        out.append(torch.cat(parts) if parts else torch.zeros(0))
    return out


def _instrumented_4layer(seed: int) -> Dict[str, List]:
    """Single depth=4, width=64 run that records *everything* needed for
    figures 2 and 6: cosine_sim_total, grad_norm_true, true_loss, and
    per-layer cosines per epoch."""
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

    cos_total, gnorm, tloss, cos_per_layer = [], [], [], []

    for _ in range(cfg["n_epochs"]):
        opt_true.zero_grad()
        loss_true = loss_fn(true_net(X), y)
        loss_true.backward()
        gt_layers = _per_layer_grads(true_net)
        gt_flat = torch.cat(gt_layers)

        surr_net.load_state_dict(true_net.state_dict())
        opt_surr.zero_grad()
        loss_surr = loss_fn(surr_net(X), y)
        loss_surr.backward()
        gs_layers = _per_layer_grads(surr_net)
        gs_flat = torch.cat(gs_layers)

        cos_total.append(cosine_similarity(gt_flat, gs_flat))
        gnorm.append(float(torch.norm(gt_flat).item()))
        tloss.append(float(loss_true.item()))
        cos_per_layer.append([
            float("nan") if gt.numel() == 0 else cosine_similarity(gt, gs)
            for gt, gs in zip(gt_layers, gs_layers)
        ])

        opt_true.step()

    return dict(cosine_sim=cos_total, grad_norm_true=gnorm,
                true_loss=tloss, cos_per_layer=cos_per_layer,
                n_layers=len(cos_per_layer[0]))


def _cosine_trace(hidden_dim: int, num_layers: int, seed: int) -> List[float]:
    cfg = dict(hidden_dim=hidden_dim, num_layers=num_layers, beta_f=50,
               beta_sg=5, dataset="moons", n_epochs=N_EPOCHS, lr=0.01,
               seed=seed)
    h = run_experiment(cfg)
    return list(map(float, h["cosine_sim"]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    out: Dict = {"meta": dict(n_seeds=N_SEEDS, n_epochs=N_EPOCHS)}

    # 3) Instrumented depth=4 baseline (used for fig 2, fig 6, and reused
    #    as both depth_uncontrolled[4] and depth_controlled[4] traces).
    print(f"\n[1/3] Instrumented depth=4, width=64 ({N_SEEDS} seeds)")
    inst_runs = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        inst_runs.append(_instrumented_4layer(seed))
        print(f"  seed={seed}: final cos={inst_runs[-1]['cosine_sim'][-1]:.3f}"
              f"  ({time.time() - t0:.1f}s)")
    out["layerwise_4layer"] = dict(
        cosine_sim=[r["cosine_sim"] for r in inst_runs],
        grad_norm_true=[r["grad_norm_true"] for r in inst_runs],
        true_loss=[r["true_loss"] for r in inst_runs],
        cos_per_layer=[r["cos_per_layer"] for r in inst_runs],
        n_layers=inst_runs[0]["n_layers"],
    )

    # 1) Uncontrolled depth (width=64, depths 1..4). depth=4 reuses inst_runs.
    print("\n[2/3] Uncontrolled depth (width=64) cosine traces")
    out["depth_uncontrolled"] = {}
    for d in (1, 2, 3, 4):
        if d == 4:
            traces = [r["cosine_sim"] for r in inst_runs]
            print(f"  depth={d}: reusing instrumented runs")
        else:
            traces = []
            t0 = time.time()
            for seed in range(N_SEEDS):
                traces.append(_cosine_trace(64, d, seed))
            print(f"  depth={d}: {N_SEEDS} seeds  ({time.time() - t0:.1f}s)")
        out["depth_uncontrolled"][str(d)] = traces

    # 2) Capacity-controlled depth.
    ref_params = sum(p.numel() for p in MLPTrue(2, 64, 4, beta_f=50).parameters())
    out["depth_controlled"] = {"target_params": ref_params, "widths": {},
                               "traces": {}}
    print(f"\n[3/3] Capacity-controlled depth (target params = {ref_params})")
    for d in (1, 2, 3, 4):
        w = controlled_width(d, ref_params)
        out["depth_controlled"]["widths"][str(d)] = w
        if d == 4 and w == 64:
            traces = [r["cosine_sim"] for r in inst_runs]
            print(f"  depth={d}, width={w}: reusing instrumented runs")
        else:
            traces = []
            t0 = time.time()
            for seed in range(N_SEEDS):
                traces.append(_cosine_trace(w, d, seed))
            print(f"  depth={d}, width={w}: {N_SEEDS} seeds  "
                  f"({time.time() - t0:.1f}s)")
        out["depth_controlled"]["traces"][str(d)] = traces

    OUT.write_text(json.dumps(out))
    print(f"\nSaved {OUT}  ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
