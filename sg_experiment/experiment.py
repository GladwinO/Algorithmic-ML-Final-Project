"""Main per-epoch experiment loop."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .data import get_dataset
from .device import DEVICE
from .metrics import (
    cosine_similarity,
    get_flat_gradients,
    relative_magnitude,
    sign_agreement,
)
from .models import MLPSurrogate, MLPTrue


def run_experiment(config: Dict) -> Dict[str, List[float]]:
    """Train ``true_net`` while at every step also computing the surrogate gradient
    of an identically-weighted ``surr_net`` and recording comparison metrics.
    """
    torch.manual_seed(config.get("seed", 0))

    X, y = get_dataset(config["dataset"])

    true_net = MLPTrue(2, config["hidden_dim"], config["num_layers"], config["beta_f"]).to(DEVICE)
    surr_net = MLPSurrogate(
        2,
        config["hidden_dim"],
        config["num_layers"],
        config["beta_f"],
        config["beta_sg"],
    ).to(DEVICE)
    surr_net.load_state_dict(true_net.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    opt_true = torch.optim.SGD(true_net.parameters(), lr=config["lr"])
    opt_surr = torch.optim.SGD(surr_net.parameters(), lr=config["lr"])

    history: Dict[str, List[float]] = {
        "cosine_sim": [],
        "sign_agreement": [],
        "relative_magnitude": [],
        "true_loss": [],
        "surr_loss": [],
        "epoch": [],
    }

    for epoch in range(config["n_epochs"]):
        true_net.train()
        opt_true.zero_grad()
        pred_true = true_net(X)
        loss_true = loss_fn(pred_true, y)
        loss_true.backward()
        g_true = get_flat_gradients(true_net)

        surr_net.load_state_dict(true_net.state_dict())
        surr_net.train()
        opt_surr.zero_grad()
        pred_surr = surr_net(X)
        loss_surr = loss_fn(pred_surr, y)
        loss_surr.backward()
        g_surr = get_flat_gradients(surr_net)

        history["cosine_sim"].append(cosine_similarity(g_true, g_surr))
        history["sign_agreement"].append(sign_agreement(g_true, g_surr))
        history["relative_magnitude"].append(relative_magnitude(g_true, g_surr))
        history["true_loss"].append(loss_true.item())
        history["surr_loss"].append(loss_surr.item())
        history["epoch"].append(epoch)

        opt_true.step()

    return history
