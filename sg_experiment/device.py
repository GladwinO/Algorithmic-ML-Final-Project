"""Single source of truth for the compute device.

Honours ``SG_DEVICE`` env var (e.g. ``cpu``/``cuda``/``cuda:0``); otherwise
auto-selects CUDA when available.
"""
from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    override = os.environ.get("SG_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
