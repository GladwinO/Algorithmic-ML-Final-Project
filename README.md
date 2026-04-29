# Surrogate vs True Gradient — Empirical Extension

Empirical extension of Gygax & Zenke (2024), Section 4. We compare the
surrogate gradient (SG) against the true gradient in deeper, wider sigmoid
MLPs trained on nonconvex 2D classification tasks (`moons`, `circles`).

## Layout

- `sg_experiment/models.py` — `MLPTrue`, `MLPSurrogate`, custom autograd op.
- `sg_experiment/data.py` — moons/circles loaders.
- `sg_experiment/metrics.py` — cosine sim, sign agreement, relative magnitude.
- `sg_experiment/experiment.py` — per-epoch training loop recording both gradients.
- `sg_experiment/plots.py` — figure helpers.
- `main.py` — runs the four experiments (depth, beta_sg, width, dataset).

## Run

All dependencies are installed into a project-local virtual environment so
nothing leaks onto the host machine.

```bash
# 1. create an isolated env (one-time)
python3 -m venv .venv

# 2. activate it
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows PowerShell

# 3. install pinned dependencies into the env
pip install --upgrade pip
pip install -r requirements.txt

# 4. run
python main.py
python regression_experiment.py
```

### GPU support

The code auto-detects a CUDA GPU and moves all tensors/models to it via
`sg_experiment/device.py`. On Linux the default `torch` wheel from PyPI
already bundles the CUDA 12.x / 13.x runtime, so no extra steps are needed
on a CUDA-capable machine — just run `pip install -r requirements.txt`
inside the venv. On a CPU-only machine the same wheel still runs (slower).

To force a specific device, set `SG_DEVICE`:

```bash
SG_DEVICE=cpu python main.py        # force CPU
SG_DEVICE=cuda:0 python main.py     # force first GPU
```

Quick sanity check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Figures land in `figures/`; raw histories in `results/all_histories.json`.
