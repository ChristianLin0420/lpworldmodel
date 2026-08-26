# lpwm_swm — LpWM on `stable-worldmodel` (Piecewise & OGBench-Cube)

This folder holds the **Piecewise** and **OGBench-Cube** experiments from the LpWM
paper. Unlike the top-level PushT/Wall codebase (built on DINO-WM), these are
built on **[`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel)**. Following the [`le-wm`](https://github.com/lucas-maes/le-wm)
pattern, this is a **minimal package**: it depends on the public
`stable_worldmodel` and `stable_pretraining` libraries and owns only the LpWM
method — the model, the sparse **RepReLU** link, and the **RDMReg**
regularizer. The Piecewise and Cube **environments themselves come from upstream
stable-worldmodel** (`swm/Piecewise-v0`, `swm/OGBCube-v0`); we add only the model,
configs, and entry points.

## What's here

```
lpwm_swm/
├── model.py      # LpWM: encoder -> projector(link) -> emb; AdaLN predictor; rollout (Dynamics contract)
├── module.py     # RepReLU link, MLP, AdaLN Predictor, Embedder
├── loss.py       # RDMReg (sliced-Wasserstein to a Rectified Generalized Gaussian) + TemporalJaccard
├── metrics.py    # l0/l1 sparsity, per-dim variance, off-diagonal covariance diagnostics
├── train.py      # Hydra training app (spt.Module/Manager); loss = pred_loss + w*rdmreg [+ w*temporal_jaccard]
├── eval.py       # Dataset-driven MPC eval (ShootingCostEvaluator + GoalMSE + CEM solver)
└── config/       # train config (config/lpwm.yaml, data/) and plan config (config/plan/)
```

The planning **cost** lives in the public
`stable_worldmodel.planning.ShootingCostEvaluator` + a pluggable `Objective`
(`GoalMSE` reproduces the model-owned criterion), not on the model — matching the
current stable-worldmodel API.

## Dependency & environment

Depends on public **stable-worldmodel** with the `train`, `env`, and `format`
extras (which pull `stable-pretraining>=0.1.8`, `ogbench`, `gymnasium[all]`,
`mujoco`, and the HDF5 dataset backend).

### Installation

This is a **separate environment** from the top-level `lpwm` env (different torch
and stack) — keep the two distinct. `environment.yaml` in this directory pins the
whole thing (torch 2.7.1 + stable-worldmodel from the known-good upstream commit):

```bash
# 1) system libs first (mujoco EGL renderer, video IO, box2d build):
sudo apt-get install -y libegl1 libopengl0 libgles2 libglib2.0-0 ffmpeg swig git

# 2) create + activate the lpwm_swm env (run from the repo root):
conda env create -f lpwm_swm/environment.yaml
conda activate lpwm_swm
```

At runtime set `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` for headless rendering, and
`STABLEWM_HOME` for the dataset/checkpoint root (see **Data** below). Run the
entry points from *inside* `lpwm_swm/` (see **Train**).

> **Version note.** The `stable_worldmodel.planning` architecture this code
> targets is on upstream `main` (commit `addbab4`, `0.1.1-41`), **not** the PyPI
> `0.1.1` release. Until a PyPI release includes it, install from git:
> `pip install "stable-worldmodel[train,env,format] @ git+https://github.com/galilai-group/stable-worldmodel@main"`.
>
> **torch pin.** Pin the torch stack to a version with a working `torchaudio`
> (the base container's `torch==2.7.1`); an unconstrained install may pull a
> bleeding-edge torch whose prebuilt `torchaudio` fails to load and breaks the
> `transformers`-based ViT encoder.

## Data

- **Piecewise** — collect with the upstream env + a random policy (dataset name
  `piecewise/random_grid_n_2_render_none_2222.h5` under `$STABLEWM_HOME/datasets`).
- **OGBench-Cube** — the `ogbench` `cube_single_expert` dataset.

Datasets and checkpoints live under `$STABLEWM_HOME` (set it, e.g.
`export STABLEWM_HOME=~/.stable_worldmodel`). The OGBench-Cube dataset can be found in https://huggingface.co/datasets/quentinll/lewm-cube/tree/main. We will release the dataset files for Piecewise soon. 

## Train

Run the commands below **from this `lpwm_swm/` directory** (flat le-wm-style
layout: the entry points import their sibling modules as top-level, so `lpwm_swm/`
must be the working directory — this also keeps the parent repo's `datasets/`
package from shadowing HuggingFace `datasets`).

```bash
# sparse LpWM on Piecewise (grid_n=2), RDMReg weight 25.
# target_dist_rms_norm defaults to OFF; the paper's sparse runs opt in (=true)
# to RMS-normalize the target and stabilize feature scale.
python train.py data=piecewise output_model_name=lpwm_piecewise \
    loss.rdmreg.weight=25 loss.rdmreg.kwargs.target_dist_rms_norm=true

# dense LeWM control: identity link, Gaussian target
python train.py data=piecewise output_model_name=lewm_piecewise \
    loss.rdmreg.kwargs.link_function_type=Identity loss.rdmreg.kwargs.p=2.0

# OGBench-Cube (Temporal-Jaccard on, weight 0.1)
python train.py data=ogb_cube output_model_name=lpwm_ogb_cube \
    loss.temporal_jaccard.enabled=true loss.temporal_jaccard.weight=0.1
```

## Evaluate (dataset-driven MPC)

```bash
python eval.py --config-name=piecewise policy=lpwm_piecewise
```

## Attribution

Built on **stable-worldmodel** (galilai-group). The Piecewise and OGBench-Cube
environments are upstream stable-worldmodel components. See `NOTICE`.
