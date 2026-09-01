# `scripts/` — portable run scripts

Plain shell wrappers around the two Python entry points (`train.py`, `plan.py`). They run
directly on the current machine — no scheduler or container assumed. On a cluster, wrap each
invocation in your own job submission.

## Prerequisites

1. Activate the environment (see the top-level [`README.md`](../README.md#installation)):
   ```bash
   conda activate lpwm
   ```
2. Point `DATASET_DIR` at the dataset root (the folder holding `pusht_noise/` and `wall_single/`):
   ```bash
   export DATASET_DIR=/path/to/data
   ```
3. Optional: `wandb login` to log runs, or `export WANDB_MODE=offline` to skip.

PushT (`pymunk`/`pygame`) and Wall (`numpy`) are pure-Python — no simulator install is needed, and
the scripts set `SDL_VIDEODRIVER=dummy` for headless pygame rendering automatically.

Checkpoints/run dirs are written under `CKPT_BASE` (default `./runs`); override with `export CKPT_BASE=...`.

## Scripts

| script | what it does |
|---|---|
| `train.sh` | one training run of the from-scratch JEPA world model |
| `plan.sh`  | CEM + MPC planning eval of a trained checkpoint |
| `reproduce_pusht.sh` | drives the full PushT sparsity-vs-linearity grid (train + eval per cell) |

### SLURM harness (4-hour wall limit)

Every GPU partition here caps at 4h, so a run that needs longer must span several
jobs. `train.py` checkpoints on a timer and on `SIGUSR1`, records `batch_idx`, and
skips forward on resume, so a chopped run trains each batch exactly once.

| script | what it does |
|---|---|
| `train_slurm.sbatch` | one 3h55m window for a single cell; forwards `SIGUSR1` to python |
| `submit_until_done.sh` | pre-submits N windows chained `afterany`; each no-ops once `DONE` exists |
| `noop_slurm.sbatch` | ~1 min job with the same header, to prove it schedules and sees a GPU |
| `launch_when_ready.sh` | blocks until the controller answers, validates the header, fires the probe |
| `verify_preemption.sh` | deliberate mid-epoch `scancel`, then checks the epoch count and wandb id |
| `run_campaign.sh` | submits one Pi-WM gate's arms x seeds, or CEM-evals them with `EVAL=1` |

Each script's header comment lists its positional args and env-var knobs.

## Examples

```bash
# sparse LpWM (mu=0), Deep-AdaLN(k) predictor, D=384, on PushT, 2 epochs:
PREDICTOR=ar_adaln PROJ_DIM=384 MU=0 MUP=1 MUP_LR=1e-4 REG_WEIGHT=0.5 RUN_NAME=my_lpwm \
  scripts/train.sh pusht 5 3 2 64 reprelu cls 1 b

# plan with the trained checkpoint:
scripts/plan.sh plan_lewm.yaml my_lpwm latest 50 10

# preview the full reproduction grid (prints the per-cell commands), then run it:
bash scripts/reproduce_pusht.sh
RUN=1 bash scripts/reproduce_pusht.sh

# --- Pi-WM campaign, chained across 4h windows ---
# probe cell (fp32, 2 epochs), whose wall-clock sets the precision decision
# and whose measured l0_frac sets k for the matched-rho arm:
scripts/launch_when_ready.sh

# preemption check: kill mid-epoch and assert the resume is lossless
scripts/verify_preemption.sh

# gates. KWTA_MATCHED comes from the probe's l0_frac: k = round(rho * 384).
KWTA_MATCHED=192 scripts/run_campaign.sh step2
scripts/run_campaign.sh step3 step4       # 18 runs, not 21: the controls are shared
EVAL=1 scripts/run_campaign.sh step2      # CEM eval once training is DONE

# analysis and figures
python analysis/predictive_jaccard.py --run_dir runs/outputs/probe_pusht_mlpvar_pd384_s0
python analysis/figures.py --step1 runs/outputs/probe_pusht_mlpvar_pd384_s0/analysis_step1.json \
    --campaign campaign.json --runs 'runs/outputs/*' --out figures/
python analysis/figures.py --selftest --out /tmp/figs   # render all 13 panels on synthetic data
```
