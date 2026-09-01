#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly
# Same partition list as train_slurm.sbatch, and for the same reason: naming a
# partition this cluster does not define (batch_block1/2/3 used to be here) does not
# fail at submit time, it parks the job on Reason=PartitionConfig forever.
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# CEM / MPC planning eval of ONE trained checkpoint.
#
# Why this file exists: scripts/plan.sh runs `python plan.py` INLINE. plan.py's
# submitit path (launch_plan_jobs) is only reached from train.py's epoch-end hook,
# never from plan.sh -- so `EVAL=1 scripts/run_campaign.sh <gate>` would have run
# every CEM eval synchronously on the submit host, which has no GPU at all
# (`nvidia-smi: command not found`). This wraps it in a real GPU allocation.
#
# Not preemption-chained like training: an eval is short and idempotent, so a lost
# window is just resubmitted rather than resumed. There is no DONE sentinel; the
# planner writes its own output dir under plan_outputs/.
#
# Usage (normally via EVAL=1 scripts/run_campaign.sh <gate>):
#   RUN_NAME=LpWM-ltv_pd384_bf16_s0 SEED=0 NEVALS=50 MAXITER=10 \
#     sbatch --job-name=eval_LpWM-ltv_s0 scripts/plan_slurm.sbatch
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO=${SLURM_SUBMIT_DIR}
else
    REPO=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "${REPO}"
mkdir -p slurm_logs

if [ -f "${REPO}/.env" ]; then
    set -a; . "${REPO}/.env"; set +a
else
    echo "ERROR: ${REPO}/.env not found" >&2; exit 1
fi
: "${RUN_NAME:?RUN_NAME is required (the run dir under \$CKPT_BASE/outputs)}"
: "${DATASET_DIR:?set DATASET_DIR in .env}"

RUN_DIR="${CKPT_BASE:-${REPO}/runs}/outputs/${RUN_NAME}"
# Evaluating a half-trained checkpoint silently produces a success rate for a model
# that is not the one the gate is about, so refuse rather than mislead.
if [ ! -f "${RUN_DIR}/DONE" ]; then
    echo "ERROR: ${RUN_DIR}/DONE absent -- training has not finished; refusing to eval." >&2
    exit 1
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LPWM_CONDA_ENV:-lpwm}"

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}   # headless pygame: the planner
                                                   # steps and renders the real env
export TOKENIZERS_PARALLELISM=false
# The planner logs nothing to wandb (plan.py sets wandb_logging=False on the
# submitit path); keep it off here too so an eval cannot create stray runs.
export WANDB_MODE=${LPWM_WANDB_MODE:-disabled}

echo "=== eval job ${SLURM_JOB_ID} on $(hostname) | $(date -Is) ==="
echo "=== run=${RUN_NAME} seed=${SEED:-0} n_evals=${NEVALS:-50} max_iter=${MAXITER:-10} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SEED="${SEED:-0}" scripts/plan.sh plan_lewm.yaml "${RUN_NAME}" latest \
    "${NEVALS:-50}" "${MAXITER:-10}"
echo "=== eval complete: ${RUN_NAME} | $(date -Is) ==="
