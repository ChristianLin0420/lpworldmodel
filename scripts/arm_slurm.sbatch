#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# One PLANNER-VARIANT eval (V2 PiWM-gd, V3 PiWM-2lvl) of one trained checkpoint.
# Same allocation and same DONE-sentinel refusal as scripts/vote_slurm.sbatch, of which
# this is a single-member copy: evaluating a half-trained checkpoint silently produces a
# success rate for a model that is not the one the contrast is about.
#
#   CKPT=LpWM-ltv_pd384_bf16_s3 LABEL=PiWM-gd_pd384_bf16_s3 SEED=3 \
#     PLAN_CONFIG=plan_gd.yaml \
#     sbatch --job-name=eval_PiWM-gd_pd384_bf16_s3 scripts/arm_slurm.sbatch
#
# The job name MUST be eval_${LABEL}: analysis/collect_evals.py reads the eval-seed
# scheme out of slurm_logs/ by that name.
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO=${SLURM_SUBMIT_DIR}; else REPO=$(cd "$(dirname "$0")/.." && pwd); fi
cd "${REPO}"; mkdir -p slurm_logs
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; else echo "ERROR: ${REPO}/.env not found" >&2; exit 1; fi
: "${CKPT:?run dir under \$CKPT_BASE/outputs}"; : "${LABEL:?output label}"
: "${PLAN_CONFIG:?plan config, e.g. plan_gd.yaml or plan_two_level.yaml}"
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
if [ ! -f "${CKPT_BASE}/outputs/${CKPT%@*}/DONE" ]; then
    echo "ERROR: ${CKPT%@*} has no DONE sentinel -- refusing to eval a half-trained checkpoint." >&2
    exit 1
fi
if [ ! -f "${REPO}/conf/${PLAN_CONFIG}" ]; then
    echo "ERROR: conf/${PLAN_CONFIG} does not exist." >&2; exit 1
fi
CONDA_BASE=$(conda info --base); . "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LPWM_CONDA_ENV:-lpwm}"
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${LPWM_WANDB_MODE:-disabled}
echo "=== arm eval job ${SLURM_JOB_ID} on $(hostname) | $(date -Is) ==="
echo "=== label=${LABEL} config=${PLAN_CONFIG} ckpt=${CKPT} seed=${SEED:-0} extra='${EXTRA_OVERRIDES:-}' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
scripts/plan_arm.sh
echo "=== arm eval complete: ${LABEL} | $(date -Is) ==="
