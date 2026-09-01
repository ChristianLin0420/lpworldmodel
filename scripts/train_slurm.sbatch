#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly
# Only partitions this cluster actually defines. Listing a NON-EXISTENT one
# (batch_block1/2/3 were here) does not fail at submit time -- sbatch accepts
# the job because some name in the list resolves -- and the scheduler then
# parks it on Reason=PartitionConfig forever. A whole campaign sits PENDING
# looking like ordinary queueing. Regenerate with:
#   for p in <candidates>; do scontrol show partition $p >/dev/null 2>&1 && echo $p; done
#SBATCH --time=03:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# One 4h training window for a single LpWM cell. Preemption-safe: train.py saves
# every training.save_every_x_min and on SIGUSR1, then exits 0 without a DONE
# sentinel, so scripts/submit_until_done.sh chains another window.
#
# RUN_NAME is REQUIRED: it pins the hydra run dir, which is what makes
# auto-resume (checkpoints/model_latest.pth) and wandb run-id reuse work.
#
# Usage (via the wrapper, normally):
#   RUN_NAME=probe_mlp_var_pd384_seed0 PREDICTOR=mlp_var PROJ_DIM=384 \
#   MUP=1 MUP_LR=5e-4 REG_WEIGHT=0.1 MU=0 SEED=0 EPOCHS=2 \
#     sbatch --job-name=probe scripts/train_slurm.sbatch pusht 5 3 2 64 reprelu cls 1 b
#
# Single GPU is deliberate: RDMReg's swd() sorts over dim 0 with no all-gather, so
# sharding the batch across ranks would silently shrink the sliced-Wasserstein
# sample count and change the regularizer. Parallelism goes across runs.
#
# cpus-per-task=30 is the fair share of a 248-CPU / 8-GPU node, and it is load
# bearing, not cosmetic. Training is dataloader-bound (PushT frames are decoded on
# the fly), so at 16 CPUs with num_workers=20 a solo job showed 67% GPU util and a
# load average of 41 -- 2.5x oversubscribed. It also caps co-tenancy at 8 jobs per
# node, which matches the GPU count; packing 7 jobs into 16-CPU slots had cut
# per-job throughput from ~4.5 to ~1.2 batch/s.
set -euo pipefail

# NB: $0 is the spooled copy under /var/spool/slurmd inside a batch job, so the
# repo has to come from SLURM_SUBMIT_DIR (falls back to $0 when run outside SLURM).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO=${SLURM_SUBMIT_DIR}
else
    REPO=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "${REPO}"
mkdir -p slurm_logs

# secrets + paths (WANDB_API_KEY, HF_TOKEN, DATASET_DIR, CKPT_BASE, LPWM_CONDA_ENV)
if [ -f "${REPO}/.env" ]; then
    set -a; . "${REPO}/.env"; set +a
else
    echo "ERROR: ${REPO}/.env not found (see plan Phase 0a)" >&2; exit 1
fi
: "${RUN_NAME:?RUN_NAME is required so the run dir is stable across resubmits}"
: "${DATASET_DIR:?set DATASET_DIR in .env}"

# Windows are pre-chained with --dependency=afterany, so later ones must no-op
# once training has finished rather than starting a fresh run.
RUN_DIR="${CKPT_BASE:-${REPO}/runs}/outputs/${RUN_NAME}"
if [ -f "${RUN_DIR}/DONE" ]; then
    echo "DONE sentinel present at ${RUN_DIR}; nothing to do."
    exit 0
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LPWM_CONDA_ENV:-lpwm}"

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}   # headless pygame (PushT)
export WANDB_RESUME=allow
export TOKENIZERS_PARALLELISM=false

# sbatch exports the SUBMITTING shell's environment (--export=ALL by default), so a
# stray WANDB_MODE=offline in whatever terminal ran the submit silently sends the
# whole campaign offline -- it is read as a wandb setting, not as an error, so the
# job looks healthy. Pin it here rather than trusting the caller. Set
# LPWM_WANDB_MODE=offline to opt out deliberately.
export WANDB_MODE=${LPWM_WANDB_MODE:-online}
unset WANDB_DISABLED WANDB_SILENT
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in .env for online logging}"

echo "=== job ${SLURM_JOB_ID} on $(hostname) | restart #${SLURM_RESTART_COUNT:-0} ==="
echo "=== run_name=${RUN_NAME} | $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# SLURM signals the batch shell (B: prefix), not our python child, so forward it.
CHILD=""
forward() { [ -n "${CHILD}" ] && kill -"$1" "${CHILD}" 2>/dev/null || true; }
trap 'echo "[sbatch] forwarding USR1"; forward USR1' USR1
trap 'echo "[sbatch] forwarding TERM"; forward TERM' TERM

bash scripts/train.sh "$@" &
CHILD=$!

# wait returns 128+signum when interrupted by a trap while the child still lives
rc=0
while true; do
    set +e; wait "${CHILD}"; rc=$?; set -e
    kill -0 "${CHILD}" 2>/dev/null || break
done

echo "=== child exited rc=${rc} | $(date -Is) ==="
exit "${rc}"
