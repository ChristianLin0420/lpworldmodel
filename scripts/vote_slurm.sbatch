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
# One consensus (multi-column vote) CEM/MPC eval. Same allocation as
# plan_slurm.sbatch; refuses to run unless EVERY member finished training, because a
# half-trained column silently changes what the vote is over.
#
#   MEMBERS="runA,runB,runC" LABEL=PiWM-vote3-mean_pd384_bf16_s3 RULE=mean SEED=3 \
#     sbatch --job-name=eval_PiWM-vote3-mean_pd384_bf16_s3 scripts/vote_slurm.sbatch
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO=${SLURM_SUBMIT_DIR}; else REPO=$(cd "$(dirname "$0")/.." && pwd); fi
cd "${REPO}"; mkdir -p slurm_logs
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; else echo "ERROR: ${REPO}/.env not found" >&2; exit 1; fi
: "${MEMBERS:?comma-separated run dirs}"; : "${LABEL:?output label}"
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
IFS=',' read -ra _M <<< "${MEMBERS}"
for m in "${_M[@]}"; do
    if [ ! -f "${CKPT_BASE}/outputs/${m%@*}/DONE" ]; then
        echo "ERROR: ${m%@*} has no DONE sentinel -- refusing to vote with a half-trained column." >&2
        exit 1
    fi
done
CONDA_BASE=$(conda info --base); . "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LPWM_CONDA_ENV:-lpwm}"
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${LPWM_WANDB_MODE:-disabled}
echo "=== vote job ${SLURM_JOB_ID} on $(hostname) | $(date -Is) ==="
echo "=== label=${LABEL} rule=${RULE:-mean} seed=${SEED:-0} members=${MEMBERS} ==="
scripts/plan_vote.sh
echo "=== vote eval complete: ${LABEL} | $(date -Is) ==="
