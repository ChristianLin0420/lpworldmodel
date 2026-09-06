#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# ROUND 8 / S2. Compute one whitening matrix W = (C + eps I)^(-1/2) per checkpoint.
#
# Runs on a GPU node because the login node has none and this encodes ~4k real frames
# through the ViT per checkpoint. It reads runs/probe_cache.pt, which is already on disk --
# no dataset, no rollout, no env.
#
#   ARM=LpWM-ltv SEEDS="3 4 5 6 7 8 9 10" sbatch --job-name=wscore scripts/wscore_slurm.sbatch
#
# Writes assets/wscore/<run>.pt. The eval that consumes it passes
#   PLAN_EXTRA="+objective.wmat_path=assets/wscore/<run>.pt"
# and plan_slurm.sbatch's LABEL gives the result its own arm name, so the whitened score
# never overwrites the unwhitened one it is compared against.
set -euo pipefail

# The repo has to come from SLURM_SUBMIT_DIR, falling back to $0 only OUTSIDE Slurm: under
# sbatch, $0 is the SPOOLED COPY of this script, so dirname "$0"/.. resolves into Slurm's
# spool directory -- which is not writable, and the first run of this file died on exactly
# that with `mkdir: cannot create directory 'assets': Permission denied`.
# train_slurm.sbatch:46 and plan_slurm.sbatch:33 already carry this idiom; this did not.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO=${SLURM_SUBMIT_DIR}
else
    REPO=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "${REPO}"
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; fi

PY=${PY:-/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python}
ARM=${ARM:-LpWM-ltv}
SEEDS=${SEEDS:-"3 4 5 6 7 8 9 10"}

echo "=== wscore: arm=${ARM} seeds=${SEEDS} | $(date -Is) ==="
mkdir -p assets/wscore
PYTHONPATH="${REPO}" "${PY}" analysis/wscore_weights.py \
    --arm "${ARM}" --seeds ${SEEDS} --out assets/wscore
echo "=== wscore complete: $(ls assets/wscore/*.pt 2>/dev/null | wc -l) matrices | $(date -Is) ==="
