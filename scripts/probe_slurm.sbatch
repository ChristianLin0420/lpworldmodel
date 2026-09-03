#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Run one analysis script that needs a GPU over every checkpoint in the campaign.
#
# The submit host has no GPU (`nvidia-smi: command not found`), and a ViT-small forward
# per run on CPU costs ~30 s -- four hours for the archive. This is the same wrapper
# plan_slurm.sbatch uses, minus the RUN_NAME/DONE machinery, because a sweep reads every
# run rather than one and a partially-trained checkpoint is still a legitimate data point
# for a weights/activations diagnostic.
#
# Usage:
#   PROBE="analysis/d_action_probe.py --batch 32 --out assets/d_action_probe.json" \
#     sbatch --job-name=probe_daction scripts/probe_slurm.sbatch
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${REPO}"
mkdir -p slurm_logs assets

if [ -f "${REPO}/.env" ]; then
    set -a; . "${REPO}/.env"; set +a
else
    echo "ERROR: ${REPO}/.env not found" >&2; exit 1
fi
: "${PROBE:?PROBE is required (the analysis script and its arguments)}"
: "${DATASET_DIR:?set DATASET_DIR in .env}"

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
. "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LPWM_CONDA_ENV:-lpwm}"

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

echo "probe: ${PROBE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086
srun python ${PROBE}
