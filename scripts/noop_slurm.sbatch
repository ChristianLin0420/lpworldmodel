#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --signal=B:USR1@30
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Header-validation no-op (plan Phase 0g). Everything above this line is byte-for-byte
# the header of scripts/train_slurm.sbatch except --time and the signal lead time.
#
# A bad --gres or an over-limit --time fails at SUBMIT, not at run, so a training
# chain can look accepted and then never land. This job proves the header schedules
# and that the allocation really has a GPU, in about a minute instead of four hours.
#
#   sbatch --job-name=noop scripts/noop_slurm.sbatch
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${REPO}"

echo "=== job ${SLURM_JOB_ID} on $(hostname) | $(date -Is) ==="
echo "partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ -f "${REPO}/.env" ]; then
    set -a; . "${REPO}/.env"; set +a
    echo "dataset_dir=${DATASET_DIR:-unset}"
    # the probe cannot start without these, so fail here rather than 4h in
    test -d "${DATASET_DIR}/pusht_noise" && echo "pusht_noise: OK" || {
        echo "pusht_noise MISSING under ${DATASET_DIR}" >&2; exit 1; }
    CONDA_BASE=$(conda info --base)
    # shellcheck disable=SC1091
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${LPWM_CONDA_ENV:-lpwm}"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
else
    echo "ERROR: ${REPO}/.env not found" >&2; exit 1
fi

echo "=== NOOP OK | $(date -Is) ==="
