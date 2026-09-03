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
# T7 - train the frozen-encoder ranking head for ONE already-trained checkpoint.
#
# This trains NO world model: the checkpoint is loaded, frozen
# (`requires_grad_(False)`, asserted) and used only to encode. So `run_campaign.sh` is
# not involved -- it exists to launch train.py under the name/precision contract, and
# there is no name or precision to contract here.
#
# MODE=both trains BOTH arms from the SAME latent cache, the SAME triples and the SAME
# negatives: PiWM-energy (cross-entropy rank) and its control PiWM-energy-distill
# (regression on the rollout's own terminal latent MSE). Sharing the cache is what makes
# the control a control -- the arms differ in the objective and in nothing else.
#
#   RUN=LpWM-ltv_pd384_bf16_s3 STEPS=20000 EPISODES=3000 \
#     sbatch --job-name=energy_s3 scripts/train_energy.sbatch
#
# Env: RUN [MODE STEPS EPISODES BATCH NEG LR SEED OUTDIR EPOCH TAG EXTRA]
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO=${SLURM_SUBMIT_DIR}; else REPO=$(cd "$(dirname "$0")/.." && pwd); fi
cd "${REPO}"; mkdir -p slurm_logs assets/energy
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; else echo "ERROR: ${REPO}/.env not found" >&2; exit 1; fi
: "${RUN:?run dir under \$CKPT_BASE/outputs}"
: "${DATASET_DIR:?set DATASET_DIR in .env}"
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
# Same refusal as plan_slurm.sbatch: a half-trained encoder is a different frozen
# representation, so the head would be fitted to a model no eval will ever load.
if [ ! -f "${CKPT_BASE}/outputs/${RUN}/DONE" ]; then
    echo "ERROR: ${RUN} has no DONE sentinel -- refusing to freeze a half-trained encoder." >&2
    exit 1
fi
# Heads live NEXT TO the checkpoints, not in the repo: 1.4 M params is 5.6 MB per head
# and assets/ is not gitignored.
OUTDIR=${OUTDIR:-${CKPT_BASE}/energy_heads}
mkdir -p "${OUTDIR}"
# TAG names the output files; override it for a canary so a short run cannot
# overwrite the real head or its log.
TAG=${TAG:-${RUN}}

CONDA_BASE=$(conda info --base); . "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LPWM_CONDA_ENV:-lpwm}"
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy} TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${LPWM_WANDB_MODE:-disabled}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

echo "=== energy-head job ${SLURM_JOB_ID:-inline} on $(hostname) | $(date -Is) ==="
echo "=== run=${RUN} mode=${MODE:-both} steps=${STEPS:-20000} episodes=${EPISODES:-3000} out=${OUTDIR}/${TAG} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python train_energy.py \
    --run "${RUN}" --epoch "${EPOCH:-latest}" --ckpt-base "${CKPT_BASE}" \
    --mode "${MODE:-both}" --steps "${STEPS:-20000}" --episodes "${EPISODES:-3000}" \
    --batch "${BATCH:-64}" --neg "${NEG:-63}" --lr "${LR:-3e-4}" --seed "${SEED:-0}" \
    --out "${OUTDIR}/${TAG}" ${EXTRA:-}

# the training log (small JSON) is kept in the repo for the diary; the .pt files are not
cp -f "${OUTDIR}/${TAG}_log.json" "assets/energy/${TAG}_log.json" 2>/dev/null || true
echo "=== energy head complete: ${RUN} | $(date -Is) ==="
