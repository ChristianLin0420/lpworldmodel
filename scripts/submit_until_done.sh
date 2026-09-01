#!/bin/bash
# Chain N 4-hour training windows for one cell until it finishes.
#
# Every GPU partition on this cluster caps at 4h, so a run that needs longer must
# span several jobs. Rather than keeping a driver process alive, we pre-submit
# WINDOWS jobs chained with --dependency=afterany. Each window exits immediately
# if the DONE sentinel exists, so the tail of the chain costs nothing once
# training completes, and 'afterany' means a preempted or failed window still
# lets the next one pick up from checkpoints/model_latest.pth.
#
# Usage:
#   RUN_NAME=probe_mlp_var_pd384_s0 PREDICTOR=mlp_var PROJ_DIM=384 MUP=1 \
#   MUP_LR=5e-4 REG_WEIGHT=0.1 MU=0 SEED=0 WINDOWS=4 \
#     scripts/submit_until_done.sh pusht 5 3 2 64 reprelu cls 1 b
#
#   WINDOWS  number of chained 4h windows (default 4)
#   DRYRUN=1 print the sbatch commands without submitting
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; fi

: "${RUN_NAME:?RUN_NAME is required (pins the run dir so resume + wandb reuse work)}"
WINDOWS=${WINDOWS:-4}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
RUN_DIR="${CKPT_BASE}/outputs/${RUN_NAME}"

if [ -f "${RUN_DIR}/DONE" ]; then
    echo "Already complete: ${RUN_DIR}/DONE exists. Nothing submitted."
    exit 0
fi

mkdir -p slurm_logs
echo "run_name = ${RUN_NAME}"
echo "run_dir  = ${RUN_DIR}"
echo "windows  = ${WINDOWS} x 3h55m"
echo "args     = $*"

DEP=""
for w in $(seq 1 "${WINDOWS}"); do
    CMD=(sbatch --job-name="${RUN_NAME}_w${w}")
    [ -n "${DEP}" ] && CMD+=(--dependency="afterany:${DEP}")
    CMD+=(scripts/train_slurm.sbatch "$@")

    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[dry-run] ${CMD[*]}"
        DEP="<jobid_w${w}>"
        continue
    fi

    OUT=$("${CMD[@]}")
    echo "  window ${w}: ${OUT}"
    DEP=$(echo "${OUT}" | awk '{print $NF}')
done

echo
echo "Monitor:  squeue -u \$USER -n ${RUN_NAME}_w1 -o '%.18i %.30j %.8T %.10M %R'"
echo "Progress: tail -f ${RUN_DIR}/train.log  (or slurm_logs/${RUN_NAME}_w*.out)"
echo "Cancel:   scancel --name=${RUN_NAME}_w1  # repeat per window, or scancel -u \$USER"
