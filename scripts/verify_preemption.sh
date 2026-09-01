#!/bin/bash
# Phase 0g on the real cluster: kill a run mid-epoch and prove the resume is lossless.
#
# tests/test_resume.py already proves the bookkeeping (every batch trained exactly
# once, bit-identical parameters after three preemptions) against the real
# Trainer.train/run on CPU. What only the cluster can prove is the rest of the
# chain: that SIGUSR1 reaches python through sbatch and train.sh, that
# --dependency=afterany hands over correctly, and that wandb resumes into one run.
#
# What it does:
#   1. submits a 2-window chain for a short cell,
#   2. waits until a mid-epoch checkpoint exists (batch_idx > 0),
#   3. scancels window 1, which is the deliberate preemption,
#   4. waits for the chain to write DONE,
#   5. asserts the final epoch is exactly training.epochs and reports the wandb run id
#      (one id across both windows == one continuous curve).
#
# Usage:
#   scripts/verify_preemption.sh            # EPOCHS=2, one cell, ~1 window
#   EPOCHS=1 scripts/verify_preemption.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; fi

RUN_NAME=${RUN_NAME:-verify_preempt_$(date +%m%d-%H%M)}
EPOCHS=${EPOCHS:-2}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
RUN_DIR="${CKPT_BASE}/outputs/${RUN_NAME}"
LATEST="${RUN_DIR}/checkpoints/model_latest.pth"
PY=${PY:-python}

log() { echo "[$(date -Is)] $*"; }

ckpt_field() {  # $1 = key in the checkpoint dict
    ${PY} - "$1" "${LATEST}" <<'EOF'
import sys, torch
print(torch.load(sys.argv[2], map_location="cpu").get(sys.argv[1]))
EOF
}

log "submitting a 2-window chain for ${RUN_NAME} (epochs=${EPOCHS})"
env RUN_NAME="${RUN_NAME}" PREDICTOR=mlp_var PROJ_DIM=384 MUP=1 MUP_LR=5e-4 \
    REG_WEIGHT=0.1 MU=0 SEED=0 REGULARIZER=rdmreg WINDOWS=2 \
    OVERRIDES="training.save_every_x_min=2" \
    scripts/submit_until_done.sh pusht 5 3 "${EPOCHS}" 64 reprelu cls 1 b

W1=$(squeue -u "${USER}" -h -n "${RUN_NAME}_w1" -o %i | head -1)
log "window 1 = job ${W1}"

log "waiting for a mid-epoch checkpoint (batch_idx > 0)..."
for _ in $(seq 1 180); do
    if [ -f "${LATEST}" ]; then
        B=$(ckpt_field batch_idx 2>/dev/null || echo 0)
        E=$(ckpt_field epoch 2>/dev/null || echo 0)
        log "  checkpoint at epoch=${E} batch_idx=${B}"
        [ "${B}" != "0" ] && [ "${B}" != "None" ] && break
    fi
    sleep 30
done
[ -f "${LATEST}" ] || { log "ERROR: no checkpoint appeared; check slurm_logs/"; exit 1; }

log "PREEMPTING: scancel ${W1} (window 2 should pick up via afterany)"
scancel "${W1}"

log "waiting for the chain to finish..."
for _ in $(seq 1 480); do
    [ -f "${RUN_DIR}/DONE" ] && break
    sleep 30
done

echo
echo "================ Phase 0g result ================"
if [ ! -f "${RUN_DIR}/DONE" ]; then
    echo "FAIL: no DONE sentinel at ${RUN_DIR}"
    tail -30 "${RUN_DIR}/train.log" 2>/dev/null || true
    exit 1
fi
echo "DONE: $(cat "${RUN_DIR}/DONE")"
FINAL_E=$(ckpt_field epoch)
echo "final checkpoint epoch = ${FINAL_E}  (expected ${EPOCHS})"
echo "wandb run id           = $(grep -m1 wandb_run_id "${RUN_DIR}/hydra.yaml" || echo '?')"
echo "epochs logged (train.log, must be 1..${EPOCHS} with no gaps and no repeats):"
grep -o "Epoch [0-9]* *Training loss" "${RUN_DIR}/train.log" | awk '{print $2}' | tr '\n' ' '
echo
echo "resume points (one per window handover):"
grep -c "Resuming from epoch" "${RUN_DIR}/train.log" || true
[ "${FINAL_E}" = "${EPOCHS}" ] && echo "PASS: exact epoch count" || {
    echo "FAIL: trained to epoch ${FINAL_E}, expected ${EPOCHS}"; exit 1; }
echo "================================================="
