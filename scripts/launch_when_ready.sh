#!/bin/bash
# Wait for the SLURM controller, then run Phase 0g validation and fire the Phase 1a probe.
#
# Written because the controller was unreachable (cs-oci-ord-a did not resolve) while
# the code was being finished. Rather than watching for it by hand, this blocks until
# sbatch works and then does the two things that have to happen first, in order:
#
#   1. a ~1 minute no-op with the real header, proving a job lands and the allocation
#      has a GPU and can see the dataset (a bad header fails at submit, not at run);
#   2. the probe cell, PushT x mlp_var x D=384 sparse, EPOCHS=2, seed 0, in fp32,
#      chained across 4h windows.
#
# Usage:
#   nohup scripts/launch_when_ready.sh > slurm_logs/launch_when_ready.log 2>&1 &
#
#   WAIT_MIN   give up after this many minutes (default 720)
#   SKIP_NOOP=1  go straight to the probe
#   DRYRUN=1   print what would be submitted
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
mkdir -p slurm_logs
WAIT_MIN=${WAIT_MIN:-720}

log() { echo "[$(date -Is)] $*"; }

wait_for_controller() {
    log "waiting for the SLURM controller (up to ${WAIT_MIN} min)..."
    for _ in $(seq 1 "${WAIT_MIN}"); do
        if scontrol ping >/dev/null 2>&1; then
            log "controller is up: $(scontrol ping 2>&1 | head -1)"
            return 0
        fi
        sleep 60
    done
    log "ERROR: controller still unreachable after ${WAIT_MIN} min"
    return 1
}

wait_for_job() {  # $1 = jobid; echoes the final state
    local id=$1
    while squeue -j "${id}" -h -o %T 2>/dev/null | grep -q .; do sleep 20; done
    sacct -j "${id}" -n -o State -X 2>/dev/null | head -1 | tr -d ' '
}

run_noop() {
    log "submitting header-validation no-op"
    local out id state
    out=$(sbatch --job-name=lpwm_noop scripts/noop_slurm.sbatch)
    id=$(echo "${out}" | awk '{print $NF}')
    log "no-op job ${id}; waiting for it to land"
    state=$(wait_for_job "${id}")
    log "no-op finished: state=${state}"
    sed -n '1,40p' "slurm_logs/lpwm_noop_${id}.out" 2>/dev/null || true
    case "${state}" in
        COMPLETED) log "header validated" ;;
        *) log "ERROR: no-op did not complete (${state}); NOT submitting the probe"
           sed -n '1,40p' "slurm_logs/lpwm_noop_${id}.err" 2>/dev/null || true
           return 1 ;;
    esac
}

# fp32 at HEAD-equivalent defaults; the precision decision comes FROM this run's
# wall-clock, so it must not pre-empt that choice by enabling bf16.
run_probe() {
    log "submitting the probe: pusht x mlp_var x D=384 sparse, EPOCHS=2, seed 0"
    env RUN_NAME="${PROBE_RUN_NAME:-probe_pusht_mlpvar_pd384_s0}" \
        PREDICTOR=mlp_var PROJ_DIM=384 MUP=1 MUP_LR=5e-4 REG_WEIGHT=0.1 MU=0 \
        SEED=0 REGULARIZER=rdmreg WINDOWS="${WINDOWS:-3}" \
        scripts/submit_until_done.sh pusht 5 3 2 64 reprelu cls 1 b
}

wait_for_controller || exit 1
if [ "${DRYRUN:-0}" = "1" ]; then
    log "dry run: would submit noop then probe"
    DRYRUN=1 run_probe
    exit 0
fi
[ "${SKIP_NOOP:-0}" = "1" ] || run_noop || exit 1
run_probe
log "done submitting; monitor with: squeue -u \$USER -o '%.18i %.34j %.8T %.10M %R'"
