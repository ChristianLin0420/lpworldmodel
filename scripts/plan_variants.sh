#!/bin/bash
# ROUND 8 / S2 + ST5 -- the two PLAN-TIME variants.
#
# Neither retrains anything. Both re-plan checkpoints that already exist, which makes their
# contrast SAME-CHECKPOINT paired: the 82% training-seed variance that dominates every
# retrained comparison is fully controlled, so their bar is +0.05 rather than +0.09, and
# their control (the base arm's existing eval) is free.
#
#   S2  PiWM-wscore    rank CEM candidates in the WHITENED basis. W = (C + eps I)^(-1/2)
#                      per checkpoint, measured from that checkpoint's own codes.
#   ST5 PiWM-st-goal   pool the goal over space and score the trajectory over time,
#                      instead of scoring the terminal frame token by token.
#
# THE COLLISION THIS AVOIDS. `model_name` is BOTH the output-dir label and the path the
# checkpoint loads from, so re-planning LpWM-ltv_..._s3 with a different objective would be
# collected as LpWM-ltv -- overwriting, and averaging into, the very arm it is compared
# against. plan_slurm.sbatch's LABEL sets hydra.run.dir so each variant gets its own arm
# name in collect_evals while model_name still points at the real run dir.
#
#   BASE=LpWM-ltv SEEDS="3 4 5 6 7 8 9 10" scripts/plan_variants.sh [s2|st5|all]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"

BASE=${BASE:-LpWM-ltv}
SEEDS=${SEEDS:-"3 4 5 6 7 8 9 10"}
SUFFIX=${SUFFIX:-_pd384_bf16}
WHAT=${1:-all}
NEVALS=${NEVALS:-50}
MAXITER=${MAXITER:-10}

submitted=0

# ---- S2: needs the whitening matrices first, so the evals depend on that one job --------
if [ "${WHAT}" = "s2" ] || [ "${WHAT}" = "all" ]; then
    need=0
    for s in ${SEEDS}; do
        [ -f "assets/wscore/${BASE}${SUFFIX}_s${s}.pt" ] || need=1
    done
    DEP=""
    if [ "${need}" = "1" ]; then
        if squeue -u "${USER}" -h -o "%j" | grep -qx "wscore"; then
            echo "  wscore job already queued; not submitting a second one"
            DEP=$(squeue -u "${USER}" -h -o "%i %j" | awk '$2=="wscore"{print $1;exit}')
        else
            OUT=$(ARM="${BASE}" SEEDS="${SEEDS}" sbatch --job-name=wscore \
                  scripts/wscore_slurm.sbatch)
            DEP=$(echo "${OUT}" | awk '{print $NF}')
            echo "  wscore matrices: job ${DEP}"
        fi
    else
        echo "  wscore matrices already on disk for every seed"
    fi
    for s in ${SEEDS}; do
        run="${BASE}${SUFFIX}_s${s}"
        label="PiWM-wscore${SUFFIX}_s${s}"
        if grep -lq "final_eval/success_rate" plan_outputs/*_${label}_gH5/logs.json 2>/dev/null; then
            echo "  done already: ${label}"; continue
        fi
        squeue -u "${USER}" -h -o "%j" | grep -qx "eval_${label}" && { echo "  in flight: ${label}"; continue; }
        CMD=(sbatch --job-name="eval_${label}")
        # afterOK, not afterany: without the matrix the objective would silently fall back to
        # the unwhitened path and the arm would be a duplicate of its own control.
        [ -n "${DEP}" ] && CMD+=(--dependency="afterok:${DEP}")
        CMD+=(scripts/plan_slurm.sbatch)
        RUN_NAME="${run}" SEED="${s}" NEVALS="${NEVALS}" MAXITER="${MAXITER}" \
            LABEL="${label}" \
            PLAN_EXTRA="+objective.wmat_path=${REPO}/assets/wscore/${run}.pt" \
            "${CMD[@]}" >/dev/null && { echo "  S2  SUBMITTED ${label}"; submitted=$((submitted+1)); }
    done
fi

# ---- ST5: pure objective override, nothing to precompute ---------------------------------
if [ "${WHAT}" = "st5" ] || [ "${WHAT}" = "all" ]; then
    for s in ${SEEDS}; do
        run="${BASE}${SUFFIX}_s${s}"
        label="PiWM-st-goal${SUFFIX}_s${s}"
        if grep -lq "final_eval/success_rate" plan_outputs/*_${label}_gH5/logs.json 2>/dev/null; then
            echo "  done already: ${label}"; continue
        fi
        squeue -u "${USER}" -h -o "%j" | grep -qx "eval_${label}" && { echo "  in flight: ${label}"; continue; }
        RUN_NAME="${run}" SEED="${s}" NEVALS="${NEVALS}" MAXITER="${MAXITER}" \
            LABEL="${label}" PLAN_EXTRA="objective.mode=pooled" \
            sbatch --job-name="eval_${label}" scripts/plan_slurm.sbatch >/dev/null \
            && { echo "  ST5 SUBMITTED ${label}"; submitted=$((submitted+1)); }
    done
fi

echo "=== plan_variants: ${submitted} submitted ==="
