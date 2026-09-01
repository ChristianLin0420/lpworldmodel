#!/bin/bash
# Submit one gate's arms x seeds, each as its own chained single-GPU run.
#
# Every arm carries its matched upstream control at the same predictor, D, HP and
# seed, so each contrast is paired. Steps 3 and 4 share a predictor (ltv) and
# therefore share their control arm (LpWM-ltv). Submitting
# both gates reuses those three runs rather than retraining them, because
# submit_until_done.sh exits early when the DONE sentinel already exists.
#
# Usage:
#   scripts/run_campaign.sh sparse            # 3 arms x 3 seeds
#   scripts/run_campaign.sh gate            # 4 arms x 3 seeds
#   scripts/run_campaign.sh union            # 3 arms x 3 seeds
#   scripts/run_campaign.sh gate union      # both, sharing the control arm
#   scripts/run_campaign.sh wave2           # 3 arms x 3 seeds (task-sensitivity wave)
#   EVAL=1 scripts/run_campaign.sh sparse     # CEM eval trained arms instead of training
#
#   KWTA_MATCHED=<k>  k for the matched-rho arm; REQUIRED for sparse, since it comes
#                     from the probe's measured l0_frac (rho * D). No default: guessing
#                     it would silently turn the matched control into a second tight arm.
#   SEEDS="0 1 2"     seeds (default "0 1 2"); DRYRUN=1; NEVALS=50; WINDOWS=3
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; fi

SEEDS=${SEEDS:-"0 1 2"}
NEVALS=${NEVALS:-50}
WINDOWS=${WINDOWS:-4}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
# namespaced on purpose: a bare D is commonly already exported in a login shell,
# and inheriting one would silently corrupt every run name below
PROJ_D=${PROJ_D:-384}

# Set once here, for every arm including the controls. bf16 perturbs numerics, so
# an arm that differed in precision would confound it with the intervention; the
# value is chosen from the probe wall-clock (3.9h/epoch in fp32 on A100).
PRECISION=${PRECISION:-bf16}
export PRECISION

# Must stay under train_slurm.sbatch's cpus-per-task (30), leaving cores for the
# main process and the GPU feed. train.sh otherwise defaults to 20, which was
# oversubscribing the old 16-CPU allocation and thrashing the dataloader.
WORKERS=${WORKERS:-24}

# One project for the whole campaign so the arms are comparable in a single wandb
# workspace. train.py sets name=<arm>/s<seed>, group=<arm> and job_type=<gate>, so
# grouping by "group" gives one row per arm with a seed band.
WANDB_PROJECT=${WANDB_PROJECT:-PiWM-pushT}
export WANDB_PROJECT

# arm -> "PREDICTOR REG_WEIGHT MUP_LR <extra env assignments...>"
# HPs are the reproduce_pusht.sh sparse table entries for that predictor at D=384.
declare -A ARMS
declare -A ORDER
# arm -> "LINK TARGET_P". These are POSITIONAL args to train.sh, so they cannot ride
# along in the extra-env field above; the dense LeWM control is the only arm that
# overrides the sparse default (reprelu, p=1).
declare -A ARM_LINK

sparse_arms() {
    : "${KWTA_MATCHED:?set KWTA_MATCHED=<k> from the measured rho of the probe (k = round(rho*D))}"
    ORDER[sparse]="LpWM-base PiWM-sparse-matched PiWM-sparse-2pct"
    ARMS[LpWM-base]="mlp_var 0.1 5e-4"
    ARMS[PiWM-sparse-matched]="mlp_var 0.1 5e-4 KWTA_K=${KWTA_MATCHED}"
    ARMS[PiWM-sparse-2pct]="mlp_var 0.1 5e-4 KWTA_K=$(python -c "print(round(0.02*${PROJ_D}))")"
}

# ltv gate factorized into input x normalization; (magnitude, sigmoid) is upstream.
gate_arms() {
    ORDER[gate]="LpWM-ltv PiWM-gate-sup-sigmoid PiWM-gate-mag-softmax PiWM-gate-sup-softmax"
    ARMS[LpWM-ltv]="ltv 1.0 5e-4"
    ARMS[PiWM-gate-sup-sigmoid]="ltv 1.0 5e-4 GATE_INPUT=support"
    ARMS[PiWM-gate-mag-softmax]="ltv 1.0 5e-4 GATE_NORM=softmax"
    ARMS[PiWM-gate-sup-softmax]="ltv 1.0 5e-4 GATE_INPUT=support GATE_NORM=softmax"
}

# LpWM-ltv is the J=1 arm: identical config, so it is deliberately not retrained.
union_arms() {
    ORDER[union]="LpWM-ltv PiWM-union4 PiWM-union4-entropy"
    ARMS[LpWM-ltv]="ltv 1.0 5e-4"
    ARMS[PiWM-union4]="ltv 1.0 5e-4 N_HEADS=4 HEAD_ENT=0.0"
    ARMS[PiWM-union4-entropy]="ltv 1.0 5e-4 N_HEADS=4 HEAD_ENT=${HEAD_ENT:-0.1}"
}

# Wave 2: does PushT discriminate a representation change AT ALL? PushT's own state
# has participation ratio ~4.3, so the healthy arms' effective dim of 15-18 is already
# 3.5-4x over-complete. If forcing the code far below that does not move CEM success,
# no gate in this campaign is measuring anything.
wave2_arms() {
    local K; K=$(python -c "print(round(0.02*${PROJ_D}))")
    ORDER[wave2]="PiWM-sparse-2pct LeWM-ltv PiWM-union4-kwta8"
    # k = round(0.02*D): the SDR-canonical sparsity, far below the task dimension.
    ARMS[PiWM-sparse-2pct]="mlp_var 0.1 5e-4 KWTA_K=${K}"
    # The DENSE control (LeWM): identity link + Gaussian target. Reproduces the
    # sparse-vs-dense claim inside OUR pipeline, which has never been checked here.
    ARMS[LeWM-ltv]="ltv 0.1 5e-4"
    ARM_LINK[LeWM-ltv]="identity 2"
    # Union head AT k-WTA sparsity. A 4-way union is 88.7% ON at rho=0.42 (saturated,
    # carries nothing) but 7.8% at 2% -- separating "unions do not help" from "the
    # union ran at an operating point where it could not help".
    ARMS[PiWM-union4-kwta8]="ltv 1.0 5e-4 N_HEADS=4 KWTA_K=${K}"
}

submit_arm() {  # $1 = arm name, $2 = seed
    local arm=$1 seed=$2
    read -r pred rw mlr extra <<< "${ARMS[$arm]}"
    # precision is in the run name so a mixed-precision comparison is visible
    # rather than silent if PRECISION is ever changed mid-campaign
    local run="${arm}_pd${PROJ_D}_${PRECISION}_s${seed}"
    local dir="${CKPT_BASE}/outputs/${run}"

    if [ "${EVAL:-0}" = "1" ]; then
        if [ ! -f "${dir}/DONE" ]; then
            echo "  SKIP eval ${run}: not finished training yet"
            return
        fi
        # sbatch, NOT inline. scripts/plan.sh runs `python plan.py` in the calling
        # shell -- plan.py's submitit path is only reached from train.py's epoch-end
        # hook -- so calling it here ran every CEM eval on the submit host, which has
        # no GPU. plan_slurm.sbatch gives it a real allocation.
        if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "^eval_${run}$"; then
            echo "  eval in flight already, skipping: ${run}"
            return
        fi
        echo "  eval ${run}"
        [ "${DRYRUN:-0}" = "1" ] && { echo "    [dry-run] sbatch --job-name=eval_${run} scripts/plan_slurm.sbatch"; return; }
        RUN_NAME="${run}" SEED="${seed}" NEVALS="${NEVALS}" MAXITER=10 \
            sbatch --job-name="eval_${run}" scripts/plan_slurm.sbatch | sed 's/^/    /'
        return
    fi

    if [ -f "${dir}/DONE" ]; then
        echo "  done already: ${run}"
        return
    fi
    # Never submit a second chain for a run that already has jobs in flight: both
    # would write ${dir}/checkpoints/model_latest.pth and corrupt each other's
    # resume point. This is the case when re-running the launcher to repair one
    # broken arm while the rest of the campaign is still training.
    if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "^${run}_w[0-9]*$"; then
        echo "  in flight already, skipping: ${run}"
        return
    fi
    read -r lnk tp <<< "${ARM_LINK[$arm]:-reprelu 1}"
    echo "  submit ${run}  (pred=${pred} rw=${rw} mup_lr=${mlr} link=${lnk} p=${tp} ${extra})"
    # shellcheck disable=SC2086
    [ "${DRYRUN:-0}" = "1" ] && { DRYRUN=1 env RUN_NAME="${run}" PREDICTOR="${pred}" \
        PROJ_DIM="${PROJ_D}" MUP=1 MUP_LR="${mlr}" REG_WEIGHT="${rw}" MU=0 SEED="${seed}" \
        REGULARIZER=rdmreg WINDOWS="${WINDOWS}" ${extra} \
        scripts/submit_until_done.sh pusht 5 3 "${EPOCHS:-2}" 64 "${lnk}" cls "${tp}" b "${WORKERS}" | sed 's/^/    /'; return; }
    env RUN_NAME="${run}" PREDICTOR="${pred}" PROJ_DIM="${PROJ_D}" MUP=1 MUP_LR="${mlr}" \
        REG_WEIGHT="${rw}" MU=0 SEED="${seed}" REGULARIZER=rdmreg WINDOWS="${WINDOWS}" \
        ${extra} \
        scripts/submit_until_done.sh pusht 5 3 "${EPOCHS:-2}" 64 "${lnk}" cls "${tp}" b "${WORKERS}" | sed 's/^/    /'
}

[ $# -gt 0 ] || { sed -n '2,25p' "$0"; exit 1; }

SUBMITTED=""
for gate in "$@"; do
    case "${gate}" in
        sparse|step2) sparse_arms; gate=sparse ;;
        gate|step3)   gate_arms;   gate=gate   ;;
        union|step4)  union_arms;  gate=union  ;;
        wave2)        wave2_arms;  gate=wave2  ;;
        *) echo "unknown gate '${gate}' (expected sparse|gate|union|wave2)" >&2; exit 1 ;;
    esac
    echo "=== ${gate}: $(echo "${ORDER[$gate]}" | wc -w) arms x $(echo "${SEEDS}" | wc -w) seeds ==="
    for arm in ${ORDER[$gate]}; do
        for seed in ${SEEDS}; do
            case " ${SUBMITTED} " in
                *" ${arm}_s${seed} "*) echo "  shared control, already handled: ${arm}_s${seed}"; continue ;;
            esac
            submit_arm "${arm}" "${seed}"
            SUBMITTED="${SUBMITTED} ${arm}_s${seed}"
        done
    done
    echo
done

echo "Monitor: squeue -u \$USER -o '%.18i %.34j %.8T %.10M %R'"
echo "When all DONE:  EVAL=1 scripts/run_campaign.sh $*"
