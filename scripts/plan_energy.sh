#!/bin/bash
# T7 - CEM eval whose LEAF SCORE is the trained ranking head (planning/energy.py).
#
# The head is NOT a world model: it replaces `objective_fn(rollout(...))` at the CEM
# leaf, so the proposal, the elite fraction, the budget, the MPC loop and the evaluator
# are all the ones the archive was measured with. `wm.rollout` is never called
# (asserted in tests/test_energy_planner.py).
#
# ONE EVAL (this is what runs inside the GPU job):
#   CKPT=LpWM-ltv_pd384_bf16_s3 LABEL=PiWM-energy_pd384_bf16_s3 SEED=3 \
#   EHEAD=$CKPT_BASE/energy_heads/LpWM-ltv_pd384_bf16_s3_rank.pt scripts/plan_energy.sh
#
# THE WAVE (submits 8 seeds x {rank, distill} through scripts/arm_slurm.sbatch):
#   SUBMIT=1 scripts/plan_energy.sh                 # optionally DEP=<jobid[,jobid]>
#   DRYRUN=1 SUBMIT=1 scripts/plan_energy.sh        # print the sbatch lines only
#
# WHY IT GOES THROUGH scripts/arm_slurm.sbatch + scripts/plan_arm.sh
#   `plan_slurm.sbatch` hard-codes `scripts/plan.sh <config> <run> <epoch>` and has no
#   way to carry a hydra override, and both files belong to other agents this wave.
#   `plan_arm.sh` already forwards `EXTRA_OVERRIDES` verbatim and already relabels the
#   output dir via `ensemble_members=[<one run>]` (the M=1 ensemble path is verified
#   inert), which is exactly what an energy eval of an LpWM-ltv checkpoint needs: the
#   baseline's own number for that seed must not be overwritten.
#
# WHY THE PATH IS PASSED AND NOT DEFAULTED
#   `+planner.sub_planner.energy_ckpt=<abs path>` is REQUIRED. EnergyCEMPlanner raises
#   if it is absent, and load_energy_head prints the parameter checksum it loaded, which
#   must equal the checksum train_energy.py printed at save. `path_int` was silently
#   randomly initialised at plan time and produced a retracted result (diary sec 13.3);
#   here that is impossible by construction.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
[ -f "${REPO}/.env" ] && { set -a; . "${REPO}/.env"; set +a; }
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
HEADDIR=${HEADDIR:-${CKPT_BASE}/energy_heads}
SEEDS=${SEEDS:-"3 4 5 6 7 8 9 10"}
NEVALS=${NEVALS:-50}; MAXITER=${MAXITER:-10}
BASE_ARM=${BASE_ARM:-LpWM-ltv}
MODES=${MODES:-"rank distill"}   # NB the default must be assigned, not inlined into
                                 # `for m in ${X:-"a b"}` -- bash keeps those quotes

if [ "${SUBMIT:-0}" = "1" ]; then
    SUB="sbatch --parsable"
    [ "${DRYRUN:-0}" = "1" ] && SUB="echo sbatch"
    [ -n "${DEP:-}" ] && SUB="${SUB} --dependency=afterok:${DEP//,/:}"
    for s in ${SEEDS}; do
        ckpt="${BASE_ARM}_pd384_bf16_s${s}"
        for mode in ${MODES}; do
            arm=$([ "${mode}" = "rank" ] && echo "PiWM-energy" || echo "PiWM-energy-distill")
            label="${arm}_pd384_bf16_s${s}"
            head="${HEADDIR}/${ckpt}_${mode}.pt"
            # `env` and not a bare assignment prefix: bash stops recognising assignments
            # at the first word that is not literally NAME=... (see plan_round5_v123.sh).
            env CKPT="${ckpt}" LABEL="${label}" PLAN_CONFIG=plan_lewm.yaml \
                SEED="${s}" NEVALS="${NEVALS}" MAXITER="${MAXITER}" \
                EXTRA_OVERRIDES="planner.sub_planner.target=planning.energy.EnergyCEMPlanner +planner.sub_planner.energy_ckpt=${head}" \
                ${SUB} --job-name="eval_${label}" scripts/arm_slurm.sbatch
        done
    done
    exit 0
fi

: "${CKPT:?run dir under \$CKPT_BASE/outputs, e.g. LpWM-ltv_pd384_bf16_s3}"
: "${LABEL:?output label, e.g. PiWM-energy_pd384_bf16_s3}"
: "${EHEAD:?path to a train_energy.py head (.pt)}"
[ -f "${EHEAD}" ] || { echo "ERROR: no energy head at ${EHEAD}" >&2; exit 1; }
EHEAD=$(cd "$(dirname "${EHEAD}")" && pwd)/$(basename "${EHEAD}")   # plan.py chdirs
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
exec env CKPT="${CKPT}" LABEL="${LABEL}" PLAN_CONFIG="${PLAN_CONFIG:-plan_lewm.yaml}" \
    SEED="${SEED:-0}" NEVALS="${NEVALS}" MAXITER="${MAXITER}" CKPT_BASE="${CKPT_BASE}" \
    EXTRA_OVERRIDES="planner.sub_planner.target=planning.energy.EnergyCEMPlanner +planner.sub_planner.energy_ckpt=${EHEAD} ${EXTRA_OVERRIDES:-}" \
    scripts/plan_arm.sh
