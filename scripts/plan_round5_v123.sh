#!/bin/bash
# Round 5 planner wave: V1 (pessimistic consensus), V2 (gradient planner), V3 (two-level).
#
# All three are PLAN-TIME arms over already-trained LpWM-ltv checkpoints; nothing is
# trained. Seeds 3..10 are round 4's eight blocks, so every contrast against the recorded
# LpWM-ltv / PiWM-vote5-median numbers is paired on the SAME episodes (plan.py:141) and,
# for V2/V3, on the SAME checkpoint.
#
#   DRYRUN=1 scripts/plan_round5_v123.sh          # print the sbatch lines, submit nothing
#   WAVE=v1 scripts/plan_round5_v123.sh           # v1 | v2 | v3 | all (default all)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
SEEDS=${SEEDS:-"3 4 5 6 7 8 9 10"}
NEVALS=${NEVALS:-50}; MAXITER=${MAXITER:-10}
WAVE=${WAVE:-all}
SUB="sbatch --parsable"
[ "${DRYRUN:-0}" = "1" ] && SUB="echo sbatch"

# The vote-5 member block, verified from slurm_logs/eval_PiWM-vote5-median_*.out: the
# recorded arm used s3..s7 for blocks 3-7 and s8..s12 for blocks 8-12. Reusing it
# exactly is what makes the new rules comparable to the +0.228 median result.
members_for() {
    if [ "$1" -le 7 ]; then B=3; else B=8; fi
    echo "LpWM-ltv_pd384_bf16_s$((B)),LpWM-ltv_pd384_bf16_s$((B+1)),LpWM-ltv_pd384_bf16_s$((B+2)),LpWM-ltv_pd384_bf16_s$((B+3)),LpWM-ltv_pd384_bf16_s$((B+4))"
}

vote_arm() {   # <arm> <rule> [lam]
    local arm=$1 rule=$2 lam=${3:-}
    for s in ${SEEDS}; do
        local label="${arm}_pd384_bf16_s${s}"
        # `env` and not a bare assignment prefix: bash stops recognising assignments at
        # the first word that is not literally NAME=..., and ${lam:+LAM=...} is not.
        env MEMBERS="$(members_for "$s")" LABEL="${label}" RULE="${rule}" ${lam:+LAM=${lam}} \
            SEED="${s}" NEVALS="${NEVALS}" MAXITER="${MAXITER}" \
            ${SUB} --job-name="eval_${label}" scripts/vote_slurm.sbatch
    done
}

single_arm() {  # <arm> <plan_config> [extra hydra overrides]
    local arm=$1 cfg=$2 extra=${3:-}
    for s in ${SEEDS}; do
        local label="${arm}_pd384_bf16_s${s}"
        env CKPT="LpWM-ltv_pd384_bf16_s${s}" LABEL="${label}" PLAN_CONFIG="${cfg}" \
            SEED="${s}" NEVALS="${NEVALS}" MAXITER="${MAXITER}" EXTRA_OVERRIDES="${extra}" \
            ${SUB} --job-name="eval_${label}" scripts/arm_slurm.sbatch
    done
}

if [ "${WAVE}" = "all" ] || [ "${WAVE}" = "v1" ]; then
    # the pessimism ladder: lam=0 (borda, the never-run variance-reduction-only control)
    # -> cvar 1 -> cvar 2 -> max (minimax). Same rank matrix, same members, same budget.
    vote_arm PiWM-vote5-borda borda
    vote_arm PiWM-vote5-cvar1 cvar 1.0
    vote_arm PiWM-vote5-cvar2 cvar 2.0
    vote_arm PiWM-vote5-max   max
fi
if [ "${WAVE}" = "all" ] || [ "${WAVE}" = "v2" ]; then
    single_arm PiWM-gd        plan_gd.yaml
    single_arm PiWM-gd-noise0 plan_gd.yaml "planner.sub_planner.action_noise=0"
fi
if [ "${WAVE}" = "all" ] || [ "${WAVE}" = "v3" ]; then
    single_arm PiWM-2lvl plan_two_level.yaml
fi
