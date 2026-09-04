#!/bin/bash
# ROUND 6, R5: the ENSEMBLE M-CURVE. Plan-time only -- nothing is trained here.
#
# WHAT THIS IS FOR. The one positive result in five rounds and ~60 arms is a plan-time
# ensemble: PiWM-vote5-borda scores 0.608 against LpWM-ltv's 0.357 (+0.215 for the
# median rule, diary s16). That is not a method result -- it is five models where the
# baseline has one -- and treating it as one would be the same mistake the round-6 audit
# was written to stop. This wave is DIAGNOSIS, not a scale proposal. The SHAPE of
# success(M) is the measurement:
#   * still climbing at M=12  => the members' errors are largely INDEPENDENT, the vote is
#     averaging away seed noise, and round 7's target is what makes one model's error
#     look like another draw of that noise;
#   * saturating by M=3 or M=5 => the errors are CORRELATED, the ensemble has bought all
#     it is ever going to buy, and round 7 must attack the shared failure instead.
# Either answer constrains round 7, which is why the curve is worth eight seeds.
#
# THE MEMBER SETS ARE NESTED, and this is load bearing. Each seed block keeps the member
# block the recorded PiWM-vote5-* runs used (verified in slurm_logs/eval_PiWM-vote5-*.out:
# s3..s7 for blocks 3-7, s8..s12 for blocks 8-12) and GROWS it: M members = the M
# checkpoints from that block's base, walking up and wrapping over the 16 trained
# LpWM-ltv seeds. So the M=2 set is inside the M=3 set is inside the recorded M=5 set is
# inside M=8 is inside M=12, and a rise from M to M+1 is the effect of ADDING a member
# rather than of swapping the panel. Without nesting, a flat curve could just as well be
# two unlucky committees.
#
#   M=1 is already on record and is not re-run: it is LpWM-ltv itself (n=16).
#   M=5 is already on record at exactly these seeds and rule (PiWM-vote5-borda, 8/8
#   COMPLETED) and is skipped by the already-evaluated guard below rather than
#   resubmitted -- collect_evals.py is newest-timestamp-wins, so a re-run would REPLACE
#   a finished number with a fresh 50-episode draw.
#
# RULE=borda for every point, so the only thing that varies along the curve is M. borda
# is the variance-reduction-only rule (mean rank, no pessimism term: create_vote_objective_fn
# with lam=0), and it is the rule the existing M=5 point uses.
#
# Usage:
#   DRYRUN=1 scripts/plan_mcurve.sh          # print the sbatch lines, submit nothing
#   scripts/plan_mcurve.sh                   # M = 2 3 5 8 12, seeds 3..10, rule=borda
#   MS="8 12" scripts/plan_mcurve.sh         # a subset
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
if [ -f "${REPO}/.env" ]; then set -a; . "${REPO}/.env"; set +a; fi

SEEDS=${SEEDS:-"3 4 5 6 7 8 9 10"}
MS=${MS:-"2 3 5 8 12"}
RULE=${RULE:-borda}
NEVALS=${NEVALS:-50}; MAXITER=${MAXITER:-10}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
# The trained column pool: LpWM-ltv_pd384_bf16_s0 .. s15, all with a DONE sentinel.
# POOL_N is the wrap modulus, so M can exceed the distance from a block base to s15.
POOL_N=${POOL_N:-16}
SUB="sbatch --parsable"
[ "${DRYRUN:-0}" = "1" ] && SUB="echo sbatch"

members_for() {   # <eval seed block> <M>  ->  comma-separated run dirs, nested in M
    local s=$1 m=$2 b i out=""
    if [ "$s" -le 7 ]; then b=3; else b=8; fi
    for i in $(seq 0 $((m - 1))); do
        out="${out}${out:+,}LpWM-ltv_pd384_bf16_s$(( (b + i) % POOL_N ))"
    done
    echo "${out}"
}

for m in ${MS}; do
    label_arm="PiWM-vote${m}-${RULE}"
    for s in ${SEEDS}; do
        label="${label_arm}_pd384_bf16_s${s}"

        # 1. every column must be trained. vote_slurm.sbatch refuses at run time too, but
        #    failing here costs no allocation and names the missing checkpoint.
        mem="$(members_for "${s}" "${m}")"
        missing=""
        IFS=',' read -ra _M <<< "${mem}"
        for c in "${_M[@]}"; do
            [ -f "${CKPT_BASE}/outputs/${c}/DONE" ] || missing="${missing} ${c}"
        done
        if [ -n "${missing}" ]; then
            echo "  SKIP ${label}: untrained column(s):${missing}"
            continue
        fi

        # 2. already evaluated? The test is a logs.json that actually REACHED
        #    final_eval/success_rate -- not merely a logs.json, because round 5's
        #    vote7/vote9 jobs all timed out and left dirs full of mpc rows and no result.
        #    Using "the dir exists" here would permanently skip exactly the M values this
        #    wave exists to fill in.
        done_already=""
        for f in plan_outputs/*_"${label}"_gH*/logs.json; do
            [ -e "${f}" ] || continue
            if grep -ql "final_eval/success_rate" "${f}"; then done_already=1; break; fi
        done
        if [ -n "${done_already}" ]; then
            echo "  already evaluated, skipping: ${label}"
            continue
        fi

        # 3. never two jobs for one label: they would write two plan_outputs dirs for the
        #    same key and collect_evals.py would keep whichever finished last.
        if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "^eval_${label}$"; then
            echo "  in flight already, skipping: ${label}"
            continue
        fi

        # 4. the walltime. Measured cost of one consensus eval (sacct, this campaign):
        #    M=1 0:40, M=3 1:45, M=5 2:50 -- 40 min + 32.5 min per extra member. The 4h
        #    GPU partitions cap vote_slurm.sbatch at 03:55, which is why every M=7 and
        #    M=9 job in round 5 TIMED OUT and produced nothing. M >= 6 (predicted >= 3:23,
        #    i.e. inside the margin) goes to the long-window script instead.
        if [ "${m}" -ge 6 ]; then sb=scripts/vote_long_slurm.sbatch; else sb=scripts/vote_slurm.sbatch; fi

        echo "  submit ${label}  (M=${m} rule=${RULE} sbatch=${sb##*/})"
        echo "    members=${mem}"
        env MEMBERS="${mem}" LABEL="${label}" RULE="${RULE}" SEED="${s}" \
            NEVALS="${NEVALS}" MAXITER="${MAXITER}" \
            ${SUB} --job-name="eval_${label}" "${sb}" | sed 's/^/    /'
    done
done

echo
echo "Monitor: squeue -u \$USER -o '%.18i %.40j %.8T %.10M %R' | grep vote"
echo "Collect: python analysis/collect_evals.py --out campaign.json"
