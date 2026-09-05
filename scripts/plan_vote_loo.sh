#!/bin/bash
# Consensus with NO member-count hyperparameter: LEAVE-ONE-OUT committees.
#
# Round 5/6 ran vote{2,3,5,8,12} x {borda,median,cvar1,cvar2,max} -- TWO hyperparameters, and
# the audit (diary/2026-09-03 §16.9) showed the M grid was worse than useless: the member set
# was a step function of the episode block, so all ten "seeds" of vote5-median were really TWO
# committees, and the five rules correlate r = 0.87-0.98. Two knobs, one measurement.
#
# This removes both knobs.
#
#   MEMBERS  = every finished checkpoint of the arm EXCEPT the one whose index matches the
#              episode block. So M is n-1, set by the data, not chosen. Nothing to sweep, and
#              the committee changes with the block, which is what the audit found the old
#              design was missing (there, two committees masqueraded as ten samples).
#   RULE     = median of ranks, fixed. It is scale-free (so heterogeneous members are fine),
#              it is defined at every M, and it was the best-powered rule in the sweep. borda
#              and cvar/max are order statistics of the SAME rank matrix and measured within
#              0.05 of it -- there is nothing left to choose between them.
#
# Leaving out the matching index also means no committee is scored on the block its excluded
# member would have been scored on alone, which keeps the committee-vs-member comparison clean.
#
#   ARM=PiWM-columns DRYRUN=1 scripts/plan_vote_loo.sh
#   ARM=PiWM-columns scripts/plan_vote_loo.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "${REPO}"
ARM=${ARM:?arm name, e.g. PiWM-columns or LpWM-ltv}
FEAT=${FEAT:-}                   # "" for cls arms, "patch" for patch arms
TAG=${TAG:-loo}
SUFFIX=""; [ -n "${FEAT}" ] && SUFFIX="_${FEAT}"

# every finished seed of this arm
mapfile -t SEEDS < <(for d in runs/outputs/${ARM}_pd*${SUFFIX}_bf16_s*/; do
    [ -f "$d/DONE" ] || continue
    b=$(basename "$d"); case "$b" in CANARY-*) continue ;; esac
    echo "${b##*_s}"
done | sort -n)
if [ "${#SEEDS[@]}" -lt 3 ]; then
    echo "  need >=3 finished seeds of ${ARM}; found ${#SEEDS[@]}"; exit 1
fi
# M must be MATCHED across the arms compared, or the committee contrast confounds the feature
# with committee size -- the same class of error as comparing a 98304-dim patch latent against a
# 384-dim cls one. Ideally M = n-1 (data-determined); in practice it is capped by MEMORY.
#
# planning/ensemble.py stacks members along the PATCH axis, so a patch member costs
# num_patches slots where a cls member costs 1: at M = 11 that is 11 x 256 = 2816 stacked slots
# against 11, and every patch committee died with CUDA OOM in planning/evaluator.py while every
# cls committee ran. M = 5 (1280 slots) is the largest that fits, so BOTH sides run at M = 5 --
# the cls side is capped down to match rather than allowed to keep its feasible 11, because an
# unmatched M is precisely the confound this script exists to remove.
M=$(( ${#SEEDS[@]} - 1 ))
if [ -n "${M_CAP:-}" ] && [ "${M_CAP}" -lt "$M" ]; then M=${M_CAP}; fi
echo "  ${ARM}: ${#SEEDS[@]} finished seeds -> leave-one-out committees of M=${M}"

for b in "${SEEDS[@]}"; do
    mem=""; k=0
    for s in "${SEEDS[@]}"; do
        [ "$s" = "$b" ] && continue          # leave out the matching index
        [ "$k" -ge "$M" ] && break           # take the first M, deterministically
        mem="${mem}${mem:+,}${ARM}_pd384${SUFFIX}_bf16_s${s}"; k=$((k+1))
    done
    label="PiWM-${TAG}${M}-$(echo "$ARM" | sed 's/^PiWM-//;s/^LpWM-//')_pd384_bf16_s${b}"
    if grep -lq "final_eval/success_rate" plan_outputs/*_${label}_gH5/logs.json 2>/dev/null; then
        echo "  already evaluated: ${label}"; continue
    fi
    if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "eval_${label}"; then
        echo "  in flight: ${label}"; continue
    fi
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "  [dry] ${label}  M=${M}  (first ${M} of all but s${b})"; continue
    fi
    # Pick the walltime from the PREDICTED cost, not from M alone. Measured single-model eval,
    # this campaign (sacct, COMPLETED only): cls 40 min, patch 64 min. A committee costs
    # base + (M-1) x increment, the increment scaling like the base (cls 32.5 min, so patch
    # ~= 32.5 x 64/40 = 52). Checks out: cls M=5 predicts 172, measured 169.
    #
    # vote_slurm.sbatch is capped at 03:55 = 235 min by the 4h GPU partitions. Routing on
    # "M >= 6", as plan_mcurve.sh does, is a CLS rule and silently mis-routes patch: patch M=5
    # predicts 272 min, went to the 3:55 script, and all 10 jobs TIMED OUT at 03:55:23 having
    # produced nothing. Routing on the prediction gets both features right.
    if [ "${FEAT:-cls}" = "patch" ]; then base=64; incr=52; else base=40; incr=33; fi
    pred=$(( base + (M - 1) * incr ))
    if [ "${pred}" -gt 200 ]; then sb=scripts/vote_long_slurm.sbatch; else sb=scripts/vote_slurm.sbatch; fi
    echo "    ${label}  M=${M} feat=${FEAT:-cls} pred=${pred}min -> ${sb##*/}"
    MEMBERS="${mem}" LABEL="${label}" RULE=median SEED="${b}" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${label}" "${sb}" | sed 's/^/    /'
done
