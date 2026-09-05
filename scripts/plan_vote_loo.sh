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
# M is data-determined, but it must also be MATCHED across the arms being compared, or the
# committee contrast confounds the feature with the committee size -- the same class of error
# as comparing a 98304-dim patch latent against a 384-dim cls one. M_CAP is therefore set to
# min(n_seeds) over the arms in the comparison, NOT chosen: with LpWM-ltv at 16 seeds and
# PiWM-columns at 12, both run at M = 11.
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
    MEMBERS="${mem}" LABEL="${label}" RULE=median SEED="${b}" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${label}" scripts/vote_slurm.sbatch | sed 's/^/    /'
done
