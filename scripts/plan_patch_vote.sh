#!/bin/bash
# S3 -- does plan-time consensus over PATCH models beat consensus over CLS models?
#
# This composes the campaign's only two leads, and it costs no training at all.
#
#   The one positive: an M=5 committee beats a COMPUTE-MATCHED single model by
#   +0.232 [+0.130, +0.334] (2026-09-03 §16.9). Its mechanism is measured -- members'
#   errors are independent at K=1 and K=5 and average away at the 1/sqrt(M) rate --
#   and what CAPS it is the quality of the best member, not correlation between members.
#
#   The one representation result: the cls latent cannot decode block orientation at all
#   (15.72 deg against a 14.51 deg constant-prediction bound), width does not fix it
#   (d2048: 14.64), and patch tokens do (PiWM-columns: 9.58) -- 2026-09-04 §7.4.
#
# If the cap really is member quality, then a committee of members that can SEE the task's
# controlled variable should sit above a committee that cannot. That is a prediction with a
# direction, and it is falsifiable: if the patch committee lands at the same ~0.60 as the cls
# committee, then the ceiling is not member quality and §16.9's mechanism story is wrong.
#
# Members come from the SAME two families the cls committees used, so the comparison is
# committee-vs-committee on the same episode blocks: blocks 3-7 take member seeds 3-7 and
# blocks 8-12 take 8-12 (scripts/plan_round5_v123.sh:23-26). PiWM-columns holds seeds 3-15,
# so both families exist.
#
#   DRYRUN=1 scripts/plan_patch_vote.sh      # print, submit nothing
#   scripts/plan_patch_vote.sh               # submit
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "${REPO}"
RULE=${RULE:-median}          # median is the cls committee's best-powered rule (n=10)
ARM=${ARM:-PiWM-columns}
TAG=${TAG:-patchvote5}

members_for() {   # same block->family rule the cls committees used
    local b=$1 base
    if [ "$b" -le 7 ]; then base=3; else base=8; fi
    local m=""
    for i in 0 1 2 3 4; do
        m="${m}${m:+,}${ARM}_pd384_patch_bf16_s$((base+i))"
    done
    echo "$m"
}

for b in 3 4 5 6 7 8 9 10 11 12; do
    label="PiWM-${TAG}-${RULE}_pd384_bf16_s${b}"
    # "an output dir exists" is NOT "it was evaluated". A FAILED eval leaves a dir with no
    # terminal marker, and skipping on the dir alone strands it forever -- the exact bug
    # that had to be fixed in both autopilots (commit 0aeede8). Skip only on the terminal
    # marker, or while a dir is still FRESH enough to be the one running.
    if grep -lq "final_eval/success_rate" plan_outputs/*_${label}_gH5/logs.json 2>/dev/null; then
        echo "  already evaluated, skipping: ${label}"; continue
    fi
    newest=$(ls -dt plan_outputs/*_${label}_gH5 2>/dev/null | head -1)
    if [ -n "${newest}" ] && [ $(( $(date +%s) - $(stat -c %Y "${newest}") )) -lt 14400 ] \
       && squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "eval_${label}"; then
        echo "  in flight already: ${label}"; continue
    fi
    mem=$(members_for "$b")
    # refuse a member that has not finished training -- a half-trained column silently
    # produces a success rate for a model that is not the one the contrast is about
    ok=1
    for m in ${mem//,/ }; do
        [ -f "runs/outputs/${m}/DONE" ] || { echo "  SKIP ${label}: member ${m} has no DONE"; ok=0; break; }
    done
    [ "$ok" = 1 ] || continue
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "  [dry] ${label}  rule=${RULE}  members=${mem}"; continue
    fi
    MEMBERS="${mem}" LABEL="${label}" RULE="${RULE}" SEED="${b}" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${label}" scripts/vote_slurm.sbatch | sed 's/^/    /'
done
