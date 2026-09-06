#!/bin/bash
# Reclaim quota. DRY_RUN=1 prints what would go and touches nothing.
#
# The account is warned it will lose GPU access on the stale-data limit, and runs/outputs
# holds 458 GB. Four categories, in increasing order of how much thought each needs.
#
# GUARDS, applied to every deletion:
#   * never touch a run with a job in flight -- a window resumes from model_latest.pth and
#     a half-deleted checkpoints/ dir would kill the chain
#   * never prune checkpoints from a run without DONE -- it is still training
#   * never prune unless model_latest.pth exists AND is non-trivial in size
#   * never touch plan_outputs dirs holding final_eval/success_rate -- those ARE the results
#     collect_evals reads; deleting one silently changes an arm's n
set -uo pipefail
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
R=/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs
DRY=${DRY_RUN:-1}
freed=0

# one squeue call; matching per-run against a live list is what keeps this safe
squeue -u "$USER" -h -o "%j" 2>/dev/null | sed 's/_w[0-9]*$//' | sed 's/^eval_//' | sort -u \
    > /home/chrislin/.claude/jobs/f091ebc4/tmp/inflight_now.txt
inflight() { grep -qx "$1" /home/chrislin/.claude/jobs/f091ebc4/tmp/inflight_now.txt; }

rm_path() {  # $1 = path, $2 = reason
    local sz; sz=$(du -sm "$1" 2>/dev/null | cut -f1); sz=${sz:-0}
    freed=$((freed + sz))
    if [ "$DRY" = "1" ]; then
        printf "  [dry] %6s MB  %-58s %s\n" "$sz" "$(basename "$1")" "$2"
    else
        rm -rf "$1" && printf "  removed %6s MB  %-54s %s\n" "$sz" "$(basename "$1")" "$2"
    fi
}

echo "=== 1. CANARY smoke-test runs (no scientific value once the arm launches) ==="
for d in "$R"/CANARY-*; do
    [ -d "$d" ] || continue
    r=$(basename "$d")
    inflight "$r" && { echo "  SKIP (in flight): $r"; continue; }
    rm_path "$d" "canary"
done

echo "=== 2. the tok* ladder -- CANCELLED 2026-09-05 as confounded, never any DONE ==="
for d in "$R"/PiWM-tok25_* "$R"/PiWM-tok50_* "$R"/PiWM-tok75_* "$R"/PiWM-tok90_*; do
    [ -d "$d" ] || continue
    r=$(basename "$d")
    [ -f "$d/DONE" ] && { echo "  SKIP (has DONE, unexpected): $r"; continue; }
    inflight "$r" && { echo "  SKIP (in flight): $r"; continue; }
    rm_path "$d" "cancelled ladder"
done

echo "=== 3. lagdil -- the dilation that cannot be expressed; 45 jobs died, no DONE ==="
for d in "$R"/PiWM-lagdil_*; do
    [ -d "$d" ] || continue
    r=$(basename "$d")
    [ -f "$d/DONE" ] && { echo "  SKIP (has DONE): $r"; continue; }
    inflight "$r" && { echo "  SKIP (in flight): $r"; continue; }
    rm_path "$d" "dead arm"
done

echo "=== 4. per-epoch snapshots (model_<N>.pth) on FINISHED runs ==="
echo "    Every eval in this archive loads model_latest.pth (plan_slurm.sbatch passes"
echo "    'latest'), and no archived plan run requested another epoch. What is lost is the"
echo "    ability to re-evaluate a run AT epoch 1 or 2, which nothing has ever done."
pruned=0
for d in "$R"/*/checkpoints; do
    [ -d "$d" ] || continue
    run=$(basename "$(dirname "$d")")
    case "$run" in CANARY-*) continue ;; esac
    [ -f "$(dirname "$d")/DONE" ] || continue          # still training
    inflight "$run" && continue                        # a window may resume from it
    lat="$d/model_latest.pth"
    [ -f "$lat" ] || continue
    [ "$(stat -c %s "$lat" 2>/dev/null || echo 0)" -gt 1000000 ] || continue
    for f in "$d"/model_[0-9]*.pth; do
        [ -f "$f" ] || continue
        sz=$(du -sm "$f" 2>/dev/null | cut -f1); freed=$((freed + ${sz:-0}))
        pruned=$((pruned + 1))
        [ "$DRY" = "1" ] || rm -f "$f"
    done
done
echo "  $( [ "$DRY" = "1" ] && echo '[dry] would prune' || echo 'pruned' ) ${pruned} snapshot files"

echo
echo "=== TOTAL $( [ "$DRY" = "1" ] && echo 'RECLAIMABLE' || echo 'FREED' ): $((freed/1024)) GB ==="
