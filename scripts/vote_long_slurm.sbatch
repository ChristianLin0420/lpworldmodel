#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=backfill_singlenode,backfill_block1
#SBATCH --time=09:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# scripts/vote_slurm.sbatch with a LONGER WALL CLOCK. Same body, byte for byte -- this
# file is only a different set of #SBATCH headers plus an exec of that script, so the two
# can never drift and a vote at M=12 is the same instrument as the vote at M=5 on record.
#
# WHY IT EXISTS. A consensus eval costs, measured from sacct over this campaign:
#     M = 1  ->  0:40      M = 3  ->  1:45      M = 5  ->  2:50
# i.e. 40 min + 32.5 min per extra member, linear to within a minute. The GPU partitions
# this cluster gives us (polar4/polar3/polar/grizzly) all cap at MaxTime=04:00:00, so
# vote_slurm.sbatch asks for 03:55:00 -- and that is exactly why every PiWM-vote7-median
# (7/7 seeds) and every PiWM-vote9-median (9/9 seeds) job in round 5 ended in
# State=TIMEOUT with ZERO final_eval rows written. Seven GPU-days produced no numbers,
# and the M-curve above M=5 is still missing because of it. The fit says M=8 needs 4:27
# and M=12 needs 6:37; neither can ever finish in a 4h partition.
#
# backfill_singlenode is the only partition our account can reach that allows more
# (MaxTime=7 days, AllowAccounts=ALL, 1223 nodes) and it is drawn from the SAME physical
# nodes as polar4 -- batch-block5-04092 is in both, A100-SXM4-80GB either way -- so the
# hardware is not a new factor in the curve. backfill_block1 (16h, 9 nodes) is listed
# second as a fallback. Both are PreemptMode=REQUEUE, hence --requeue: a preempted eval
# restarts from the beginning (plan.py has no eval checkpoint) and writes a SECOND
# plan_outputs dir. That is safe, and deliberately so: analysis/collect_evals.py:145-152
# treats a dir whose logs.json never reached final_eval/success_rate as `pending` and
# skips it, and `sorted()` makes the newest COMPLETE eval win -- so a partial restart can
# neither be read as a result nor overwrite one.
#
# Usage: identical to vote_slurm.sbatch, and normally via scripts/plan_mcurve.sh.
#   MEMBERS="runA,...,runL" LABEL=PiWM-vote12-borda_pd384_bf16_s3 RULE=borda SEED=3 \
#     sbatch --job-name=eval_PiWM-vote12-borda_pd384_bf16_s3 scripts/vote_long_slurm.sbatch
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO=${SLURM_SUBMIT_DIR}; else REPO=$(cd "$(dirname "$0")/.." && pwd); fi
echo "=== long-window vote job ${SLURM_JOB_ID:-local} on $(hostname) | restart #${SLURM_RESTART_COUNT:-0} | $(date -Is) ==="
# The #SBATCH lines in the target are comments to bash, so this runs its body and
# nothing else. exec, so signals reach it directly.
exec bash "${REPO}/scripts/vote_slurm.sbatch"
