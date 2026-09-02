#!/bin/bash
# Plan-time CONSENSUS eval: M already-trained checkpoints vote on every CEM candidate.
#
# The columns never share a latent space (every run trains its own encoder), so the
# vote happens on the OBJECTIVE, not on the latents: each column rolls out in its own
# space and planning/objectives.create_vote_objective_fn combines the M opinions.
# There is no training-time aggregator here at all, so nothing can admit z==0.
#
# Usage:
#   MEMBERS="LpWM-ltv_pd384_bf16_s4,LpWM-ltv_pd384_bf16_s5,LpWM-ltv_pd384_bf16_s6" \
#   LABEL=PiWM-vote3-mean_pd384_bf16_s3 RULE=mean SEED=3 scripts/plan_vote.sh
#
# LABEL is what analysis/collect_evals.py files the result under: keep the
# "<arm>_pd<D>_<precision>_s<block>" shape so figures.run_arm/run_seed parse it and
# paired_effect pairs it against LpWM-ltv on the SAME episode block. SEED must equal
# the <block> in LABEL -- the episodes are [SEED*n_evals+1 ... ] (plan.py:134).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
: "${MEMBERS:?comma-separated run dirs under \$CKPT_BASE/outputs}"
: "${LABEL:?output label, e.g. PiWM-vote3-mean_pd384_bf16_s3}"
: "${DATASET_DIR:?set DATASET_DIR to the dataset root}"
RULE=${RULE:-mean}; NEVALS=${NEVALS:-50}; MAXITER=${MAXITER:-10}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
N=$(awk -F, '{print NF}' <<< "${MEMBERS}")

python plan.py --config-name plan_lewm.yaml \
    ckpt_base_path="${CKPT_BASE}" model_name="${LABEL}" model_epoch="${EPOCH:-latest}" \
    "ensemble_members=[${MEMBERS}]" \
    objective._target_=planning.objectives.create_vote_objective_fn \
    +objective.n_members="${N}" +objective.rule="${RULE}" \
    n_evals="${NEVALS}" planner.max_iter="${MAXITER}" ${SEED:+seed=$SEED} ${GOAL_H:+goal_H=$GOAL_H}
