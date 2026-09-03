#!/bin/bash
# Planner-VARIANT eval of ONE already-trained checkpoint (V2 PiWM-gd, V3 PiWM-2lvl).
#
# Why this is not scripts/plan.sh: plan.py builds the output dir from `model_name`
# (conf/plan_lewm.yaml:12) and analysis/collect_evals.py parses the arm back out of that
# dir name with newest-wins. Evaluating LpWM-ltv_pd384_bf16_s5 under a DIFFERENT planner
# with model_name=LpWM-ltv_pd384_bf16_s5 would therefore silently overwrite the CEM
# baseline number for that seed. `ensemble_members=[<one run>]` is the one hook that
# decouples the label from the checkpoint dir; the M=1 ensemble path is verified inert
# (PiWMvoteM1 scored 0.66 on block s5, exactly LpWM-ltv s5's 0.66).
#
# Usage:
#   CKPT=LpWM-ltv_pd384_bf16_s3 LABEL=PiWM-gd_pd384_bf16_s3 SEED=3 \
#     PLAN_CONFIG=plan_gd.yaml scripts/plan_arm.sh
#
# LABEL must keep the "<arm>_pd<D>_<precision>_s<block>" shape (analysis/figures.py
# _ARM_STRIP) and SEED must equal <block>: the episodes are [SEED*n_evals+1 ...]
# (plan.py:141), so a mismatch un-pairs the contrast against LpWM-ltv.
#
# Env: CKPT LABEL PLAN_CONFIG [SEED NEVALS MAXITER GOAL_H EPOCH CKPT_BASE
#      EXTRA_OVERRIDES]  -- EXTRA_OVERRIDES is word-split hydra overrides, e.g.
#      "planner.sub_planner.action_noise=0" for the PiWM-gd-noise0 control.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO}"
: "${CKPT:?run dir under \$CKPT_BASE/outputs}"
: "${LABEL:?output label, e.g. PiWM-gd_pd384_bf16_s3}"
: "${DATASET_DIR:?set DATASET_DIR to the dataset root}"
PLAN_CONFIG=${PLAN_CONFIG:-plan_gd.yaml}
NEVALS=${NEVALS:-50}; MAXITER=${MAXITER:-10}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}

python plan.py --config-name "${PLAN_CONFIG}" \
    ckpt_base_path="${CKPT_BASE}" model_name="${LABEL}" model_epoch="${EPOCH:-latest}" \
    "ensemble_members=[${CKPT}]" \
    n_evals="${NEVALS}" planner.max_iter="${MAXITER}" \
    ${SEED:+seed=$SEED} ${GOAL_H:+goal_H=$GOAL_H} ${EXTRA_OVERRIDES:-}
