#!/bin/bash
# The two controls that decide whether consensus is a METHOD or an ARTEFACT.
#
# An adversarial audit of the campaign's only positive (plan-time consensus, +0.228 vs the
# single-model baseline) returned two WEAKENS verdicts and one SURVIVES. The statistics are
# sound -- no optional stopping (all 10 vote5-median seeds share one sacct Submit second,
# 1.75 h before any result existed), 7.8 sigma against a binomial null, McNemar p=8e-20 over
# 500 paired episodes, and it survives Westfall-Young over all 65 campaign contrasts. What it
# does NOT survive is re-baselining:
#
#   vs the members' MEAN  +0.228  [+0.176, +0.280]   18/18 cells positive
#   vs the members' BEST  +0.014  [-0.032, +0.060]   NULL, 4 of 8 rules negative
#
# i.e. consensus removes the risk of picking a bad model; it does not beat the good one. And
# it spends 4.3x plan-time wall-clock, against which NO compute-matched control exists --
# 549/549 archived plan runs use the identical CEM budget.
#
# Two gaps block the verdict, and both are cheap:
#
# A. COMPUTE-MATCHED CONTROL (10 runs). An M=5 ensemble does M rollouts per candidate x 300
#    candidates = 1500 rollouts per opt step. One model at num_samples=1500 does exactly the
#    same. If it reaches ~0.60, consensus is a compute story and the method claim dies.
#
# B. THE OFF-DIAGONAL MEMBER MATRIX (40 runs). Every member has exactly ONE solo number, on
#    its own episode block -- checkpoint quality and block difficulty are perfectly
#    confounded, and "members' best" rests entirely on that diagonal. Evaluating each member
#    on its family's other four blocks makes best/mean same-block quantities and permits a
#    leave-block-out model-selection baseline, which is the honest rival to an ensemble.
#
# Families are fixed by the launchers: blocks 3-7 use members {s3..s7}, blocks 8-12 use
# {s8..s12} (scripts/plan_round5_v123.sh:23-26).
#
#   DRYRUN=1 scripts/plan_consensus_controls.sh        # print, submit nothing
#   scripts/plan_consensus_controls.sh                 # submit both
#   ONLY=A scripts/plan_consensus_controls.sh          # just the compute control
#   ONLY=C scripts/plan_consensus_controls.sh          # just the over-optimisation control
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "${REPO}"
ONLY=${ONLY:-ABC}
sub() {  # label ckpt seed extra
    local label=$1 ckpt=$2 seed=$3 extra=${4:-}
    if compgen -G "plan_outputs/*_${label}_gH5" >/dev/null 2>&1; then
        echo "  already evaluated, skipping: ${label}"; return
    fi
    if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "eval_${label}"; then
        echo "  in flight already: ${label}"; return
    fi
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "  [dry] ${label}  ckpt=${ckpt} seed=${seed} extra='${extra}'"; return
    fi
    CKPT="${ckpt}" LABEL="${label}" SEED="${seed}" PLAN_CONFIG=plan_lewm.yaml \
        NEVALS=50 MAXITER=10 EXTRA_OVERRIDES="${extra}" \
        sbatch --job-name="eval_${label}" scripts/arm_slurm.sbatch | sed 's/^/    /'
}

if [[ "${ONLY}" == *A* ]]; then
  echo "=== A. compute-matched control: one model, 1500 CEM samples (= M=5's rollout count) ==="
  for b in 3 4 5 6 7 8 9 10 11 12; do
      sub "PiWM-cem5x_pd384_bf16_s${b}" "LpWM-ltv_pd384_bf16_s${b}" "${b}" \
          "planner.sub_planner.num_samples=1500"
  done
fi

if [[ "${ONLY}" == *C* ]]; then
  # C. THE OTHER SIDE OF THE SAME KNOB -- the over-optimisation test.
  #
  # A tests whether MORE single-model search reaches the ensemble's 0.602. But the same knob,
  # turned down, tests a different and sharper hypothesis. M4 measured that CEM's predicted
  # improvement beats the realised one on 38% of baseline episodes and only 6.7% under ORACLE
  # dynamics -- the signature of a planner finding and exploiting the places its model is
  # wrong. If that is what limits planning, then SHRINKING the search should HELP, because a
  # weaker maximiser cannot find the model's failure modes.
  #
  # Two directions on one axis makes this a real test rather than a fishing trip:
  #   success rises with samples  -> the planner is search-limited; compute explains consensus
  #   success falls with samples  -> the planner is EXPLOITING the model; more search is worse
  #   flat                        -> neither; search is not the binding constraint at all
  # Nothing in six rounds has varied the planner's budget: 549/549 archived plan runs use an
  # identical CEM configuration, so this axis is completely unmeasured in either direction.
  echo "=== C. over-optimisation control: one model, 60 CEM samples (1/5 of default) ==="
  for b in 3 4 5 6 7 8 9 10 11 12; do
      sub "PiWM-cem0p2x_pd384_bf16_s${b}" "LpWM-ltv_pd384_bf16_s${b}" "${b}" \
          "planner.sub_planner.num_samples=60"
  done
fi

if [[ "${ONLY}" == *B* ]]; then
  echo "=== B. off-diagonal member matrix: each member on its family's other blocks ==="
  for fam in "3 4 5 6 7" "8 9 10 11 12"; do
      for m in ${fam}; do
          for b in ${fam}; do
              [ "${m}" = "${b}" ] && continue      # the diagonal already exists as LpWM-ltv
              sub "PiWM-solo${m}_pd384_bf16_s${b}" "LpWM-ltv_pd384_bf16_s${m}" "${b}"
          done
      done
  done
fi
