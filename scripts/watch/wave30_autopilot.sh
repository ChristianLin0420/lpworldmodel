#!/bin/bash
# ROUND 8, the LAST group: wave30 (T3 / ST1 / S3) and wave31 (ST4 + the hist8 top-up).
#
# Written before the first run finishes, and it exists because nothing else covers these
# arms: wave29_autopilot's ARMS list names the sixteen arms of ITS wave, so all 80 of these
# checkpoints would have trained and then sat unevaluated with every monitor reporting
# "normal" -- the exact failure round 6 had, where 33 finished runs read as healthy.
#
# Carries every guard wave29 had to learn:
#   * ALL wave30/31 arms in ARMS, and the gate each one belongs to (chain re-extension has
#     to call the RIGHT gate, or run_campaign silently submits nothing).
#   * The TERMINAL MARKER (final_eval/success_rate) is the only proof of evaluation.
#   * squeue AND a freshness check, but NEVER "skip whenever a dir exists" -- that strands
#     failed evals forever (41 of 742 dirs in this archive lack the marker).
#   * resolve_arm for names: patch arms key as "PiWM-st-tube_patch", not "PiWM-st-tube".
#     That mismatch has silently reported n=0 with data on disk three separate times.
#   * WINDOW COUNT FROM COST: the patch + EMA + extra-encoder-forward arms (st-tube) are the
#     most expensive per step in the campaign, so they get 8 windows, not 4.
#   * The PLAN-TIME arms (wscore, st-goal) are deliberately NOT here: they re-plan existing
#     checkpoints via scripts/plan_variants.sh, have no training to watch, and their eval
#     carries a LABEL this loop would not reproduce.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
R=/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs

W30_ARMS="PiWM-tjepa PiWM-tjepa-w1 PiWM-st-tube PiWM-st-tube-w5 PiWM-st-tube-iid PiWM-white-zt PiWM-white-dz PiWM-white-shuf"
W31_ARMS="PiWM-st-metric PiWM-st-metric-w1 PiWM-hist8"
ARMS="$W30_ARMS $W31_ARMS"
EVALED=""

gate_for() {   # chain re-extension must call the gate that DEFINES the arm
  case " $W30_ARMS " in *" $1 "*) echo wave30; return ;; esac
  echo wave31
}

windows_for() {
  case "$1" in
    *st-tube*) echo 8 ;;   # patch tokens + EMA teacher + a SECOND encoder forward per step
    *hist8*)   echo 6 ;;   # 8 lags => more compute per step
    *tjepa*)   echo 6 ;;   # num_pred=5 widens the batch's frame window
    *) echo 6 ;;
  esac
}

while true; do
  # ---- 1. submit evals for anything trained and unevaluated -------------------------
  for a in $ARMS; do
    for d in "$R"/${a}_pd*_s*/; do
      [ -d "$d" ] || continue
      r=$(basename "$d")
      case "$r" in CANARY-*) continue ;; esac
      grep -qx "epochs=2" "$d/DONE" 2>/dev/null || continue
      case " $EVALED " in *" $r "*) continue ;; esac
      if grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null; then
        EVALED="$EVALED $r"; continue
      fi
      squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
      # a markerless dir younger than 4h is a live eval; older is dead and is RETRIED
      if compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1; then
        newest=$(ls -dt plan_outputs/*_${r}_gH5 2>/dev/null | head -1)
        [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -lt 14400 ] && continue
      fi
      s=${r##*_s}
      RUN_NAME="$r" SEED="$s" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
        && { echo "W30 EVAL SUBMITTED $r"; EVALED="$EVALED $r"; }
    done
  done

  # ---- 2. re-extend chains that exhausted without finishing --------------------------
  for a in $ARMS; do
    for d in "$R"/${a}_pd*_s*/; do
      [ -d "$d" ] || continue
      r=$(basename "$d")
      case "$r" in CANARY-*) continue ;; esac
      grep -qx "epochs=2" "$d/DONE" 2>/dev/null && continue
      # any window still queued => the chain is alive; topping it up is what makes duplicates
      [ "$(squeue -u "$USER" -h -o '%j' | grep -c "^${r}_w")" -gt 0 ] && continue
      # SECOND guard, and the squeue one is not enough on its own. squeue goes to zero for a
      # run whose windows are all COMPLETING/draining, so a re-extension fired there lands a
      # whole new chain on top of one that is still finishing -- PiWM-hist8 s6 reached SEVEN
      # queued windows exactly this way. A live job checkpoints every save_every_x_min (20),
      # so a model_latest.pth touched inside 30 minutes means something is still writing.
      ck="$d/checkpoints/model_latest.pth"
      if [ -f "$ck" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$ck" 2>/dev/null || echo 0) ))
        [ "$age" -lt 1800 ] && continue
      fi
      w=$(windows_for "$r"); g=$(gate_for "$a"); seed=${r##*_s}
      echo "W30 CHAIN EXHAUSTED, re-extending $r via $g to WINDOWS=$w"
      if [ "$g" = "wave30" ]; then
        WINDOWS=$w SEEDS="$seed" WAVE30_ARMS="$a" bash scripts/run_campaign.sh wave30 >/dev/null 2>&1
      else
        WINDOWS=$w SEEDS="$seed" WAVE31_ARMS="$a" bash scripts/run_campaign.sh wave31 >/dev/null 2>&1
      fi
    done
  done

  # ---- 3. report contrasts on SHARED seeds, each against its OWN control -------------
  $PY - <<'EOF' 2>/dev/null
import numpy as np
from analysis.collect_evals import collect, resolve_arm, ArmNameError
A = collect(scheme="fixed")[0]
# Each arm against the control its design names -- never a registered mean, and never the
# global baseline where a nearer control exists.
PAIRS = [
    ("PiWM-tjepa",        "PiWM-jump5"),      # same data, same horizon, different estimator
    ("PiWM-tjepa-w1",     "PiWM-jump5"),
    ("PiWM-st-tube",      "PiWM-st-tube-iid"),  # structure alone
    ("PiWM-st-tube-w5",   "PiWM-st-tube-iid"),
    ("PiWM-white-zt",     "PiWM-white-shuf"),   # alignment alone
    ("PiWM-white-dz",     "PiWM-white-shuf"),
    ("PiWM-st-metric",    "LpWM-ltv"),
    ("PiWM-st-metric-w1", "LpWM-ltv"),
    ("PiWM-hist8",        "LpWM-ltv"),
    ("PiWM-wscore",       "LpWM-ltv"),          # plan-time, same checkpoint
    ("PiWM-st-goal",      "LpWM-ltv"),
]
out = []
for k, c in PAIRS:
    try:
        v = A[resolve_arm(A, k)]; cv = A[resolve_arm(A, c)]
    except (ArmNameError, KeyError):
        continue
    sh = sorted(set(v) & set(cv), key=int)
    if len(sh) < 2:
        continue
    d = float(np.mean([float(v[s]) - float(cv[s]) for s in sh]))
    out.append(f"{k.replace('PiWM-','')} vs {c.replace('PiWM-','').replace('LpWM-','')}: "
               f"n={len(sh)} d={d:+.3f}")
if out:
    print("W30 || " + " || ".join(out))
EOF
  sleep 600
done
