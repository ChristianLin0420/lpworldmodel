#!/bin/bash
# ROUND 8 eval autopilot. Written BEFORE the first run finishes, not after.
#
# Round 6 had no eval autopilot at all: 33 runs finished training, the progress monitor
# reported "trained 33, evald 0", and that read as normal. This carries every guard the
# earlier autopilots had to learn the hard way:
#
#   * ALL round-8 arms in ARMS. wave24's list omitted its own arms and left 23 checkpoints
#     idle while the monitor looked healthy.
#   * The TERMINAL MARKER (final_eval/success_rate) is the only proof of evaluation. A DONE
#     sentinel means training finished, not that anything was scored.
#   * squeue AND a freshness check: between "left the queue" and "wrote its marker" another
#     process's eval is invisible, and collect_evals is newest-wins, so a duplicate silently
#     RESAMPLES the arm (observed on PiWM-vp_s10).
#   * ...but NEVER "skip whenever a dir exists", which strands every FAILED eval forever.
#     41 of 742 eval dirs in this archive lack the marker and 7 had no live job.
#   * Arm names resolved by exact match else unique `name_<tag>`: patch arms key as
#     "PiWM-pdpred_patch", not "PiWM-pdpred". That mismatch has silently reported n=0 with
#     data on disk three separate times (analysis/collect_evals.py:resolve_arm).
#   * WINDOW COUNT FROM PREDICTED COST, not a fixed 4. K=8 stranded 8 of 16 runs at 4 windows,
#     and re-extension has to run AGAIN after old chains drain, because the duplicate-chain
#     guard correctly skips runs that still have windows queued (2 of 16 fell in that gap).
#   * Health gates at the CORRECTED threshold effective_dim < 10, not == 0: 48 runs sit at
#     0 < ed < 10 and pass every gate, LpWM-ltv-d1536 s5 among them at 1.04.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
R=/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs

ARMS="PiWM-hist1 PiWM-hist2 PiWM-hist5 PiWM-hist8 PiWM-detach PiWM-pdpred PiWM-pdpred-w003 PiWM-ssm PiWM-mlpvar PiWM-lagmask25 PiWM-lagmask50 PiWM-lagdil"
EVALED=""

# A K=8 run needs 6 windows, not 4. Anything whose per-window cost is high gets more.
windows_for() {
  case "$1" in
    *hist8*|*hist5*) echo 6 ;;      # more lags => more compute per step
    *) echo 4 ;;
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
      # a markerless dir younger than 4h is a live eval; older is a dead one and is RETRIED
      if compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1; then
        newest=$(ls -dt plan_outputs/*_${r}_gH5 2>/dev/null | head -1)
        [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -lt 14400 ] && continue
      fi
      s=${r##*_s}
      RUN_NAME="$r" SEED="$s" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
        && { echo "W29 EVAL SUBMITTED $r"; EVALED="$EVALED $r"; }
    done
  done

  # ---- 2. re-extend chains that exhausted without finishing --------------------------
  for a in $ARMS; do
    for d in "$R"/${a}_pd*_s*/; do
      [ -d "$d" ] || continue
      r=$(basename "$d")
      grep -qx "epochs=2" "$d/DONE" 2>/dev/null && continue
      # no windows left in the queue => the chain drained without producing DONE
      [ "$(squeue -u "$USER" -h -o '%j' | grep -c "^${r}_w")" -gt 0 ] && continue
      w=$(windows_for "$r")
      echo "W29 CHAIN EXHAUSTED, re-extending $r to WINDOWS=$w"
      seed=${r##*_s}
      WINDOWS=$w SEEDS="$seed" WAVE29_ARMS="$a" bash scripts/run_campaign.sh wave29 >/dev/null 2>&1
    done
  done

  # ---- 3. report contrasts on shared seeds (never a registered mean) -----------------
  $PY - <<'EOF' 2>/dev/null
import numpy as np
from analysis.collect_evals import collect, resolve_arm, ArmNameError
A = collect(scheme="fixed")[0]
b = A.get("LpWM-ltv", {})
out = []
for k in ["PiWM-hist1","PiWM-hist2","PiWM-hist5","PiWM-hist8","PiWM-detach"]:
    try: v = A[resolve_arm(A, k)]
    except (ArmNameError, KeyError): continue
    sh = sorted(set(v) & set(b), key=int)
    if len(sh) < 2: continue
    d = float(np.mean([float(v[s]) - float(b[s]) for s in sh]))
    out.append(f"{k.replace('PiWM-','')}: n={len(sh)} d={d:+.3f}")
for k, c in [("PiWM-pdpred","PiWM-patchdecode"), ("PiWM-pdpred-w003","PiWM-patchdecode")]:
    try:
        v = A[resolve_arm(A, k)]; cv = A[resolve_arm(A, c)]
    except (ArmNameError, KeyError): continue
    sh = sorted(set(v) & set(cv), key=int)
    if len(sh) < 2: continue
    d = float(np.mean([float(v[s]) - float(cv[s]) for s in sh]))
    out.append(f"{k.replace('PiWM-','')} vs patchdecode: n={len(sh)} d={d:+.3f}")
if out: print("W29 || " + " || ".join(out))
EOF
  sleep 600
done
