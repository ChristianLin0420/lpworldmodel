#!/bin/bash
# Round 7 EVAL autopilot. Built BEFORE the first run finishes, not after.
#
# Round 6 taught this the expensive way: it had no eval autopilot at all, 33 runs finished
# training, and the progress monitor reported "trained 33, evald 0" in a way that read as
# normal. The round would have stalled indefinitely (commit 79b7b02).
#
# Carries every guard the earlier autopilots had to learn:
#   * CANARY-* skipped.
#   * The TERMINAL MARKER (final_eval/success_rate) is the only proof of evaluation. A
#     DONE sentinel means training finished, not that anything was scored.
#   * squeue AND a freshness check, because between "left the queue" and "wrote its marker"
#     another process's eval is invisible and a duplicate gets submitted -- collect_evals is
#     newest-wins, so a duplicate silently RESAMPLES the arm (observed on PiWM-vp_s10).
#   * ...but NOT "skip whenever a dir exists", which strands every FAILED eval forever. 41 of
#     742 eval dirs in this archive lack the marker and 7 had no live job (commit 0aeede8).
#     An older markerless dir is a dead eval and is retried.
#   * Arm names resolved by exact match else unique `name_<tag>`: these arms are ALL
#     feature=patch, so collect_evals keys them "PiWM-tok25_patch", not "PiWM-tok25". That
#     mismatch has silently reported n=0 with data on disk three separate times
#     (analysis/collect_evals.py:resolve_arm).
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
ARMS="PiWM-tok25 PiWM-tok50 PiWM-tok75 PiWM-tok90 PiWM-columns \
PiWM-patchdecode PiWM-patchdecode-detach LpWM-ltv-lr9e5 LpWM-ltv-d2048-hilr"
EVALED=""
LAST=""
while true; do
  for a in $ARMS; do
    for d in runs/outputs/${a}_pd*/; do
      [ -f "$d/DONE" ] || continue
      r=$(basename "$d"); s=${r##*_s}
      case "$r" in CANARY-*) continue ;; esac
      case " $EVALED " in *" $r "*) continue ;; esac
      if grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null; then
        EVALED="$EVALED $r"; continue
      fi
      squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
      if compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1; then
        newest=$(ls -dt plan_outputs/*_${r}_gH5 2>/dev/null | head -1)
        if [ -n "${newest}" ] && [ $(( $(date +%s) - $(stat -c %Y "${newest}") )) -lt 14400 ]; then
          continue
        fi
      fi
      RUN_NAME="$r" SEED="$s" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
        && { echo "W26 EVAL SUBMITTED $r"; EVALED="$EVALED $r"; }
    done
  done
  $PY analysis/collect_evals.py --out /tmp/w26.json >/dev/null 2>&1
  CUR=$($PY - <<'EOF' 2>/dev/null
import json, numpy as np
from scipy import stats
A = json.load(open("/tmp/w26.json"))["arms"]

def R(n):
    if n in A: return n
    c = [k for k in A if k.startswith(n + "_")]
    return c[0] if len(c) == 1 else None

# S1's ladder is read against ITS OWN family endpoint (PiWM-columns, TOKEN_DROP=0), never
# against the cls baseline -- comparing a patch arm to a cls arm is the confound S1 exists
# to avoid. S2/S4/S5 are read against the control each one's spec names.
P = [("S1 tok25 vs columns","PiWM-tok25","PiWM-columns"),
     ("S1 tok50 vs columns","PiWM-tok50","PiWM-columns"),
     ("S1 tok75 vs columns","PiWM-tok75","PiWM-columns"),
     ("S1 tok90 vs columns","PiWM-tok90","PiWM-columns"),
     ("S2 columns vs base","PiWM-columns","LpWM-ltv"),
     ("S4 patchdec vs detach","PiWM-patchdecode","PiWM-patchdecode-detach"),
     ("S5 d384-lowlr vs base","LpWM-ltv-lr9e5","LpWM-ltv"),
     ("S5 d2048-hilr vs base","LpWM-ltv-d2048-hilr","LpWM-ltv"),
     ("S3 patchvote vs clsvote","PiWM-patchvote5-median","PiWM-vote5-median")]
out, ready = [], 0
for nm, x, y in P:
    rx, ry = R(x), R(y)
    if rx is None or ry is None:
        continue
    X, Y = A[rx], A[ry]
    s = sorted(set(X) & set(Y), key=int)
    if len(s) < 3:
        out.append(f"{nm}: n={len(s)}"); continue
    if len(s) >= 8: ready += 1
    d = np.array([X[k]-Y[k] for k in s]); se = d.std(ddof=1)/np.sqrt(len(d))
    t = stats.t.ppf(.975, len(d)-1)
    out.append(f"{nm}: n={len(s)} d={d.mean():+.3f}[{d.mean()-t*se:+.2f},{d.mean()+t*se:+.2f}]")
if out:
    print(("ROUND7 ALL READY || " if ready == len(P) else "W26 || ") + " || ".join(out))
EOF
)
  [ -n "$CUR" ] && [ "$CUR" != "$LAST" ] && { echo "$CUR"; LAST="$CUR"; }
  sleep 900
done
