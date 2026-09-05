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
# Round 7 v2. The TOKEN_DROP ladder and the S2/S4 extra seeds were cancelled (confounded at
# fixed capacity, and under-powered by 5x respectively -- diary/2026-09-05 §2b, §2c). What
# replaces them is the DIMENSION-MATCHED grid: each row a cls-vs-patch contrast at equal total
# latent, each column a capacity ladder at fixed feature.
ARMS="LpWM-ltv-p1 PiWM-cols-p4 PiWM-cols-p16 LpWM-ltv-d1536 LpWM-ltv-d6144 \
LpWM-ltv-lr9e5 LpWM-ltv-d2048-hilr PiWM-columns PiWM-patchdecode PiWM-patchdecode-detach"
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
# Rows of the aligned grid: cls vs patch at EQUAL total latent dimension. Comparing across
# rows would confound the feature with capacity, which is the fault this grid exists to fix.
P = [("dims  384: patch vs cls","LpWM-ltv-p1","LpWM-ltv"),
     ("dims 1536: patch vs cls","PiWM-cols-p4","LpWM-ltv-d1536"),
     ("dims 6144: patch vs cls","PiWM-cols-p16","LpWM-ltv-d6144"),
     # Columns of the same grid: capacity at fixed feature.
     ("cls  384 -> 1536","LpWM-ltv-d1536","LpWM-ltv"),
     ("cls  384 -> 6144","LpWM-ltv-d6144","LpWM-ltv"),
     ("patch  1 -> 4 tokens","PiWM-cols-p4","LpWM-ltv-p1"),
     ("patch  4 -> 16 tokens","PiWM-cols-p16","PiWM-cols-p4"),
     # S5: the width x LR 2x2 the campaign designed and never ran.
     ("S5 d384 @ low lr","LpWM-ltv-lr9e5","LpWM-ltv"),
     ("S5 d2048 @ base lr","LpWM-ltv-d2048-hilr","LpWM-ltv"),
     # Leave-one-out committees, M matched at 11 across both arms.
     ("LOO committee: patch vs cls","PiWM-loo11-columns","PiWM-loo11-ltv")]
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
