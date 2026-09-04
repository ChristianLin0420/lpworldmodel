#!/bin/bash
# Round 6 (wave25) EVAL autopilot: submit CEM evals for finished training runs, then report the
# paired contrasts as they become readable.
#
# WHY THIS FILE EXISTS. Round 5 had wave23_autopilot.sh; round 6 had nothing. 33 wave25 runs
# finished training and sat with zero evals submitted and zero queued -- the round would have
# stalled indefinitely while the progress line cheerfully reported "trained 33, evald 0". This is
# the third time in one day that work was silently blocked by an arm missing from a list:
#   * wave24's T4 arms were absent from wave23_autopilot.sh's ARMS (23 idle checkpoints).
#   * R6's support arms were absent from w25_watch.sh's progress GROUP (invisible while training).
#   * and now: an entire round with no eval autopilot.
# The lesson each time is the same -- absence produces no output, and no output reads as healthy.
#
# EVERY arm is evaluated, including the ones the guards have already condemned. R4 is collapsed on
# every seed, but its whole purpose is a CLOSURE ("no eps beats the baseline, so the reweighting
# family is closed"), and a closure needs CEM numbers, not a guard reading. Skipping the dead arms
# would leave exactly the question the round was built to settle.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
ARMS="PiWM-support-w0p03 PiWM-support-w0p1 PiWM-support-w0p3 \
PiWM-consist-w0p03 PiWM-consist-w0p1 PiWM-consist-w0p3 PiWM-consist-w0p1-data \
PiWM-sam-r0p01 PiWM-sam-r0p03 PiWM-sam-r0p1 \
PiWM-incr-eps0p001 PiWM-incr-eps0p01 PiWM-incr-eps0p041 PiWM-incr-eps0p041-clip10 \
PiWM-jump2 PiWM-overshoot2 PiWM-jump3 PiWM-overshoot3 PiWM-jump8 PiWM-overshoot8"
EVALED=""
LAST=""
while true; do
  for a in $ARMS; do
    for d in runs/outputs/${a}_pd*/; do
      [ -f "$d/DONE" ] || continue
      r=$(basename "$d"); s=${r##*_s}
      case "$r" in CANARY-*) continue ;; esac
      case " $EVALED " in *" $r "*) continue ;; esac
      # terminal marker = actually evaluated
      if grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null; then
        EVALED="$EVALED $r"; continue
      fi
      squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
      # An output dir means an eval has at least STARTED. Without this there is a window between
      # "left the queue" and "wrote its marker" where another process's eval is invisible, and a
      # duplicate gets submitted -- observed on PiWM-vp_s10. collect_evals is newest-wins, so a
      # duplicate silently resamples the arm.
      compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1 && continue
      RUN_NAME="$r" SEED="$s" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
        && { echo "W25 EVAL SUBMITTED $r"; EVALED="$EVALED $r"; }
    done
  done
  $PY analysis/collect_evals.py --out /tmp/w25evals.json >/dev/null 2>&1
  CUR=$($PY - <<'EOF' 2>/dev/null
import json, numpy as np
from scipy import stats
A = json.load(open("/tmp/w25evals.json"))["arms"]

def R(name):
    if name in A: return name
    c = [k for k in A if k.startswith(name + "_")]
    return c[0] if len(c) == 1 else None

# Each arm against the control ITS OWN SPEC names -- never against a generic baseline.
# R1's control is the matched OVERSHOOT arm at the SAME K; R2's is the same lambda with
# dataset actions; everything else is the untreated LpWM-ltv.
P = [("R6 w.03","PiWM-support-w0p03","LpWM-ltv"), ("R6 w.1","PiWM-support-w0p1","LpWM-ltv"),
     ("R6 w.3","PiWM-support-w0p3","LpWM-ltv"),
     ("R2 w.03","PiWM-consist-w0p03","LpWM-ltv"), ("R2 w.1","PiWM-consist-w0p1","LpWM-ltv"),
     ("R2 w.3","PiWM-consist-w0p3","LpWM-ltv"),
     ("R2 cem-vs-data","PiWM-consist-w0p1","PiWM-consist-w0p1-data"),
     ("R3 r.01","PiWM-sam-r0p01","LpWM-ltv"), ("R3 r.03","PiWM-sam-r0p03","LpWM-ltv"),
     ("R3 r.1","PiWM-sam-r0p1","LpWM-ltv"),
     ("R4 e1e-3","PiWM-incr-eps0p001","LpWM-ltv"), ("R4 e1e-2","PiWM-incr-eps0p01","LpWM-ltv"),
     ("R4 e4.1e-2","PiWM-incr-eps0p041","LpWM-ltv"), ("R4 clip","PiWM-incr-eps0p041-clip10","LpWM-ltv"),
     ("R1 K2","PiWM-jump2","PiWM-overshoot2"), ("R1 K3","PiWM-jump3","PiWM-overshoot3"),
     ("R1 K8","PiWM-jump8","PiWM-overshoot8")]
out, ready = [], 0
for nm, x, y in P:
    rx, ry = R(x), R(y)
    if rx is None or ry is None:
        continue                       # arm has no evals yet; silence is correct here
    X, Y = A[rx], A[ry]
    s = sorted(set(X) & set(Y), key=int)
    if len(s) < 3:
        out.append(f"{nm}: n={len(s)}"); continue
    if len(s) >= 8: ready += 1
    d = np.array([X[k]-Y[k] for k in s]); se = d.std(ddof=1)/np.sqrt(len(d))
    t = stats.t.ppf(.975, len(d)-1)
    out.append(f"{nm}: n={len(s)} d={d.mean():+.3f}[{d.mean()-t*se:+.2f},{d.mean()+t*se:+.2f}]")
if out:
    tag = "ROUND6 ALL READY || " if ready == len(P) else "W25 || "
    print(tag + " || ".join(out))
EOF
)
  [ -n "$CUR" ] && [ "$CUR" != "$LAST" ] && { echo "$CUR"; LAST="$CUR"; }
  sleep 900
done
