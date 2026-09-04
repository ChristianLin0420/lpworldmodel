#!/bin/bash
# Auto-evaluate wave23 (round-5 training) runs as they finish, then report paired contrasts.
# Mirrors the wave22 autopilot: RUN_NAME is an ENV VAR for plan_slurm.sbatch (passing it
# positionally makes every eval die instantly, which has happened in this campaign), and a
# run counts as evaluated only when logs.json carries the terminal marker.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
ARMS="PiWM-decode PiWM-decode-detach PiWM-decode-w1 PiWM-patchdecode PiWM-patchdecode-detach PiWM-contact PiWM-contact-shuf PiWM-contact-g05 PiWM-jump5 PiWM-overshoot5"
EVALED=""
LAST=""
while true; do
  for a in $ARMS; do
    for d in runs/outputs/${a}_pd*/; do
      [ -f "$d/DONE" ] || continue
      r=$(basename "$d"); s=${r##*_s}
      case " $EVALED " in *" $r "*) continue ;; esac
      # already evaluated on disk? terminal marker only
      if grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null; then
        EVALED="$EVALED $r"; continue
      fi
      squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
      RUN_NAME="$r" SEED="$s" NEVALS=50 MAXITER=10 \
        sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
        && { echo "EVAL SUBMITTED $r"; EVALED="$EVALED $r"; }
    done
  done
  $PY analysis/collect_evals.py --out /tmp/w23.json >/dev/null 2>&1
  CUR=$($PY - <<'EOF' 2>/dev/null
import json, numpy as np
from scipy import stats
A = json.load(open("/tmp/w23.json"))["arms"]
PAIRS = [("decode","PiWM-decode","PiWM-decode-detach"), ("decode-w1","PiWM-decode-w1","PiWM-decode-detach"),
         ("patchdec","PiWM-patchdecode","PiWM-patchdecode-detach"),
         ("contact","PiWM-contact","PiWM-contact-shuf"), ("contact-g05","PiWM-contact-g05","PiWM-contact-shuf"),
         ("jump5","PiWM-jump5","PiWM-overshoot5"), ("jump5-vs-base","PiWM-jump5","LpWM-ltv")]
out = []
for nm, x, y in PAIRS:
    X, Y = A.get(x, {}), A.get(y, {})
    s = sorted(set(X) & set(Y), key=int)
    if not s: continue
    d = np.array([X[k] - Y[k] for k in s])
    if len(s) < 3:
        out.append(f"{nm}: n={len(s)} d={d.mean():+.3f}"); continue
    se = d.std(ddof=1)/np.sqrt(len(d)); t = stats.t.ppf(.975, len(d)-1)
    out.append(f"{nm}: n={len(s)} mean={np.mean([X[k] for k in s]):.3f} "
               f"d={d.mean():+.3f}[{d.mean()-t*se:+.2f},{d.mean()+t*se:+.2f}]")
if out: print("W23 || " + " || ".join(out))
EOF
)
  # report only when a contrast actually moves; the wave is long and most cycles are quiet
  [ -n "$CUR" ] && [ "$CUR" != "$LAST" ] && { echo "$CUR"; LAST="$CUR"; }
  sleep 900
done
