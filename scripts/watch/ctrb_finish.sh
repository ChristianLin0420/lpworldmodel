#!/bin/bash
# Finish round 4: submit evals for the remaining PiWM-ctrb seeds as they finish training,
# and report ONLY when the paired contrast actually changes. Replaces the wave22 autopilot,
# which reprinted an identical summary every cycle once the other nine arms hit n=8.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
LAST=""
while true; do
  for d in runs/outputs/PiWM-ctrb_pd384_bf16_s*/; do
    [ -f "$d/DONE" ] || continue
    r=$(basename "$d"); s=${r##*_s}
    grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null && continue
    squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
    RUN_NAME="$r" SEED="$s" NEVALS=50 MAXITER=10 \
      sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
      && echo "EVAL SUBMITTED $r"
  done
  $PY analysis/collect_evals.py --out /tmp/ctrb.json >/dev/null 2>&1
  CUR=$($PY - <<'EOF' 2>/dev/null
import json, numpy as np
from scipy import stats
A = json.load(open("/tmp/ctrb.json"))["arms"]
x, y = A.get("PiWM-ctrb", {}), A.get("LpWM-linvar", {})
s = sorted(set(x) & set(y), key=int)
if not s: raise SystemExit
d = np.array([x[k] - y[k] for k in s])
if len(s) < 3:
    print(f"P4 ctrb: n={len(s)} mean={np.mean([x[k] for k in s]):.3f} d={d.mean():+.3f}")
else:
    se = d.std(ddof=1)/np.sqrt(len(d)); t = stats.t.ppf(.975, len(d)-1)
    done = "  [COMPLETE]" if len(s) >= 8 else ""
    print(f"P4 ctrb vs linvar: n={len(s)} mean={np.mean([x[k] for k in s]):.3f} "
          f"d={d.mean():+.3f}[{d.mean()-t*se:+.2f},{d.mean()+t*se:+.2f}]{done}")
EOF
)
  [ -n "$CUR" ] && [ "$CUR" != "$LAST" ] && { echo "$CUR"; LAST="$CUR"; }
  sleep 900
done
