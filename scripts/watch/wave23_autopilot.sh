#!/bin/bash
# Auto-evaluate wave23 AND wave24 (round-5 training) runs as they finish, then report paired
# contrasts. Mirrors the wave22 autopilot: RUN_NAME is an ENV VAR for plan_slurm.sbatch (passing
# it positionally makes every eval die instantly, which has happened in this campaign), and a run
# counts as evaluated only when logs.json carries the terminal marker.
#
# TWO BUGS THIS FILE HAS ALREADY HAD -- both were SILENT, which is what makes them worth naming:
#
#   1. ARMS listed wave23 only, while the header claimed wave23/24. The three wave24 arms
#      (PiWM-vp{,-mc,-geom}) finished training and sat unevaluated for hours; 23 checkpoints,
#      no eval, no error. An arm missing from ARMS produces no output of any kind, so the
#      monitor looked healthy the entire time.
#   2. PAIRS referred to "PiWM-patchdecode", but collect_evals keys that arm
#      "PiWM-patchdecode_patch" -- the feature tag is part of the key. The lookup missed, the
#      `if not s: continue` swallowed it, and the contrast reported n=0 while the data existed.
#
# Both are the same failure: a name that does not resolve reads as "no data yet" and is
# indistinguishable from an arm that is legitimately still training. Hence _resolve() below,
# which prefix-matches against the keys actually present and SHOUTS when a pair cannot be
# resolved rather than skipping it.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
ARMS="PiWM-decode PiWM-decode-detach PiWM-decode-w1 PiWM-patchdecode PiWM-patchdecode-detach \
PiWM-contact PiWM-contact-shuf PiWM-contact-g05 PiWM-jump5 PiWM-overshoot5 \
PiWM-vp PiWM-vp-mc PiWM-vp-geom"
EVALED=""
LAST=""
while true; do
  for a in $ARMS; do
    for d in runs/outputs/${a}_pd*/; do
      [ -f "$d/DONE" ] || continue
      r=$(basename "$d"); s=${r##*_s}
      case "$r" in CANARY-*) continue ;; esac
      case " $EVALED " in *" $r "*) continue ;; esac
      # already evaluated on disk? terminal marker only
      if grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null; then
        EVALED="$EVALED $r"; continue
      fi
      squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
      # RACE, observed on PiWM-vp_s10 (two eval dirs, 062624 and 064048). The two checks
      # above are "has it FINISHED?" and "is it QUEUED?" -- there is a window between them
      # where a job has left the queue but has not yet written the terminal marker, and in
      # that window an eval submitted by another process (a manual EVAL=1 run_campaign.sh,
      # say) is invisible here. collect_evals is newest-timestamp-wins, so the duplicate
      # REPLACES the first value with a fresh noisy draw -- not corrupt, both draws are
      # valid 50-episode evals, but it silently resamples an arm and wastes an allocation.
      # An existing output dir means an eval has at least STARTED; that closes the window.
      # An output dir means an eval has at least STARTED, which closes the race where a job
      # has left the queue but not yet written its terminal marker (observed on
      # PiWM-vp_s10, which got two dirs; collect_evals is newest-wins, so a duplicate
      # silently resamples the arm).
      #
      # BUT "dir exists -> never resubmit" is too strong and strands genuine failures: an
      # eval that dies leaves a dir with NO terminal marker and would then never be retried.
      # Measured on this archive: 41 of 742 eval dirs lack the marker and 7 had no live job,
      # i.e. they were permanently stuck. So only skip while the dir is FRESH -- younger
      # than the 03:55 walltime cap, so it could still be the one running. An older
      # markerless dir is a dead eval and is retried.
      if compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1; then
          newest=$(ls -dt plan_outputs/*_${r}_gH5 2>/dev/null | head -1)
          if [ -n "${newest}" ] && [ $(( $(date +%s) - $(stat -c %Y "${newest}") )) -lt 14400 ]; then
              continue
          fi
      fi
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

def _resolve(name):
    """Exact key, else the unique key that is `name` plus a feature tag. Never silently None."""
    if name in A:
        return name
    cand = [k for k in A if k == name or k.startswith(name + "_")]
    if len(cand) == 1:
        return cand[0]
    return None

PAIRS = [("decode","PiWM-decode","PiWM-decode-detach"), ("decode-w1","PiWM-decode-w1","PiWM-decode-detach"),
         ("patchdec","PiWM-patchdecode","PiWM-patchdecode-detach"),
         ("contact","PiWM-contact","PiWM-contact-shuf"), ("contact-g05","PiWM-contact-g05","PiWM-contact-shuf"),
         ("jump5","PiWM-jump5","PiWM-overshoot5"), ("jump5-vs-base","PiWM-jump5","LpWM-ltv"),
         # wave24 / T4. vp is the TD variant, -mc isolates TD from "any learned scalar that
         # correlates with progress", -geom isolates temporal structure. All read against the
         # shared baseline; vp-vs-mc is the contrast the T4 spec actually turns on.
         ("vp-vs-base","PiWM-vp","LpWM-ltv"), ("vpmc-vs-base","PiWM-vp-mc","LpWM-ltv"),
         ("vpgeom-vs-base","PiWM-vp-geom","LpWM-ltv"), ("vp-vs-mc","PiWM-vp","PiWM-vp-mc")]
out = []
for nm, x, y in PAIRS:
    rx, ry = _resolve(x), _resolve(y)
    if rx is None or ry is None:
        # An unresolvable name is a CONFIG bug, not "no data". Say so; do not skip.
        out.append(f"{nm}: UNRESOLVED({x if rx is None else y})")
        continue
    X, Y = A[rx], A[ry]
    s = sorted(set(X) & set(Y), key=int)
    if not s:
        out.append(f"{nm}: n=0"); continue
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
