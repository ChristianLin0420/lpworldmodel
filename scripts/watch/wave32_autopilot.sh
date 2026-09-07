#!/bin/bash
# ROUND 9 (wave32 pushT / wave33 wall). Evaluation is chained at SUBMIT time
# (submit_until_done.sh --dependency=afterany on the last window), so this loop is not the
# primary path to an eval -- it is the safety net for the two ways that chain breaks:
#   * the chain exhausts its windows without ever writing DONE, so the eval fires, finds no
#     DONE, and exits 1 (plan_slurm.sbatch refuses half-trained checkpoints on purpose);
#   * a window is preempted in a way that kills the dependency.
#
# Carries every guard wave29/wave30 had to learn:
#   * ALL arms named, each mapped to the gate that DEFINES it -- a chain re-extension that
#     calls the wrong gate silently submits nothing.
#   * The TERMINAL MARKER (final_eval/success_rate) is the only proof of evaluation; 41 of
#     742 archived plan dirs exist WITHOUT it.
#   * squeue AND a checkpoint-mtime freshness check. squeue goes to zero for a run whose
#     windows are all COMPLETING/draining, and a re-extension fired there lands a whole new
#     chain on top of one still finishing -- PiWM-hist8 reached SEVEN queued windows this way.
#   * EXACT arm keys, never resolve_arm's prefix fallback. collect() keys a wall run as
#     "<arm>_wall", and resolve_arm("PiWM-enc-ssm") would match "PiWM-enc-ssm_wall" by
#     prefix and silently report wall numbers as pushT ones before any pushT data lands.
#   * Env-scoped arm lists, so a cancellation stays cancelled.
#
# REPORTS SUCCESS RATE AND NOTHING ELSE. Diagnostics are still recorded by the runs; no
# mechanism analysis is produced here.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
R=/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs

W32_ARMS="${W32_ARMS:-PiWM-enc-ssm PiWM-enc-ssm-frozen PiWM-enc-deep PiWM-enc-deep-frozen PiWM-enc-scan PiWM-enc-scan-frozen PiWM-enc-st-scan PiWM-enc-st-scan-frozen PiWM-state-sinv PiWM-state-sinv-shuf PiWM-state-vel PiWM-state-sum PiWM-state-nce PiWM-state-nce-shuf}"
W33_ARMS="${W33_ARMS:-LpWM-ltv $W32_ARMS}"
SEEDS="${SEEDS:-3 4 5}"
EVALED=""
SR_LAST=""

windows_for() {
  case "$1" in
    *scan*)       echo 16 ;;  # measured 4.29x baseline per step -- must match the launch,
                              # or a re-extension quietly gives the arm a shorter budget
                              # than the one its window count was chosen for
    *state-sinv*) echo 6 ;;   # a SECOND student encoder pass, with activations
    *) echo 4 ;;
  esac
}

# every run dir this wave owns: pushT (no tag) and wall (_wall tag)
run_dirs() {
  for a in $W32_ARMS; do for s in $SEEDS; do echo "${a}_pd384_bf16_s${s}"; done; done
  for a in $W33_ARMS; do for s in $SEEDS; do echo "${a}_wall_pd384_bf16_s${s}"; done; done
}

while true; do
  # ---- 1. evals for anything trained whose chained eval did not land ------------------
  for r in $(run_dirs); do
    d="$R/$r"; [ -d "$d" ] || continue
    grep -qx "epochs=2" "$d/DONE" 2>/dev/null || continue
    case " $EVALED " in *" $r "*) continue ;; esac
    if grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null; then
      EVALED="$EVALED $r"; continue
    fi
    squeue -u "$USER" -h -o "%j" | grep -qx "eval_${r}" && continue
    # a markerless dir younger than 4h is a live eval; older is dead and IS retried
    if compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1; then
      newest=$(ls -dt plan_outputs/*_${r}_gH5 2>/dev/null | head -1)
      [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -lt 14400 ] && continue
    fi
    pcfg=plan_lewm.yaml; case "$r" in *_wall_*) pcfg=plan_wall.yaml ;; esac
    RUN_NAME="$r" SEED="${r##*_s}" NEVALS=50 MAXITER=10 PLAN_CFG="$pcfg" \
      sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
      && { echo "W32 EVAL SUBMITTED $r ($pcfg)"; EVALED="$EVALED $r"; }
  done

  # ---- 2. re-extend chains that exhausted without finishing ---------------------------
  for r in $(run_dirs); do
    d="$R/$r"; [ -d "$d" ] || continue
    grep -qx "epochs=2" "$d/DONE" 2>/dev/null && continue
    [ "$(squeue -u "$USER" -h -o '%j' | grep -c "^${r}_w")" -gt 0 ] && continue
    ck="$d/checkpoints/model_latest.pth"
    if [ -f "$ck" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$ck" 2>/dev/null || echo 0) ))
      [ "$age" -lt 1800 ] && continue      # something is still writing: leave it alone
    fi
    arm="${r%%_pd384_bf16_s*}"; seed="${r##*_s}"; w=$(windows_for "$r")
    case "$arm" in
      *_wall) g=wave33; arm="${arm%_wall}"; v=WAVE33_ARMS ;;
      *)      g=wave32; v=WAVE32_ARMS ;;
    esac
    echo "W32 CHAIN EXHAUSTED, re-extending $r via $g to WINDOWS=$w"
    WINDOWS=$w SEEDS="$seed" SUBMIT_EVAL=1 env "$v=$arm" \
      bash scripts/run_campaign.sh "$g" >/dev/null 2>&1
  done

  # ---- 3. SUCCESS RATE ONLY ----------------------------------------------------------
  # Printed only when it CHANGES. An arm that has finished reports the same contrast every
  # cycle forever, so an unconditional print buries each new result under repeats of the
  # old ones -- and a watcher whose output is mostly noise stops being read.
  SR_NOW=$($PY - <<'EOF' 2>/dev/null
import numpy as np
from analysis.collect_evals import collect
A = collect(scheme="fixed")[0]

# EXACT keys. resolve_arm's prefix fallback would match "<arm>_wall" for a bare pushT name.
PAIRS = [("PiWM-enc-ssm", "PiWM-enc-ssm-frozen"),
         ("PiWM-enc-deep", "PiWM-enc-deep-frozen"),
         ("PiWM-enc-scan", "PiWM-enc-scan-frozen"),
         ("PiWM-enc-st-scan", "PiWM-enc-st-scan-frozen"),
         ("PiWM-state-sinv", "PiWM-state-sinv-shuf"),
         ("PiWM-state-vel", "PiWM-state-sum"),
         ("PiWM-state-nce", "PiWM-state-nce-shuf")]

def d(a, b):
    va, vb = A.get(a), A.get(b)
    if not va or not vb:
        return None
    sh = sorted(set(va) & set(vb), key=int)
    if not sh:
        return None
    return len(sh), float(np.mean([float(va[s]) - float(vb[s]) for s in sh]))

for env, tag, base in (("pusht", "", "LpWM-ltv"), ("wall", "_wall", "LpWM-ltv_wall")):
    out = []
    for arm, ctrl in PAIRS:
        vc = d(arm + tag, ctrl + tag)
        vb = d(arm + tag, base)
        if vc is None and vb is None:
            continue
        p = arm.replace("PiWM-", "")
        s = f"{p}:"
        if vc: s += f" vs_ctrl n={vc[0]} {vc[1]:+.3f}"
        if vb: s += f" vs_base n={vb[0]} {vb[1]:+.3f}"
        out.append(s)
    if out:
        print(f"W32 SR [{env}] || " + " || ".join(out))
EOF
)
  if [ "$SR_NOW" != "$SR_LAST" ]; then
    [ -n "$SR_NOW" ] && echo "$SR_NOW"
    SR_LAST="$SR_NOW"
  fi
  sleep 600
done
