#!/bin/bash
# ROUND 9 REPAIRED (wave34 pushT / wave35 wall), plus the two arms kept from the first
# attempt because they ARE the finding: PiWM-enc-ssm and PiWM-enc-deep with their frozen
# controls.
#
# WHY THIS FILE EXISTS RATHER THAN REUSING wave32_autopilot.sh: that one names the first
# attempt's arms, and while the repaired round was being submitted it re-extended
# PiWM-enc-scan -- an arm that had just been deliberately cancelled and whose run dir held
# checkpoints trained under the OLD a=0 scan. Its own header warns "env-scoped arm lists, so
# a cancellation stays cancelled"; the lists were simply stale. A superseded arm must never
# appear in a live watcher's ARMS.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
R=/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs

NEW_ARMS="${NEW_ARMS:-PiWM-scan2 PiWM-scan2-fixed PiWM-st2 PiWM-st2-fixed PiWM-sinv2 PiWM-sinv2-shuf PiWM-vel2 PiWM-sum2 PiWM-nce2 PiWM-nce2-shuf}"
KEEP_ARMS="${KEEP_ARMS:-PiWM-enc-ssm PiWM-enc-ssm-frozen PiWM-enc-deep PiWM-enc-deep-frozen}"
SEEDS="${SEEDS:-3 4 5}"
EVALED=""
SR_LAST=""

windows_for() {
  case "$1" in
    *scan2*|*st2*) echo 16 ;;   # the scan is 4.29x baseline per step (measured)
    *sinv2*)       echo 6  ;;   # a second student encode, subsampled
    *) echo 6 ;;
  esac
}
gate_for() {   # $1 = run name; the repaired arms live in wave34/35, the kept ones in 32/33
  case "$1" in
    *_wall_*) case " $NEW_ARMS " in *" ${1%%_wall_*} "*) echo wave35;; *) echo wave33;; esac ;;
    *)        case " $NEW_ARMS " in *" ${1%%_pd384*} "*) echo wave34;; *) echo wave32;; esac ;;
  esac
}
run_dirs() {
  for a in $NEW_ARMS $KEEP_ARMS; do for s in $SEEDS; do
    echo "${a}_pd384_bf16_s${s}"; echo "${a}_wall_pd384_bf16_s${s}"
  done; done
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
    if compgen -G "plan_outputs/*_${r}_gH5" >/dev/null 2>&1; then
      newest=$(ls -dt plan_outputs/*_${r}_gH5 2>/dev/null | head -1)
      [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y "$newest") )) -lt 14400 ] && continue
    fi
    pcfg=plan_lewm.yaml; case "$r" in *_wall_*) pcfg=plan_wall.yaml ;; esac
    RUN_NAME="$r" SEED="${r##*_s}" NEVALS=50 MAXITER=10 PLAN_CFG="$pcfg" \
      sbatch --job-name="eval_${r}" scripts/plan_slurm.sbatch >/dev/null 2>&1 \
      && { echo "W34 EVAL SUBMITTED $r ($pcfg)"; EVALED="$EVALED $r"; }
  done

  # ---- 2. re-extend chains that exhausted without finishing ---------------------------
  for r in $(run_dirs); do
    d="$R/$r"; [ -d "$d" ] || continue
    grep -qx "epochs=2" "$d/DONE" 2>/dev/null && continue
    [ "$(squeue -u "$USER" -h -o '%j' | grep -c "^${r}_w")" -gt 0 ] && continue
    ck="$d/checkpoints/model_latest.pth"
    if [ -f "$ck" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$ck" 2>/dev/null || echo 0) ))
      [ "$age" -lt 1800 ] && continue
    fi
    arm="${r%%_pd384_bf16_s*}"; arm="${arm%_wall}"; seed="${r##*_s}"
    w=$(windows_for "$r"); g=$(gate_for "$r")
    case "$g" in
      wave34) v=WAVE34_ARMS ;; wave35) v=WAVE35_ARMS ;;
      wave32) v=WAVE32_ARMS ;; *) v=WAVE33_ARMS ;;
    esac
    echo "W34 CHAIN EXHAUSTED, re-extending $r via $g to WINDOWS=$w"
    WINDOWS=$w SEEDS="$seed" SUBMIT_EVAL=1 NEVALS=50 env "$v=$arm" \
      bash scripts/run_campaign.sh "$g" >/dev/null 2>&1
  done

  # ---- 3. SUCCESS RATE ONLY, and only when it changes ---------------------------------
  SR_NOW=$($PY - <<'EOF' 2>/dev/null
import numpy as np
from analysis.collect_evals import collect
A = collect(scheme="fixed")[0]
PAIRS = [("PiWM-scan2","PiWM-scan2-fixed"),("PiWM-st2","PiWM-st2-fixed"),
         ("PiWM-sinv2","PiWM-sinv2-shuf"),("PiWM-vel2","PiWM-sum2"),
         ("PiWM-nce2","PiWM-nce2-shuf")]
def d(a,b):
    va,vb=A.get(a),A.get(b)
    if not va or not vb: return None
    sh=sorted(set(va)&set(vb),key=int)
    return (len(sh), float(np.mean([float(va[s])-float(vb[s]) for s in sh]))) if sh else None
for env,tag,base in (("pusht","","LpWM-ltv"),("wall","_wall","LpWM-ltv_wall")):
    out=[]
    for arm,ctrl in PAIRS:
        vc=d(arm+tag,ctrl+tag); vb=d(arm+tag,base)
        if vc is None and vb is None: continue
        s=arm.replace("PiWM-","")+":"
        if vc: s+=f" vs_ctrl n={vc[0]} {vc[1]:+.3f}"
        if vb: s+=f" vs_base n={vb[0]} {vb[1]:+.3f}"
        out.append(s)
    if out: print(f"W34 SR [{env}] || " + " || ".join(out))
EOF
)
  if [ "$SR_NOW" != "$SR_LAST" ]; then
    [ -n "$SR_NOW" ] && echo "$SR_NOW"
    SR_LAST="$SR_NOW"
  fi
  sleep 600
done
