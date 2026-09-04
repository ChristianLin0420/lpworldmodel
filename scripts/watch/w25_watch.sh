#!/bin/bash
# Report (a) each round-6 canary as it PASSES or FAILS its own guard, and (b) the moment the
# full 8-seed wave25 grids are submitted. Reports only on change.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
LASTC=""; LASTF=""
while true; do
  CUR=$($PY - <<'EOF' 2>/dev/null
import json, glob, os, re
# Each round-6 family: the key that proves its term is LIVE, and the guard that would damn it.
FAM = {
    "support":   ("train/support_s",     "train/support_z_rms"),
    "consist":   ("train/consist_loss",  "train/consist_rel"),
    "sam":       ("train/sam_sharpness", "train/sam_d_action_over_scale"),
    "incr-eps":  ("train/incr_ess",      "train/incr_span"),
    "jump2": (None, None), "jump3": (None, None), "jump8": (None, None),
    "overshoot2": (None, None), "overshoot3": (None, None), "overshoot8": (None, None),
}
# Longest prefix first, so "jump2" is not shadowed by a shorter family, and so round 5's
# PiWM-jump5 / PiWM-overshoot5 canaries -- which are NOT round-6 arms -- never match at all.
ORDER = sorted(FAM, key=len, reverse=True)
out = []
for d in sorted(glob.glob("runs/outputs/CANARY-PiWM-*/")):
    arm = os.path.basename(d.rstrip("/")).replace("CANARY-", "").split("_pd")[0]
    fam = next((f for f in ORDER if arm.startswith("PiWM-" + f)), None)
    if fam is None or not os.path.exists(d + "DONE"):
        continue
    f = d + "wandb/latest-run/files/wandb-summary.json"
    if not os.path.exists(f):
        continue
    try: s = json.load(open(f))
    except Exception: continue
    rm = s.get("err/rel_mse")
    key, guard = FAM[fam]
    bits = [f"rel_mse={rm:.4f}" if rm is not None else "rel_mse=?"]
    verdict = "PASS"
    if rm is None or rm >= 0.5:
        verdict = "FAIL(death condition)"
    if key:
        v = s.get(key)
        if v is None:
            verdict = "FAIL(term not logged)"
        elif v != v:
            verdict = "FAIL(NaN)"
        else:
            bits.append(f"{key.split('/')[-1]}={v:.4g}")
        g = s.get(guard) if guard else None
        if g is not None: bits.append(f"{guard.split('/')[-1]}={g:.4g}")
    out.append(f"{arm} {verdict} " + " ".join(bits))
if out: print("CANARY || " + " || ".join(sorted(out)))
EOF
)
  [ -n "$CUR" ] && [ "$CUR" != "$LASTC" ] && { echo "$CUR"; LASTC="$CUR"; }
  # full 8-seed grids: non-canary wave25 arms on disk
  N=$(ls -d runs/outputs/PiWM-{support,consist,sam,incr-eps,jump2,jump3,jump8,overshoot2,overshoot3,overshoot8}*/ 2>/dev/null | wc -l)
  J=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "^PiWM-(support|consist|sam|incr-eps|jump[238]|overshoot[238])")
  F="${N}|${J}"
  if [ "$F" != "$LASTF" ] && [ "$N" -gt 0 ]; then
    echo "WAVE25 FULL LAUNCH || $N run dirs || $J training jobs queued"
    LASTF="$F"
  fi
  sleep 420
done
