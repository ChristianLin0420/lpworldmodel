#!/bin/bash
# Two standing reports the user asked for:
#   1. ROUND 6 LAUNCHED  -- the moment any wave25 arm appears in the queue or on disk
#   2. ROUND 5 COMPLETE  -- when every remaining round-5 contrast reaches n=8
# Reports only on change, so a quiet cluster stays quiet.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
R6_ARMS="PiWM-support PiWM-consist PiWM-sam PiWM-eps PiWM-jump PiWM-overshoot"
LAST6=""; LAST5=""; ANNOUNCED=0
while true; do
  # --- 2. round 5 completion ---
  $PY analysis/collect_evals.py --out /tmp/r5w.json >/dev/null 2>&1
  CUR5=$($PY - <<'EOF' 2>/dev/null
import json, numpy as np
from scipy import stats
A = json.load(open("/tmp/r5w.json"))["arms"]
P = [("T2 patchdecode","PiWM-patchdecode","PiWM-patchdecode-detach"),
     ("T6 jump5","PiWM-jump5","PiWM-overshoot5"),
     ("T4 vp","PiWM-vp","LpWM-ltv"),
     ("T4 vp-mc","PiWM-vp-mc","LpWM-ltv"),
     ("T4 vp-geom","PiWM-vp-geom","LpWM-ltv")]
def _resolve(name):
    """Exact key, else the unique `name_<featuretag>`. collect_evals keys the patch-feature
    arms "PiWM-patchdecode_patch", not "PiWM-patchdecode" -- A.get() missed, the pair read
    n=0 forever, and because `done` requires EVERY pair at n>=8, ROUND5 COMPLETE could never
    fire. A name that does not resolve must be loud, never an empty dict."""
    if name in A: return name
    c = [k for k in A if k == name or k.startswith(name + "_")]
    return c[0] if len(c) == 1 else None

out, done = [], True
for nm, x, y in P:
    rx, ry = _resolve(x), _resolve(y)
    if rx is None or ry is None:
        out.append(f"{nm}: UNRESOLVED({x if rx is None else y})"); done = False; continue
    X, Y = A[rx], A[ry]
    s = sorted(set(X) & set(Y), key=int)
    if len(s) < 8: done = False
    if len(s) < 3:
        out.append(f"{nm}: n={len(s)}"); continue
    d = np.array([X[k]-Y[k] for k in s]); se = d.std(ddof=1)/np.sqrt(len(d))
    t = stats.t.ppf(.975, len(d)-1)
    out.append(f"{nm}: n={len(s)} d={d.mean():+.3f}[{d.mean()-t*se:+.2f},{d.mean()+t*se:+.2f}]")
print(("ROUND5 COMPLETE || " if done else "R5 || ") + " || ".join(out))
EOF
)
  [ -n "$CUR5" ] && [ "$CUR5" != "$LAST5" ] && { echo "$CUR5"; LAST5="$CUR5"; }
  sleep 600
done
