"""How much of the reported success required no block motion at all?

Per-episode LATCHED outcome for episodes 0-9 of every archived run is recoverable
from the plotted filenames `output_final_real_{i}_{success|failure}.png`; whether
the episode needed the block to move at all comes from plan_targets.pkl at t=0.
"""
import glob, os, pickle, re, json
import numpy as np

recs = []
for f in sorted(glob.glob("plan_outputs/*/plan_targets.pkl")):
    d = os.path.dirname(f); base = os.path.basename(d)
    t = pickle.load(open(f, "rb"))
    s0 = np.asarray(t["state_0"], float); sg = np.asarray(t["state_g"], float)
    if s0.ndim != 2: continue
    bl = np.linalg.norm(sg[:, 2:4] - s0[:, 2:4], axis=1)
    an = np.abs(sg[:, 4] - s0[:, 4]); an = np.minimum(an, 2*np.pi - an)
    free0 = (bl < 20) & (an < np.pi/9)          # block already at goal at t=0
    for p in glob.glob(os.path.join(d, "output_final_real_*_*.png")):
        m = re.match(r"output_final_real_(\d+)_(success|failure)\.png$", os.path.basename(p))
        if not m: continue
        i = int(m.group(1))
        if i >= len(bl): continue
        recs.append((base, i, m.group(2) == "success", bool(free0[i]), bl[i], an[i]))

if not recs:
    raise SystemExit("no plotted episodes found")
succ = np.array([r[2] for r in recs]); free = np.array([r[3] for r in recs])
runs = sorted({r[0] for r in recs})
print(f"runs={len(runs)}  plotted episodes={len(recs)}")
print(f"latched success rate over plotted episodes: {succ.mean():.4f}")
print()
print(f"P(needs NO block motion at t=0)                    = {free.mean():.4f}")
print(f"P(needs NO block motion | LATCHED SUCCESS)         = {free[succ].mean():.4f}   <-- 'free'/agent-parking share of reported success")
print(f"P(needs block motion   | LATCHED SUCCESS)          = {1-free[succ].mean():.4f}")
print(f"success rate | episode needs NO block motion       = {succ[free].mean():.4f}  (n={free.sum()})")
print(f"success rate | episode needs    block motion       = {succ[~free].mean():.4f}  (n={(~free).sum()})")
odds = (succ[free].mean()/(1-succ[free].mean()))/(succ[~free].mean()/(1-succ[~free].mean()))
print(f"odds ratio (free vs must-push)                     = {odds:.2f}x")
json.dump([[r[0], r[1], bool(r[2]), bool(r[3])] for r in recs], open("/tmp/m2_rescore/free.json","w"))
