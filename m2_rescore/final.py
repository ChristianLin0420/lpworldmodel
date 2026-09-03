import json, re, os, pickle, collections
import numpy as np
from scipy import stats
fr = json.load(open("/tmp/m2_rescore/frames.json"))
BIAS = np.array([0.607, 0.472]); THR, ATHR = 20.0, np.pi/9
tg = {d: np.asarray(pickle.load(open(f"plan_outputs/{d}/plan_targets.pkl","rb"))["state_g"],float)
      for d in {r["dir"] for r in fr}}
for r in fr:
    g = tg[r["dir"]][r["ep"]]
    r["bl"] = float(np.linalg.norm(np.array([r["bx"],r["by"]])-BIAS-g[2:4]))
    r["sb"] = bool(r["bl"]<THR and r["angle_diff"]<ATHR)
    r["free0"] = bool(r["block_pos_diff_t0"]<THR and r["angle_diff_t0"]<ATHR)
byrun = collections.defaultdict(list)
for r in fr: byrun[r["dir"]].append(r)
runs = [d for d,v in byrun.items() if len(v)>=5]
def per_run(key, sub=None):
    out=[]
    for d in runs:
        v = byrun[d]
        if sub: v=[x for x in v if not x["free0"]] if sub=="push" else v
        out.append(np.mean([x[key] for x in v]) if v else np.nan)
    return np.array(out)
old = per_run("latched_success"); new = per_run("sb")
oldp = per_run("latched_success","push"); newp = per_run("sb","push")
mask = ~np.isnan(oldp)
arch = json.load(open("/tmp/archive_rows.json")); sr={r["dir"]:r["sr"] for r in arch}
a50 = np.array([sr.get(d,np.nan) for d in runs])
def sp(x,y,lbl):
    m=~(np.isnan(x)|np.isnan(y)); r,p=stats.spearmanr(x[m],y[m])
    print(f"  {lbl:66s} rho={r:+.4f}  n={m.sum()}")
print("PER-RUN Spearman (all on the same 10 plotted episodes unless stated):")
sp(old,new,"old latched success  vs  terminal block-only")
sp(oldp,newp,"old latched success  vs  block-only, must-push episodes only")
sp(old,oldp,"old latched (all eps) vs old latched (must-push only)")
sp(a50,new,"archive 50-ep success_rate vs terminal block-only (10 ep)")
sp(a50,newp,"archive 50-ep success_rate vs block-only must-push (10 ep)")
print()
free = np.array([r["free0"] for r in fr]); sb=np.array([r["sb"] for r in fr]); su=np.array([r["latched_success"] for r in fr])
print(f"do-nothing floor for success_block (block already in tolerance at t=0) = {free.mean():.4f}")
print(f"pooled: old latched {su.mean():.4f} | block-only {sb.mean():.4f} | block-only on must-push {sb[~free].mean():.4f} | old on must-push {su[~free].mean():.4f}")
