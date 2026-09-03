import json, re, collections
import numpy as np
from scipy import stats

fr = json.load(open("/tmp/m2_rescore/frames.json"))
BIAS = np.array([0.607, 0.472])          # measured on 522 archive t=0 frames
THR, ATHR = 20.0, np.pi / 9

for r in fr:
    r["bl"] = float(np.linalg.norm(np.array([r["bx"], r["by"]]) - BIAS -
                                   (np.array([r["bx"], r["by"]]) - 0)) * 0)  # placeholder
# recompute block_pos_diff with the bias removed
tg = {}
import pickle, os
for d in {r["dir"] for r in fr}:
    t = pickle.load(open(os.path.join("plan_outputs", d, "plan_targets.pkl"), "rb"))
    tg[d] = np.asarray(t["state_g"], float)
for r in fr:
    g = tg[r["dir"]][r["ep"]]
    p = np.array([r["bx"], r["by"]]) - BIAS
    r["bl"] = float(np.linalg.norm(p - g[2:4]))
    r["sb"] = bool(r["bl"] < THR and r["angle_diff"] < ATHR)
    r["free0"] = bool(r["block_pos_diff_t0"] < THR and r["angle_diff_t0"] < ATHR)
    r["ambig"] = bool(abs(r["bl"] - THR) < 3.4)

n = len(fr)
suc = np.array([r["latched_success"] for r in fr])
sb = np.array([r["sb"] for r in fr])
free = np.array([r["free0"] for r in fr])
amb = np.array([r["ambig"] for r in fr])
fit = np.array([r["fit_iou"] for r in fr])
print(f"episodes = {n} over {len({r['dir'] for r in fr})} runs;  fit IoU median {np.median(fit):.3f}, p1 {np.percentile(fit,1):.3f}")
print(f"episodes within the +-3.4px recovery band of the 20px threshold: {amb.mean()*100:.1f}%")
print()
print("--- pooled per-episode, terminal (image-recovered) vs latched (archive) ---")
print(f"latched  success (archive metric, agent+block) = {suc.mean():.4f}")
print(f"terminal success_block (BLOCK ONLY)           = {sb.mean():.4f}")
print(f"consistency check  P(success_block | latched success) = {sb[suc].mean():.4f}   (must be ~1: success => success_block)")
print(f"P(latched success | success_block)                    = {suc[sb].mean():.4f}")
print()
print(f"share of latched successes that needed NO block motion at t=0: {free[suc].mean():.4f}")
print(f"share of terminal block successes that needed NO block motion : {free[sb].mean():.4f}")
print(f"success_block rate | needs block motion  = {sb[~free].mean():.4f}  (n={(~free).sum()})")
print(f"success_block rate | needs no block motion = {sb[free].mean():.4f} (n={free.sum()})")
print()
# ---- per-run ranking ----
byrun = collections.defaultdict(list)
for r in fr: byrun[r["dir"]].append(r)
runs, old, new, newmove, oldmove = [], [], [], [], []
for d, v in sorted(byrun.items()):
    if len(v) < 5: continue
    s = np.array([x["latched_success"] for x in v]); b = np.array([x["sb"] for x in v])
    f = np.array([x["free0"] for x in v])
    runs.append(d); old.append(s.mean()); new.append(b.mean())
    if (~f).sum() >= 3:
        oldmove.append(s[~f].mean()); newmove.append(b[~f].mean())
old, new = np.array(old), np.array(new)
print(f"--- per-run, over the {len(runs)} runs x 10 plotted episodes ---")
print(f"mean over runs: old latched {old.mean():.4f} -> new terminal block-only {new.mean():.4f}")
rho, p = stats.spearmanr(old, new)
print(f"Spearman(old latched success, new terminal success_block) = {rho:.4f}  (p={p:.2e}, n={len(runs)})")
r2, p2 = stats.pearsonr(old, new); print(f"Pearson  = {r2:.4f}")
om, nm = np.array(oldmove), np.array(newmove)
rho2, p3 = stats.spearmanr(om, nm)
print(f"Spearman(old latched, new block-only restricted to must-push episodes) = {rho2:.4f} (n={len(om)})")

# ---- per-run using the FULL 50-episode archive success rate as 'old' ----
rows = json.load(open("/tmp/archive_rows.json"))
sr = {r["dir"]: r["sr"] for r in rows}
o50 = np.array([sr[d] for d in runs if d in sr])
n10 = np.array([new[i] for i, d in enumerate(runs) if d in sr])
rho3, p4 = stats.spearmanr(o50, n10)
print(f"Spearman(archive final_eval/success_rate over 50 eps, new block-only over 10 eps) = {rho3:.4f} (n={len(o50)})")

# ---- per-ARM ranking ----
arm_old, arm_new = collections.defaultdict(list), collections.defaultdict(list)
for i, d in enumerate(runs):
    m = re.match(r"^\d{14}_(.+)_s\d+_gH\d+$", d)
    a = m.group(1) if m else d
    arm_old[a].append(old[i]); arm_new[a].append(new[i])
A = sorted(arm_old)
ao = np.array([np.mean(arm_old[a]) for a in A]); an_ = np.array([np.mean(arm_new[a]) for a in A])
rho4, p5 = stats.spearmanr(ao, an_)
print(f"Spearman over the {len(A)} ARMS (mean over seeds) = {rho4:.4f}")
big = [a for a in A if len(arm_old[a]) >= 6]
if big:
    rho5, _ = stats.spearmanr([np.mean(arm_old[a]) for a in big], [np.mean(arm_new[a]) for a in big])
    print(f"Spearman over the {len(big)} arms with >=6 seeds = {rho5:.4f}")
print()
print("top arms by NEW terminal block-only rate (>=6 evaluated seeds):")
for a in sorted(big, key=lambda x: -np.mean(arm_new[x]))[:12]:
    print(f"  {a:42s} old={np.mean(arm_old[a]):.3f}  new={np.mean(arm_new[a]):.3f}  n={len(arm_old[a])}")
json.dump(dict(runs=runs, old=old.tolist(), new=new.tolist()), open("/tmp/m2_rescore/perrun.json","w"))
