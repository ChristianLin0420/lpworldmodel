import json, re, os, pickle, collections
import numpy as np
from scipy import stats

fr = json.load(open("/tmp/m2_rescore/frames.json"))
BIAS = np.array([0.607, 0.472]); THR, ATHR = 20.0, np.pi/9
tg = {}
for d in {r["dir"] for r in fr}:
    tg[d] = np.asarray(pickle.load(open(f"plan_outputs/{d}/plan_targets.pkl","rb"))["state_g"], float)
for r in fr:
    g = tg[r["dir"]][r["ep"]]
    r["bl"] = float(np.linalg.norm(np.array([r["bx"],r["by"]])-BIAS - g[2:4]))
    r["sb"] = bool(r["bl"] < THR and r["angle_diff"] < ATHR)
    r["free0"] = bool(r["block_pos_diff_t0"] < THR and r["angle_diff_t0"] < ATHR)

def arm_of(d):
    m = re.match(r"^\d{14}_(.+)_pd\d+.*_s\d+_gH\d+$", d)
    m2 = re.match(r"^\d{14}_(.+)_s\d+_gH\d+$", d)
    return m.group(1) if m else (m2.group(1) if m2 else d)

byrun = collections.defaultdict(list)
for r in fr: byrun[r["dir"]].append(r)

fixed = set(json.load(open("campaign_fixed.json"))["arms"])
arm = collections.defaultdict(lambda: dict(o=[], n=[], om=[], nm=[]))
for d, v in byrun.items():
    if len(v) < 5: continue
    a = arm_of(d)
    s = np.array([x["latched_success"] for x in v]); b = np.array([x["sb"] for x in v])
    f = np.array([x["free0"] for x in v])
    arm[a]["o"].append(s.mean()); arm[a]["n"].append(b.mean())
    if (~f).sum() >= 3:
        arm[a]["om"].append(s[~f].mean()); arm[a]["nm"].append(b[~f].mean())

def rho(A, ko, kn, minseed=1):
    A = [a for a in A if len(arm[a][ko]) >= minseed and len(arm[a][kn]) >= minseed]
    x = [np.mean(arm[a][ko]) for a in A]; y = [np.mean(arm[a][kn]) for a in A]
    r, p = stats.spearmanr(x, y); return r, p, len(A)

allA = sorted(arm)
fixedA = sorted(a for a in allA if a in fixed)
print(f"arms present: {len(allA)}; of which in campaign_fixed.json: {len(fixedA)}")
for name, A, ms in [("all arms", allA, 1), ("all arms, >=4 seeds", allA, 4),
                    ("campaign_fixed arms", fixedA, 1), ("campaign_fixed, >=4 seeds", fixedA, 4)]:
    r, p, n = rho(A, "o", "n", ms)
    rm, pm, nm = rho(A, "om", "nm", ms)
    print(f"{name:28s} n={n:3d}  Spearman(old,new_blockonly)={r:.4f}   restricted-to-must-push={rm:.4f}")

# bootstrap CI over runs for the per-run spearman
runs = [d for d,v in byrun.items() if len(v)>=5]
o = np.array([np.mean([x["latched_success"] for x in byrun[d]]) for d in runs])
nn = np.array([np.mean([x["sb"] for x in byrun[d]]) for d in runs])
rs = []
rng = np.random.RandomState(0)
for _ in range(2000):
    idx = rng.randint(0, len(runs), len(runs))
    rs.append(stats.spearmanr(o[idx], nn[idx]).correlation)
print(f"per-run Spearman = {stats.spearmanr(o,nn).correlation:.4f}  95% CI [{np.percentile(rs,2.5):.4f}, {np.percentile(rs,97.5):.4f}]  n={len(runs)}")

# rank movement of the arms
A = [a for a in allA if len(arm[a]['o'])>=4]
o_ = np.array([np.mean(arm[a]['o']) for a in A]); n_ = np.array([np.mean(arm[a]['n']) for a in A])
ro = stats.rankdata(-o_); rn = stats.rankdata(-n_)
mv = sorted(zip(A, ro, rn, o_, n_), key=lambda t: -(t[1]-t[2]))
print(f"\nlargest rank GAINS under block-only (of {len(A)} arms with >=4 seeds):")
for a,r1,r2,x,y in mv[:6]: print(f"  {a:32s} rank {r1:4.1f} -> {r2:4.1f}   old={x:.3f} new={y:.3f}")
print("largest rank LOSSES:")
for a,r1,r2,x,y in mv[-6:]: print(f"  {a:32s} rank {r1:4.1f} -> {r2:4.1f}   old={x:.3f} new={y:.3f}")
print(f"\nmean |rank change| = {np.abs(ro-rn).mean():.2f} of {len(A)} arms; max = {np.abs(ro-rn).max():.0f}")
