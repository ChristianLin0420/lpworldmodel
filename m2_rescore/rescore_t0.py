"""Part A of the M2 rescore: what the SCORED quantity is made of at t=0.

Reads only plan_outputs/*/plan_targets.pkl (present in every dir).  No GPU, no env.
"""
import glob, os, pickle, re, json, sys
import numpy as np

dirs = sorted(glob.glob("plan_outputs/*/plan_targets.pkl"))
rows, per_ep = [], []
for f in dirs:
    d = os.path.basename(os.path.dirname(f))
    try:
        t = pickle.load(open(f, "rb"))
    except Exception as ex:
        print("skip", d, ex); continue
    s0 = np.asarray(t["state_0"], dtype=np.float64)
    sg = np.asarray(t["state_g"], dtype=np.float64)
    if s0.ndim != 2 or s0.shape[1] < 5:
        continue
    ag = np.linalg.norm(sg[:, :2] - s0[:, :2], axis=1)
    bl = np.linalg.norm(sg[:, 2:4] - s0[:, 2:4], axis=1)
    an = np.abs(sg[:, 4] - s0[:, 4]); an = np.minimum(an, 2*np.pi - an)
    pos = np.linalg.norm(sg[:, :4] - s0[:, :4], axis=1)
    m = re.match(r"^(\d{14})_(.+)_s(\d+)_gH(\d+)$", d)
    rows.append(dict(dir=d, arm=(m.group(2) if m else d), seed=(int(m.group(3)) if m else -1),
                     n=len(ag),
                     agent_share=float((ag**2).sum() / (pos**2).sum()),
                     med_agent=float(np.median(ag)), med_block=float(np.median(bl)),
                     frac_block_ok_t0=float(np.mean((bl < 20) & (an < np.pi/9))),
                     frac_full_ok_t0=float(np.mean((pos < 20) & (an < np.pi/9)))))
    for i in range(len(ag)):
        per_ep.append((d, i, ag[i], bl[i], an[i], pos[i]))

A = np.array([[r["agent_share"], r["med_agent"], r["med_block"],
               r["frac_block_ok_t0"], r["frac_full_ok_t0"]] for r in rows])
ag = np.array([p[2] for p in per_ep]); bl = np.array([p[3] for p in per_ep])
an = np.array([p[4] for p in per_ep]); pos = np.array([p[5] for p in per_ep])
print(f"dirs={len(rows)}  episodes={len(per_ep)}")
print(f"agent share of sum pos_diff^2 at t=0 (pooled) = {100*(ag**2).sum()/(pos**2).sum():.1f}%")
print(f"median agent displacement to goal = {np.median(ag):.2f} px")
print(f"median block displacement to goal = {np.median(bl):.2f} px")
print(f"goals already satisfying BLOCK criterion at t=0 = {100*np.mean((bl<20)&(an<np.pi/9)):.1f}%")
print(f"goals already satisfying FULL  criterion at t=0 = {100*np.mean((pos<20)&(an<np.pi/9)):.1f}%")
print(f"episodes needing block motion (>20px or >pi/9)  = {100*np.mean(~((bl<20)&(an<np.pi/9))):.1f}%")
json.dump({"rows": rows}, open("/tmp/m2_rescore/t0.json", "w"))
np.savez("/tmp/m2_rescore/t0_per_ep.npz",
         d=np.array([p[0] for p in per_ep]), i=np.array([p[1] for p in per_ep]),
         agent=ag, block=bl, angle=an, pos=pos)
