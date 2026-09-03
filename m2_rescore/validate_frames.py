"""Validate the pose recovery ON THE ARCHIVE IMAGES THEMSELVES.

Frame 0 of every strip is the t=0 render, whose true state is `state_0` in
plan_targets.pkl.  Recovering it and comparing gives a ground-truth error
distribution on exactly the images Part B consumes."""
import glob, os, pickle, re, sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/tmp/m2_rescore")
import imageio.v2 as iio, poserec
R = 224


def do_dir(d):
    try:
        t = pickle.load(open(os.path.join(d, "plan_targets.pkl"), "rb"))
    except Exception:
        return []
    s0 = np.asarray(t["state_0"], float)
    out = []
    for p in sorted(glob.glob(os.path.join(d, "output_final_real_*_*.png"))):
        m = re.match(r"output_final_real_(\d+)_(success|failure)\.png$", os.path.basename(p))
        if not m: continue
        i = int(m.group(1))
        if i >= len(s0): continue
        fr = iio.imread(p)[:, :R, :3]
        pose, iou, n = poserec.fit_pose(fr)
        if pose is None: continue
        out.append((float(np.linalg.norm(pose[:2] - s0[i, 2:4])),
                    float(abs((pose[2] - s0[i, 4] + np.pi) % (2*np.pi) - np.pi)),
                    float(iou), float(pose[0]-s0[i,2]), float(pose[1]-s0[i,3])))
    return out


if __name__ == "__main__":
    dirs = sorted(os.path.dirname(f) for f in glob.glob("plan_outputs/*/plan_targets.pkl"))
    import random; random.seed(0); dirs = random.sample(dirs, min(60, len(dirs)))
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        res = [r for c in pool.imap_unordered(do_dir, dirs) for r in c]
    a = np.array(res)
    print(f"n={len(a)} validation frames (t=0, true pose known exactly)")
    print(f"pos err (world px): median={np.median(a[:,0]):.3f}  p90={np.percentile(a[:,0],90):.3f}  p99={np.percentile(a[:,0],99):.3f}  max={a[:,0].max():.3f}")
    print(f"ang err (deg):      median={np.degrees(np.median(a[:,1])):.3f}  p90={np.degrees(np.percentile(a[:,1],90)):.3f}  max={np.degrees(a[:,1].max()):.3f}")
    print(f"fit IoU: median={np.median(a[:,2]):.4f}  p1={np.percentile(a[:,2],1):.3f}")
    print(f"signed bias dx={a[:,3].mean():.3f} dy={a[:,4].mean():.3f}")
    print(f"frac pos err > 2px: {np.mean(a[:,0]>2):.4f} ; > 5px: {np.mean(a[:,0]>5):.4f}")
    np.save("/tmp/m2_rescore/valid.npy", a)
