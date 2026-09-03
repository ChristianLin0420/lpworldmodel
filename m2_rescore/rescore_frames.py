"""Part B of the M2 rescore: recover the TERMINAL block pose of every plotted
archive episode from its final rendered frame, and re-score it block-only."""
import glob, os, pickle, re, sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/tmp/m2_rescore")
sys.path.insert(0, os.environ.get("REPO", "."))
import imageio.v2 as iio
import poserec

R = 224
SEP = 12


def last_exec_frame(strip):
    """strip = [f0 .. f_{nf-1}] + 12px separator + goal panel, each frame R wide"""
    w = strip.shape[1]
    nf = (w - SEP - R) // R
    if nf < 1 or nf * R + SEP + R != w:
        return None, nf
    return strip[:, (nf - 1) * R: nf * R], nf


def do_dir(d):
    try:
        t = pickle.load(open(os.path.join(d, "plan_targets.pkl"), "rb"))
    except Exception:
        return []
    sg = np.asarray(t["state_g"], float)
    s0 = np.asarray(t["state_0"], float)
    if sg.ndim != 2:
        return []
    out = []
    for p in sorted(glob.glob(os.path.join(d, "output_final_real_*_*.png"))):
        m = re.match(r"output_final_real_(\d+)_(success|failure)\.png$", os.path.basename(p))
        if not m:
            continue
        i = int(m.group(1))
        if i >= len(sg):
            continue
        try:
            strip = iio.imread(p)[:, :, :3]
        except Exception:
            continue
        fr, nf = last_exec_frame(strip)
        if fr is None:
            continue
        pose, iou_fit, npx = poserec.fit_pose(fr)
        if pose is None:
            continue
        bl = float(np.linalg.norm(pose[:2] - sg[i, 2:4]))
        an = abs((pose[2] - sg[i, 4] + np.pi) % (2 * np.pi) - np.pi)
        bl0 = float(np.linalg.norm(s0[i, 2:4] - sg[i, 2:4]))
        an0 = abs((s0[i, 4] - sg[i, 4] + np.pi) % (2 * np.pi) - np.pi)
        out.append(dict(dir=os.path.basename(d), ep=i,
                        latched_success=(m.group(2) == "success"),
                        bx=float(pose[0]), by=float(pose[1]), bth=float(pose[2]),
                        fit_iou=float(iou_fit), n_px=int(npx), n_frames=int(nf),
                        block_pos_diff=bl, angle_diff=float(an),
                        block_pos_diff_t0=bl0, angle_diff_t0=float(an0)))
    return out


if __name__ == "__main__":
    dirs = sorted(os.path.dirname(f) for f in glob.glob("plan_outputs/*/plan_targets.pkl"))
    with Pool(int(os.environ.get("NPROC", 24))) as pool:
        res = [r for chunk in pool.imap_unordered(do_dir, dirs) for r in chunk]
    json.dump(res, open("/tmp/m2_rescore/frames.json", "w"))
    print("episodes recovered:", len(res), "over", len({r['dir'] for r in res}), "dirs")
