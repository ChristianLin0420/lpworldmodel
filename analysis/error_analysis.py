"""M4: why does a failed episode fail? Four counts, declared before looking at anything.

Reads the per-episode traces M2 added to `PlanEvaluator` (`traces_output_final.npz`, one row
per episode), so it needs no GPU and no re-evaluation.

The counts, fixed in advance (docs/round5-specs.md, M4):
  (a) does the goal even require block motion --  ||block_goal - block_init|| > 5 px
  (b) which term fails, separately: agent xy, block xy, angle
  (c) "never even tried": the block never moved
  (d) per episode, CEM-PREDICTED improvement vs REALISED improvement -- this separates
      "the model was wrong" from "no good action was ever sampled"

NOTE ON (c). The spec asks for a zero-CONTACT count, which needs `n_contacts` out of a widened
`pusht_wrapper.rollout`. That does not exist yet, so (c) is reported as a strictly weaker proxy:
the block's total displacement over the episode. Zero displacement implies no effective contact;
nonzero displacement does not imply the agent ever intended it. Labelled PROXY in the output.

    python analysis/error_analysis.py --out assets/error_analysis.json [--arm LpWM-ltv]
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np

RUN_RE = re.compile(r"\d{14}_(.+?)_pd\d+_\w*?_?bf16_s(\d+)_gH\d+$")
MOVE_PX = 5.0          # a goal needs block motion if the block must travel further than this
STILL_PX = 1.0         # the block "never moved" if it travelled less than this in total
POS_TOL, ANG_TOL = 20.0, np.pi / 9      # env/pusht/pusht_wrapper.py:62


def _runs(pattern="plan_outputs/*/traces_output_final.npz"):
    for f in sorted(glob.glob(pattern)):
        m = RUN_RE.match(os.path.basename(os.path.dirname(f)))
        if m:
            yield m.group(1), int(m.group(2)), f


def analyse(arm_filter=None):
    per_arm = collections.defaultdict(lambda: collections.defaultdict(list))
    for arm, seed, f in _runs():
        if arm_filter and arm_filter not in arm:
            continue
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        need = ("state_0", "state_g", "e_state_final", "success", "block_pos_diff",
                "agent_pos_diff", "angle_diff", "d_pred", "d_real")
        if any(k not in d.files for k in need):
            continue
        s0, sg, sf = d["state_0"], d["state_g"], d["e_state_final"]
        a = per_arm[arm]
        a["must_push"].append(np.linalg.norm(sg[:, 2:4] - s0[:, 2:4], axis=1) > MOVE_PX)
        a["block_moved"].append(np.linalg.norm(sf[:, 2:4] - s0[:, 2:4], axis=1))
        a["success"].append(np.asarray(d["success"], bool))
        a["fail_agent"].append(np.asarray(d["agent_pos_diff"], float) > POS_TOL)
        a["fail_block"].append(np.asarray(d["block_pos_diff"], float) > POS_TOL)
        a["fail_angle"].append(np.asarray(d["angle_diff"], float) > ANG_TOL)
        a["d_pred"].append(np.asarray(d["d_pred"], float))
        a["d_real"].append(np.asarray(d["d_real"], float))
        a["seeds"].append(seed)

    out = {}
    for arm, a in per_arm.items():
        cat = {k: np.concatenate(v) for k, v in a.items() if k != "seeds"}
        ok, mp = cat["success"], cat["must_push"]
        fail = ~ok
        n = int(ok.size)
        row = dict(
            n_runs=len(a["seeds"]), n_episodes=n, success_rate=float(ok.mean()),
            # (a) does the goal require the block to move at all
            frac_goals_needing_block_motion=float(mp.mean()),
            success_when_no_motion_needed=float(ok[~mp].mean()) if (~mp).any() else None,
            success_when_motion_needed=float(ok[mp].mean()) if mp.any() else None,
            # (b) which term fails, among failures (terms overlap -- an episode can fail on
            # several, so these do not sum to 1)
            fail_on_agent=float(cat["fail_agent"][fail].mean()) if fail.any() else None,
            fail_on_block=float(cat["fail_block"][fail].mean()) if fail.any() else None,
            fail_on_angle=float(cat["fail_angle"][fail].mean()) if fail.any() else None,
            fail_on_angle_only=float((cat["fail_angle"] & ~cat["fail_agent"]
                                      & ~cat["fail_block"])[fail].mean())
            if fail.any() else None,
            # (c) PROXY for "never even tried"
            frac_block_never_moved=float((cat["block_moved"] < STILL_PX).mean()),
            frac_never_moved_among_must_push=float(
                (cat["block_moved"][mp] < STILL_PX).mean()) if mp.any() else None,
            median_block_displacement_px=float(np.median(cat["block_moved"])),
            # (d) predicted vs realised improvement
            median_d_pred=float(np.median(cat["d_pred"])),
            median_d_real=float(np.median(cat["d_real"])),
            frac_optimistic=float((cat["d_pred"] < cat["d_real"]).mean()),
        )
        with np.errstate(invalid="ignore"):
            good = np.isfinite(cat["d_pred"]) & np.isfinite(cat["d_real"])
            row["corr_pred_real"] = (float(np.corrcoef(cat["d_pred"][good],
                                                       cat["d_real"][good])[0, 1])
                                     if good.sum() > 2 else None)
        out[arm] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="assets/error_analysis.json")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--min-episodes", type=int, default=100)
    a = ap.parse_args()
    res = analyse(a.arm)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    big = {k: v for k, v in res.items() if v["n_episodes"] >= a.min_episodes}
    print(f"{len(res)} arms traced; {len(big)} with >= {a.min_episodes} episodes\n")
    hdr = (f"{'arm':26s}{'eps':>5s}{'succ':>6s}{'push%':>7s}{'s|no':>6s}{'s|yes':>7s}"
           f"{'fA':>6s}{'fB':>6s}{'fAng':>6s}{'still':>7s}{'optim':>7s}")
    print(hdr); print("-" * len(hdr))
    f = lambda x, w=6, p=3: (" " * w if x is None else f"{x:{w}.{p}f}")
    for arm, r in sorted(big.items(), key=lambda t: -t[1]["success_rate"]):
        print(f"{arm:26s}{r['n_episodes']:5d}{f(r['success_rate'])}"
              f"{f(r['frac_goals_needing_block_motion'],7,3)}"
              f"{f(r['success_when_no_motion_needed'])}{f(r['success_when_motion_needed'],7,3)}"
              f"{f(r['fail_on_agent'])}{f(r['fail_on_block'])}{f(r['fail_on_angle'])}"
              f"{f(r['frac_block_never_moved'],7,3)}{f(r['frac_optimistic'],7,3)}")
    print("\n  push% = goals needing block motion; s|no / s|yes = success rate given that")
    print("  fA/fB/fAng = share of FAILURES violating agent-xy / block-xy / angle (overlapping)")
    print("  still = PROXY, block moved < 1 px all episode;  optim = CEM predicted better than realised")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
