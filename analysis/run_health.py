"""The two failure detectors this campaign actually needed, and did not have.

Six rounds ran with one pre-registered gate, `err/rel_mse >= 0.5`. It misses both failure
modes the campaign has actually met:

  * TOTAL REPRESENTATIONAL COLLAPSE. A constant-output model has rel_mse ~ 0, so the gate
    scores it healthy. Measured: `sparsity/effective_dim == 0` is the only perfect predictor
    of SR ~ 0 in the archive -- 24 runs across 12 arms, every evaluated one at SR <= 0.04.
    PiWM-drop95 and PiWM-incr-eps0p001 have their *highest-scoring* seeds in this state, so
    reading SR alone inverts the ranking (see diary/2026-09-04 §7.0b).

  * A DIVERGING ROLLOUT. Nothing in the training-time logs sees it, because the model can be
    a perfectly good 1-step predictor and still drive the simulation off the table. Measured
    from the eval traces, `median agent_pos_diff` separates every healthy arm from every
    failing one with NO OVERLAP:

        LpWM-ltv 17.1 px   vote5-borda 13.3   patchdecode 16.7   sam-r0p03 27.8
        consist-w0p3 2689.6   contact-shuf 1290.9   incr-clip10 1281.5   support-w0p3 1019.3

    This is why "re-score on the block alone" is a trap: dropping the agent term removes the
    only signal that catches divergence, and it lifted those arms from 0.003-0.06 to
    0.22-0.29 (diary/README §7, the retraction).

Both are free. The first reads a wandb summary, the second reads traces already on disk.

    python analysis/run_health.py                 # every arm
    python analysis/run_health.py --arm PiWM-tok  # substring filter
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN = re.compile(r"(.+)_pd\d+.*_s(\d+)$")
TRACE = re.compile(r"^\d+_(.+)_gH5$")

# 100 px is five times the 20 px success radius: not "did it park", but "is the simulation
# still on the table". Every healthy arm measured sits under 30.
DIVERGENCE_PX = 100.0


def summary(run):
    """The run's FINAL wandb summary. glob() is unsorted and a chained run has one window per
    resume, so [0] returns an arbitrary -- often mid-training -- window."""
    d = f"runs/outputs/{run}/"
    for f in ([d + "wandb/latest-run/files/wandb-summary.json"]
              + sorted(glob.glob(d + "wandb/run-*/files/wandb-summary.json"))[::-1]):
        if os.path.exists(f):
            try:
                return json.load(open(f))
            except Exception:
                continue
    return {}


def _get(s, sub):
    return next((s[k] for k in s if sub in k), None)


def collapse_check(run):
    """effective_dim == 0 -> the encoder emits a constant. rel_mse cannot see this."""
    s = summary(run)
    ed = _get(s, "effective_dim")
    rm = _get(s, "rel_mse")
    return ed, rm, (ed is not None and ed <= 0.0)


def divergence_check():
    """median ||agent - goal|| per run, from the eval traces. -> {run: (median, frac_diverged)}"""
    out = {}
    for d in sorted(glob.glob("plan_outputs/*_gH5/")):
        f = os.path.join(d, "traces_output_final.npz")
        m = TRACE.match(os.path.basename(d.rstrip("/")))
        if not (m and os.path.exists(f)):
            continue
        try:
            z = np.load(f)
        except Exception:
            continue
        if "agent_pos_diff" not in z:
            continue
        ap = np.asarray(z["agent_pos_diff"])
        if ap.ndim > 1:
            ap = ap[:, -1]
        out[m.group(1)] = (float(np.median(ap)), float((ap > DIVERGENCE_PX).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default=None, help="substring filter on the arm name")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    div = divergence_check()
    rows = []
    for d in sorted(glob.glob("runs/outputs/*/")):
        run = os.path.basename(d.rstrip("/"))
        if run.startswith("CANARY-") or not os.path.exists(d + "DONE"):
            continue
        m = RUN.match(run)
        if not m or (a.arm and a.arm not in m.group(1)):
            continue
        ed, rm, collapsed = collapse_check(run)
        md, fd = div.get(run, (None, None))
        rows.append(dict(run=run, arm=m.group(1), seed=int(m.group(2)),
                         effective_dim=ed, rel_mse=rm, collapsed=collapsed,
                         agent_med=md, agent_frac_diverged=fd,
                         diverged=(md is not None and md > DIVERGENCE_PX)))

    bad = [r for r in rows if r["collapsed"] or r["diverged"]]
    print(f"  {len(rows)} finished runs, {len(bad)} flagged\n")
    if bad:
        print(f"  {'run':44s} {'eff_dim':>8s} {'rel_mse':>8s} {'agent px':>9s}  why")
        for r in sorted(bad, key=lambda x: x["run"]):
            why = ",".join([w for w, c in (("COLLAPSED", r["collapsed"]),
                                           ("DIVERGED", r["diverged"])) if c])
            print(f"  {r['run']:44s} {r['effective_dim'] if r['effective_dim'] is not None else float('nan'):8.1f} "
                  f"{r['rel_mse'] if r['rel_mse'] is not None else float('nan'):8.4f} "
                  f"{r['agent_med'] if r['agent_med'] is not None else float('nan'):9.1f}  {why}")
        # the point of the module: how many would rel_mse alone have caught?
        miss = [r for r in bad if not (r["rel_mse"] is not None and r["rel_mse"] >= 0.5)]
        print(f"\n  of those, {len(miss)} would NOT be caught by the pre-registered "
              f"`rel_mse >= 0.5` gate alone")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\n  wrote {a.json}")


if __name__ == "__main__":
    main()
