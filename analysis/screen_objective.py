"""Does a candidate quantity predict planning, or is it just a collapse detector?

This campaign has now caught TWO quantities that look like strong predictors of CEM success and
are not:

    d_action    raw Spearman +0.599 over 235 runs   ->  partial (rho, rel_mse removed) +0.545,
                                                        but NON-MONOTONE, an inverted U whose
                                                        optimum the baseline already sat at, and
                                                        nine arms across two rounds moved it the
                                                        wrong way (diary 2026-09-03 s12b)
    h8/h1       raw Spearman +0.558 over 323 runs   ->  partial -0.017, p = 0.77
                                                        +0.002 among healthy predictors

Both would have been caught here, before an arm was built. The failure mode both share: a
quantity that separates DEAD models from LIVE ones will correlate with anything, because dead
models plan at zero. What matters is whether it orders the models that already work.

So this screen reports four things, and a candidate must survive all four:

  1. raw Spearman against CEM                    -- the number that fools you
  2. rank-partial with rho and rel_mse removed,  -- the number that matters, with a permutation
     against a permutation null                     null so no distributional assumption is made
  3. the same, restricted to predictors that     -- a cut, reported as a robustness check and
     actually predict                                never as the headline
  4. binned means over quartiles                 -- because a partial correlation cannot see a
                                                    non-monotone relationship, which is exactly
                                                    what d_action turned out to be

NOTHING IN THIS FILE THRESHOLDS THE CANDIDATE. Item 3's cut is on rel_mse, a property of the
metric rather than of the candidate, and the campaign's rule against fitted thresholds (design
rule 5) is why item 2 is the headline.

    python analysis/screen_objective.py --key rollout/val_z_visual_err_rollout_h8 \
        --over rollout/val_z_visual_err_rollout_h1 --campaign campaign.json
    python analysis/screen_objective.py --key sparsity/effective_dim --campaign campaign.json
    python analysis/screen_objective.py --list          # what keys are available to screen
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np
from scipy import stats

RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")
CAMPAIGN_ALIAS = {"PiWM-columns": "PiWM-columns_patch"}
# The two covariates every candidate must be partialled against. rho is code density and
# rel_mse is prediction error: between them they identify a dead model, which is the confound.
COVARS = ("sparsity/val_l0_frac", "err/rel_mse")
HEALTHY_MAX_REL_MSE = 0.05      # a cut on the METRIC, not on the candidate


def _rank(v):
    return stats.rankdata(np.asarray(v, float))


def partial_spearman(x, y, covars, n_perm=20000, seed=0):
    """Rank partial correlation with a permutation p-value. Same estimator as causal_figs."""
    x, y = _rank(x), _rank(y)
    C = np.column_stack([np.ones(len(x))] + [_rank(c) for c in covars])
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    rx, ry = resid(x), resid(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(n_perm)])
    return r, float((np.abs(null) >= abs(r)).mean())


def harvest(repo, campaign, key, over=None):
    """One row per evaluated run: the candidate, the covariates and the CEM number."""
    cem = json.load(open(campaign))["arms"] if os.path.exists(campaign) else {}
    rows = []
    for d in sorted(glob.glob(os.path.join(repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        m = RUN_RE.match(name)
        f = os.path.join(d, "wandb/latest-run/files/wandb-summary.json")
        # CANARY-* are 200-step liveness probes, not experiments (see causal_figs.harvest)
        if not m or name.startswith("CANARY-") or not os.path.exists(f):
            continue
        try:
            s = json.load(open(f))
        except Exception:
            continue
        arm, seed = m.group(1), m.group(2)
        c = cem.get(CAMPAIGN_ALIAS.get(arm, arm), {}).get(seed)
        v, cov = s.get(key), [s.get(k) for k in COVARS]
        if over is not None:
            den = s.get(over)
            v = None if (v is None or not den) else v / den
        if c is None or v is None or any(x is None for x in cov):
            continue
        rows.append(dict(arm=arm, seed=seed, val=float(v), cem=float(c),
                         rho=float(cov[0]), rel_mse=float(cov[1])))
    return rows


def screen(rows, label):
    v = np.array([r["val"] for r in rows])
    c = np.array([r["cem"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    e = np.array([r["rel_mse"] for r in rows])
    raw = stats.spearmanr(v, c)
    par, p = partial_spearman(v, c, (rho, e))
    print(f"\n=== {label} ===")
    print(f"  n runs, {len(rows)} over {len(set(r['arm'] for r in rows))} arms")
    print(f"  1. raw Spearman vs CEM              {raw.correlation:+.3f}   p = {raw.pvalue:.2g}")
    print(f"  2. PARTIAL, rho and rel_mse removed {par:+.3f}   p = {p:.4f}   <- the headline")
    h = [r for r in rows if r["rel_mse"] < HEALTHY_MAX_REL_MSE]
    hs = None
    if len(h) >= 20:
        hv = np.array([r["val"] for r in h]); hc = np.array([r["cem"] for r in h])
        hs = stats.spearmanr(hv, hc).correlation
        print(f"  3. healthy only (rel_mse < {HEALTHY_MAX_REL_MSE}) {hs:+.3f}   n = {len(h)}"
              f"   (a cut -- robustness check, not the headline)")
        q = np.quantile(hv, [0, .25, .5, .75, 1.0])
        print("  4. binned means over the healthy band (a partial cannot see a non-monotone "
              "relation):")
        for i in range(4):
            m = (hv >= q[i]) & (hv <= q[i + 1] if i == 3 else hv < q[i + 1])
            if m.sum():
                print(f"       {q[i]:10.4f} - {q[i + 1]:10.4f}   n = {m.sum():3d}   "
                      f"mean CEM {hc[m].mean():.3f}")
        # Monotonicity is a property of the ORDERING of the binned means, not of the sign of a
        # quadratic term: a strictly decreasing but convex relation (S_model: .407 .365 .249
        # .058) has a large negative quadratic and is perfectly monotone. Judging it by the
        # quadratic alone mislabels a usable target as an inverted U, which is the mistake this
        # line originally made.
        means = [hc[(hv >= q[i]) & (hv <= q[i + 1] if i == 3 else hv < q[i + 1])].mean()
                 for i in range(4)]
        mono = (all(np.diff(means) >= 0) or all(np.diff(means) <= 0))
        z = (_rank(hv) - len(hv) / 2) / len(hv)
        quad = np.linalg.lstsq(np.column_stack([np.ones(len(hv)), z, z ** 2]),
                               _rank(hc), rcond=None)[0][2]
        shape = ("MONOTONE" if mono else
                 "NON-MONOTONE -- an interior optimum, so raising it is not a direction")
        print(f"     binned means {shape}   (rank-quadratic {quad:+.1f}, i.e. curvature)")
    verdict = ("FAILS -- this is a collapse detector, not a direction"
               if abs(par) < 0.15 or p > 0.05 else
               "SURVIVES the partial; check item 4 for monotonicity before targeting it")
    print(f"  VERDICT: {verdict}")
    return dict(n=len(rows), raw=float(raw.correlation), partial=par, p=p, healthy=hs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--campaign", default="campaign.json")
    ap.add_argument("--key", help="wandb-summary key to screen")
    ap.add_argument("--over", default=None, help="divide --key by this key (e.g. an h1 baseline)")
    ap.add_argument("--list", action="store_true", help="list screenable keys and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.list:
        keys = collections.Counter()
        for d in sorted(glob.glob(os.path.join(a.repo, "runs/outputs/*/")))[:400]:
            f = os.path.join(d, "wandb/latest-run/files/wandb-summary.json")
            if os.path.exists(f):
                try:
                    keys.update(k for k, v in json.load(open(f)).items()
                                if isinstance(v, (int, float)))
                except Exception:
                    pass
        for k, n in sorted(keys.items()):
            print(f"  {n:4d}  {k}")
        return

    if not a.key:
        ap.error("--key is required (or --list)")
    rows = harvest(a.repo, a.campaign, a.key, a.over)
    if len(rows) < 20:
        print(f"only {len(rows)} evaluated runs carry '{a.key}' -- too few to screen. "
              f"NOTE: a key logged only by recent arms gives a SELECTION-BIASED population, "
              f"which is exactly how section 8's d_action conclusion went wrong.")
        return
    label = a.key + (f"  /  {a.over}" if a.over else "")
    res = screen(rows, label)
    if a.out:
        json.dump(dict(key=a.key, over=a.over, **res), open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
