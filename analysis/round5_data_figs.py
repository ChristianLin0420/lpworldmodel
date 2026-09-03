"""Round-5 data figures: the dissociation, the motion tail, and the P1-P6 contrasts.

These are the three measurements the round-5 proposals rest on, drawn in the same style as
analysis/causal_figs.py (threshold-free, arm colours fixed by identity, single-hue sequential
where a magnitude is encoded).

    python analysis/round5_data_figs.py --out diary/assets/2026-09-03 [--campaign campaign.json]
"""
import argparse
import collections
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.causal_figs import (ARM_COLOR, GRID, INK, LABEL, MUTED, _style,
                                  harvest, partial_spearman)

ACCENT = {"blue": "#0055af", "magenta": "#912d59", "green": "#006e00",
          "amber": "#9a6700", "crit": "#b3261e", "slate": "#3d3d3a"}

# Whether the arm's predictor keeps a nonlinear readout. This is the factor that turned out to
# matter; d_action, which two rounds were designed around, did not.
NONLINEAR = {
    "LpWM-ltv": True, "LpWM-ltv-mupfix": True, "PiWM-actgain-b03": True,
    "PiWM-actgain-b30": True, "PiWM-columns": True, "LpWM-ltv-relu-p2": True,
    "LpWM-linvar": False, "PiWM-multact": False, "PiWM-lie": False, "PiWM-lie-sim": False,
    "PiWM-ctrb": False,
}
DATASET = "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/dinowm/pusht_noise/train"


# ------------------------------------------------------------------ figure 1
PROBE = "assets/d_action_probe.json"


def _probe_rows(campaign):
    """One row per evaluated run, with d_action re-measured from the checkpoint.

    `causal/d_action` is only in the summaries of runs trained after the diagnostic was
    added, and those exclude LpWM-ltv, d2048, vfloor, mupfix and the whole gate family --
    i.e. every high-CEM arm. Section 8 was computed on what was left. This uses
    analysis/d_action_probe.py, which re-measures every checkpoint on one fixed batch with
    one fixed permutation, and agrees with the logged value at Spearman +0.992 (n = 145).
    """
    probe = {x["run"]: x for x in json.load(open(PROBE))}
    cem = json.load(open(campaign))["arms"]
    alias = {"PiWM-columns": "PiWM-columns_patch"}
    run_re = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")
    out = []
    for d in sorted(glob.glob("runs/outputs/*/")):
        n = os.path.basename(d.rstrip("/"))
        m = run_re.match(n)
        f = os.path.join(d, "wandb/latest-run/files/wandb-summary.json")
        if not m or n not in probe or not os.path.exists(f):
            continue
        try:
            s = json.load(open(f))
        except Exception:
            continue
        c = cem.get(alias.get(m.group(1), m.group(1)), {}).get(m.group(2))
        if c is None or s.get("err/rel_mse") is None or s.get("sparsity/val_l0_frac") is None:
            continue
        out.append(dict(arm=m.group(1), da=probe[n]["d_action_over_scale"], cem=c,
                        rel_mse=s["err/rel_mse"], rho=s["sparsity/val_l0_frac"]))
    return out


def fig_dissociation(rows, out, campaign="/tmp/p16.json"):
    """d_action, re-measured on every checkpoint, against planning success.

    The name is historical: this figure was drawn to show a double dissociation and the
    measurement refuted it. What it shows instead is an inverted U with the baseline
    already sitting just below the peak.
    """
    if not os.path.exists(PROBE):
        return None
    R = _probe_rows(campaign)
    if len(R) < 50:
        return None
    d = np.array([r["da"] for r in R])
    c = np.array([r["cem"] for r in R])
    e = np.array([r["rel_mse"] for r in R])
    rho = np.array([r["rho"] for r in R])
    r_par, _, _, p_par = partial_spearman(d, c, (rho, e))
    z = (stats.rankdata(d) - len(d) / 2) / len(d)
    quad = np.linalg.lstsq(np.column_stack([np.ones(len(d)), z, z ** 2]),
                           stats.rankdata(c), rcond=None)[0][2]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.0, 5.9),
                                 gridspec_kw={"width_ratios": [1.75, 1]})
    NAMED = {"LpWM-ltv": ("baseline", ACCENT["crit"], (-52, 4)),
             "LpWM-ltv-d2048": ("d2048 (best arm)", ACCENT["green"], (48, 10)),
             "PiWM-actinfo": ("V2 actinfo", ACCENT["magenta"], (0, 16)),
             "PiWM-lie": ("P1 lie", ACCENT["blue"], (-4, 14))}
    for r in R:
        col, zo, sz = MUTED, 3, 34
        if r["arm"] in NAMED:
            col, zo, sz = NAMED[r["arm"]][1], 5, 78
        ax.scatter(r["da"], r["cem"], s=sz, color=col, alpha=0.55 if zo == 3 else 0.95,
                   edgecolor="white" if zo == 5 else "none", lw=0.8, zorder=zo)
    for arm, (lab, col, off) in NAMED.items():
        v = [r for r in R if r["arm"] == arm]
        if not v:
            continue
        mx, my = np.median([r["da"] for r in v]), np.median([r["cem"] for r in v])
        ax.annotate(lab, (mx, my), xytext=off, textcoords="offset points",
                    ha="center", fontsize=11.5, color=col, weight="bold", zorder=7)
    # binned mean of the healthy predictors -- the shape, without a curve fit
    h = np.array([(r["da"], r["cem"]) for r in R if r["rel_mse"] < 0.05])
    q = np.quantile(h[:, 0], np.linspace(0, 1, 8))
    sel = [(h[:, 0] >= q[i]) & (h[:, 0] <= q[i + 1]) for i in range(7)]
    mid = [(h[m, 1].mean(), float(np.median(h[m, 0]))) for m in sel if m.sum()]
    ax.plot([m[1] for m in mid], [m[0] for m in mid], color=INK, lw=2.4, zorder=6,
            marker="o", ms=7, mfc="white", mew=2)
    ax.annotate("septile means,\nhealthy predictors only", (mid[4][1], mid[4][0]),
                xytext=(66, -18), textcoords="offset points", fontsize=10.5, color=INK,
                ha="left")
    ax.set_xlabel(r"$d_{action}\,/\,\|z\|$, re-measured on every checkpoint", fontsize=12.5)
    ax.set_ylabel("CEM success rate", fontsize=12.5)
    ax.set_xlim(-0.05, 1.55)
    ax.set_ylim(-0.04, 0.82)
    _style(ax)
    ax.annotate(f"partial Spearman, removing $\\rho$ and rel_mse:  "
                f"{r_par:+.3f}   p < 0.0001   (n = {len(R)})\n"
                f"rank-quadratic term {quad:+.0f}: the relation is an INVERTED U, "
                "not monotone",
                (0.025, 0.97), xycoords="axes fraction", va="top", ha="left",
                fontsize=11.5,
                color=INK, bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f0", ec=GRID))

    # --- right: what round 4 intended to do to d_action, and what it did ---
    probe = {x["run"]: x for x in json.load(open(PROBE))}
    base = float(np.median([x["d_action_over_scale"] for k, x in probe.items()
                            if k.startswith("LpWM-ltv_pd")]))
    arms = [("actgain 3.0", "PiWM-actgain-b30"), ("actgain 0.3", "PiWM-actgain-b03"),
            ("P4 ctrb", "PiWM-ctrb"), ("P2 multact", "PiWM-multact"),
            ("P2 linvar", "LpWM-linvar"), ("P1 lie", "PiWM-lie")]
    ys = np.arange(len(arms))[::-1]
    for y, (lab, arm) in zip(ys, arms):
        v = [x["d_action_over_scale"] for k, x in probe.items() if k.startswith(arm + "_pd")]
        if not v:
            continue
        m = float(np.median(v))
        bx.barh(y, m, height=0.6, color=ACCENT["crit"], alpha=0.75, zorder=3)
        bx.annotate(f"{m / base:.2f}x", (m, y), xytext=(-8, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=11.5, color="white", weight="bold")
    bx.axvline(base, color=INK, lw=2.2, zorder=4)
    bx.annotate(f"baseline {base:.2f}", (base, 2.5), xytext=(9, 0), rotation=90,
                textcoords="offset points", fontsize=11.5, color=INK, weight="bold",
                va="center", ha="center")
    bx.set_yticks(ys)
    bx.set_yticklabels([a[0] for a in arms], fontsize=11.5)
    bx.set_xlim(0, 0.78)
    bx.set_xlabel(r"$d_{action}\,/\,\|z\|$", fontsize=12.5)
    _style(bx)
    bx.set_title("Every round-4 arm LOWERED it -- including both\narms built to raise it",
                 fontsize=12.5, color=INK, loc="left", pad=8)

    fig.suptitle("$d_{action}$ re-measured: the baseline was never action-inert, and the "
                 "relation to planning is an inverted U\n"
                 f"LpWM-ltv sits at {base:.2f}, not the 1.9e-04 quoted in train.py -- a "
                 "factor of 2900. That number motivated P1, P3, P4 and P5.\n"
                 "The peak of the curve is at ~0.6, just above the baseline, which is why "
                 "every arm that moved $d_{action}$ moved it the wrong way.",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    p = os.path.join(out, "dissociation.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 2
def _motion_stats(cache="assets/pusht_motion_stats.json"):
    """Per-model-step motion of agent and block. Cached: the load is ~20 s."""
    if os.path.exists(cache):
        return json.load(open(cache))
    import torch
    st = torch.load(f"{DATASET}/states.pth").float()
    ag, bl = st[..., :2], st[..., 2:4]
    fs = 5
    d_ag = (ag[:, fs::fs] - ag[:, :-fs:fs]).norm(dim=-1).flatten()
    d_bl = (bl[:, fs::fs] - bl[:, :-fs:fs]).norm(dim=-1).flatten()
    m = torch.isfinite(d_ag) & torch.isfinite(d_bl)
    d_ag, d_bl = d_ag[m].numpy(), d_bl[m].numpy()
    s = np.sort(d_bl)[::-1]
    tot = s.sum()
    out = dict(
        n=int(d_bl.size),
        block_q=[float(np.percentile(d_bl, q)) for q in (50, 75, 90, 95, 99)],
        agent_q=[float(np.percentile(d_ag, q)) for q in (50, 75, 90, 95, 99)],
        frac_block_moves=float((d_bl > 0.5).mean()),
        frac_agent_moves=float((d_ag > 0.5).mean()),
        lorenz=[float(s[: max(1, int(s.size * k / 1000))].sum() / tot) for k in range(1, 1001)],
        block_hist=np.histogram(np.log10(np.clip(d_bl, 1e-3, None)), bins=60)[0].tolist(),
        block_edges=np.histogram(np.log10(np.clip(d_bl, 1e-3, None)), bins=60)[1].tolist(),
    )
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def fig_motion_tail(out):
    """The task lives in a tail the mean objective cannot see."""
    S = _motion_stats()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    edges = np.array(S["block_edges"])
    a1.bar(edges[:-1], S["block_hist"], width=np.diff(edges), align="edge",
           color=ACCENT["blue"], alpha=0.55, edgecolor="white", lw=0.3)
    still = (1 - S["frac_block_moves"]) * 100
    a1.axvline(-3, color=ACCENT["crit"], lw=2.4)
    a1.annotate(f"median = 0.000 px\n{still:.0f}% of transitions move the\n"
                "block less than half a pixel",
                (0.30, 0.93), xycoords="axes fraction", va="top", fontsize=11.5,
                color=ACCENT["crit"], weight="bold")
    a1.set_xlim(-3.25, 3.1)
    a1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    a1.set_xticklabels(["0", "0.01", "0.1", "1", "10", "100", "1000"])
    a1.set_xlabel("block displacement per model step (px, log scale)", fontsize=11.5)
    a1.set_ylabel("transitions (thousands)", fontsize=11)
    a1.set_yticklabels([f"{int(t / 1000)}" for t in a1.get_yticks()])
    a1.set_title("The block does not move", fontsize=12.5, color=INK, loc="left")
    _style(a1)

    k = np.arange(1, 1001) / 10.0
    a2.plot(k, np.array(S["lorenz"]) * 100, color=ACCENT["blue"], lw=2.6, zorder=4)
    a2.plot([0, 100], [0, 100], color=MUTED, lw=1.3, ls=":", zorder=2)
    a2.annotate("uniform", (68, 68), rotation=31, fontsize=10, color=MUTED,
                ha="center", va="bottom")
    for pct, dy in ((1, -34), (5, 6)):
        y = S["lorenz"][int(pct * 10) - 1] * 100
        a2.plot([pct, pct], [0, y], color=ACCENT["crit"], lw=1.2, ls="--", zorder=3)
        a2.scatter([pct], [y], s=70, color=ACCENT["crit"], zorder=5)
        a2.annotate(f"top {pct}% of transitions carry\n{y:.1f}% of all block motion",
                    (pct, y), xytext=(30, dy), textcoords="offset points", fontsize=11.5,
                    color=ACCENT["crit"], weight="bold", va="center")
    a2.set_xlim(-2, 102)
    a2.set_ylim(-3, 108)
    a2.set_xlabel("share of transitions, largest block motion first (%)", fontsize=11.5)
    a2.set_ylabel("cumulative share of block motion (%)", fontsize=11)
    a2.set_title("...and when it does, it is concentrated", fontsize=12.5, color=INK,
                 loc="left")
    _style(a2)

    fig.suptitle("The objective is a mean over transitions; the task lives in the tail\n"
                 f"{S['n']:,} transitions at the model's own frameskip. The agent moves in "
                 f"{S['frac_agent_moves'] * 100:.0f}% of them and its motion is exactly linear "
                 "in the action -- so a linear predictor can match\nthe baseline on one-step "
                 "error (0.0095 vs 0.0092) while planning 4.5x worse. The 1-step MSE is "
                 "dominated by the 75% of steps where nothing happens.",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(out, "motion-tail.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 3
CONTRASTS = [
    ("P2  linvar", "LpWM-linvar", "LpWM-ltv"),
    ("P2  multact", "PiWM-multact", "LpWM-linvar"),
    ("P1  lie", "PiWM-lie", "LpWM-ltv"),
    ("P1b lie-sim", "PiWM-lie-sim", "PiWM-lie"),
    ("P3  actgain 0.3", "PiWM-actgain-b03", "LpWM-ltv"),
    ("P3  actgain 3.0", "PiWM-actgain-b30", "LpWM-ltv"),
    ("P4  ctrb", "PiWM-ctrb", "LpWM-linvar"),
    ("P5  actinfo-knn", "PiWM-actinfo-cond", "PiWM-actinfo"),
    ("P5b knn+sigreg", "PiWM-actinfo-cond-sigreg", "PiWM-actinfo-cond"),
]
XLO, XHI = -0.80, 0.32


def fig_p16(campaign, out):
    """Forest plot of every round-4 contrast against its own matched control."""
    arms = json.load(open(campaign))["arms"]
    rows = []
    for nm, a, c in CONTRASTS:
        A, C = arms.get(a, {}), arms.get(c, {})
        s = sorted(set(A) & set(C), key=int)
        # a 2-seed interval is t_{.975,1} = 12.7 standard errors wide -- wider than the
        # whole outcome range. It is reported as a point with its n, not as an interval.
        if len(s) < 3:
            rows.append((nm, c, len(s),
                         float(np.mean([A[k] - C[k] for k in s])) if s else None, None, None))
            continue
        d = np.array([A[k] - C[k] for k in s])
        se = d.std(ddof=1) / np.sqrt(len(d))
        tc = stats.t.ppf(.975, len(d) - 1)
        rows.append((nm, c, len(d), d.mean(), d.mean() - tc * se, d.mean() + tc * se))

    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ys = np.arange(len(rows))[::-1]
    for y, (nm, c, n, m, lo, hi) in zip(ys, rows):
        if m is None:
            ax.annotate("trained, no paired eval yet", (XLO + 0.02, y), fontsize=10.5,
                        va="center", color=MUTED, style="italic")
            continue
        col = ACCENT["crit"] if (hi is not None and hi < 0) else (
            ACCENT["green"] if (lo is not None and lo > 0) else ACCENT["slate"])
        if lo is not None:
            l, h = max(lo, XLO), min(hi, XHI)
            ax.plot([l, h], [y, y], color=col, lw=2.8, solid_capstyle="round", zorder=4)
            for edge, val, d in ((l, lo, -1), (h, hi, +1)):     # arrow cap where clipped
                if abs(val - edge) > 1e-9:                       # head lands ON the edge
                    ax.annotate("", (edge, y), xytext=(edge - d * 0.035, y), zorder=6,
                                arrowprops=dict(arrowstyle="-|>,head_width=0.32",
                                                color=col, lw=2.8, shrinkA=0, shrinkB=0))
            txt = f"{m:+.3f}   [{lo:+.2f}, {hi:+.2f}]"
        else:
            txt = f"{m:+.3f}   (n too small for an interval)"
        ax.scatter([m], [y], s=140, color=col, zorder=5, edgecolor="white", lw=1.2)
        ax.annotate(txt, (XHI + 0.03, y), fontsize=11, va="center", color=col,
                    annotation_clip=False, family="monospace")
    ax.axvline(0, color=INK, lw=1.4, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{nm}   (n={n})\nvs {c}" for nm, c, n, *_ in rows], fontsize=10.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(XLO, XHI)
    ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
    ax.set_xlabel("paired difference in CEM success rate vs the arm's own control  "
                  "(95% CI; arrow = interval runs past the axis)", fontsize=11.5)
    _style(ax)
    ax.set_title("Round 4: every contrast against its matched control\n"
                 "No positive. The two arms that remove the predictor's nonlinearity are the "
                 "round's largest effects and both are negative.",
                 fontsize=12.5, color=INK, loc="left", pad=12)
    fig.subplots_adjust(right=0.68)
    p = os.path.join(out, "p16-results.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--campaign", default="/tmp/p16.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = harvest(a.repo, a.campaign)
    for p in (fig_dissociation(rows, a.out), fig_motion_tail(a.out), fig_p16(a.campaign, a.out)):
        if p:
            print("  wrote", p)


if __name__ == "__main__":
    main()
