"""Round-6 result figures: the dose ladders, and the contrast forest.

Two figures, deliberately separate because they answer different questions and one of them is
answerable long before the other:

  round6-dose.png     TRAINING-TIME HEALTH per family, as the strength knob rises. Available as
                      soon as runs finish. Says nothing about planning -- 2026-09-04 §2 shows
                      nearly every in-isolation metric here is a collapse detector.
  round6-contrasts.png  CEM CONTRASTS against each arm's OWN control. The actual result. Renders
                      a visible "pending" row for any contrast below n=8 rather than drawing a
                      point that invites reading, because this campaign has four retractions from
                      early reads.

    python analysis/round6_results_figs.py --out diary/assets/2026-09-04
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from scipy import stats                                        # noqa: E402

from analysis.collect_evals import collect, resolve_arm, ArmNameError   # noqa: E402
from analysis.style import (C, FILL, GRID, INK, MUTED, ax_style,        # noqa: E402
                            fit, panel_title, suptitle, use_style)

PROBE = "assets/d_action_probe.json"

# Floor for the log-scaled error panel: where an EXACTLY-zero rel_mse is parked so it stays
# visible instead of sending log() to -inf.
FLOOR_B = 1e-4

# dose ladders, ordered by INCREASING intervention strength. R4's eps is an inverse knob --
# smaller eps is stronger reweighting -- so it is listed reversed to keep every row reading
# left-to-right as "stronger".
LADDERS = [
    ("R6 support", "green",  [("0.03", "support-w0p03"), ("0.1", "support-w0p1"),
                              ("0.3", "support-w0p3")]),
    ("R2 consist", "purple", [("0.03", "consist-w0p03"), ("0.1", "consist-w0p1"),
                              ("0.3", "consist-w0p3")]),
    ("R3 SAM",     "amber",  [("0.01", "sam-r0p01"), ("0.03", "sam-r0p03"),
                              ("0.1", "sam-r0p1")]),
    ("R4 incr",    "crimson", [("4.1e-2", "incr-eps0p041"), ("1e-2", "incr-eps0p01"),
                               ("1e-3", "incr-eps0p001")]),
]

# (label, treated, control). The control is the one each arm's OWN spec names -- R1's is the
# matched OVERSHOOT arm at the same K, R2's cem arm is against its -data twin.
CONTRASTS = [
    ("R6  support w=0.03", "PiWM-support-w0p03", "LpWM-ltv", "green"),
    ("R6  support w=0.1", "PiWM-support-w0p1", "LpWM-ltv", "green"),
    ("R6  support w=0.3", "PiWM-support-w0p3", "LpWM-ltv", "green"),
    ("R2  consist w=0.03", "PiWM-consist-w0p03", "LpWM-ltv", "purple"),
    ("R2  consist w=0.1", "PiWM-consist-w0p1", "LpWM-ltv", "purple"),
    ("R2  consist w=0.3", "PiWM-consist-w0p3", "LpWM-ltv", "purple"),
    ("R2  cem vs data", "PiWM-consist-w0p1", "PiWM-consist-w0p1-data", "plum"),
    ("R3  SAM rho=0.01", "PiWM-sam-r0p01", "LpWM-ltv", "amber"),
    ("R3  SAM rho=0.03", "PiWM-sam-r0p03", "LpWM-ltv", "amber"),
    ("R3  SAM rho=0.1", "PiWM-sam-r0p1", "LpWM-ltv", "amber"),
    ("R4  eps=4.1e-2", "PiWM-incr-eps0p041", "LpWM-ltv", "crimson"),
    ("R4  eps=1e-2", "PiWM-incr-eps0p01", "LpWM-ltv", "crimson"),
    ("R4  eps=1e-3", "PiWM-incr-eps0p001", "LpWM-ltv", "crimson"),
    ("R4  eps=4.1e-2 clip", "PiWM-incr-eps0p041-clip10", "LpWM-ltv", "crimson"),
    ("R1  jump vs overshoot K=2", "PiWM-jump2", "PiWM-overshoot2", "teal"),
    ("R1  jump vs overshoot K=3", "PiWM-jump3", "PiWM-overshoot3", "teal"),
    ("R1  jump vs overshoot K=8", "PiWM-jump8", "PiWM-overshoot8", "teal"),
]


def _summary(d):
    """The run's FINAL wandb summary.

    `glob.glob` is unsorted, and a preempted/chained run has one `run-*` window per
    resume, so `glob(...)[0]` returns an ARBITRARY window -- often a mid-training one.
    Measured over this archive: 101 of 314 multi-window runs disagree between `[0]` and
    the last window, and 8 of those flip a guard verdict (e.g. PiWM-support-w0p3_s8 reads
    d_action_over_scale 0.1663 one way and 0.2834 the other, either side of the 0.2746
    threshold). `wandb/latest-run` is the symlink wandb maintains for exactly this, and
    is what diary/README.md §3 prescribes; sorted()[-1] is the fallback when a run
    predates the symlink. collect_evals.py already has the mirror-image care ("sorted()
    puts the newest timestamp last"); this module simply did not inherit it.
    """
    cands = [os.path.join(d, "wandb/latest-run/files/wandb-summary.json")]
    cands += sorted(glob.glob(os.path.join(d, "wandb/run-*/files/wandb-summary.json")))[::-1]
    for f in cands:
        if not os.path.exists(f):
            continue
        try:
            return json.load(open(f))
        except Exception:
            continue
    return {}


def _get(s, exact=None, sub=None):
    if exact and exact in s:
        return s[exact]
    if sub:
        return next((s[k] for k in s if sub in k), None)
    return None


def arm_health(arm):
    """Finished, non-canary runs of one arm -> lists of (d_a/scale, rel_mse, S_model)."""
    ov, rm, sm = [], [], []
    for d in sorted(glob.glob(f"runs/outputs/PiWM-{arm}_pd*/")):
        if "CANARY-" in d or not os.path.exists(os.path.join(d, "DONE")):
            continue
        s = _summary(d)
        if not s:
            continue
        for box, val in ((ov, _get(s, exact="causal/d_action_over_scale")),
                         (rm, _get(s, sub="rel_mse")),
                         (sm, _get(s, exact="jacc/S_model"))):
            if isinstance(val, (int, float)):
                box.append(val)
    return ov, rm, sm


def baseline_probe():
    if not os.path.exists(PROBE):
        return None, None
    P = json.load(open(PROBE))
    v = sorted(r["d_action_over_scale"] for r in P if r.get("arm") == "LpWM-ltv")
    return (st.median(v), min(v)) if v else (None, None)


def fig_dose(out):
    """Health vs dose. Three panels: action sensitivity, prediction error, support dissimilarity.

    Layout notes, because the first draft had every one of these defects: the 3-line figure
    subtitle collided with the panel titles (fixed by reserving top margin explicitly rather than
    relying on tight_layout); the per-point "n=" labels overprinted the x tick labels in all three
    panels (fixed by showing n ONCE, in panel A, with per-family vertical offsets); and the
    "baseline" / "death condition" labels ran off the right edge (fixed by placing them inside the
    axes, left-aligned above their own line).
    """
    use_style()
    bmed, bmin = baseline_probe()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4))
    # Reserve the top band for the 3-line header. tight_layout cannot see figure-coordinate text.
    fig.subplots_adjust(wspace=0.30, top=0.74, bottom=0.13, left=0.055, right=0.985)
    panels = [
        ("A", "Action sensitivity", "d_action / |z|, relative to baseline", 0),
        ("B", "Prediction error", "err/rel_mse   (log scale)", 1),
        ("C", "Support dissimilarity", "jacc/S_model   (lower = R6's objective working)", 2),
    ]
    for letter, title, sub, idx in panels:
        ax = axes[idx]
        xs = np.arange(3)
        for fam, key, rungs in LADDERS:
            ys, ns = [], []
            for _, arm in rungs:
                ov, rm, sm = arm_health(arm)
                pick = (ov, rm, sm)[idx]
                ys.append(st.median(pick) if pick else np.nan)
                ns.append(len(pick))
            ys = np.array(ys, dtype=float)
            if idx == 0 and bmed:
                ys = ys / bmed
            # Panel B is log-scaled and R4's eps=1e-3 arm has rel_mse EXACTLY 0.0000 -- the ratio
            # of two collapsed quantities. log(0) is -inf and expands the canvas without bound
            # (observed: 405085 px). A degenerate arm must be drawn AS degenerate: parked on the
            # floor with a hollow marker, never silently dropped.
            zero = np.zeros(len(ys), dtype=bool)
            if idx == 1:
                zero = ys <= 0
                ys = np.where(zero, FLOOR_B, ys)
            good = ~np.isnan(ys)
            if not good.any():
                continue
            lab = fam if idx else f"{fam}   n={'/'.join(str(v) for v in ns)}"
            ax.plot(xs[good], ys[good], "-o", color=C[key], lw=2.1, ms=7.5,
                    label=lab, zorder=4, clip_on=False)
            if zero.any():
                ax.scatter(xs[zero], ys[zero], s=130, facecolor="white",
                           edgecolor=C[key], lw=1.9, zorder=5, clip_on=False)
                ax.annotate("exactly 0", (xs[zero][0], FLOOR_B), xytext=(-4, 12),
                            textcoords="offset points", ha="right", fontsize=8,
                            color=C[key], fontweight="bold")
        if idx == 0:
            ax.axhline(1.0, color=MUTED, lw=1.0, ls=":", zorder=1)
            ax.annotate("dotted = baseline", (2.22, 1.155), fontsize=8.5, color=MUTED,
                        va="bottom", ha="right")
            if bmin and bmed:
                ax.axhspan(0, bmin / bmed, color=FILL["crimson"], alpha=0.4, lw=0, zorder=0)
                # bottom-LEFT: the R4 line descends left-to-right, so it is clear here
                # while the right half of the band is exactly where R4 lands.
                ax.annotate("shaded = below the baseline's own minimum", (-0.2, 0.025),
                            fontsize=8, color=C["crimson"], va="bottom", ha="left")
            ax.set_ylim(0, 1.22)
            ax.set_ylabel("x baseline", fontsize=9.5, color=MUTED)
        if idx == 1:
            ax.set_yscale("log")
            ax.set_ylim(FLOOR_B * 0.5, 3.2)
            ax.axhline(0.0092, color=MUTED, lw=1.0, ls=":", zorder=1)
            ax.annotate("dotted = baseline 0.0092", (-0.2, 1.6e-4), fontsize=8.5,
                        color=MUTED, va="bottom", ha="left")
            ax.axhline(0.5, color=C["crimson"], lw=1.2, ls="--", zorder=1)
            ax.annotate("death condition", (2.22, 0.58), fontsize=8.5, ha="right",
                        color=C["crimson"], va="bottom", fontweight="bold")
        if idx == 2:
            ax.set_ylim(0, 1.14)
            ax.axhline(0.082, color=MUTED, lw=1.0, ls=":", zorder=1)
            ax.annotate("dotted = baseline 0.082", (-0.2, 1.05), fontsize=8.5,
                        color=MUTED, va="bottom", ha="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(["weakest", "middle", "strongest"], fontsize=9.5)
        ax.set_xlim(-0.25, 2.25)
        ax_style(ax)
        panel_title(ax, letter, title, sub)
        if idx == 0:
            ax.legend(loc="center left", fontsize=8.5, ncol=1, handlelength=1.6,
                      bbox_to_anchor=(0.015, 0.31), labelspacing=0.55,
                      title="n per rung, weakest→strongest", title_fontsize=8)
    fig.text(0.0, 0.985,
             "Every round-6 objective trades action sensitivity as its dose rises — except SAM",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.0, 0.935,
             "Training-time health only — NO CEM numbers. §2 shows nearly every in-isolation "
             "metric here is a collapse detector, so none of this may be read as a result.\n"
             "R4's ε is an inverse knob (smaller ε = stronger reweighting), so its ladder is "
             "reversed to keep every row reading left-to-right as 'stronger'.",
             fontsize=9.5, color=MUTED, ha="left", va="top")
    p = os.path.join(out, "round6-dose.png")
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return p


def fig_contrasts(out, floor=8):
    """The forest. A contrast below the n=8 floor is drawn as PENDING, never as a point."""
    use_style()
    A = collect()[0]
    rows = []
    for label, x, y, key in CONTRASTS:
        try:
            X, Y = A[resolve_arm(A, x)], A[resolve_arm(A, y)]
        except ArmNameError:
            rows.append((label, key, 0, None, None, None, False))
            continue
        s = sorted(set(X) & set(Y), key=int)
        if len(s) < 3:
            rows.append((label, key, len(s), None, None, None, False))
            continue
        d = np.array([X[k] - Y[k] for k in s])
        se = d.std(ddof=1) / np.sqrt(len(d))
        # A DEGENERATE interval is not a precise one. When every paired delta is identical the
        # sd is exactly 0 and the t-interval collapses to a point, which draws as infinite
        # confidence while actually carrying NO variance information. This is not hypothetical:
        # PiWM-blockcausal is 0.000 on 3/3 seeds against a baseline that is 0.380 on the same
        # 3, giving "-0.380 [-0.380, -0.380]". R4's arms are collapsing to identical values too
        # (rel_mse exactly 0.0000), so a round-6 contrast can reach the same state. Such a row
        # is flagged, not plotted as an interval.
        # NOT `se == 0`: np.std on three identical floats returns 6.8e-17, not 0.0, so an
        # exact test silently misses the very case it is written for (checked: it did).
        # Compare against the spread of the values themselves, scale-free.
        degenerate = bool(np.ptp(d) <= 1e-9 * max(1.0, abs(float(d.mean()))))
        t = stats.t.ppf(0.975, len(d) - 1)
        lo, hi = (d.mean(), d.mean()) if degenerate else (d.mean() - t * se, d.mean() + t * se)
        rows.append((label, key, len(s), d.mean(), lo, hi, degenerate))

    fig, ax = plt.subplots(figsize=(11.4, 0.42 * len(rows) + 2.5))
    ys = np.arange(len(rows))[::-1]
    for y, (label, key, n, d, lo, hi, degenerate) in zip(ys, rows):
        if n >= floor and degenerate:
            ax.scatter([d], [y], s=90, facecolor="white", edgecolor=C["crimson"], lw=1.9, zorder=4)
            ax.annotate(f"{d:+.3f} on {n}/{n} seeds — identical, no interval", (0.62, y),
                        fontsize=9, color=C["crimson"], va="center", ha="left",
                        annotation_clip=False, style="italic")
            continue
        if n < floor:
            ax.annotate(f"pending — n = {n} of {floor}", (0.0, y), ha="center", va="center",
                        fontsize=9, color=MUTED, style="italic",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F4F4F4", ec="#E0E0E0", lw=0.8))
            continue
        sig = lo > 0 or hi < 0
        col = C[key] if sig else MUTED
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([d], [y], s=78, color=col, zorder=4,
                   marker="D" if sig else "o")
        ax.annotate(f"{d:+.3f}  n={n}", (0.62, y), fontsize=9,
                    color=col, va="center", ha="left", annotation_clip=False,
                    fontweight="bold" if sig else "normal")
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel("paired Δ CEM success vs the arm's own control", fontsize=10.5)
    ax_style(ax, grid="x")
    for y in ys:
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#FAFAFA", zorder=0)
    n_ready = sum(1 for r in rows if r[2] >= floor)
    suptitle(fig, f"Round 6 contrasts — {n_ready} of {len(rows)} at the n = {floor} floor",
             "Each arm against the control its own spec names, never a generic baseline. "
             "A contrast below the floor is shown as pending, not as a point:\n"
             "T1 read −0.233 [−0.39, −0.08] at n = 3 and settled at a null by n = 8.")
    fit(fig)
    p = os.path.join(out, "round6-contrasts.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-04")
    ap.add_argument("--only", choices=["dose", "contrasts"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only in (None, "dose"):
        print("  wrote", fig_dose(a.out))
    if a.only in (None, "contrasts"):
        print("  wrote", fig_contrasts(a.out))


if __name__ == "__main__":
    main()
