"""Round-5 data figures: the dissociation, the motion tail, the P1-P6 forest, and the M1 probe.

Four measurements the round-5 proposals rest on. Every number is harvested at call time from
the checkpoints, the probe caches and the campaign JSON, so re-running after more evals land
refreshes the figures without editing anything.

Drawn in the one design system — analysis/style.py, derived from figures/motivation_teaser.svg.
Hues are assigned by IDENTITY, never by rank: green is the system, purple the contrasting
condition, amber the intervention under test, crimson failure, slate everything else.

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
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter
from scipy import stats

from analysis.causal_figs import LABEL, partial_spearman
from analysis.style import C, EDGE, FILL, GRID, HEAD, INK, MUTED, ax_style, use_style

use_style()

# A white halo behind a direct label, so a label that lands near a mark stays legible without
# a box. The reference uses white marker edges for the same reason.
HALO = [withStroke(linewidth=3.0, foreground="white")]

DATASET = "/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/dinowm/pusht_noise/train"


def _header(fig, title, subs=(), head_room=0.66):
    """Left-aligned figure header: bold finding, muted detail lines, in a reserved band.

    Returns the `top` for tight_layout(rect=...), reserving `head_room` inches underneath for
    the panel headers that _heads() draws afterwards. Everything is measured in INCHES and
    then converted, so the header cannot collide with the panels at any figure size — which
    is what the old `suptitle(y=1.0) + rect=[0, 0, 1, 0.86]` pair could not promise.
    """
    h = fig.get_figheight()
    y = 1.0 - 0.06 / h
    fig.text(0.004, y, title, fontsize=13.0, fontweight="bold", color=INK, ha="left", va="top")
    y -= 0.255 / h
    for s_ in subs:
        fig.text(0.004, y, s_, fontsize=10.5, color=MUTED, ha="left", va="top")
        y -= 0.205 / h
    return max(0.50, y - head_room / h)


def _heads(fig, specs, top=None, gap=0.26):
    """The reference's panel header — coloured letter, bold title, muted subtitle — drawn in
    FIGURE coordinates after the layout is final.

    style.panel_title() places the header in AXES coordinates, which (a) puts every panel's
    header at a different height when the panels differ in size and (b) sets the gap between
    the letter and the title as a fraction of the panel width, so the letter collides with the
    title on a narrow panel. Both are visible defects in the figures this replaces. Here the
    offsets are inches and the baseline is shared, so a row of headers always lines up.
    """
    w, h = fig.get_figwidth(), fig.get_figheight()
    if top is None:
        top = max(a.get_position().y1 for a, *_ in specs)
    y = top + gap / h
    for ax, letter, title, sub in specs:
        x = ax.get_position().x0
        fig.text(x, y, letter, fontsize=11.5, fontweight="bold", color=HEAD,
                 ha="left", va="bottom")
        fig.text(x + 0.20 / w, y, title, fontsize=11.5, fontweight="bold", color=INK,
                 ha="left", va="bottom")
        if sub:
            fig.text(x + 0.20 / w, y - 0.045 / h, sub, fontsize=9.5, color=MUTED,
                     ha="left", va="top")


def _leader(ax, xy, xytext, color, rad=0.0, lw=0.9, **kw):
    """A hairline leader from a direct label to the mark it owns."""
    ax.annotate("", xy=xy, xytext=xytext, textcoords=kw.pop("textcoords", "offset points"),
                zorder=kw.pop("zorder", 6),
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, shrinkA=3, shrinkB=5,
                                connectionstyle=f"arc3,rad={rad}"), **kw)


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
    use_style()
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

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.4, 6.6),
                                 gridspec_kw={"width_ratios": [1.72, 1]})

    # Hues by identity: the baseline is the system (green), d2048 its scaled sibling (teal),
    # V2 actinfo the contrasting condition (purple), P1 lie the failure (crimson).
    #   arm: (label, hue, label offset in points, ha, leader curvature)
    NAMED = {
        "LpWM-ltv":       ("baseline", "green", (-76, 34), "center", -0.18),
        "LpWM-ltv-d2048": ("d2048 (best arm)", "teal", (30, 22), "left", 0.22),
        "PiWM-actinfo":   ("V2 actinfo", "purple", (2, 52), "center", -0.22),
        "PiWM-lie":       ("P1 lie", "crimson", (-16, 44), "center", 0.20),
    }
    for r in R:
        if r["arm"] in NAMED:
            continue
        ax.scatter(r["da"], r["cem"], s=26, color=C["slate"], alpha=0.34, lw=0, zorder=2)
    for arm, (lab, key, off, ha, rad) in NAMED.items():
        v = [r for r in R if r["arm"] == arm]
        if not v:
            continue
        ax.scatter([r["da"] for r in v], [r["cem"] for r in v], s=62, color=C[key],
                   alpha=0.95, edgecolor="white", lw=0.9, zorder=5)
        mx, my = np.median([r["da"] for r in v]), np.median([r["cem"] for r in v])
        _leader(ax, (mx, my), off, C[key], rad=rad)
        ax.annotate(lab, (mx, my), xytext=off, textcoords="offset points", ha=ha,
                    va="center", fontsize=11.5, color=C[key], weight="bold", zorder=7,
                    path_effects=HALO, linespacing=1.25)

    # binned mean of the healthy predictors — the shape, without a curve fit
    h = np.array([(r["da"], r["cem"]) for r in R if r["rel_mse"] < 0.05])
    q = np.quantile(h[:, 0], np.linspace(0, 1, 8))
    sel = [(h[:, 0] >= q[i]) & (h[:, 0] <= q[i + 1]) for i in range(7)]
    mid = [(h[m, 1].mean(), float(np.median(h[m, 0]))) for m in sel if m.sum()]
    ax.plot([m[1] for m in mid], [m[0] for m in mid], color=INK, lw=2.4, zorder=6,
            marker="o", ms=7.5, mfc="white", mew=2.2)
    ax.annotate("septile means,\nhealthy predictors only", (mid[-1][1], mid[-1][0]),
                xytext=(1.02, 0.44), textcoords="data", fontsize=10.5, color=INK,
                ha="left", va="center", linespacing=1.3, path_effects=HALO, zorder=8,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkA=4, shrinkB=9,
                                connectionstyle="arc3,rad=0.22"))

    ax.set_xlabel(r"$d_{action}\,/\,\|z\|$, re-measured on every checkpoint", fontsize=11.5)
    ax.set_ylabel("CEM success rate", fontsize=11.5)
    ax.set_xlim(-0.05, 1.60)
    ax.set_ylim(-0.045, 0.90)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_style(ax, grid="both")
    ax.annotate(f"partial Spearman, removing $\\rho$ and rel_mse\n"
                f"{r_par:+.3f},   p < 0.0001   (n = {len(R)})\n"
                f"rank-quadratic term {quad:+.0f}: an INVERTED U,\nnot a monotone gain",
                (0.020, 0.978), xycoords="axes fraction", va="top", ha="left", fontsize=10.5,
                color=C["slate"], linespacing=1.45, zorder=9,
                bbox=dict(boxstyle="round,pad=0.55", fc=FILL["slate"], ec=C["slate"], lw=0.9))

    # --- right: what round 4 intended to do to d_action, and what it did ---
    probe = {x["run"]: x for x in json.load(open(PROBE))}
    base = float(np.median([x["d_action_over_scale"] for k, x in probe.items()
                            if k.startswith("LpWM-ltv_pd")]))
    arms = [("actgain 3.0", "PiWM-actgain-b30"), ("actgain 0.3", "PiWM-actgain-b03"),
            ("P4 ctrb", "PiWM-ctrb"), ("P2 multact", "PiWM-multact"),
            ("P2 linvar", "LpWM-linvar"), ("P1 lie", "PiWM-lie")]
    ys = np.arange(len(arms))[::-1]
    XB = 0.80
    for y, (lab, arm) in zip(ys, arms):
        v = [x["d_action_over_scale"] for k, x in probe.items() if k.startswith(arm + "_pd")]
        if not v:
            continue
        m = float(np.median(v))
        bx.barh(y, m, height=0.56, color=C["crimson"], alpha=0.9, lw=0, zorder=3)
        # value inside the bar while it is long enough to hold the text, outside otherwise
        inside = m > 0.16 * XB
        bx.annotate(f"{m / base:.2f}x", (m, y), xytext=(-9 if inside else 8, 0),
                    textcoords="offset points", va="center",
                    ha="right" if inside else "left", fontsize=11,
                    color="white" if inside else C["crimson"], weight="bold", zorder=5)
    bx.axvline(base, color=C["green"], lw=2.0, zorder=4)
    bx.annotate(f"baseline  {base:.2f}", (base, len(arms) - 0.62), ha="center", va="bottom",
                fontsize=10.5, color=C["green"], weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.34", fc=FILL["green"], ec=C["green"], lw=0.9))
    bx.set_yticks(ys)
    bx.set_yticklabels([a[0] for a in arms], fontsize=11)
    for t in bx.get_yticklabels():
        t.set_color(INK)
    bx.set_ylim(-0.60, len(arms) - 0.10)
    bx.set_xlim(0, XB)
    bx.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    bx.set_xlabel(r"median $d_{action}\,/\,\|z\|$ per arm", fontsize=11.5)
    ax_style(bx, grid="x")

    top = _header(
        fig,
        "$d_{action}$ re-measured: the baseline was never action-inert, and the relation to "
        "planning is an inverted U",
        [f"LpWM-ltv sits at {base:.2f}, not the 1.9e-04 quoted in train.py — a factor of "
         "2900. That number motivated P1, P3, P4 and P5.",
         "The peak of the curve is at ~0.6, just above the baseline, which is why every arm "
         "that moved $d_{action}$ moved it the wrong way."])
    fig.tight_layout(rect=[0, 0, 1, top], w_pad=2.8)
    _heads(fig, [(ax, "A", "Planning peaks at a middling $d_{action}$",
                  "one point per checkpoint; the four named arms are drawn over the rest"),
                 (bx, "B", "Every round-4 arm LOWERED it",
                  "including both arms that were built to raise it")])
    p = os.path.join(out, "dissociation.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight", pad_inches=0.06)
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
    use_style()
    S = _motion_stats()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.6))

    # --- A: the distribution of per-step block displacement ---
    edges = np.array(S["block_edges"])
    hist = np.array(S["block_hist"])
    HALF = np.log10(0.5)                 # "half a pixel": the threshold behind frac_block_moves
    still = (1 - S["frac_block_moves"]) * 100
    # The zero pile is 30× the tallest tail bin. Clip the axis so the tail is legible at all
    # and label the clipped bar with its own height; on a full-range axis the entire tail --
    # which is what this figure is about — is a flat line one pixel high.
    YMAX, YTOP = 24000.0, 27000.0
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        key = "crimson" if hi <= HALF else "green"
        a1.bar(lo, min(n, YTOP), width=hi - lo, align="edge", color=FILL[key],
               edgecolor=C[key], lw=0.55, zorder=3)
    a1.set_xlim(-3.30, 3.15)
    a1.set_ylim(0, YTOP)
    a1.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    a1.set_xticklabels(["0", "0.01", "0.1", "1", "10", "100", "1000"])
    a1.set_yticks([0, 5000, 10000, 15000, 20000])
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    a1.set_xlabel("block displacement per model step (px, log scale)", fontsize=11.5)
    a1.set_ylabel("transitions (thousands)", fontsize=11.5)
    ax_style(a1, grid="y")

    # the zero bar runs off the top of the axis: cap it with an upward caret and say so
    zb, zw = edges[0], edges[1] - edges[0]
    a1.plot([zb + zw / 2], [YTOP], marker="^", ms=9, color=C["crimson"], zorder=8,
            clip_on=False)
    a1.annotate(f"{hist[0] / 1000:.0f}k transitions here --\nthe bar runs off the axis",
                (zb + zw / 2, YTOP), xytext=(0.115, 0.945), textcoords="axes fraction",
                fontsize=10.5, color=C["crimson"], weight="bold", va="top", ha="left",
                linespacing=1.35, zorder=8,
                arrowprops=dict(arrowstyle="-", color=C["crimson"], lw=0.9, shrinkA=4,
                                shrinkB=8, connectionstyle="arc3,rad=0.25"))
    a1.axvline(HALF, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
    a1.annotate("half a pixel", (HALF, YMAX * 0.86), xytext=(6, 0),
                textcoords="offset points", fontsize=9.5, color=MUTED, ha="left",
                va="center", zorder=8)
    a1.annotate(f"median = 0.000 px\n{still:.0f}% of transitions move\n"
                "the block less than half\na pixel  (crimson bins)",
                (HALF, YMAX * 0.60), xytext=(-11, 0), textcoords="offset points",
                fontsize=10.5, color=C["crimson"], weight="bold", ha="right", va="center",
                linespacing=1.4, zorder=8,
                bbox=dict(boxstyle="round,pad=0.5", fc=FILL["crimson"], ec=C["crimson"],
                          lw=0.9))
    a1.annotate("the tail that the task\nactually lives in", (1.30, 21600),
                xytext=(0, 0), textcoords="offset points", fontsize=10.5, color=C["green"],
                weight="bold", ha="center", va="bottom", linespacing=1.35, zorder=8,
                path_effects=HALO)

    # --- B: how concentrated the motion is when it happens ---
    k = np.arange(1, 1001) / 10.0
    a2.plot([0, 100], [0, 100], color=MUTED, lw=1.1, ls=(0, (1.5, 3)), zorder=2)
    a2.annotate("uniform: every transition\ncarries the same motion", (48, 48),
                xytext=(10, -13), textcoords="offset points", rotation=29,
                rotation_mode="anchor", fontsize=10, color=MUTED, ha="left", va="top",
                linespacing=1.35)
    a2.plot(k, np.array(S["lorenz"]) * 100, color=C["green"], lw=2.6, zorder=4)
    for pct, tx, ty in ((1, 14.0, 22.0), (5, 22.0, 60.0)):
        y = S["lorenz"][int(pct * 10) - 1] * 100
        a2.plot([pct, pct], [0, y], color=C["crimson"], lw=1.0, ls=(0, (3, 3)), zorder=3)
        a2.scatter([pct], [y], s=64, color=C["crimson"], zorder=6, edgecolor="white", lw=1.2)
        a2.annotate(f"top {pct}% of transitions\ncarry {y:.1f}% of all block motion",
                    (pct, y), xytext=(tx, ty), textcoords="data", fontsize=10.5,
                    color=C["crimson"], weight="bold", va="center", ha="left",
                    linespacing=1.35, zorder=7, path_effects=HALO,
                    arrowprops=dict(arrowstyle="-", color=C["crimson"], lw=0.9,
                                    shrinkA=8, shrinkB=6,
                                    connectionstyle="arc3,rad=-0.18"))
    a2.set_xlim(-2, 103)
    a2.set_ylim(-3, 108)
    a2.set_xticks([0, 20, 40, 60, 80, 100])
    a2.set_yticks([0, 20, 40, 60, 80, 100])
    a2.set_xlabel("share of transitions, largest block motion first (%)", fontsize=11.5)
    a2.set_ylabel("cumulative share of block motion (%)", fontsize=11.5)
    ax_style(a2, grid="both")

    top = _header(
        fig,
        "The objective is a mean over transitions; the task lives in the tail",
        [f"{S['n']:,} transitions at the model's own frameskip. The agent moves in "
         f"{S['frac_agent_moves'] * 100:.0f}% of them and its motion is exactly linear in the "
         "action — so a linear predictor can match the baseline",
         "on one-step error (0.0095 vs 0.0092) while planning 4.5× worse. The 1-step MSE is "
         "dominated by the 75% of steps where nothing happens."])
    fig.tight_layout(rect=[0, 0, 1, top], w_pad=3.0)
    _heads(fig, [(a1, "A", "The block does not move",
                  "per-step displacement over the whole training set"),
                 (a2, "B", "...and when it does, it is concentrated",
                  "Lorenz curve of block motion over the same transitions")])
    p = os.path.join(out, "motion-tail.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight", pad_inches=0.06)
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
XLO, XHI = -0.56, 0.20


def fig_p16(campaign, out):
    """Forest plot of every round-4 contrast against its own matched control.

    The numbers live in their own axes to the right of the forest, not in clipped
    annotations hanging off it — which is how the numeric column used to overrun the panel.
    """
    use_style()
    arms = json.load(open(campaign))["arms"]
    rows = []
    for nm, a, c in CONTRASTS:
        A, B = arms.get(a, {}), arms.get(c, {})
        s = sorted(set(A) & set(B), key=int)
        # a 2-seed interval is t_{.975,1} = 12.7 standard errors wide — wider than the
        # whole outcome range. It is reported as a point with its n, not as an interval.
        if len(s) < 3:
            rows.append((nm, c, len(s),
                         float(np.mean([A[k] - B[k] for k in s])) if s else None, None, None))
            continue
        d = np.array([A[k] - B[k] for k in s])
        se = d.std(ddof=1) / np.sqrt(len(d))
        tc = stats.t.ppf(.975, len(d) - 1)
        rows.append((nm, c, len(d), d.mean(), d.mean() - tc * se, d.mean() + tc * se))

    fig = plt.figure(figsize=(12.6, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    tx = fig.add_subplot(gs[0, 1], sharey=ax)
    ys = np.arange(len(rows))[::-1]

    for y, (nm, c, n, m, lo, hi) in zip(ys, rows):
        if m is None:
            ax.annotate("trained, no paired eval yet", (XLO + 0.02, y), fontsize=10,
                        va="center", color=MUTED, style="italic")
            tx.text(0.44, y, "--", fontsize=11, va="center", ha="right", color=MUTED,
                    family="monospace")
            continue
        # by identity of the verdict, not by rank: crimson = worse than its control,
        # green = better, slate = the interval straddles zero
        key = "crimson" if (hi is not None and hi < 0) else (
            "green" if (lo is not None and lo > 0) else "slate")
        col = C[key]
        if lo is not None:
            l, h = max(lo, XLO), min(hi, XHI)
            ax.plot([l, h], [y, y], color=col, lw=2.6, solid_capstyle="round", zorder=4)
            for edge, val, dd in ((l, lo, -1), (h, hi, +1)):    # arrow cap where clipped
                if abs(val - edge) > 1e-9:                       # head lands ON the edge
                    ax.annotate("", (edge, y), xytext=(edge - dd * 0.035, y), zorder=6,
                                arrowprops=dict(arrowstyle="-|>,head_width=0.30",
                                                color=col, lw=2.6, shrinkA=0, shrinkB=0))
        ax.scatter([m], [y], s=118, color=col, zorder=5, edgecolor="white", lw=1.2)
        tx.text(0.44, y, f"{m:+.3f}", fontsize=11, va="center", ha="right", color=col,
                weight="bold", family="monospace")
        tx.text(1.00, y, "n too small" if lo is None else f"[{lo:+.2f}, {hi:+.2f}]",
                fontsize=10.5, va="center", ha="right", color=col, family="monospace")

    ax.axvline(0, color=INK, lw=1.2, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{nm}   (n={n})\nvs {c}" for nm, c, n, *_ in rows], fontsize=10.5)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.set_ylim(-0.66, len(rows) - 0.22)
    ax.set_xlim(XLO, XHI)
    ax.set_xticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_xlabel("paired difference in CEM success rate vs the arm's own control\n"
                  "(95% CI; an arrow cap means the interval runs past the axis)",
                  fontsize=11, linespacing=1.45)
    ax_style(ax, grid="x")
    ax.tick_params(axis="y", length=0)          # the rows are categories, not a scale
    ax.annotate("worse than its control  ←", (-0.012, len(rows) - 0.42), ha="right",
                va="center", fontsize=10, color=MUTED, style="italic", zorder=6)
    ax.annotate("→  better", (0.012, len(rows) - 0.42), ha="left", va="center",
                fontsize=10, color=MUTED, style="italic", zorder=6)

    tx.set_xlim(0, 1.06)
    tx.axis("off")
    tx.text(0.44, len(rows) - 0.42, "difference", fontsize=10, ha="right", va="center",
            color=MUTED, weight="bold")
    tx.text(1.00, len(rows) - 0.42, "95% CI", fontsize=10, ha="right", va="center",
            color=MUTED, weight="bold")
    tx.plot([0.0, 1.04], [len(rows) - 0.66] * 2, color=EDGE, lw=0.9, clip_on=False)

    # One data panel, so the figure header is the panel header — no second title, and no
    # tight_layout: an axes with axis("off") is not tight_layout-compatible and the call was
    # silently dropping the reserved band, which is what pushed the two titles into each other.
    top = _header(
        fig,
        "Round 4: not one contrast is positive",
        ["The two arms that remove the predictor's nonlinearity — P2 linvar and P1 lie — are "
         "the round's largest effects, and both are negative.",
         "Every remaining interval straddles zero. Each arm is compared with its own matched "
         "control, seed by seed."],
        head_room=0.10)
    fig.subplots_adjust(left=0.137, right=0.995, bottom=0.165, top=top, wspace=0.03)
    p = os.path.join(out, "p16-results.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 4 (M1)
PROBE_M1 = "assets/latent_probe.json"
# Arms whose predictor has no nonlinear readout. This is the factor round 4 varied, and it
# turns out to sort the LATENT PROBE as strongly as it sorts planning — in opposite directions.
LINEAR_ARMS = {"LpWM-linvar", "PiWM-lie", "PiWM-lie-sim", "PiWM-multact", "PiWM-ctrb",
               "LpWM-ltv-ident-p1"}


def fig_probe(out, campaign="/tmp/p16.json"):
    """M1: the latent encodes the block far better than any control — and the arms that
    encode it BEST are the ones that plan WORST."""
    use_style()
    if not os.path.exists(PROBE_M1):
        return None
    H = "_heldout"
    R = json.load(open(PROBE_M1))
    cem = json.load(open(campaign))["arms"]
    alias = {"PiWM-columns": "PiWM-columns_patch"}
    for x in R:
        x["cem"] = cem.get(alias.get(x["arm"], x["arm"]), {}).get(str(x["seed"]))
    ev = [x for x in R if x["cem"] is not None and x.get("err_pos_px" + H) is not None]
    if len(ev) < 40:
        return None

    # plt.subplots, not add_gridspec(wspace=...): a gridspec with an explicit wspace makes its
    # axes tight_layout-incompatible, and tight_layout then silently drops the reserved header
    # band, which is how the panel titles ended up on top of the figure title.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.6, 6.8),
                                 gridspec_kw={"width_ratios": [1.52, 1]})

    XLIM = (96.0, 7.7)                      # inverted log: a BETTER latent is to the RIGHT
    YLIM = (-0.115, 0.91)                   # the band below 0 is a gutter for direct labels
    BAND = 17.0                             # the six arms whose median latent is sharpest

    # the band first, so everything else sits on top of it
    ax.axvspan(BAND, XLIM[1], color=FILL["slate"], alpha=0.6, lw=0, zorder=0)
    ax.axvline(BAND, color=EDGE, lw=1.0, zorder=1)

    # --- left: probe quality against planning, split by predictor class ---
    #   faint marks   = one checkpoint
    #   stars         = one arm's median, which is what the claim is about
    for lin, key, mk in ((True, "crimson", "s"), (False, "green", "o")):
        S = [x for x in ev if (x["arm"] in LINEAR_ARMS) == lin]
        ax.scatter([x["err_pos_px" + H] for x in S], [x["cem"] for x in S], s=34, marker=mk,
                   color="white" if lin else FILL[key], edgecolor=C[key], lw=1.1, alpha=0.9,
                   zorder=3)

    by = collections.defaultdict(list)
    for x in ev:
        by[x["arm"]].append(x)
    med = {a_: (float(np.median([x["err_pos_px" + H] for x in v])),
                float(np.median([x["cem"] for x in v])))
           for a_, v in by.items() if len(v) >= 5}
    for a_, (mx, my) in med.items():
        key = "crimson" if a_ in LINEAR_ARMS else "green"
        ax.scatter([mx], [my], s=175, marker="*", color=C[key], zorder=6,
                   edgecolor="white", lw=1.1)

    # (label offset in points, ha, leader curvature). The two gutter labels sit below y = 0,
    # where no data can ever be.
    NAMED = {
        "LpWM-ltv-d2048": ((-6, 40), "center", -0.45),
        "LpWM-ltv":       ((-92, 16), "right", 0.18),
        "PiWM-columns":   ((36, 30), "left", -0.20),
        "LpWM-linvar":    ((2, 40), "center", -0.22),
        "PiWM-ctrb":      ((-10, -34), "center", 0.22),
        "PiWM-lie":       ((10, -34), "center", -0.22),
    }
    for arm, (off, ha, rad) in NAMED.items():
        if arm not in med:
            continue
        mx, my = med[arm]
        key = "crimson" if arm in LINEAR_ARMS else "green"
        _leader(ax, (mx, my), off, C[key], rad=rad, zorder=5)
        ax.annotate(LABEL.get(arm, arm).replace(" (control)", ""), (mx, my), xytext=off,
                    textcoords="offset points", ha=ha, va="center", fontsize=11,
                    color=C[key], weight="bold", zorder=9, path_effects=HALO)

    # THE inversion, drawn as one move: the baseline's latent sharpens 2x and its planning
    # collapses. Halo-outlined so it reads as an annotation layer over the marks.
    if "LpWM-ltv" in med and "LpWM-linvar" in med:
        ax.annotate("", xy=med["LpWM-linvar"], xytext=med["LpWM-ltv"], zorder=7,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.34,head_length=0.7",
                                    color=C["crimson"], lw=2.0, shrinkA=11, shrinkB=8,
                                    connectionstyle="arc3,rad=0.30",
                                    path_effects=[withStroke(linewidth=4.6,
                                                             foreground="white")]))
        ax.annotate("better latent,\nworse planner", (12.4, 0.325), fontsize=11.5,
                    color=C["crimson"], weight="bold", ha="center", va="center",
                    linespacing=1.3, zorder=9, path_effects=HALO)

    ax.set_xscale("log")                    # 9 px to 90 px; linear squashes the live arms
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([10, 15, 20, 30, 50, 90])
    ax.set_xticklabels(["10", "15", "20", "30", "50", "90"])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel("held-out block position error from the frozen latent  (px, ridge probe; "
                  "a better latent is to the right)", fontsize=11.5)
    ax.set_ylabel("CEM success rate", fontsize=11.5)
    ax_style(ax, grid="both")
    n_band = sum(1 for v in med.values() if v[0] <= BAND)
    ax.annotate(f"the {n_band} arms with the sharpest latents", (BAND, 0.872), xytext=(7, 0),
                textcoords="offset points", fontsize=10, color=MUTED, ha="left", va="center")

    n_lin = sum(1 for x in ev if x["arm"] in LINEAR_ARMS)
    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=FILL["green"], mec=C["green"],
                      mew=1.2, label=f"nonlinear predictor,  n = {len(ev) - n_lin}"),
               Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec=C["crimson"],
                      mew=1.4, label=f"linear predictor,  n = {n_lin}"),
               Line2D([], [], ls="", marker="*", ms=13, mfc=MUTED, mec="white",
                      label=f"per-arm median,  {len(med)} arms")]
    # the marker SHAPES need a key that a direct label cannot carry; parked in the empty
    # upper-left, which is the one region of this panel with no marks in it
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5,
                    handletextpad=0.5, labelspacing=0.6, borderaxespad=0.7)
    for t, col in zip(leg.get_texts(), (C["green"], C["crimson"], MUTED)):
        t.set_color(col)

    # --- right: the control ladder ---
    # hues by identity: the green family is the model's own latent, the purple family the
    # information controls, slate and crimson the nulls.
    LAD = [("z, pre-link", "prelink_err", "green"),
           ("z, post-link (fed to\nthe predictor)", "err", "teal"),
           ("agent pose alone", "ctrl_agent_err", "purple"),
           ("random encoder", "ctrl_rand_err", "plum"),
           ("constant predictor", "const_err", "slate"),
           ("shuffled labels", "ctrl_shuf_err", "crimson")]
    # the ladder is a statement about a WORKING latent, so it is taken over the arms that
    # both predict and plan — not over the collapsed arms, whose probe is at chance.
    HEALTHY = {"LpWM-ltv", "LpWM-ltv-d2048", "PiWM-columns", "LpWM-ltv-relu-p2",
               "LpWM-ltv-vfloor", "PiWM-gate-both"}
    good = [x for x in R if x["arm"] in HEALTHY]
    ys = np.arange(len(LAD))[::-1]
    # fixed right-aligned numeric columns past the end of the longest bar, so no label can
    # ever be pushed off the panel; the axis itself stops at 80 px.
    XB, PX_COL, DEG_COL = 150.0, 118.0, 147.0
    for y, (lab, pre, key) in zip(ys, LAD):
        v = [x[pre + "_pos_px" + H] for x in good if x.get(pre + "_pos_px" + H) is not None]
        if not v:
            continue
        m = float(np.median(v))
        bx.barh(y, m, height=0.56, color=C[key], alpha=0.9, lw=0, zorder=3)
        a_ = [x[pre + "_ang_deg" + H] for x in good if x.get(pre + "_ang_deg" + H) is not None]
        bx.text(PX_COL, y, f"{m:.1f} px", fontsize=10.5, va="center", ha="right",
                color=C[key], weight="bold")
        bx.text(DEG_COL, y, f"{np.median(a_):.1f}°", fontsize=10.5, va="center",
                ha="right", color=C[key], weight="bold")
    bx.set_yticks(ys)
    bx.set_yticklabels([l for l, _, _ in LAD], fontsize=10.5)
    for t in bx.get_yticklabels():
        t.set_color(INK)
    bx.set_ylim(-0.62, len(LAD) - 0.18)
    bx.set_xlim(0, XB)
    bx.set_xticks([0, 20, 40, 60, 80])
    bx.set_xlabel("held-out block position error (px)", fontsize=11.5)
    ax_style(bx, grid="none")
    bx.xaxis.grid(True, color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    bx.tick_params(axis="y", length=0)
    bx.spines["bottom"].set_bounds(0, 80)   # the scale stops where the bars do
    bx.axhline(len(LAD) - 2.5, color=EDGE, lw=1.0, zorder=1)   # model's latent | controls
    bx.text(PX_COL, len(LAD) - 0.40, "position", fontsize=9.5, ha="right", va="center",
            color=MUTED, weight="bold")
    bx.text(DEG_COL, len(LAD) - 0.40, "angle", fontsize=9.5, ha="right", va="center",
            color=MUTED, weight="bold")

    top = _header(
        fig,
        "M1: the latent has the block's POSITION, and the arms that have it best are the ones "
        "that plan worst",
        ["LpWM-linvar localises the block to 7.8 px and its orientation to 3.5° on the "
         "validation split — 2x and 4.5× the baseline — and plans at 0.09 vs 0.38.",
         "The baseline's ANGLE is not decoded at all: 15.7° against a 14.5° "
         "constant-prediction floor, with the task tolerance at 20°."])
    fig.tight_layout(rect=[0, 0, 1, top], w_pad=3.4)
    _heads(fig, [(ax, "A", "The best representations belong to the worst planners",
                  "one faint mark per checkpoint; one star per arm"),
                 (bx, "B", "Healthy arms only: every control is beaten",
                  "the model's own latent above the rule, the controls below; the link "
                  "costs 4 px and 4 degrees")])
    p = os.path.join(out, "m1-probe.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--campaign", default="/tmp/p16.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in (fig_dissociation(None, a.out, a.campaign), fig_motion_tail(a.out),
              fig_p16(a.campaign, a.out), fig_probe(a.out, a.campaign)):
        if p:
            print("  wrote", p)


if __name__ == "__main__":
    main()
