"""Round-5 result figures: the whole campaign as one forest plot, and the M3 oracle ladder.

Both figures are drawn from the frozen round-5 contrast table in diary/2026-09-03.md section 16
(the paired, matched-seed numbers), not re-derived from the campaign JSON — those intervals are
the authoritative ones and several of the arms they cover are still finishing.

DESIGN. Everything visual comes from ``analysis/style.py``, the one design system for this
project (derived from ``figures/motivation_teaser.svg``): sans-serif throughout, hues assigned by
IDENTITY and never by rank, recessive furniture, pale callouts with saturated text, direct
labels, titles that state the finding, no dual axes.

    green    the thing that works        the ensemble; oracle dynamics
    crimson  failure / alert             a contrast whose interval excludes zero, negative
    slate    neutral                     a null, and every piece of furniture
    amber    the intervention under test (unused here: nothing in round 5 earned it)

    python analysis/round5_result_figs.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory as blend
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style as S  # noqa: E402
from analysis.style import C, EDGE, FILL, GRID, HEAD, INK, MUTED  # noqa: E402

S.use_style()

DIM = "#AEB8B4"          # an arm with no paired evaluation yet — present, not scored
BAND = "#F1F3F2"         # the pale band behind a group header row


# ------------------------------------------------------------------ figure 1: the forest
# (label, control, n, d, lo, hi). Groups are ordered by their most negative member, so the
# round reads top-to-bottom from "the one thing that worked" down to "the model collapse".
GROUPS = [
    ("V1", "consensus rule over 5 independent rollouts", [
        ("borda", "single-model baseline", 8, +0.215, +0.089, +0.341),
        ("median", "borda", 8, +0.000, -0.027, +0.027),
        ("cvar1", "borda", 8, -0.022, -0.055, +0.010),
        ("cvar2", "borda", 8, -0.022, -0.081, +0.036),
        ("max", "borda", 8, -0.040, -0.088, +0.008),
    ]),
    ("T1", "attach the decoder, gradient into the encoder", [
        ("detach", "baseline", 8, +0.055, -0.085, +0.195),
        ("decode-w1", "detach (own control)", 8, -0.048, -0.148, +0.053),
        ("decode", "detach (own control)", 8, -0.060, -0.205, +0.085),
    ]),
    ("V3", "two-level contact planner", [
        ("2lvl", "baseline", 8, -0.067, -0.158, +0.023),
    ]),
    ("V2", "gradient planner instead of CEM", [
        ("gd-noise0", "baseline", 8, -0.070, -0.118, -0.022),
        ("gd", "baseline", 8, -0.077, -0.123, -0.032),
    ]),
    ("T7", "frozen-encoder ranking head as the CEM leaf", [
        ("energy", "energy-distill (own control)", 8, -0.015, -0.085, +0.055),
        ("energy", "baseline", 8, -0.143, -0.246, -0.039),
    ]),
    ("T3", "weight transitions by unexplained visual change", [
        ("contact", "contact-shuf (own control)", 8, -0.022, -0.140, +0.095),
        ("contact-g05", "contact-shuf (own control)", 6, -0.027, -0.201, +0.147),
        ("contact-shuf", "baseline", 8, -0.345, -0.476, -0.214),
    ]),
]
TRAINING = [("T2", "patchdecode", 0), ("T6", "jump5", 1), ("T4", "vp", 0)]
XLO, XHI = -0.52, 0.38

# column geometry, in axes fractions. Everything outside [0, 1] is drawn with clipping off:
# the label gutter on the left, the numeric table on the right.
X_GROUP = -0.372        # group header text, flush left
X_NAME = -0.268         # arm name, right-aligned into the gutter
X_CTRL = -0.258         # "vs <control>", left-aligned out of the gutter
X_EFF = 1.115           # effect size, right-aligned
X_CI = 1.145            # the interval, left-aligned
X_N = 1.330             # n, right-aligned
X_L, X_R = X_GROUP, X_N + 0.015     # the extent of a group-header band


def _colour(lo, hi):
    """Hue by identity of the READING, not by rank: positive/negative/null."""
    if lo > 0:
        return C["green"]
    if hi < 0:
        return C["crimson"]
    return C["slate"]


def fig_forest(out):
    """Every round-5 contrast, paired and matched-seed, on one axis."""
    # --- lay the rows out: a header per proposal, then its contrasts ---
    rows = []                                    # (kind, payload)
    for tag, blurb, items in GROUPS:
        rows.append(("head", (tag, blurb)))
        for it in items:
            rows.append(("row", it))
    rows.append(("head", ("", "still training — no paired evaluation exists yet")))
    for tag, nm, n in TRAINING:
        rows.append(("train", (tag, nm, n)))
    N = len(rows)

    fig, ax = plt.subplots(figsize=(14.6, 10.4))
    L, R, TOP, BOT = 0.235, 0.800, 0.895, 0.072
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT)
    bx = blend(ax.transAxes, ax.transData)       # x in axes fractions, y in row units
    ys = np.arange(N)[::-1]

    ax.plot([0, 0], [-0.8, N - 0.55], color=INK, lw=1.2, zorder=2)
    for y, (kind, payload) in zip(ys, rows):
        if kind == "head":
            tag, blurb = payload
            ax.add_patch(Rectangle((X_L, y - 0.5), X_R - X_L, 1.0, transform=bx,
                                   fc=BAND, ec="none", zorder=2.5, clip_on=False))
            x = X_GROUP + 0.010
            if tag:
                ax.text(x, y, tag, transform=bx, fontsize=11.5, fontweight="bold",
                        color=HEAD, va="center", ha="left", zorder=4, clip_on=False)
                x += 0.042
            ax.text(x, y, blurb, transform=bx, fontsize=11.5, color=INK if tag else MUTED,
                    va="center", ha="left", zorder=4, clip_on=False)
            continue

        if kind == "train":
            tag, nm, n = payload
            ax.text(X_NAME, y, f"{tag}  {nm}", transform=bx, fontsize=11, color=MUTED,
                    va="center", ha="right", zorder=4, clip_on=False)
            ax.plot([XLO + 0.02, XHI - 0.02], [y, y], color=DIM, lw=1.0, ls=(0, (2, 3)),
                    zorder=3)
            ax.text(0.5 * (XLO + XHI), y, f"still training  ·  n = {n}", fontsize=10.5,
                    va="center", ha="center", color=MUTED, style="italic", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="none"))
            ax.text(X_EFF, y, "—", transform=bx, fontsize=11, color=DIM, va="center",
                    ha="right", zorder=4, clip_on=False)
            ax.text(X_CI, y, "—", transform=bx, fontsize=11, color=DIM, va="center",
                    ha="left", zorder=4, clip_on=False)
            ax.text(X_N, y, f"{n}", transform=bx, fontsize=11, color=DIM, va="center",
                    ha="right", zorder=4, clip_on=False)
            continue

        nm, ctrl, n, d, lo, hi = payload
        col = _colour(lo, hi)
        hero = d > 0.1                                   # V1 borda: the round's one positive
        if hero:
            ax.add_patch(Rectangle((X_L, y - 0.5), X_R - X_L, 1.0, transform=bx,
                                   fc=FILL["green"], ec="none", zorder=2.4, clip_on=False))
        ax.text(X_NAME, y, nm, transform=bx, fontsize=11, color=INK, va="center",
                ha="right", zorder=4, clip_on=False,
                fontweight="bold" if hero else "normal")
        ax.text(X_CTRL, y, f"vs {ctrl}", transform=bx, fontsize=11, color=MUTED,
                va="center", ha="left", zorder=4, clip_on=False)

        lw, ms = (4.6, 300) if hero else (2.4, 118)
        ax.plot([lo, hi], [y, y], color=col, lw=lw, solid_capstyle="butt", zorder=4)
        for e in (lo, hi):                               # interval end caps
            ax.plot([e, e], [y - 0.20, y + 0.20], color=col, lw=lw * 0.75,
                    solid_capstyle="butt", zorder=4)
        ax.scatter([d], [y], s=ms, color=col, zorder=5, edgecolor="white",
                   lw=1.5 if hero else 1.1, marker="D" if hero else "o")

        w = "bold" if hero else "normal"
        ax.text(X_EFF, y, f"{d:+.3f}", transform=bx, fontsize=11, color=col, va="center",
                ha="right", zorder=4, clip_on=False, fontweight=w)
        ax.text(X_CI, y, f"[{lo:+.3f}, {hi:+.3f}]", transform=bx, fontsize=11, color=col,
                va="center", ha="left", zorder=4, clip_on=False, fontweight=w)
        ax.text(X_N, y, f"{n}", transform=bx, fontsize=11, color=MUTED, va="center",
                ha="right", zorder=4, clip_on=False, fontweight=w)

    # --- the one positive, labelled directly inside its own band ---
    # The hero interval starts at +0.089, so the whole left half of that row is empty: the
    # label goes there, on the row itself. No leader, no box, nothing to collide with.
    y_hero = ys[[i for i, (k, p) in enumerate(rows) if k == "row" and p[3] > 0.1][0]]
    ax.text(XLO + 0.015, y_hero, "THE ONLY POSITIVE — five rollouts averaged, "
                                 "a result already known",
            fontsize=11, fontweight="bold", color=C["green"], va="center", ha="left",
            zorder=6)

    # --- the table header, on its own strip above the first group ---
    y_hdr = N - 0.30
    ax.plot([X_L, X_R], [N - 0.52, N - 0.52], transform=bx, color=EDGE, lw=1.0,
            clip_on=False, zorder=3)
    for x, t, ha in ((X_GROUP + 0.010, "contrast", "left"),
                     (X_EFF, "effect", "right"), (X_CI, "95% CI", "left"),
                     (X_N, "n", "right")):
        ax.text(x, y_hdr, t, transform=bx, fontsize=10, color=MUTED, va="center", ha=ha,
                clip_on=False, zorder=4)

    ax.set_yticks([])
    ax.set_ylim(-0.8, N + 0.7)
    ax.set_xlim(XLO, XHI)
    ax.set_xticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("paired difference in CEM success rate against the arm's own control "
                  "(95% CI, matched seeds)", fontsize=11.5, color=INK, labelpad=8)
    S.ax_style(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    handles = [Line2D([], [], color=C["green"], lw=3.2, marker="D", ms=8.5, mec="white",
                      label="interval excludes zero, positive"),
               Line2D([], [], color=C["crimson"], lw=2.4, marker="o", ms=7.5, mec="white",
                      label="interval excludes zero, negative"),
               Line2D([], [], color=C["slate"], lw=2.4, marker="o", ms=7.5, mec="white",
                      label="interval spans zero (a null)"),
               Line2D([], [], color=DIM, lw=1.4, ls=(0, (2, 3)),
                      label="still training, no paired eval")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(X_GROUP, 1.012),
              frameon=False, fontsize=10.5, ncol=4, handlelength=2.6, handletextpad=0.6,
              columnspacing=2.4, borderpad=0.0)

    xt = L + X_GROUP * (R - L)
    fig.text(xt, 0.988, "Round 5: sixteen proposals, one positive — and it is the ensemble "
                        "that was already known", fontsize=13, fontweight="bold", color=INK,
             ha="left", va="bottom")
    fig.text(xt, 0.980,
             "Every paired contrast in the campaign, matched seeds and a 95% interval. Four "
             "of the five intervals that exclude zero are negative;\nT3's −0.345 is its own "
             "control arm, so both T3 arms sit at the floor.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.45)

    p = os.path.join(out, "round5-forest.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 2: M3's 2x2
# rows are dynamics (oracle on top so "up" is "better"), columns are objective.
SEEDS = {                       # (dynamics row, objective col) -> (s3, s4, s5), mean, bound?
    ("oracle", "learned"): ((0.94, 0.92, 0.88), 0.91, True),
    ("oracle", "oracle"): ((0.88, 0.82, 0.92), 0.87, True),
    ("learned", "learned"): ((0.24, 0.38, 0.66), 0.427, False),
    ("learned", "oracle"): ((0.20, 0.10, 0.32), 0.207, False),
}
# One hue, light -> dark, for a magnitude. The design system's green ramp.
SEQ = LinearSegmentedColormap.from_list("m3", S.SEQ_GREEN)
# The whole panel is laid out in INCHES (one data unit = one inch, and the axes fills the
# figure) so nothing — title included — can drift into anything else: every element below
# is positioned against the same ruler.
FW, FH = 13.8, 8.9
CW, CH = 3.55, 2.05                                 # cell width, cell height
CX = {"learned": 4.85, "oracle": 8.68}              # column centres  (objective)
CY = {"learned": 3.625, "oracle": 5.975}            # row centres     (dynamics)


def fig_oracle_ladder(out):
    """M3: swapping the objective does nothing; swapping the dynamics is the whole gap."""
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.set_aspect("equal")
    ax.set_axis_off()
    norm = Normalize(0.0, 1.0)

    for (rdyn, cobj), (seeds, mean, lb) in SEEDS.items():
        x, y = CX[cobj], CY[rdyn]
        ax.add_patch(Rectangle((x - CW / 2, y - CH / 2), CW, CH,
                               facecolor=SEQ(norm(mean)), edgecolor="white", lw=2.0,
                               zorder=2))
        fg = "white" if mean > 0.50 else INK
        sub = "#CFE3D8" if fg == "white" else MUTED
        ax.text(x, y + 0.40, ("\u2265 " if lb else "") + (f"{mean:.2f}" if lb else f"{mean:.3f}"),
                ha="center", va="center", fontsize=38, fontweight="bold", color=fg, zorder=4)
        ax.text(x, y - 0.06, "mean of the three seeds", ha="center", va="center",
                fontsize=10.5, color=sub, zorder=4)
        ge = "\u2265" if lb else ""
        ax.text(x, y - 0.44, "   ".join(f"s{s} {ge}{v:.2f}"
                                        for s, v in zip((3, 4, 5), seeds)),
                ha="center", va="center", fontsize=12, color=fg, zorder=4)
        ax.text(x, y - 0.80, "lower bound \u2014 latched, hit walltime" if lb
                else "run to completion",
                ha="center", va="center", fontsize=10, color=sub,
                fontweight="bold" if lb else "normal", zorder=4)

    # today's system, tagged inside its own cell so nothing sits between the cells
    x, y = CX["learned"], CY["learned"]
    ax.add_patch(Rectangle((x - CW / 2, y - CH / 2), CW, CH, facecolor="none",
                           edgecolor=C["crimson"], lw=2.4, zorder=5))
    ax.text(x - CW / 2 + 0.16, y + CH / 2 - 0.21, "TODAY'S SYSTEM", ha="left", va="center",
            fontsize=10.5, color=C["crimson"], fontweight="bold", zorder=6)

    # --- headers ---
    ytop = CY["oracle"] + CH / 2
    for cobj, sub in (("learned", "terminal latent MSE"),
                      ("oracle", "true task distance")):
        ax.text(CX[cobj], ytop + 0.56, f"objective:  {cobj.upper()}", ha="center",
                va="baseline", fontsize=13.5, fontweight="bold", color=INK)
        ax.text(CX[cobj], ytop + 0.22, sub, ha="center", va="baseline", fontsize=11,
                color=MUTED)
    xlab = CX["learned"] - CW / 2 - 0.34
    for rdyn, sub in (("oracle", "the true simulator, replayed"),
                      ("learned", "the model's own rollout")):
        ax.text(xlab, CY[rdyn] + 0.10, f"dynamics:  {rdyn.upper()}", ha="right",
                va="baseline", fontsize=13.5, fontweight="bold", color=INK)
        ax.text(xlab, CY[rdyn] - 0.30, sub, ha="right", va="baseline", fontsize=11,
                color=MUTED)

    # --- the two arrows. The vertical one is the finding; the horizontal one is the null.
    # The weight difference between them is the whole point of the figure, so it is large:
    # 9pt of saturated green against 1.6pt of muted grey.
    xa = CX["oracle"] + CW / 2 + 0.52
    ax.add_patch(FancyArrowPatch((xa, CY["learned"] - 0.55), (xa, CY["oracle"] + 0.62),
                                 arrowstyle="-|>,head_width=13,head_length=21",
                                 color=C["green"], lw=9.0, shrinkA=0, shrinkB=0, zorder=4))
    xt = xa + 0.48
    ax.text(xt, CY["oracle"] + 0.72, "swap the DYNAMICS", ha="left", va="baseline",
            fontsize=15, color=C["green"], fontweight="bold")
    ax.text(xt, CY["oracle"] + 0.30, "same latent-MSE objective,\nsame planner,\n"
                                     "same 300 \u00d7 30 budget",
            ha="left", va="top", fontsize=10.5, color=MUTED, linespacing=1.5)
    ymid = (CY["learned"] + CY["oracle"]) / 2
    ax.text(xt, ymid + 0.30, "+0.48", ha="left", va="center", fontsize=44,
            color=C["green"], fontweight="bold")
    ax.text(xt, ymid - 0.28, "0.427  \u2192  \u2265 0.91", ha="left", va="center",
            fontsize=13, color=C["green"])
    ax.text(xt, CY["learned"] - 0.30, "the whole gap between\nthe system and the task",
            ha="left", va="top", fontsize=10.5, color=MUTED, linespacing=1.5)

    ya = CY["learned"] - CH / 2 - 0.45
    ax.add_patch(FancyArrowPatch((CX["learned"], ya), (CX["oracle"], ya),
                                 arrowstyle="-|>,head_width=3.2,head_length=6.5",
                                 color=MUTED, lw=1.6, shrinkA=0, shrinkB=0, zorder=4))
    xm = (CX["learned"] + CX["oracle"]) / 2
    ax.text(xm, ya - 0.38, "swap the OBJECTIVE", ha="center", va="center", fontsize=12,
            color=MUTED, fontweight="bold")
    ax.text(xm - 0.12, ya - 0.78, "\u22120.22", ha="right", va="center", fontsize=19,
            color=MUTED, fontweight="bold")
    ax.text(xm + 0.12, ya - 0.78, "nothing \u2014 if anything, worse", ha="left",
            va="center", fontsize=11.5, color=MUTED, style="italic")

    # --- the control fact, along the bottom ---
    ax.add_patch(FancyBboxPatch((0.34, 0.18), 9.50, 0.92, boxstyle="round,pad=0.04",
                                facecolor=FILL["slate"], edgecolor=EDGE, lw=1.0, zorder=2))
    ax.text(0.62, 0.64,
            "CONTROL: the (learned, learned) cell reproduces the archive EXACTLY \u2014 "
            "0.24 / 0.38 / 0.66 are LpWM-ltv s3 / s4 / s5 on record.\n"
            "Same harness, same 50 episodes, same 300 samples \u00d7 30 iterations \u00d7 "
            "H = 5. It is the same experiment, with one part swapped at a time.",
            ha="left", va="center", fontsize=11, color=INK, zorder=4, linespacing=1.6)

    # --- the ramp legend, drawn in inches so it cannot land on anything ---
    cx0, cy0, cw, ch = 10.55, 0.62, 2.70, 0.17
    for i in range(120):
        ax.add_patch(Rectangle((cx0 + cw * i / 120, cy0), cw / 120 + 0.004, ch,
                               facecolor=SEQ(i / 119), edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((cx0, cy0), cw, ch, facecolor="none", edgecolor=EDGE, lw=0.8,
                           zorder=4))
    for v in (0.0, 0.5, 1.0):
        ax.text(cx0 + cw * v, cy0 - 0.10, f"{v:.1f}", ha="center", va="top", fontsize=9.5,
                color=MUTED)
    ax.text(cx0, cy0 + ch + 0.12, "CEM success rate, 50 episodes", ha="left", va="baseline",
            fontsize=10, color=MUTED)

    # --- the header, on the same ruler as everything else ---
    ax.text(0.34, FH - 0.34,
            "M3: the rollout is the bottleneck \u2014 swapping the DYNAMICS is worth +0.48, "
            "swapping the OBJECTIVE is worth nothing",
            fontsize=13, fontweight="bold", color=INK, ha="left", va="baseline")
    ax.text(0.34, FH - 0.60,
            "LpWM-ltv s3/s4/s5, 50 episodes per cell, the planner's real budget. Feed the "
            "same latent-MSE score a true terminal frame and planning goes 0.427 \u2192 "
            "\u2265 0.91;\nreplace that score with the true task distance and it goes 0.427 "
            "\u2192 0.207. The oracle-dynamics cells are latched lower bounds, so the "
            "vertical gap is understated.",
            fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.5)

    p = os.path.join(out, "oracle-ladder.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in (fig_forest(a.out), fig_oracle_ladder(a.out)):
        print("  wrote", p)


if __name__ == "__main__":
    main()
