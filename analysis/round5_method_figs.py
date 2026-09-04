"""Round-5 METHOD figures: the two exhibits that are about how the campaign was read.

Both figures are about inference, not about a model. Neither harvests anything: every
number in them is a settled round-5 result quoted from diary/2026-09-03.md sections 16.13
to 16.15 (and section 12 for the round-4 `actgain` correction), so they are literals here
rather than a re-derivation that could drift from the write-up.

    n3-lesson.png     the T1 decode-vs-detach contrast as seeds landed: an interval that
                      excluded zero at n = 3 and was a null at n = 8, plus the campaign's
                      two other instances of the same error.
    t3-reweighting.png  why T3 failed twice, and how the shuffled control — identical ESS,
                      random transitions — separates the two causes.

DESIGN. Everything visual comes from ``analysis/style.py``, the one design system for this
project (derived from ``figures/motivation_teaser.svg``): sans-serif throughout, hues by
IDENTITY and never by rank, recessive furniture, panel headers with a coloured letter and a
rule, pale callouts with saturated text, direct labels instead of a legend, no dual axes.

    slate    LpWM-ltv, the baseline, and every neutral reading
    crimson  the reading that was wrong; the arm whose training collapsed
    purple   the contrasting condition — the shuffled control
    green    the prescription that comes out of the exhibit

Usage:  python analysis/round5_method_figs.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style as S  # noqa: E402
from analysis.style import C, EDGE, FILL, GRID, HEAD, INK, MUTED  # noqa: E402

S.use_style()

# Hues by identity. LpWM-ltv keeps the slate it has in every other figure of this round;
# the collapsed arm is crimson wherever it appears and its control is the purple
# "contrasting condition", so neither can ever be mistaken for a result that worked.
ARM = {"LpWM-ltv": C["slate"], "PiWM-contact": C["crimson"],
       "PiWM-contact-shuf": C["purple"]}
# Roles in figure 1, again by identity rather than by rank: the reading that was wrong is
# crimson wherever it appears, the settled reading is ink, everything else is slate.
ROLE = {"wrong": C["crimson"], "final": INK, "interim": C["slate"]}

MINUS = "−"          # a real minus sign; the hyphen is a different, shorter glyph


def _num(v, fmt="{:+.3f}"):
    return fmt.format(v).replace("-", MINUS)


def _panel(ax, letter, title, sub=None, y=1.105, x0=0.0, x1=1.0):
    """The reference's panel header: coloured letter, bold title, muted sub, and a rule."""
    S.panel_title(ax, letter, title, sub, y=y)
    ax.plot([x0, x1], [y - 0.075, y - 0.075], transform=ax.transAxes, color=EDGE, lw=1.0,
            clip_on=False, zorder=5)


def _callout(ax, x, y, text, key="slate", fontsize=10.5, ha="left", va="top",
             coords="data", weight="normal", **kw):
    """Pale fill, saturated matching text, thin matching border — never the other way."""
    return ax.annotate(text, (x, y), xycoords=coords, ha=ha, va=va, fontsize=fontsize,
                       color=C[key], fontweight=weight, linespacing=1.5, zorder=8,
                       bbox=dict(boxstyle="round,pad=0.46", fc=FILL[key], ec=C[key],
                                 lw=0.9), **kw)


# ------------------------------------------------------------------ figure 1
# T1 `PiWM-decode` vs `PiWM-decode-detach`, the same paired contrast as seeds landed
# (diary section 16.14). n = 2 carries no interval: t(.975, 1) = 12.71 standard errors is
# wider than the whole outcome range, so it is a point with its n.
LADDER = [(2, -0.240, None, None, "interim"),
          (3, -0.233, -0.39, -0.08, "wrong"),
          (6, -0.123, -0.27, +0.03, "interim"),
          (8, -0.060, -0.21, +0.09, "final")]
# where each reading's number goes: (n, y of the label, alignment)
LADDER_LABEL_Y = {2: -0.300, 3: -0.445, 6: -0.325, 8: -0.265}

# The same error, three times in one campaign. Each card: what was claimed, and what the
# evidence actually was at the time.
CASES = [
    ("P3   actgain-b03", "round 4", "interim",
     ["a null, " + MINUS + "0.067, on three seeds"],
     ["the interval was [" + MINUS + "0.52, +0.42]: 'destroys planning' to",
      "'doubles it'. An absence of evidence, not a null.",
      "At n = 8 it genuinely IS a null."]),
    ("P4   ctrb", "round 4", "interim",
     ["'a null by construction', from a mid-training",
      "val/ctrb_loss of 0.04 read off a partial run"],
     ["a fuller read gave 0.23; the converged median over",
      "8 runs is 0.033. Right conclusion, unearned evidence."]),
    ("T1   decode vs detach", "this figure", "wrong",
     ["an effect: " + MINUS + "0.233 [" + MINUS + "0.39, " + MINUS + "0.08], n = 3"],
     [MINUS + "0.060 [" + MINUS + "0.21, +0.09] at n = 8. A null, after halving twice."]),
]


def fig_n3(out):
    """The n = 3 lesson: a thin interval is three numbers agreeing, not an effect."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.6, 7.4),
                                 gridspec_kw={"width_ratios": [1.30, 1]})
    fig.subplots_adjust(left=0.062, right=0.985, top=0.800, bottom=0.095, wspace=0.19)

    # --- left: the point estimate and its interval as seeds accumulated ---
    YLO, YHI = -0.54, 0.42
    ax.add_patch(Rectangle((2.60, YLO), 0.80, 0.80, facecolor=FILL["crimson"], alpha=0.55,
                           edgecolor="none", zorder=1))
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    ax.annotate("no effect", (1.44, 0.012), fontsize=10.5, color=INK, ha="left",
                va="bottom", zorder=4)

    ns = [r[0] for r in LADDER]
    ax.plot(ns, [r[1] for r in LADDER], color=MUTED, lw=1.8, zorder=4)

    for n, m, lo, hi, role in LADDER:
        col = ROLE[role]
        if lo is not None:
            ax.plot([n, n], [lo, hi], color=col, lw=2.8, solid_capstyle="butt", zorder=5)
            for v in (lo, hi):
                ax.plot([n - 0.10, n + 0.10], [v, v], color=col, lw=2.8, zorder=5)
        ax.scatter([n], [m], s=190 if role != "interim" else 120, color=col, zorder=6,
                   edgecolor="white", lw=1.5)
        txt = _num(m) if lo is None else f"{_num(m)}   [{_num(lo, '{:+.2f}')}, " \
                                         f"{_num(hi, '{:+.2f}')}]"
        ax.annotate(txt, (n, LADDER_LABEL_Y[n]), ha="center", va="center", fontsize=11,
                    color=col, fontweight="bold", zorder=7)

    # the two direct labels that replace a legend
    ax.annotate("reported as an effect.\nIt was not one.", (3.05, 0.028), ha="center",
                va="bottom", fontsize=12, color=C["crimson"], fontweight="bold",
                linespacing=1.4, zorder=7)
    ax.annotate("n = 8: the settled reading — a null", (7.90, 0.115), ha="center",
                va="bottom", fontsize=11, color=INK, fontweight="bold", zorder=7)

    _callout(ax, 1.44, 0.405,
             "n = 2 carries no interval: on two seeds a paired 95%\n"
             "interval is t(.975, 1) = 12.7 standard errors wide,\n"
             "wider than the whole outcome range",
             key="slate", fontsize=10, ha="left", va="top")
    _callout(ax, 9.30, 0.405,
             "the estimate halves twice on the way to zero:\n"
             "0.233 \u2192 0.123 \u2192 0.060, and the interval\n"
             "covers zero from n = 6 onwards",
             key="green", fontsize=10.5, ha="right", va="top", weight="bold")

    ax.set_xlim(1.35, 9.40)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"n = {n}" for n in ns], fontsize=11)
    ax.set_ylim(YLO, YHI)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("paired seeds available when the reading was taken", fontsize=11.5,
                  color=INK, labelpad=7)
    ax.set_ylabel("PiWM-decode minus PiWM-decode-detach\n(paired difference in CEM success)",
                  fontsize=11.5, color=INK)
    S.ax_style(ax, grid="y")
    _panel(ax, "A", "The same contrast, read four times",
           "T1 decode vs decode-detach, as its eight seeds landed")

    # --- right: the campaign's three instances of the same error ---
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis("off")
    _panel(bx, "B", "The same error, three times in one campaign",
           "what was claimed, and what the evidence actually was")

    HEAD_H, LINE, PAD, GAPY, BLOCKGAP = 0.060, 0.042, 0.026, 0.034, 0.013
    XLAB, XTXT = 0.030, 0.145
    y = 0.955
    for head, when, role, called, was in CASES:
        h = HEAD_H + LINE * (len(called) + len(was)) + BLOCKGAP + PAD
        y -= h
        col = ROLE[role]
        key = "crimson" if role == "wrong" else "slate"
        bx.add_patch(FancyBboxPatch((0.004, y), 0.992, h, boxstyle="round,pad=0.0,rounding_size=0.012",
                                    transform=bx.transAxes, zorder=2,
                                    fc=FILL[key], ec=C[key] if role == "wrong" else EDGE,
                                    lw=1.0))
        bx.add_patch(Rectangle((0.004, y), 0.010, h, transform=bx.transAxes, zorder=3,
                               fc=col, ec="none"))
        bx.annotate(head, (XLAB, y + h - 0.026), xycoords="axes fraction", va="top",
                    ha="left", fontsize=11.5, color=col, fontweight="bold", zorder=4)
        bx.annotate(when, (0.975, y + h - 0.028), xycoords="axes fraction", va="top",
                    ha="right", fontsize=10, color=MUTED, zorder=4)
        yy = y + h - HEAD_H - 0.012
        for lab, lines, c in (("called", called, INK), ("was", was, MUTED)):
            bx.annotate(lab, (XLAB, yy), xycoords="axes fraction", va="top", ha="left",
                        fontsize=10, color=col, fontweight="bold", zorder=4)
            for i, ln in enumerate(lines):
                bx.annotate(ln, (XTXT, yy - LINE * i), xycoords="axes fraction", va="top",
                            ha="left", fontsize=10, color=c, zorder=4)
            yy -= LINE * len(lines) + BLOCKGAP
        y -= GAPY

    bx.annotate("Design rule 6: an uninformative interval is not a null,\n"
                "and a thin one at n = 3 is not an effect.",
                (0.004, 0.012), xycoords="axes fraction", va="bottom", ha="left",
                fontsize=11.5, color=C["green"], fontweight="bold", linespacing=1.45,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.52", fc=FILL["green"], ec=C["green"],
                          lw=0.9))

    fig.text(0.006, 0.995,
             "An interval that excludes zero at n = 3 is evidence that three numbers "
             "agreed, not evidence of an effect",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.006, 0.988,
             "The T1 decode contrast read " + MINUS + "0.233 [" + MINUS + "0.39, " + MINUS
             + "0.08] at n = 3 and " + MINUS + "0.060 [" + MINUS + "0.21, +0.09] at n = 8 "
             "— a null. Nothing changed but the number of seeds.\nA paired 95% interval at "
             "n = 3 spans ±4.30 standard errors, so three seeds that happen to agree "
             "produce a thin interval with no effect underneath it.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.5)

    p = os.path.join(out, "n3-lesson.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 2
# Converged training diagnostics, diary section 16.13. Different seed ids per arm because
# the two arms were launched on different seed blocks; the CEM contrasts on the right are
# the paired n = 8 numbers from section 16.15.
T3 = {"PiWM-contact": dict(label="PiWM-contact",
                           sub="gamma = 1.0, contact-weighted",
                           seeds=["s3", "s7", "s8"], mse=[0.838, 1.0000, 1.0000],
                           rho=[0.206, 0.039, 0.014], ess=0.38),
      "PiWM-contact-shuf": dict(label="PiWM-contact-shuf",
                                sub="control: same weights, random transitions",
                                seeds=["s4", "s5", "s9"], mse=[0.063, 0.044, 0.046],
                                rho=[0.576, 0.607, 0.415], ess=0.38)}
CEM = [("LpWM-ltv", "LpWM-ltv", "baseline", 0.357, 13),
       ("PiWM-contact-shuf", "PiWM-contact-shuf", "control", 0.048, 8),
       ("PiWM-contact", "PiWM-contact", "treatment", 0.025, 8)]


def fig_t3(out):
    """T3 fails twice: the reweighting costs -0.345 on its own, and the contact selection
    additionally collapses training."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.0, 7.0),
                                 gridspec_kw={"width_ratios": [1.06, 1]})
    fig.subplots_adjust(left=0.058, right=0.995, top=0.790, bottom=0.150, wspace=0.20)

    # --- left: converged rel_mse per seed, with the identical ESS as the punchline ---
    xs = {"PiWM-contact": np.array([0.0, 1.0, 2.0]),
          "PiWM-contact-shuf": np.array([4.2, 5.2, 6.2])}
    ticks, ticklabels = [], []
    for arm, d in T3.items():
        x, col = xs[arm], ARM[arm]
        ax.bar(x, d["mse"], width=0.72, color=col, zorder=3)
        for xi, m in zip(x, d["mse"]):
            ax.annotate(f"{m:.3f}", (xi, m), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=11.5, color=col, fontweight="bold", zorder=5)
        ticks += list(x)
        ticklabels += [f"{sd}\n" + r"$\rho$ " + f"{r:.3f}"
                       for sd, r in zip(d["seeds"], d["rho"])]
        # ESS bracket: the identity of these two numbers is the whole argument
        lo, hi, y = x[0] - 0.42, x[-1] + 0.42, 1.150
        ax.plot([lo, lo, hi, hi], [y - 0.035, y, y, y - 0.035], color=INK, lw=1.4, zorder=6)
        ax.annotate(f"ESS {d['ess']:.2f}", ((lo + hi) / 2, y + 0.025), ha="center",
                    va="bottom", fontsize=13.5, color=INK, fontweight="bold", zorder=6)
        # the arm's identity, directly under its own group — no legend needed
        ax.annotate(d["label"], ((lo + hi) / 2, -0.175), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=11.5, color=col, fontweight="bold",
                    annotation_clip=False, zorder=6)
        ax.annotate(d["sub"], ((lo + hi) / 2, -0.235), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=10.5, color=MUTED,
                    annotation_clip=False, zorder=6)

    xmid = (xs["PiWM-contact"][-1] + 0.42 + xs["PiWM-contact-shuf"][0] - 0.42) / 2
    ax.annotate("=", (xmid, 1.150), ha="center", va="center", fontsize=20, color=INK,
                fontweight="bold", zorder=6)
    ax.annotate("identical", (xmid, 1.088), ha="center", va="top", fontsize=10.5,
                color=MUTED, zorder=6)

    ax.axhline(0.5, color=C["crimson"], lw=1.6, ls=(0, (6, 4)), zorder=4)
    _callout(ax, 6.90, 0.545, "pre-registered death condition:  rel_mse \u2265 0.5",
             key="crimson", fontsize=10.5, ha="right", va="bottom", weight="bold")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=10.5)
    ax.set_xlim(-0.80, 7.00)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("converged val rel_mse\n"
                  "(1.0 = the predictor emits the conditional mean)", fontsize=11.5,
                  color=INK)
    S.ax_style(ax, grid="y")
    _panel(ax, "A", "Same weight distribution, same effective sample size",
           "only WHICH transitions are upweighted differs — and one arm trains, one does not")

    # --- right: what it costs at plan time ---
    ys = np.arange(len(CEM))[::-1]
    bx.add_patch(Rectangle((0, -0.80), 0.06, 3.42, facecolor=FILL["slate"],
                           edgecolor="none", zorder=1))
    bx.annotate("the floor", (0.078, -0.60), ha="left", va="center", fontsize=10.5,
                color=MUTED, style="italic", zorder=5)
    for y, (arm, lab, role, v, n) in zip(ys, CEM):
        bx.barh(y, v, height=0.46, color=ARM[arm], zorder=3)
        bx.annotate(f"{v:.3f}", (v, y), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=12, color=ARM[arm], fontweight="bold", zorder=5)
        bx.annotate(f"arm mean, n = {n}", (v, y), xytext=(58, 0),
                    textcoords="offset points", va="center", fontsize=10.5, color=MUTED,
                    zorder=5)
    XA = 0.60
    for y0, y1, txt, key in (
            (2, 1, _num(-0.345) + "  [" + _num(-0.476) + ", " + _num(-0.214) + "]\n"
                   "the reweighting alone, with nothing\n"
                   "selected for: a third of the success rate", "crimson"),
            (1, 0, _num(-0.022) + "  [" + _num(-0.140) + ", " + _num(+0.095) + "]\n"
                   "a null — but only because both\narms are already on the floor",
             "slate")):
        bx.annotate("", (XA, y0), xytext=(XA, y1), zorder=4,
                    arrowprops=dict(arrowstyle="<|-|>,head_width=0.26", color=C[key],
                                    lw=1.8, shrinkA=3, shrinkB=3))
        _callout(bx, XA + 0.035, (y0 + y1) / 2, txt, key=key, fontsize=10.5, ha="left",
                 va="center", weight="bold")
    bx.set_yticks(ys)
    bx.set_yticklabels([f"{lab}\n({role})" for _, lab, role, _, _ in CEM], fontsize=11)
    bx.set_ylim(-0.80, 2.62)
    bx.set_xlim(0, 1.30)
    bx.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    bx.set_xlabel("CEM success rate          (arrows: paired difference over 8 matched "
                  "seeds)", fontsize=11.5, color=INK, labelpad=7)
    S.ax_style(bx, grid="x")
    bx.spines["bottom"].set_bounds(0, 0.44)     # the axis stops where the data does; the
    bx.spines["left"].set_visible(False)        # rest of the panel is an annotation gutter
    bx.tick_params(axis="y", length=0)
    _panel(bx, "B", "The informative contrast is the control against the baseline",
           "not the treatment against its control — both of those sit on the floor")

    fig.text(0.006, 0.995,
             "The reweighting machinery is fatal on its own, independent of what it "
             "upweights",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.006, 0.988,
             "Both T3 arms carry the same weight distribution at ESS 0.38. The control "
             "attaches it to RANDOM transitions and trains fine (rel_mse 0.046 vs 1.0000) "
             "— yet still loses\n" + MINUS + "0.345 [" + MINUS + "0.476, " + MINUS
             + "0.214] of CEM success. So contact selection collapses training, and "
             "shrinking the effective batch ~20× destroys planning; these are separate "
             "failures.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.5)

    p = os.path.join(out, "t3-reweighting.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in (fig_n3(a.out), fig_t3(a.out)):
        if p:
            print("  wrote", p)


if __name__ == "__main__":
    main()
