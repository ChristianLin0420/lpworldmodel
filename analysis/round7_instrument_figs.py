"""Why most of this campaign's nulls were unreadable: the instrument, in one figure.

    python analysis/round7_instrument_figs.py --out diary/assets/2026-09-05
    -> diary/assets/2026-09-05/instrument.png

Six rounds moved the training objective and returned one positive (plan-time consensus).
The success-rate reanalysis (README §7, 2026-09-04 §7.0b, 2026-09-05 §2b) says that is
roughly what this design could have returned whatever was true, and names two detectors the
campaign never had. Three panels, in the order the argument runs.

  A  RESOLUTION. What a paired n-seed contrast can see. Two contrast types: a RETRAINED arm
     (median sd of the paired difference 0.147 over 33 contrasts, MDE 0.150 at n = 8) and a
     same-checkpoint planner/eval contrast (0.084 over 13, MDE 0.086). Against them, the
     between-arm sd of the entire healthy-arm population, 0.099 noise-corrected. The reading:
     at the campaign's n = 8 floor a retrained contrast cannot resolve anything smaller than
     1.51x the whole spread of the population it is sampling; it needs n = 19 seeds to get
     under that spread, where the planner/eval contrast needs 7.

  B  WHY. The dominant axis is the TRAINING SEED. `PiWM-soloM` is a fully crossed 5x5
     model x episode-block matrix -- `PiWM-solo{m}` evaluates checkpoint `LpWM-ltv_s{m}` on
     block `s{b}` (scripts/plan_consensus_controls.sh:93), so the diagonal m == b is just
     `LpWM-ltv_s{m}` on its own block. Identical configs; only the training seed differs.
     Family A splits 92 / 1 / 7 across model / block / residual, with model means running
     0.180 -> 0.600 while every block mean lies inside 0.348-0.392. And pairing does not
     remove it: the mean pairwise correlation of demeaned seed profiles across healthy arms
     is 0.150, so the seed effect is an arm x seed INTERACTION and differencing cancels
     almost none of it.

  C  THE TWO MISSING DETECTORS. `sparsity/effective_dim == 0` (a constant-output model) and
     `median agent_pos_diff > 100 px` (a diverging rollout). One point per finished run that
     has an eval trace, coloured by whether its success rate clears 0.1. 139 of the 174 runs
     these two flag would NOT fire the pre-registered `err/rel_mse >= 0.5` gate, because a
     constant-output model has rel_mse ~ 0 and a diverging rollout is invisible to a
     one-step training loss.

PROVENANCE. Every number is either recomputed here from the archive or carried from the
round's notes, and the figure says which on its face.

  RECOMPUTED HERE (analysis.collect_evals.collect(), scheme="fixed"; assets/run_health.json):
    * the whole of panel B's arithmetic -- both 5x5 matrices, the model and block means, the
      92/1/7 and 63/18/19 sums of squares, the residual sds 0.048 and 0.057;
    * the cross-arm correlation of demeaned seed profiles, 0.1497 over 703 pairs of the 38
      healthy arms -- which is the brief's 0.150;
    * the between-arm sd of the healthy-arm population (mean >= 0.20, n >= 8): 0.1092 raw,
      0.0991 after subtracting the mean sampling variance of an arm mean -- the brief's 0.099;
    * the three effects at stake, from analysis.figures.paired_effect: columns +0.072
      (sd 0.213, n = 12), patchdecode +0.140 against its own -detach control (sd 0.224,
      n = 8), consensus vote5-median +0.228 (sd 0.136, n = 10);
    * the required n for columns, 69, from its own paired sd by the two-sided normal
      approximation n = ((z_.975 + z_.80) sd / delta)^2 -- 69.4, which is the notes' 69;
    * all of panel C: 574 finished runs, 289 with an eval trace, 174 flagged, 139 of them
      missed by `rel_mse >= 0.5`, 22 collapsed runs (7 of them with traces, every one at
      SR <= 0.04), and the 100 px / effective_dim == 0 thresholds themselves.

  CARRIED FROM THE BRIEF / the round's notes, NOT recomputed:
    * the split of the campaign's contrasts into 33 retrained and 13 same-checkpoint, and
      their median paired sds 0.147 and 0.084 with MDEs 0.150 and 0.086 at n = 8;
    * the training-seed sd 0.144 against a binomial sd of 0.043;
    * the healthy 13-28 px / failing 1000-2700 px arm-level agent bands;
    * patchdecode's required n = 36. Flagged on the figure, because the same normal
      approximation that reproduces columns' 69 exactly gives n = 20 from patchdecode's
      measured paired sd of 0.224. The two disagree and the figure does not hide it.

  DERIVED, and labelled as arithmetic rather than as a measurement:
    * the MDE curves. MDE scales as 1/sqrt(n) at fixed sd, so each curve is anchored on the
      notes' own n = 8 value and scaled: MDE(n) = MDE(8) sqrt(8/n). No second power
      calculation is done here, so the curve cannot disagree with the number printed on it.
    * "pairing buys ~7 %": sqrt(1 - 0.150) = 0.922, i.e. a 7.8 % cut in the sd of the paired
      difference (15 % of the variance), against the 100 % that "pairing removes the seed
      effect" would need.

STYLE. analysis/style.py, imported and never edited. Hues by IDENTITY:
    green    the training side -- the retrained-arm contrast (A), SS(model) (B), and the runs
             that plan successfully (C). The reference calls green "the system".
    purple   the contrasting condition -- the same-checkpoint planner/eval contrast (A),
             SS(block), i.e. the evaluation axis (B), and the incumbent `rel_mse` gate (C).
    amber    the intervention under test -- the two proposed detectors and their thresholds.
    crimson  failure / alert ONLY: the 0.099 population sd and the region a retrained
             contrast cannot resolve (A), the runs at SR <= 0.1 (C). No arm owns crimson.
    slate    the nulls and neutral furniture. teal is the campaign's one positive.
  The two under-powered nulls are drawn as nulls -- slate, hollow, no diamond -- and the one
  positive as a filled teal diamond. Repainting by rank is what this rule exists to stop.

LAYOUT. No tight_layout: panel B's two heatmaps carry their marginal means INSIDE the axes
(the limits are extended and the extra row/column is labelled by a tick) precisely so that
nothing hangs outside where tight_layout cannot see it. `audit()` runs on every render and
reports any pair of text boxes that intersect, any data-coordinate label that crosses a
spine, and any text that leaves the canvas; it is the reason for the panel limits below,
which are the values at which it stops complaining rather than values chosen by taste.
"""
import argparse
import itertools
import json
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import matplotlib.patheffects as pe                                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                   # noqa: E402
from matplotlib.ticker import NullLocator                               # noqa: E402
from matplotlib.transforms import Bbox                                  # noqa: E402

from analysis import figures as FG                                      # noqa: E402
from analysis.collect_evals import collect, resolve_arm                 # noqa: E402
from analysis.style import (C, FILL, INK, MUTED, SEQ_GREEN,             # noqa: E402
                            ax_style, callout, panel_title, use_style)

HEALTH_JSON = "assets/run_health.json"

#: Two-sided alpha = .05 at 80 % power. Only used for the required-n arithmetic, which is
#: the form that reproduces the notes' n = 69 for the columns contrast exactly.
Z_SUM = 1.959963985 + 0.841621234

#: The campaign's seed floor, and the n every MDE in the notes is quoted at.
N_FLOOR = 8

# --- numbers carried from the brief / the round's notes ---------------------------
#: (label, median paired sd, MDE at n = 8, number of contrasts, hue key).
#: README §7 table "the instrument is coarser than the effects it was applied to".
CONTRAST_TYPES = [
    ("retrained arm", 0.147, 0.150, 33, "green"),
    ("same checkpoints, planner/eval varied", 0.084, 0.086, 13, "purple"),
]
#: README §7 §3. Recomputed here only as far as the SS split and the model means; the sd
#: pair itself is quoted, not re-derived.
BRIEF_SEED_SD, BRIEF_BINOM_SD = 0.144, 0.043
#: 2026-09-04 §7 retraction box: arm-level medians over pooled episodes.
BRIEF_HEALTHY_PX, BRIEF_FAILING_PX = (13.0, 28.0), (1000.0, 2700.0)
#: 2026-09-05 §2b / §S4. columns' 69 is reproduced below; patchdecode's 36 is not.
BRIEF_N_REQ = {"columns": 69, "patchdecode": 36}
#: run_health.py's own constants, restated so the panel cannot drift from the detector.
DIVERGENCE_PX, SR_OK = 100.0, 0.1

#: Footer wrap width, in characters. 8.2 pt on an 18.6 in canvas holds about 300;
#: unwrapped, one of these paragraphs measures 5700 px against a 2976 px canvas.
WRAP = 300

#: A white halo for any label that has to cross a curve or a rule.
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]
CMAP = LinearSegmentedColormap.from_list("seqgreen", SEQ_GREEN)


def num(x, fmt="+.3f"):
    """Format with a real minus sign, so a drawn number matches the tick labels."""
    return format(x, fmt).replace("-", "−")


# --- panel A: resolution ----------------------------------------------------------
def mde_at(mde8, n):
    """MDE at n seeds, anchored on the notes' value at n = 8.

    At a fixed sd the minimum detectable effect is proportional to 1/sqrt(n) whatever the
    exact power calculation, so anchoring on the published n = 8 number and scaling keeps
    the curve and the printed value from ever disagreeing.
    """
    return mde8 * np.sqrt(float(N_FLOOR) / np.asarray(n, dtype=float))


def n_for_mde(mde8, target):
    """Seeds needed for the MDE of that contrast type to reach `target`."""
    return N_FLOOR * (mde8 / target) ** 2


def n_required(delta, sd):
    """n for 80 % power at alpha = .05 on a paired difference, normal approximation."""
    return (Z_SUM * sd / delta) ** 2


def effects(arms):
    """The three effects at stake, recomputed from the archive.

    columns and patchdecode are the two the round's own screens ride on and both are nulls;
    consensus is the campaign's only positive. Each carries its OWN paired sd, which is the
    whole point -- columns' 0.213 is 1.45x the population median of 0.147, which is why its
    honest price is n = 69 and not the ~35 the median-sd curve would suggest.
    """
    def pe_(ctrl, arm):
        return FG.paired_effect(arms, resolve_arm(arms, ctrl), resolve_arm(arms, arm))

    out = {
        "columns": dict(pe_("LpWM-ltv", "PiWM-columns"), key="slate", null=True,
                        label="columns  +0.072  vs LpWM-ltv", ctrl="vs LpWM-ltv"),
        "patchdecode": dict(pe_("PiWM-patchdecode-detach", "PiWM-patchdecode"),
                            key="slate", null=True,
                            label="patchdecode  +0.140  vs −detach",
                            ctrl="vs its own −detach control"),
        "consensus": dict(pe_("LpWM-ltv", "PiWM-vote5-median"), key="teal", null=False,
                          label="consensus  +0.228  vote5-median",
                          ctrl="vote5-median vs LpWM-ltv"),
    }
    for k, e in out.items():
        e["n_req"] = n_required(e["mean"], e["sd"])
        e["spans_zero"] = bool(e["lo"] < 0 < e["hi"])
        assert e["null"] == e["spans_zero"], f"{k}: null encoding disagrees with its interval"
    return out


def population_sd(arms, min_mean=0.20, min_n=8):
    """Between-arm sd of the healthy-arm population, raw and noise-corrected.

    An arm mean over n seeds carries sampling noise var/n of its own; subtracting the mean of
    those inflations is what makes 0.109 into 0.099. Reported as `corr`.
    """
    H = {a: np.array([float(x) for x in v.values()])
         for a, v in arms.items() if len(v) >= min_n}
    H = {a: v for a, v in H.items() if v.mean() >= min_mean}
    means = np.array([v.mean() for v in H.values()])
    inflation = np.mean([v.var(ddof=1) / v.size for v in H.values()])
    raw = float(means.std(ddof=1))
    return dict(arms=len(H), raw=raw, corr=float(np.sqrt(max(raw ** 2 - inflation, 0.0))))


def panel_resolution(ax, E, pop):
    """A: MDE against n for both contrast types, against the spread of the population."""
    xlo, xhi, ytop = 2.62, 96.0, 0.305
    grid = np.geomspace(xlo, xhi, 400)
    ret, pln = CONTRAST_TYPES[0], CONTRAST_TYPES[1]

    # everything under the retrained curve is invisible to a retrained contrast at that n
    ax.fill_between(grid, 0, mde_at(ret[2], grid), color=FILL["crimson"], alpha=0.60,
                    lw=0, zorder=0)
    for label, sd, mde8, k, key in CONTRAST_TYPES:
        ax.plot(grid, mde_at(mde8, grid), color=C[key], lw=2.6,
                ls="-" if key == "green" else (0, (5.5, 2.2)), zorder=4,
                solid_capstyle="round")

    # the reference: the whole spread of the population these contrasts sample
    ax.axhline(pop["corr"], color=C["crimson"], lw=1.7, zorder=3)
    ax.text(xhi * 0.97, pop["corr"] + 0.0075,
            f"between-arm sd of the whole healthy-arm population   {pop['corr']:.3f}",
            ha="right", va="bottom", fontsize=9.2, fontweight="bold", color=C["crimson"],
            path_effects=HALO)

    # where each contrast type finally gets under that spread
    for label, sd, mde8, k, key in CONTRAST_TYPES:
        nx = n_for_mde(mde8, pop["corr"])
        ax.scatter([nx], [pop["corr"]], s=96, facecolor="white", edgecolor=C[key],
                   linewidth=2.0, zorder=6)
        ax.plot([nx, nx], [0, pop["corr"]], color=C[key], lw=0.9, ls=(0, (2, 2)), zorder=2)
        ax.text(nx, pop["corr"] - 0.008, f"n = {int(np.ceil(nx))}", ha="center",
                va="top", fontsize=9.4, fontweight="bold", color=C[key],
                path_effects=HALO)

    # the campaign's floor, and what each instrument reads there
    ax.plot([N_FLOOR, N_FLOOR], [0, 0.170], color=C["slate"], lw=1.0,
            ls=(0, (2, 2.4)), zorder=2)
    ax.text(N_FLOOR, 0.176, "n = 8\nthe campaign's floor", ha="center", va="bottom",
            fontsize=9.0, color=C["slate"], linespacing=1.3)
    for label, sd, mde8, k, key in CONTRAST_TYPES:
        ax.scatter([N_FLOOR], [mde8], s=74, color=C[key], edgecolor="white", linewidth=1.0,
                   zorder=6)
        ax.text(N_FLOOR * 1.13, mde8, f"MDE {mde8:.3f}", ha="left", va="center",
                fontsize=9.6, fontweight="bold", color=C[key], path_effects=HALO)

    # the curves, labelled where they run
    ax.text(xlo * 1.03, mde_at(ret[2], xlo) + 0.008,
            f"RETRAINED arm contrast\nmedian paired sd {ret[1]:.3f}  ·  {ret[3]} contrasts",
            ha="left", va="bottom", fontsize=9.4, fontweight="bold", color=C["green"],
            linespacing=1.35, path_effects=HALO)
    ax.text(xhi * 0.97, 0.011,
            f"same checkpoints, planner/eval varied  ·  median paired sd {pln[1]:.3f}"
            f"  ·  {pln[3]} contrasts",
            ha="right", va="bottom", fontsize=9.4, fontweight="bold", color=C["purple"],
            path_effects=HALO)

    # the effects at stake
    place = {"columns": (-1, "left", "needs n = %d"),
             "patchdecode": (-1, "left", "needs n = %d"),
             "consensus": (+1, "right", None)}
    for name, e in E.items():
        y, key = e["mean"], e["key"]
        side, at, tmpl = place[name]
        ax.plot([xlo, xhi], [y, y], color=C[key], lw=1.2, ls=(0, (1.6, 2.0)), zorder=3)
        ax.text(xlo * 1.03 if at == "left" else xhi * 0.97, y + side * 0.0075, e["label"],
                ha=at, va="bottom" if side > 0 else "top", fontsize=9.2,
                fontweight="bold", color=C[key], path_effects=HALO)
        if tmpl:
            nx = BRIEF_N_REQ[name]
            ax.scatter([nx], [y], s=96, facecolor="white", edgecolor=C[key], linewidth=2.0,
                       zorder=6)
            ax.text(nx, y + 0.0075, tmpl % nx, ha="center", va="bottom", fontsize=9.2,
                    fontweight="bold", color=C[key], path_effects=HALO)
        else:
            ax.scatter([e["n"]], [y], s=110, color=C[key], marker="D", edgecolor="white",
                       linewidth=1.0, zorder=6)
            ax.text(e["n"] * 1.16, y, f"resolved at n = {e['n']}", ha="left", va="center",
                    fontsize=9.2, fontweight="bold", color=C[key], path_effects=HALO)

    callout(ax, 13.6, 0.200,
            "A RETRAINED CONTRAST AT n = 8 CANNOT RESOLVE\n"
            f"ANYTHING SMALLER THAN {CONTRAST_TYPES[0][2] / pop['corr']:.2f}× THE ENTIRE\n"
            "SPREAD OF THE POPULATION IT SAMPLES",
            key="crimson", fontsize=9.6, ha="left", va="top")

    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, ytop)
    ax.set_xticks([3, 4, 6, 8, 12, 16, 24, 36, 50, 69])
    ax.set_xticklabels([str(v) for v in (3, 4, 6, 8, 12, 16, 24, 36, 50, 69)])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("paired seeds per contrast, n   (log scale)", fontsize=10, color=INK)
    ax.set_ylabel("effect on CEM success", fontsize=10, color=INK)
    ax_style(ax, grid="y")
    panel_title(ax, "A", "The instrument is coarser than the effects it was applied to",
                "minimum detectable effect, 80 % power, α = .05  ·  shaded = "
                "invisible to a retrained contrast at that n")


# --- panel B: the training seed ---------------------------------------------------
#: `PiWM-solo{m}` = checkpoint LpWM-ltv_s{m} on block s{b}; the diagonal is LpWM-ltv itself.
FAMILIES = [("A", list(range(3, 8))), ("B", list(range(8, 13)))]


def solo_matrix(arms, seeds):
    """The 5x5 model x block success matrix, diagonal filled from LpWM-ltv."""
    M = np.zeros((len(seeds), len(seeds)))
    for i, m in enumerate(seeds):
        for j, b in enumerate(seeds):
            M[i, j] = float(arms["LpWM-ltv"][str(m)] if m == b
                            else arms[f"PiWM-solo{m}"][str(b)])
    return M


def two_way(M):
    """Sums of squares for a fully crossed design with one observation per cell."""
    g, n = M.mean(), M.shape[0]
    ss_m = n * float(((M.mean(1) - g) ** 2).sum())
    ss_b = n * float(((M.mean(0) - g) ** 2).sum())
    ss_t = float(((M - g) ** 2).sum())
    ss_r = ss_t - ss_m - ss_b
    return dict(model=100 * ss_m / ss_t, block=100 * ss_b / ss_t, resid=100 * ss_r / ss_t,
                resid_sd=float(np.sqrt(ss_r / ((n - 1) ** 2))), grand=float(g),
                row=M.mean(1), col=M.mean(0))


def cross_arm_rho(arms, min_mean=0.20, min_n=8, min_share=4):
    """Mean pairwise correlation of demeaned per-seed profiles across healthy arms.

    If the training-seed effect were common-mode, these profiles would be near-perfectly
    correlated and a paired difference would cancel it. They are not: rho-bar ~ 0.15.
    """
    P = {}
    for a, v in arms.items():
        if len(v) < min_n:
            continue
        vals = np.array([float(x) for x in v.values()])
        if vals.mean() < min_mean:
            continue
        P[a] = {s: float(x) - vals.mean() for s, x in v.items()}
    rs = []
    for a, b in itertools.combinations(sorted(P), 2):
        sh = sorted(set(P[a]) & set(P[b]))
        if len(sh) < min_share:
            continue
        x = np.array([P[a][s] for s in sh])
        y = np.array([P[b][s] for s in sh])
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    return dict(arms=len(P), pairs=len(rs), rho=float(np.mean(rs)))


def heat(ax, fam, seeds, M, D, letter=None):
    """One family's 5x5 matrix, with its marginal means drawn INSIDE the axes."""
    n = len(seeds)
    lo, hi = 0.12, 0.68
    ax.imshow(M, cmap=CMAP, vmin=lo, vmax=hi, aspect="auto", zorder=1,
              extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if i == j else "normal",
                    color="white" if v > 0.44 else INK, zorder=3)
            if i == j:      # the diagonal is LpWM-ltv on its own block
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=C["slate"], lw=1.6, ls=(0, (2.6, 1.8)),
                                           zorder=4))
    # marginals, inside the axes so nothing hangs where a layout engine cannot see it
    for i in range(n):
        ax.text(n + 0.35, i, f"{D['row'][i]:.3f}", ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=C["green"], zorder=3)
    for j in range(n):
        ax.text(j, n + 0.32, f"{D['col'][j]:.3f}", ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=C["purple"], zorder=3)
    ax.set_xlim(-0.5, n + 0.95)
    ax.set_ylim(n + 0.95, -0.5)
    ax.set_xticks(list(range(n)) + [n + 0.35])
    ax.set_xticklabels([f"s{s}" for s in seeds] + ["model\nmean"], fontsize=9)
    ax.set_yticks(list(range(n)) + [n + 0.32])
    ax.set_yticklabels([f"s{s}" for s in seeds] + ["block mean"], fontsize=9)
    ax.tick_params(length=0, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_xlabel("episode block (eval seed)", fontsize=9.6, color=INK, labelpad=2)
    ax.set_ylabel("model (training seed)", fontsize=9.6, color=INK, labelpad=2)
    sub = (f"model means {D['row'].min():.3f} → {D['row'].max():.3f}"
           f"  ({D['row'].max() / D['row'].min():.1f}×)   ·   "
           f"blocks {D['col'].min():.3f}–{D['col'].max():.3f}")
    if letter:
        panel_title(ax, letter,
                    "The dominant axis is the TRAINING SEED, and pairing does not remove it",
                    y=1.272)
        ax.text(0.0, 1.187, "identical configs, one observation per cell  ·  "
                            "dashed cells are the diagonal: LpWM-ltv on its own block",
                transform=ax.transAxes, fontsize=9.2, color=MUTED, va="bottom", ha="left")
    ax.text(0.0, 1.092, f"family {fam}  ·  seeds {seeds[0]}–{seeds[-1]}",
            transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK,
            va="bottom", ha="left")
    ax.text(0.0, 1.010, sub, transform=ax.transAxes, fontsize=8.6, color=MUTED,
            va="bottom", ha="left")


def ss_bars(ax, decomp, rho):
    """The variance split, and the note that pairing does not undo it."""
    keys = [("model", "green", "training seed (model)"),
            ("block", "purple", "episode block"),
            ("resid", "slate", "residual")]
    for r, (fam, D) in enumerate(decomp):
        y, left = 1.15 - 1.15 * r, 0.0
        for k, key, _ in keys:
            w = D[k]
            ax.barh([y], [w], left=left, height=0.46, color=C[key],
                    edgecolor="white", linewidth=1.0, zorder=2)
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f} %", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=3)
            else:
                # a 1 % sliver cannot hold its own label; it gets a leader instead, and it
                # goes UP, because down is the other family's bar.
                ax.annotate(f"{w:.0f} %", xy=(left + w / 2, y + 0.23),
                            xytext=(left + w / 2, y - 0.40), ha="center", va="top",
                            fontsize=9, fontweight="bold", color=C[key], zorder=3,
                            arrowprops=dict(arrowstyle="-", color=C[key], lw=0.9,
                                            shrinkA=1, shrinkB=1))
            left += w
        ax.text(101.0, y, f"residual sd {D['resid_sd']:.3f}", ha="left", va="center",
                fontsize=9, color=MUTED)
    for x, (k, key, name) in zip((3.0, 46.0, 74.0), keys):
        ax.text(x, 1.72, name, ha="left", va="bottom", fontsize=9.2, fontweight="bold",
                color=C[key])
    # The two facts the matrices exist to establish, under the bars where they belong.
    ax.text(0.0, -0.30, f"training-seed sd {BRIEF_SEED_SD:.3f}   against a binomial sd of "
                        f"{BRIEF_BINOM_SD:.3f}   (both from the round's notes)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.6, fontweight="bold",
            color=C["green"])
    ax.text(0.0, -0.62, "and pairing does NOT cancel it: the mean cross-arm correlation of "
                        f"demeaned seed profiles is {rho['rho']:.3f} over {rho['pairs']} "
                        f"pairs of {rho['arms']} healthy arms,\nso differencing sheds "
                        f"{100 * (1 - np.sqrt(1 - rho['rho'])):.1f} % of the sd — the seed "
                        "effect is an arm × seed INTERACTION, not a common-mode offset.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2, color=MUTED,
            linespacing=1.5)
    ax.set_xlim(0, 128)
    ax.set_ylim(-0.45, 2.15)
    ax.set_yticks([1.15, 0])
    ax.set_yticklabels(["family A", "family B"], fontsize=9.6)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_xlabel("share of the total sum of squares", fontsize=9.6, color=INK, labelpad=1)


# --- panel C: the two missing detectors -------------------------------------------
def health_rows(arms_detail, path=HEALTH_JSON):
    """assets/run_health.json joined to the planner's success rate, run by run."""
    rows = json.load(open(path))
    for r in rows:
        d = arms_detail.get(r["run"])
        r["sr"] = None if d is None else float(d["success"])
    flagged = [r for r in rows if r["collapsed"] or r["diverged"]]
    missed = [r for r in flagged if not (r["rel_mse"] is not None and r["rel_mse"] >= 0.5)]
    pts = [r for r in rows if r["agent_med"] is not None and r["sr"] is not None]
    return dict(all=rows, flagged=flagged, missed=missed, pts=pts,
                collapsed=[r for r in rows if r["collapsed"]],
                diverged=[r for r in rows if r["diverged"]])


def panel_detectors(ax, H):
    """C: effective_dim against median agent error, one point per run with a trace."""
    pts = H["pts"]
    # A clear lane above the data (max effective_dim is 43.5) and one below the zero rule
    # (nothing can sit under it) is what lets every label be a DIRECT label.
    xlo, xhi, ylo, yhi = 4.2, 11000.0, -10.5, 56.0
    x = np.array([r["agent_med"] for r in pts])
    y = np.array([r["effective_dim"] for r in pts])
    ok = np.array([r["sr"] > SR_OK for r in pts])
    gate = np.array([r["rel_mse"] is not None and r["rel_mse"] >= 0.5 for r in pts])

    # the arm-level bands the retraction box records, drawn as the arm-level claim they are
    ax.axvspan(*BRIEF_HEALTHY_PX, color=FILL["green"], alpha=0.85, lw=0, zorder=0)
    ax.axvspan(*BRIEF_FAILING_PX, color=FILL["crimson"], alpha=0.85, lw=0, zorder=0)
    ax.text(np.sqrt(BRIEF_HEALTHY_PX[0] * BRIEF_HEALTHY_PX[1]), ylo + 0.5,
            "every healthy ARM\n13–28 px", ha="center", va="bottom", fontsize=8.8,
            fontweight="bold", color=C["green"], linespacing=1.3)
    ax.text(np.sqrt(BRIEF_FAILING_PX[0] * BRIEF_FAILING_PX[1]), yhi - 1.4,
            "every failing ARM\n1000–2700 px", ha="center", va="top", fontsize=8.8,
            fontweight="bold", color=C["crimson"], linespacing=1.3)

    # the two detectors
    ax.axvline(DIVERGENCE_PX, color=C["amber"], lw=1.9, zorder=2)
    ax.axhline(0.0, color=C["amber"], lw=1.9, zorder=2)
    ax.text(DIVERGENCE_PX * 1.10, yhi - 1.4,
            "DETECTOR 2   median agent_pos_diff > 100 px\na diverging rollout — 5× "
            "the 20 px success radius",
            ha="left", va="top", fontsize=9.4, fontweight="bold", color=C["amber"],
            linespacing=1.35, path_effects=HALO)
    ax.text(xhi * 0.93, -0.9,
            "DETECTOR 1   effective_dim == 0  —  a constant-output encoder",
            ha="right", va="top", fontsize=9.4, fontweight="bold", color=C["amber"],
            path_effects=HALO)

    # the incumbent gate, as a ring: it fires only where a ring is drawn
    ax.scatter(x[gate], y[gate], s=132, facecolor="none", edgecolor=C["purple"],
               linewidth=1.5, zorder=4)
    for m, lab, key in ((ok, f"SR > {SR_OK:g}", "green"), (~ok, f"SR ≤ {SR_OK:g}", "crimson")):
        ax.scatter(x[m], y[m], s=34, color=C[key], alpha=0.85, edgecolor="white",
                   linewidth=0.5, zorder=5, label=lab)

    n_gate_pts = int(gate.sum())
    ax.text(xhi * 0.93, 34.0,
            f"purple ring  =  the pre-registered gate `err/rel_mse ≥ 0.5` fires\n"
            f"{n_gate_pts} of the {len(pts)} plotted runs",
            ha="right", va="center", fontsize=9.2, fontweight="bold", color=C["purple"],
            linespacing=1.35, path_effects=HALO)
    # x 30-110 above effective_dim 29 is the one gap in the cloud wide enough for the key.
    for yy, lab, key in ((40.5, f"SR > {SR_OK:g}   plans", "green"),
                         (35.8, f"SR ≤ {SR_OK:g}   does not", "crimson")):
        ax.scatter([58.0], [yy], s=34, color=C[key], edgecolor="white", linewidth=0.5,
                   zorder=5)
        ax.text(67.0, yy, lab, ha="left", va="center", fontsize=9.6,
                fontweight="bold", color=C[key], path_effects=HALO)

    n_dead = int((y <= 0).sum())
    worst = max(r["sr"] for r in pts if r["effective_dim"] <= 0)
    ax.annotate(f"{n_dead} runs at effective_dim == 0\nbest of them scores SR = {worst:.2f}",
                xy=(float(np.median(x[y <= 0])), 0.0), xytext=(430.0, -1.0),
                fontsize=9.2, fontweight="bold", color=C["amber"], ha="center", va="top",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=C["amber"], lw=1.0,
                                shrinkA=2, shrinkB=6))

    # top-left: the only rectangle of this panel with no runs in it at all.
    callout(ax, xlo * 1.14, yhi - 3.2,
            f"{len(H['flagged'])} of {len(H['all'])} finished runs are flagged by these two "
            f"checks.\n{len(H['missed'])} of them would NOT be caught by the pre-registered\n"
            "`rel_mse ≥ 0.5` gate: a constant-output model has rel_mse ≈ 0, and a\n"
            "one-step training loss cannot see a rollout leaving the table.",
            key="crimson", fontsize=9.4, ha="left", va="top", zorder=8)

    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    # ticks only where the quantity exists: the lane under the zero rule is label space,
    # and a "-10" on an axis that cannot go negative reads as a data range.
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xticks([10, 30, 100, 300, 1000, 3000, 10000])
    ax.set_xticklabels(["10", "30", "100", "300", "1000", "3000", "10000"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("median  agent_pos_diff  over the run's eval episodes  (px, log scale)",
                  fontsize=10, color=INK)
    ax.set_ylabel("sparsity/effective_dim", fontsize=10, color=INK)
    ax_style(ax, grid="both")
    panel_title(ax, "C",
                "The two detectors the campaign never had — and the gate that has both "
                "blind spots",
                f"one point per finished run with an eval trace ({len(pts)} of "
                f"{len(H['all'])}), coloured by whether it plans at all")


# --- the overlap audit ------------------------------------------------------------
def text_boxes(fig):
    """(owner, first line, bbox) for every visible piece of text, in device pixels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []


    def add(t, owner):
        if t is None or not t.get_visible() or not str(t.get_text()).strip():
            return
        bb = t.get_window_extent(renderer=r)
        items.append((owner, str(t.get_text()).split("\n")[0][:48],
                      Bbox.from_extents(bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2), bb))

    for i, ax in enumerate(fig.axes):
        ab = ax.get_window_extent()
        for t in list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]:
            add(t, f"ax{i}")
        # matplotlib keeps tick labels for ticks outside the view limits and does not draw
        # them; counting those as ink reports collisions that are not on the page.
        for t, axis in ([(t, "x") for t in ax.get_xticklabels()]
                        + [(t, "y") for t in ax.get_yticklabels()]):
            bb = t.get_window_extent(renderer=r)
            if axis == "x" and (bb.x1 < ab.x0 - 1 or bb.x0 > ab.x1 + 1):
                continue
            if axis == "y" and (bb.y1 < ab.y0 - 1 or bb.y0 > ab.y1 + 1):
                continue
            add(t, f"ax{i}")
    for t in fig.texts:
        add(t, "fig")
    return items


def audit(fig, slack=1.0):
    """Text pairs that intersect, data-coordinate labels across a spine, text off-canvas.

    "Nothing may overlap" is not checkable by eye at this size; this is what checks it.
    Only DATA-coordinate labels are tested against the spines -- panel titles and marginal
    labels live on axes/blended transforms and are meant to sit outside the data box.
    """
    items, bad = text_boxes(fig), []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ov = Bbox.intersection(items[i][2], items[j][2])
            if ov is not None and ov.width > slack and ov.height > slack:
                bad.append(f"OVERLAP  {items[i][0]} {items[i][1]!r}  x  "
                           f"{items[j][0]} {items[j][1]!r}   "
                           f"({ov.width:.0f}x{ov.height:.0f} px)")
    r = fig.canvas.get_renderer()
    for i, ax in enumerate(fig.axes):
        ab = ax.get_window_extent()
        for t in ax.texts:
            if t.get_transform() is not ax.transData:
                continue
            bb = t.get_window_extent(renderer=r)
            inside = ab.x0 - 1 <= bb.x0 and bb.x1 <= ab.x1 + 1
            outside = bb.x1 < ab.x0 or bb.x0 > ab.x1
            if not inside and not outside:
                bad.append(f"STRADDLE ax{i} {str(t.get_text())[:46]!r} crosses a spine "
                           f"(text {bb.x0:.0f}-{bb.x1:.0f}, axes {ab.x0:.0f}-{ab.x1:.0f})")
    fb = fig.get_window_extent()
    for owner, first, _pad, bb in items:
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
            bad.append(f"OFFCANVAS {owner} {first!r} ({bb.x0:.0f}-{bb.x1:.0f}, "
                       f"{bb.y0:.0f}-{bb.y1:.0f}) vs canvas "
                       f"({fb.x0:.0f}-{fb.x1:.0f}, {fb.y0:.0f}-{fb.y1:.0f})")
    return bad


# --- figure -----------------------------------------------------------------------
def fig_instrument(out, check=True):
    use_style()
    arms, detail, _ = collect()
    E = effects(arms)
    pop = population_sd(arms)
    decomp = [(fam, two_way(solo_matrix(arms, seeds))) for fam, seeds in FAMILIES]
    rho = cross_arm_rho(arms)
    H = health_rows(detail)

    fig = plt.figure(figsize=(18.6, 11.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.00, 1.24], height_ratios=[1.0, 0.80],
                          left=0.050, right=0.988, top=0.838, bottom=0.163,
                          wspace=0.26, hspace=0.50)
    ax_a = fig.add_subplot(gs[0, 0])
    gsb = gs[0, 1].subgridspec(2, 2, height_ratios=[1.0, 0.46], hspace=0.62, wspace=0.34)
    ax_b1 = fig.add_subplot(gsb[0, 0])
    ax_b2 = fig.add_subplot(gsb[0, 1])
    ax_bs = fig.add_subplot(gsb[1, :])
    ax_c = fig.add_subplot(gs[1, :])

    panel_resolution(ax_a, E, pop)
    heat(ax_b1, "A", FAMILIES[0][1], solo_matrix(arms, FAMILIES[0][1]), decomp[0][1],
         letter="B")
    heat(ax_b2, "B", FAMILIES[1][1], solo_matrix(arms, FAMILIES[1][1]), decomp[1][1])
    ss_bars(ax_bs, decomp, rho)
    panel_detectors(ax_c, H)

    fig.text(0.0, 0.996,
             "Why most of this campaign's nulls were unreadable",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.0, 0.958,
             "Six rounds moved the training objective and returned one positive. At the "
             "campaign's n = 8 floor a retrained-arm contrast could only ever have seen an "
             "effect LARGER THAN THE ENTIRE SPREAD of the arm population it was sampling — "
             "because the dominant\naxis of variation is the training seed, which pairing "
             "does not cancel. Two free checks would have caught both failure modes the "
             "campaign actually met; the pre-registered gate caught neither.",
             fontsize=10.4, color=MUTED, ha="left", va="top", linespacing=1.45)

    # The footer is WRAPPED by hand: at 8.2 pt one of these paragraphs is 5700 px wide
    # against a 2976 px canvas, and matplotlib will happily draw it off the page.
    def para(txt):
        return textwrap.fill(" ".join(txt.split()), width=WRAP)

    fig.text(0.0, 0.006, "\n".join(para(t) for t in (
        f"RECOMPUTED HERE from analysis.collect_evals.collect() (scheme=\"fixed\") and "
        f"{HEALTH_JSON}. B in full: PiWM-solo{{m}} is checkpoint LpWM-ltv_s{{m}} planned on "
        f"block s{{b}} (scripts/plan_consensus_controls.sh:93), so the dashed diagonal is "
        f"LpWM-ltv on its own block. The healthy-arm population sd is {pop['raw']:.4f} raw "
        f"over {pop['arms']} arms (mean >= 0.20, n >= 8) and {pop['corr']:.4f} once the "
        f"sampling variance of an arm mean is removed. Mean cross-arm correlation of "
        f"demeaned seed profiles: {rho['rho']:.4f} over {rho['pairs']} pairs of those arms "
        f"— so pairing cuts the sd of the difference by "
        f"{100 * (1 - np.sqrt(1 - rho['rho'])):.1f} %, not to zero. The three effects come "
        f"from analysis.figures.paired_effect: columns {num(E['columns']['mean'])} "
        f"[{num(E['columns']['lo'])}, {num(E['columns']['hi'])}] sd "
        f"{E['columns']['sd']:.3f} n = {E['columns']['n']}; patchdecode "
        f"{num(E['patchdecode']['mean'])} [{num(E['patchdecode']['lo'])}, "
        f"{num(E['patchdecode']['hi'])}] sd {E['patchdecode']['sd']:.3f} n = "
        f"{E['patchdecode']['n']}; consensus {num(E['consensus']['mean'])} "
        f"[{num(E['consensus']['lo'])}, {num(E['consensus']['hi'])}] sd "
        f"{E['consensus']['sd']:.3f} n = {E['consensus']['n']}. All of C is recomputed too.",

        f"CARRIED FROM THE ROUND'S NOTES, not recomputed: the 33 / 13 split of the "
        f"campaign's contrasts and their median paired sds 0.147 / 0.084 with MDEs "
        f"0.150 / 0.086 at n = 8 (README §7); the training-seed sd {BRIEF_SEED_SD:.3f} "
        f"against a binomial sd of {BRIEF_BINOM_SD:.3f}; and C's arm-level bands, which are "
        f"medians over POOLED episodes for the eight arms tabulated in the 2026-09-04 §7 "
        f"retraction box. Per-run points do fall between those bands — the no-overlap claim "
        f"is an ARM-level one and is drawn as one.",

        f"The MDE curves are the notes' n = 8 values scaled by 1/sqrt(n), not a second "
        f"power calculation, so a curve cannot disagree with the number printed on it. "
        f"Required n is priced at each contrast's OWN paired sd, which is why columns costs "
        f"n = 69 and not the ~35 the median-sd curve suggests: "
        f"n = ((z.975 + z.80)·sd/delta)^2 gives "
        f"{n_required(E['columns']['mean'], E['columns']['sd']):.0f} for columns, the notes' "
        f"69. The same formula gives "
        f"{n_required(E['patchdecode']['mean'], E['patchdecode']['sd']):.0f} for "
        f"patchdecode, not the 36 the notes record and this figure draws; that discrepancy "
        f"is unresolved and is marked rather than smoothed over.")),
        fontsize=8.2, color=MUTED, ha="left", va="bottom", linespacing=1.5)

    if check:
        for line in audit(fig):
            print("LAYOUT:", line)

    p = os.path.join(out, "instrument.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-05")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the text-overlap audit (it is on by default)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    print(fig_instrument(a.out, check=not a.no_check))


if __name__ == "__main__":
    main()
