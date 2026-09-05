"""Round 7's premise, in one figure: the cls latent cannot see the block's orientation.

    python analysis/round7_figs.py --out diary/assets/2026-09-05
    -> diary/assets/2026-09-05/orientation-probe.png

WHAT IS DRAWN, AND IN WHAT ORDER OF EMPHASIS

  A  The probe (2026-09-04 §7.4). Ridge-decode the block's angle from ONE FROZEN FRAME --
     no dynamics, no planner. Every checkpoint is a point; the box is the interquartile
     range; the bold tick is the median. The reference is `const_err_ang_deg` = 14.51 deg,
     the best achievable CONSTANT prediction, which is the bound that matters because
     PushT's T settles into a canonical orientation and 90 deg is NOT the right reference.
     The reading order down the panel is the argument's order:
       1. `LpWM-ltv` (cls, the baseline) sits at 15.72 deg -- WORSE than the bound. Every
          one of ~120 contrasts was built on this latent.
       2. `LpWM-ltv-d2048` (cls, 5x width) lands at 14.64 deg -- AT the bound. Capacity is
          not the constraint.
       3. `PiWM-columns` (patch tokens, a single-factor change from the baseline) reaches
          9.58 deg, the only arm below the bound -- and `PiWM-drop95`, ALSO patch but with
          95 % of its tokens discarded, falls back to 16.01 deg. So it is the TOKENS, not
          the word "patch".
     `ctrl_rand` -- the same encoder architecture, freshly initialised and never trained --
     is drawn per row so the reader can see the probe has headroom and is not saturated.

  B  Angular error against CEM success, one point per arm. This is FOUR ARMS and an
     OBSERVATIONAL association, and the panel says so on its face: §7.3 records three
     separate quantities that passed the screen on exactly this shape and then failed
     under intervention.

  C  The one clean single-factor CEM contrast, `PiWM-columns` vs `LpWM-ltv`. It is
     +0.072 [-0.064, +0.207] at n = 12 -- an interval that SPANS ZERO, so it is drawn with
     the null encoding (slate, hollow circle, no diamond, no bold), never as a positive.

DATA. Nothing here is typed in by hand.

  * `assets/latent_probe*.json` (13 shard files) for `err_ang_deg`, `const_err_ang_deg`,
    `ctrl_rand_err_ang_deg`, `arm`, `run`, `n_feat`, `n_dead`. Every run appears in exactly
    two shards; the duplicate objects are byte-identical (checked: 328 duplicate pairs, 0
    differing), so rows are deduplicated by `run` and a shard is never double-counted.
    `const_err_ang_deg` is identical (14.51303695789509) on all 328 runs -- it is a property
    of the evaluation set, not of an arm -- so it is drawn once, as a single line.
  * `analysis.collect_evals.collect()` for CEM, default `scheme="fixed"` (the only eval
    instrument valid for a comparison; see that module's note on the degenerate seed-0
    eval_seed).
  * `analysis.figures.paired_effect` for panel C's interval, reused rather than re-derived
    so it cannot drift from the other contrast figures.

  QUANTITIES COMPUTED HERE and stated as such, because they are not in the round's notes:
    - the interquartile ranges drawn in panel A (numpy linear interpolation);
    - the +-1 SE bars and the seed counts in panel B;
    - the Spearman rho over the four arms in panel B's caption (-0.80, p = 0.20, n = 4);
    - `LpWM-ltv-d2048`'s CEM mean, 0.587 over 6 seeds, and its paired contrast against the
      baseline, +0.020 [-1.464, +1.504] at n = 3 -- which is why the footer warns that the
      panel-B means are unpaired and over different seed sets;
    - the count of degenerate `PiWM-drop95` probes: 4 of 7 report n_dead == n_feat == 98304,
      i.e. every probe feature is constant, and those four share one intercept-only value.
      A tie that large would otherwise read as a suspiciously tidy cluster, so it is labelled.
  The four medians, the bound, the ctrl_rand values, the CEM means for columns/drop95/
  baseline and the +0.072 [-0.064, +0.207] contrast all reproduce the round's notes exactly.

STYLE. analysis/style.py. Hues by IDENTITY, never by rank:
    green   `LpWM-ltv`        -- the system: the representation the whole campaign was built on
    teal    `LpWM-ltv-d2048`  -- the same cls family, wider (style.py provides teal for exactly
                                 this: "a lighter green, for a second same-family series")
    amber   `PiWM-columns`    -- the intervention under test (encoder.feature: cls -> patch)
    purple  `PiWM-drop95`     -- the contrasting condition for that intervention (also patch,
                                 tokens removed), which is what makes it a dose control
    crimson  RESERVED for the alert -- the bound, the "worse than the bound" region, and the
             verdict text of the arms that fall in it. No arm owns crimson, so a crimson mark
             always means the same thing.
    slate    the null (panel C) and the untrained ctrl_rand reference.
  Re-running this after more seeds land re-orders nothing and re-paints nothing.

LAYOUT RULES, and the defect each one exists to stop

  * NO tight_layout. The row labels of panel A hang outside the axes on a blended transform
    and tight_layout cannot see them, so margins are set explicitly with subplots_adjust and
    the figure header/footer get reserved bands. (First draft: tight_layout pulled the axes
    left and the arm names ran off the canvas.)
  * Panel A gives each arm a band of height 1.0 and spends it from the centre outwards on a
    fixed budget: points within +-0.21, median tick +-0.28, the value at +0.36, the verdict
    at +0.545. Nothing is placed by eye, so no row can grow into its neighbour.
  * Ties are STACKED, not jittered. `_swarm` is deterministic -- points closer than TOL in x
    take the next free level -- so drop95's four identical 16.01 values draw as a visible
    column of four rather than one dot hiding three, and the figure is byte-reproducible.
  * `ctrl_rand` is a dashed vertical segment, not a marker on the row line. drop95 has a seed
    at 21.38 and its ctrl_rand is 21.45; a marker there landed on top of that point and read
    as a rendering fault rather than as the fact it is.
  * Panel B labels sit on a fixed above/below schedule per point, chosen from where the four
    points actually are, because d2048/baseline/drop95 are crowded into 1.4 deg of x.
  * Any label that has to cross the bound rule carries a white halo (HALO). The 1.7 pt
    crimson line showed through the letter gaps of "14.64 deg" and read as a printing fault.
  * Panel C hand-places its own subtitle. style.panel_title offsets the subtitle by 0.062 of
    the AXES height, which on a 1.2 in panel is under 6 pt, and title and subtitle printed on
    top of each other. style.py is imported, never edited -- other figures depend on it.
  * `audit()` runs on every render and prints any pair of text boxes that intersect and any
    label that straddles a spine. It is not decoration: "nothing may overlap" is not checkable
    by eye at 3800 px wide, and it found five collisions in this figure that a read of the
    PNG had already missed once. Panel C's right x-limit and panel A's top headroom are the
    values at which it stops complaining, not numbers chosen by taste.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import matplotlib.patheffects as pe                                    # noqa: E402
from matplotlib.transforms import Bbox                                 # noqa: E402
from scipy import stats                                                # noqa: E402

from analysis import figures as FG                                     # noqa: E402
from analysis.collect_evals import collect, resolve_arm                # noqa: E402
from analysis.style import (C, FILL, INK, MUTED, ax_style, callout,    # noqa: E402
                            panel_title, use_style)

PROBE_GLOB = "assets/latent_probe*.json"

#: A white halo for the labels that sit across the bound rule. Without it the 1.7 pt crimson
#: line shows through the letter gaps of "14.64°" and reads as a printing fault.
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]


def num(x, fmt="+.3f"):
    """Format with a real minus sign, so a drawn number matches the axis tick labels."""
    return format(x, fmt).replace("-", "−")

#: (arm, hue key, descriptor). Order is the ARGUMENT's order, top to bottom in panel A:
#: the baseline fails, width does not fix it, tokens do, removing tokens undoes it.
ARMS = [
    ("LpWM-ltv",       "green",  "cls  ·  the baseline"),
    ("LpWM-ltv-d2048", "teal",   "cls  ·  5× width"),
    ("PiWM-columns",   "amber",  "patch tokens  ·  cls → patch only"),
    ("PiWM-drop95",    "purple", "patch  ·  95 % of tokens dropped"),
]

#: Panel A's vertical budget, in row units. A row is 1.0 tall and these must stay inside it.
PT_SPAN, MED_SPAN, VAL_Y, VERDICT_Y, NOTE_Y = 0.21, 0.28, 0.36, 0.545, -0.40
#: x separation below which two points stack instead of sitting side by side, in degrees.
TOL = 0.35


# --- data ------------------------------------------------------------------------
def probe_rows(pattern=PROBE_GLOB):
    """Every probe row, deduplicated by `run`.

    The shards overlap: each run is written twice and the two objects are identical, so a
    naive concatenation doubles every n (26/12/32/14 instead of 13/6/16/7) without changing
    a single median -- which is exactly the kind of error a median hides.
    """
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for r in (d if isinstance(d, list) else [d]):
            if r.get("run") in seen:
                continue
            seen.add(r.get("run"))
            rows.append(r)
    return rows


def probe_table(rows):
    """Per-arm summary: vals, n, med, q1/q3, ctrl_rand, and the degenerate-probe bookkeeping.

    `n_dead_all` / `dead_val` / `live` split the runs into the probes that fitted something
    and the probes whose every feature was constant, which is the difference between a
    measurement and an intercept.
    """
    out = {}
    for arm, _, _ in ARMS:
        rs = [r for r in rows if r.get("arm") == arm]
        v = np.array(sorted(float(r["err_ang_deg"]) for r in rs))
        rand = sorted({round(float(r["ctrl_rand_err_ang_deg"]), 6) for r in rs})
        assert len(rand) == 1, f"{arm}: ctrl_rand is not constant within the arm: {rand}"
        # a probe whose every feature is constant has decoded nothing at all: its number
        # is an intercept-only fit, and several such runs report the SAME value.
        dead = sorted(float(r["err_ang_deg"]) for r in rs if r["n_dead"] == r["n_feat"])
        out[arm] = dict(
            vals=v, n=len(v), med=float(np.median(v)),
            q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
            rand=float(rand[0]), n_dead_all=len(dead),
            dead_val=dead[0] if dead else None,
            live=sorted(float(r["err_ang_deg"]) for r in rs if r["n_dead"] != r["n_feat"]),
            n_feat=int(rs[0]["n_feat"]),
        )
    return out


def const_bound(rows):
    """The best achievable CONSTANT angular prediction. One number, identical on every run."""
    v = {round(float(r["const_err_ang_deg"]), 9) for r in rows if "const_err_ang_deg" in r}
    assert len(v) == 1, f"const_err_ang_deg is not constant across runs: {sorted(v)}"
    return float(next(iter(v)))


def cem_table(arms_dict):
    """{arm: dict(vals, mean, se, n)} from collect(), fixed eval scheme only."""
    out = {}
    for arm, _, _ in ARMS:
        v = np.array(sorted(float(x) for x in arms_dict[resolve_arm(arms_dict, arm)].values()))
        out[arm] = dict(vals=v, n=len(v), mean=float(v.mean()),
                        se=float(v.std(ddof=1) / np.sqrt(len(v))))
    return out


# --- drawing helpers -------------------------------------------------------------
def _swarm(vals, tol=TOL, step=0.105):
    """Deterministic stacking offsets: points within `tol` in x take the next free level.

    Not a random jitter. Ties must be VISIBLE (drop95 has four identical values) and the
    figure must render identically every time, which rules out an RNG.
    """
    levels = [0]
    while len(levels) < 24:
        k = len(levels) // 2 + 1
        levels += [k, -k]
    off, placed = np.zeros(len(vals)), []
    for i in np.argsort(np.asarray(vals), kind="stable"):
        x = float(vals[i])
        used = {lv for xx, lv in placed if abs(xx - x) < tol}
        lv = next(l for l in levels if l not in used)
        placed.append((x, lv))
        off[i] = lv * step
    return off


def text_boxes(fig):
    """(owner, first line, bbox) for every visible piece of text, in device pixels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []

    def add(t, owner):
        if t is None or not t.get_visible() or not str(t.get_text()).strip():
            return
        bb = t.get_window_extent(renderer=r)
        bb = Bbox.from_extents(bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2)   # ink padding
        items.append((owner, str(t.get_text()).split("\n")[0][:46], bb))

    for i, ax in enumerate(fig.axes):
        for t in (list(ax.texts) + list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                  + [ax.xaxis.label, ax.yaxis.label]):
            add(t, f"ax{i}")
    for t in fig.texts:
        add(t, "fig")
    return items


def audit(fig, slack=1.0):
    """Every pair of text boxes that intersect, and every label straddling a spine.

    This exists because "nothing may overlap" cannot be checked by eye at 3800 px wide --
    the first draft of this figure had five text collisions and a label across a spine, and
    a read of the rendered PNG had already caught only one of them.
    """
    items, bad = text_boxes(fig), []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ov = Bbox.intersection(items[i][2], items[j][2])
            if ov is not None and ov.width > slack and ov.height > slack:
                bad.append(f"OVERLAP  {items[i][0]} {items[i][1]!r}  x  "
                           f"{items[j][0]} {items[j][1]!r}   "
                           f"({ov.width:.0f}x{ov.height:.0f} px)")
    for i, ax in enumerate(fig.axes):
        ab = ax.get_window_extent()
        for t in ax.texts:
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inside = ab.x0 - 1 <= bb.x0 and bb.x1 <= ab.x1 + 1
            outside = bb.x1 < ab.x0 or bb.x0 > ab.x1
            if not inside and not outside:
                bad.append(f"STRADDLE ax{i} {str(t.get_text())[:46]!r} crosses a spine "
                           f"(text {bb.x0:.0f}-{bb.x1:.0f}, axes {ab.x0:.0f}-{ab.x1:.0f})")
    return bad


# --- panels ----------------------------------------------------------------------
def panel_probe(ax, P, bound):
    """A: the probe. Rows top-to-bottom follow the argument, not the ranking."""
    n_rows = len(ARMS)
    xlo, xhi = 6.3, 28.3
    ax.axvspan(bound, xhi, color=FILL["crimson"], alpha=0.55, lw=0, zorder=0)
    ax.axvline(bound, color=C["crimson"], lw=1.7, zorder=3)

    for i, (arm, key, desc) in enumerate(ARMS):
        y = n_rows - 1 - i                      # row 0 of ARMS at the TOP
        d = P[arm]
        # interquartile box, behind the points
        ax.plot([d["q1"], d["q3"]], [y, y], color=FILL[key], lw=13,
                solid_capstyle="butt", zorder=1)
        ax.plot([d["q1"], d["q3"]], [y, y], color=C[key], lw=0.9, alpha=0.45, zorder=1)
        # every checkpoint, ties stacked
        off = _swarm(d["vals"])
        # the vertical budget is a contract, not a hope: a deeper stack would reach into the
        # median tick and then into the row above.
        assert np.abs(off).max() <= PT_SPAN + 1e-9, f"{arm}: stack exceeds PT_SPAN"
        ax.scatter(d["vals"], y + off, s=34, color=C[key], alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=4)
        # the median
        ax.plot([d["med"], d["med"]], [y - MED_SPAN, y + MED_SPAN], color=C[key],
                lw=3.0, solid_capstyle="round", zorder=5)
        ax.text(d["med"], y + VAL_Y, f"{d['med']:.2f}°", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=C[key], path_effects=HALO)
        # the verdict against the bound: crimson ONLY when the arm is beaten by a constant
        delta = d["med"] - bound
        if delta < -0.5:
            # left-aligned at the panel edge: this verdict is the long one, and centring it
            # on a median of 9.58 pushed its first characters off the left spine.
            vtxt, vcol, vx, vha = (f"−{abs(delta):.2f}° — the ONLY arm below the bound",
                                   C[key], xlo + 0.3, "left")
        elif delta < 0.5:
            vtxt, vcol, vx, vha = (f"+{delta:.2f}° — AT the bound",
                                   C["slate"], d["med"], "center")
        else:
            vtxt, vcol, vx, vha = (f"+{delta:.2f}° — WORSE than the bound",
                                   C["crimson"], d["med"], "center")
        ax.text(vx, y + VERDICT_Y, vtxt, ha=vha, va="bottom",
                fontsize=9.2, fontweight="bold", color=vcol, path_effects=HALO)
        # ctrl_rand: a reference, not a series -- dashed segment so it cannot be misread
        # as a median, and so drop95's 21.38 seed does not collide with a marker at 21.45.
        ax.plot([d["rand"]] * 2, [y - 0.30, y + 0.30], color=C["slate"], lw=1.4,
                ls=(0, (3, 2)), zorder=3)
        ax.text(d["rand"], y + VAL_Y, f"random encoder {d['rand']:.2f}°", ha="center",
                va="bottom", fontsize=8.2, color=C["slate"])
        # row label, outside the axes on a blended transform
        tr = ax.get_yaxis_transform()
        ax.text(-0.012, y + 0.10, arm, transform=tr, ha="right", va="bottom",
                fontsize=11, fontweight="bold", color=C[key])
        ax.text(-0.012, y - 0.10, f"{desc}   ·   n = {d['n']}", transform=tr, ha="right",
                va="top", fontsize=8.6, color=MUTED)
        # the degenerate probes, labelled where they sit
        if d["n_dead_all"]:
            ax.text(d["med"] + 0.35, y + NOTE_Y,
                    f"{d['n_dead_all']} of {d['n']} probes: every feature dead "
                    f"(n_dead = n_feat = {d['n_feat']}), one shared value",
                    ha="left", va="center", fontsize=8.2, color=MUTED)

    ax.text(bound + 0.45, n_rows - 0.15,
            "WORSE THAN THE BOUND — beaten by one constant angle",
            ha="left", va="bottom", fontsize=9.6, fontweight="bold", color=C["crimson"])
    # two lines, not one: on one line this label is wider than the space between the
    # bound and the left spine, and it ran outside the axes.
    ax.text(bound - 0.45, n_rows - 0.15,
            f"best achievable CONSTANT\nprediction   {bound:.2f}°",
            ha="right", va="bottom", fontsize=9.6, fontweight="bold", color=C["crimson"],
            linespacing=1.35)
    ax.set_xlim(xlo, xhi)
    # the top band holds the two-line bound label; without the headroom it grew into the
    # panel subtitle, which sits immediately above the axes.
    ax.set_ylim(-0.66, n_rows + 0.34)
    ax.set_yticks([])
    ax.set_xticks(np.arange(8, 29, 2))
    ax.set_xlabel("median absolute angular error of a ridge decode of block orientation,\n"
                  "from ONE FROZEN FRAME (degrees)   —   lower is better",
                  fontsize=10, color=INK)
    ax_style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    panel_title(ax, "A", "The baseline's latent cannot see the block's angle; patch tokens can",
                "one point per checkpoint  ·  box = interquartile range  ·  bold tick = "
                "median  —  PiWM-columns' box clears the baseline's entirely")


def panel_assoc(ax, P, E, bound, rho, p_rho):
    """B: the observational association, labelled as one."""
    xlo, xhi = 7.6, 18.9
    ax.axvspan(bound, xhi, color=FILL["crimson"], alpha=0.55, lw=0, zorder=0)
    ax.axvline(bound, color=C["crimson"], lw=1.4, ls=(0, (4, 2)), zorder=1)
    ax.text(bound - 0.25, 0.862, f"constant bound {bound:.2f}°", ha="right", va="top",
            fontsize=8.6, color=C["crimson"], fontweight="bold", path_effects=HALO)
    # above/below is fixed per point: d2048, the baseline and drop95 live inside 1.4 deg of x
    # (y of the label, x offset, alignment). d2048's median is 0.13 deg off the bound, so a
    # centred label printed across the bound's dashed line; it is pushed to the right of it.
    place = {"LpWM-ltv": (0.268, 0.0, "center"), "LpWM-ltv-d2048": (0.735, 0.28, "left"),
             "PiWM-columns": (0.512, 0.0, "center"), "PiWM-drop95": (0.058, 0.0, "center")}
    for arm, key, _ in ARMS:
        x, e = P[arm]["med"], E[arm]
        ax.plot([x, x], [e["mean"] - e["se"], e["mean"] + e["se"]], color=C[key],
                lw=2.0, solid_capstyle="round", zorder=3)
        ax.scatter([x], [e["mean"]], s=105, color=C[key], zorder=4,
                   edgecolor="white", linewidth=0.9)
        ty, dx, ha = place[arm]
        ax.text(x + dx, ty, f"{arm}   {e['mean']:.3f}  (n = {e['n']})", ha=ha,
                va="bottom" if ty > e["mean"] else "top",
                fontsize=9, fontweight="bold", color=C[key], path_effects=HALO)
    # Four lines, not six. The long form ran into PiWM-drop95's direct label; the detail
    # it carried now lives in the figure footer, which has the full canvas width.
    callout(ax, xlo + 0.25, 0.035,
            "4 arms. OBSERVATIONAL — nothing was intervened on.\n"
            f"Spearman ρ = {num(rho, '+.2f')} (p = {p_rho:.2f}, n = 4), and NOT monotone:\n"
            "d2048 has the best CEM mean yet sits AT the bound.\n"
            "§7.3: this shape failed under intervention 3 times.",
            key="slate", fontsize=8.2, ha="left", va="bottom")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.045, 0.89)
    ax.set_xlabel("median angular error (degrees)", fontsize=10, color=INK)
    ax.set_ylabel("CEM success   (mean over seeds, ±1 SE)", fontsize=10, color=INK)
    ax_style(ax, grid="both")
    panel_title(ax, "B", "Probe against planning — an association, not a lever",
                "four arms, observational; the footer records what happened last time")


def panel_contrast(ax, eff):
    """C: the single-factor CEM contrast, with the NULL encoding it has earned."""
    lo, hi, m, n = eff["lo"], eff["hi"], eff["mean"], eff["n"]
    spans_zero = lo < 0 < hi
    col = C["slate"] if spans_zero else C["amber"]
    ax.axvspan(-0.40, 0.0, color="#F4F4F4", lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.plot([lo, hi], [0, 0], color=col, lw=2.6, solid_capstyle="round", zorder=3)
    for b in (lo, hi):
        ax.plot([b, b], [-0.13, 0.13], color=col, lw=1.6, zorder=3)
    # hollow circle, never a filled diamond: the diamond is this project's "excludes zero"
    ax.scatter([m], [0], s=110, facecolor="white", edgecolor=col, linewidth=2.0, zorder=4)
    ax.text(hi + 0.04, 0.10, f"{num(m)}  [{num(lo)}, {num(hi)}]   n = {n}", ha="left",
            va="bottom", fontsize=9.6, fontweight="bold", color=col)
    ax.text(hi + 0.04, -0.10, "the interval SPANS ZERO — a null, not a positive", ha="left",
            va="top", fontsize=9, fontweight="bold", color=col)
    # The "worse / better than the baseline" hints that used to sit here are folded into
    # the axis label instead: this panel is 1.2 in tall, and no matter which edge they were
    # parked against they touched either the header or the "spans zero" line.
    # right edge chosen so the widest annotation ends inside the spine: the audit reports
    # a straddle at 0.90, which is how this number was picked rather than guessed.
    ax.set_xlim(-0.40, 0.98)
    ax.set_ylim(-1.20, 0.98)
    ax.set_yticks([])
    ax.set_xlabel("paired Δ CEM success, PiWM-columns − LpWM-ltv   (arm specs identical "
                  "except encoder.feature; shaded = worse than the baseline)",
                  fontsize=9.5, color=INK)
    ax_style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    # style.panel_title puts its subtitle 0.062 of the AXES height under the title, which on
    # a panel this short is under 6 pt -- the two printed on top of each other. The title is
    # raised and the subtitle placed by hand at a distance measured in axes height.
    panel_title(ax, "C", "The one clean single-factor contrast", y=1.34)
    ax.text(0.0, 1.15, "the other two arms share too few seeds with the baseline to "
                       "contrast at all", transform=ax.transAxes, fontsize=9.5,
            color=MUTED, va="bottom", ha="left")


# --- figure ----------------------------------------------------------------------
def fig_orientation_probe(out, check=True):
    use_style()
    rows = probe_rows()
    P, bound = probe_table(rows), const_bound(rows)
    arms_dict = collect()[0]
    E = cem_table(arms_dict)
    rho, p_rho = stats.spearmanr([P[a]["med"] for a, _, _ in ARMS],
                                 [E[a]["mean"] for a, _, _ in ARMS])
    eff = FG.paired_effect(arms_dict, "LpWM-ltv", resolve_arm(arms_dict, "PiWM-columns"))
    d2048 = FG.paired_effect(arms_dict, "LpWM-ltv", resolve_arm(arms_dict, "LpWM-ltv-d2048"))

    fig = plt.figure(figsize=(17.4, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0], height_ratios=[1.0, 0.30])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    # Explicit margins, no tight_layout: panel A's row labels hang outside its axes on a
    # blended transform and tight_layout, which measures only what is inside them, both
    # crops them and steals the bands reserved for the header and the footer.
    fig.subplots_adjust(left=0.135, right=0.988, top=0.812, bottom=0.155,
                        wspace=0.235, hspace=1.05)

    panel_probe(ax_a, P, bound)
    panel_assoc(ax_b, P, E, bound, rho, p_rho)
    panel_contrast(ax_c, eff)

    fig.text(0.0, 0.995,
             "Round 7's premise: the representation cannot represent the variable PushT "
             "is about",
             fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.0, 0.945,
             "Ridge decode of the block's angle from ONE FROZEN FRAME — no dynamics, no "
             "planner. Privileged state is used for MEASUREMENT only, never in a loss.\n"
             f"The reference is the best achievable CONSTANT prediction, {bound:.2f}°, not "
             "90°: PushT's T settles into a canonical orientation, so the angle is strongly "
             "non-uniform.",
             fontsize=9.8, color=MUTED, ha="left", va="top")

    # No legend. Both reference marks are already labelled where they are drawn, and a
    # legend block under panel A landed on top of this footer.
    dp = P["PiWM-drop95"]
    fig.text(0.0, 0.010,
             "Probe: assets/latent_probe*.json, 13 shards deduplicated by `run` — every run "
             "is written to exactly two shards and the duplicates are byte-identical. "
             "const_err_ang_deg is the same value on all 328 runs, so it is drawn once. "
             "The dashed slate lines are ctrl_rand: the same encoder architecture, freshly "
             "initialised and never trained.\n"
             f"PiWM-drop95: {dp['n_dead_all']} of its {dp['n']} probes report "
             f"n_dead = n_feat = {dp['n_feat']} — every feature constant — and share one "
             f"intercept-only value, {dp['dead_val']:.2f}°; the "
             f"{len(dp['live'])} that did fit read "
             + ", ".join(f"{v:.2f}°" for v in dp["live"])
             + f", at or above its own untrained control ({dp['rand']:.2f}°).\n"
               "CEM: analysis.collect_evals.collect(), fixed eval scheme. Panel B's means are "
               "UNPAIRED and over different seed sets (n = "
             + " / ".join(f"{E[a]['n']} for {a}" for a, _, _ in ARMS)
             + f"); LpWM-ltv-d2048 shares only {d2048['n']} seeds with the baseline, so its "
               f"paired contrast is {num(d2048['mean'])} [{num(d2048['lo'])}, "
               f"{num(d2048['hi'])}] — no information, and it is not drawn in C.\n"
               "§7.3, the reason panel B is labelled the way it is: three quantities were "
               "selected on exactly that shape and then failed under intervention — "
               "d_action / h8·h1 rejected by the screen, S_model −0.260 at a healthy dose, "
               "K = 5 rollout error −0.200 across a 33× range of the quantity.",
             fontsize=8.4, color=MUTED, ha="left", va="bottom")

    if check:
        for line in audit(fig):
            print("LAYOUT:", line)

    p = os.path.join(out, "orientation-probe.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-05")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the text-overlap audit (it is on by default)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    print(fig_orientation_probe(a.out, check=not a.no_check))


if __name__ == "__main__":
    main()
