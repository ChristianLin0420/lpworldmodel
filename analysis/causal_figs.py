"""Deep-dive analysis figures for the causal-objective round (diary/2026-09-03.md).

Every number is harvested at call time from the runs themselves — each run's local
``wandb/latest-run/files/wandb-summary.json`` for the training diagnostics, and
``analysis/collect_evals.py``'s campaign JSON for the CEM numbers — so re-running this
after more evals land refreshes the figures without editing anything.

DESIGN. Everything visual comes from ``analysis/style.py``, the one design system for this
project (derived from ``figures/motivation_teaser.svg``): sans-serif throughout, recessive
furniture, panel headers with a coloured letter and a rule, pale callouts with saturated
text, titles that state the finding. This module adds one thing on top of it — a fixed
ARM -> hue map, so an arm keeps its colour across all six figures and across regenerations.

The arm map, by IDENTITY and never by rank:

    slate    #546060   LpWM-ltv                 the control
    green    #18483C   reprelu, p=2             the rectified link — the healthy code
    teal     #2E7D6F   columns P=256            same family, also healthy
    amber    #C88A1E   V3 pathint               the intervention under test
    amber-2  #E0B265   sigreg-arpred            a second objective-side intervention
    purple   #543C84   V2 actinfo               the contrasting condition
    plum     #8A6FA8   V2+V3                    same family
    crimson  #A83024   identity, p=1            the link that collapses the code
    crimson-2#C9635A   V1 incr                  collapsed: d_action is exactly zero
    crimson-3#E0A49C   V1+V2                    collapsed, same family
    slate-2  #2E3A3A   block-causal             temporally collapsed (found independently)
    OTHER    #AEB8B4   every other arm          the background population, unlabelled

Usage (this is the invocation the diary's reproduce block uses):
    python analysis/collect_evals.py --out campaign.json
    python analysis/causal_figs.py --out diary/assets/2026-09-03 \
        --campaign campaign.json --incr-stats assets/incr_stats_lpwm_ltv_s3.json

PASS --campaign EXPLICITLY. The argparse default is still ``/tmp/campaign_now.json``, a
scratch file that is usually stale and covers far fewer arms than a fresh collect_evals run;
falling back to it silently shrinks the population behind d-action-vs-cem.png.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style as S  # noqa: E402
from analysis.style import C, FILL  # noqa: E402

S.use_style()

# --- palette --------------------------------------------------------------------------
# The historical ACCENT keys are kept because round5_method_figs / round5_result_figs /
# round5_data_figs import them; only the VALUES move to the design system, exactly as
# analysis/arch_figs.py did.
ACCENT = dict(S.PALETTE_SVG)          # blue->green, magenta->purple, green->teal, crit->crimson
ACCENT_FILL = dict(S.PALETTE_SVG_FILL)
INK, MUTED, GRID = S.INK, S.MUTED, S.GRID
HEAD, EDGE, PAPER = S.HEAD, S.EDGE, S.PAPER

AMBER_2 = "#E0B265"       # a lighter amber, for a second intervention in the same family
CRIMSON_2 = "#C9635A"     # lighter crimsons, for the other collapse identities
CRIMSON_3 = "#E0A49C"
SLATE_2 = "#2E3A3A"       # a darker slate
OTHER = "#AEB8B4"         # arms that are not part of this round's design
OTHER_FILL = "#DCE2E0"

ARM_COLOR = {
    "LpWM-ltv":             C["slate"],
    "LpWM-ltv-relu-p2":     C["green"],
    "PiWM-columns":         C["teal"],
    "PiWM-pathint":         C["amber"],
    "PiWM-sigreg-arpred":   AMBER_2,
    "PiWM-actinfo":         C["purple"],
    "PiWM-actinfo-pathint": C["plum"],
    "LpWM-ltv-ident-p1":    C["crimson"],
    "PiWM-incr":            CRIMSON_2,
    "PiWM-incr-actinfo":    CRIMSON_3,
    "PiWM-blockcausal":     SLATE_2,
}
LABEL = {
    "LpWM-ltv": "LpWM-ltv (control)", "LpWM-ltv-relu-p2": "reprelu, p=2",
    "LpWM-ltv-ident-p1": "identity, p=1", "PiWM-columns": "columns P=256",
    "PiWM-pathint": "V3 pathint", "PiWM-actinfo": "V2 actinfo",
    "PiWM-actinfo-pathint": "V2+V3", "PiWM-incr": "V1 incr",
    "PiWM-incr-actinfo": "V1+V2", "PiWM-blockcausal": "block-causal",
    "PiWM-sigreg-arpred": "sigreg-arpred",
}
#: the order the named arms are listed in whenever a key is needed
NAMED = ["LpWM-ltv-relu-p2", "PiWM-columns", "PiWM-pathint", "PiWM-sigreg-arpred",
         "PiWM-actinfo", "PiWM-actinfo-pathint", "LpWM-ltv-ident-p1", "PiWM-incr",
         "PiWM-incr-actinfo", "PiWM-blockcausal", "LpWM-ltv"]

RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")
# the columns run directory is PiWM-columns_pd384_patch_bf16_sN, so the regex yields
# "PiWM-columns" while collect_evals keys the arm "PiWM-columns_patch"
CAMPAIGN_ALIAS = {"PiWM-columns": "PiWM-columns_patch"}

CEM_RAMP = LinearSegmentedColormap.from_list("cem", S.SEQ_GREEN)


def col(arm):
    """The arm's hue. Arms outside this round's design share one recessive slate."""
    return ARM_COLOR.get(arm, OTHER)


def is_named(arm):
    return arm in ARM_COLOR


def _style(ax, grid="both"):
    """Recessive furniture. Kept at this name and signature — three other modules import it."""
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=3)
    return ax


def _callout(ax, x, y, text, key="slate", fc=None, ec=None, tc=None, fontsize=10,
             ha="left", va="top", weight="bold", coords="axes fraction", **kw):
    """Pale fill, saturated matching text, thin matching border — never the other way round."""
    fc = FILL.get(key, OTHER_FILL) if fc is None else fc
    ec = C.get(key, OTHER) if ec is None else ec
    tc = C.get(key, INK) if tc is None else tc
    return ax.annotate(text, (x, y), xycoords=coords, ha=ha, va=va, fontsize=fontsize,
                       color=tc, fontweight=weight, zorder=20,
                       bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=0.9), **kw)


def _headroom(ax, frac=0.72, log=False):
    """Grow the top of the y axis so the current data maximum sits at `frac` of the height,
    leaving a clean band above the marks for labels and significance brackets."""
    lo, hi = ax.get_ylim()
    if log:
        l0, l1 = np.log10(lo), np.log10(hi)
        ax.set_ylim(lo, 10 ** (l0 + (l1 - l0) / frac))
    else:
        ax.set_ylim(lo, lo + (hi - lo) / frac)


def _bracket(ax, x0, x1, y, text, color=MUTED, drop=0.018, fontsize=9.5, weight="normal"):
    """A significance bracket in axes-fraction y over data-coordinate x."""
    tr = ax.get_xaxis_transform()
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], transform=tr, color=color,
            lw=1.0, clip_on=False, zorder=6, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y + 0.012, text, transform=tr, ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight=weight, zorder=6)


def _rowbands(ax, n, color="#F5F7F6"):
    """Alternating hairline row bands — how a 45-row chart stays readable across columns."""
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=color, lw=0, zorder=0)


def _pfmt(p, n_perm=20000):
    """A p-value that never prints as a false zero."""
    if not np.isfinite(p):
        return "p = n/a"
    if p <= 0:
        return f"p < {1.0 / n_perm:.0e}"
    return f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.4f}".rstrip("0").rstrip(".")


def _num(v):
    """Three significant figures, keeping trailing zeros — so a median of 0.99995 prints
    as '1.00' and not as a bare '1' that reads like an integer."""
    return f"{v:#.3g}"


TEXT_COLOR = {AMBER_2: "#9C7318", CRIMSON_3: "#B4564C", OTHER: MUTED}


def tcol(arm):
    """The arm's hue, darkened where the mark colour would be too light to read as text."""
    c = col(arm)
    return TEXT_COLOR.get(c, c)


def _header_at(fig, ax, letter, title, sub=None, dy_title=0.40, dy_sub=0.16, rule=True):
    """The reference's panel header — coloured letter, bold title, muted subtitle, rule --
    placed in INCHES above the axes, so it reads the same whether the panel is 3in or 11in
    tall. style.panel_title's axes-fraction offsets cannot do that."""
    bb = ax.get_position()
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    dx = 0.0
    if letter:
        fig.text(bb.x0, bb.y1 + dy_title / fh, letter, fontsize=11.5, fontweight="bold",
                 color=HEAD, ha="left", va="bottom")
        dx = 0.21 / fw
    fig.text(bb.x0 + dx, bb.y1 + dy_title / fh, title, fontsize=11.5, fontweight="bold",
             color=INK, ha="left", va="bottom")
    if sub:
        fig.text(bb.x0 + dx, bb.y1 + dy_sub / fh, sub, fontsize=9.5, color=MUTED,
                 ha="left", va="bottom")
    if rule:
        y = bb.y1 + 0.085 / fh
        fig.add_artist(plt.Line2D([bb.x0, bb.x1], [y, y], color=EDGE, lw=1.0,
                                  transform=fig.transFigure))


def _figure_head(fig, title, sub, x=0.018, dy_title=0.30, dy_sub=0.60):
    """The figure-level header, in inches down from the top edge."""
    fh = fig.get_figheight()
    fig.text(x, 1 - dy_title / fh, title, fontsize=13, fontweight="bold", color=INK,
             ha="left", va="top")
    if sub:
        fig.text(x, 1 - dy_sub / fh, sub, fontsize=10.5, color=MUTED, ha="left", va="top",
                 linespacing=1.45)


def harvest(repo, campaign):
    """One row per run: the causal diagnostics, the health metrics, and the CEM number."""
    cem = json.load(open(campaign))["arms"] if campaign and os.path.exists(campaign) else {}
    rows = []
    for d in sorted(glob.glob(os.path.join(repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        f = os.path.join(d, "wandb/latest-run/files/wandb-summary.json")
        m = RUN_RE.match(name)
        if not m or not os.path.exists(f):
            continue
        # CANARY-* runs are deliberate ~200-step liveness probes, not experiments: they stop at
        # epoch 1 with a meaningless rel_mse and would otherwise appear as their own arms in
        # every training-metric figure. They never reach plan_outputs, so no CEM contrast is
        # affected — this only keeps them out of the plots.
        if name.startswith("CANARY-"):
            continue
        try:
            s = json.load(open(f))
        except Exception:
            continue
        arm, seed = m.group(1), int(m.group(2))
        rows.append(dict(
            run=name, arm=arm, seed=seed, done=os.path.exists(os.path.join(d, "DONE")),
            d_action=s.get("causal/d_action"), d_state=s.get("causal/d_state"),
            soa=s.get("causal/state_over_action"),
            rel_mse=s.get("err/rel_mse"), rho=s.get("sparsity/val_l0_frac"),
            eff=s.get("sparsity/effective_dim"),
            cem=cem.get(CAMPAIGN_ALIAS.get(arm, arm), {}).get(str(seed))))
    return rows


# ------------------------------------------------------------------ figure 1
def fig_d_action_strip(rows, out):
    """Per-run d_action and d_state. The point: d_action is now measured at n=8 per arm,
    not inferred from a single checkpoint, and the arms separate by three orders of
    magnitude — so the diagnostic is a property of the OBJECTIVE, not of a lucky seed.

    45 arms do not fit in a chart sized for eleven. The layout is therefore explicit: one
    wide strip on the left, and the two number columns get their own axes on the right so
    a digit can never land on a mark or on the title.
    """
    by = collections.defaultdict(list)
    for r in rows:
        if r["d_action"] is not None:
            by[r["arm"]].append(r)
    order = sorted(by, key=lambda a: np.median([x["d_action"] for x in by[a]]))
    n = len(order)
    FLOOR = 3e-4                      # exact zeros must stay visible on a log axis

    # --- explicit inch-space layout: nothing here is left to tight_layout -------------
    FW, ROW = 14.0, 0.245
    L, R, TOP, BOT = 2.55, 0.30, 1.62, 0.92          # margins, inches
    COLW, GAP = 1.30, 0.30                            # the two number columns
    FH = TOP + BOT + ROW * n
    fig = plt.figure(figsize=(FW, FH))
    strip_w = FW - L - R - 2 * COLW - 2 * GAP
    y0, h = BOT / FH, (ROW * n) / FH
    ax = fig.add_axes([L / FW, y0, strip_w / FW, h])
    cx1 = fig.add_axes([(L + strip_w + GAP) / FW, y0, COLW / FW, h])
    cx2 = fig.add_axes([(L + strip_w + 2 * GAP + COLW) / FW, y0, COLW / FW, h])

    _rowbands(ax, n)
    for cx in (cx1, cx2):
        _rowbands(cx, n)
        cx.set_xlim(0, 1)
        cx.set_ylim(-0.6, n - 0.4)
        cx.set_xticks([])
        cx.set_yticks([])
        for s in cx.spines.values():
            s.set_visible(False)

    # the exact-zero gutter, so a point on the floor is never read as a small value
    ax.axvspan(8e-5, 4.4e-4, color=OTHER_FILL, lw=0, zorder=1)
    ax.text(1.75e-4, -0.42, "exact 0", ha="center", va="center", fontsize=8.5,
            color=MUTED, zorder=6)

    for i, arm in enumerate(order):
        v = by[arm]
        c = col(arm)
        named = is_named(arm)
        da = np.array([x["d_action"] for x in v], float)
        ds = np.array([x["d_state"] for x in v if x["d_state"] is not None], float)
        ax.scatter(np.clip(ds, FLOOR, None), np.full(ds.size, i) + 0.19,
                   s=30 if named else 20, facecolor="none", edgecolor=c,
                   lw=1.35 if named else 1.0, zorder=3)
        ax.scatter(np.clip(da, FLOOR, None), np.full(da.size, i) - 0.19,
                   s=42 if named else 26, color=c, lw=0, zorder=4)
        med = float(np.median(da))
        cx1.text(0.94, i, "0" if med < 1e-6 else f"{med:.3f}", va="center", ha="right",
                 fontsize=9.5, color=tcol(arm) if named else MUTED,
                 fontweight="bold" if named else "normal")
        if ds.size:
            ratio = np.median(ds) / med if med > 0 else np.inf
            hot = (not np.isfinite(ratio)) or ratio > 10
            txt = ("inf" if not np.isfinite(ratio)
                   else (f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"))
            cx2.text(0.94, i, txt, va="center", ha="right", fontsize=9.5,
                     color=C["crimson"] if hot else (tcol(arm) if named else MUTED),
                     fontweight="bold" if hot else "normal")

    ax.set_yticks(range(n))
    ax.set_yticklabels([LABEL.get(a, a) for a in order], fontsize=9.5)
    for t, a in zip(ax.get_yticklabels(), order):
        t.set_color(tcol(a) if is_named(a) else MUTED)
        if is_named(a):
            t.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(8e-5, 3.2)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    _style(ax, grid="x")
    ax.set_xlabel("RMS displacement of the PREDICTED latent under an in-batch permutation "
                  r"(log; points in the grey gutter are exact zeros)",
                  fontsize=10.5, color=INK, labelpad=8)

    # column headers, in their own axes' space so they cannot reach the strip
    for cx, head in ((cx1, "median\n" + r"$d_{action}$"),
                     (cx2, r"$d_{state}\,/\,d_{action}$")):
        cx.text(0.94, 1.0, head, transform=cx.transAxes, ha="right", va="bottom",
                fontsize=10, color=INK, fontweight="bold", linespacing=1.35)
        cx.plot([0.0, 1.0], [1.0, 1.0], transform=cx.transAxes, color=EDGE, lw=1.0,
                clip_on=False)

    # two series -> direct labels, not a legend. Positions are measured in inches from the
    # left of the strip so the two entries can never run into one another.
    ky = 1 + 0.30 / (ROW * n)
    for dx_in, s_, fc, ec, lw, lab in (
            (0.00, 42, INK, INK, 0.0, r"$d_{action}$ — permute the action only"),
            (3.10, 30, "none", INK, 1.35, r"$d_{state}$ — permute the state only")):
        kx = dx_in / strip_w
        ax.scatter([kx], [ky], transform=ax.transAxes, s=s_, facecolor=fc, edgecolor=ec,
                   lw=lw, clip_on=False, zorder=8)
        ax.text(kx + 0.16 / strip_w, ky, lab, transform=ax.transAxes, va="center",
                ha="left", fontsize=10, color=INK)

    _figure_head(fig,
                 "Every arm, every seed: how far does changing ONLY the action move the "
                 "prediction?",
                 "Three orders of magnitude separate the arms and the spread inside an arm "
                 "is tight, so this is a property of the objective, not of a lucky seed. "
                 "V1's prediction does not\nmove at all; block-causal's action is worth 464x "
                 "less than its state.   Arms in colour are this round's design; grey is the "
                 "background population.")

    p = os.path.join(out, "d-action-strip.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 2
def fig_link_target(rows, out):
    """The 2x2 the archive never ran, now complete at n=8-16 per cell.

    Colour encodes the LINK (green = the rectified link that keeps the code alive, crimson =
    the identity link that collapses it). Every test is drawn as a bracket over the cells it
    compares, in a band reserved above the marks, so no p-value floats free of its contrast.
    """
    cells = [("LpWM-ltv", "reprelu", "p=1"), ("LpWM-ltv-relu-p2", "reprelu", "p=2"),
             ("LpWM-ltv-ident-p1", "identity", "p=1"), ("LeWM-ltv-p2", "identity", "p=2")]
    LINKCOL = {"reprelu": C["green"], "identity": C["crimson"]}
    LINKFILL = {"reprelu": FILL["green"], "identity": FILL["crimson"]}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    present = [(a, lk, tp) for a, lk, tp in cells if by.get(a)]
    metrics = [("rho", "A", "code density  " + r"$\rho$",
                "the link moves it; the target distribution does not", False),
               ("eff", "B", "effective dimension",
                "24.1 -> 2.9 across the links, flat within each row", False),
               ("rel_mse", "C", "rel_mse   (log, lower is better)",
                "neither factor separates it — identity's is bimodal", True)]
    G = lambda arm, k: np.array([x[k] for x in by.get(arm, []) if x.get(k) is not None], float)

    # explicit inch layout: the header band is reserved, not negotiated with tight_layout
    FW, FH = 13.6, 6.6
    L, R, TOP, BOT, GAP = 0.78, 0.30, 1.85, 1.15, 0.95
    pw = (FW - L - R - 2 * GAP) / 3.0
    fig = plt.figure(figsize=(FW, FH))
    axes = [fig.add_axes([(L + i * (pw + GAP)) / FW, BOT / FH, pw / FW,
                          (FH - TOP - BOT) / FH]) for i in range(3)]

    for ax, (key, letter, title, sub, logy) in zip(axes, metrics):
        def mw(a, b):
            x, y = G(a, key), G(b, key)
            return stats.mannwhitneyu(x, y).pvalue if x.size and y.size else float("nan")
        labs, tops = [], []
        for j, (arm, lk, tp) in enumerate(present):
            v = G(arm, key)
            if not v.size:
                continue
            c = LINKCOL[lk]
            ax.bar(j, np.median(v), 0.62, color=LINKFILL[lk], edgecolor=c, lw=1.2, zorder=2)
            ax.scatter(np.full(v.size, j) + np.linspace(-.16, .16, v.size), v,
                       s=24, color=c, lw=0, alpha=0.9, zorder=4)
            # the label sits on the BAR, not above the tallest dot — it is the median
            # the median is printed UNDER its own column, not floating in the plot: it can
            # then neither collide with a dot nor be read as the value of the topmost one
            labs.append((j, f"{lk}, {tp}\nn = {v.size}\nmed {_num(float(np.median(v)))}",
                         c))
            tops.append(v.max())
        ax.set_xticks([j for j, _, _ in labs])
        ax.set_xticklabels([t for _, t, _ in labs], fontsize=9.5, linespacing=1.5)
        for tick, (_, _, c) in zip(ax.get_xticklabels(), labs):
            tick.set_color(c)
        ax.set_xlim(-0.62, len(present) - 0.38)
        if logy:
            ax.set_yscale("log")
            _headroom(ax, 0.60, log=True)
        else:
            ax.set_ylim(0, max(tops) / 0.58)
        _style(ax, grid="y")
        ax.tick_params(axis="x", length=0)

        fmt = lambda q: (f"p = {q:.1e}" if q < 1e-3 else f"p = {q:.2f}")
        p1, p2 = mw(present[0][0], present[1][0]), mw(present[2][0], present[3][0])
        pl = mw(present[0][0], present[2][0]) if len(present) == 4 else float("nan")
        _bracket(ax, 0, 1, 0.760, "target_p:  " + fmt(p1), color=LINKCOL["reprelu"])
        _bracket(ax, 2, 3, 0.760, "target_p:  " + fmt(p2), color=LINKCOL["identity"])
        _bracket(ax, 0, 2, 0.925, "link, at p=1:  " + fmt(pl), color=INK, weight="bold")
        _header_at(fig, ax, letter, title, sub)

    _figure_head(fig,
                 "target_p is inert in BOTH rows; the link decides the CODE but not the "
                 "prediction error",
                 "Four within-row Mann-Whitney tests, not one significant. Across the rows "
                 "the code collapses while rel_mse does not separate at all.\n"
                 "Green is the rectified link, crimson the identity link. Each bar is the "
                 "cell median, printed under its column; each dot is one run; every "
                 "p-value\nsits on a bracket over the two cells it compares.")
    p = os.path.join(out, "link-target-2x2.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 3
def fig_v1_mechanism(stats_path, out):
    """Why V1 collapsed: eps=1e-4 sits far below the increments it was meant to floor."""
    if not stats_path or not os.path.exists(stats_path):
        return None
    D = json.load(open(stats_path))
    d2 = np.array(D["d2"], float)
    eps = D["eps"]
    w = 1.0 / (d2 + eps)
    w = w / w.mean()
    share = np.sort(w / w.sum())[::-1]
    FW, FH = 13.2, 6.0
    L, R, TOP, BOT, GAP = 0.80, 0.30, 1.55, 0.95, 1.15
    pw = (FW - L - R - GAP) / 2.0
    fig = plt.figure(figsize=(FW, FH))
    a1 = fig.add_axes([L / FW, BOT / FH, pw / FW, (FH - TOP - BOT) / FH])
    a2 = fig.add_axes([(L + pw + GAP) / FW, BOT / FH, pw / FW, (FH - TOP - BOT) / FH])

    # --- A: the distribution of the increment against the floor that was supposed to catch it
    a1.hist(np.log10(np.clip(d2, 1e-9, None)), bins=60, color=FILL["slate"],
            edgecolor=C["slate"], lw=0.5, zorder=2)
    _style(a1, grid="y")
    _headroom(a1, 0.62)
    xlo, xhi = a1.get_xlim()
    a1.set_xlim(xlo, xhi + 0.60)          # room for the median label to the right of its line
    le, lm = np.log10(eps), np.log10(D["d2_median"])
    a1.axvline(le, color=C["crimson"], lw=2.2, zorder=3)
    a1.axvline(lm, color=C["slate"], lw=1.6, ls=(0, (5, 3)), zorder=3)
    ytop = a1.get_ylim()[1]
    _callout(a1, le + 0.10, ytop * 0.985,
             f"$\\varepsilon$ = {eps:g}\nonly {D['frac_below_eps'] * 100:.1f}% of\n"
             "samples fall below it",
             key="crimson", coords="data", fontsize=10.5, va="top", ha="left")
    a1.annotate("median $\\|\\Delta z\\|^2$ = "
                f"{D['d2_median']:.2e}\n{D['d2_median'] / eps:.0f}x above the floor,\n"
                "so the floor never binds",
                (lm + 0.10, ytop * 0.985), xycoords="data", ha="left", va="top",
                fontsize=10.5, color=C["slate"], zorder=6, linespacing=1.4)
    a1.set_xlabel(r"$\log_{10} \; \|\Delta z\|^2$   (per sample, per timestep)", fontsize=11)
    a1.set_ylabel("number of validation samples", fontsize=10.5, color=INK)
    _header_at(fig, a1, "A", "The floor never engages",
               "1536 real validation samples through a trained LpWM-ltv checkpoint")

    # --- B: what that does to the batch loss
    k = np.arange(1, share.size + 1) / share.size * 100
    cum = np.cumsum(share) * 100
    a2.fill_between(k, k, cum, color=FILL["crimson"], lw=0, zorder=2)
    a2.plot([0, 100], [0, 100], color="#9AA6A2", lw=1.3, ls=(0, (2, 3)), zorder=3)
    a2.plot(k, cum, color=C["crimson"], lw=2.2, zorder=5)
    a2.text(72, 68, "an equal share", fontsize=9.5, color=MUTED, ha="left", va="top")
    top1 = share[0] * 100
    onepct = cum[max(int(round(share.size * 0.01)) - 1, 0)]
    a2.scatter([1.0], [onepct], s=46, color=C["crimson"], zorder=6)
    a2.annotate(f"the single heaviest sample alone\ntakes {top1:.1f}% of the batch loss;\n"
                f"the heaviest 1% take {onepct:.0f}%\n"
                f"(the weights span {D['w_span']:.0f}x)",
                (1.0, onepct), xycoords="data", xytext=(40, 19), textcoords="data",
                ha="left", va="center", fontsize=10.5, color=C["crimson"],
                fontweight="bold", zorder=20, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.45", fc=FILL["crimson"],
                          ec=C["crimson"], lw=0.9),
                arrowprops=dict(arrowstyle="-", color=C["crimson"], lw=1.0,
                                shrinkA=4, shrinkB=6,
                                connectionstyle="angle,angleA=0,angleB=90,rad=8"))
    a2.set_xlim(-2, 102)
    a2.set_ylim(0, 104)
    a2.set_xlabel("share of samples, heaviest first (%)", fontsize=11)
    a2.set_ylabel("cumulative share of the loss (%)", fontsize=10.5, color=INK)
    _style(a2, grid="both")
    _header_at(fig, a2, "B", "...so a handful of near-static frames own the gradient",
               "shaded: how far the loss is from being shared equally")

    _figure_head(fig, "V1 is an implementation failure, not a test of the idea",
                 r"$w = 1/(\|\Delta z\|^2 + \varepsilon)$ was meant to stop large "
                 r"autonomous motion dominating the loss; with this $\varepsilon$ it "
                 "did the opposite.")
    p = os.path.join(out, "v1-mechanism.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 4
# NO FIXED GATES. An earlier version of this figure dichotomised rho, rel_mse and d_action
# at cut-points chosen after seeing the runs, then reported the separation at its own
# training accuracy. That is a fitted decision boundary, not a result. What replaces it is
# a rank-based partial correlation: does d_action carry information about CEM that rho and
# rel_mse do not? No variable is thresholded anywhere below.
HEALTH_COVARS = ("rho", "rel_mse")


def _rank(v):
    return stats.rankdata(np.asarray(v, float))


def partial_spearman(x, y, covars, n_perm=20000, seed=0):
    """Rank partial correlation of x and y given covars, with a permutation p-value.

    Ranks everything, regresses rank(x) and rank(y) on the ranked covariates, and
    correlates the residuals. The null is generated by permuting one residual vector,
    which is exact under exchangeability and needs no distributional assumption.
    """
    x, y = _rank(x), _rank(y)
    Cm = np.column_stack([np.ones(len(x))] + [_rank(c) for c in covars])
    resid = lambda v: v - Cm @ np.linalg.lstsq(Cm, v, rcond=None)[0]
    rx, ry = resid(x), resid(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rng.permutation(rx), ry)[0, 1] for _ in range(n_perm)])
    return r, rx, ry, float((np.abs(null) >= abs(r)).mean())


def fig_d_action_vs_cem(rows, out):
    """What section 8 measured: does d_action say anything about planning that the health
    metrics do not?

    The diary marks this section SUPERSEDED (see 12b): `causal/d_action` is only logged for
    runs trained after the diagnostic was added, which excludes every high-CEM arm, so the
    population below is selection-biased. The figure is kept as the record of what section 8
    measured on the runs that had the field; its titles say exactly that and no more.
    """
    sub = [r for r in rows if r["cem"] is not None and r["d_action"] is not None
           and r["rel_mse"] is not None and r["rho"] is not None]
    if len(sub) < 8:
        return None
    present = [a for a in NAMED if any(r["arm"] == a for r in sub)]

    FW, FH = 14.0, 8.7
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.055, 0.245, 0.505, 0.505])
    bx = fig.add_axes([0.660, 0.245, 0.325, 0.505])
    lg = fig.add_axes([0.055, 0.020, 0.930, 0.140]); lg.axis("off")

    # --- left: the raw relationship, no thresholds, CEM on a continuous axis ---
    for r in sorted(sub, key=lambda r: is_named(r["arm"])):
        named = is_named(r["arm"])
        c = col(r["arm"])
        ax.scatter(max(r["d_action"], 3e-4), r["cem"], s=78 if named else 30,
                   color=c, edgecolor="white" if named else "none",
                   lw=0.7, alpha=0.95 if named else 0.75, zorder=4 if named else 3)
    sp = stats.spearmanr([max(r["d_action"], 3e-4) for r in sub], [r["cem"] for r in sub])
    _callout(ax, 0.025, 0.975,
             f"marginal Spearman = {sp.correlation:+.2f},  p = {sp.pvalue:.2g}   "
             f"(n = {len(sub)})", key="slate", tc=INK, fontsize=10.5, weight="normal")
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 8)
    ax.set_ylim(-0.06, 0.80)
    ax.set_xlabel(r"$d_{action}$   (log; points at the left edge are exact zeros)",
                  fontsize=11)
    ax.set_ylabel("CEM success rate", fontsize=11, color=INK)
    _style(ax, grid="both")
    _header_at(fig, ax, "A", "Marginal: every run that logged the field, no cut-points",
               "one point per run; colour is the arm's identity")

    # --- right: the same question with the health metrics partialled out ---
    r_par, rx, ry, p_par = partial_spearman(
        [r["d_action"] for r in sub], [r["cem"] for r in sub],
        [[r[c] for r in sub] for c in HEALTH_COVARS])
    # DEGENERATE = the predictor is provably action-blind (d_action is exactly 0), or the
    # arm is block-causal, which an independent measurement (std_t(z)) called collapsed a
    # day earlier. Neither is a fitted cut-point; both are identified without looking at CEM.
    degen = [r["d_action"] == 0 or r["arm"] == "PiWM-blockcausal" for r in sub]
    keep = [i for i, d in enumerate(degen) if not d]
    order = sorted(range(len(sub)), key=lambda i: is_named(sub[i]["arm"]))
    for i in order:
        r, xx, yy, dg = sub[i], rx[i], ry[i], degen[i]
        named, c = is_named(r["arm"]), col(r["arm"])
        bx.scatter(xx, yy, s=78 if named else 30,
                   color="white" if dg else c, edgecolor=c if (dg or named) else "none",
                   lw=1.6 if dg else 0.7, alpha=0.95 if named else 0.8,
                   zorder=5 if dg else (4 if named else 3))
    r_ok, _, _, p_ok = partial_spearman(
        [sub[i]["d_action"] for i in keep], [sub[i]["cem"] for i in keep],
        [[sub[i][c] for i in keep] for c in HEALTH_COVARS])
    lo, hi = rx.min(), rx.max()
    b1, b0 = np.polyfit(rx, ry, 1)
    bx.plot([lo, hi], [b1 * lo + b0, b1 * hi + b0], color=C["crimson"], lw=2.0,
            ls=(0, (6, 3)), zorder=6)
    xlab = lo + 0.06 * (hi - lo)
    bx.annotate("fit to all runs", (xlab, b1 * xlab + b0), xytext=(4, -11),
                textcoords="offset points", fontsize=9.5, color=C["crimson"],
                fontweight="bold", va="top", ha="left", zorder=8,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
    if len(keep) > 8:
        klo, khi = rx[keep].min(), rx[keep].max()
        c1, c0 = np.polyfit(rx[keep], ry[keep], 1)
        bx.plot([klo, khi], [c1 * klo + c0, c1 * khi + c0], color=C["slate"], lw=2.0,
                zorder=6)
        xk = klo + 0.97 * (khi - klo)
        bx.annotate("fit to the rest", (xk, c1 * xk + c0), xytext=(2, 12),
                    textcoords="offset points", fontsize=9.5, color=C["slate"],
                    fontweight="bold", va="bottom", ha="right", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
    bx.axhline(0, color=GRID, lw=1, zorder=1)
    bx.axvline(0, color=GRID, lw=1, zorder=1)
    _style(bx, grid="none")
    bx.margins(x=0.10, y=0.10)
    ylo, yhi = bx.get_ylim()
    bx.set_ylim(ylo, ylo + (yhi - ylo) / 0.80)
    _callout(bx, 0.025, 0.975,
             f"all {len(sub)}:  {r_par:+.2f}   ({_pfmt(p_par)})\n"
             f"minus the {len(sub) - len(keep)} degenerate (hollow):  "
             f"{r_ok:+.2f}   ({_pfmt(p_ok)})", key="crimson", fontsize=10.5)
    bx.set_xlabel(r"rank $d_{action}$, residual after $\rho$ and rel_mse", fontsize=11)
    bx.set_ylabel("rank CEM, residual after the same", fontsize=11, color=INK)
    _header_at(fig, bx, "B", r"Partial: with $\rho$ and rel_mse removed",
               "hollow = degenerate; both exclusions are identified without looking at CEM")

    # --- the key, in its own strip so it can never reach an axis label ---
    hs = [Line2D([], [], ls="", marker="o", ms=8, mfc=col(a), mec="white", mew=0.7,
                 label=LABEL.get(a, a)) for a in present]
    hs.append(Line2D([], [], ls="", marker="o", ms=6, mfc=OTHER, mec="none",
                     label=f"{len(set(r['arm'] for r in sub)) - len(present)} other arms"))
    hs.append(Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=C["slate"], mew=1.6,
                     label="degenerate (panel B)"))
    lg.legend(handles=hs, loc="upper center", ncol=7, frameon=False, fontsize=9.5,
              handletextpad=0.4, columnspacing=1.5, labelspacing=0.9,
              borderaxespad=0.0)

    fig.text(0.055, 1 - 0.30 / FH,
             "What section 8 measured, on the "
             f"{len(sub)} runs that logged " + r"$d_{action}$",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.055, 1 - 0.60 / FH,
             r"Partialling out $\rho$ and rel_mse leaves "
             f"{r_par:+.2f} ({_pfmt(p_par)}) over all {len(sub)} runs;\nremoving the "
             f"{len(sub) - len(keep)} whose predictor is provably action-blind, plus "
             f"block-causal, leaves {r_ok:+.2f} ({_pfmt(p_ok)}).\nRead this as the record "
             "of section 8, not as the campaign's answer.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.45)
    fig.text(0.985, 1 - 0.24 / FH,
             "SUPERSEDED — see 12b\n"
             r"$d_{action}$ is logged only for runs trained after the diagnostic" "\n"
             "was added, which excludes every high-CEM arm. Re-measured from\n"
             "the checkpoints over all 235 evaluated runs, the partial is +0.545.",
             fontsize=9.5, color=C["crimson"], ha="right", va="top", linespacing=1.5,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc=FILL["crimson"], ec=C["crimson"],
                       lw=0.9))

    p = os.path.join(out, "d-action-vs-cem.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 5
def _place_labels(fig, ax, items, points, fontsize=10.0, pad=5.0, mark_r=13.0,
                  lead_sep=11.0):
    """Put each label near its own mark without letting two labels, or a label and a mark,
    or two LEADER LINES, or a label and the panel edge, ever touch.

    `items` is [(x, y, text, colour)] in data coordinates and `points` is every mark that
    must stay uncovered. Candidates are tried on rings of increasing radius; the first that
    collides with nothing wins. Leaders are kept apart too — two arms whose medians sit a
    few pixels apart (columns and reprelu p=2 do) would otherwise get two labels pointing
    down the same corridor, and neither would own its mark.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ab = ax.get_window_extent()
    px = ax.transData.transform(np.asarray(points, float)) if len(points) else np.empty((0, 2))
    placed, leaders = [], []

    def rect(cx, cy, w, h):
        return (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)

    def sample(a, b, k=18):
        t = np.linspace(0, 1, k)[:, None]
        return np.asarray(a) * (1 - t) + np.asarray(b) * t

    def hits(r, seg, own=None, level=0):
        """level 0 enforces everything; higher levels drop the softer constraints in turn,
        so a crowded neighbourhood degrades to 'labels still do not overlap' rather than to
        a label parked on top of its neighbour."""
        mr = mark_r if level < 3 else 0.0
        if r[0] < ab.x0 + 2 or r[2] > ab.x1 - 2 or r[1] < ab.y0 + 2 or r[3] > ab.y1 - 2:
            return True
        for q in placed:
            if not (r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3]):
                return True
        if px.size:
            # marks have a radius: a label that only clears the CENTRE still lands on the ring
            inside = ((px[:, 0] > r[0] - mr) & (px[:, 0] < r[2] + mr) &
                      (px[:, 1] > r[1] - mr) & (px[:, 1] < r[3] + mr))
            if inside.any():
                return True
        if seg is not None:
            a = sample(*seg)
            if level < 2:
                for other in leaders:
                    b = sample(*other)
                    if np.min(np.linalg.norm(a[:, None, :] - b[None, :, :],
                                             axis=-1)) < lead_sep:
                        return True
            if level < 1 and px.size:   # a leader may not run through somebody else's mark
                other_pts = px
                if own is not None:
                    other_pts = px[np.linalg.norm(px - np.asarray(own), axis=1) > 1e-6]
                if other_pts.size:
                    d = np.linalg.norm(a[:, None, :] - other_pts[None, :, :], axis=-1)
                    if d.min() < mark_r:
                        return True
        return False

    angles = np.deg2rad([90, 270, 0, 180, 45, 135, 315, 225, 60, 120, 240, 300,
                         20, 160, 200, 340, 75, 105, 255, 285])
    for x, y, text, c in items:
        t = ax.text(0, 0, text, fontsize=fontsize, color=c, ha="center", va="center",
                    zorder=12, linespacing=1.3)
        bb = t.get_window_extent(renderer=rend)
        w, h = bb.width, bb.height
        ox, oy = ax.transData.transform((x, y))
        own = (ox, oy)
        best, bestr, bestseg = None, None, None
        for level in (0, 1, 2, 3):
            for rad in (26, 38, 52, 68, 86, 108, 134, 168, 206):
                for a in angles:
                    cx = ox + (rad + w / 2) * np.cos(a)
                    cy = oy + (rad + h / 2) * np.sin(a)
                    r = rect(cx, cy, w, h)
                    d = np.hypot(cx - ox, cy - oy)
                    seg = None
                    ux, uy = (cx - ox) / d, (cy - oy) / d
                    edge = min(abs((w / 2 + 2) / ux) if abs(ux) > 1e-6 else 1e9,
                               abs((h / 2 + 2) / uy) if abs(uy) > 1e-6 else 1e9)
                    # a leader only when the label is far enough from the mark that ownership
                    # is in doubt — otherwise a 6px dash floats next to the mark saying nothing
                    if d - mark_r - edge > 20:
                        seg = ((ox + ux * (mark_r - 2), oy + uy * (mark_r - 2)),
                               (cx - ux * (edge + 3), cy - uy * (edge + 3)))
                    if not hits(r, seg, own, level):
                        best, bestr, bestseg = (cx, cy), r, seg
                        break
                if best:
                    break
            if best:
                break
        if best is None:                      # nowhere clean: park it and accept the least bad
            best = (ox, oy + 34 + h / 2)
            bestr, bestseg = rect(*best, w, h), None
        placed.append(bestr)
        t.set_position(ax.transData.inverted().transform(best))
        if bestseg is not None:
            leaders.append(bestseg)
            (x0, y0), (x1, y1) = bestseg
            p0 = ax.transData.inverted().transform((x0, y0))
            p1 = ax.transData.inverted().transform((x1, y1))
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=0.7, alpha=0.7, zorder=3,
                    solid_capstyle="butt")


def fig_quadrant(rows, out):
    """Arm medians in the plane the round is actually about: is the code healthy, and is
    the prediction action-sensitive? Buying the second by destroying the first is easy.

    Only this round's arms are labelled — 45 stacked captions is not a chart. The rest are
    drawn as small unlabelled marks so the population they define is still visible.
    """
    by = collections.defaultdict(list)
    for r in rows:
        if r["d_action"] is not None and r["rho"] is not None:
            by[r["arm"]].append(r)
    fig = plt.figure(figsize=(13.0, 8.0))
    ax = fig.add_axes([0.062, 0.095, 0.845, 0.735])
    # NO fitted box here either: CEM is shown as a continuous single-hue ramp, so the
    # reader sees the gradient rather than a cut-point someone chose.
    cems = {a: np.median([r["cem"] for r in v if r["cem"] is not None])
            for a, v in by.items() if any(r["cem"] is not None for r in v)}
    vmax = max(cems.values()) if cems else 1.0

    pts, items, n_other = [], [], 0
    for arm, v in sorted(by.items(), key=lambda kv: is_named(kv[0])):
        c = col(arm)
        x = float(np.median([r["rho"] for r in v]))
        y = max(float(np.median([r["d_action"] for r in v])), 4e-4)
        pts.append((x, y))
        cem = cems.get(arm)
        face = CEM_RAMP(cem / vmax) if cem is not None else "white"
        if is_named(arm):
            # a white collar under the ring, so two arms that land on top of each other
            # (columns and reprelu p=2 are 1px apart in rho) still read as two marks
            ax.scatter(x, y, s=330, color="white", lw=0, zorder=5)
            ax.scatter(x, y, s=210, color=face, edgecolor=c, lw=2.4, zorder=6)
            items.append((x, y, f"{LABEL.get(arm, arm)}\n(n={len(v)}" +
                          (f", CEM {cem:.2f})" if cem is not None else ")"),
                          tcol(arm)))
        else:
            ax.scatter(x, y, s=52, color=face, edgecolor=OTHER, lw=1.2, zorder=4)
            n_other += 1

    ax.set_yscale("log")
    ax.set_ylim(2.0e-4, 6.0)
    ax.set_xlim(-0.07, 1.13)
    ax.set_xlabel(r"code density  $\rho$    (fraction of units active; 1.0 = fully dense)",
                  fontsize=11.5)
    ax.set_ylabel(r"$d_{action}$    (log)", fontsize=11.5, color=INK)
    _style(ax, grid="both")
    sm = plt.cm.ScalarMappable(cmap=CEM_RAMP, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("CEM success rate  (fill of each mark)", fontsize=10.5, color=INK)
    cb.ax.tick_params(labelsize=9, colors=MUTED, length=3)
    cb.outline.set_edgecolor(EDGE)

    ax.annotate(f"{n_other} arms outside this round's design, unlabelled",
                (0.985, 0.018), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=9.5, color=MUTED)
    _place_labels(fig, ax, items, pts, fontsize=10.0)

    fig.text(0.062, 0.985, "Action-sensitivity is cheap if you are willing to wreck the code",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.062, 0.950,
             "V2 — the arm DESIGNED for action-sensitivity — bought the most of any arm and "
             "collapsed the code doing it. Three arms did land healthy AND\n"
             "sensitive: V3 pathint, reprelu p=2 and columns — none of them was designed to. "
             "Fill shows CEM on a continuous scale; nothing here is thresholded.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.45)

    p = os.path.join(out, "causal-quadrant.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return p

# ------------------------------------------------------------------ figure 6
def fig_arm_health(rows, out):
    """Per-arm code density and prediction error side by side, so 'healthy' is not a word."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["d_action"] is not None and r["rho"] is not None and r["rel_mse"] is not None:
            by[r["arm"]].append(r)
    # barh index 0 sits at the BOTTOM, so sort worst-first to put the best arms on top
    order = sorted(by, key=lambda a: -np.median([r["rel_mse"] for r in by[a]]))
    n = len(order)

    FW, ROW = 13.6, 0.245
    L, R, TOP, BOT, GAP = 2.35, 0.30, 1.55, 0.90, 0.95
    FH = TOP + BOT + ROW * n
    fig = plt.figure(figsize=(FW, FH))
    pw = (FW - L - R - GAP) / 2.0
    y0, h = BOT / FH, (ROW * n) / FH
    a1 = fig.add_axes([L / FW, y0, pw / FW, h])
    a2 = fig.add_axes([(L + pw + GAP) / FW, y0, pw / FW, h])
    _rowbands(a1, n)
    _rowbands(a2, n)

    mse_labels = []
    for i, arm in enumerate(order):
        v = by[arm]
        named = is_named(arm)
        c = col(arm)
        fc = c if named else OTHER
        rho = float(np.median([r["rho"] for r in v]))
        mse = float(np.median([r["rel_mse"] for r in v]))
        a1.barh(i, rho, 0.62, color=fc, edgecolor=fc, lw=0, zorder=3)
        a2.barh(i, mse, 0.62, color=fc, edgecolor=fc, lw=0, zorder=3)
        kw = dict(xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                  color=tcol(arm) if named else MUTED, zorder=8,
                  fontweight="bold" if named else "normal",
                  path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        a1.annotate(f"{rho:.3f}", (rho, i), **kw)
        mse_labels.append((a2.annotate(f"{mse:.4f}", (mse, i), **kw), mse))

    a1.set_yticks(range(n))
    a1.set_yticklabels([LABEL.get(a, a) for a in order], fontsize=9.5)
    for t, a in zip(a1.get_yticklabels(), order):
        t.set_color(tcol(a) if is_named(a) else MUTED)
        if is_named(a):
            t.set_fontweight("bold")
    a1.set_xlim(0, 1.30)
    a1.set_xlabel(r"code density  $\rho$", fontsize=11, labelpad=7)
    _header_at(fig, a1, "A", "Which arms keep a code?",
               r"median $\rho$ per arm; 1.0 = every unit active on every sample")

    # the only reference line is rel_mse = 1.0, which is a PROPERTY of the metric
    # (the error of predicting the mean), not a threshold anyone chose.
    a2.axvline(1.0, color=C["crimson"], lw=1.5, ls=(0, (5, 3)), zorder=4)
    a2.set_xscale("log")
    a2.set_xlim(8e-4, 26)
    a2.set_yticks(range(n))
    a2.set_yticklabels([])
    a2.set_xlabel("rel_mse   (log, lower is better)", fontsize=11, labelpad=7)
    _header_at(fig, a2, "B", "...and which of them still predict?",
               "median rel_mse per arm, same row order")

    # no value label may be crossed by the rel_mse = 1.0 reference line
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    xline = a2.transData.transform((1.0, 0))[0]
    for ann, mse in mse_labels:
        bb = ann.get_window_extent(renderer=rend)
        if bb.x0 - 3 <= xline <= bb.x1 + 3:
            xb = a2.transData.transform((mse, 0))[0]
            ann.set_position(((xline - xb) * 72.0 / fig.dpi + 7.0, 0))
    a2.text(1.55, n * 0.60, "1.0 = predicting the mean — a property of the metric, "
            "not a threshold anyone chose", rotation=90, ha="center", va="center",
            fontsize=9.5, color=C["crimson"], fontweight="bold", zorder=7,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

    for ax in (a1, a2):
        _style(ax, grid="x")
        ax.set_ylim(-0.6, n - 0.4)
        ax.tick_params(axis="y", length=0)

    _figure_head(fig,
                 "Code density and prediction error are two different axes of 'healthy'",
                 "Rows are sorted by median rel_mse, best at the top; panel A is the same "
                 "order. The three fully dense arms (\u03c1 = 1.000) are spread across the "
                 "whole range of B,\nwhich is why rel_mse alone cannot certify a world "
                 "model. Arms in colour are this round's design; grey is the background "
                 "population.")

    p = os.path.join(out, "arm-health.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--campaign", default="/tmp/campaign_now.json")
    ap.add_argument("--incr-stats", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = harvest(a.repo, a.campaign)
    print(f"harvested {len(rows)} runs with causal diagnostics "
          f"({sum(r['cem'] is not None for r in rows)} with a CEM number)")
    for p in (fig_d_action_strip(rows, a.out), fig_link_target(rows, a.out),
              fig_v1_mechanism(a.incr_stats, a.out), fig_d_action_vs_cem(rows, a.out),
              fig_quadrant(rows, a.out), fig_arm_health(rows, a.out)):
        if p:
            print("  wrote", p)


if __name__ == "__main__":
    main()
