"""The one design system for every figure in this project.

Derived from figures/motivation_teaser.svg, which is the submission-quality reference. Both the
matplotlib figures (PNG) and the hand-built architecture diagrams (SVG) import from here, so the
palette and typography are defined exactly once.

WHAT THE REFERENCE ESTABLISHES

    hue           role                                 primary    fill
    green         the system / the thing that works    #18483C    #D8F0E4
    purple        the contrasting condition            #543C84    #DED6EA
    amber         the intervention under test          #C88A1E    #F7E8C8
    crimson       failure, retraction, alert           #A83024    #F5E2E0
    slate         neutral / everything else            #546060    #DFE4E4

Rules the reference follows and every figure here must too:

  * Sans-serif throughout. Maths is the only italic.
  * Hues are assigned by IDENTITY, never by rank -- regenerating a figure after more evals land
    must not repaint the arms.
  * Recessive furniture: a hairline grid in GRID, no top/right spine, ticks in MUTED.
  * A panel letter (A, B, C) in the accent colour with a rule under the header.
  * Direct labels beside marks; a legend only when there are more than four series.
  * Callout boxes are a pale fill with the matching saturated text, never a saturated fill.
  * Nothing overlaps. `fit()` is provided because that is the failure this system exists to stop.

Usage (matplotlib):
    from analysis.style import C, FILL, INK, MUTED, GRID, use_style, ax_style, panel_title, fit
    use_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ...
    ax_style(ax); fit(fig)

Usage (SVG): import PALETTE_SVG / FONT_SVG from here; see analysis/arch_figs.py.
"""
from __future__ import annotations

# --- palette ---------------------------------------------------------------------
C = {
    "green":   "#18483C",
    "purple":  "#543C84",
    "amber":   "#C88A1E",
    "crimson": "#A83024",
    "slate":   "#546060",
    "teal":    "#2E7D6F",     # a lighter green, for a second same-family series
    "plum":    "#8A6FA8",     # a lighter purple, likewise
}
FILL = {
    "green":   "#D8F0E4",
    "purple":  "#DED6EA",
    "amber":   "#F7E8C8",
    "crimson": "#F5E2E0",
    "slate":   "#DFE4E4",
    "teal":    "#DCEFE9",
    "plum":    "#E9E1F1",
}

INK = "#14231C"          # body text, axis labels
HEAD = "#18483C"         # section/panel headers
MUTED = "#6C786C"        # secondary text, tick labels
GRID = "#E4E4E4"         # gridlines
EDGE = "#D8D8D8"         # panel borders
PAPER = "#FFFFFF"

# The categorical order. Never cycle past the end -- fold into "other" or facet instead.
ORDER = ["green", "purple", "amber", "crimson", "slate", "teal", "plum"]
CATEGORICAL = [C[k] for k in ORDER]

# Sequential ramp for a magnitude (single hue, light -> dark), from the reference's green.
SEQ_GREEN = ["#EAF6F0", "#C4E4D6", "#90CCB4", "#4E9E82", "#246048", "#12362C"]
# Diverging pair with a NEUTRAL grey midpoint -- never a hue at the middle.
DIVERGING = ["#A83024", "#D08A80", "#E8E8E8", "#7FA9A0", "#18483C"]

FONT_STACK = ["Nimbus Sans", "Liberation Sans", "DejaVu Sans", "sans-serif"]
FONT_SVG = "Nimbus Sans, Liberation Sans, Helvetica, Arial, sans-serif"

# What the SVG modules consume. Keys match arch_figs.py's historical ACCENT keys so the
# architecture diagrams re-colour without restructuring.
PALETTE_SVG = {
    "blue": C["green"], "magenta": C["purple"], "green": C["teal"],
    "amber": C["amber"], "crit": C["crimson"], "slate": C["slate"],
}
PALETTE_SVG_FILL = {
    "blue": FILL["green"], "magenta": FILL["purple"], "green": FILL["teal"],
    "amber": FILL["amber"], "crit": FILL["crimson"], "slate": FILL["slate"],
}


# --- matplotlib ------------------------------------------------------------------
def use_style():
    """Set the rcParams once per figure module. Safe to call repeatedly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "axes.titlecolor": INK,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # embed as TrueType, not Type 3 -- required by most venues
        "ps.fonttype": 42,
    })


def ax_style(ax, grid="y"):
    """Recessive furniture: hairline grid on one axis, no top/right spine."""
    ax.set_axisbelow(True)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    if grid == "none":
        ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=9, length=3)
    return ax


def panel_title(ax, letter, title, sub=None, y=1.06):
    """The reference's header: a coloured panel letter, the title, and a rule beneath."""
    ax.set_title("")
    t = ax.text(0, y, f"{letter}", transform=ax.transAxes, fontsize=11.5, fontweight="bold",
                color=HEAD, va="bottom", ha="left")
    ax.text(0.032, y, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold",
            color=INK, va="bottom", ha="left")
    if sub:
        ax.text(0.032, y - 0.058, sub, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom", ha="left")
    return t


def callout(ax, x, y, text, key="green", fontsize=10, ha="left", va="center", **kw):
    """Pale fill, saturated matching text -- the reference's callout box."""
    return ax.annotate(text, (x, y), fontsize=fontsize, color=C[key], ha=ha, va=va,
                       fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.42", fc=FILL[key], ec=C[key],
                                 lw=0.9, alpha=0.95), **kw)


def fit(fig, **kw):
    """tight_layout with the reference's padding. This exists because overlapping text is the
    single most common defect in this project's figures."""
    fig.tight_layout(pad=0.9, w_pad=1.6, h_pad=1.2, **kw)
    return fig


def suptitle(fig, title, sub=None):
    """Left-aligned figure header in the reference's style: bold finding, muted subtitle."""
    fig.suptitle("")
    fig.text(0.0, 1.005, title, fontsize=12.5, fontweight="bold", color=INK,
             ha="left", va="bottom")
    if sub:
        fig.text(0.0, 0.995, sub, fontsize=10, color=MUTED, ha="left", va="top")
