"""The visual design of the LpWM / Pi-WM campaign, in exactly one place.

Every function here is a pure function of plain numpy / dict inputs that RETURNS a
matplotlib Figure. Nothing in this module saves a file, touches wandb, or reads the
filesystem, which is what lets the two consumers share one design:

  train.py            wraps the returned Figure in wandb.Image, live during training
  analysis/figures.py saves the returned Figure as a PNG, offline after a campaign

A panel never raises on empty or degenerate input -- it returns a figure carrying a
"no data" message instead, so a live training run cannot be killed by a missing
diagnostic. Deciding whether an input is worth plotting at all is the CALLER's job
(analysis/figures.py records a skip reason; train.py just renders what it has).

Colour system, defined once and obeyed everywhere:

  * IDENTITY is composite: hue = campaign step, lightness = the factor under test,
    marker/dash = the second factor. An arm is the same colour in every panel, and
    the legend reads as the experimental design rather than as nine arbitrary hues
  * the two flags-off CONTROLS are neutral grey, not hues. A control is context you
    read a variant against, and demoting it is what leaves enough separable palette
    for the seven variants -- nine categorical hues provably cannot be separated
  * single-hue sequential (orange, `SEQ`) for magnitudes and densities, so a
    magnitude ramp is never mistaken for an arm
  * diverging RdBu_r, centred EXACTLY at zero, for anything signed (effects,
    deltas, correlations) -- a sequential map on signed data hides the sign, and an
    off-centre diverging map invents one
  * teal (#2ec4b6) is RESERVED for contact-onset markers and appears nowhere else,
    so teal always means "contact" without a legend
  * STATUS (pass / fail / underpowered) rides on marker fill and glyph, never on
    hue, so a verdict and an arm identity can occupy the same mark

Every hue was validated with a CVD simulator rather than chosen by eye; `SERIES_LADDER`
and `needs_facet()` carry the consequence -- past six series on one axis, colour stops
working and the panel must facet.
"""
import hashlib
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors as mcolors  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# --- colour system --------------------------------------------------------------
#
# Derived, not chosen. Every hex below was produced by stepping OKLCH lightness
# inside the categorical band and then run through the CVD validator; the numbers
# quoted in `validate_palette()` are that run's output, and the test suite re-derives
# them so a well-meaning tweak cannot silently break colourblind separation.
#
# The predecessor palette -- one flat hue per arm, nine of them -- failed: blue
# `#4c72b0` and purple `#8172b3` sat 1.9 dE apart under protanopia, and the Step 3
# control `#55a868` was 7.3 dE from `#c44e52`, so a deuteranope could not tell the
# CONTROL from the worst arm. Nine categorical hues cannot be separated; that is
# arithmetic, not taste (see `SERIES_LADDER`).

#: teal, reserved exclusively for contact-onset markers. Never a series colour, so
#: teal always means "contact" without a legend.
CONTACT = "#2ec4b6"
#: single-hue sequential ramp for magnitudes / densities. Orange, because the three
#: family hues take blue / magenta / green and a magnitude ramp must not be mistaken
#: for an arm. `SEQ_ALT` is the second simultaneous sequential context.
SEQ, SEQ_ALT = "lpwm_orange", "lpwm_slate"
#: diverging map for signed quantities, always centred at 0 by `symmetric_limits`
DIV = "RdBu_r"
#: the ghost colour used to draw a reference series behind the highlighted one
GHOST = "#dcdcdc"
#: neutral ink for annotations
INK = "0.15"

#: Chart surfaces the palette was validated against.
SURFACE = {"light": "#ffffff", "dark": "#161616"}

#: Family ramps: (light hex, dark hex) per level, level 0 = lighter. Validated
#: all-pairs in BOTH modes -- worst CVD dE 8.4 light / 7.9 dark (>= 8 target, 6-8
#: legal with secondary encoding), worst normal-vision dE 23.0 / 16.3 (>= 15 floor).
FAMILY_RAMP = {
    "blue":    (("#59a6ff", "#4795f4"), ("#0055af", "#0761bc")),
    "magenta": (("#e97ca5", "#e45f8e"), ("#912d59", "#aa285e")),
    "green":   (("#56c050", "#43ae3e"), ("#006e00", "#007a00")),
}
#: The control ink. Deliberately BELOW the chroma floor -- it must read as grey.
#: A control is context, not a competing tenth hue, and demoting the two controls
#: to neutral is exactly what buys back enough palette for the seven variants.
CONTROL_INK = {"light": "#3d3d3a", "dark": "#c3c2b7"}

#: arm -> (family, level, marker, dashed). `family=None` means the neutral control.
#:
#: Hue = campaign step. Lightness = the factor under test. Marker / dash = the
#: second factor -- so Step 3's 2x2 (gate input x normalisation) is legible as a
#: factorial design rather than three arbitrary colours, and the legend teaches the
#: experiment. Marker doubles as the secondary encoding the 6-8 dE band requires.
ARM_SPEC = {
    "LpWM-base":     (None,      0, "s", False),
    "PiWM-sparse-matched": ("blue",    0, "o", False),
    "PiWM-sparse-2pct":     ("blue",    1, "o", False),
    "LpWM-ltv":    (None,      0, "D", False),
    "PiWM-gate-sup-sigmoid":  ("magenta", 0, "o", False),
    "PiWM-gate-sup-softmax":  ("magenta", 0, "^", True),
    "PiWM-gate-mag-softmax":  ("magenta", 1, "^", True),
    "PiWM-union4":      ("green",   0, "o", False),
    "PiWM-union4-entropy":       ("green",   1, "o", False),
    # Wave 2. LeWM is the DENSE counterpart (identity link, Gaussian target) -- a
    # reference arm, not an intervention, so it is neutral like the other controls.
    "LeWM-ltv":         (None,      0, "v", False),
    # Step 2 x Step 4: a union head at k-WTA sparsity. Green family (union head) with
    # a distinct marker+dash, because its two lightness steps are already spoken for.
    "PiWM-union4-kwta8": ("green",   1, "^", True),
    # Wave 3. gate-both feeds [z ; 1[z>0]] -- the information-lossless gate. Magenta
    # (support-gating family) at the darker step, square marker: it is the repair of
    # the two magenta variants above, not a third variant of them.
    "PiWM-gate-both":    ("magenta", 1, "s", False),
    # Wave 4. Variance-floor arms. LpWM-ltv-vfloor is a CONTROL (below): it exists to
    # show the floor does nothing to a healthy code, so it must stay neutral.
    "LpWM-ltv-vfloor":   (None,      0, "P", False),
    "PiWM-union4-vfloor": ("green",  0, "s", False),
    # J=1 + ltv + k-WTA: the control that was missing under union4-kwta8, without which
    # that arm's 0.00 is fully explained by the k-WTA main effect and attributes
    # nothing to the union. Blue (k-WTA family), since that is the factor it isolates.
    "PiWM-kwta8-J1":     ("blue",    1, "D", False),
    # Wave 5, D=2048: k-WTA at 2% gives w=41 active units, inside Numenta's SDR band
    # (n=2048-10000, w=10-40). The D=384 sparse arms sat at w=8, OUTSIDE it on both
    # axes -- so "sparsity hurts" was never measured on an actual SDR.
    "LpWM-ltv-d2048":    (None,      0, "X", False),
    "PiWM-sdr-d2048-k41": ("blue",   0, "*", False),
}

#: Human-readable family label, for legends and facet headers.
FAMILY_LABEL = {None: "baseline (LpWM)", "blue": "sparse codes (k-WTA)",
                "magenta": "support gating", "green": "union head"}

#: Reference arms, in the order panels prefer them as a baseline. These are drawn
#: neutral rather than hued: a control is context you read a variant against, and
#: demoting them is what leaves enough separable palette for the variants. "LeWM-ltv"
#: is here because the dense baseline is a reference too, not a PiWM intervention.
CONTROL_ARMS = ("LpWM-base", "LpWM-ltv", "LeWM-ltv",
                # matched controls for the wave-4/5 interventions: each exists only to
                # isolate its variant's one changed factor, so it is drawn neutral.
                "LpWM-ltv-vfloor", "LpWM-ltv-d2048")

#: How many series may share one axis before hue stops working. Derived: three
#: lightness steps of a single family measure dE 12.9 against normal vision, under
#: the 15 floor, so nine arms CANNOT be told apart by colour on one axis. Past
#: `FACET_ABOVE` a panel must facet instead of adding series.
SERIES_LADDER = {"hue_alone": 3, "hue_plus_labels": 6}
FACET_ABOVE = SERIES_LADDER["hue_plus_labels"]

#: Fallback hues for arms outside the canonical campaign (probe runs, renamed arms).
#: Hashed, not enumerated, so one extra probe run cannot shift every other colour.
_EXTRA_COLORS = ("#7a5195", "#ef5675", "#2f4b7c", "#ffa600", "#003f5c", "#bc5090")

_ARM_STRIP = re.compile(r"_pd\d+|_(bf16|fp16|no)(?=_|$)|_s\d+$")


def _ramp(hue, lo=0.14, hi=0.92, n=256):
    """One-hue sequential colormap from a family anchor, light -> dark."""
    base = mcolors.to_rgb(hue)
    stops = [tuple(1 - t + t * c for c in base) for t in np.linspace(lo, 1.0, n // 2)]
    stops += [tuple(c * k for c in base) for k in np.linspace(1.0, 0.45, n - len(stops))]
    return mcolors.LinearSegmentedColormap.from_list("lpwm", stops)


#: Registered so `cmap=SEQ` works anywhere matplotlib takes a colormap name. The
#: `_r` variants are registered too: matplotlib synthesises `_r` only for its own
#: built-ins, so `SEQ + "_r"` raises on a custom map unless it is registered here.
for _name, _anchor in (("lpwm_orange", "#eb6834"), ("lpwm_slate", "#2a78d6")):
    _cm = _ramp(_anchor)
    for _n, _c in ((_name, _cm), (_name + "_r", _cm.reversed())):
        try:
            matplotlib.colormaps.register(_c, name=_n, force=True)
        except (AttributeError, ValueError):  # matplotlib < 3.5
            pass


def canon_arm(name):
    """Run dir / wandb run name -> campaign arm, matching train.py's group derivation.

    'PiWM-union4-entropy_pd384_bf16_s0' -> 'PiWM-union4-entropy'. Idempotent, so an already-canonical
    arm name passes through and `arm_color` can be handed either form.
    """
    return _ARM_STRIP.sub("", str(name or ""))


def is_control(arm):
    """True for the flags-off controls, which are drawn neutral rather than hued."""
    return ARM_SPEC.get(canon_arm(arm), (0,))[0] is None and canon_arm(arm) in ARM_SPEC


def arm_family(arm):
    """The arm's campaign step ('blue' / 'magenta' / 'green'), or None for a control."""
    return ARM_SPEC.get(canon_arm(arm), (None,))[0]


def arm_color(arm, mode="light"):
    """Stable colour for an arm, canonical or not.

    Stability matters more than prettiness: the same arm must get the same colour in
    every process, every panel and every run of the suite, so unknown names are
    hashed rather than enumerated (an enumeration would shift every colour as soon
    as one probe run appeared in the glob -- which is exactly the bug the old
    `tab10[i % 10]` had).
    """
    a = canon_arm(arm)
    spec = ARM_SPEC.get(a)
    if spec is None:
        h = hashlib.md5(a.encode("utf-8")).hexdigest()
        return _EXTRA_COLORS[int(h[:8], 16) % len(_EXTRA_COLORS)]
    family, level, _, _ = spec
    if family is None:
        return CONTROL_INK[mode]
    return FAMILY_RAMP[family][level][0 if mode == "light" else 1]


def arm_style(arm, mode="light", lw=2.0):
    """Everything needed to draw one arm: colour, marker, dash and weight.

    Returned as a dict so a caller can splat it into `ax.plot(**style)`. Marker and
    dash are not decoration -- they are the secondary encoding that keeps the Step 3
    variants separable when their two lightness steps sit in the 6-8 dE band.
    """
    a = canon_arm(arm)
    family, _, marker, dashed = ARM_SPEC.get(a, ("?", 0, "o", False))
    return {"color": arm_color(a, mode), "marker": marker,
            "dashes": (4.5, 1.8) if dashed else (None, None),
            "lw": lw + (0.5 if family is None else 0.0),
            "zorder": 2 if family is None else 3}


def arm_palette(arms, mode="light"):
    """{arm: colour} for a collection of arms, in the given order."""
    return {a: arm_color(a, mode) for a in arms}


def arm_ink(arm, target=0.62, mode="light"):
    """The arm's colour, darkened enough to read as TEXT on white.

    A line at 2pt and a 9pt label need different luminance: the light step of a
    family is fine as a stroke and near-illegible as a glyph. Same hue, so the
    identity survives.
    """
    r, g, b = mcolors.to_rgb(arm_color(arm, mode))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum <= target:
        return (r, g, b)
    k = target / max(lum, 1e-6)
    return (r * k, g * k, b * k)


def wrap_arm(name, width=13):
    """Arm name broken onto at most two lines at a separator, for an x tick.

    Mechanism names ("PiWM-gate-sup-sigmoid") are long enough that nine of them on
    one axis overlap. Breaking at a hyphen keeps each fragment meaningful, which
    truncation would not: "PiWM-gate-sup-s..." and "PiWM-gate-sup-so..." are the
    same label to a reader.
    """
    n = str(name)
    if len(n) <= width:
        return n
    parts = re.split(r"[-_]", n)
    head, out = "", []
    for i, part in enumerate(parts):
        cand = f"{head}-{part}" if head else part
        if len(cand) > width and head:
            out.append(head)
            head = part
        else:
            head = cand
    out.append(head)
    return "\n".join(out[:2]) if len(out) <= 2 else out[0] + "\n" + "-".join(out[1:])


def needs_facet(arms):
    """True when `arms` is too many to separate by colour on a single axis.

    Callers use this to choose between an overlay and a facet grid rather than
    guessing. Nine overlaid arms is the current suite's most common unreadable
    panel, and it is unreadable for a measurable reason.
    """
    return len({canon_arm(a) for a in arms}) > FACET_ABOVE


def symmetric_limits(values, floor=1e-12):
    """(-m, +m) covering `values`, so a diverging map is centred exactly at zero.

    Letting matplotlib autoscale a diverging map is the classic signed-data bug: an
    asymmetric range puts the map's white point somewhere other than 0, and every
    cell then reads with the wrong sign.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    m = float(np.max(np.abs(v))) if v.size else 0.0
    m = max(m, floor)
    return -m, m


def epoch_colors(n, cmap=SEQ, lo=0.38, hi=0.95):
    """n sequential colours for an ordered series (epochs, steps, horizons).

    `lo` starts well past the ramp's near-white end: those steps are for FILLS on a
    white surface, and a line or a label drawn in them is invisible.
    """
    n = max(int(n), 1)
    return plt.get_cmap(cmap)(np.linspace(lo, hi, n))


def head_colors(n_heads):
    """Colours for the union head's J readouts.

    Heads are a small ORDERED set, not campaign arms, so they take steps of the
    sequential ramp rather than categorical slots: a head index has a natural order,
    and spending arm hues on it would put "head 2" and "Step 3" in the same colour
    on pages that show both. `tab10` was doing exactly that.
    """
    return epoch_colors(max(int(n_heads), 1), cmap=SEQ_ALT, lo=0.30, hi=0.92)


def arm_ramp(arm, mode="light"):
    """A one-hue sequential map in the ARM's own colour, light -> dark.

    Lets a single trace carry two channels at once: hue is the arm's identity,
    lightness is time within the run. Used by `phase_plane`, where "where did this
    arm walk to, and when" is the whole question.
    """
    return _ramp(arm_color(arm, mode))


# --- shared figure furniture ----------------------------------------------------

def _new_fig(figsize, **kw):
    fig = plt.figure(figsize=figsize, facecolor="white", **kw)
    return fig


def _subplots(*a, **kw):
    kw.setdefault("facecolor", "white")
    fig, axes = plt.subplots(*a, **kw)
    fig.patch.set_facecolor("white")
    return fig, axes


#: figure label marking a panel that had nothing to draw; see `is_no_data`
NO_DATA = "lpwm_no_data"


def is_no_data(fig):
    """True if `fig` is a no_data placeholder rather than a real panel.

    Panels never return None, so callers that want to distinguish "rendered" from
    "had nothing to render" -- the offline suite's skip accounting, and the live
    logger deciding whether a panel is worth a wandb slot -- need this rather than a
    truthiness check.
    """
    return getattr(fig, "get_label", lambda: "")() == NO_DATA


def no_data(message, figsize=(6.4, 2.4), title=None):
    """A figure that says why it is empty, instead of an exception or a blank panel.

    Live training logs this rather than nothing at all: an empty panel on the wandb
    page is indistinguishable from a broken logger, whereas this names the input
    that was missing.
    """
    fig, ax = _subplots(figsize=figsize)
    fig.set_label(NO_DATA)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10.5)
    ax.text(0.5, 0.5, str(message), ha="center", va="center", fontsize=10,
            color="0.35", wrap=True, transform=ax.transAxes)
    return fig


def _n_note(ax, text, loc="upper right"):
    """Sample size, on the figure. A panel whose n is only in the caller's head is
    not reviewable, and n is what decides whether any of these shapes mean anything.
    """
    xy = {"upper right": (0.99, 0.99, "right", "top"),
          "upper left": (0.01, 0.99, "left", "top"),
          "lower right": (0.99, 0.01, "right", "bottom"),
          "lower left": (0.01, 0.01, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=7.5, color="0.35",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.78))


def _finite(x):
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _declutter(ax, items, *, gap=0.052, lift=0.038, fontsize=8):
    """Place point labels in axes-fraction space without letting them overlap.

    Nine arms whose sparsity differs by 2% land within a few pixels of each other,
    and matplotlib will happily stack all nine labels on the same line. Greedy
    vertical separation of labels whose horizontal extents overlap costs nothing and
    is the difference between a readable panel and an unreadable one.

    items : list of (x_data, y_data, text, colour).
    """
    if not items:
        return
    placed = []
    prepared = []
    for x, y, text, color in items:
        fx, fy = _axes_frac(ax, x, y)
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        prepared.append([float(fx), float(fy) + lift, text, color])
    prepared.sort(key=lambda r: -r[1])
    for row in prepared:
        fx, fy, text, color = row
        half = 0.5 * (0.0075 * fontsize * max(len(text), 1))
        for px, py, pw in placed:
            if abs(fx - px) < (half + pw) and abs(fy - py) < gap:
                fy = py - gap
        placed.append((fx, fy, half))
        ax.annotate(text, (np.clip(fx, 0.02, 0.98), np.clip(fy, 0.015, 0.985)),
                    xycoords="axes fraction", ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                              edgecolor="none", alpha=0.72))


def _axes_frac(ax, x, y):
    """(x, y) in data units -> axes fraction, WITHOUT needing a draw.

    The obvious `ax.transData + ax.transAxes.inverted()` silently returns the input
    unchanged before the first draw: until then the axes bbox is the unit square, so
    the composite is the identity. Every label then lands at its data coordinate
    interpreted as a fraction, which is why decluttered labels used to pile up on
    top of each other in panels that were saved without an explicit draw. Deriving
    the fraction from the limits is exact and has no such ordering hazard.
    """
    def frac(v, lo, hi, log):
        if log:
            if min(v, lo, hi) <= 0:
                return np.nan
            v, lo, hi = np.log10(v), np.log10(lo), np.log10(hi)
        return (v - lo) / (hi - lo) if hi != lo else 0.5
    return (frac(x, *ax.get_xlim(), ax.get_xscale() == "log"),
            frac(y, *ax.get_ylim(), ax.get_yscale() == "log"))


def declutter(ax, items, *, gap=0.052, lift=0.038, fontsize=8):
    """Public alias for the greedy label placer, for callers outside this module.

    Labels land INSIDE the axes on a white pad rather than in the right margin:
    a margin label collides with the next subplot in any multi-panel figure, and
    which subplot it lands on depends on the layout, so it cannot be tuned away.
    """
    return _declutter(ax, items, gap=gap, lift=lift, fontsize=fontsize)


def _density(samples, edges, smooth=1.5):
    """Histogram of `samples` on `edges`, lightly smoothed, peak-normalised to 1.

    Peak-normalised because a ridgeline compares SHAPES across epochs; normalising
    by area instead would make a sharpening distribution grow vertically and read as
    "more mass", which is the opposite of what happened.
    """
    h, _ = np.histogram(samples, bins=edges)
    h = h.astype(float)
    if smooth and smooth > 0 and h.size >= 3:
        r = int(max(1, round(3 * smooth)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
        h = np.convolve(h, k / k.sum(), mode="same")
    return h / h.max() if h.max() > 0 else h


# --- 1. ridgeline ---------------------------------------------------------------

def ridgeline(series, labels=None, *, bins=72, xlabel=r"nonzero activation magnitude  $|z_i|$",
              title=None, subtitle=None, xlim=None, overlap=0.62, cmap=SEQ,
              max_rows=14):
    """Distribution SHAPE per epoch, one filled ridge per epoch.

    The panel exists because a mean line cannot distinguish "the code shrank" from
    "the code split into a near-zero mode plus a surviving mode" -- and that
    distinction is the whole Pi-WM sparsification claim. Ridges are drawn newest at
    the bottom with a white separator so overlapping tails stay readable.

    series : list of 1-D sample arrays, one per epoch (oldest first), OR a tuple
             (edges, matrix) of pre-binned rows, which is what the live path uses so
             it never has to keep raw samples around.
    labels : row labels; defaults to 'ep 1..n'.
    """
    edges, rows = None, None
    if isinstance(series, tuple) and len(series) == 2:
        edges = np.asarray(series[0], dtype=float)
        mat = np.asarray(series[1], dtype=float)
        if mat.ndim == 1:
            mat = mat[None]
        rows = [r / r.max() if np.isfinite(r).any() and np.nanmax(r) > 0 else r
                for r in np.nan_to_num(mat)]
        counts = [float(np.nansum(r)) for r in np.nan_to_num(mat)]
    else:
        samples = [_finite(s) for s in (series or [])]
        samples = [s for s in samples if s.size]
        if not samples:
            return no_data("no nonzero activations recorded yet\n"
                           "(needs at least one logged batch with a rectified link)",
                           title=title)
        lo = min(float(s.min()) for s in samples)
        hi = max(float(np.quantile(s, 0.999)) for s in samples)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, int(bins) + 1)
        rows = [_density(s, edges) for s in samples]
        counts = [float(s.size) for s in samples]
    if rows is None or not len(rows):
        return no_data("no distributions to draw", title=title)

    if labels is None:
        labels = [f"ep {i + 1}" for i in range(len(rows))]
    labels = list(labels)[: len(rows)]
    # too many epochs makes every ridge a sliver: keep the newest `max_rows`
    if len(rows) > max_rows:
        rows, labels = rows[-max_rows:], labels[-max_rows:]
        counts = counts[-max_rows:]

    x = 0.5 * (edges[:-1] + edges[1:])
    n = len(rows)
    fig, ax = _subplots(figsize=(7.6, 1.05 + 0.42 * n))
    cols = epoch_colors(n, cmap)
    for k, y in enumerate(rows):
        base = -k * overlap
        ax.fill_between(x, base, base + np.nan_to_num(y), color=cols[k], alpha=0.9,
                        lw=0, zorder=n - k)
        ax.plot(x, base + np.nan_to_num(y), color="white", lw=1.1, zorder=n - k)
    ax.set_yticks([-k * overlap for k in range(n)])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*(xlim if xlim else (edges[0], edges[-1])))
    ax.spines[["left", "right", "top"]].set_visible(False)
    head = title or "Ridgeline: distribution of surviving magnitudes per epoch"
    sub = subtitle or ("a second mode growing near zero = the code is SPARSIFYING, "
                       "not merely shrinking")
    ax.set_title(f"{head}\n{sub}", fontsize=10.5)
    _n_note(ax, f"n = {int(sum(counts)):,} samples over {n} rows")
    fig.tight_layout()
    return fig


# --- 2. joint hexbin with marginals ---------------------------------------------

def joint_hexbin(x, y, onset=None, *, xlabel=r"$S_{world}$  (observed support change)",
                 ylabel=r"$S_{model}$  (predicted-vs-actual mismatch)", title=None,
                 gridsize=40, onset_label="contact onset"):
    """The two support-change statistics as a JOINT density, onsets overlaid in teal.

    Step 1's claim is that the two statistics fire at different TIMES, so the object
    that can falsify it is the joint distribution and where the contact onsets sit
    inside it -- not two separate marginal curves. Colour is log-count, because the
    interesting structure is in the tail and a linear count map shows only the mode.

    Axes: joint, top marginal, right marginal (plus a colourbar axis).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if onset is not None:
        onset = np.asarray(onset).ravel()[:n]
        on = m & (np.nan_to_num(onset.astype(float)) > 0.5)
    else:
        on = np.zeros(n, dtype=bool)
    x, y, on = x[m], y[m], on[m]
    if x.size < 3:
        return no_data("fewer than 3 finite (S_world, S_model) pairs\n"
                       "(needs a rectified link and at least one multi-frame batch)",
                       title=title)

    fig = _new_fig((6.9, 6.5))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  hspace=0.05, wspace=0.05, figure=fig)
    axj = fig.add_subplot(gs[1, 0])
    hb = axj.hexbin(x, y, gridsize=int(gridsize), cmap=f"{SEQ_ALT}_r", bins="log",
                    mincnt=1)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    axj.plot(lim, lim, color="grey", ls="--", lw=1, zorder=2)
    if on.any():
        axj.scatter(x[on], y[on], s=18, facecolor="none", edgecolor=CONTACT, lw=1.0,
                    label=f"{onset_label} (n={int(on.sum())})", zorder=3)
        axj.legend(frameon=False, fontsize=8, loc="lower right")
    axj.set(xlabel=xlabel, ylabel=ylabel)
    axj.grid(alpha=0.15)

    axt = fig.add_subplot(gs[0, 0], sharex=axj)
    axt.hist(x, 50, color="#4c72b0", alpha=0.85)
    if on.any():
        axt.hist(x[on], 50, color=CONTACT)
    axt.axis("off")
    axr = fig.add_subplot(gs[1, 1], sharey=axj)
    axr.hist(y, 50, orientation="horizontal", color="#c44e52", alpha=0.85)
    if on.any():
        axr.hist(y[on], 50, orientation="horizontal", color=CONTACT)
    axr.axis("off")
    # the top-right cell of the joint grid is dead space; putting the colourbar
    # there leaves the right marginal its full width instead of a sliver
    cax = fig.add_subplot(gs[0, 1])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0, pos.width * 0.30, pos.height])
    fig.colorbar(hb, cax=cax, label="frames per hex (log)")
    cax.yaxis.set_label_position("right")

    head = title or "Joint density of the two support-change statistics"
    fig.suptitle(f"{head}\nn = {x.size:,} frames"
                 + (f",  {int(on.sum())} contact onsets (teal)" if on.any() else ""),
                 y=0.97, fontsize=10.5)
    return fig


# --- 3. peri-event raster + average ---------------------------------------------

def peri_event_raster(panels, lags=None, *, title=None,
                      xlabel="frames relative to contact onset", ci=0.95):
    """One row per contact onset, plus the mean +- CI underneath, per statistic.

    A mean-only peri-event curve cannot say whether an effect is every event doing
    it a little or three events doing it enormously, and those two have completely
    different implications for the Step 1 gate. The raster answers that directly;
    the dotted line marks each statistic's own peak lag, which is the lead/lag the
    gate is actually about.

    panels : list of (name, rows) where rows is (n_events, n_lags).
    """
    panels = [(nm, np.atleast_2d(np.asarray(r, dtype=float)))
              for nm, r in (panels or []) if r is not None and np.size(r)]
    panels = [(nm, r) for nm, r in panels if r.ndim == 2 and r.shape[0] >= 1]
    if not panels:
        return no_data("no contact onsets with a full window\n"
                       "(needs an env-eval rollout whose states cross the contact "
                       "threshold away from the episode edges)", title=title)
    n_lag = min(r.shape[1] for _, r in panels)
    panels = [(nm, r[:, :n_lag]) for nm, r in panels]
    if lags is None:
        half = n_lag // 2
        lags = np.arange(-half, n_lag - half)
    lags = np.asarray(lags, dtype=float)[:n_lag]

    ncol = len(panels)
    fig, axes = _subplots(2, ncol, figsize=(4.7 * ncol, 5.7), squeeze=False,
                          sharex=True,
                          gridspec_kw={"height_ratios": [2.6, 1.4], "hspace": 0.09,
                                       "wspace": 0.34})
    seqs = (SEQ_ALT, SEQ, "cividis")
    lines = ("#c44e52", "#4c72b0", "#8172b3")
    z = 1.96 if ci >= 0.95 else 1.0
    for col, (name, rows) in enumerate(panels):
        finite = rows[np.isfinite(rows).all(axis=1)]
        if finite.size == 0:
            finite = np.nan_to_num(rows)
        ax = axes[0][col]
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        if not np.isfinite(vmin) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmin(finite)) + 1.0
        im = ax.imshow(finite, aspect="auto", cmap=seqs[col % len(seqs)],
                       extent=[lags[0], lags[-1], finite.shape[0], 0],
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axvline(0, color="white", lw=1.6)
        ax.set_title(f"{name}: one row per onset event  (n={finite.shape[0]})",
                     fontsize=10)
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)

        a2 = axes[1][col]
        mu = np.nanmean(finite, axis=0)
        se = np.nanstd(finite, axis=0) / max(np.sqrt(finite.shape[0]), 1.0)
        c = lines[col % len(lines)]
        a2.plot(lags, mu, color=c, lw=2.2, marker="o", ms=3.6)
        a2.fill_between(lags, mu - z * se, mu + z * se, color=c, alpha=0.25, lw=0)
        a2.axvline(0, color="k", lw=1.2)
        peak = float(lags[int(np.nanargmax(mu))]) if np.isfinite(mu).any() else 0.0
        a2.axvline(peak, color=c, ls=":", lw=1.4)
        # in a corner, not at the peak: at lag 0 the label would sit straight on the
        # curve it is describing
        a2.text(0.02, 0.93, f"peak lag {peak:+.0f}", transform=a2.transAxes,
                fontsize=8.5, color=c, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))
        a2.set_xlabel(xlabel)
        a2.grid(alpha=0.25)
    # only the leftmost column gets y labels, so a colourbar never collides with one
    axes[0][0].set_ylabel("onset event")
    axes[1][0].set_ylabel(f"mean $\\pm${int(ci * 100)}% CI")
    fig.suptitle(title or "Peri-event raster + average: does the statistic fire "
                          "BEFORE or AFTER contact?", y=0.985, fontsize=11)
    return fig


# --- 4. parallel coordinates ----------------------------------------------------

def parallel_coordinates(values, metrics, *, higher_better=None, highlight=(),
                         title=None, ylabel="normalised within metric (0=worst, 1=best)",
                         already_normalised=False):
    """Every arm across every metric at once, normalised within metric.

    Bars can show one metric; the campaign's real question is the TRADE-OFF (an arm
    that wins on sparsity and loses on success is not an improvement). Two or three
    arms are highlighted and the rest are ghosted, because nine equally-weighted
    lines is a plate of spaghetti.

    values : {arm: sequence of len(metrics)}; NaNs are allowed and simply break the
             line, rather than being imputed.
    higher_better : per-metric +1 / -1; -1 flips the metric so 1 always means best.
    """
    metrics = list(metrics or [])
    values = {a: np.asarray(v, dtype=float) for a, v in (values or {}).items()
              if v is not None and len(v)}
    if not values or not metrics:
        return no_data("no (arm x metric) table to draw\n"
                       "(needs a campaign.json with at least two arms)", title=title)
    arms = list(values)
    mat = np.full((len(arms), len(metrics)), np.nan)
    for i, a in enumerate(arms):
        v = values[a][: len(metrics)]
        mat[i, : len(v)] = v

    if not already_normalised:
        hb = np.ones(len(metrics)) if higher_better is None else \
            np.asarray(higher_better, dtype=float)[: len(metrics)]
        norm = np.full_like(mat, np.nan)
        for j in range(len(metrics)):
            col = mat[:, j]
            f = col[np.isfinite(col)]
            if f.size == 0:
                continue
            lo, hi = float(f.min()), float(f.max())
            unit = np.full_like(col, 0.5) if hi <= lo else (col - lo) / (hi - lo)
            norm[:, j] = unit if hb[j] >= 0 else 1.0 - unit
        mat = norm

    highlight = [canon_arm(h) for h in (highlight or [])]
    if not highlight:  # default: the controls plus the most extreme arm
        highlight = [a for a in arms if canon_arm(a) in CONTROL_ARMS][:2]
    fig, ax = _subplots(figsize=(1.35 * max(len(metrics), 4) + 2.4, 4.8))
    xs = np.arange(len(metrics))
    for x in xs:
        ax.axvline(x, color="grey", lw=0.8, alpha=0.5, zorder=0)
    for i, a in enumerate(arms):
        hot = canon_arm(a) in highlight
        lw, alpha, zo = (3.0, 1.0, 3) if hot else (1.3, 0.45, 2)
        ax.plot(xs, mat[i], color=arm_color(a), lw=lw, alpha=alpha, marker="o",
                ms=5 if hot else 3.5, label=canon_arm(a), zorder=zo)
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set(ylabel=ylabel, ylim=(-0.06, 1.06))
    ax.set_title(title or "Parallel coordinates: multivariate trade-offs across arms",
                 fontsize=11)
    _n_note(ax, f"{len(arms)} arms x {len(metrics)} metrics", loc="lower left")
    ax.legend(frameon=False, fontsize=8, ncol=min(5, max(len(arms), 1)),
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    return fig


# --- 5. small multiples / trellis ------------------------------------------------

def small_multiples(series, ref=None, *, ncol=4, xlabel="epoch", ylabel=None,
                    title=None, ref_label="upstream control", band=None):
    """One panel per arm on shared axes, with the control ghosted in EVERY panel.

    Nine overlaid trajectories are unreadable and nine independent plots are not
    comparable. Small multiples give both: the shared axes make the panels
    comparable, and repeating the ghosted reference means every single panel already
    contains its own baseline, so no cross-panel eye-tracking is needed.

    series : {arm: (x, y)} or {arm: (x, y, halfwidth)}.
    ref    : (x, y) drawn in grey behind every panel.
    """
    items = []
    for a, s in (series or {}).items():
        if s is None or len(s) < 2:
            continue
        x, y = np.asarray(s[0], dtype=float), np.asarray(s[1], dtype=float)
        n = min(x.size, y.size)
        if n < 2:
            continue
        hw = np.asarray(s[2], dtype=float)[:n] if len(s) > 2 and s[2] is not None else None
        items.append((a, x[:n], y[:n], hw))
    if not items:
        return no_data("no per-arm trajectory to draw\n"
                       "(needs at least one run with 2+ logged points for this metric)",
                       title=title)

    ncol = max(1, min(int(ncol), len(items)))
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = _subplots(nrow, ncol, figsize=(2.9 * ncol, 2.5 * nrow + 0.5),
                          squeeze=False, sharex=True, sharey=True)
    flat = axes.ravel()
    rx = ry = None
    if ref is not None and len(ref) >= 2:
        rx, ry = np.asarray(ref[0], dtype=float), np.asarray(ref[1], dtype=float)
        k = min(rx.size, ry.size)
        rx, ry = rx[:k], ry[:k]
    for ax, (a, x, y, hw) in zip(flat, items):
        if rx is not None and rx.size >= 2:
            ax.plot(rx, ry, color=GHOST, lw=1.3, zorder=1, label=ref_label)
        c = arm_color(a)
        if hw is not None:
            ax.fill_between(x, y - hw, y + hw, color=c, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, y, color=c, lw=2.0, zorder=3)
        ax.set_title(canon_arm(a), fontsize=9, color=arm_ink(a))
        ax.grid(alpha=0.25)
    for ax in flat[len(items):]:
        ax.set_axis_off()
    # sharex hides tick labels on every row but the last, so a partly-filled last
    # row would silently strip the x axis off the panels sitting above its gaps
    for k, ax in enumerate(flat[:len(items)]):
        below = k + ncol
        if below >= len(items):
            ax.set_xlabel(xlabel)
            ax.tick_params(labelbottom=True)
    if ylabel:
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
    if rx is not None and rx.size >= 2:
        flat[0].legend(frameon=False, fontsize=7.5, loc="best")
    head = title or (f"Small multiples: per-arm trajectory "
                     f"(grey = {ref_label} in every panel)")
    n_pts = int(sum(x.size for _, x, _, _ in items))
    fig.suptitle(f"{head}\n{len(items)} arms, {n_pts:,} logged points, shared axes",
                 y=1.0, fontsize=10.5)
    fig.tight_layout()
    return fig


# --- 6. diverging effect map -----------------------------------------------------

def effect_map(eff, arms, metrics, *, sig=None, title=None,
               cbar_label="standardised effect vs matched control", n_note=None,
               annotate=True, symlog=None):
    """(arm x metric) grid of SIGNED standardised effects, diverging and zero-centred.

    This is the panel that replaces most of the bar charts: a bar chart shows one
    metric for a few arms and needs one figure per metric, while this shows sign,
    magnitude and across-seed significance for the whole campaign in one grid. Cells
    whose effect exceeds one across-seed std are boxed, so "big" and "reliably
    nonzero" stay visually separate.
    """
    arms, metrics = list(arms or []), list(metrics or [])
    eff = np.asarray(eff, dtype=float)
    if eff.size == 0 or not arms or not metrics:
        return no_data("no effect table to draw\n"
                       "(needs >=2 arms sharing seeds with a matched control)",
                       title=title)
    eff = np.atleast_2d(eff)[: len(arms), : len(metrics)]
    vmin, vmax = symmetric_limits(eff)
    sig = np.zeros_like(eff, dtype=bool) if sig is None else \
        np.atleast_2d(np.asarray(sig, dtype=bool))[: eff.shape[0], : eff.shape[1]]

    # One arm can move a metric by 20 seed-sd (k-WTA on l0) while every other cell
    # lives inside +-3, and on a linear map that single cell renders the rest of the
    # grid white. Compressing beyond +-1 sd keeps the zero centre and the sign --
    # which is what a diverging map must never lose -- while leaving the small
    # effects, the ones the gates are actually about, distinguishable.
    a = np.abs(eff[np.isfinite(eff)])
    span = max(abs(vmin), abs(vmax))
    if symlog is None:
        symlog = bool(a.size and span > 4.0 * max(np.median(a), 1e-9)
                      and span > 3.0)
    norm = mcolors.SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax, base=10) \
        if symlog else mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = _subplots(figsize=(1.15 * len(metrics) + 3.4, 0.52 * len(arms) + 2.3))
    im = ax.imshow(eff, cmap=DIV, norm=norm, aspect="auto")
    # what "a strong colour" means, for the white-vs-ink text decision below
    span = 1.0 if symlog else span
    for i in range(eff.shape[0]):
        for j in range(eff.shape[1]):
            v = eff[i, j]
            if annotate:
                txt = "n/a" if not np.isfinite(v) else f"{v:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(v) and abs(v) > 0.55 * span
                        else INK,
                        fontweight="bold" if sig[i, j] else "normal")
            if sig[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="k", lw=1.8))
    ax.set_xticks(range(eff.shape[1]))
    ax.set_xticklabels(metrics[: eff.shape[1]], rotation=22, ha="right", fontsize=9)
    ax.set_yticks(range(eff.shape[0]))
    ax.set_yticklabels([canon_arm(a) for a in arms[: eff.shape[0]]], fontsize=9)
    for tick, a in zip(ax.get_yticklabels(), arms[: eff.shape[0]]):
        tick.set_color(arm_ink(a))
    fig.colorbar(im, ax=ax,
                 label=cbar_label + (" [symlog beyond $\\pm$1]" if symlog else ""))
    head = title or ("Effect map: signed, standardised, boxed where "
                     "|effect| > 1 seed-std")
    sub = f"{eff.shape[0]} arms x {eff.shape[1]} metrics, " \
          f"{int(sig.sum())} cells beyond 1 seed-std"
    ax.set_title(f"{head}\n{sub}" + (f";  {n_note}" if n_note else ""), fontsize=10.5)
    fig.tight_layout()
    return fig


# --- 7. ECDF overlay -------------------------------------------------------------

def ecdf_overlay(samples, *, colors=None, xlabel=r"per-sample $\|z\|_0$  (units active)",
                 ylabel="empirical CDF", title=None, log_x="auto", vlines=(),
                 legend_loc="lower right"):
    """ECDF per group, which is the right tool for COMPARING distributions.

    Overlaid histograms of five arms are unreadable and bin-width dependent; ECDFs
    have no bins, so a shift, a spread change and a hard cap (k-WTA pinning L0 at
    exactly k) are all directly legible, and the median is on the legend.

    samples : {label: 1-D array}. Coloured by arm unless `colors` overrides.
    vlines  : (x, label) pairs, e.g. the k-WTA k.
    log_x   : "auto" (default) uses a log axis only when the data spans >= 1 decade;
              True / False force it. A forced choice is never overridden.
    """
    groups = {k: _finite(v) for k, v in (samples or {}).items() if v is not None}
    groups = {k: v for k, v in groups.items() if v.size}
    if not groups:
        return no_data("no per-sample sparsity samples\n"
                       "(needs dist/z_l0_per_sample, i.e. a tier-2 diagnostic log)",
                       title=title)
    # A log x-axis earns its place across decades -- k-WTA at k=8 against a dense arm
    # at 190. Inside one decade it only crowds the ticks into "1.93e2 1.94e2 ...",
    # colliding with each other. "auto" picks; an explicit True/False is OBEYED,
    # because a caller that asked for a scale outranks a heuristic about the data.
    if log_x == "auto":
        pos = np.concatenate([v[v > 0] for v in groups.values() if (v > 0).any()]
                             or [np.array([])])
        log_x = bool(pos.size and pos.max() / max(pos.min(), 1e-12) >= 10.0)
    log_x = bool(log_x)
    fig, ax = _subplots(figsize=(7.2, 4.5))
    for label, v in groups.items():
        x = np.sort(v)
        if log_x:
            x = np.clip(x, max(np.min(x[x > 0]) if (x > 0).any() else 1e-3, 1e-3), None)
        c = (colors or {}).get(label) or arm_color(label)
        ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=c, lw=2.1,
                label=f"{label}  (median {np.median(v):.3g}, n={v.size:,})")
    for item in (vlines or ()):
        xv, lab = (item if isinstance(item, (tuple, list)) else (item, None))
        ax.axvline(float(xv), color="0.4", ls=":", lw=1.2)
        if lab:
            ax.annotate(lab, (float(xv), 0.5), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color="0.3")
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel + ("  [log scale]" if log_x else ""), ylabel=ylabel,
           ylim=(0, 1.02))
    total = int(sum(v.size for v in groups.values()))
    ax.set_title((title or "ECDF of code sparsity")
                 + f"\n{len(groups)} groups, n = {total:,} samples total",
                 fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    fig.tight_layout()
    return fig


# --- 8. streamgraph of head usage ------------------------------------------------

#: Head colours are DERIVED -- see `head_colors`, which steps the sequential ramp.
#: A head index is an ORDERED small set, not a campaign arm, so it must not consume
#: categorical slots. The literal that used to live here was four hexes lifted from
#: the previous arm palette, which meant "head 1" and a Step-2 arm rendered in the
#: same colour on any page showing both.
HEAD_COLORS = None


def head_stream(panels, *, title=None, xlabel="epoch", ylabel=r"head usage $\bar p_j$",
                collapse_line=0.9, colors=None):
    """Union-head usage as a stacked stream over training.

    Collapse is the failure mode this arm has to rule out, and it has an unmistakable
    signature here: one colour swallowing the band. Healthy specialisation stays
    braided. The dashed line is the pre-registered 0.9 precondition -- above it the
    J-head model is numerically J=1 and cannot falsify anything.

    panels : list of (title, x, P) with P of shape (J, T) or (T, J).
    """
    clean = []
    for item in (panels or []):
        if item is None or len(item) < 3:
            continue
        name, x, p = item[0], np.asarray(item[1], dtype=float), np.asarray(item[2], dtype=float)
        if p.ndim != 2 or p.size == 0 or x.size < 2:
            continue
        if p.shape[0] == x.size and p.shape[1] != x.size:  # (T, J) -> (J, T)
            p = p.T
        t = min(p.shape[1], x.size)
        p, x = p[:, :t], x[:t]
        p = np.nan_to_num(p, nan=0.0)
        p = np.clip(p, 1e-6, None)
        s = p.sum(axis=0, keepdims=True)
        p = p / np.where(s > 0, s, 1.0)
        clean.append((name, x, p))
    if not clean:
        return no_data("no union-head usage recorded\n"
                       "(only arms with n_heads > 1 emit heads/*; J=1 arms have no "
                       "stream to draw)", title=title)

    fig, axes = _subplots(1, len(clean), figsize=(5.3 * len(clean), 4.0),
                          squeeze=False, sharey=True)
    for ax, (name, x, p) in zip(axes[0], clean):
        j = p.shape[0]
        pal = list(colors) if colors else list(head_colors(j))
        ax.stackplot(x, p, colors=[pal[k % len(pal)] for k in range(j)],
                     labels=[f"head {k}" for k in range(j)], alpha=0.95)
        ax.axhline(collapse_line, color="k", ls="--", lw=1.2)
        ax.set(xlabel=xlabel, ylim=(0, 1))
        ax.set_title(name, fontsize=10)
        _n_note(ax, f"J={j}, {x.size} logged points", loc="lower left")
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="center left",
                       bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(title or f"Streamgraph: does the union head specialise or collapse? "
                          f"(dashed = {collapse_line} precondition)",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


# --- 9. bubble: regulariser floor vs sparsity ------------------------------------

def bubble_reg_vs_sparsity(points, *, ref_level=None, title=None,
                           xlabel=r"$l_0$ fraction", ylabel="RDMReg loss",
                           size_label="CEM success", log_x=True,
                           ref_text="upstream RDMReg level -- an arm ABOVE this is "
                                    "fighting the regulariser"):
    """The mu-matching claim as a RELATIONSHIP, not two unrelated bar charts.

    If density matching worked, the k-WTA arms should sit at low l0 WITHOUT being
    pushed up onto a raised RDMReg floor. That is a statement about a 2-D position,
    which two bar charts structurally cannot make; here it is one glance, with
    bubble area carrying the outcome metric and one bubble per seed so the spread is
    visible rather than averaged away.

    points : {arm: [(l0, reg, size), ...]} -- one tuple per seed.
    """
    clean = {}
    for a, rows in (points or {}).items():
        keep = []
        for r in (rows or []):
            if r is None or len(r) < 2:
                continue
            l0, reg = float(r[0]), float(r[1])
            sz = float(r[2]) if len(r) > 2 and r[2] is not None and np.isfinite(r[2]) \
                else np.nan
            if np.isfinite(l0) and np.isfinite(reg) and (l0 > 0 or not log_x):
                keep.append((l0, reg, sz))
        if keep:
            clean[a] = keep
    if not clean:
        return no_data("no (l0, RDMReg) points\n"
                       "(needs both sparsity/l0_frac and the reg loss for >=1 run)",
                       title=title)

    sizes = np.array([p[2] for rows in clean.values() for p in rows], dtype=float)
    finite = sizes[np.isfinite(sizes)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    fig, ax = _subplots(figsize=(7.9, 5.3))
    n_pts = 0
    for a, rows in clean.items():
        c = arm_color(a)
        for l0, reg, sz in rows:
            frac = 0.5 if not np.isfinite(sz) or hi <= lo else (sz - lo) / (hi - lo)
            ax.scatter(l0, reg, s=90 + 780 * frac ** 2, color=c, alpha=0.45,
                       edgecolor="k", lw=0.7, zorder=2)
            n_pts += 1
    if log_x:
        ax.set_xscale("log")
    if ref_level is not None and np.isfinite(ref_level):
        ax.axhline(float(ref_level), color="grey", ls="--", lw=1.2, zorder=1)
        ax.annotate(ref_text, (0.02, float(ref_level)), xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points", fontsize=8, color="0.3")
    # headroom BEFORE labelling, so the declutter pass lays labels out against the
    # final axes limits rather than ones tight_layout is about to change
    ax.margins(x=0.13, y=0.22)
    _declutter(ax, [(float(np.mean([p[0] for p in rows])),
                     float(np.mean([p[1] for p in rows])),
                     canon_arm(a), arm_ink(a)) for a, rows in clean.items()])
    ax.set(xlabel=xlabel + ("  [log scale]" if log_x else ""), ylabel=ylabel)
    head = title or f"Bubble: sparsity vs regulariser floor (area = {size_label})"
    n_line = (f"{len(clean)} arms, {n_pts} points; bubble area = {size_label} over "
              f"[{lo:.3g}, {hi:.3g}]" if finite.size else
              f"{len(clean)} arms, {n_pts} points; {size_label} unavailable "
              "(all bubbles equal)")
    ax.set_title(f"{head}\n{n_line}", fontsize=10.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --- auxiliary: latent error aligned with a rendered env rollout ------------------

def latent_error_timeline(frames_x, curves, *, onset=None, xlabel="env-eval frame",
                          title=None, contact_label="contact onset", markers=None):
    """Latent-space error over the SAME trajectory a rendered video shows.

    has_decoder is False for this campaign, so a video can only ever show the real
    environment -- never the model's imagination. This panel is what makes the video
    interpretable anyway: it puts the model's one-step and open-loop latent error on
    the video's own time axis, with contact onsets marked in teal, so "the video
    looks fine but the model is lost after frame 12" is readable.

    curves : {label: 1-D array over frames_x}.
    """
    x = np.asarray(frames_x, dtype=float).ravel()
    curves = {k: np.asarray(v, dtype=float).ravel()
              for k, v in (curves or {}).items() if v is not None and np.size(v)}
    if x.size < 2 or not curves:
        return no_data("no latent error series for the env rollout\n"
                       "(needs an env-eval episode long enough to roll out)",
                       title=title)
    fig, ax = _subplots(figsize=(8.0, 3.9))
    cols = epoch_colors(max(len(curves), 2), SEQ, 0.15, 0.8)
    for (label, y), c in zip(curves.items(), cols):
        n = min(x.size, y.size)
        ax.plot(x[:n], y[:n], lw=2.0, marker="o", ms=3.4, color=c, label=label)
    if onset is not None:
        on = np.asarray(onset, dtype=float).ravel()
        idx = np.flatnonzero(np.nan_to_num(on) > 0.5)
        for k, t in enumerate(idx):
            if t < x.size:
                ax.axvline(x[t], color=CONTACT, lw=1.6, alpha=0.85,
                           label=contact_label if k == 0 else None)
    for item in (markers or ()):
        xv, lab = (item if isinstance(item, (tuple, list)) else (item, None))
        ax.axvline(float(xv), color="0.55", ls="--", lw=1.0)
        if lab:
            ax.annotate(lab, (float(xv), 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -11), textcoords="offset points", fontsize=8,
                        color="0.35")
    ax.set(xlabel=xlabel, ylabel="latent error")
    n_on = int((np.nan_to_num(np.asarray(onset, dtype=float)) > 0.5).sum()) \
        if onset is not None else 0
    ax.set_title((title or "Latent-space error over the rendered env rollout")
                 + f"\nn = {x.size} frames, {n_on} contact onsets (teal)",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


#: The nine approved designs, by the name the suite files them under. Kept as data
#: so train.py, analysis/figures.py and the tests cannot disagree about the list.
NINE = (
    ("ridgeline", ridgeline),
    ("joint_hexbin", joint_hexbin),
    ("peri_event_raster", peri_event_raster),
    ("parallel_coordinates", parallel_coordinates),
    ("small_multiples", small_multiples),
    ("effect_map", effect_map),
    ("ecdf_overlay", ecdf_overlay),
    ("head_stream", head_stream),
    ("bubble_reg_vs_sparsity", bubble_reg_vs_sparsity),
)


# --- 11. estimation plot --------------------------------------------------------

def estimation_plot(seeds, contrasts, *, mde=None, ylabel="CEM success rate",
                    title=None, subtitle=None, order=None):
    """Raw seeds on top, the paired bootstrap difference below. Replaces bar charts.

    A bar of |effect| next to a threshold line -- which is what this suite drew
    before -- destroys the sign, hides the seed spread that decides whether the
    effect is real, and spends colour on a boolean the threshold already encodes.
    The estimation plot is the standard fix: the reader sees the raw data, the point
    estimate and its uncertainty on one axis, in the units of the measurement.

    STATUS is carried by marker FILL (solid = resolved, hollow = inside the detection
    floor), never by hue, so "underpowered" and "which arm" coexist on one mark.
    A hollow dot is the panel's whole point: at n=3 an effect below the MDE is an
    underpowered test, not evidence of no effect.

    seeds     : {arm: [per-seed value, ...]}
    contrasts : [(control_arm, variant_arm), ...] -- paired, seed i to seed i
    mde       : minimum detectable effect; shades a band and sets the fill rule
    """
    seeds = {a: _finite(v) for a, v in (seeds or {}).items()}
    seeds = {a: v for a, v in seeds.items() if v.size}
    pairs = [(c, v) for c, v in (contrasts or []) if c in seeds and v in seeds]
    if not pairs:
        return no_data("no contrast has a control and a variant with seeds\n"
                       "(needs matched seeds for at least one control/variant pair)",
                       title=title)
    if order is None:
        order, seen = [], set()
        for c, v in pairs:
            for a in (c, v):
                if a not in seen:
                    seen.add(a)
                    order.append(a)
    xpos = {a: i for i, a in enumerate(order)}

    fig = _new_fig((max(7.2, 1.45 * len(order) + 2.6), 6.1))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.12], hspace=0.10, figure=fig)
    a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for a in order:
        v, x = seeds[a], xpos[a]
        st = arm_style(a)
        jit = np.linspace(-0.10, 0.10, v.size) if v.size > 1 else np.zeros(1)
        a0.scatter(x + jit, v, s=36, color=st["color"], marker=st["marker"],
                   edgecolor="white", linewidth=1.4, zorder=3)
        a0.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=arm_ink(a), lw=2.3, zorder=4)
    for c, v in pairs:
        a0.plot([xpos[c], xpos[v]], [seeds[c].mean(), seeds[v].mean()],
                color=GHOST, lw=1.0, zorder=1)
    a0.set(ylabel=ylabel, xlim=(-0.6, len(order) - 0.4))
    a0.set_xticks([])
    a0.grid(axis="y", alpha=0.35)
    a0.spines[["top", "right"]].set_visible(False)

    if mde and np.isfinite(mde):
        a1.axhspan(-mde, mde, color="#f2f2f2", zorder=0)
    a1.axhline(0, color="#9a9a9a", lw=1.0, zorder=1)
    n_res = 0
    for c, v in pairs:
        n = min(seeds[c].size, seeds[v].size)
        d = seeds[v][:n] - seeds[c][:n]
        rng = np.random.default_rng(0)
        bs = rng.choice(d, size=(4000, n), replace=True).mean(1) if n > 1 else d
        lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        x, col = xpos[v], arm_color(v)
        if bs.size > 1 and hi > lo:
            dens, edges = np.histogram(bs, 60, density=True)
            yc = 0.5 * (edges[:-1] + edges[1:])
            w = 0.34 * dens / max(dens.max(), 1e-12)
            a1.fill_betweenx(yc, x, x + w, color=col, alpha=0.22, lw=0, zorder=2)
            a1.plot(x + w, yc, color=col, lw=1.1, zorder=3)
        a1.plot([x, x], [lo, hi], color=arm_ink(v), lw=2.4,
                solid_capstyle="round", zorder=4)
        resolved = bool(mde and np.isfinite(mde) and abs(d.mean()) >= mde)
        n_res += resolved
        a1.scatter([x], [d.mean()], s=64, zorder=5, linewidth=1.8,
                   color=col if resolved else "white", edgecolor=arm_ink(v))
        # left of the dot, not above it: the half-violin occupies the right side and
        # the next arm's CI occupies the space above, so "above" collides on both.
        a1.annotate(f"{d.mean():+.3f}", (x, d.mean()), xytext=(-11, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.4, color=arm_ink(v))
    a1.set_xticks(list(xpos.values()))
    a1.set_xticklabels([wrap_arm(a) for a in order], fontsize=7.2)
    for t, a in zip(a1.get_xticklabels(), order):
        t.set_color(arm_ink(a))
    a1.set(ylabel="paired effect vs control\n(bootstrap 95% CI)",
           xlim=(-0.6, len(order) - 0.4))
    a1.grid(axis="y", alpha=0.35)
    a1.spines[["top", "right"]].set_visible(False)

    head = title or "Every arm's seeds, and the paired difference from its control"
    sub = subtitle or ("squares / diamonds = flags-off control  ·  each dot = one seed"
                       + (f"  ·  hollow = inside the n={min(len(v) for v in seeds.values())}"
                          f" detection floor (MDE ±{mde:.3f}), i.e. underpowered, not null"
                          if mde and np.isfinite(mde) else ""))
    a0.set_title(f"{head}\n{sub}", fontsize=10.5, loc="left")
    _n_note(a0, f"{len(pairs)} contrasts, {n_res} resolved", loc="lower left")
    # explicit rather than tight_layout: the two panels come from a GridSpec whose
    # hspace is load-bearing (the seeds must sit directly above their own contrast),
    # and tight_layout would renegotiate it.
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.13)
    return fig


# --- 12. phase plane ------------------------------------------------------------

def phase_plane(traj, *, xlabel=r"code density $\rho$", ylabel="RDMReg loss",
                title=None, subtitle=None, ncol=3, annotate_end=True, smooth=None):
    """Two metrics against EACH OTHER over training, one facet per arm.

    A bar of "final regulariser loss per arm" answers a question nobody asked. The
    question is whether an arm is sitting on an irreducible floor -- and a floor is a
    place a trajectory STOPS MOVING, which is a shape, not a scalar. Reading the
    trace right-to-left in time shows an arm walk its density down and then stall.

    Time is encoded as lightness inside the arm's OWN hue, so one trace carries both
    "which arm" and "when" without spending a second colour channel or a legend.

    traj : {arm: (x_array, y_array)} in training order.
    """
    traj = {a: (np.asarray(x, float), np.asarray(y, float))
            for a, (x, y) in (traj or {}).items()}
    traj = {a: (x, y) for a, (x, y) in traj.items()
            if x.size == y.size and np.isfinite(x).any() and np.isfinite(y).any()}
    if not traj:
        return no_data("no arm logged both metrics over training\n"
                       "(needs two aligned scalar series per run)", title=title)
    arms = list(traj)
    ncol = min(ncol, len(arms))
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = _subplots(nrow, ncol, figsize=(3.7 * ncol, 2.9 * nrow + 0.7),
                          sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, a in zip(flat, arms):
        for b, (bx, by) in traj.items():
            if b != a:
                ax.plot(bx, by, color=GHOST, lw=0.9, zorder=1)
        x, y = traj[a]
        if smooth and smooth > 1 and x.size > smooth:
            k = np.ones(int(smooth)) / int(smooth)
            x = np.convolve(x, k, mode="valid")
            y = np.convolve(y, k, mode="valid")
        t = np.linspace(0, 1, max(x.size - 1, 1))
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        if len(segs):
            lc = LineCollection(segs, cmap=arm_ramp(a), array=t, lw=2.1, zorder=3)
            ax.add_collection(lc)
        ax.scatter([x[0]], [y[0]], s=30, facecolor="white", edgecolor=arm_ink(a),
                   lw=1.5, zorder=4)
        ax.scatter([x[-1]], [y[-1]], s=44, color=arm_ink(a), edgecolor="white",
                   lw=1.4, zorder=5)
        if annotate_end and np.isfinite(y[-1]):
            ax.annotate(f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=7.4, color=arm_ink(a))
        ax.set_title(a, loc="left", fontsize=9.2, color=arm_ink(a), pad=4)
        ax.grid(alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat[len(arms):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    head = title or "Where each arm walks to, and where it stops"
    sub = subtitle or ("hollow = first step,  filled = last step,  colour lightens to "
                       "darkens with training;  grey = every other arm")
    if smooth and smooth > 1:
        sub += f"  ·  traces smoothed over {int(smooth)} logged steps"
    # the end annotation must report the RAW last value, not the smoothed one

    fig.suptitle(f"{head}\n{sub}", fontsize=10.5, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93 if nrow > 1 else 0.86])
    return fig


# --- 13. facet grid -------------------------------------------------------------

def facet_grid(series, *, xlabel="epoch", ylabel=None, title=None, subtitle=None,
               ncol=3, band=None, marks=None, share_y=True):
    """One trajectory per facet, every other trajectory ghosted behind it.

    This is the answer to "nine arms on one axis". `needs_facet()` says when it is
    mandatory: three lightness steps of one family sit 12.9 dE apart against normal
    vision, under the 15 floor, so nine overlaid arms cannot be told apart no matter
    how good the palette is. Ghosting keeps the comparison -- a facet is still read
    against the whole campaign -- while giving each arm an unambiguous line.

    series : {arm: (x, y)} or {arm: y}
    band   : optional {arm: (lo, hi)} seed min-max, drawn as a soft ribbon
    marks  : optional {arm: [x positions]} for resume / preemption rules
    """
    prep = {}
    for a, v in (series or {}).items():
        if isinstance(v, tuple) and len(v) == 2:
            x, y = np.asarray(v[0], float), np.asarray(v[1], float)
        else:
            y = np.asarray(v, float)
            x = np.arange(y.size, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            prep[a] = (x[m], y[m])
    if not prep:
        return no_data("no run carries this metric\n(needs >=2 finite points for one arm)",
                       title=title)
    arms = list(prep)
    ncol = min(ncol, len(arms))
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = _subplots(nrow, ncol, figsize=(3.7 * ncol, 2.5 * nrow + 0.8),
                          sharex=True, sharey=share_y, squeeze=False)
    flat = axes.ravel()
    for ax, a in zip(flat, arms):
        for b, (bx, by) in prep.items():
            if b != a:
                ax.plot(bx, by, color=GHOST, lw=0.9, zorder=1)
        x, y = prep[a]
        st = arm_style(a)
        ax.plot(x, y, color=st["color"], lw=st["lw"], dashes=st["dashes"],
                zorder=3, solid_joinstyle="round")
        if band and a in band:
            lo, hi = np.asarray(band[a][0], float), np.asarray(band[a][1], float)
            if lo.size == x.size:
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.16, lw=0, zorder=2)
        for mx in (marks or {}).get(a, []):
            ax.axvline(mx, color="0.62", lw=0.8, zorder=1)
        ax.scatter([x[-1]], [y[-1]], s=26, color=st["color"], marker=st["marker"],
                   edgecolor="white", lw=1.2, zorder=4)
        ax.annotate(a, (0.035, 0.94), xycoords="axes fraction", fontsize=8.6,
                    color=arm_ink(a), va="top")
        ax.annotate(f"{y[-1]:.4g}", (0.97, 0.07), xycoords="axes fraction",
                    fontsize=7.5, color=arm_ink(a), ha="right")
        ax.grid(alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat[len(arms):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    if ylabel:
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
    head = title or (ylabel or "metric") + " by arm"
    sub = subtitle or ("shared axes, every other arm ghosted behind -- a facet is read "
                       "against the campaign without a legend")
    fig.suptitle(f"{head}\n{sub}", fontsize=10.5, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93 if nrow > 1 else 0.84])
    return fig


# --- 14. arm ridgeline ----------------------------------------------------------

def arm_ridgeline(samples, *, xlabel="value", title=None, subtitle=None,
                  refs=(), bins=90, xlim=None, row_h=0.62):
    """One filled density per ARM, in arm colour. Replaces "mean +- sd" bar charts.

    A mean with an error bar cannot distinguish "centred on target" from "bimodal
    with no mass at the target", and that distinction is exactly what the Step 3
    normalisation check is asking about. The ridge shows the shape; the tick shows
    the mean, so nothing is lost relative to the bar it replaces.

    samples : {arm: 1-D sample array}
    refs    : [(x, label), ...] reference rules, e.g. (0.5, "sigmoid"), (1.0, "target")
    """
    samples = {a: _finite(v) for a, v in (samples or {}).items()}
    samples = {a: v for a, v in samples.items() if v.size}
    if not samples:
        return no_data("no arm logged samples of this quantity", title=title)
    arms = list(samples)
    if xlim is None:
        lo = min(float(v.min()) for v in samples.values())
        hi = max(float(np.quantile(v, 0.995)) for v in samples.values())
        for r, _ in refs:
            lo, hi = min(lo, r), max(hi, r)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        pad = 0.04 * (hi - lo)
        xlim = (lo - pad, hi + pad)
    edges = np.linspace(xlim[0], xlim[1], int(bins) + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = _subplots(figsize=(8.0, 1.6 + row_h * len(arms) * 1.55))
    for i, a in enumerate(reversed(arms)):
        v, y0, col = samples[a], i * 1.0, arm_color(a)
        h = _density(v, edges) * 0.92
        ax.fill_between(xc, y0, y0 + h, color=col, alpha=0.34, lw=0, zorder=10 + i)
        ax.plot(xc, y0 + h, color=arm_ink(a), lw=1.5, zorder=10 + i)
        ax.plot([xc[0], xc[-1]], [y0, y0], color="#cfcfcf", lw=0.7, zorder=9 + i)
        m = float(v.mean())
        ax.plot([m, m], [y0, y0 + 0.30], color=arm_ink(a), lw=2.0, zorder=30 + i)
        # the arm label lives in a gutter OUTSIDE the data area: inside, it lands on
        # whichever ridge happens to be tall there, which is data-dependent and so
        # cannot be fixed by choosing a better constant.
        ax.annotate(a, (-0.012, y0 + 0.10), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8.4, color=arm_ink(a), zorder=40)
        ax.annotate(f"mean {m:.3g}  (n={v.size:,})", (m, y0 + 0.02), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.1, color=arm_ink(a),
                    va="bottom", zorder=40,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.80))
    for r, lab in refs:
        ax.axvline(r, color="#a8a8a8", lw=1.0, zorder=5)
        ax.annotate(lab, (r, len(arms) - 0.02), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.6, color="0.35", va="top")
    ax.set(xlabel=xlabel, xlim=xlim, ylim=(-0.06, len(arms) + 0.05))
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    head = title or "Distribution by arm"
    sub = subtitle or "the whole shape, not its mean -- a mean can sit on target with no mass there"
    ax.set_title(f"{head}\n{sub}", fontsize=10.5, loc="left")
    fig.tight_layout()
    # widen the left margin so the arm-label gutter is inside the canvas
    fig.subplots_adjust(left=max(0.20, fig.subplotpars.left))
    return fig


# --- 15. strip plot -------------------------------------------------------------

def strip_plot(groups, *, xlabel="value", title=None, subtitle=None, ref=None,
               ref_label=None, log_x=False, jitter=0.16, show_dist=True):
    """Every observation as a dot, grouped by arm. Replaces "median per run" bars.

    A bar of the median throws away the spread that answers the actual question --
    a run is slow because it is *sometimes* slow, and a median hides the dips. The
    strip shows every point; the ridge behind it shows where the mass is.

    groups : {arm: 1-D array of observations}
    ref    : optional reference value drawn as a rule (e.g. campaign median)
    """
    groups = {a: _finite(v) for a, v in (groups or {}).items()}
    groups = {a: v for a, v in groups.items() if v.size}
    if not groups:
        return no_data("no run carries this metric", title=title)
    arms = list(groups)
    fig, ax = _subplots(figsize=(8.6, 1.5 + 0.46 * len(arms)))
    rng = np.random.default_rng(0)
    for i, a in enumerate(reversed(arms)):
        v, col = groups[a], arm_color(a)
        y = i + rng.uniform(-jitter, jitter, v.size)
        if show_dist and v.size > 8:
            q = np.percentile(v, [25, 50, 75])
            ax.plot([q[0], q[2]], [i, i], color=col, lw=5.5, alpha=0.26,
                    solid_capstyle="round", zorder=2)
            ax.plot([q[1], q[1]], [i - 0.20, i + 0.20], color=arm_ink(a), lw=2.0, zorder=4)
        ax.scatter(v, y, s=9, color=col, alpha=0.55, lw=0, zorder=3)
        ax.annotate(f"{a}  (n={v.size:,})", (0.0, i + 0.30), xycoords=("axes fraction", "data"),
                    xytext=(2, 0), textcoords="offset points", fontsize=8.0,
                    color=arm_ink(a), va="bottom")
    if ref is not None and np.isfinite(ref):
        ax.axvline(ref, color="0.35", lw=1.1, zorder=5)
        ax.annotate(ref_label or f"{ref:.3g}", (ref, len(arms) - 0.55), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.6, color="0.35")
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylim=(-0.6, len(arms) - 0.1))
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=0.35)
    head = title or "Every observation, by arm"
    sub = subtitle or "bar = interquartile range, tick = median, dots = individual samples"
    ax.set_title(f"{head}\n{sub}", fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig


# --- 16. stat tiles -------------------------------------------------------------

def stat_tiles(tiles, *, title=None, subtitle=None, ncol=None, ax=None):
    """A row of headline numbers. Replaces one-bar and two-bar "charts".

    A two-bar chart of between-arm vs within-arm variance is a picture of one
    number -- the ICC -- drawn as two rectangles whose ratio the reader has to
    estimate by eye. Printing the number is strictly more informative and honest
    about how little there is to see.

    tiles : [(key, value, detail)] or [(key, value, detail, status)] where status is
            one of "good" / "warn" / "crit" / None; status colours the DETAIL line
            and adds a glyph, never the value itself.
    """
    tiles = [t for t in (tiles or []) if t]
    if not tiles:
        return no_data("nothing to summarise", title=title)
    ncol = ncol or min(4, len(tiles))
    nrow = int(np.ceil(len(tiles) / ncol))
    # `ax` lets a composite panel host the tiles directly. The alternative -- render
    # a tile figure and imshow its buffer -- rasterises at the wrong DPI and turns
    # the numbers into mush at any print size.
    host = ax is not None
    if host:
        fig = ax.figure
    else:
        fig, ax = _subplots(figsize=(2.85 * ncol, 1.42 * nrow + (0.62 if title else 0.12)))
    ax.set_axis_off()
    ax.set(xlim=(0, ncol), ylim=(-nrow, 0))
    glyph = {"good": ("▲", "#1a7f37"), "warn": ("◆", "#9a6700"),
             "crit": ("■", "#b3261e"), None: ("", "0.35")}
    for i, t in enumerate(tiles):
        k, v, d = t[0], t[1], (t[2] if len(t) > 2 else "")
        status = t[3] if len(t) > 3 else None
        gy, gc = glyph.get(status, glyph[None])
        cx, cy = i % ncol, -(i // ncol) - 1
        ax.add_patch(FancyBboxPatch((cx + 0.045, cy + 0.10), 0.91, 0.80,
                                    boxstyle="round,pad=0.012,rounding_size=0.045",
                                    fc="#f7f7f6", ec="#e6e6e4", lw=0.8,
                                    transform=ax.transData, zorder=1))
        ax.text(cx + 0.10, cy + 0.78, str(k), fontsize=8.4, color="0.42", va="top", zorder=2)
        ax.text(cx + 0.10, cy + 0.58, str(v), fontsize=20, color="0.08", va="top",
                fontweight="semibold", zorder=2)
        # wrapped, not clipped: a detail line that runs past its tile reads as a
        # label belonging to the NEXT tile, which is worse than a second row
        detail = "\n".join(textwrap.wrap(str(d), 34)[:2]) if d else ""
        ax.text(cx + 0.10, cy + 0.22, (gy + "  " if gy else "") + detail, fontsize=7.6,
                color=gc, va="center", linespacing=1.35, zorder=2)
    if title:
        ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                     fontsize=10.5, loc="left")
    if host:
        return fig
    # not tight_layout: the tiles are drawn in data coordinates on a hidden axis,
    # which tight_layout cannot measure and warns about.
    fig.subplots_adjust(left=0.012, right=0.988, top=0.86 if title else 0.98, bottom=0.03)
    return fig


# --- 17. forest plot ------------------------------------------------------------

def forest(rows, *, xlabel="observed effect (95% CI)", title=None, subtitle=None,
           zero=0.0, arms_for_colour=True):
    """Effect + interval per contrast, with pre-registered thresholds and verdicts.

    Rows sorted as given, because a forest plot re-sorted by magnitude stops being
    comparable between renders. The verdict is a GLYPH plus a word in status colour,
    never a fill -- so the row keeps its arm identity while carrying pass/fail.

    rows : [{"label", "mean", "lo", "hi", "threshold"(opt), "verdict"(opt),
             "arm"(opt), "n"(opt)}]
    """
    rows = [r for r in (rows or []) if np.isfinite(r.get("mean", np.nan))]
    if not rows:
        return no_data("no contrast has a finite effect estimate", title=title)
    fig, ax = _subplots(figsize=(9.0, 1.5 + 0.52 * len(rows)))
    status_c = {"PASS": "#1a7f37", "FAIL": "#b3261e", "WARN": "#9a6700",
                "UNDERPOWERED": "#9a6700"}
    status_g = {"PASS": "▲", "FAIL": "✕", "WARN": "◆",
                "UNDERPOWERED": "◆"}
    ax.axvline(zero, color="#9a9a9a", lw=1.0, zorder=1)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        arm = r.get("arm") or r.get("label", "")
        col = arm_color(arm) if arms_for_colour else "#2a78d6"
        ink = arm_ink(arm) if arms_for_colour else "#1c5cab"
        lo, hi = r.get("lo", np.nan), r.get("hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        spans = np.isfinite(lo) and np.isfinite(hi) and lo <= zero <= hi
        ax.scatter([r["mean"]], [y], s=62, zorder=4, linewidth=1.8,
                   color="white" if spans else col, edgecolor=ink)
        if np.isfinite(r.get("threshold", np.nan)):
            ax.plot([r["threshold"]] * 2, [y - 0.30, y + 0.30], color="0.15",
                    lw=1.8, zorder=5)
        v = str(r.get("verdict") or "").upper()
        if v:
            ax.annotate(f"{status_g.get(v, '')} {v}", (1.005, y),
                        xycoords=("axes fraction", "data"), fontsize=8.0,
                        color=status_c.get(v, "0.3"), va="center", fontweight="semibold")
        if r.get("n"):
            ax.annotate(f"n={r['n']}", (r["mean"], y), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=6.8, color="0.45")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.get("label", "") for r in reversed(rows)], fontsize=8.4)
    for t, r in zip(ax.get_yticklabels(), reversed(rows)):
        t.set_color(arm_ink(r.get("arm") or r.get("label", "")) if arms_for_colour else "0.2")
    ax.set(xlabel=xlabel, ylim=(-0.6, len(rows) - 0.4))
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right", "left"]].set_visible(False)
    head = title or "Effects with intervals"
    sub = subtitle or ("hollow dot = interval spans zero  ·  black bar = pre-registered "
                       "threshold  ·  verdict is a glyph, never a fill colour")
    ax.set_title(f"{head}\n{sub}", fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig


# --- 18. dot plot with intervals ------------------------------------------------

def dot_ci(groups, *, xlabel="group", ylabel="value", title=None, subtitle=None,
           ncol=None, ref=None, ref_label=None, colors=None):
    """Mean + bootstrap CI per category, as a dot. Replaces bar+errorbar.

    A bar starting at an arbitrary zero encodes the distance from zero, which is
    rarely the quantity of interest; the dot encodes the estimate, and the interval
    encodes what we know about it. When there are several measures, each gets a
    facet on its own scale rather than a shared axis that flattens all of them.

    groups : {measure: {category: (mean, lo, hi, n)}}
    """
    groups = {k: v for k, v in (groups or {}).items() if v}
    if not groups:
        return no_data("no measure has a finite estimate", title=title)
    keys = list(groups)
    ncol = ncol or min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = _subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow + 0.7),
                          squeeze=False)
    flat = axes.ravel()
    for ax, k in zip(flat, keys):
        cats = list(groups[k])
        for i, c in enumerate(cats):
            m, lo, hi, n = (list(groups[k][c]) + [None] * 4)[:4]
            col = (colors or {}).get(c) or plt.get_cmap(SEQ)(0.35 + 0.42 * i / max(len(cats) - 1, 1))
            if lo is not None and np.isfinite(lo) and np.isfinite(hi):
                ax.plot([i, i], [lo, hi], color=col, lw=2.2, solid_capstyle="round", zorder=3)
            ax.scatter([i], [m], s=58, color=col, edgecolor="white", lw=1.5, zorder=4)
            if n:
                ax.annotate(f"n={n}", (i, m), xytext=(0, 12), textcoords="offset points",
                            ha="center", fontsize=6.8, color="0.45")
        if ref is not None and np.isfinite(ref):
            ax.axhline(ref, color="0.35", lw=1.1, zorder=1)
            if ref_label:
                ax.annotate(ref_label, (0.99, ref), xycoords=("axes fraction", "data"),
                            xytext=(0, 4), textcoords="offset points", ha="right",
                            fontsize=7.4, color="0.35")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([str(c) for c in cats], fontsize=8)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(-0.55, len(cats) - 0.45))
        ax.set_title(str(k), fontsize=9.4, loc="left")
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat[len(keys):]:
        ax.set_axis_off()
    head = title or "Estimates with bootstrap intervals"
    sub = subtitle or "dot = mean, bar = 95% CI -- no fill, so distance from zero is not over-read"
    fig.suptitle(f"{head}\n{sub}", fontsize=10.5, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90 if nrow > 1 else 0.84])
    return fig


# --- design tokens for non-matplotlib consumers ---------------------------------

def css_tokens():
    """The palette as {mode: {css-var-name: hex}}, for the HTML report.

    Exported from here rather than restated in the report so the two media cannot
    drift: an arm that changes colour in a PNG changes colour in the dashboard in
    the same commit. This is the same reason figures.py imports this module instead
    of carrying its own constants -- that duplication is what let one arm be green
    in one panel and red in the next.
    """
    out = {}
    for mode in ("light", "dark"):
        i = 0 if mode == "light" else 1
        t = {"surface": SURFACE[mode], "control": CONTROL_INK[mode],
             "contact": CONTACT, "ghost": GHOST if mode == "light" else "#3a3a38"}
        for fam, levels in FAMILY_RAMP.items():
            for lvl, pair in enumerate(levels):
                t[f"{fam}{lvl}"] = pair[i]
        t.update({"ink": "#0b0b0b" if mode == "light" else "#ffffff",
                  "ink-2": "#52514e" if mode == "light" else "#c3c2b7",
                  "ink-3": "#8a8a85" if mode == "light" else "#8b8a83",
                  "line": "#e6e6e4" if mode == "light" else "#333331",
                  "hair": "#efefed" if mode == "light" else "#2a2a29",
                  "surface-2": "#f7f7f6" if mode == "light" else "#1f1f1e",
                  "good": "#1a7f37" if mode == "light" else "#3fb950",
                  "warn": "#9a6700" if mode == "light" else "#d29922",
                  "crit": "#b3261e" if mode == "light" else "#f85149"})
        out[mode] = t
    return out


def arm_slot(arm):
    """The CSS variable name carrying this arm's colour ('blue1', 'control', ...)."""
    spec = ARM_SPEC.get(canon_arm(arm))
    if spec is None:
        return "ink-2"
    family, level, _, _ = spec
    return "control" if family is None else f"{family}{level}"
