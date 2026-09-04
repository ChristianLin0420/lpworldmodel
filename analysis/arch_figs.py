"""SVG architecture diagrams for every PiWM variant, and the primitives the other
architecture modules are built from.

Eight modules import from here (arch_figs_causal, arch_figs_predictor, arch_figs_defects,
round5_figs_repr / _planners / _measure / _obj, causal_figs), so the whole 27-figure SVG set
re-colours and re-types from this one file.

STYLE.  Everything comes from analysis/style.py, which is the single design system for the
project (derived from figures/motivation_teaser.svg).  Sans-serif throughout; hues assigned by
IDENTITY, never by rank; pale fills with a matching saturated border; recessive furniture.

The base schematic's identities, fixed for all 27 figures:

    observation / action circles   pale slate    a value the world hands us
    encoder  f                     pale green    the system
    link (ReLU / identity)         pale amber    the piece a variant most often swaps
    predictor  g                   pale purple   the contrasting half of the model
    code stack (active unit)       dark slate    data, never a hue -- the PATTERN is the signal
    MSE                            crimson       the loss
    wires, neutral borders         slate

Variant accents keep their historical keys so the dependent modules import unchanged:
blue -> green, magenta -> purple, green -> teal, amber, crit -> crimson, slate.

Usage:  python analysis/arch_figs.py --out diary/assets/2026-09-02
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.style import (  # noqa: E402,F401
    C, FILL, EDGE, FONT_SVG, GRID, PAPER,     # EDGE/GRID/PAPER are re-exported for the
    INK as _S_INK, MUTED as _S_MUTED,         # dependent modules to import from here
)

# --- palette, re-expressed on the design system -----------------------------------
# old name          ->  new value                       role
# ACCENT["blue"]    ->  C["green"]    #18483C           sparse-code family
# ACCENT["magenta"] ->  C["purple"]   #543C84           gating family
# ACCENT["green"]   ->  C["teal"]     #2E7D6F           union / consensus family
# ACCENT["amber"]   ->  C["amber"]    #C88A1E           the intervention under test
# ACCENT["crit"]    ->  C["crimson"]  #A83024           failure / retraction
# ACCENT["slate"]   ->  C["slate"]    #546060           neutral
ACCENT = {
    "blue": C["green"], "magenta": C["purple"], "green": C["teal"],
    "amber": C["amber"], "crit": C["crimson"], "slate": C["slate"],
}
ACCENT_FILL = {
    "blue": FILL["green"], "magenta": FILL["purple"], "green": FILL["teal"],
    "amber": FILL["amber"], "crit": FILL["crimson"], "slate": FILL["slate"],
}

INK = _S_INK             # #14231C  body text          (was #1A1A1A)
MUTED = _S_MUTED         # #6C786C  secondary text     (new export)
WHITE = PAPER            # #FFFFFF
DIM = "#AAB4AF"          # a disabled / not-scored mark

ARROW = C["slate"]       # #546060  wires              (was #4A6274)
BOXLINE = C["slate"]     # #546060  neutral border     (was #55707F)
MSERED = C["crimson"]    # #A83024  the MSE link       (was #9E3B3B)

# the base schematic's block fills
PEACH = FILL["slate"]    # #DFE4E4  observations/actions  (was #FAD9BE peach)
TEAL = FILL["green"]     # #D8F0E4  encoder               (was #A9D5D1)
YELLOW = FILL["amber"]   # #F7E8C8  link                  (was #F9F2C4)
LAVENDER = FILL["purple"]  # #DED6EA  predictor           (was #E2CDE9)
NAVY = C["slate"]        # #546060  an active code unit   (was #40566E)

FONT = FONT_SVG          # Nimbus Sans / Liberation Sans  (was Georgia, serif)

# stroke weights -- one scale for the whole set, thinner than the old hand-set values
SW_BLOCK = 1.8           # encoder / predictor outline   (was 2.4)
SW_BOX = 1.3             # a plain box                   (was 1.4)
SW_CELL = 1.1            # one code unit                 (was 1.3)
SW_WIRE = 1.8            # a wire                        (was 2.0)
RX = 6                   # box corner radius             (was 4)

# A pale fill implies its own saturated border: every shape drawn with a system fill gets the
# matching hue without the call site having to name it.
_BORDER_FOR = {v: C[k] for k, v in FILL.items()}
_BORDER_FOR.update({
    WHITE: BOXLINE, "#ffffff": BOXLINE, "white": BOXLINE,
    "#eeeeea": C["slate"], "#f2f2ef": C["slate"],
    "#cfe3e1": C["green"], "#EAF6F0": C["green"], "#eaf6f0": C["green"],
})

# Legacy ad-hoc greys in the dependent modules, folded into the system's two greys.
_INK_FOR = {
    "#1a1a1a": INK, "#1A1A1A": INK, "#000": INK, "#000000": INK,
    "#444": MUTED, "#444444": MUTED, "#555": MUTED, "#555555": MUTED,
    "#666": MUTED, "#666666": MUTED, "#6b6b66": MUTED, "#77776f": MUTED,
    "#9a9a94": MUTED, "#bcbcb7": DIM, "#b9b9b4": DIM, "#cfcfca": GRID,
}


def _c(color, table):
    return table.get(color, table.get(str(color).lower(), color))


def _ink(color):
    """Fold a legacy grey onto the system's INK / MUTED / DIM."""
    return _c(color, _INK_FOR)


def _border(fill, given=None):
    """The border a fill implies, unless the call site named one."""
    if given is not None:
        return _ink(given)
    return _c(fill, _BORDER_FOR) if fill in _BORDER_FOR or str(fill).lower() in _BORDER_FOR \
        else BOXLINE


# --- arrowheads -------------------------------------------------------------------
# One marker per colour, so an accent-coloured wire never ends in a slate head (it did:
# aw()/apoly() coloured the LINE but every arrowhead came out ARROW-grey).  Registered
# lazily while the body is built; _hdr() runs last (emit takes an already-built body) and
# emits whatever was used.
_MARKERS = {}


def _marker(color, kind="end"):
    key = (str(color), kind)
    if key not in _MARKERS:
        _MARKERS[key] = f"{'s' if kind == 'start' else 'e'}{len(_MARKERS)}"
    return _MARKERS[key]


def _marker_defs():
    out = []
    for (color, kind), mid in _MARKERS.items():
        if kind == "start":
            out.append(
                f'  <marker id="{mid}" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" '
                f'refX="0.9" refY="5" markerWidth="11" markerHeight="11" orient="auto">'
                f'<path d="M9.4,1.6 L0.4,5 L9.4,8.4 z" fill="{color}"/></marker>\n')
        else:
            out.append(
                f'  <marker id="{mid}" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" '
                f'refX="9.1" refY="5" markerWidth="11" markerHeight="11" '
                f'orient="auto-start-reverse">'
                f'<path d="M0.6,1.6 L9.6,5 L0.6,8.4 z" fill="{color}"/></marker>\n')
    return "".join(out)


def _hdr(w, h):
    # "a" / "ar" / "am" are kept because call sites write marker-end="url(#ar)" by hand.
    for col, kind in ((ARROW, "end"), (MSERED, "end"), (MSERED, "start")):
        _marker(col, kind)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
        '<defs>\n'
        f'  <marker id="a" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="9.1" '
        f'refY="5" markerWidth="11" markerHeight="11" orient="auto-start-reverse">'
        f'<path d="M0.6,1.6 L9.6,5 L0.6,8.4 z" fill="{ARROW}"/></marker>\n'
        f'  <marker id="ar" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="9.1" '
        f'refY="5" markerWidth="11" markerHeight="11" orient="auto-start-reverse">'
        f'<path d="M0.6,1.6 L9.6,5 L0.6,8.4 z" fill="{MSERED}"/></marker>\n'
        f'  <marker id="am" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="0.9" '
        f'refY="5" markerWidth="11" markerHeight="11" orient="auto">'
        f'<path d="M9.4,1.6 L0.4,5 L9.4,8.4 z" fill="{MSERED}"/></marker>\n'
        + _marker_defs() +
        '</defs>\n'
        f'<rect width="{w}" height="{h}" fill="{PAPER}"/>\n'
    )


# --- text metrics -----------------------------------------------------------------
# cairosvg re-anchors EVERY <tspan> when text-anchor="middle", so a mixed-weight or
# mixed-colour run drawn that way piles up on itself (the old title did not, only because
# it was a single run).  The cure is to place such runs with text-anchor="start" at a
# computed x, which needs advance widths.  Nimbus Sans and Liberation Sans -- the two faces
# actually installed -- both carry Helvetica/Arial metrics, so the AFM table below is exact.
_W_REG = {" ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
          "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
          ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015,
          "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333,
          "{": 334, "|": 260, "}": 334, "~": 584}
_W_BOLD = {" ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722, "'": 238,
           "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
           ":": 333, ";": 333, "<": 584, "=": 584, ">": 584, "?": 611, "@": 975,
           "[": 333, "\\": 278, "]": 333, "^": 584, "_": 556, "`": 333,
           "{": 389, "|": 280, "}": 389, "~": 584}
for _c_ in "0123456789":
    _W_REG[_c_] = _W_BOLD[_c_] = 556
_W_REG.update(dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    (667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833,
     722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611))))
_W_REG.update(dict(zip("abcdefghijklmnopqrstuvwxyz",
    (556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833,
     556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500))))
_W_BOLD.update(dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    (722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833,
     722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611))))
_W_BOLD.update(dict(zip("abcdefghijklmnopqrstuvwxyz",
    (556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889,
     611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500))))
for _t_ in (_W_REG, _W_BOLD):
    _t_.update({"\u00a0": 278, "\u2013": 556, "\u2014": 1000, "\u2192": 838,
                "\u21d2": 838, "\u00d7": 584, "\u2264": 549, "\u2265": 549,
                "\u00b7": 278, "\u2212": 584, "\u2026": 1000})

_ENT = {"&#160;": "\u00a0", "&#8211;": "\u2013", "&#8212;": "\u2014", "&#8594;": "\u2192",
        "&#8658;": "\u21d2", "&#215;": "\u00d7", "&#8804;": "\u2264", "&#8805;": "\u2265",
        "&#183;": "\u00b7", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"'}
_TSPAN = re.compile(r"<tspan([^>]*)>|</tspan>")


def text_width(s, size=19, weight="normal"):
    """Advance width, in user units, of an SVG text run -- tspans (their own font-size and
    weight), numeric entities and all.  Exact for the installed Helvetica-metric faces."""
    stack = [(float(size), weight == "bold")]
    w, i = 0.0, 0
    for m in _TSPAN.finditer(s):
        w += _run_w(s[i:m.start()], *stack[-1])
        i = m.end()
        if m.group(0).startswith("</"):
            if len(stack) > 1:
                stack.pop()
        else:
            attrs = m.group(1) or ""
            fs = re.search(r"font-size=[\"']([\d.]+)", attrs)
            fw = re.search(r"font-weight=[\"'](\w+)", attrs)
            stack.append((float(fs.group(1)) if fs else stack[-1][0],
                          (fw.group(1) == "bold") if fw else stack[-1][1]))
    w += _run_w(s[i:], *stack[-1])
    return w


def _run_w(text, size, bold):
    for k, v in _ENT.items():
        text = text.replace(k, v)
    text = re.sub(r"&#\d+;", "\u25a1", text)
    t = _W_BOLD if bold else _W_REG
    return sum(t.get(ch, 556) for ch in text) * size / 1000.0


# --- text -------------------------------------------------------------------------
def txt(x, y, s, size=19, anchor="middle", fill=INK, style="", weight="normal"):
    a = f' font-style="{style}"' if style and style != "normal" else ""
    b = f' font-weight="{weight}"' if weight and weight != "normal" else ""
    if "<tspan" in s and anchor in ("middle", "end"):
        # cairosvg anchors only the first chunk, so a label with a subscript came out
        # off-centre by half its subscript.  Place it ourselves.
        w = text_width(s, size, weight)
        x, anchor = round(x - (w / 2 if anchor == "middle" else w), 1), "start"
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{_ink(fill)}"{a}{b}>{s}</text>\n')


def sub(main, s, size=13):
    """Subscript via dy (baseline-shift is ignored by cairosvg/rsvg), with the baseline
    PUT BACK by an empty tspan.  Without the reset, anything concatenated after a sub()
    was drawn 6px low -- e.g. `sub("z","t") + " (P tokens)"`.  An empty tspan carries no
    advance width, so centred text stays centred."""
    d = round(0.42 * size, 1)
    return (f'{main}<tspan font-size="{size}" dy="{d}">{s}</tspan>'
            f'<tspan dy="{-d}"></tspan>')


def sup(main, s, size=12):
    """Superscript, same construction as sub().  The old version padded with a trailing
    space to restore the baseline, which pushed every centred label half a space left."""
    d = round(0.55 * size, 1)
    return (f'{main}<tspan font-size="{size}" dy="{-d}">{s}</tspan>'
            f'<tspan dy="{d}"></tspan>')


def nrm(inner, size=12):
    """|| inner ||^2.  ASCII bars: U+2016 is absent from Nimbus Sans and Liberation Sans
    alike (verified by rendering), exactly as it was from the old serif face."""
    return "|| " + inner + " ||" + f'<tspan font-size="{size}" dy="{-round(0.55*size,1)}">2' \
                                   f'</tspan><tspan dy="{round(0.55*size,1)}"></tspan>'


def caret(cx, cy, w=9, h=5.5, color=None):
    """A drawn circumflex.  Neither the old serif face nor the new sans stack has the
    precomposed z-hat (U+1E91) or the combining accent (U+0302) -- re-verified by rendering
    against Nimbus Sans / Liberation Sans.  So the hat stays geometry, not text."""
    c = _ink(color or INK)
    return (f'<path d="M{cx-w/2},{cy} L{cx},{cy-h} L{cx+w/2},{cy}" stroke="{c}" '
            f'stroke-width="1.6" fill="none" stroke-linecap="round" '
            f'stroke-linejoin="round"/>\n')


# --- wires ------------------------------------------------------------------------
def arrow(x1, y1, x2, y2, color=ARROW, marker="a", w=SW_WIRE, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = _marker(color) if marker == "a" else marker
    return (f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{w}" '
            f'fill="none" stroke-linecap="round" marker-end="url(#{m})"{d}/>\n')


def poly(pts, color=ARROW, marker="a", w=SW_WIRE, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " L".join(f"{x},{y}" for x, y in pts)
    m = _marker(color) if marker == "a" else marker
    return (f'<path d="M{p}" stroke="{color}" stroke-width="{w}" fill="none" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'marker-end="url(#{m})"{d}/>\n')


# --- shapes -----------------------------------------------------------------------
def circle(cx, cy, r, label, fill=PEACH, size=19, stroke=None):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{_border(fill, stroke)}" stroke-width="1.5"/>\n'
            ) + txt(cx, cy + size * 0.35, label, size)


def domeup(x, y, w, h, label, fill=TEAL, sub="", stroke=None):
    """Encoder shape: flat bottom, rounded top."""
    r = w / 2
    s = (f'<path d="M{x},{y+h} L{x},{y+r} A{r},{r} 0 0 1 {x+w},{y+r} L{x+w},{y+h} Z" '
         f'fill="{fill}" stroke="{_border(fill, stroke)}" stroke-width="{SW_BLOCK}" '
         f'stroke-linejoin="round"/>\n')
    s += txt(x + w / 2, y + h - (30 if sub else 24), label, 21)
    if sub:
        s += txt(x + w / 2, y + h - 10, sub, 13, fill=MUTED)
    return s


def domeright(x, y, w, h, label, fill=LAVENDER, sub="", stroke=None):
    """Predictor shape: flat left, rounded right."""
    r = h / 2
    s = (f'<path d="M{x},{y} L{x+w-r},{y} A{r},{r} 0 0 1 {x+w-r},{y+h} L{x},{y+h} Z" '
         f'fill="{fill}" stroke="{_border(fill, stroke)}" stroke-width="{SW_BLOCK}" '
         f'stroke-linejoin="round"/>\n')
    cx = x + (w - r) / 2 + 8
    s += txt(cx, y + h / 2 + (0 if sub else 7), label, 21)
    if sub:
        s += txt(cx, y + h / 2 + 22, sub, 13, fill=MUTED)
    return s


def box(x, y, w, h, label, fill=YELLOW, size=17, rx=RX, stroke=None, sw=SW_BOX, sub="",
        dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
         f'stroke="{_border(fill, stroke)}" stroke-width="{sw}"{d}/>\n')
    s += txt(x + w / 2, y + h / 2 + (6 if not sub else 1), label, size)
    if sub:
        s += txt(x + w / 2, y + h / 2 + 19, sub, 12.5, fill=MUTED)
    return s


def stack(x, y, pattern, cell=22, gap=3, label=None, lab_dx=0, accent=None, hat=False):
    """Vertical code stack. pattern: iterable of 0/1 (1 = active) / 2 (accent)."""
    s = ""
    for i, v in enumerate(pattern):
        f = accent if (v == 2 and accent) else (NAVY if v else WHITE)
        s += (f'<rect x="{x}" y="{y + i*(cell+gap)}" width="{cell}" height="{cell}" '
              f'fill="{f}" stroke="{NAVY}" stroke-width="{SW_CELL}"/>\n')
    if label:
        lx = x + cell / 2 + lab_dx
        s += txt(lx, y - 13, label, 19)
        if hat:
            # over the leading glyph of the label, one hair above its x-height
            s += caret(lx - 10, y - 26)
    return s


def callout(x, y, w, h, text_lines, key="blue", size=13):
    """The reference's callout: pale fill, matching saturated text, thin matching border.
    Dashed, because in these diagrams a callout always names a PROPOSED change."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{f}" '
         f'stroke="{c}" stroke-width="1.4" stroke-dasharray="6,4"/>\n')
    for i, ln in enumerate(text_lines):
        weight = "bold" if i == 0 else "normal"
        s += txt(x + w / 2, y + 22 + i * (size + 6), ln.replace(" -- ", " &#8211; "),
                 size, fill=c, weight=weight)
    return s


# --- the title line ---------------------------------------------------------------
_STATUS_HUE = {
    "COLLAPSED": "crimson", "DEAD": "crimson", "REFUTED": "crimson",
    "NEGATIVE": "crimson", "FAILED": "crimson", "BROKEN": "crimson",
    "NULL": "slate", "UNTESTED": "slate", "PLANNED": "slate", "OPEN": "slate",
    "WORKS": "green", "POSITIVE": "green", "CONFIRMED": "green",
}
NBSP, ENDASH = "&#160;", "&#8211;"


def title_line(x, y, title, size=20):
    """`(TAG) name -- clause  [STATUS]` as one centred line: the tag in the system green,
    the variant name bold, the clause muted, the status in its identity hue.  One <text>
    element, so the mixed run self-centres and nothing can drift apart."""
    body, status = title, ""
    m = re.search(r"\s*\[([A-Z][A-Z ]*)\]\s*$", title)
    if m:
        status, body = m.group(1).strip(), title[:m.start()]
    tag = ""
    m = re.match(r"\s*(\([^)]*\))\s*(.*)$", body, re.S)
    if m:
        tag, body = m.group(1), m.group(2)
    body = " ".join(body.split())
    name, clause = body, ""
    for seprt in (" -- ", " – "):
        if seprt in body:
            name, clause = body.split(seprt, 1)
            break
    runs = []
    if tag:
        runs.append((tag + NBSP, C["green"], "bold"))
    runs.append((name.strip(), INK, "bold"))
    if clause:
        runs.append((NBSP + ENDASH + NBSP + clause.strip(), MUTED, "normal"))
    if status:
        runs.append((NBSP * 3 + "[" + status + "]", C[_STATUS_HUE.get(status, "slate")],
                     "bold"))
    inner = "".join(f'<tspan fill="{c}" font-weight="{w}">{t}</tspan>' for t, c, w in runs)
    wide = sum(text_width(t, size, w) for t, _, w in runs)
    return (f'<text x="{round(x - wide / 2, 1)}" y="{y}" font-size="{size}" '
            f'text-anchor="start">{inner}</text>\n')


# --- the shared LpWM base ----------------------------------------------------------
SP_L = [0, 1, 1, 0, 1, 0]      # z_t          (sparse: white = 0, filled = active)
SP_P = [0, 1, 0, 1, 1, 0]      # z_hat_{t+1}
SP_R = [0, 1, 1, 1, 0, 0]      # z_{t+1}


def base(title, reg_label="RDMReg", reg_sub="", link_label="ReLU",
         zl=SP_L, zp=SP_P, zr=SP_R, pred_label="Pred", pred_sub="g",
         enc_sub="", skip=()):
    """The LpWM schematic every variant starts from. `skip` omits pieces a variant redraws.

    Every anchor point is fixed -- the dependent modules hook connectors onto x=880,
    (766, 315..362), (300, 190), (362, 430), (850, 648), (852, 742), (812, 782) -- so this
    function may be re-COLOURED but not re-LAID-OUT."""
    s = ""
    # observations
    if "obs" not in skip:
        s += circle(120, 742, 34, sub("o", "t", 13))
        s += circle(812, 742, 34, sub("o", "t+1", 13))
        s += arrow(120, 706, 120, 646)
        s += arrow(812, 706, 812, 646)
    # encoders
    if "enc" not in skip:
        s += domeup(64, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>", sub=enc_sub)
        s += domeup(756, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>", sub=enc_sub)
        s += arrow(120, 520, 120, 470)
        s += arrow(812, 520, 812, 470)
    # links
    if "link" not in skip:
        s += box(78, 420, 84, 44, link_label, size=16)
        s += box(770, 420, 84, 44, link_label, size=16)
        s += arrow(120, 420, 120, 352)
        s += arrow(812, 420, 812, 352)
    # code stacks
    if "codes" not in skip:
        s += stack(109, 190, zl, label=sub("z", "t", 13))
        s += stack(801, 190, zr, label=sub("z", "t+1", 13))
        s += stack(701, 190, zp, label=sub("z", "t+1", 13), hat=True)
        # MSE double arrow, in the 78px gutter between the two right-hand stacks
        s += (f'<path d="M737,262 L795,262" stroke="{MSERED}" stroke-width="1.8" '
              f'stroke-linecap="round" marker-end="url(#ar)" marker-start="url(#am)"/>\n')
        s += txt(762, 240, "MSE (min)", 14, fill=MSERED)
    # predictor + action
    if "pred" not in skip:
        s += domeright(232, 196, 190, 132, pred_label, sub=pred_sub)
        s += circle(318, 430, 38, sub("a", "t", 13))
        s += arrow(318, 392, 318, 332)
        s += arrow(146, 262, 226, 262)
        s += txt(186, 248, sub("z", "t", 12), 15, fill=MUTED)
        s += arrow(424, 262, 697, 262)
    # regulariser
    if "reg" not in skip:
        s += box(430, 556, 160, 52, reg_label, fill=WHITE, sw=1.5, size=17, sub=reg_sub)
        s += poly([(162, 442), (206, 442), (206, 582), (426, 582)])
        s += poly([(770, 442), (726, 442), (726, 582), (594, 582)])
    s += title_line(470, 872, title)
    return s


def emit(name, body, out, w=940, h=900):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --- the variants -----------------------------------------------------------------
def build(out):
    made = []

    # 0. LpWM baseline (reference reproduction)
    made.append(emit("arch-00-lpwm.svg",
        base("(0) LpWM baseline -- sparse JEPA world model"), out))

    # 1. Step 2: k-WTA sparse codes
    b = base("(1) PiWM-sparse -- k-WTA imposed on the code  [REFUTED]",
             link_label="ReLU", skip=("link",))
    b += box(78, 420, 84, 44, "ReLU", size=16) + box(770, 420, 84, 44, "ReLU", size=16)
    # the wire enters and leaves the inserted stage, instead of running underneath it
    b += arrow(120, 420, 120, 402) + arrow(812, 420, 812, 402)
    b += box(60, 358, 120, 40, "k-WTA", fill=ACCENT_FILL["blue"],
             stroke=ACCENT["blue"], sw=2.0, size=16, dash="6,4")
    b += box(752, 358, 120, 40, "k-WTA", fill=ACCENT_FILL["blue"],
             stroke=ACCENT["blue"], sw=2.0, size=16, dash="6,4")
    b += arrow(120, 358, 120, 343) + arrow(812, 358, 812, 343)
    b += callout(268, 626, 366, 100, [
        "Proposed: hard top-k after the link",
        "Realised density = min(k, #positive)/D,",
        "so the arm never ran at its own k:",
        "0.19 of 8 units (D=384), 29 of 41 (D=2048)."], "blue")
    made.append(emit("arch-01-sparse-kwta.svg", b, out))

    # 2. Step 3: support gating
    b = base("(2) PiWM-gate -- gate driven by the code's SUPPORT  [NULL]",
             pred_label="Pred", pred_sub="g (LTV)")
    # the gate box stops at x=286: at the old width its bottom-right corner was 0.5px
    # off the a_t circle (centre 318,430 r=38), so the two shapes touched.
    b += box(176, 356, 110, 40, "1[z &gt; 0]", fill=ACCENT_FILL["magenta"],
             stroke=ACCENT["magenta"], sw=2.0, size=16, dash="6,4")
    # out of the code stack (x=131), down its own corridor (x=145), into the gate's edge.
    # The final leg is 28px, not 7px: at 7px the 11px arrowhead swallowed the whole
    # segment and came to rest ON the elbow, reading as a blob rather than an arrow.
    b += poly([(133, 314), (145, 314), (145, 376), (173, 376)],
              color=ACCENT["magenta"], w=1.8, dash="6,4")
    # and straight up into the predictor's flat bottom -- x=250 keeps it off the action
    # wire at x=318, which the old route crossed
    b += arrow(250, 356, 250, 334, color=ACCENT["magenta"], w=1.8, dash="6,4")
    b += callout(258, 626, 392, 100, [
        "Proposed: gate on the binary support only",
        "s = 1[z&gt;0] is a deterministic function of z,",
        "so I(s;Y) &lt;= I(z;Y): bounded above by magnitude.",
        "Measured: 76% of bits discarded, 63% of info kept."], "magenta")
    made.append(emit("arch-02-gate-support.svg", b, out))

    # 3. Step 4: union head
    b = base("(3) PiWM-union -- J parallel readouts, loss = min_j  [DEAD]",
             skip=("pred",))
    b += domeright(232, 150, 190, 108, "Pred", sub="g, head 1")
    b += (f'<path d="M244,272 L406,272 A54,54 0 0 1 406,380 L244,380 Z" '
          f'fill="{ACCENT_FILL["green"]}" stroke="{ACCENT["green"]}" '
          f'stroke-width="2.0" stroke-dasharray="7,4"/>\n')
    # centred on the shape the way domeright() centres its own label: the flat body runs
    # 244..406 and the arc bulges to 460, so the optical centre is 333, not 314.
    b += txt(333, 332, "heads 2 ... J", 19, fill=ACCENT["green"])
    b += circle(318, 466, 36, sub("a", "t", 13))
    b += arrow(318, 430, 318, 386)
    b += arrow(146, 204, 226, 204)
    b += arrow(424, 204, 697, 204)
    b += arrow(462, 326, 494, 326, color=ACCENT["green"], w=1.8)
    b += box(498, 303, 190, 46, "loss = min_j  L_j",
             fill=ACCENT_FILL["green"], stroke=ACCENT["green"], sw=2.0, size=17,
             dash="6,4")
    # h=100, not 86: four lines at 19px leading put the last baseline 79px down, which
    # left 7px under it and 12px over the first -- visibly bottom-crowded.
    b += callout(268, 694, 366, 100, [
        "Proposed: any head may be right (SDR union)",
        "min_j L_j admits z = 0 as a GLOBAL optimum;",
        "a variance floor prevents collapse and the arm",
        "still scores 0.000 on 6/6 seeds (p = 0.0115)."], "green")
    made.append(emit("arch-03-union-head.svg", b, out))

    # 4. consensus voting -- the positive result
    b = _consensus()
    made.append(emit("arch-04-consensus-vote.svg", b, out, w=980, h=880))

    # 5. reference frame (pose)
    b = base("(4) PiWM-refframe -- allocentric pose bound to the action")
    # the pose column drops BELOW the regulariser wire (y=582), which the old layout
    # ran straight through the middle of the p_t circle
    b += circle(318, 664, 30, sub("p", "t", 13), fill=ACCENT_FILL["amber"])
    b += box(258, 498, 120, 40, "Enc pose", fill=ACCENT_FILL["amber"],
             stroke=ACCENT["amber"], sw=2.0, size=15, dash="6,4")
    # a_t sits above the regulariser wire and p_t below it, and both x-positions are fixed
    # by base(), so the pose column HAS to cross y=582.  It crosses with a hop -- a 9px
    # semicircle centred on the crossing -- instead of a plain intersection, which read as
    # a T-junction into the regulariser.
    b += (f'<path d="M318,634 L318,591 A9,9 0 0 0 318,573 L318,542" '
          f'stroke="{ACCENT["amber"]}" stroke-width="1.8" fill="none" '
          f'stroke-linecap="round" marker-end="url(#{_marker(ACCENT["amber"])})"/>\n')
    b += arrow(318, 498, 318, 472, color=ACCENT["amber"], w=2.0)
    # the "+" used to sit at (352,480), overlapping the a_t circle's lower-right stroke
    b += txt(344, 486, "+", 20, fill=ACCENT["amber"], weight="bold")
    # x=372 (was 386) keeps the right border off the right encoder dome, which starts at 756
    b += callout(372, 636, 364, 100, [
        "Proposed: bind location to the action embedding",
        "obs['proprio'] was loaded every batch and DROPPED",
        "(adaln path), so the pose encoder got zero gradient.",
        "Rollout must roll pose forward, not freeze it."], "amber")
    made.append(emit("arch-05-refframe-pose.svg", b, out))

    # 6. columns (patch tokens)
    b = base("(5) PiWM-columns -- P=256 patch tokens instead of one CLS",
             skip=("codes", "pred"), enc_sub="feature = patch")
    for k, xoff in enumerate((0, 26, 52)):
        b += stack(95 + xoff, 190 + k * 6, SP_L, cell=18, gap=3)
    # centred on the three sub-stacks (95..165), not on 148
    b += txt(130, 168, sub("z", "t", 13) + " (P tokens)", 17)
    for k, xoff in enumerate((0, 26, 52)):
        b += stack(787 + xoff, 190 + k * 6, SP_R, cell=18, gap=3)
    b += txt(822, 168, sub("z", "t+1", 13), 17)
    # the predicted stack sits 32 further left than the single-token version.  Three
    # sub-stacks are 70px wide where one is 22, so at the old offset the gutter was 77px
    # and the 68px "MSE (min)" label came within 2px of the target group; 89px gives the
    # label the same clearance the single-token base gives it.
    for k, xoff in enumerate((0, 26, 52)):
        b += stack(628 + xoff, 190 + k * 6, SP_P, cell=18, gap=3)
    b += txt(668, 168, sub("z", "t+1", 13), 17) + caret(658, 155)
    b += (f'<path d="M715,258 L775,258" stroke="{MSERED}" stroke-width="1.8" '
          f'stroke-linecap="round" marker-end="url(#ar)" marker-start="url(#am)"/>\n')
    b += txt(745, 236, "MSE (min)", 14, fill=MSERED)
    # the predictor block, redrawn so its wires start and stop clear of the token stacks
    b += domeright(232, 196, 190, 132, "Pred", sub="g")
    b += circle(318, 430, 38, sub("a", "t", 13))
    b += arrow(318, 392, 318, 332)
    b += arrow(176, 262, 226, 262)
    b += txt(201, 248, sub("z", "t", 12), 15, fill=MUTED)
    b += arrow(424, 262, 620, 262)
    b += callout(268, 694, 366, 100, [
        "Proposed: give the model columns at all",
        "feature=cls gives num_patches = 1, so the predictor",
        "saw ONE token and shared one operator across it.",
        "TBT needs a population of located columns."], "slate")
    made.append(emit("arch-06-columns-patch.svg", b, out))

    # 7. SIGReg / isotropic Gaussian target
    b = base("(6) PiWM-sigreg -- LeJEPA objective: isotropic-Gaussian target",
             reg_label="SIGReg", reg_sub="Epps-Pulley, 1024 proj.",
             link_label="Identity",
             zl=[1, 1, 1, 1, 1, 1], zp=[1, 1, 1, 1, 1, 1], zr=[1, 1, 1, 1, 1, 1],
             skip=("reg",))
    b += box(418, 552, 184, 60, "SIGReg", fill=ACCENT_FILL["crit"],
             stroke=ACCENT["crit"], sw=2.0, size=18, dash="6,4",
             sub="Epps-Pulley &#8594; N(0, I)")
    b += poly([(162, 442), (206, 442), (206, 582), (414, 582)])
    b += poly([(770, 442), (726, 442), (726, 582), (606, 582)])
    b += callout(258, 644, 392, 100, [
        "Proposed: replace the sparse target with N(0, I)",
        "LeJEPA claims the isotropic Gaussian minimises",
        "worst-case probing risk. Here the SAME target under",
        "RDMReg also dies -- the target, not the regulariser."], "crit")
    made.append(emit("arch-07-sigreg-gaussian.svg", b, out))

    # 8. token dropping
    b = base("(7) PiWM-drop95 -- 95% of patch tokens dropped  [COLLAPSED]",
             enc_sub="feature = patch", skip=("enc",))
    # the dome keeps its bottom at 644 (so o_t and the sub-label are where every other
    # figure puts them) but starts 20px lower, because "one continuous wire with the
    # inserted stage straddling it" meant the opaque box covered y=480..510 of the only
    # wire between the encoder and the link: the encoder read as unconnected.
    for x in (64, 756):
        b += domeup(x, 540, 112, 104, "Enc <tspan font-style='italic'>f</tspan>",
                    sub="feature = patch")
    for x in (120, 812):
        # the wire enters the inserted stage and leaves it, and both legs are visible
        b += arrow(x, 540, x, 524)
        b += box(x - 60, 488, 120, 30, "drop 95%", fill=ACCENT_FILL["amber"],
                 stroke=ACCENT["amber"], sw=2.0, size=16, dash="6,4")
        b += arrow(x, 488, x, 470)
    b += callout(258, 644, 392, 100, [
        "Proposed: LeVJEPA's strongest single gain",
        "In a JEPA the encoder output is ALSO the target, so",
        "dropping removes signal from both sides. Paired with",
        "RDMReg (dead code costs only 0.51) it collapsed: rho=0."], "amber")
    made.append(emit("arch-08-token-drop.svg", b, out))

    # 9. block-causal attention
    b = base("(8) PiWM-blockcausal -- temporal structure inside the encoder",
             skip=("enc",))
    b += domeup(64, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                fill=ACCENT_FILL["green"])
    b += domeup(756, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                fill=ACCENT_FILL["green"])
    b += arrow(120, 520, 120, 470)
    b += arrow(812, 520, 812, 470)
    # under both encoders, in the empty band between them and the observations --
    # the old straight line at y=582 ran through the RDMReg box and its label
    b += poly([(150, 650), (150, 676), (782, 676), (782, 652)],
              color=ACCENT["green"], w=2.0, dash="8,5")
    b += txt(466, 662, "block-causal attention across frames", 15, fill=ACCENT["green"])
    b += callout(258, 696, 392, 100, [
        "Proposed: the encoder sees the clip, causally",
        "encode_obs folded t into the batch, so frames were",
        "encoded INDEPENDENTLY -- no temporal structure at all.",
        "Verified causal: frame 0 is unchanged by frame T."], "green")
    made.append(emit("arch-09-block-causal.svg", b, out))

    return made


def _consensus():
    """M independently trained columns voting at PLAN time (the positive result)."""
    s = ""
    ys = [96, 300, 504]
    BX, BY, BW, BH = 566, 246, 196, 120        # consensus box
    cy = BY + BH / 2
    for i, y in enumerate(ys):
        fill = TEAL if i < 2 else "#EAF6F0"
        s += domeup(70, y, 96, 104,
                    sub("Enc <tspan font-style='italic'>f</tspan>", f"{i+1}", 12), fill=fill)
        s += domeright(232, y + 12, 150, 80, "Pred", sub=sub("g", f"{i+1}", 11))
        s += arrow(166, y + 52, 226, y + 52)
        # the stack sits at 418, not 404: the predictor's arc tips at x=382, so at 404 the
        # connecting arrow was 14px long carrying an 11px head -- a floating arrowhead with
        # a 3px shaft. 28px gives the head a shaft to sit on.
        s += stack(418, y + 6, [SP_P, SP_L, SP_R][i], cell=16, gap=3)
        s += arrow(384, y + 52, 412, y + 52)
        s += txt(118, y - 12, f"column {i+1}", 15, fill=MUTED)
        # each member gets its OWN corridor and its OWN entry point on the vote box:
        # the old routing put all three on the shared segment x=508 -> the box.
        cor, ey = (500, 512, 524)[i], (BY + 30, cy, BY + BH - 30)[i]
        s += poly([(440, y + 52), (cor, y + 52), (cor, ey), (BX - 6, ey)])
    # drawn ellipsis (no font here has a reliable U+22EE), with its caption beside it
    for k in range(3):
        s += f'<circle cx="118" cy="{644 + k*11}" r="2.4" fill="{MUTED}"/>\n'
    # the caption centres on the dot column (dots at 644/655/666, so mid 655), instead of
    # sitting level with the LAST dot as if it belonged to that dot alone.
    s += txt(138, 661, "M independently trained models", 15, anchor="start", fill=MUTED)
    s += (f'<rect x="{BX}" y="{BY}" width="{BW}" height="{BH}" rx="9" '
          f'fill="{ACCENT_FILL["green"]}" stroke="{ACCENT["green"]}" stroke-width="2.0"/>\n')
    s += txt(BX + BW / 2, BY + 34, "CONSENSUS", 19, fill=ACCENT["green"], weight="bold")
    s += txt(BX + BW / 2, BY + 60, "median over", 15, fill=ACCENT["green"])
    s += txt(BX + BW / 2, BY + 81, "per-member ranks", 15, fill=ACCENT["green"])
    s += txt(BX + BW / 2, BY + 104, "(at PLAN time)", 13, fill=MUTED)
    s += arrow(BX + BW, cy, BX + BW + 50, cy, color=ACCENT["green"], w=2.0)
    s += circle(BX + BW + 96, cy, 42, "CEM", fill=PEACH, size=17)
    # h=100 like every other four-line callout in the set; raised to 694 so the extra
    # 14px does not close on the title's cap-height at y=824.
    s += callout(238, 694, 494, 100, [
        "Proposed: vote among columns -- and it WORKS",
        "M=5: delta = +0.228 vs a single model (p = 0.0005), beats its",
        "members' mean 10/10, matches best-of-M, 0/10 catastrophes.",
        "On the OBJECTIVE, not in the loss: min_j L_j is degenerate."], "green")
    s += title_line(490, 824,
                    "(9) PiWM-vote -- plan-time consensus over independently "
                    "trained columns")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-02")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in build(a.out):
        print(f"  wrote {a.out}/{n}")


if __name__ == "__main__":
    main()
