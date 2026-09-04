"""Architecture diagrams for the causal-objective round (V1-V3), in the LpWM paper's style.

The top half of each figure is the unmodified LpWM schematic from analysis/arch_figs.py
(slate observation/action circles, pale-green encoder, pale-amber link, pale-purple
predictor, slate/white code stacks, crimson MSE). The bottom half is a MODULE PANEL: the
proposed addition drawn the same way -- blocks, operator nodes, and arrows -- with the
equation it implements underneath and the measured outcome as a value strip. No prose.

This module also carries the primitives five other architecture modules import
(panel/mbox/pill/opnode/aw/apoly/eq/strip/nrm/sup/emit). Every name and positional
signature here is fixed; the only changes are additive keywords.

LAYOUT RULES, and why each one exists (each is a defect this file used to have)

  * Strict left-to-right dataflow on fixed row baselines; a wire that changes rows does so
    on its OWN vertical corridor. V3 used to route p_t down and a_t up through the same
    x=136 corridor, so the two wires were literally the same line.
  * base()'s wires form a CLOSED LOOP, so a connector from the module panel into the
    model's interior must cross one of them. It crosses ONCE, at a named point, with a
    hop -- see wire_to_action / wire_to_pred / wire_to_loss. An unhopped crossing reads
    as a junction, which is what V1 and P4 used to draw.
  * Nothing is sized by hand where it can be measured: mbox, pill and strip fit their own
    text (strip wraps rather than shrink past 15), and eqrow guarantees a gap between
    clauses. P5's strip used to spill its cells into their neighbours.
  * Text goes in free space and is owned by the mark next to it. P4's "L_ctrb" used to sit
    on top of the z_{t+1} code stack.

Usage:  python analysis/arch_figs_causal.py --out diary/assets/2026-09-03
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, ARROW, BOXLINE, DIM, INK, MSERED, MUTED, NAVY, WHITE,
    _W_BOLD, _W_REG, _hdr, arrow, base, box, caret, circle, poly, sub, text_width, txt,
)

W, H = 940, 1310
PX, PY, PW, PH = 34, 848, 872, 434          # module panel
PIN, PIX = PX + 18, PX + PW - 18            # the panel's inner text margins
ROW_A, ROW_B = 940, 1050                    # the two module-flow baselines
EQ1, EQ2 = 1140, 1170                       # the two equation baselines
SY = 1198                                   # the value strip's top edge

DY = -60                                    # the schematic group's vertical nudge: base()
                                            # draws from y=158, which left a 164px band of
                                            # white above every figure in the set.

# --- glyph metrics ----------------------------------------------------------------
# arch_figs.text_width carries the Helvetica AFM table, which has no Greek and no maths.
# Measured from NimbusSans-Regular/-Bold.otf (fontTools); (regular, bold) advance x1000.
# setdefault only -- this ADDS the glyphs the base table lacks and overrides nothing, so
# the five modules that import from here get more accurate centring and nothing else.
_GLYPHS = {
    "θ": (556, 541), "φ": (648, 715), "λ": (500, 556),   # theta phi lambda
    "τ": (395, 446), "ρ": (569, 619), "ε": (446, 475),   # tau rho epsilon
    "β": (575, 610), "Σ": (618, 600), "Δ": (668, 719),   # beta Sigma Delta
    "γ": (500, 556), "α": (578, 615), "μ": (576, 612),   # gamma alpha mu
    "σ": (617, 684), "π": (690, 766),                         # sigma pi
    "∥": (719, 719), "∈": (1000, 1000), "∑": (996, 996), # || in sum
    "√": (453, 549), "⊗": (1000, 1000), "⊙": (996, 996), # sqrt (x) (.)
    "²": (333, 333), "¹": (333, 333), "ⁿ": (365, 396),   # sup 2 1 n
    "⁰": (333, 333), "³": (333, 333), "⁴": (333, 333),
    "⁵": (333, 333), "⁶": (333, 333), "⁷": (333, 333),
    "⁸": (333, 333), "⁹": (333, 333),
    "°": (400, 400), "†": (556, 556), "′": (191, 238),   # deg dagger prime
}
for _ch, (_wr, _wb) in _GLYPHS.items():
    _W_REG.setdefault(_ch, _wr)
    _W_BOLD.setdefault(_ch, _wb)

BAR = "∥"      # U+2225 PARALLEL TO -- the real double bar. U+2016 is absent from
                    # Nimbus Sans and Liberation Sans alike, which is why the norm used
                    # to be typed as two ASCII pipes.
SQ = "²"       # a real superscript-two glyph; no raised tspan to collide
DOT = "·"
TIMES = "×"
MINUS = "−"
ARROWC = "→"
IMPLIES = "⇒"
NDASH = "–"
MDASH = "—"
THETA, PHI, LAMB, TAU, RHO, EPS, BETA, SIGMA = (
    "θ", "φ", "λ", "τ", "ρ", "ε", "β", "Σ")


# --- text primitives --------------------------------------------------------------
def sup(main, s, size=12):
    """Superscript, with the baseline PUT BACK by an empty tspan.

    arch_figs.sup raises by 0.55x the SUPERSCRIPT size, which clears a capital but not a
    short base glyph: `v` followed by a raised `T` came out as a v with a bar through it.
    0.95x puts the superscript's baseline on the base glyph's cap line."""
    d = round(0.95 * size, 1)
    return (f'{main}<tspan font-size="{size}" dy="-{d}">{s}</tspan>'
            f'<tspan dy="{d}"></tspan>')


def nrm(inner, size=12):
    """|| inner ||^2, set with the real double bar (U+2225) and a real superscript two. A bare "." means "whatever
    came in on the wire" and is set as a centred dot."""
    body = DOT if inner.strip() == "." else inner
    return f"{BAR}{body}{BAR}²"


def fit_size(s, size, maxw, floor=10.0, weight="normal"):
    """The largest size <= `size` at which `s` fits `maxw`. Nothing in this file is sized
    by hand where it can be measured."""
    while size > floor and text_width(s, size, weight) > maxw:
        size -= 0.25
    return round(size, 2)


ASCENDERS = "bdfhklti"          # glyphs a circumflex has to clear, not sit inside


def _hat_dy(ch, size):
    return round((0.90 if ch in ASCENDERS else 0.66) * size, 1)


def hatrun(x, y, main, sub_s="", size=17, fill=INK, weight="normal"):
    """`main` with a DRAWN circumflex over its first glyph, plus an optional subscript,
    start-anchored at x. Neither Nimbus Sans nor Liberation Sans has a precomposed z-hat
    (U+1E91) or a combining accent (U+0302), so the hat has to be geometry. Returns
    (svg, advance) so a caller can chain runs on one baseline.

    This is what replaced the literal strings "zhat", "phat_t+1" and "bhat_t+1"."""
    body = main if not sub_s else sub(main, sub_s, round(size * 0.72))
    s = txt(x, y, body, size, anchor="start", fill=fill, weight=weight)
    w0 = text_width(main[0], size, weight)
    s += caret(round(x + w0 / 2, 1), y - _hat_dy(main[0], size),
               w=max(7.0, 0.5 * size), h=max(4.0, 0.32 * size), color=fill)
    return s, text_width(body, size, weight)


_TAG = re.compile(r"<[^>]*>")


def nb(s):
    """Spaces outside tags -> U+00A0.

    SVG collapses runs of whitespace and strips them at the edges of a <text> element, so
    an equation laid out as several elements lost the space on either side of every join:
    `L_z  =  w` came out `L_z= w`, and the measured advance no longer matched the drawn
    glyphs. Non-breaking spaces are the same width and are never collapsed."""
    out, i = [], 0
    for m in _TAG.finditer(s):
        out.append(s[i:m.start()].replace(" ", "\u00a0"))
        out.append(m.group(0))
        i = m.end()
    out.append(s[i:].replace(" ", "\u00a0"))
    return "".join(out)


def seq_w(parts, size=17, weight="normal"):
    """Advance width of an eqseq() part list."""
    w = 0.0
    for p in parts:
        if isinstance(p, tuple):
            main, sub_s = p[1], (p[2] if len(p) > 2 else "")
            w += text_width(main if not sub_s else sub(main, sub_s, round(size * 0.72)),
                            size, weight)
        else:
            w += text_width(p, size, weight)
    return w


def eqseq(x, y, parts, size=17, fill=INK, weight="normal"):
    """Lay a run out left to right at a known baseline. A part is a string, or the tuple
    ("hat", main, sub) for a hatted variable."""
    s, cx = "", x
    for p in parts:
        if isinstance(p, tuple):
            frag, adv = hatrun(cx, y, p[1], p[2] if len(p) > 2 else "", size, fill, weight)
            s += frag
            cx += adv
        else:
            s += txt(cx, y, nb(p), size, anchor="start", fill=fill, weight=weight)
            cx += text_width(p, size, weight)
    return s


def eqmid(cx, cy, parts, size=16, fill=INK, weight="normal"):
    """eqseq(), centred on cx."""
    return eqseq(round(cx - seq_w(parts, size, weight) / 2, 1), cy, parts, size, fill,
                 weight)


def eq(x, y, s, size=17, weight="normal"):
    return txt(x, y, s, size, anchor="start", fill=INK, weight=weight)


def eqfit(x, y, s, size=17, weight="normal", maxx=None):
    """A single equation line that CANNOT run past the panel. P3's bold conclusion used to
    overflow the panel border by a few px; this is why."""
    maxx = PIX - 8 if maxx is None else maxx
    return eq(x, y, s, fit_size(s, size, maxx - x, floor=12.5, weight=weight), weight)


def eqrow(y, cols, size=17, x0=None, x1=None, gap=44, weight="normal"):
    """Several clauses on ONE baseline with a guaranteed gap between them. Shrinks the gap
    first and the type second, rather than letting two clauses touch -- which is what P5's
    `E_k = ...` and `a_0 = a_t` used to do, with 4px between them.

    A column is a string or an eqseq() part list."""
    x0 = PX + 26 if x0 is None else x0
    x1 = PIX - 8 if x1 is None else x1
    cols = [c if isinstance(c, list) else [c] for c in cols]
    n, avail = len(cols), (x1 - x0)
    sz = float(size)
    while True:
        ws = [seq_w(c, sz, weight) for c in cols]
        if sum(ws) + gap * (n - 1) <= avail or n == 1:
            break
        if gap > 26:
            gap -= 2
        elif sz > 13.0:
            sz -= 0.25
        else:
            break
    g = gap if n < 2 else max(gap, min(2.4 * gap, (avail - sum(ws)) / (n - 1)))
    s, cx = "", x0
    for c, w in zip(cols, ws):
        s += eqseq(round(cx, 1), y, c, round(sz, 2), INK, weight)
        cx += w + g
    return s


# --- shapes -----------------------------------------------------------------------
def panel(key, title):
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.2"/>\n')
    s += txt(PX + 20, PY + 32, title, fit_size(title, 18, PW - 44, floor=14),
             anchor="start", fill=c, weight="bold")
    s += (f'<path d="M{PIN},{PY + 44} L{PIX},{PY + 44}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def mbox(x, y, w, h, label, key, size=16, fill=WHITE, sub_=""):
    """A block in the panel's accent colour. Both lines FIT: P4's `- logdet(...)` used to
    push its caption 8px past each border of its own box."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
         f'stroke="{c}" stroke-width="1.8"/>\n')
    s += txt(x + w / 2, y + h / 2 + (6 if not sub_ else 0), label,
             fit_size(label, size, w - 16, floor=11.5))
    if sub_:
        s += txt(x + w / 2, y + h / 2 + 17, sub_,
                 fit_size(sub_, 11.5, w - 14, floor=9.5), fill=MUTED)
    return s


def pill(cx, cy, label, key, rx=34, ry=19, fill=None, hat=False):
    """A rounded value/variable node. `rx` is a minimum: the type shrinks to 12.5 before
    the pill grows, so a long label can never widen a pill into its neighbour."""
    c = ACCENT[key]
    f = fill or ACCENT_FILL[key]
    full = label if not isinstance(hat, str) else sub(label, hat, 12)
    size = fit_size(full, 16, 2 * rx - 20, floor=12.5)
    rx = max(rx, text_width(full, size) / 2 + 12)
    s = (f'<rect x="{round(cx - rx, 1)}" y="{cy - ry}" width="{round(2 * rx, 1)}" '
         f'height="{2 * ry}" rx="{ry}" fill="{f}" stroke="{c}" stroke-width="1.8"/>\n')
    if hat:
        w = text_width(full, size)
        s += txt(round(cx - w / 2, 1), cy + 6, full, size, anchor="start")
        s += caret(round(cx - w / 2 + text_width(label[0], size) / 2, 1),
                   cy + 6 - _hat_dy(label[0], size),
                   w=max(7.0, 0.5 * size), h=max(4.0, 0.32 * size))
        return s
    return s + txt(cx, cy + 6, full, size)


def opnode(cx, cy, sym, key, r=16):
    """Operator node: circle + glyph, drawn (not typeset) so it never misses a font."""
    c = ACCENT[key]
    s = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{c}" '
         f'stroke-width="1.8"/>\n')
    if sym == "minus":
        s += (f'<path d="M{cx - 8},{cy} L{cx + 8},{cy}" stroke="{c}" stroke-width="2.2" '
              f'stroke-linecap="round"/>\n')
    elif sym == "times":
        s += (f'<path d="M{cx - 6},{cy - 6} L{cx + 6},{cy + 6} M{cx + 6},{cy - 6} '
              f'L{cx - 6},{cy + 6}" stroke="{c}" stroke-width="2.2" '
              f'stroke-linecap="round"/>\n')
    elif sym == "plus":
        s += (f'<path d="M{cx - 8},{cy} L{cx + 8},{cy} M{cx},{cy - 8} L{cx},{cy + 8}" '
              f'stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>\n')
    return s


def dot(cx, cy, key, r=4):
    """A fan-out junction. Drawn wherever one signal legitimately feeds two blocks, so a
    junction is never confused with a crossing (and a crossing never with a junction)."""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{ACCENT[key]}"/>\n'


def aw(x1, y1, x2, y2, key, w=1.8, dash=""):
    return arrow(x1, y1, x2, y2, color=ACCENT[key], w=w, dash=dash)


def apoly(pts, key, w=1.8, dash=""):
    return poly(pts, color=ACCENT[key], w=w, dash=dash)


def _hop_d(pts, hops, r=8):
    """The `d` of a polyline that HOPS over each listed crossing point.

    Without this a connector and a model wire that merely cross read as joined -- V1 and
    P4 both drew an unhopped T over the right-hand link wire at x=812."""
    hops = [(float(a), float(b)) for a, b in hops]
    d = f"M{pts[0][0]},{pts[0][1]}"
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg = []
        for hx, hy in hops:
            if abs(dx) > abs(dy):                       # horizontal run
                if abs(hy - y1) < 3 and min(x1, x2) + r < hx < max(x1, x2) - r:
                    seg.append((abs(hx - x1), hx, y1, 1 if dx > 0 else 0))
            else:                                        # vertical run
                if abs(hx - x1) < 3 and min(y1, y2) + r < hy < max(y1, y2) - r:
                    seg.append((abs(hy - y1), x1, hy, 1 if dy > 0 else 0))
        for _, hx, hy, sweep in sorted(seg):
            if abs(dx) > abs(dy):
                sx = hx - r if dx > 0 else hx + r
                ex = hx + r if dx > 0 else hx - r
                d += f" L{round(sx,1)},{hy} A{r},{r} 0 0 {sweep} {round(ex,1)},{hy}"
            else:
                sy = hy - r if dy > 0 else hy + r
                ey = hy + r if dy > 0 else hy - r
                d += f" L{hx},{round(sy,1)} A{r},{r} 0 0 {sweep} {hx},{round(ey,1)}"
        d += f" L{x2},{y2}"
    return d


def hpoly(pts, key, w=1.8, dash="", hops=(), r=8):
    """apoly() with a hop at each named crossing."""
    from analysis.arch_figs import _marker
    c = ACCENT[key]
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{_hop_d(pts, hops, r)}" stroke="{c}" stroke-width="{w}" '
            f'fill="none" stroke-linejoin="round" stroke-linecap="round" '
            f'marker-end="url(#{_marker(c)})"{ds}/>\n')


# --- the measured-outcome strip ---------------------------------------------------
def _wrap2(s):
    """Split a value onto two lines at the most balanced comma, else space."""
    for seps in (", ", " "):
        parts = s.split(seps)
        if len(parts) < 2:
            continue
        best, bw = None, None
        for i in range(1, len(parts)):
            a = seps.join(parts[:i]) + ("," if seps == ", " else "")
            b = seps.join(parts[i:])
            m = max(text_width(a, 15, "bold"), text_width(b, 15, "bold"))
            if bw is None or m < bw:
                best, bw = (a, b), m
        if best:
            return list(best)
    return [s]


def strip(x, y, items, key, cw=196, h=44):
    """Measured-outcome value strip: label above, number below, one cell per item.

    Each cell is as wide as its OWN content and never narrower than `cw`; if the row still
    will not fit the panel the value type steps 16 -> 15 and only then wraps onto a second
    line. It never clips and never spills, which is what P3's `||W|| action 20.3 vs encoder
    354.7` and P5's `encode action identity, drop the state` both used to do."""
    c = ACCENT[key]
    n, gap = len(items), 8
    avail = (PIX - 8) - x
    for vs, ls, lines in ((16, 12, 1), (15, 11.5, 1), (14.5, 11, 1), (16, 12, 2),
                          (15, 11.5, 2), (14, 11, 2)):
        vals = [(_wrap2(v) if lines == 2 else [v]) for _, v, _ in items]
        ws = [max(cw, text_width(items[i][0], ls) + 20,
                  max(text_width(p, vs, "bold") for p in vals[i]) + 20)
              for i in range(n)]
        if sum(ws) + gap * (n - 1) <= avail:
            break
    wrapped = any(len(v) > 1 for v in vals)
    h = max(h, 58) if wrapped else h
    s, cx = "", x
    for i, (lab, _, hot) in enumerate(items):
        s += (f'<rect x="{round(cx,1)}" y="{y}" width="{round(ws[i],1)}" height="{h}" '
              f'rx="6" fill="{WHITE}" stroke="{c}" stroke-width="1.3" '
              f'stroke-opacity="0.75"/>\n')
        two = len(vals[i]) > 1
        s += txt(cx + ws[i] / 2, y + (15 if two else 17), lab, ls, fill=MUTED)
        col = ACCENT["crit"] if hot else INK
        for j, ln in enumerate(vals[i]):
            s += txt(cx + ws[i] / 2, y + (33 + j * 18 if two else 36), ln, vs,
                     fill=col, weight="bold")
        cx += ws[i] + gap
    return s


# --- the schematic group and its three connector routes ---------------------------
# base() fixes every anchor and its wires form a closed loop, so a connector from the
# module panel to the model's interior must cross exactly one of them. There are three
# places a variant can attach, and each has ONE route, used by every figure that attaches
# there -- so the reader learns the route once.
def sch(body):
    """The schematic, nudged up by DY. Connectors are drawn INSIDE this group, so their
    panel-side endpoint is PY - DY in group coordinates."""
    return f'<g transform="translate(0,{DY})">\n{body}</g>\n'


PYL = PY - DY                      # the panel's top edge, in schematic-group coordinates


def wire_to_action(key, label):
    """Into the action circle. Crosses the predictor's output wire once, at (620, 262)."""
    s = hpoly([(880, PYL), (880, 120), (620, 120), (620, 430), (358, 430)], key,
              dash="6,4", hops=[(620, 262)])
    return s + txt(870, 106, label, 15.5, anchor="end", fill=ACCENT[key])


def wire_to_pred(key, label):
    """Into the predictor's flat top. Runs in two empty corridors; crosses nothing."""
    s = apoly([(880, PYL), (880, 120), (300, 120), (300, 190)], key, dash="6,4")
    return s + txt(310, 106, label, 15.5, anchor="start", fill=ACCENT[key])


def wire_to_loss(key, sym, label):
    """Into the MSE, via an operator node in the empty gutter between the two right-hand
    code stacks. Crosses the right-hand link wire once, at (812, 380)."""
    s = hpoly([(880, PYL), (880, 380), (766, 380), (766, 347)], key, dash="6,4",
              hops=[(812, 380)])
    s += opnode(766, 330, sym, key, r=15)
    s += apoly([(766, 315), (766, 276)], key, dash="6,4")
    return s + txt(688, 330, label, 15.5, anchor="end", fill=ACCENT[key])


# --- V1 ---------------------------------------------------------------------------
def v1_incr():
    K = "blue"
    b = base("(V1) PiWM-incr -- per-sample increment normalisation   [COLLAPSED]")
    b += wire_to_loss(K, "times", "w scales the prediction loss")
    b = sch(b)

    s = panel(K, "module:  a per-sample weight on the prediction loss")
    yA, yB = ROW_A, ROW_B
    # row A -- the weight, built from the TRUE increment
    s += pill(98, yA - 25, sub("z", "t+1", 12), K)
    s += pill(98, yA + 25, sub("z", "t", 12), K)
    s += opnode(198, yA, "minus", K)
    s += aw(132, yA - 25, 180, yA - 8, K)
    s += aw(132, yA + 25, 180, yA + 8, K)
    s += mbox(234, yA - 24, 104, 48, nrm("."), K)
    s += aw(214, yA, 230, yA, K)
    s += mbox(374, yA - 24, 176, 48, f"1 / ( {DOT} + {EPS} )", K)
    s += aw(338, yA, 370, yA, K)
    s += pill(620, yA, "w", K, rx=30)
    s += aw(550, yA, 586, yA, K)
    # row B -- the prediction error it multiplies
    s += pill(98, yB - 25, "z", K, fill=WHITE, hat="t+1")
    s += pill(98, yB + 25, sub("z", "t+1", 12), K, fill=WHITE)
    s += opnode(198, yB, "minus", K)
    s += aw(132, yB - 25, 180, yB - 8, K)
    s += aw(132, yB + 25, 180, yB + 8, K)
    s += mbox(234, yB - 24, 104, 48, nrm("."), K)
    s += aw(214, yB, 230, yB, K)
    s += opnode(620, yB, "times", K)
    s += aw(338, yB, 600, yB, K)
    s += aw(620, yA + 19, 620, yB - 18, K)
    s += mbox(692, yB - 24, 190, 48, sub("L", "z", 12), K, fill=ACCENT_FILL[K], size=18)
    s += aw(636, yB, 688, yB, K)
    # equations + measured
    s += eqrow(EQ1, [
        ["L", sub("", "z"), f"  =  w {DOT} {BAR}", ("hat", "z", "t+1"),
         f" {MINUS} ", "z", sub("", "t+1"), f"{BAR}²"],
        [f"w  =  1 / ( {BAR}", "z", sub("", "t+1"), f" {MINUS} ", "z", sub("", "t"),
         f"{BAR}²  +  {EPS} )"]], 17)
    s += eqfit(PX + 26, EQ2, "w is formed PER SAMPLE. A batch-level w would only rescale "
               "L_z and leave every gradient direction unchanged.", 15)
    s += strip(PX + 26, SY, [
        (f"{EPS}", "1e-4", False),
        (f"median {BAR}dz{BAR}²", "4.1e-2", True),
        (f"samples floored by {EPS}", "0.1%", True),
        ("w_max / w_min", f"3884{TIMES}", True)], K, cw=198)
    return b + s


# --- V2 ---------------------------------------------------------------------------
def v2_actinfo():
    K = "magenta"
    b = base("(V2) PiWM-actinfo -- InfoNCE over actions   [NEGATIVE]")
    b += wire_to_action(K, f"K permuted copies of a{sub('', 't', 11)}")
    b = sch(b)

    s = panel(K, "module:  the true action must beat K permuted ones at the SAME state")
    rows = [(936, sub("a", "t", 11), "E0", True),
            (1002, "a'", "E1", False), (1068, "a''", "E2", False)]
    BUS = 110
    s += pill(64, 1002, sub("z", "t", 11), K, rx=26)
    s += (f'<path d="M92,1002 L{BUS},1002 M{BUS},956 L{BUS},1088" '
          f'stroke="{ACCENT[K]}" stroke-width="1.8" fill="none" '
          f'stroke-linecap="round"/>\n')
    s += dot(BUS, 1002, K, r=3.6)
    for y, lab, ek, is_true in rows:
        col = K if is_true else "slate"
        wl = 2.0 if is_true else 1.4
        s += circle(160, y - 14, 14, lab, size=13,
                    fill=ACCENT_FILL[K] if is_true else "#eeeeea")
        s += aw(174, y - 13, 196, y - 12, col, w=wl)                  # a_k -> g
        s += apoly([(BUS, y + 20), (186, y + 20), (186, y + 12), (196, y + 12)],
                   col, w=wl)                                          # z_t -> g
        s += mbox(200, y - 22, 68, 44, "g", col, size=19)
        s += mbox(300, y - 22, 200, 44, nrm(f"h(g) {MINUS} z{sub('', 't+1', 10)}"),
                  col, size=15)
        s += aw(268, y, 296, y, col, w=wl)
        s += pill(540, y, ek, col, rx=26, fill=WHITE)
        s += aw(500, y, 512, y, col, w=wl)
    # three private corridors at x=592 into the softmax
    s += mbox(610, 948, 172, 108, f"softmax( {MINUS}E / {TAU} )", K, size=16,
              sub_="positive = the TRUE action")
    s += apoly([(566, 936), (592, 936), (592, 974), (606, 974)], K, w=2.0)
    s += apoly([(566, 1002), (606, 1002)], "slate", w=1.4)
    s += apoly([(566, 1068), (592, 1068), (592, 1030), (606, 1030)], "slate", w=1.4)
    s += pill(842, 1002, "L_ctrl", K, rx=42)
    s += aw(782, 1002, 796, 1002, K)
    s += eqrow(EQ1, [
        ["E", sub("", "k"), f"  =  {BAR}h( g(z", sub("", "t"), " , a", sub("", "k"),
         f") ) {MINUS} z", sub("", "t+1"), BAR + "²"],
        ["a", sub("", "0"), " = a", sub("", "t"), " ,   a", sub("", "1..K"),
         " = perm(a) within the batch"],
        [f"{TAU} = detached mean E"]], 16, gap=36)
    s += eqrow(EQ2, [
        [f"L_ctrl  =  {MINUS}log [ exp({MINUS}E", sub("", "0"), f"/{TAU}) / {SIGMA}",
         sub("", "k"), f" exp({MINUS}E", sub("", "k"), f"/{TAU}) ]"],
        [f"{IMPLIES}   max  I( a", sub("", "t"), " ; z", sub("", "t+1"), " | z",
         sub("", "t"), " )"]], 16)
    s += strip(PX + 26, SY, [
        ("d_action", "0.699  (max of any arm)", False),
        (f"code density {RHO}", f"0.448 {ARROWC} 0.052", True),
        ("CEM vs control", f"{MINUS}0.280  (p = 0.017)", True)], K, cw=264)
    return b + s


# --- V3 ---------------------------------------------------------------------------
def v3_pathint():
    K = "amber"
    b = base("(V3) PiWM-pathint -- the reference frame is path-integrated by the action"
             "   [NULL]")
    b += wire_to_action(K, "pose joins the action embedding")
    b = sch(b)

    s = panel(K, "module:  a learned pose map, rolled forward by the model itself")
    yA, yB = 936, 1048
    # Both blocks consume BOTH inputs, so the pair is bundled ONCE and fanned out. The
    # old drawing sent p_t down and a_t up through the same x=136 corridor: one line.
    s += circle(74, yA, 22, sub("p", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += circle(74, yB, 22, sub("a", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += mbox(108, 973, 86, 38, "[ p ; a ]", K, size=14)
    s += aw(90, yA + 18, 106, 976, K)
    s += aw(90, yB - 18, 106, 1008, K)
    s += aw(194, 981, 212, yA + 16, K)
    s += aw(194, 1003, 212, yB - 16, K)
    # row A -- the embedding the predictor consumes
    s += mbox(216, yA - 24, 190, 48, "W_a a_t  +  W_p p_t", K, size=16)
    s += pill(462, yA, "to g", K, rx=40, fill=WHITE)
    s += aw(406, yA, 418, yA, K)
    # row B -- the path-integration head and its supervision
    s += mbox(216, yB - 24, 190, 48, "f_path", K, size=19, sub_="[ p_t ; a_t ]")
    s += pill(470, yB, "p", K, rx=44, fill=WHITE, hat="t+1")
    s += aw(406, yB, 418, yB, K)
    s += pill(572, 978, sub("p", "t+1", 11), K, rx=36)
    s += opnode(572, yB, "minus", K)
    s += aw(512, yB, 554, yB, K)
    s += aw(572, 997, 572, yB - 18, K)
    s += mbox(626, yB - 24, 104, 48, nrm("."), K)
    s += aw(590, yB, 622, yB, K)
    s += mbox(770, yB - 24, 118, 48, "L_path", K, fill=ACCENT_FILL[K], size=18)
    s += aw(730, yB, 766, yB, K)
    s += eqrow(EQ1, [
        [("hat", "p", "t+1"), "  =  f_path( [ p", sub("", "t"), " ; a", sub("", "t"),
         " ] )"],
        [f"L  =  L_z  +  {LAMB} {DOT} {BAR}", ("hat", "p", "t+1"), f" {MINUS} p",
         sub("", "t+1"), BAR + "²"]], 17)
    s += eqrow(EQ2, [
        "rollout uses the SAME head:  p_k+1 = f_path( [ p_k ; a_k ] )",
        "warm start: the dataset fit (95.2% of the 1-step pose change)"], 15, gap=40)
    s += strip(PX + 26, SY, [
        (f"code density {RHO}", "0.573", False), ("rel_mse", "0.0094", False),
        ("d_action", "0.460", False),
        ("CEM vs control", f"{MINUS}0.025  (null)", True)], K, cw=198)
    return b + s


# --- the 2x2 the archive never ran ------------------------------------------------
def head(x, y, title, sub_):
    """The design system's left-aligned figure header: bold finding, muted subtitle."""
    s = txt(x, y, title, 22, anchor="start", weight="bold")
    return s + txt(x, y + 25, sub_, 14, anchor="start", fill=MUTED)


def badge(x, y, label, key, size=12.5, anchor="start"):
    """Pale fill, matching saturated text, thin matching border -- never the reverse. `y`
    is the badge's TOP edge, so a badge can be added to one cell of a grid without pushing
    that cell's other rows out of alignment with its neighbours'."""
    c = ACCENT[key]
    w = text_width(label, size, "bold") + 22
    x = x if anchor == "start" else round(x - w, 1)
    s = (f'<rect x="{x}" y="{y}" width="{round(w, 1)}" height="22" rx="11" '
         f'fill="{WHITE}" stroke="{c}" stroke-width="1.3"/>\n')
    return s + txt(x + w / 2, y + 15.5, label, size, fill=c, weight="bold")


def statrow(x, y, w, stats, key):
    """Three label/value pairs on a shared baseline. The values used to be one string with
    runs of spaces in it, and SVG collapses those: `rho 0.448   eff_dim 24.1` rendered as
    `rho 0.448 eff_dim 24.1`, with the numbers touching their next label."""
    s = ""
    for i, (lab, val) in enumerate(stats):
        cx = x + w * (i + 0.5) / len(stats)
        s += txt(cx, y, lab, 11.5, fill=MUTED)
        s += txt(cx, y + 21, val, 17, weight="bold")
    return s


def link_target_2x2():
    """The 2x2 the archive never ran: link x target_p. A data table, not a method."""
    X0, CW, GAPX = 182, 340, 28
    Y0, CH, GAPY = 152, 158, 26
    s = head(40, 52, f"The 2x2 that had never been run: link {TIMES} target_p",
             "173 runs at (reprelu, p=1), 35 at (identity, p=2), ZERO off-diagonal")
    for j, lab in enumerate(("target_p = 1   (Laplace)", "target_p = 2   (Gaussian)")):
        s += txt(X0 + j * (CW + GAPX) + CW / 2, Y0 - 18, lab, 14, fill=MUTED)
    cells = [
        (0, 0, "reprelu,  p = 1", "LpWM-ltv   (n = 16)", None,
         [(f"{RHO}", "0.448"), ("eff_dim", "24.1"), ("rel_mse", "0.0092")], "green"),
        (1, 0, "reprelu,  p = 2", "n = 8", "NEW this round",
         [(f"{RHO}", "0.509"), ("eff_dim", "26.1"), ("rel_mse", "0.0082")], "green"),
        (0, 1, "identity,  p = 1", "n = 8", "NEW this round",
         [(f"{RHO}", "1.000"), ("eff_dim", "2.9"), ("rel_mse", "0.052")], "crit"),
        (1, 1, "identity,  p = 2", "LeWM-ltv-p2   (n = 8)", None,
         [(f"{RHO}", "1.000"), ("eff_dim", "4.1"), ("rel_mse", "0.83")], "crit"),
    ]
    for cx, cy, title, who, tag, stats, key in cells:
        x, y = X0 + cx * (CW + GAPX), Y0 + cy * (CH + GAPY)
        s += (f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="10" '
              f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="2.2"/>\n')
        if tag:
            s += badge(x + 16, y + 14, tag, key)
        s += txt(x + CW / 2, y + 66, title, 19, fill=ACCENT[key], weight="bold")
        s += txt(x + CW / 2, y + 88, who, 12.5, fill=MUTED)
        s += statrow(x, y + 116, CW, stats, key)
    for i, lab in enumerate(("link = reprelu", "link = identity")):
        s += txt(X0 - 18, Y0 + i * (CH + GAPY) + CH / 2 + 6, lab, 15, anchor="end")
    s += txt(40, Y0 + 2 * CH + GAPY + 48, "Rows differ (link: p = 2.7e-06).  Columns do "
             "not (target_p: p = 0.21 / 0.80).", 16, anchor="start")
    s += txt(40, Y0 + 2 * CH + GAPY + 74, "A 1-D projection of a D=384 Laplace sample IS "
             f"Gaussian (Diaconis{NDASH}Freedman), and RDMReg only sees 1-D projections.",
             13, anchor="start", fill=MUTED)
    return s


def emit(name, body, out, w=W, h=H):
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + _hdr(w, h) + body + "</svg>\n")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, body, wh in (
        ("arch-v1-incr.svg", v1_incr(), (W, H)),
        ("arch-v2-actinfo.svg", v2_actinfo(), (W, H)),
        ("arch-v3-pathint.svg", v3_pathint(), (W, H)),
        ("arch-link-target-2x2.svg", link_target_2x2(), (940, 600)),
    ):
        print("  wrote", a.out + "/" + emit(name, body, a.out, *wh))


if __name__ == "__main__":
    main()
