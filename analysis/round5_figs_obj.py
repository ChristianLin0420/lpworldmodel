"""Architecture diagrams for the round-5 OBJECTIVE proposals (T4-T7).

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py: the
unmodified LpWM schematic from base() on top, and a MODULE PANEL below carrying the proposal
as blocks, operator nodes and arrows, with the equation it implements underneath and a short
strip of measured values. No prose paragraphs inside a diagram.

T4  value    -- hindsight relabelling gives a goal-conditioned V( z , g ) its labels for free.
T5  valueeq  -- the model loss becomes a matched Bellman backup, with V frozen.
T6  jumpy    -- one predictor call spans k frames, instead of composing the 1-step operator.
T7  energy   -- a frozen encoder and a scalar energy that RANKS whole action sequences.

STYLE.  analysis/style.py, via analysis/arch_figs.py: sans throughout, hues by IDENTITY, pale
fills with a matching saturated border, recessive furniture.  Maths is typeset -- subscripts,
a drawn circumflex, a drawn macron, a drawn composition ring, real Greek and a real norm --
because the ASCII stand-ins ("V-bar", "gamma", "zhat", "SUM_k", "h o g") were the loudest
thing on the page and read as identifiers rather than as maths.

LAYOUT RULES, and the defect each one exists to stop

  * Strict left-to-right dataflow on fixed row baselines; a wire that changes rows does so on
    its OWN vertical corridor. No two wires share a segment. The a_t circle at (318, 430) is
    fenced in by the link/encoder/RDMReg wiring, so a connector enters the predictor's flat
    top edge or the MSE gutter instead.
  * SHIFT reclaims the empty band above the schematic: base() draws nothing above y=158,
    which used to leave 158px of white at the top of every figure in this set.
  * Every footer line goes through frow(), which guarantees a gap between clauses and a hard
    right margin. Hand-placed footers used to reach the panel border: T5's "...so the 48%
    no-ops cost ~0" ended at x=905 inside a panel whose right edge is x=906.
  * Nothing is placed where a mark already is. T6's five action circles used to be centred on
    y=952, which is 5px BELOW its own panel's title rule -- the rule ran through all five.

Nothing drawn here puts privileged state into a training loss.

Usage:  python analysis/round5_figs_obj.py --out diary/assets/2026-09-03
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, INK, MUTED, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, BAR, DOT, MINUS, NDASH, SQ, TIMES, aw, apoly, dot, eq, eqseq, fit_size,
    hatrun, mbox, nb, nrm, opnode, pill, seq_w, strip, sup,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
PIN, PIX = PX + 26, PX + PW - 26      # the footer's inner margins
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-03")

# The whole figure is drawn in base()'s coordinates and then lifted. base() puts no ink
# above y=158; SHIFT turns that dead band into a normal top margin.
SHIFT = -80

GAMMA, TAU, SIGMA_, EPS_, NABLA, IN_ = "γ", "τ", "Σ", "ε", "∇", "∈"


def lift(body):
    return f'<g transform="translate(0,{SHIFT})">\n{body}</g>\n'


# --- primitives copied from analysis/arch_figs_predictor.py -----------------------
def panel(x, y, w, h, key, title):
    """Accent-bordered module panel with a title rule. Explicit geometry so each figure
    can size its own flow area; the title fits, so it can never cross the rule's end."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.0"/>\n')
    s += txt(x + 20, y + 32, title, fit_size(title, 18, w - 44, floor=14),
             anchor="start", fill=c, weight="bold")
    s += (f'<path d="M{x + 18},{y + 44} L{x + w - 18},{y + 44}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


# --- typeset maths ----------------------------------------------------------------
# Nimbus Sans and Liberation Sans both lack U+0304 (combining macron), U+1E91 (z-hat) and
# U+2218 (ring operator) -- verified by cmap lookup against the two installed faces -- so
# the bar, the hat and the ring are all drawn, exactly as arch_figs.caret() is.
def barrun(x, y, main, sub_s="", size=17, fill=INK, weight="normal"):
    """`main` under a drawn MACRON, plus an optional subscript, start-anchored at x.
    Returns (svg, advance) so a caller can chain runs on one baseline."""
    body = main if not sub_s else sub(main, sub_s, round(size * 0.72))
    s = txt(x, y, body, size, anchor="start", fill=fill, weight=weight)
    w0 = text_width(main[0], size, weight)
    # 0.98x, not 0.82x: at 0.82 the bar landed ON a capital V's apex and the pair read
    # as a nabla.  The cap height of these faces is 0.717em, so this clears it by 0.26em.
    y0 = round(y - 0.98 * size, 1)
    s += (f'<path d="M{round(x + 0.05 * w0, 1)},{y0} L{round(x + 0.95 * w0, 1)},{y0}" '
          f'stroke="{fill}" stroke-width="{max(1.1, 0.065 * size):.2f}" '
          f'stroke-linecap="round"/>\n')
    return s, text_width(body, size, weight)


def run_w(parts, size=17, weight="normal"):
    """Advance width of an mrun() part list."""
    w = 0.0
    for p in parts:
        if isinstance(p, tuple) and p[0] == "ring":
            w += 0.62 * size
        elif isinstance(p, tuple):
            main, sub_s = p[1], (p[2] if len(p) > 2 else "")
            w += text_width(main if not sub_s else sub(main, sub_s, round(size * 0.72)),
                            size, weight)
        else:
            w += text_width(p, size, weight)
    return w


def mrun(x, y, parts, size=17, fill=INK, weight="normal"):
    """Lay a maths run out left to right at a known baseline. A part is a string, or one of
    the tuples ("hat", main, sub) / ("bar", main, sub) / ("ring",)."""
    s, cx = "", float(x)
    for p in parts:
        if isinstance(p, tuple) and p[0] == "ring":
            r = round(0.17 * size, 1)
            s += (f'<circle cx="{round(cx + 0.31 * size, 1)}" '
                  f'cy="{round(y - 0.30 * size, 1)}" r="{r}" fill="none" '
                  f'stroke="{fill}" stroke-width="{max(1.1, 0.072 * size):.2f}"/>\n')
            cx += 0.62 * size
        elif isinstance(p, tuple):
            fn = barrun if p[0] == "bar" else hatrun
            frag, adv = fn(round(cx, 1), y, p[1], p[2] if len(p) > 2 else "", size, fill,
                           weight)
            s += frag
            cx += adv
        else:
            s += txt(round(cx, 1), y, nb(p), size, anchor="start", fill=fill,
                     weight=weight)
            cx += text_width(p, size, weight)
    return s


def mmid(cx, cy, parts, size=17, fill=INK, weight="normal"):
    """mrun(), centred on cx."""
    return mrun(round(cx - run_w(parts, size, weight) / 2, 1), cy, parts, size, fill,
                weight)


def rbox(x, y, w, h, parts, key, size=16, fill=WHITE, sub_=""):
    """mbox() whose label is a drawn maths run rather than a plain string."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
         f'stroke="{c}" stroke-width="1.8"/>\n')
    while size > 11.5 and run_w(parts, size) > w - 16:
        size -= 0.25
    s += mmid(x + w / 2, y + h / 2 + (6 if not sub_ else 0), parts, round(size, 2))
    if sub_:
        s += txt(x + w / 2, y + h / 2 + 17, sub_,
                 fit_size(sub_, 11.5, w - 14, floor=9.5), fill=MUTED)
    return s


def rpill(cx, cy, parts, key, rx=34, ry=19, fill=None, size=16):
    """pill() whose label is a drawn maths run."""
    c, f = ACCENT[key], (fill or ACCENT_FILL[key])
    while size > 12.5 and run_w(parts, size) > 2 * rx - 20:
        size -= 0.25
    rx = max(rx, run_w(parts, size) / 2 + 12)
    s = (f'<rect x="{round(cx - rx, 1)}" y="{cy - ry}" width="{round(2 * rx, 1)}" '
         f'height="{2 * ry}" rx="{ry}" fill="{f}" stroke="{c}" stroke-width="1.8"/>\n')
    return s + mmid(cx, cy + 6, parts, round(size, 2))


# --- the footer -------------------------------------------------------------------
def frow(y, cols, size=16, x0=PIN, x1=PIX, gap=34, floor=12.0):
    """One footer line of several clauses, justified between x0 and x1 with a GUARANTEED
    gap and a hard right margin.

    A clause is (text, fill, weight) or (text, fill, weight, scale); `text` may also be an
    mrun() part list, so a clause can carry a hatted or barred variable. The gap shrinks
    first and the type second -- a footer can never run past x1, which is what every
    overflow in this set was."""
    cols = [(c if isinstance(c, list) else [c], f, w, (p[3] if len(p) > 3 else 1.0))
            for p in cols for c, f, w in (p[:3],)]
    n, avail = len(cols), x1 - x0
    sz = float(size)
    while True:
        ws = [run_w(c, sz * k, w) for c, _, w, k in cols]
        if sum(ws) + gap * (n - 1) <= avail or n == 1:
            break
        if gap > 18:
            gap -= 2
        elif sz > floor:
            sz -= 0.25
        else:
            break
    if n == 1:
        c, f, w, k = cols[0]
        while sz > floor and run_w(c, sz * k, w) > avail:
            sz -= 0.25
        ws = [run_w(c, sz * k, w)]
    g = gap if n < 2 else max(gap, (avail - sum(ws)) / (n - 1))
    s, cx = "", float(x0)
    for (c, f, w, k), wd in zip(cols, ws):
        s += mrun(round(cx, 1), y, c, round(sz * k, 2), f, w)
        cx += wd + g
    return s


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h + SHIFT) + lift(body) + "</svg>\n")
    return name


# --- connectors into the unmodified schematic -------------------------------------
def hook(key, label, note=""):
    """Dashed connector into the objective the model minimises: the empty gutter BETWEEN
    the two code stacks (x 723..801), entered from BELOW their bottom edge (y 337) so it
    clears the MSE arrow and its label."""
    s = mbox(732, 312, 66, 36, label, key, size=15)
    s += apoly([(880, 898), (880, 366), (766, 366), (766, 352)], key, w=1.8, dash="6,4")
    s += apoly([(766, 312), (766, 284)], key, w=1.8, dash="6,4")
    if note:
        s += txt(866, 390, note, 13.5, anchor="end", fill=ACCENT[key], weight="bold")
    return s


def topedge(key, label):
    """The other legal entry point: the predictor's flat top edge, via the two empty
    corridors x=880 and y=120. The label sits BELOW its own corridor, so lifting the
    figure by SHIFT cannot push it off the top of the page."""
    s = apoly([(880, 898), (880, 142), (300, 142), (300, 190)], key, w=1.8, dash="6,4")
    s += txt(312, 164, label, 15.5, anchor="start", fill=ACCENT[key])
    return s


def strikeout(cx, cy, key="crit", r=13, w=2.6, gap=False):
    """A drawn cross. With `gap`, a white disc is punched through whatever it crosses
    first, so severing a wire reads as a break and not as a blot on top of it."""
    c = ACCENT[key]
    s = f'<circle cx="{cx}" cy="{cy}" r="{r + 3}" fill="{WHITE}"/>\n' if gap else ""
    return s + (f'<path d="M{cx - r},{cy - r} L{cx + r},{cy + r} M{cx + r},{cy - r} '
                f'L{cx - r},{cy + r}" stroke="{c}" stroke-width="{w}" '
                f'stroke-linecap="round"/>\n')


# --- local primitives -------------------------------------------------------------
def inset(x, y, w, h, title, key, note=""):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.6"/>\n')
    s += txt(x + w / 2, y + 26, title, fit_size(title, 17, w - 40, floor=13.5),
             fill=c, weight="bold")
    if note:
        s += txt(x + w / 2, y + 45, note, fit_size(note, 12, w - 30, floor=10),
                 fill=MUTED)
    return s


def sgbar(cx, cy, key="crit", h=11, gap=4):
    """The stop-gradient mark: two bars across a wire."""
    c = ACCENT[key]
    return (f'<path d="M{cx - gap},{cy - h} L{cx - gap},{cy + h} M{cx + gap},{cy - h} '
            f'L{cx + gap},{cy + h}" stroke="{c}" stroke-width="2.4" '
            f'stroke-linecap="round"/>\n')


def freeze(cx, cy, key, r=12, w=2.0):
    """A drawn snowflake: 'these weights do not move'. Geometry, not a glyph."""
    c = ACCENT[key]
    s = ""
    for deg in (90.0, 30.0, 150.0):
        a = math.radians(deg)
        dx, dy = r * math.cos(a), -r * math.sin(a)
        s += (f'<path d="M{cx - dx:.1f},{cy - dy:.1f} L{cx + dx:.1f},{cy + dy:.1f}" '
              f'stroke="{c}" stroke-width="{w}" stroke-linecap="round"/>\n')
        for sgn in (1.0, -1.0):
            ex, ey = cx + sgn * dx, cy + sgn * dy
            for off in (140.0, -140.0):
                b = math.radians(deg + off)
                bx = ex + sgn * 0.42 * r * math.cos(b)
                by = ey - sgn * 0.42 * r * math.sin(b)
                s += (f'<path d="M{ex:.1f},{ey:.1f} L{bx:.1f},{by:.1f}" stroke="{c}" '
                      f'stroke-width="{w * 0.8:.2f}" stroke-linecap="round"/>\n')
    return s


def cells(x, y, pattern, key, cell=18, gap=2, ghost=False):
    """A little code stack drawn horizontally: 1 = active."""
    c = ACCENT[key]
    s = ""
    for i, v in enumerate(pattern):
        f = (ACCENT[key] if v else WHITE) if not ghost else WHITE
        d = ' stroke-dasharray="4,3"' if ghost else ""
        s += (f'<rect x="{x + i * (cell + gap)}" y="{y}" width="{cell}" height="{cell}" '
              f'fill="{f}" fill-opacity="{0.85 if v and not ghost else 1}" '
              f'stroke="{DIM if ghost else c}" stroke-width="1.3"{d}/>\n')
    return s


def tick(cx, cy, key, r=9):
    c = ACCENT[key]
    s = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{c}" '
         f'stroke-width="1.5"/>\n')
    s += (f'<path d="M{cx - 4.5},{cy} L{cx - 1},{cy + 3.8} L{cx + 4.8},{cy - 4.2}" '
          f'stroke="{c}" stroke-width="2" fill="none" stroke-linecap="round" '
          f'stroke-linejoin="round"/>\n')
    return s


def checklist(x, y, w, items, key, h=58):
    """The NEW HEAD CHECKLIST: the lines path_int was missing, one ticked cell each. The
    cells are as wide as the row allows and every label fits its own cell."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.6"/>\n')
    s += txt(x + 14, y + 24, "NEW HEAD CHECKLIST", 13, anchor="start", fill=c,
             weight="bold")
    s += txt(x + 14, y + 44, "path_int had NONE of these", 11.5, anchor="start",
             fill=ACCENT["crit"])
    x0, cw = x + 196, (w - 206) / len(items)
    for i, (lab, where) in enumerate(items):
        cx = x0 + i * cw
        s += tick(cx + 13, y + 22, key)
        s += txt(cx + 30, y + 27, lab, fit_size(lab, 13, cw - 36, floor=10.5),
                 anchor="start")
        s += txt(cx + 30, y + 46, where, fit_size(where, 11, cw - 36, floor=9),
                 anchor="start", fill=MUTED)
    return s


# ------------------------------------------------------------------------ T4: value
def t4_value():
    """A goal-conditioned value head. Hindsight relabelling supplies the labels."""
    K = "blue"
    b = base("(T4) PiWM-value -- a goal-conditioned V( z , g ) trained by hindsight "
             "relabelling")
    b += hook(K, "V", "a NEW head on the codes")

    PY, PH = 898, 600
    s = panel(PX, PY, PW, PH, K,
              "module:  every later latent is a goal;  the step count is the label")
    s += txt(470, 966, f"hindsight relabelling {NDASH} no reward, no extra data, no "
             f"privileged state", 13, fill=MUTED)

    # -- row A: the trajectory, read as a set of labelled (state, goal) pairs.  The step
    # counts sit under the ARROWS, not under the pills: the goal's own corridor drops out
    # of pill 3's bottom edge, and a label centred there was 10px from it.
    xs = (96, 214, 332, 450, 568, 686)
    for i, cx in enumerate(xs):
        s += pill(cx, 1006, sub("z", "t" if not i else f"t+{i}", 10 + (i == 0)), K,
                  rx=34, ry=17, fill=ACCENT_FILL[K] if i in (0, 3) else WHITE)
        if i:
            s += aw(xs[i - 1] + 34, 1006, cx - 34, 1006, K)
            s += txt((xs[i - 1] + cx) / 2, 1044, f"{MINUS}{i}", 13, fill=ACCENT[K],
                     weight="bold")
    s += txt(450, 982, "goal  g", 13, fill=ACCENT[K], weight="bold")
    s += txt(792, 1044, "label = steps to the goal", 12, fill=MUTED)

    # -- z_t drops down its own corridor and fans to both rows off one spine
    s += (f'<path d="M96,1023 L96,1140 L136,1140" stroke="{ACCENT[K]}" stroke-width="1.8" '
          f'fill="none" stroke-linejoin="round"/>\n')
    s += (f'<path d="M136,1128 L136,1238" stroke="{ACCENT[K]}" stroke-width="1.8" '
          f'fill="none"/>\n')
    s += dot(136, 1140, K)
    s += aw(136, 1128, 158, 1128, K)
    s += aw(136, 1238, 158, 1238, K)
    # -- the goal drops out of node 3's bottom edge on its own corridor and runs in along
    # the empty y = 1066 line
    s += apoly([(450, 1023), (450, 1066), (191, 1066), (191, 1100)], K)

    # -- row B: the prediction  V( z , g )
    s += mbox(162, 1104, 58, 48, "[ ; ]", K, size=17)
    s += mbox(244, 1104, 146, 48, "V", K, size=20, sub_=f"MLP ( 2 {TIMES} 512 )")
    s += aw(220, 1128, 240, 1128, K)
    s += pill(466, 1128, "V( z , g )", K, rx=58, fill=WHITE)
    s += aw(390, 1128, 404, 1128, K)
    s += apoly([(524, 1128), (760, 1128), (760, 1167)], K)

    # -- row C: the TD target, with an expectile in place of the max
    s += mbox(162, 1214, 106, 48, "P", K, size=20, sub_="predictor")
    s += txt(346, 1188, "K sampled actions", 11.5, fill=MUTED)
    for k, yy in enumerate((1198, 1226, 1254)):
        s += rbox(292, yy, 108, 24, [("bar", "V")], K, size=13)
        s += aw(268, 1238 + (-10 if k == 0 else (0 if k == 1 else 10)),
                288, yy + 12, K, w=1.6)
        s += aw(400, yy + 12, 420, 1238 + (-10 if k == 0 else (0 if k == 1 else 10)),
                K, w=1.6)
    s += mbox(424, 1214, 132, 48, sub("max", "a", 12), K, size=18,
              sub_=f"expectile, {TAU}")
    s += rpill(660, 1238, [f"y  =  {MINUS}1 + {GAMMA} m"], K, rx=86,
               fill=ACCENT_FILL[K])
    s += aw(556, 1238, 570, 1238, K)
    s += apoly([(746, 1238), (760, 1238), (760, 1199)], K)
    s += mmid(346, 1292, [("bar", "V"), "  =  a frozen copy of V"], 11.5, MUTED)

    # -- the TD error
    s += opnode(760, 1183, "minus", K)
    s += aw(776, 1183, 792, 1183, K)
    s += mbox(796, 1159, 92, 48, nrm("."), K)
    s += txt(842, 1228, sub("L", "V", 12), 16, fill=ACCENT[K], weight="bold")

    s += frow(1326, [
        ("V( z , g )  =  MLP( [ z ; g ] )", INK, "normal"),
        (f"hindsight:   g  =  {sub('z', 't+k', 12)} ,    label  =  {MINUS}k",
         INK, "normal")])
    s += frow(1352, [
        ([f"y  =  {MINUS}1  +  {GAMMA} {DOT} {sub('max', 'a', 12)}  ", ("bar", "V"),
          "( P( z , a ) , g )"], INK, "normal"),
        ([f"expectile:   {sub('L', TAU, 12)}( d )  =  | {TAU} {MINUS} 1[ d &lt; 0 ] | "
          f"{DOT} " + sup("d", "2", 11)], INK, "normal")])
    s += checklist(PIN, 1366, PIX - PIN, [
        ("optimizer group", "train.py init_optimizers"),
        ("checkpoint key", "_keys_to_save"),
        ("lazy .to(device)", "built inside VWorldModel"),
        ("arm mirror", "tests/lpwm_build.py")], K)
    s += strip(PIN, 1432, [
        ("relabelled pairs", "free, from the demos", False),
        ("latent vs task dist", "Spearman +0.398", True),
        ("the V5 planner needs", "exactly this head", False),
        ("path_int precedent", "3 lines missing", True)], K, cw=198)
    return b + s


# ---------------------------------------------------------------------- T5: valueeq
def t5_valueeq():
    """The model loss becomes a matched Bellman backup instead of a latent MSE."""
    K = "green"
    b = base("(T5) PiWM-valueeq -- match Bellman backups, not next latents")
    b += hook(K, "V", "V REPLACES the MSE")
    # the MSE link is SEVERED: the cross punches a white gap through the arrow first, so
    # it reads as a break rather than as a crimson blot on a crimson arrow
    b += strikeout(766, 262, "crit", r=10, w=2.6, gap=True)

    PY, PH = 898, 580
    s = panel(PX, PY, PW, PH, K,
              "module:  the model is scored by the VALUE it implies, not by the latent "
              "it emits")

    # -- row A: the model's own backup
    s += pill(84, 1014, sub("z", "t", 11), K, rx=30, ry=16)
    s += circle(84, 1060, 20, sub("a", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += mbox(136, 1008, 96, 48, "P", K, size=20, sub_="predictor")
    s += aw(114, 1014, 132, 1022, K)
    s += aw(106, 1054, 132, 1044, K)
    s += rpill(296, 1032, [("hat", "z", "t+1")], K, rx=46, fill=WHITE)
    s += aw(232, 1032, 246, 1032, K)
    s += rbox(400, 1008, 150, 48, [("bar", "V"), "( . , g )"], K, size=16)
    s += freeze(534, 1020, K, r=8, w=1.5)
    s += aw(342, 1032, 396, 1032, K)
    s += pill(620, 1032, sub("v", "pred", 11), K, rx=46, fill=WHITE)
    s += aw(550, 1032, 570, 1032, K)
    s += apoly([(666, 1032), (742, 1032), (742, 1071)], K)

    # -- row B: the environment's backup
    s += circle(84, 1150, 24, sub("o", "t+1", 10), size=14)
    s += mbox(136, 1126, 96, 48, "Enc", K, size=18, sub_="trained")
    s += aw(108, 1150, 132, 1150, K)
    s += pill(296, 1150, sub("z", "t+1", 11), K, rx=46, fill=WHITE)
    s += aw(232, 1150, 246, 1150, K)
    s += rbox(400, 1126, 150, 48, [("bar", "V"), "( . , g )"], K, size=16)
    s += freeze(534, 1138, K, r=8, w=1.5)
    s += aw(342, 1150, 396, 1150, K)
    s += sgbar(370, 1150)
    s += txt(370, 1192, "stop-grad", 11, fill=ACCENT["crit"], weight="bold")
    s += pill(620, 1150, sub("v", "tgt", 11), K, rx=46, fill=WHITE)
    s += aw(550, 1150, 570, 1150, K)
    s += apoly([(666, 1150), (742, 1150), (742, 1103)], K)

    # -- the residual
    s += opnode(742, 1087, "minus", K)
    s += aw(758, 1087, 774, 1087, K)
    s += mbox(778, 1063, 110, 48, nrm("."), K)
    s += txt(833, 1128, sub("L", "veq", 12), 16, fill=ACCENT[K], weight="bold")
    note = [("bar", "V"), "  =  V from T4, its weights FROZEN"]
    nw = run_w(note, 12.5, "bold")
    s += mrun(round(480 - nw / 2, 1), 1094, note, 12.5, ACCENT[K], "bold")
    s += freeze(round(480 - nw / 2 - 20, 1), 1089, K, r=8, w=1.5)

    # -- inset: what metric that induces
    s += inset(60, 1210, 440, 134, "only value-relevant error costs anything", K)
    s += mrun(84, 1272, ["L  =  " + sup("( " + sub("v", "pred", 12) + f"  {MINUS}  "
                                        + sub("v", "tgt", 12) + " )", "2", 12)], 16)
    s += mrun(84, 1302, ["=  d" + sup("z", "T", 11) + " M dz  +  higher order"], 16)
    s += mrun(84, 1330, [f"M  =  {NABLA}V {NABLA}" + sup("V", "T", 11)
                         + f"   {NDASH}   rank 1 in D = 384"], 14)

    s += inset(520, 1210, 356, 134, "the metric it induces", K)
    s += txt(698, 1252, "level sets of V( . , g )", 11.5, fill=MUTED)
    for i in range(8):
        xx = 552 + i * 40
        s += (f'<path d="M{xx},1256 L{xx},1310" stroke="{ACCENT[K]}" stroke-width="1.2" '
              f'stroke-opacity="0.5"/>\n')
    s += aw(572, 1306, 572, 1264, "slate", w=1.8)
    s += txt(572, 1330, "no-op", 11, fill=MUTED)
    s += aw(612, 1282, 832, 1282, K, w=2.4)
    s += txt(730, 1330, "moves the block", 11, fill=ACCENT[K], weight="bold")

    s += frow(1374, [
        ([sub("L", "veq", 12) + "  =  (", ("bar", "V"),
          "( P( z , a ) , g ) " + MINUS + " ", ("bar", "V"),
          "( " + sub("z", "t+1", 12) + f" , g ) )" + SQ], INK, "normal"),
        ("the target side is detached", MUTED, "normal", 0.94)])
    s += frow(1400, [
        (f"the latent MSE weights every transition equally; this one weights each by how "
         f"much the VALUE moves {NDASH} so the 48% no-ops cost ~0", INK, "normal")],
        size=14)
    s += strip(PIN, 1414, [
        ("fully static", "48% of transitions", True),
        ("block moves &gt; 0.5 px", "25.3%", True),
        ("top 1% carry", "34.8% of all motion", True),
        ("V comes from", "T4, frozen", False)], K, cw=198)
    return b + s


# ------------------------------------------------------------------------ T6: jumpy
def t6_jumpy():
    """One predictor call spanning k frames, beside the k-step compounded rollout."""
    K = "amber"
    b = base("(T6) PiWM-jumpy -- predict z_t+5 in ONE call, not five composed calls")
    b += topedge(K, "one predictor call spans 5 frames")

    PY, PH = 898, 582
    s = panel(PX, PY, PW, PH, K,
              "module:  the horizon CEM plans over is the horizon the loss is written at")
    G = "<tspan font-style='italic'>g</tspan>"

    # -- row A: today. the 1-step operator, composed five times.  The action circles are
    # centred on y=986, below the panel's title rule at y=942: they used to be on y=952,
    # so the rule ran through all five of them.
    s += txt(52, 976, "TODAY", 14, anchor="start", fill=MUTED, weight="bold")
    s += pill(80, 1030, sub("z", "t", 11), K, rx=30, ry=17)
    s += aw(110, 1030, 124, 1030, K)
    bars = (4, 7, 11, 16, 22)
    for k in range(5):
        gx = 128 + k * 152
        s += circle(gx + 24, 978, 15, sub("a", str(k + 1), 9),
                    fill=ACCENT_FILL[K], size=12)
        s += aw(gx + 24, 993, gx + 24, 1003, K, w=1.6)
        s += mbox(gx, 1006, 48, 48, G, K, size=19)
        s += aw(gx + 48, 1030, gx + 62, 1030, K)
        cx = gx + 100
        s += rpill(cx, 1030, [("hat", "z", str(k + 1))], K, rx=34, ry=17, fill=WHITE)
        if k < 4:
            s += aw(cx + 34, 1030, gx + 152, 1030, K)
        s += (f'<rect x="{cx - 14}" y="1056" width="28" height="{bars[k]}" rx="2" '
              f'fill="{ACCENT["crit"]}" fill-opacity="0.6" stroke="{ACCENT["crit"]}" '
              f'stroke-width="1"/>\n')
    s += txt(836, 1096, "error compounds", 11.5, fill=ACCENT["crit"], weight="bold")

    # -- row B: one call.  The elbow leaves the z_t pill at x=86, clear of the T6 label.
    s += apoly([(86, 1047), (86, 1152), (124, 1152)], K)
    s += txt(52, 1090, "T6", 14, anchor="start", fill=ACCENT[K], weight="bold")
    for k in range(5):
        s += circle(146 + k * 34, 1100, 13, sub("a", str(k + 1), 8),
                    fill=ACCENT_FILL[K], size=11)
    s += aw(216, 1113, 216, 1124, K, w=1.6)
    s += mbox(128, 1128, 176, 48, G, K, size=20, sub_="one call,  k = 5")
    s += rpill(390, 1152, [("hat", "z", "t+5")], K, rx=52, fill=WHITE)
    s += aw(304, 1152, 334, 1152, K)
    s += pill(560, 1100, sub("z", "t+5", 11), K, rx=44)
    s += opnode(560, 1152, "minus", K)
    s += aw(442, 1152, 544, 1152, K)
    s += aw(560, 1119, 560, 1136, K)
    s += mbox(600, 1128, 110, 48, nrm("."), K)
    s += aw(576, 1152, 596, 1152, K)
    s += pill(802, 1152, sub("L", "jump", 12), K, rx=58, fill=ACCENT_FILL[K])
    s += aw(710, 1152, 740, 1152, K)

    # -- what the config pins today
    s += inset(60, 1218, 816, 124, "what the config pins today", K)
    for i, (lab, kk) in enumerate((("num_pred = 1", "crit"), ("num_hist = 3", K),
                                   ("frameskip = 5", K), ("CEM horizon = 5", K))):
        s += mbox(108 + i * 184, 1260, 168, 44, lab, kk, size=15,
                  fill=ACCENT_FILL["crit"] if kk == "crit" else WHITE)
    s += txt(468, 1328, "the training loss has never evaluated a 5-step rollout", 13,
             fill=ACCENT[K], weight="bold")

    s += frow(1376, [
        (["today:   ", ("hat", "z", "t+H"), "  =  ( h ", ("ring",), "  " + G + " )"
          + sup("", "H", 12) + "( " + sub("z", "t", 12) + " ; "
          + sub("a", "1:H", 12) + " )"], INK, "normal"),
        (["T6:   ", ("hat", "z", "t+k"), "  =  " + sub(G, "k", 12) + "( "
          + sub("z", "t", 12) + " ; " + sub("a", "1:k", 12) + " )"], INK, "normal")],
        size=16)
    s += frow(1402, [
        ([f"L  =  {SIGMA_}" + sub("", "k", 12) + "  " + BAR, ("hat", "z", "t+k"),
          f" {MINUS} " + sub("z", "t+k", 12) + BAR + SQ], INK, "normal"),
        (f"k {IN_} {{ 1 , 5 }} {NDASH} one head per jump, or k as a conditioning input",
         INK, "normal", 0.9)])
    s += strip(PIN, 1416, [
        ("num_pred today", "1", True),
        ("CEM horizon", "5", False),
        ("frameskip", "5  (act dim 10)", False),
        ("what compounds", "5 × ( h · g )", True)], K, cw=198)
    return b + s


# ----------------------------------------------------------------------- T7: energy
def t7_energy():
    """A scalar energy over WHOLE action sequences, on a frozen encoder."""
    K = "magenta"
    b = base("(T7) PiWM-energy -- a frozen encoder and a scalar energy over action "
             "sequences")
    b += hook(K, "E", "E replaces the MSE")
    # the freeze, drawn on both encoder domes
    for cx in (120, 812):
        b += freeze(cx, 556, K, r=13)
        b += txt(cx, 588, "frozen", 12, fill=ACCENT[K], weight="bold")

    PY, PH = 898, 604
    s = panel(PX, PY, PW, PH, K,
              "module:  E ranks the EXECUTED sequence below the ones CEM would sample")

    # -- the conditioning both branches share
    s += pill(72, 1084, sub("z", "t", 11), K, rx=26, ry=15)
    s += pill(72, 1132, sub("z", "g", 11), K, rx=26, ry=15)
    s += mbox(126, 1084, 54, 48, "[ ; ]", K, size=17)
    s += aw(98, 1084, 122, 1094, K)
    s += aw(98, 1132, 122, 1122, K)
    s += (f'<path d="M180,1108 L206,1108 M206,1054 L206,1162" stroke="{ACCENT[K]}" '
          f'stroke-width="1.8" fill="none"/>\n')
    s += dot(206, 1108, K)
    s += aw(206, 1054, 226, 1054, K)
    s += aw(206, 1162, 226, 1162, K)

    # -- positives: the sequence the demo actually executed
    s += txt(290, 972, "positives:  the EXECUTED sequence", 12.5, fill=ACCENT[K],
             weight="bold")
    for k in range(5):
        s += circle(238 + k * 26, 1000, 11, sub("a", str(k + 1), 8),
                    fill=ACCENT_FILL[K], size=10)
    s += aw(290, 1011, 290, 1026, K, w=1.6)
    s += mbox(230, 1030, 120, 48, "E", K, size=20, sub_="scalar head")
    s += rpill(420, 1054, [sup("E", "+", 11)], K, rx=38, fill=ACCENT_FILL[K])
    s += aw(350, 1054, 378, 1054, K)

    # -- negatives: CEM's own proposal
    for k in range(5):
        s += circle(238 + k * 26, 1106, 11, "", fill=WHITE, size=10)
    s += txt(366, 1110, "K sampled  " + sub("ã", "1:H", 9), 11, anchor="start",
             fill=MUTED)
    s += aw(290, 1117, 290, 1134, K, w=1.6)
    s += mbox(230, 1138, 120, 48, "E", K, size=20, sub_="the SAME weights")
    for k, yy in enumerate((1134, 1162, 1190)):
        lab = sub(sup("E", MINUS, 10), ["1", "2", "K"][k], 10)
        s += rpill(420, yy, [lab], K, rx=34, ry=11, fill=WHITE, size=13)
        s += aw(350, 1162 + (-10 if k == 0 else (0 if k == 1 else 10)),
                384, yy, K, w=1.6)
    s += txt(270, 1218, "negatives:  the sequences CEM would sample", 11.5, fill=MUTED)

    # -- the ranking loss
    s += rbox(500, 1078, 180, 60, ["softmax rank"], K, size=16,
              sub_=sup("E", "+", 10) + " must be the lowest")
    s += apoly([(458, 1054), (478, 1054), (478, 1090), (496, 1090)], K)
    s += apoly([(454, 1162), (478, 1162), (478, 1126), (496, 1126)], K)
    s += pill(790, 1108, sub("L", "rank", 12), K, rx=60, fill=ACCENT_FILL[K])
    s += aw(680, 1108, 726, 1108, K)

    # -- why V2's collapse is structurally unavailable
    s += inset(60, 1236, 816, 130, "why the V2 collapse cannot recur", K)
    s += txt(250, 1288, "V2 / P5:   the encoder was TRAINABLE", 12.5,
             fill=ACCENT["crit"], weight="bold")
    s += cells(96, 1302, (1, 1, 0, 1, 1, 0), "crit")
    s += aw(214, 1311, 262, 1311, "crit", w=1.8)
    s += cells(272, 1302, (0, 0, 0, 1, 0, 0), "crit")
    s += txt(250, 1348, f"ρ  0.448  {ARROWC}  0.052", 12,
             fill=ACCENT["crit"], weight="bold")
    s += (f'<path d="M468,1280 L468,1352" stroke="{ACCENT[K]}" stroke-width="1.2" '
          f'stroke-opacity="0.45"/>\n')
    s += txt(668, 1288, "T7:   the encoder is FROZEN", 12.5, fill=ACCENT[K],
             weight="bold")
    s += cells(520, 1302, (1, 1, 0, 1, 1, 0), K)
    s += freeze(656, 1311, K, r=9, w=1.6)
    s += aw(674, 1311, 720, 1311, "slate", w=1.8, dash="6,4")
    s += strikeout(697, 1311, "crit", r=9, w=2.2)
    s += cells(726, 1302, (0, 0, 0, 1, 0, 0), K, ghost=True)
    s += txt(668, 1348, "z is a CONSTANT of the optimisation", 12, fill=ACCENT[K],
             weight="bold")

    s += frow(1396, [
        (f"E :  ( z , {sub('a', '1:H', 12)} , {sub('z', 'g', 12)} )   {ARROWC}   scalar "
         f"cost-to-go", INK, "normal"),
        (f"z = Enc( o ) with Enc frozen {NDASH} 0 trainable encoder params",
         INK, "normal", 0.92)])
    s += frow(1422, [
        (sub("L", "rank", 12) + "  =  " + MINUS + "log [ exp( " + MINUS
         + sup("E", "+", 11) + " )  /  ( exp( " + MINUS + sup("E", "+", 11) + " )  +  "
         + SIGMA_ + sub("", "k", 11) + " exp( " + MINUS
         + sub(sup("E", MINUS, 11), "k", 11) + " ) ) ]", INK, "normal"),
        ("negatives where CEM searches", MUTED, "normal", 0.94)])
    s += strip(PIN, 1438, [
        ("what trains", "E only", False),
        ("encoder", "frozen, 0 grads", False),
        ("V2 collapse mode", "ρ 0.448 → 0.052", True),
        ("negatives from", "the CEM proposal", False)], K, cw=198)
    return b + s




# ------------------------------------------------------------------- the four defects
DW, DH = 940, 900
QW, QH = 438, 384
QX = (26, 476)
QY = (78, 486)


def quad(i, j, key, tag, title, where):
    """Quadrant frame: number chip, title, and the file:line that proves it."""
    x, y, c = QX[j], QY[i], ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{QW}" height="{QH}" rx="10" '
         f'fill="{ACCENT_FILL[key]}" fill-opacity="0.4" stroke="{c}" '
         f'stroke-width="2.4"/>\n')
    s += (f'<rect x="{x + 14}" y="{y + 12}" width="30" height="24" rx="5" fill="{c}"/>\n')
    s += txt(x + 29, y + 30, tag, 15, fill=WHITE, weight="bold")
    s += txt(x + 52, y + 30, title, 15, anchor="start", fill=c, weight="bold")
    s += txt(x + 52, y + 48, where, 11.5, anchor="start", fill=MUTED, style="italic")
    s += (f'<path d="M{x + 14},{y + 58} L{x + QW - 14},{y + 58}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def code(x, y, w, lines, key, size=11.5):
    """A monospace source excerpt."""
    c = ACCENT[key]
    h = 13 + 18 * len(lines)
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.3" stroke-opacity="0.65"/>\n')
    for k, ln in enumerate(lines):
        s += (f'<text x="{x + 11}" y="{y + 21 + 18 * k}" font-size="{size}" '
              f'font-family="monospace" fill="{INK}">{ln}</text>\n')
    return s


def verdict(x, y, w, text, key, size=13.5):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="32" rx="5" fill="{c}" '
         f'fill-opacity="0.14" stroke="{c}" stroke-width="1.6"/>\n')
    return s + txt(x + w / 2, y + 21, text, size, fill=c, weight="bold")


def d1():
    """The success metric scores the agent's own position."""
    K, x, y = "crit", QX[0], QY[0]
    s = quad(0, 0, K, "1", "Success scores the agent's own position",
             "env/pusht/pusht_wrapper.py:62")
    s += code(x + 18, y + 72, QW - 36, [
        "pos_diff = norm(goal_state[:4] - cur_state[:4])"], K)
    yb, cw = 204, 52
    x0 = x + (QW - (7 * cw + 6 * 4)) / 2
    for k, (lab, agent) in enumerate((("agent x", True), ("agent y", True),
                                      ("T x", False), ("T y", False),
                                      ("T theta", False), ("vel", False),
                                      ("vel", False))):
        cx, scored = x0 + k * (cw + 4), k < 4
        s += (f'<rect x="{cx}" y="{yb}" width="{cw}" height="40" rx="4" '
              f'fill="{ACCENT[K] if agent else WHITE}" '
              f'fill-opacity="{0.85 if agent else 1}" stroke="{ACCENT[K]}" '
              f'stroke-width="{2.2 if scored else 1.1}" '
              f'stroke-opacity="{1 if scored else 0.45}"/>\n')
        s += txt(cx + cw / 2, yb + 25, lab, 10.5,
                 fill=WHITE if agent else (INK if scored else MUTED))
    xl, xr = x0, x0 + 4 * (cw + 4) - 4
    s += (f'<path d="M{xl},{yb + 50} L{xl},{yb + 60} L{xr},{yb + 60} L{xr},{yb + 50}" '
          f'stroke="{ACCENT[K]}" stroke-width="2" fill="none"/>\n')
    s += txt((xl + xr) / 2, yb + 84, "state[:4]  --  the scored ball", 13,
             fill=ACCENT[K], weight="bold")
    s += txt(x + QW / 2, yb + 112,
             "2 of the 4 scored dimensions are the agent's own", 12.5)
    s += txt(x + QW / 2, yb + 130, "position: directly actuated, reached in one step,",
             12.5)
    s += txt(x + QW / 2, yb + 148, "no contact and no physics", 12.5)
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "half the metric never requires touching the T", K)
    return s


def d2():
    """Success is latched over up to ten replans."""
    K, x, y = "amber", QX[1], QY[0]
    s = quad(0, 1, K, "2", "Success is latched, not terminal",
             "planning/mpc.py:110-112, 60-70")
    s += code(x + 18, y + 72, QW - 36, [
        "while not all(is_success) and iter &lt; max_iter:",
        "&#160;&#160;&#160;&#160;self.is_success |= successes"], K)
    yb, x0 = 252, x + 34
    for k in range(10):
        cx, hit = x0 + k * 38, k == 5
        s += (f'<circle cx="{cx}" cy="{yb}" r="12" '
              f'fill="{ACCENT[K] if hit else WHITE}" stroke="{ACCENT[K]}" '
              f'stroke-width="{2.4 if hit else 1.4}"/>\n')
        s += txt(cx, yb + 4, str(k + 1), 11, fill=WHITE if hit else MUTED)
        if k < 9:
            s += aw(cx + 13, yb, cx + 24, yb, K, w=1.3)
    s += txt(x0 + 5 * 38, yb - 26, "in the ball once", 12, fill=ACCENT[K], weight="bold")
    s += (f'<path d="M{x0 + 5 * 38},{yb - 20} L{x0 + 5 * 38},{yb - 14}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.8"/>\n')
    s += txt(x + QW / 2, yb + 42, "reported success  =  OR over all ten replans", 13.5,
             weight="bold")
    s += txt(x + QW / 2, yb + 64, "a maximum over noisy draws, so the number rises", 12.5)
    s += txt(x + QW / 2, yb + 82, "with max_iter, with the model held fixed", 12.5)
    s += txt(x + QW / 2, yb + 106, "_apply_success_mask then freezes the episode", 11.5,
             fill=MUTED)
    s += txt(x + QW / 2, yb + 122, "at the normalised hold ( +0.0431 , -0.0340 )", 11.5,
             fill=MUTED)
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "not \"ended at the goal\" but \"passed through it\"", K)
    return s


def d3():
    """path_int has no optimizer, no checkpoint key, and is absent from every ckpt."""
    K, x, y = "magenta", QX[0], QY[1]
    s = quad(1, 0, K, "3", "path_int has no optimizer, no checkpoint",
             "train.py init_optimizers / _keys_to_save; visual_world_model.py:383-389")
    s += txt(x + QW / 2, 570, "optimizer groups built by init_optimizers", 13,
             fill=ACCENT[K], weight="bold")
    x0 = x + (QW - (4 * 92 + 3 * 8)) / 2
    for k, lab in enumerate(("encoder", "predictor", "decoder", "act / prop")):
        s += mbox(x0 + k * 100, 582, 92, 40, lab, K, size=12.5)
    s += (f'<rect x="{x + 18}" y="646" width="148" height="40" rx="5" fill="{WHITE}" '
          f'stroke="{ACCENT[K]}" stroke-width="2" stroke-dasharray="7,5"/>\n')
    s += txt(x + 92, 671, "path_int", 15, fill=MUTED)
    for k, ln in enumerate(("no optimizer group", "no _keys_to_save entry",
                            "absent from every checkpoint")):
        s += txt(x + 206, 660 + k * 18, ln, 12, anchor="start", fill=ACCENT[K])
    s += code(x + 18, 706, QW - 36, [
        "loss = (path_int(x) - proprio.detach()) ** 2",
        "# both leaves are data: it reaches no parameter"], K, size=11)
    s += txt(x + QW / 2, 790, "_roll_pose then rolls a RANDOM nn.Linear at plan time",
             12.5, fill=ACCENT[K], weight="bold")
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "V3 tested a random rollout head, not path integration", K, size=12.5)
    return s


def d4():
    """No decoder was ever attached, and the branch that exists detaches."""
    K, x, y = "blue", QX[1], QY[1]
    s = quad(1, 1, K, "4", "No decoder has ever been attached",
             "conf/train_rdmreg.yaml:126, train.py:463, visual_world_model.py:728")
    yb = 596
    s += circle(x + 52, yb, 26, sub("o", "t", 11), size=15)
    s += mbox(x + 106, yb - 26, 92, 52, "encoder", K, size=14)
    s += aw(x + 80, yb, x + 102, yb, K)
    s += pill(x + 250, yb, sub("z", "t", 11), K, rx=28, ry=17)
    s += aw(x + 198, yb, x + 218, yb, K)
    s += (f'<rect x="{x + 330}" y="{yb - 26}" width="92" height="52" rx="5" '
          f'fill="{WHITE}" stroke="{ACCENT[K]}" stroke-width="2" '
          f'stroke-dasharray="7,5" stroke-opacity="0.6"/>\n')
    s += txt(x + 376, yb + 5, "decoder", 14, fill=MUTED)
    s += aw(x + 278, yb, x + 326, yb, K, w=1.6, dash="7,5")
    s += txt(x + 376, yb - 40, "has_decoder: False", 11, fill=ACCENT[K], weight="bold")
    s += txt(x + 376, yb + 44, "and the adaln branch", 11, fill=MUTED)
    s += txt(x + 376, yb + 58, "passes z_emb.detach()", 11, fill=MUTED)
    s += txt(x + QW / 2, 690, "the only pressure on the encoder is that its own", 12.5)
    s += txt(x + QW / 2, 708, "output be predictable from its own past", 12.5)
    s += txt(x + QW / 2, 734, "RDMReg blocks DIMENSIONAL collapse", 13,
             fill=ACCENT[K], weight="bold")
    s += txt(x + QW / 2, 752, "-- a full-rank latent can still encode only proprio,",
             12)
    s += txt(x + QW / 2, 770, "which is a free input to the predictor", 12)
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "nothing has required the latent to keep visual content", K, size=12.5)
    return s


def defects_4():
    s = txt(DW / 2, 40, "Four verified defects that precede every model question", 22,
            weight="bold")
    s += txt(DW / 2, 62, "each read from the source, not inferred -- 3 retracts a "
             "published conclusion, 4 explains four separate collapses", 13, fill=MUTED)
    return s + d1() + d2() + d3() + d4()


FIGURES = (
    ("arch-t4-value.svg", t4_value, (940, 1524)),
    ("arch-t5-valueeq.svg", t5_valueeq, (940, 1504)),
    ("arch-t6-jumpy.svg", t6_jumpy, (940, 1506)),
    ("arch-t7-energy.svg", t7_energy, (940, 1528)),
    # arch-defects-4.svg is emitted by analysis/arch_figs_defects.py, which OWNS that
    # filename and carries the current layout. Two modules writing one path is how a
    # figure silently reverts, so the body below stays out of this tuple; it is kept only
    # so the alternative 940x900 layout is not lost.
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn, wh in FIGURES:
        print("  wrote", a.out + "/" + emit(name, fn(), a.out, *wh))


if __name__ == "__main__":
    main()
