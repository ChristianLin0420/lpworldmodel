"""Architecture diagrams for round 5 (V1-V5, the PLANNER proposals), in the LpWM style.

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py:
the unmodified LpWM schematic from `base()` on top -- nudged up by sch(), so the set does
not carry a band of white above every figure -- and a MODULE PANEL below carrying the
proposed change as blocks, operator nodes and arrows, with the equation it implements and
a strip of measured values. No prose inside a diagram.

These five differ from rounds 4 and 5's model proposals in one way: nothing here changes
the trained model. V1, V2, V3 and V5 change what the PLANNER does with a model that is
already trained; V4 deletes the planner.

WHAT THIS FILE OWES THE DESIGN SYSTEM (analysis/style.py, via arch_figs)

  * Hue by IDENTITY, never by rank. V1 is the campaign's one positive result, so it is
    drawn in the system green (#18483C) -- "the thing that works". V2 takes the teal;
    they used to be the other way round, which put two near-identical greens on the two
    figures a reader compares first. V3 amber (an intervention), V4 purple (the
    contrasting condition: no planner at all), V5 crimson (Sutton's warning).
  * Every connector into the schematic uses one of the SET's named routes -- wire_to_pred,
    or wire_to_score below, which hops the one model wire it has to cross. An unhopped
    crossing reads as a junction.
  * Strict left-to-right dataflow on fixed row baselines; a wire that changes rows does so
    on its own vertical corridor.
  * Nothing is sized by hand where it can be measured: mbox/pill/partbox/strip fit their
    own text and eqrow guarantees a gap between clauses. V1's two equation clauses used to
    sit 4px apart and read as one run-on sentence.
  * Text goes in free space and is owned by the mark beside it. V1's and V5's "M members
    vote" / "V replaces the MSE" used to be drawn straight across the right-hand link
    wire; V2's "leaf, requires_grad" sat 300px from the leaf it names.

Usage:  python analysis/round5_figs_planners.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, DIM, MUTED, WHITE,
    _hdr, arrow, base, circle, sub, sup, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402
    ARROWC, BAR, DOT, LAMB, MINUS, NDASH, PIX, PW, PX, PY, PYL, TIMES,
    apoly, aw, dot, eq, eqmid, eqrow, eqseq, fit_size, hpoly, mbox, pill, sch, seq_w,
    strip, wire_to_pred,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-03")
ETA = "η"
HINT = 1.06     # cairo rounds every glyph advance UP to a whole pixel at these sizes, so
                # a long line renders ~5% wider than the AFM table says, and fit_size()
                # trusts the table. V2's second equation clause reached the panel border.


def eqline(x, y, s, size=15, weight="normal"):
    """One equation line that cannot run past the panel, hinting included."""
    return eq(x, y, s, fit_size(s, size, (PIX - 8 - x) / HINT, floor=11.5,
                                weight=weight), weight)


def eqcols(y, cols, size=17, gap=44, x0=None):
    """eqrow() with the same allowance."""
    x0 = PX + 26 if x0 is None else x0
    return eqrow(y, cols, size, x0=x0, x1=x0 + (PIX - 8 - x0) / HINT, gap=gap)


def _nd(s):
    """The house dash. Labels are written with ASCII `--` and set as an en dash once."""
    return s.replace(" -- ", f"  {NDASH}  ")


# --- the module panel -------------------------------------------------------------
def panel(x, y, w, h, key, title):
    """Accent-bordered module panel with a title rule. Explicit geometry, because these
    five panels are five different heights; everything else matches arch_figs_causal."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    title = _nd(title)
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.2"/>\n')
    s += txt(x + 20, y + 32, title, fit_size(title, 18, w - 44, floor=14),
             anchor="start", fill=c, weight="bold")
    s += (f'<path d="M{x + 18},{y + 44} L{x + w - 18},{y + 44}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --- the connector this round needs -----------------------------------------------
def wire_to_score(key, label, note):
    """Into the objective the planner minimises: the empty gutter BETWEEN the two right
    code stacks (x 723..801), entered from below their bottom edge so it clears the MSE
    arrow and its label. It crosses the right-hand link wire exactly once, at (812, 380),
    and HOPS it -- an unhopped crossing there reads as a junction.

    Same route as arch_figs_causal.wire_to_loss, but what lands in the gutter is a BLOCK,
    not an operator: V1 and V5 replace the MSE rather than scaling it."""
    s = hpoly([(880, PYL), (880, 380), (762, 380), (762, 352)], key, dash="6,4",
              hops=[(812, 380)])
    s += mbox(729, 314, 66, 36, label, key, size=15, fill=ACCENT_FILL[key])
    s += apoly([(762, 314), (762, 276)], key, dash="6,4")
    return s + txt(690, 304, note, 15.5, anchor="end", fill=ACCENT[key])


# --- local primitives -------------------------------------------------------------
def partbox(x, y, w, h, parts, key, size=15, fill=WHITE, sub_=""):
    """A block whose label is an eqseq() part list -- so it can carry a HATTED variable.
    mbox() cannot: the hat is drawn geometry, not a glyph the font has."""
    c = ACCENT[key]
    while size > 10.5 and seq_w(parts, size) > w - 16:
        size -= 0.25
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
         f'stroke="{c}" stroke-width="1.8"/>\n')
    s += eqmid(x + w / 2, y + h / 2 + (6 if not sub_ else 0), parts, round(size, 2))
    if sub_:
        s += txt(x + w / 2, y + h / 2 + 17, sub_,
                 fit_size(sub_, 11.5, w - 14, floor=9.5), fill=MUTED)
    return s


def partpill(cx, cy, parts, key, rx=34, ry=19, fill=None, size=16):
    """pill(), with an eqseq() part list for a label."""
    c, f = ACCENT[key], (fill or ACCENT_FILL[key])
    while size > 12 and seq_w(parts, size) > 2 * rx - 20:
        size -= 0.25
    rx = max(rx, seq_w(parts, size) / 2 + 12)
    s = (f'<rect x="{round(cx - rx, 1)}" y="{cy - ry}" width="{round(2 * rx, 1)}" '
         f'height="{2 * ry}" rx="{ry}" fill="{f}" stroke="{c}" stroke-width="1.8"/>\n')
    return s + eqmid(cx, cy + 6, parts, round(size, 2))


def card(x, y, w, h, title, formula, note, key, hot=False):
    """One of several alternatives shown side by side."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="{2.4 if hot else 1.4}"/>\n')
    s += txt(x + w / 2, y + 27, title, 18, fill=c, weight="bold")
    s += txt(x + w / 2, y + 52, formula, fit_size(formula, 15, w - 24, floor=12.5))
    s += txt(x + w / 2, y + 70, _nd(note), fit_size(note, 12, w - 20, floor=10),
             fill=MUTED)
    return s


def inset(x, y, w, h, title, key, note=""):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.6"/>\n')
    s += txt(x + w / 2, y + 26, _nd(title), fit_size(title, 17, w - 24, floor=14),
             fill=c, weight="bold")
    if note:
        s += txt(x + w / 2, y + 45, note, 12, fill=MUTED)
    return s


def acell(x, y, w, h, label, key, fill=WHITE, size=12):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
         f'stroke="{c}" stroke-width="1.4"/>\n')
    return s + txt(x + w / 2, y + h / 2 + 4, label, size)


def ghost(x, y, w, h, label, size=15):
    """A block the proposal DELETES: neutral, dashed, struck through."""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
         f'fill="{ACCENT_FILL["slate"]}" fill-opacity="0.6" stroke="{DIM}" '
         f'stroke-width="1.6" stroke-dasharray="7,4"/>\n')
    s += txt(x + w / 2, y + h / 2 + 5, label, size, fill=MUTED)
    c = ACCENT["crit"]
    s += (f'<path d="M{x + 22},{y + h / 2} L{x + w - 22},{y + h / 2}" stroke="{c}" '
          f'stroke-width="1.8" stroke-linecap="round" stroke-opacity="0.6"/>\n')
    return s


def vdots(cx, cy, key="slate", n=3, gap=6, r=1.9):
    c = ACCENT[key]
    return "".join(f'<circle cx="{cx}" cy="{cy + i * gap}" r="{r}" fill="{c}"/>\n'
                   for i in range(n))


# ------------------------------------------------------------------ V1: pessimism
def v1_pessimism():
    """The campaign's one positive result, so it carries the system green."""
    K = "blue"
    b = base("(V1) PiWM-pessimism -- combine the M members' RANKS, pessimistically")
    b += wire_to_score(K, "vote", "M members vote")
    b = sch(b)

    PH = 500
    s = panel(PX, PY, PW, PH, K,
              "module:  M member rollouts  ->  per-member ranks  ->  one score CEM sorts on"
              .replace(" -> ", f" {ARROWC} "))
    ys = (922, 968, 1014)
    tags = ("1", "2", "M")
    s += pill(92, 968, sub("a", "1:H", 10), K, rx=44)
    s += (f'<path d="M136,968 L156,968 M156,922 L156,1014" stroke="{ACCENT[K]}" '
          f'stroke-width="1.8" fill="none" stroke-linecap="round"/>\n')
    s += dot(156, 968, K, r=3.6)
    s += vdots(168, 986, key=K, gap=5, r=1.7)
    for y, tg in zip(ys, tags):
        s += aw(156, y, 176, y, K)
        s += mbox(180, y - 20, 104, 40, sub("WM", tg, 11), K, size=16)
        s += aw(284, y, 306, y, K)
        s += partbox(310, y - 20, 180, 40,
                     [BAR, ("hat", "z", "H"), f" {MINUS} z", sub("", "g"), BAR,
                      sup("", "2", 11)], K, size=14)
        s += aw(490, y, 512, y, K)
        s += pill(542, y, sub("L", tg, 11), K, rx=30, ry=17)
        s += aw(572, y, 594, y, K)
        s += mbox(598, y - 20, 108, 40, sub("rank", tg, 11), K, size=15)
    # three private corridors into the rank bus -- no two share a segment
    s += mbox(752, 944, 132, 48, f'r{sub("", "1", 11)}  ...  r{sub("", "M", 11)}',
              K, size=15)
    s += apoly([(706, 922), (728, 922), (728, 952), (748, 952)], K)
    s += aw(706, 968, 748, 968, K)
    s += apoly([(706, 1014), (728, 1014), (728, 984), (748, 984)], K)
    # the rank spine, feeding the three combination rules
    s += (f'<path d="M818,992 L818,1048 M188,1048 L818,1048" stroke="{ACCENT[K]}" '
          f'stroke-width="1.8" fill="none" stroke-linejoin="round"/>\n')
    for cx in (188, 470, 752):
        s += aw(cx, 1048, cx, 1058, K)
    s += card(64, 1062, 248, 76, "borda",
              f's  =  mean{sub("", "m", 11)}  r{sub("", "m", 11)}',
              "already implemented (rule = borda)", K)
    s += card(346, 1062, 248, 76, "cvar", f"s  =  mean  +  {LAMB} {DOT} std",
              "penalise DISAGREEMENT", K, hot=True)
    s += card(628, 1062, 248, 76, "minimax",
              f's  =  max{sub("", "m", 11)}  r{sub("", "m", 11)}',
              "the worst member decides", K, hot=True)
    # converge on CEM's argsort, each on its own corridor
    s += apoly([(188, 1138), (188, 1184), (316, 1184)], K)
    s += aw(470, 1138, 470, 1156, K)
    s += apoly([(752, 1138), (752, 1184), (624, 1184)], K)
    s += mbox(320, 1160, 300, 48, "CEM:   argsort( s )[ :topk ]", K, size=17,
              fill=ACCENT_FILL[K])
    s += eqcols(1238, [["r", sub("", "m"), "  =  rank of the candidate under member m"],
                      "cvar and minimax are two new branches at objectives.py:95"], 16)
    s += eqline(PX + 26, 1264, "ranks are scale-free, so columns with different links and "
               "different D can vote together; the mean rule cannot mix them", 15)
    s += strip(PX + 26, 1280, [
        ("M = 5 today", "+0.228,  p = 0.0005", False),
        ("rule in use", "median over ranks", False),
        ("what is new", "cvar,  minimax", False),
        ("cost", "M rollouts per step", False)], K, cw=198)
    return b + s, PY + PH + 28


# ------------------------------------------------------------------- V2: gradient
def v2_gradient():
    K = "green"
    b = base("(V2) PiWM-gradplan -- optimise the actions by GRADIENT, not by sampling")
    b += wire_to_pred(K, "dJ / da flows back through g and h")
    b = sch(b)

    PH = 510
    s = panel(PX, PY, PW, PH, K,
              "module:  the actions are LEAVES;  the objective backprops to them")
    yA = 940
    s += txt(56, 910, "leaf, requires_grad", 12.5, anchor="start", fill=ACCENT[K])
    s += pill(104, yA, sub("a", "1:H", 10), K, rx=48)
    s += aw(152, yA, 184, yA, K)
    s += mbox(188, 916, 190, 48, "rollout  ( H steps )", K, size=15,
              sub_="z_k+1 = h( g( z_k , a_k ) )")
    s += aw(378, yA, 392, yA, K)
    s += partpill(434, yA, [("hat", "z", "H")], K, rx=40, fill=WHITE)
    s += aw(476, yA, 500, yA, K)
    s += partbox(504, 916, 230, 48,
                 [BAR, ("hat", "z", "H"), f" {MINUS} z", sub("", "g"), BAR,
                  sup("", "2", 11)], K, size=16)
    s += aw(734, yA, 762, yA, K)
    s += pill(806, yA, "J", K, rx=44, fill=ACCENT_FILL[K])
    # the backward pass gets its own corridor, under the whole forward row
    s += apoly([(806, 959), (806, 1016), (104, 1016), (104, 963)], "crit", w=2.2)
    s += txt(470, 1002, f"backward:    a  :=  a  {MINUS}  {ETA} {DOT} dJ / da", 16,
             fill=ACCENT["crit"])

    # -- inset A: gradient descent
    ix, iy, iw, ih = 60, 1042, 402, 172
    s += inset(ix, iy, iw, ih, "V2:  follow the gradient", K,
               note="1 rollout per step, exact direction")
    bx, byb, byt = ix + 201, 1176, 1104
    pts = [(bx + t * 150.0, byb - (byb - byt) * t * t)
           for t in (-1.0 + 2.0 * k / 40.0 for k in range(41))]
    d = " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    s += f'<path d="M{d}" stroke="{ACCENT[K]}" stroke-width="1.8" fill="none"/>\n'
    prev = None
    for t in (-0.95, -0.66, -0.38, -0.14):
        px, py = bx + t * 150.0, byb - (byb - byt) * t * t
        if prev is not None:
            s += arrow(prev[0], prev[1], px, py, color=ACCENT[K], w=1.5)
        s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="{ACCENT[K]}" '
              f'stroke="{WHITE}" stroke-width="1.1"/>\n')
        prev = (px, py)
    s += txt(bx + 6, 1204, "J ( a )", 13, fill=MUTED)

    # -- inset B: CEM
    jx, jy, jw, jh = 486, 1042, 390, 172
    s += inset(jx, jy, jw, jh, "CEM (today):  sample and rank", "slate",
               note="300 samples per step, ranks only")
    cx, cy = jx + 195, 1146
    s += (f'<circle cx="{cx}" cy="{cy}" r="40" fill="none" stroke="{ACCENT["slate"]}" '
          f'stroke-width="1.4" stroke-dasharray="6,4"/>\n')
    scat = [(-.86, .30), (-.55, -.52), (-.30, .66), (-.05, -.18), (.18, .44),
            (.44, -.62), (.62, .22), (.86, -.10), (-.70, -.10), (.30, -.30),
            (-.18, .10), (.70, .58)]
    elite = {3, 9, 10}
    for i, (fx, fy) in enumerate(scat):
        px, py = cx + fx * 88, cy + fy * 38
        hot = i in elite
        s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{5.2 if hot else 4.0}" '
              f'fill="{ACCENT[K] if hot else WHITE}" stroke="'
              f'{ACCENT[K] if hot else ACCENT["slate"]}" stroke-width="1.4"/>\n')
    s += arrow(cx, cy, cx + 20, cy - 12, color=ACCENT[K], w=1.6)
    s += txt(cx + 4, 1204, f"topk mean {ARROWC} new mu", 13, fill=MUTED)

    s += eqcols(1250, [
        ["J( a )  =  ", BAR, ("hat", "z", "H"), f"( a ) {MINUS} z", sub("", "g"), BAR,
         sup("", "2", 11)],
        ["grad steps replace 300-sample resampling; planning/gd.py already does this"]],
        17)
    s += eqline(PX + 26, 1276, "the rollout carries NO no_grad on its path " + NDASH +
               " the file's three no_grad sites (243, 573, 698) are all off it", 15)
    s += strip(PX + 26, 1290, [
        ("CEM cost per step", "300 rollouts", False),
        ("GD cost per step", "1 rollout + 1 backward", False),
        ("rank rules cannot be used", "borda / median are flat", True)], K, cw=268)
    return b + s, PY + PH + 28


# ------------------------------------------------------------------- V3: two-level
def v3_twolevel():
    K = "amber"
    b = base("(V3) PiWM-twolevel -- search over SUBGOALS, then a short push")
    b += wire_to_pred(K, "z_sub is a latent the predictor must reach")
    b = sch(b)

    PH = 516
    s = panel(PX, PY, PW, PH, K,
              "module:  an upper level over LATENTS, a lower level over ACTIONS")
    yA, yB = 942, 1042
    s += pill(96, 992, sub("z", "t", 11), K, rx=34)
    s += (f'<path d="M130,992 L152,992 M152,942 L152,1042" stroke="{ACCENT[K]}" '
          f'stroke-width="1.8" fill="none" stroke-linecap="round"/>\n')
    s += dot(152, 992, K, r=3.6)
    # -- upper level: search over a latent
    s += aw(152, yA, 196, yA, K)
    s += mbox(200, 918, 180, 48, "UPPER", K, size=17, sub_="CEM over z_sub")
    s += aw(380, yA, 418, yA, K)
    s += pill(470, yA, sub("z", "sub", 10), K, rx=48)
    s += txt(540, yA + 6, "proposes a latent, not an action", 13, anchor="start",
             fill=MUTED)
    # -- lower level: reach it, then push.  z_sub drops on its own corridor.
    s += apoly([(470, 961), (470, 1012)], K)
    s += aw(152, yB, 396, yB, K)
    s += mbox(400, 1016, 164, 52, f"LOWER  {chr(0x3c0)}", K, size=17,
              sub_="reach z_sub, k steps")
    s += aw(564, yB, 600, yB, K)
    s += pill(648, yB, sub("a", "1:k", 10), K, rx=44)
    s += aw(692, yB, 716, yB, K)
    s += mbox(720, 1018, 150, 48, "PUSH", K, size=17, sub_=f"H {MINUS} k free steps")

    # -- inset: flat vs two-level, on one action axis
    ix, iy, iw, ih = 60, 1084, 816, 136
    s += inset(ix, iy, iw, ih, "flat 5-step action search   vs   subgoal + short push", K)
    yF, yT = 1152, 1196
    NRM = [BAR, ("hat", "z", "5"), f" {MINUS} z", sub("", "g"), BAR, sup("", "2", 11)]
    s += txt(100, yF + 5, "flat", 13, anchor="start", fill=MUTED)
    for i in range(5):
        s += acell(180 + i * 46, yF - 14, 38, 28, sub("a", f"{i + 1}", 9), "slate")
    s += arrow(418, yF, 440, yF, color=ACCENT["slate"], w=1.5)
    s += partbox(444, yF - 14, 186, 28, NRM, "slate", size=13)
    s += txt(644, yF + 5, "one score, at H = 5", 13, anchor="start", fill=MUTED)
    s += txt(100, yT + 5, "two-level", 13, anchor="start", fill=MUTED)
    for i in range(2):
        s += acell(180 + i * 46, yT - 14, 38, 28, sub("a", f"{i + 1}", 9), K,
                   fill=ACCENT_FILL[K])
    s += pill(304, yT, sub("z", "sub", 9), K, rx=34, ry=14)
    for i in range(3):
        s += acell(348 + i * 46, yT - 14, 38, 28, sub("a", f"{i + 3}", 9), K,
                   fill=ACCENT_FILL[K])
    s += partbox(490, yT - 14, 186, 28, NRM, K, size=13)
    s += txt(690, yT + 5, "reach score + task score", 13, anchor="start", fill=MUTED)

    s += eqcols(1256, [
        ["upper:   min over z", sub("", "sub"), "  of  ", BAR, ("hat", "z", "H"),
         f" {MINUS} z", sub("", "g"), BAR, sup("", "2", 11)],
        ["lower:   a", sub("", "1:k"), "  =  argmin  ", BAR, ("hat", "z", "k"),
         f" {MINUS} z", sub("", "sub"), BAR, sup("", "2", 11)]], 16)
    s += eqline(PX + 26, 1282, "a subgoal is a LATENT, so it is searchable without the "
               "env; the push is the only part that has to move the block", 15)
    s += strip(PX + 26, 1296, [
        ("block static in", "75% of transitions", True),
        ("top 1% of steps carry", "34.8% of motion", True),
        ("flat search", "H=5, 10-D, 300 samp", False),
        ("two-level", "k=2 reach + 3 push", False)], K, cw=198)
    return b + s, PY + PH + 28


# ---------------------------------------------------------------------- V4: policy
def v4_policy():
    K = "magenta"
    PI = chr(0x3c0)
    b = base("(V4) PiWM-policy -- a goal-conditioned policy; no search at all")
    b += wire_to_pred(K, f"{PI}( z_t , z_g ) replaces the CEM loop")
    b = sch(b)

    PH = 462
    s = panel(PX, PY, PW, PH, K,
              "module:  a policy head on the frozen latents;  the evaluator is unchanged")
    yA = 946
    s += txt(470, 910, "PLAN TIME:  one forward pass per step", 13, fill=MUTED)
    s += pill(92, 922, sub("z", "t", 11), K, rx=32, ry=17)
    s += pill(92, 970, sub("z", "g", 11), K, rx=32, ry=17)
    s += mbox(170, 922, 66, 48, "[ ; ]", K, size=17)
    s += aw(124, 922, 166, 934, K)
    s += aw(124, 970, 166, 958, K)
    s += aw(236, yA, 262, yA, K)
    s += mbox(266, 922, 150, 48, PI, K, size=20, sub_=f"MLP ( 2 {TIMES} 512 )")
    s += aw(416, yA, 442, yA, K)
    s += pill(486, yA, sub("a", "t", 11), K, rx=42, fill=WHITE)
    s += aw(528, yA, 556, yA, K)
    s += mbox(560, 922, 168, 48, "env step", K, size=17, sub_="same evaluator")
    s += txt(816, 952, "1 forward pass", 15, fill=ACCENT[K], weight="bold")
    # what disappears
    s += ghost(200, 1020, 470, 56,
               f"CEM:  300 samples {TIMES} 30 opt steps {TIMES} H rollouts")
    s += txt(790, 1054, "deleted", 16, fill=ACCENT["crit"], weight="bold")
    # training
    s += txt(470, 1100, "TRAINING:  behaviour cloning with hindsight goals, offline",
             13, fill=MUTED)
    yB = 1136
    s += mbox(60, 1112, 176, 48, "offline demos", K, size=16, sub_="18,685 trajectories")
    s += aw(236, yB, 262, yB, K)
    s += mbox(266, 1112, 204, 48, "hindsight goal", K, size=16,
              sub_="z_g = z_t+k ,  k ~ U( 1 , H )")
    s += aw(470, yB, 496, yB, K)
    s += mbox(500, 1112, 220, 48, "L_bc", K, size=18, sub_="regress the demo action")
    s += aw(720, yB, 750, yB, K)
    s += pill(812, yB, PI, K, rx=56, fill=ACCENT_FILL[K])
    s += eqcols(1202, [
        ["a", sub("", "t"), f"  =  {PI}( z", sub("", "t"), " , z", sub("", "g"), " )"],
        [f"L_bc  =  {BAR}{PI}( z", sub("", "t"), " , z", sub("", "g"),
         f" ) {MINUS} a", sub("", "t"), BAR, sup("", "2", 11)]], 17)
    s += eqline(PX + 26, 1228, "same evaluator, same goals, same 50 episodes " + NDASH +
               " the only change is WHO produces the action", 15)
    s += strip(PX + 26, 1242, [
        ("what it removes", f"300 {TIMES} 30 rollouts", False),
        ("what it needs", "1 head, 1 BC loss", False),
        ("the risk", "no test-time search", True)], K, cw=268)
    return b + s, PY + PH + 28


# --------------------------------------------------------------- V5: value-guided
def v5_valueguided():
    K = "crit"
    b = base("(V5) PiWM-value -- a learned V( z , g ) at the leaves of the CEM tree")
    b += wire_to_score(K, "V", "V replaces the MSE")
    b = sch(b)

    PH = 442
    s = panel(PX, PY, PW, PH, K,
              "module:  the CEM tree is unchanged;  only what scores a LEAF changes")
    root = (96, 1026)
    l1 = ((232, 970), (232, 1082))
    leaves = (942, 998, 1054, 1110)
    s += circle(root[0], root[1], 24, sub("z", "t", 11), fill=ACCENT_FILL[K], size=15)
    for cx, cy in l1:
        s += partpill(cx, cy, [("hat", "z", "")], K, rx=21, ry=21, fill=WHITE, size=13)
    s += aw(118, 1012, 208, 978, K, w=1.6)
    s += aw(118, 1040, 208, 1074, K, w=1.6)
    s += txt(160, 976, sub("a", "1", 9), 13, fill=MUTED)
    s += txt(160, 1078, sub("a", "1", 9) + "'", 13, fill=MUTED)
    for k, ly in enumerate(leaves):
        px, py = l1[0] if k < 2 else l1[1]
        s += aw(px + 21, py + (-9 if k % 2 == 0 else 9), 352,
                ly + (6 if k % 2 == 0 else -6), K, w=1.6)
        s += partpill(374, ly, [("hat", "z", "")], K, rx=19, ry=19, fill=WHITE, size=13)
    # the value head replaces the terminal MSE at every leaf
    s += (f'<rect x="412" y="918" width="152" height="218" rx="8" fill="none" '
          f'stroke="{ACCENT[K]}" stroke-width="1.6" stroke-dasharray="7,4"/>\n')
    s += txt(488, 910, "requires T4 (the value head)", 13, fill=ACCENT[K], weight="bold")
    for ly in leaves:
        s += aw(394, ly, 416, ly, K)
        s += mbox(420, ly - 19, 136, 38, "V( z , g )", K, size=15)
        s += aw(556, ly, 596, ly, K)
    s += txt(232, 1156, "step 1", 12, fill=MUTED)
    s += txt(374, 1156, "step H", 12, fill=MUTED)
    s += mbox(600, 922, 150, 196, "argsort( s )", K, size=17,
              sub_=f"elites {ARROWC} new mu")
    s += aw(750, 1020, 786, 1020, K)
    s += pill(842, 1020, "mu", K, rx=48, fill=ACCENT_FILL[K])
    s += eqcols(1182, [
        ["today:    s  =  ", BAR, ("hat", "z", "H"), f" {MINUS} z", sub("", "g"), BAR,
         sup("", "2", 11)],
        ["V5:    s  =  " + MINUS + " V( ", ("hat", "z", "H"), " , z", sub("", "g"), " )"],
        ["cost-to-go PAST the horizon"]], 17, gap=40)
    s += eqline(PX + 26, 1210, "Sutton: search and a learned value are NOT additive "
               + NDASH + " if T4's V is good the CEM gain shrinks, and that is the "
               "prediction", 14)
    s += strip(PX + 26, 1224, [
        ("depends on", "T4 (the V head)", True),
        ("latent vs task dist", "Spearman +0.398", True),
        ("CEM today", f"H = 5,  300 {TIMES} 30", False),
        ("Sutton predicts", "not additive with T4", True)], K, cw=198)
    return b + s, PY + PH + 28


FIGURES = (
    ("arch-v1-pessimism.svg", v1_pessimism),
    ("arch-v2-gradient.svg", v2_gradient),
    ("arch-v3-twolevel.svg", v3_twolevel),
    ("arch-v4-policy.svg", v4_policy),
    ("arch-v5-valueguided.svg", v5_valueguided),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in FIGURES:
        body, h = fn()
        print("  wrote", a.out + "/" + emit(name, body, a.out, 940, h))


if __name__ == "__main__":
    main()
