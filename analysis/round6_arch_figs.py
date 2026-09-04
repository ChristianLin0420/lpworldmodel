"""Architecture diagrams for the six round-6 proposals (R1-R6).

Same construction as analysis/arch_figs_causal.py and analysis/round5_figs_obj.py: the
unmodified LpWM schematic from base() on top, a MODULE PANEL below carrying the proposal as
blocks, operator nodes and arrows, the equation it implements underneath, and a strip of
measured values. No prose paragraphs inside a diagram.

    R6  support   -- optimise S = 1 - J_S, the one logged quantity the screen endorses.
    R2  consist   -- the K-step jump must agree with K chained 1-step steps, off-policy.
    R3  sam       -- sharpness-aware minimisation in the ACTION space.
    R4  incr-eps  -- V1 completed: epsilon is the FLOOR of the per-sample weight.
    R1  ksweep    -- T6's K, swept: one call advances K action rows.
    R5  mcurve    -- the ensemble M-curve, as a DIAGNOSIS of where the ceiling is.

STYLE.  analysis/style.py, via analysis/arch_figs.py.  Sans throughout, hues by IDENTITY:
green = the system, purple = the contrasting condition, amber = the intervention under test,
crimson = failure, slate = neutral.  Panel keys follow that rule rather than the file order --
R6 is green because the screen endorses it, R5 is teal because it is the consensus family
(arch_figs._consensus), R1 is slate because a sweep is a measurement and not yet a claim.

LAYOUT RULES, and the defect each one exists to stop

  * Strict left-to-right dataflow on fixed row baselines.  NO TWO WIRES SHARE A SEGMENT: a
    wire that changes rows does so on its own vertical corridor, and where two signals
    legitimately meet there is a drawn dot().  The one exception is a LOOP -- R3's inner
    ascent and outer descent return right-to-left on their own private corridors, which is
    what makes them read as loops.
  * base()'s wires form a closed loop, so a connector from the panel into the model's
    interior crosses one of them ONCE, at a named point, with a drawn hop (hpoly).  R5's
    route down from the predicted code stack crosses TWO of base()'s wires (the RDMReg
    return at x=726 and the right-hand link wire at x=812) and hops over both; an unhopped
    crossing reads as a junction.
  * Nothing is sized by hand where it can be measured: mbox/pill/rbox/rpill/strip/frow all
    fit their own text and none of them may cross the panel border.
  * Every measured number in a strip is a value this round actually holds.  The only
    arithmetic invented anywhere is R6's worked pair of codes, whose rel_mse and S are
    computed FROM THE DRAWN VECTORS and are labelled as the illustration's own.

Usage:  python analysis/round6_arch_figs.py --out diary/assets/2026-09-04
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, INK, MSERED, MUTED, NAVY, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, BAR, DOT, EPS, IMPLIES, MINUS, NDASH, RHO, SQ, THETA, TIMES,
    apoly, aw, badge, dot, emit, eq, eqfit, eqrow, eqseq, fit_size, hatrun, hpoly,
    mbox, nb, nrm, opnode, pill, seq_w, strip, sup,
)
from analysis.round5_figs_obj import (  # noqa: E402,F401
    cells, frow, inset, mmid, mrun, panel, rbox, rpill, run_w, sgbar, strikeout, tick,
)

W = 940
PX, PW = 34, 872                      # the panel's x-extent, shared by every figure
PIN, PIX = PX + 26, PX + PW - 26      # 60 .. 880 -- the inner margins everything fits
PY = 898                              # the panel's top edge, in base() coordinates
SHIFT = -80                           # base() puts no ink above y=155; reclaim that band

SIGMA_L = "σ"                    # lowercase sigma, registered in arch_figs_causal
MU = "μ"
DELTA = "δ"
LEQ = "≤"
NBSP = "\u00a0"                  # SVG collapses runs of spaces; these survive

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-04")


def lift(body):
    return f'<g transform="translate(0,{SHIFT})">\n{body}</g>\n'


def out_emit(name, body, out, h):
    """emit(), with the schematic band reclaimed. `h` is the height in base coordinates."""
    return emit(name, lift(body), out, w=W, h=h + SHIFT)


# --- the three connector routes into the unmodified schematic ----------------------
# base() fixes every anchor, so each attachment point has ONE route and the reader learns
# it once. Each route names the wires it crosses and hops over every one of them.
def wire_to_mse(key, sym, label):
    """Into the objective, via an operator node in the empty gutter between the two
    right-hand code stacks. Crosses the right-hand link wire once, at (812, 380)."""
    s = hpoly([(880, PY), (880, 380), (766, 380), (766, 347)], key, dash="6,4",
              hops=[(812, 380)])
    s += opnode(766, 330, sym, key, r=15)
    s += apoly([(766, 315), (766, 276)], key, dash="6,4")
    return s + txt(690, 334, label, 15, anchor="end", fill=ACCENT[key])


def wire_to_action(key, label):
    """Into the action circle. Crosses the predictor's output wire once, at (620, 262)."""
    s = hpoly([(880, PY), (880, 126), (620, 126), (620, 430), (358, 430)], key,
              dash="6,4", hops=[(620, 262)])
    return s + txt(870, 152, label, 15, anchor="end", fill=ACCENT[key])


def wire_to_pred(key, label):
    """Into the predictor's flat top. Runs in two empty corridors; crosses nothing."""
    s = apoly([(880, PY), (880, 142), (300, 142), (300, 190)], key, dash="6,4")
    return s + txt(312, 166, label, 15, anchor="start", fill=ACCENT[key])


def wire_from_pred(key, label):
    """OUT of the predicted code stack, down to the panel: for a proposal that reads the
    model and changes no loss. Crosses the RDMReg return at (726, 490) and the right-hand
    link wire at (812, 490), and hops over both."""
    s = hpoly([(712, 341), (712, 490), (880, 490), (880, PY)], key, dash="6,4",
              hops=[(726, 490), (812, 490)])
    # the label owns the vertical corridor, on its LEFT: under the horizontal run it was
    # pinched between the right-hand link box (ends y=464) and the wire at y=490.
    return s + txt(700, 404, label, 14, anchor="end", fill=ACCENT[key], weight="bold")


# --- local primitives -------------------------------------------------------------
def divnode(cx, cy, key, r=16):
    """A division operator node. opnode() carries minus/times/plus only, and R6's whole
    statistic is a ratio, so the obelus is drawn here the same way those are."""
    c = ACCENT[key]
    s = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{c}" '
         f'stroke-width="1.8"/>\n')
    s += (f'<path d="M{cx - 8},{cy} L{cx + 8},{cy}" stroke="{c}" stroke-width="2.2" '
          f'stroke-linecap="round"/>\n')
    for dy in (-5.5, 5.5):
        s += f'<circle cx="{cx}" cy="{cy + dy}" r="1.9" fill="{c}"/>\n'
    return s


def codebar(x, y, vals, key, vmax=1.5, cell=42, gap=6, h=34, label="", hat=False):
    """One code vector as unit SLOTS with a magnitude fill: an empty slot is a unit that
    does not fire, a part-full slot is a unit that fires weakly.

    Support and magnitude are the two things R6 has to separate, so they are two different
    visual channels here -- the slot says WHETHER, the fill says HOW MUCH."""
    s = ""
    for i, v in enumerate(vals):
        cx = x + i * (cell + gap)
        s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{h}" rx="2" fill="{WHITE}" '
              f'stroke="{DIM}" stroke-width="1.1"/>\n')
        if v > 0:
            fh = max(3.0, min(1.0, v / vmax) * h)
            s += (f'<rect x="{cx}" y="{round(y + h - fh, 1)}" width="{cell}" '
                  f'height="{round(fh, 1)}" rx="2" fill="{ACCENT[key]}" '
                  f'fill-opacity="0.82" stroke="{ACCENT[key]}" stroke-width="1.1"/>\n')
    if label:
        lx = x - 14
        s += txt(lx, y + h - 8, label, 16, anchor="end")
        if hat:
            s += caret(lx - text_width(label, 16) / 2 - 1.5, y + h - 8 - 12, w=8, h=5)
    return s


def support_row(x, y, a, b, cell=42, gap=6, h=16, key="blue"):
    """The support agreement itself: filled where BOTH fire, crimson where exactly one
    does. This is the channel S reads and rel_mse does not."""
    s = ""
    for i, (u, v) in enumerate(zip(a, b)):
        cx = x + i * (cell + gap)
        both, one = (u > 0 and v > 0), ((u > 0) != (v > 0))
        f = ACCENT[key] if both else (ACCENT["crit"] if one else WHITE)
        st = ACCENT["crit"] if one else (ACCENT[key] if both else DIM)
        s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{h}" rx="2" fill="{f}" '
              f'fill-opacity="{0.85 if (both or one) else 1}" stroke="{st}" '
              f'stroke-width="1.2"/>\n')
    return s + txt(x - 14, y + h - 3, "support", 12, anchor="end", fill=MUTED)


def axes(x0, x1, y0, y1, key="slate"):
    """A bare L of axes -- recessive furniture, no top/right spine, as style.py asks."""
    c = ACCENT[key]
    return (f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" stroke="{c}" '
            f'stroke-width="1.3" fill="none" stroke-opacity="0.8"/>\n')


def curve(pts, key, w=2.2, dash="", opacity=1.0):
    d = " L".join(f"{round(px, 1)},{round(py, 1)}" for px, py in pts)
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{d}" stroke="{ACCENT[key]}" stroke-width="{w}" fill="none" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'stroke-opacity="{opacity}"{ds}/>\n')


def mark(cx, cy, key, r=5.5, fill=None):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill or ACCENT[key]}" '
            f'stroke="{ACCENT[key]}" stroke-width="1.8"/>\n')


def lin(v, v0, v1, p0, p1):
    return p0 + (v - v0) / (v1 - v0) * (p1 - p0)


def bars(x, y, heights, key, cw=18, gap=6, opacity=0.75):
    """A row of bars on a shared baseline `y`."""
    s = ""
    for i, hh in enumerate(heights):
        s += (f'<rect x="{x + i * (cw + gap)}" y="{round(y - hh, 1)}" width="{cw}" '
              f'height="{round(hh, 1)}" rx="2" fill="{ACCENT[key]}" '
              f'fill-opacity="{opacity}" stroke="{ACCENT[key]}" stroke-width="1"/>\n')
    return s


def ellipsis_v(cx, cy, key, n=3, r=2.6, gap=11):
    """A drawn vertical ellipsis: no installed face has a reliable U+22EE."""
    s = ""
    for k in range(n):
        s += f'<circle cx="{cx}" cy="{cy + k * gap}" r="{r}" fill="{ACCENT[key]}"/>\n'
    return s


# =================================================================== R6: support
def r6_support():
    """THE HEADLINE. rel_mse measures magnitude error; S measures which units fire."""
    K = "blue"
    b = base("(R6) PiWM-support -- optimise the one logged quantity the screen endorses")
    b += wire_to_mse(K, "plus", "S is ADDED to the MSE, never replacing it")

    PH = 600
    s = panel(PX, PY, PW, PH, K,
              "module:  S  =  1 " + MINUS + "  soft Jaccard( " + sub("z", "pred", 12)
              + " , target ) ,  added to " + sub("L", "z", 12))

    # -- row A: the statistic, left to right. Both inputs feed both branches, so the pair
    # is bundled ONCE and fanned out; two separate min/max wire pairs would have put four
    # wires through the same 40px of corridor.
    yA = 1020
    s += pill(96, 996, "z", K, rx=44, ry=18, fill=WHITE, hat="t+1")
    s += pill(96, 1046, sub("z", "t+1", 12), K, rx=44, ry=18, fill=WHITE)
    s += txt(96, 962, "a", 15, fill=ACCENT[K], weight="bold")
    s += txt(96, 1082, "b", 15, fill=ACCENT[K], weight="bold")
    s += mbox(158, 996, 58, 48, "( a , b )", K, size=13)
    s += aw(140, 996, 152, 1008, K)
    s += aw(140, 1046, 152, 1032, K)
    s += mbox(242, 974, 152, 44, "Σ min( a , b )", K, size=14)
    s += mbox(242, 1042, 152, 44, "Σ max( a , b )", K, size=14)
    s += aw(216, 1008, 238, 996, K)
    s += aw(216, 1032, 238, 1064, K)
    s += divnode(432, 1030, K, r=17)
    s += aw(394, 996, 414, 1020, K)
    s += aw(394, 1064, 414, 1040, K)
    s += pill(512, 1030, sub("J", "S", 12), K, rx=32, ry=18)
    s += aw(449, 1030, 476, 1030, K)
    s += mbox(578, 1006, 86, 48, "1 " + MINUS + " " + DOT, K, size=17, sub_="=  S")
    s += aw(544, 1030, 574, 1030, K)
    s += opnode(702, 1030, "times", K)
    s += aw(664, 1030, 682, 1030, K)
    s += txt(702, 992, "support_w", 12.5, fill=ACCENT[K], weight="bold")
    s += pill(806, 1030, sub("L", "support", 12), K, rx=66, ry=19)
    s += aw(718, 1030, 736, 1030, K)

    # -- row B: WHY the two statistics are different on a sparse code. Both pairs carry
    # the SAME squared error; only the second moves the support.
    zt = (0.0, 1.0, 0.0, 0.8, 0.0, 0.0)
    za = (0.0, 1.4, 0.0, 1.1, 0.0, 0.0)                 # same support, magnitudes over
    zb = (0.25, 1.0, 0.25, 0.8, 0.25, 0.25)             # magnitudes right, 4 units leak
    for x0, ttl, note, zp, rel, sval, hot in (
            (60, "same support, magnitudes overshoot", "rel_mse sees it",
             za, "0.152", "0.280", False),
            (478, "magnitudes right, four spurious units fire", "only S sees it",
             zb, "0.152", "0.357", True)):
        s += inset(x0, 1108, 396, 226, ttl, K, note=note)
        bx = x0 + 76
        s += codebar(bx, 1176, zp, K, label="z", hat=True)
        s += codebar(bx, 1224, zt, "slate", label=sub("z", "t+1", 11))
        s += support_row(bx, 1274, zp, zt, key=K)
        s += badge(bx - 4, 1300, "rel_mse" + NBSP * 3 + rel, "slate")
        s += badge(bx + 148, 1300, "S" + NBSP * 3 + sval,
                   "crit" if hot else K)
    # x=600, not the panel centre: at 470 the caption's left half sat on the bottom
    # edge of the "Σ max( a , b )" box, which ends at x=394.
    s += txt(600, 1096, "the SAME squared error, a different support", 13,
             fill=ACCENT[K], weight="bold")

    s += frow(1372, [
        ([sub("J", "S", 12) + "( a , b )  =  Σ min( a , b ) / Σ max( a , b )"],
         INK, "normal"),
        (["S  =  1 " + MINUS + "  ", sub("J", "S", 12), "( ", ("hat", "z", "t+1"),
          " , " + sub("z", "t+1", 12) + " )"], INK, "normal"),
        (["a , b " + "≥" + " 0"], MUTED, "normal", 0.95)])
    s += frow(1398, [
        (["L  =  ", sub("L", "z", 12), "  +  support_w " + DOT + " S"], INK, "bold"),
        ("the post-link code is rectified, so the inputs are non-negative by construction",
         MUTED, "normal", 0.92)])
    s += strip(PIN, 1414, [
        ("screened partial", MINUS + "0.549", False),
        ("monotone over bins", "0.407 / 0.365 / 0.249 / 0.058", False),
        ("evidence", "n = 296 over 36 arms", False),
        ("this quantity", "never optimised before", False)], K, cw=190)
    return b + s, PY + PH + 40


# =============================================================== R2: consistency
def r2_consistency():
    """The K-step jump against K chained 1-step steps, on the planner's own actions."""
    K = "magenta"
    b = base("(R2) PiWM-consist -- the K-step jump must agree with K chained 1-step steps")
    b += wire_to_action(K, sub("a", "1:K", 11) + " ~ the CEM proposal")

    PH = 644
    s = panel(PX, PY, PW, PH, K,
              "module:  BOTH SIDES ARE THE MODEL'S OWN OUTPUT " + NDASH
              + "  no ground truth exists for these actions")

    # -- row A: where the actions come from, and what does NOT exist for them
    s += mbox(60, 976, 206, 48, "randn " + DOT + " " + SIGMA_L + "  +  " + MU, K,
              size=17, sub_="planning/cem.py: the proposal")
    s += aw(266, 1000, 292, 1000, K)
    for k in range(5):
        s += circle(320 + k * 32, 1000, 14, sub("a", str(k + 1), 9),
                    fill=ACCENT_FILL[K], size=12)
    s += txt(384, 1046, MU + " = 0 ,   " + SIGMA_L + " = var_scale = 1", 12.5,
             fill=MUTED)
    s += aw(462, 1000, 496, 1000, "slate", w=1.6, dash="6,4")
    # the label goes ABOVE the circle: under the strikeout it was unreadable
    s += txt(526, 964, sub("o", "t+1", 10), 14, fill=MUTED)
    s += circle(526, 1000, 24, "", fill=WHITE, stroke=DIM)
    s += strikeout(526, 1000, "crit", r=17, w=2.6)
    # two lines: one line of this ran to x=936, past the panel border at x=906
    s += txt(566, 993, "no recorded next frame", 13, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    s += txt(566, 1013, "no simulator ever rolled these actions", 13, anchor="start",
             fill=ACCENT["crit"], weight="bold")

    # -- z_t fans to BOTH branches off one spine, with a drawn junction dot
    s += pill(88, 1155, sub("z", "t", 11), K, rx=30, ry=17)
    s += (f'<path d="M118,1155 L142,1155 M142,1112 L142,1198" stroke="{ACCENT[K]}" '
          f'stroke-width="1.8" fill="none" stroke-linecap="round"/>\n')
    s += dot(142, 1155, K, r=3.8)
    s += aw(142, 1112, 168, 1112, K)
    s += aw(142, 1198, 168, 1198, K)

    # -- row B: the K-step jump, one predictor call on the option action
    s += txt(52, 1078, "the K-step JUMP", 14, anchor="start", fill=ACCENT[K],
             weight="bold")
    s += mbox(172, 1088, 216, 48, sub("P", "K", 12), K, size=20,
              sub_="one call, on the option action")
    s += rpill(560, 1112, [("hat", "z", "jump")], K, rx=54, fill=WHITE)
    s += aw(388, 1112, 504, 1112, K)

    # -- row C: the same horizon, reached by chaining the 1-step map
    # centred UNDER its own row: at x=52 this read as a caption for the inset below it
    s += txt(364, 1242, "K chained 1-step steps", 14, fill=ACCENT[K], weight="bold")
    for k in range(5):
        gx = 172 + k * 82
        s += circle(gx + 28, 1152, 12, sub("a", str(k + 1), 8), fill=ACCENT_FILL[K],
                    size=11)
        s += aw(gx + 28, 1164, gx + 28, 1172, K, w=1.5)
        s += mbox(gx, 1174, 56, 48, sub("P", "1", 11), K, size=17)
        if k < 4:
            s += aw(gx + 56, 1198, gx + 78, 1198, K, w=1.6)
    s += aw(556, 1198, 600, 1198, K)
    s += rpill(662, 1198, [("hat", "z", "chain")], K, rx=56, fill=WHITE)

    # -- the residual, on two private corridors into one operator node
    s += apoly([(614, 1112), (742, 1112), (742, 1139)], K)
    s += apoly([(718, 1198), (742, 1198), (742, 1171)], K)
    s += opnode(742, 1155, "minus", K)
    s += aw(758, 1155, 774, 1155, K)
    s += mbox(778, 1131, 100, 48, nrm("."), K)
    s += txt(828, 1204, sub("L", "consist", 12), 16, fill=ACCENT[K], weight="bold")

    # -- the matched control: one factor changes. Each box carries its own arm name as
    # its caption, rather than a floating label 14px under the inset's title rule.
    s += inset(60, 1266, 816, 108, "the matched control differs by ONE factor", K)
    s += mbox(88, 1312, 250, 46, "randn " + DOT + " " + SIGMA_L + " + " + MU, K,
              size=15, sub_="arm:  consist_src = cem")
    s += mbox(362, 1312, 250, 46, "rows drawn from the batch", "slate", size=15,
              sub_="control:  consist_src = data")
    s += txt(644, 1314, "same term, same shapes,", 12, anchor="start", fill=INK)
    s += txt(644, 1332, "same predictor calls, same RNG,", 12, anchor="start", fill=INK)
    s += txt(644, 1350, "scale-matched by construction", 12, anchor="start", fill=INK)

    s += frow(1412, [
        (["L  =  " + BAR, ("hat", "z", "jump"), " " + MINUS + " ", ("hat", "z", "chain"),
          BAR + SQ + "   =   " + BAR + sub("P", "K", 12) + "( z , " + sub("a", "1:K", 12)
          + " ) " + MINUS + " chain( " + sub("P", "1", 12) + " , z , "
          + sub("a", "1:K", 12) + " )" + BAR + SQ], INK, "bold")])
    s += frow(1438, [
        ("neither side is detached: the claim is AGREEMENT, which is symmetric",
         INK, "normal"),
        ("K = consist_k = the planner's goal_H = 5", MUTED, "normal", 0.95)])
    s += strip(PIN, 1454, [
        ("non-dataset actions in training", "0 arms of ~60", True),
        ("CEM optimistic", "38% of episodes", True),
        ("under oracle dynamics", "6.7%", False)], K, cw=250)
    return b + s, PY + PH + 40


# ======================================================================= R3: SAM
def r3_sam():
    """An ascent step on the ACTION input, then the loss at the perturbed action."""
    K = "amber"
    b = base("(R3) PiWM-sam -- sharpness-aware minimisation in the ACTION space")
    b += wire_to_action(K, "a  +  " + DELTA + "  enters the predictor")

    PH = 646
    s = panel(PX, PY, PW, PH, K,
              "module:  min" + sub("", THETA, 12) + "  max" + sub("", BAR + DELTA + BAR
              + " " + LEQ + " " + RHO, 11) + "  L( z , a + " + DELTA + " ) "
              + NDASH + "  two loops, and a guard on the minimiser")

    # -- INNER LOOP: one ascent step on the action. Runs left to right and returns on its
    # OWN corridor at y=1066, which no other wire touches.
    yA = 1006
    s += pill(84, 976, sub("z", "t", 11), K, rx=28, ry=15, fill=WHITE)
    s += circle(84, 1030, 20, sub("a", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += mbox(140, 982, 88, 48, "P", K, size=20, sub_="predictor")
    s += aw(112, 976, 136, 994, K)
    s += aw(104, 1030, 136, 1018, K)
    s += aw(228, 1006, 252, 1006, K)
    s += mbox(256, 982, 132, 48, "L( z , a )", K, size=17)
    s += aw(388, 1006, 412, 1006, K)
    s += mbox(416, 982, 150, 48, "g  =  dL / da", K, size=16, sub_="autograd.grad")
    s += aw(566, 1006, 590, 1006, K)
    s += mbox(594, 982, 168, 48, RHO + " " + DOT + " g / " + BAR + "g" + BAR, K, size=17,
              sub_="the dual-norm step")
    s += aw(762, 1006, 786, 1006, K)
    s += pill(830, 1006, DELTA, K, rx=30, ry=19)
    s += apoly([(830, 1025), (830, 1066), (84, 1066), (84, 1050)], K, dash="5,4")
    s += txt(452, 1086, "inner loop:  ONE ascent step, on the ACTION " + NDASH
             + "  no weight moves", 13.5, fill=ACCENT[K], weight="bold")

    # -- OUTER LOOP: the descent, at the perturbed action, on its own return corridor
    yB = 1152
    s += pill(88, 1128, sub("z", "t", 11), K, rx=28, ry=15, fill=WHITE)
    s += pill(92, 1178, "a + " + DELTA, K, rx=34, ry=17)
    s += mbox(152, 1128, 88, 48, "P", K, size=20, sub_="same weights")
    s += aw(116, 1128, 148, 1142, K)
    s += aw(126, 1178, 148, 1164, K)
    s += aw(240, 1152, 264, 1152, K)
    s += mbox(268, 1128, 168, 48, "L( z , a + " + DELTA + " )", K, size=17)
    s += aw(436, 1152, 460, 1152, K)
    s += mbox(464, 1128, 150, 48, "dL / d" + THETA, K, size=17, sub_="the real backward")
    s += aw(614, 1152, 638, 1152, K)
    s += mbox(642, 1128, 178, 48, THETA + " step", K, size=18, sub_="the OUTER descent")
    s += apoly([(820, 1176), (846, 1176), (846, 1212), (196, 1212), (196, 1178)], K)
    s += txt(452, 1236, "outer loop:  DESCENT on " + THETA + " at the PERTURBED action",
             13.5, fill=ACCENT[K], weight="bold")

    # -- the guard, in crimson, prominently
    s += inset(60, 1256, 816, 128, "GUARD " + NDASH + " the unconstrained minimiser of "
               "the inner max", "crit")
    # TWO code rows, not one: "the SAME prediction" needs two outputs to compare, and a
    # single row left the claim to the caption alone.
    s += circle(112, 1316, 15, "a", fill=ACCENT_FILL["crit"], size=13)
    s += circle(112, 1352, 15, "a'", fill=WHITE, size=13)
    s += mbox(146, 1310, 56, 52, "P", "crit", size=18)
    s += aw(127, 1316, 142, 1324, "crit", w=1.6)
    s += aw(127, 1352, 142, 1348, "crit", w=1.6)
    s += aw(202, 1322, 232, 1322, "crit", w=1.6)
    s += aw(202, 1350, 232, 1350, "crit", w=1.6)
    s += cells(238, 1313, (1, 1, 0, 1, 0, 0), "crit")
    s += cells(238, 1341, (1, 1, 0, 1, 0, 0), "crit")
    s += txt(298, 1304, "the SAME prediction", 11.5, fill=ACCENT["crit"], weight="bold")
    s += txt(404, 1341, IMPLIES + "   d_action  =  0", 15, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    s += (f'<path d="M580,1296 L580,1372" stroke="{ACCENT["crit"]}" stroke-width="1.2" '
          f'stroke-opacity="0.45"/>\n')
    s += mbox(614, 1312, 234, 48, "d_action", "crit", size=18,
              sub_="in loss_components, every epoch")
    s += txt(731, 1300, "the number to watch is a FALL", 11.5, fill=ACCENT["crit"],
             weight="bold")

    s += frow(1422, [
        (["min" + sub("", THETA, 12) + "   max" + sub("", BAR + DELTA + BAR + " " + LEQ
          + " " + RHO, 11) + "   L( z , a + " + DELTA + " )"], INK, "bold"),
        ([DELTA + "  =  " + RHO + " " + DOT + " g / " + BAR + "g" + BAR + " ,    g  =  "
          "dL / da"], INK, "normal"),
        ([RHO + " is per action ROW"], MUTED, "normal", 0.95)])
    # shortened: at full length these two clauses met in the middle, with frow's gap
    # already collapsed to its 18px floor
    s += frow(1448, [
        ("the ascent pass runs on detached z and target", INK, "normal"),
        ("it shares no buffer with the real backward", INK, "normal"),
        (RHO + " in normalised action units, " + SIGMA_L + " = 1", MUTED,
         "normal", 0.95)])
    s += strip(PIN, 1464, [
        ("grad to actions", "verified live 2.76e-3", False),
        ("dead entries", "zero", False),
        ("guard, logged every epoch", "d_action", True)], K, cw=250)
    return b + s, PY + PH + 40


# ================================================================== R4: epsilon
def r4_eps():
    """V1 completed: epsilon is the FLOOR of the weight, and at 1e-4 it never engages."""
    K = "blue"
    b = base("(R4) PiWM-incr-" + EPS + " -- V1 completed:  " + EPS
             + " is the FLOOR of the per-sample weight")
    b += wire_to_mse(K, "times", "w scales the prediction loss")

    PH = 594
    s = panel(PX, PY, PW, PH, K,
              "module:  w  =  1 / ( " + BAR + "dz" + BAR + SQ + " + " + EPS
              + " ) ,  renormalised to unit mean")

    # -- row A: the weight, exactly V1's construction, with the knob exposed
    s += pill(94, 976, sub("z", "t+1", 12), K, rx=44, ry=18, fill=WHITE)
    s += pill(94, 1024, sub("z", "t", 12), K, rx=44, ry=18, fill=WHITE)
    s += opnode(184, 1000, "minus", K)
    s += aw(138, 976, 168, 992, K)
    s += aw(138, 1024, 168, 1008, K)
    s += aw(200, 1000, 222, 1000, K)
    s += mbox(226, 976, 96, 48, nrm("."), K)
    s += aw(322, 1000, 346, 1000, K)
    s += mbox(350, 976, 162, 48, "1 / ( " + DOT + " + " + EPS + " )", K, size=17,
              sub_="the only new knob")
    s += aw(512, 1000, 536, 1000, K)
    s += mbox(540, 976, 140, 48, "/ mean( " + DOT + " )", K, size=16, sub_="unit mean")
    s += aw(680, 1000, 704, 1000, K)
    s += pill(752, 1000, "w", K, rx=32, ry=19)
    s += aw(784, 1000, 806, 1000, K)
    s += txt(846, 1006, "to " + sub("L", "z", 12), 15, fill=ACCENT[K], weight="bold")

    # -- LEFT inset: the floor itself.
    # The y range runs to 5.4, not 4.3, so the 1/eps floor line for eps = 1e-4 lands well
    # inside the plot: at 4.3 it sat 8px under the frame and ran straight through the
    # "the knee" and "median" annotations. The curve legend is BELOW the axis for the same
    # reason -- inside the plot every position it could take is on a curve.
    s += inset(60, 1054, 470, 264, EPS + " is the FLOOR of the weight", K,
               note="w " + ARROWC + " 1/" + EPS + "  as  " + BAR + "dz" + BAR + SQ
               + " " + ARROWC + " 0")
    X0, X1, Y0, Y1 = 122, 500, 1120, 1230
    LX0, LX1, LY0, LY1 = -5.0, -0.4, 0.0, 5.4
    s += axes(X0, X1, Y0, Y1)
    for lx in range(-5, 0):
        px = lin(lx, LX0, LX1, X0, X1)
        s += (f'<path d="M{round(px,1)},{Y1} L{round(px,1)},{Y1 + 5}" '
              f'stroke="{MUTED}" stroke-width="1.1"/>\n')
        s += txt(px, Y1 + 18, f"1e{lx}", 10.5, fill=MUTED)
    s += txt(X1, Y1 + 40, BAR + "dz" + BAR + SQ + "   (log)", 11.5, anchor="end",
             fill=MUTED)
    s += txt(X0 - 2, Y0 - 8, "w   (log)", 11.5, anchor="start", fill=MUTED)
    for row, (eps_v, kk, lab) in enumerate((
            (1e-4, "crit", EPS + " = 1e-4    floor 1e4"),
            (4.1e-2, K, EPS + " = 4.1e-2    floor 24"))):
        pts = []
        for i in range(61):
            lx = LX0 + (LX1 - LX0) * i / 60.0
            ly = math.log10(1.0 / (10.0 ** lx + eps_v))
            pts.append((lin(lx, LX0, LX1, X0, X1), lin(ly, LY0, LY1, Y1, Y0)))
        fy = lin(math.log10(1.0 / eps_v), LY0, LY1, Y1, Y0)
        s += (f'<path d="M{X0},{round(fy,1)} L{X1},{round(fy,1)}" '
              f'stroke="{ACCENT[kk]}" stroke-width="1.3" stroke-dasharray="5,4" '
              f'stroke-opacity="0.7"/>\n')
        s += curve(pts, kk, w=2.3)
        ly_leg = Y1 + 36 + row * 18
        s += (f'<path d="M{X0},{ly_leg - 4} L{X0 + 22},{ly_leg - 4}" '
              f'stroke="{ACCENT[kk]}" stroke-width="2.3" stroke-linecap="round"/>\n')
        s += txt(X0 + 30, ly_leg, lab, 11.5, anchor="start", fill=ACCENT[kk],
                 weight="bold")
    mx = lin(math.log10(4.1e-2), LX0, LX1, X0, X1)
    s += (f'<path d="M{round(mx,1)},{Y0} L{round(mx,1)},{Y1}" stroke="{MUTED}" '
          f'stroke-width="1.3" stroke-dasharray="4,4"/>\n')
    s += txt(round(mx, 1) - 6, Y0 + 14, "median", 11, anchor="end", fill=MUTED)
    kx = lin(-4.0, LX0, LX1, X0, X1)
    s += (f'<path d="M{round(kx,1)},{Y0} L{round(kx,1)},{Y1}" '
          f'stroke="{ACCENT["crit"]}" stroke-width="1.3" stroke-dasharray="4,4"/>\n')
    s += txt(round(kx, 1) + 6, Y0 + 14, "the knee", 11, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    # "left of every sample" would have been an overclaim: 0.1% of samples do reach the
    # floor. The decade gap is arithmetic on the two numbers already on the axis.
    s += txt(295, 1306, "the knee at 1e-4 sits 2.6 decades below the median: span 3884"
             + TIMES + ", ESS 0.063", 11.5, fill=ACCENT["crit"], weight="bold")

    # -- RIGHT inset: what that does to a batch
    s += inset(548, 1054, 330, 264, "the batch it weights", K)
    b1 = [58, 4, 3, 5, 3, 4, 3, 6, 3, 4, 3, 5]
    b2 = [34, 30, 27, 33, 26, 30, 29, 34, 27, 31, 28, 30]
    s += txt(713, 1128, EPS + " = 1e-4" + NBSP * 6 + "ESS 0.063", 12.5,
             fill=ACCENT["crit"], weight="bold")
    s += bars(572, 1200, b1, "crit")
    s += (f'<path d="M566,1200 L860,1200" stroke="{MUTED}" stroke-width="1.1"/>\n')
    s += txt(713, 1220, "one sample carries the batch", 11, fill=MUTED)
    s += txt(713, 1252, EPS + " = 4.1e-2" + NBSP * 6 + "ESS 0.813", 12.5,
             fill=ACCENT[K], weight="bold")
    s += bars(572, 1300, b2, K)
    s += (f'<path d="M566,1300 L860,1300" stroke="{MUTED}" stroke-width="1.1"/>\n')

    s += frow(1362, [
        (["w  =  1 / ( " + BAR + sub("z", "t+1", 12) + " " + MINUS + " "
          + sub("z", "t", 12) + BAR + SQ + "  +  " + EPS + " )"], INK, "normal"),
        (["w  " + ARROWC + "  w / mean( w )"], INK, "normal"),
        ([sub("L", "z", 12) + "  =  w " + DOT + " " + BAR, ("hat", "z", "t+1"), " "
          + MINUS + " " + sub("z", "t+1", 12) + BAR + SQ], INK, "normal")])
    s += frow(1388, [
        ("the knee of 1/( x + " + EPS + " ) sits at x = " + EPS + ", and w " + LEQ
         + " 1/" + EPS + " is the floor", INK, "bold"),
        ("V1 shipped one value of " + EPS + " and had no way to change it", MUTED,
         "normal", 0.94)])
    s += strip(PIN, 1404, [
        (EPS + " = 1e-4   as shipped, " + MINUS + "0.383", "ESS 0.063 , span 3884"
         + TIMES, True),
        (EPS + " = 1e-3", "ESS 0.242", False),
        (EPS + " = 1e-2", "ESS 0.591", False),
        (EPS + " = 4.1e-2  = the median", "ESS 0.813", False)], K, cw=190)
    return b + s, PY + PH + 40


# =================================================================== R1: K sweep
def r1_ksweep():
    """T6's K, swept. One predictor call advances K action rows; overshoot chains K."""
    K = "slate"
    b = base("(R1) PiWM-jump -- T6's K, swept:  one predictor call advances K action rows")
    b += wire_to_pred(K, "one call spans K action rows")

    PH = 640
    s = panel(PX, PY, PW, PH, K,
              "module:  _option_act  =  the MEAN of the K action embeddings")

    # -- row A: the arm. mean of the K embeddings, then ONE predictor call.
    s += txt(52, 976, "arm:  jump", 14, anchor="start", fill=ACCENT["blue"],
             weight="bold")
    A = "blue"
    for k in range(5):
        s += circle(88 + k * 32, 1000, 13, sub("a", str(k + 1), 8),
                    fill=ACCENT_FILL[A], size=11)
    s += aw(230, 1000, 254, 1000, A)
    s += mbox(258, 976, 124, 48, "mean", A, size=18, sub_="over the K rows")
    s += aw(382, 1000, 406, 1000, A)
    s += pill(478, 1000, "_option_act", A, rx=68, ry=19)
    s += aw(546, 1000, 570, 1000, A)
    s += mbox(574, 976, 118, 48, "P", A, size=20, sub_="ONE call")
    # z_t enters P from directly below. The old route was a 517px horizontal at y=1064
    # that ran the width of the panel and left no clear band for row B's caption.
    s += pill(633, 1062, sub("z", "t", 11), A, rx=28, ry=16, fill=WHITE)
    s += aw(633, 1046, 633, 1026, A)
    s += aw(692, 1000, 716, 1000, A)
    s += rpill(796, 1000, [("hat", "z", "t+K")], A, rx=58, fill=WHITE)

    # -- row B: the matched control. Same horizon, reached by compounding.
    C = "magenta"
    # the caption clears the row: at y=1108 it ran through the a_1 circle at (170, 1104)
    s += txt(52, 1096, "control:  overshoot", 14, anchor="start", fill=ACCENT[C],
             weight="bold")
    s += pill(88, 1162, sub("z", "t", 11), C, rx=28, ry=16, fill=WHITE)
    s += aw(116, 1162, 138, 1162, C)
    heights = (4, 7, 11, 16, 22)
    for k in range(5):
        gx = 142 + k * 86
        s += circle(gx + 28, 1116, 12, sub("a", str(k + 1), 8), fill=ACCENT_FILL[C],
                    size=11)
        s += aw(gx + 28, 1128, gx + 28, 1138, C, w=1.5)
        s += mbox(gx, 1140, 56, 44, sub("P", "1", 11), C, size=17)
        if k < 4:
            s += aw(gx + 56, 1162, gx + 82, 1162, C, w=1.6)
        s += (f'<rect x="{gx + 14}" y="1188" width="28" height="{heights[k]}" rx="2" '
              f'fill="{ACCENT["crit"]}" fill-opacity="0.6" stroke="{ACCENT["crit"]}" '
              f'stroke-width="1"/>\n')
    s += aw(568, 1162, 596, 1162, C)
    s += rpill(660, 1162, [("hat", "z", "t+K")], C, rx=58, fill=WHITE)
    s += txt(556, 1222, "error compounds", 12, fill=ACCENT["crit"], weight="bold")

    # -- K as the swept axis
    s += inset(60, 1244, 816, 124, "K is the swept axis", K,
               note="one predictor call, K action rows")
    AX0, AX1, AY = 132, 812, 1332
    s += (f'<path d="M{AX0},{AY} L{AX1},{AY}" stroke="{ACCENT[K]}" stroke-width="1.6" '
          f'marker-end="url(#a)"/>\n')
    s += txt(838, AY + 5, "K", 16, weight="bold")
    for kk, lab, new in ((1, "1", None), (2, "2", True), (3, "3", True),
                         (5, "5", False), (8, "8", True)):
        px = lin(math.log(kk), 0.0, math.log(8.0), AX0 + 22, AX1 - 40)
        if new is None:
            s += mark(px, AY, "slate", r=8, fill=WHITE)
            s += txt(px, AY - 22, "today", 11.5, fill=MUTED)
        elif new:
            s += mark(px, AY, "blue", r=9)
            s += txt(px, AY - 22, "new", 11.5, fill=ACCENT["blue"], weight="bold")
        else:
            s += mark(px, AY, "blue", r=9, fill=WHITE)
            s += txt(px, AY - 22, "exists", 11.5, fill=ACCENT["blue"], weight="bold")
        s += txt(px, AY + 26, lab, 15, weight="bold")

    s += frow(1406, [
        (["_option_act  =  ( 1 / K ) Σ" + sub("", "k", 12) + "  E( "
          + sub("a", "k", 12) + " )"], INK, "normal"),
        ([("hat", "z", "t+K"), "  =  P( " + sub("z", "t", 12)
          + " ; _option_act )"], INK, "normal"),
        (["overshoot:  chain( P , K )"], INK, "normal", 0.95)])
    s += frow(1432, [
        ("NOT a concat: fan_in 10 " + ARROWC + " 50 moves action_encoder's muP LR by 5"
         + TIMES, ACCENT["crit"], "bold"),
        ("NOT a sum: the AdaLN conditioning norm would scale by K", ACCENT["crit"],
         "bold")])
    s += strip(PIN, 1448, [
        ("K swept", "2 , 3 , 8   (K = 5 exists)", False),
        ("block moves 0.000 px", "48.1% of 1-row steps", True),
        ("over 5 rows", "12.7%", False),
        ("median motion", "0.084 px " + ARROWC + " 24.6 px", False)], K, cw=190)
    return b + s, PY + PH + 40


# =================================================================== R5: M-curve
def r5_mcurve():
    """The ensemble M-curve. A DIAGNOSIS: the curve's shape says where the ceiling is."""
    K = "green"
    b = base("(R5) PiWM-vote M-curve -- a DIAGNOSIS: are the members' errors correlated?")
    b += wire_from_pred(K, "eval only " + NDASH + " no term in the loss")

    PH = 668
    s = panel(PX, PY, PW, PH, K,
              "module:  M independently trained models, per-member rank, borda "
              + ARROWC + " CEM's argsort( loss )[ : topk ]")

    # -- row A: M members, each in its OWN latent basis, each with its own corridor into
    # the vote. The old consensus figure put all members on one shared segment.
    BX, BY, BW, BH = 552, 972, 170, 142
    cy = BY + BH / 2
    for i, y in enumerate((992, 1048, 1104)):
        s += mbox(64, y - 22, 132, 44, "model " + str(i + 1), K, size=15,
                  sub_="its own encoder")
        s += aw(196, y, 218, y, K)
        s += mbox(222, y - 22, 112, 44, sub("MSE", str(i + 1), 11), K, size=15)
        s += aw(334, y, 356, y, K)
        s += mbox(360, y - 22, 148, 44, "rank", K, size=15,
                  sub_="within the CEM batch")
        # each member gets its OWN corridor and its OWN entry point on the vote box
        cor, ey = (524, 534, 544)[i], (BY + 30, cy, BY + BH - 30)[i]
        s += apoly([(508, y), (cor, y), (cor, ey), (BX - 6, ey)], K)
    s += ellipsis_v(138, 1140, "slate")
    s += txt(172, 1174, "M " + "∈" + " { 2 , 3 , 5 , 8 , 12 }   independently "
             "trained models", 14, anchor="start", fill=MUTED)
    s += (f'<rect x="{BX}" y="{BY}" width="{BW}" height="{BH}" rx="9" '
          f'fill="{ACCENT_FILL[K]}" stroke="{ACCENT[K]}" stroke-width="2.0"/>\n')
    s += txt(BX + BW / 2, BY + 34, "BORDA", 19, fill=ACCENT[K], weight="bold")
    s += txt(BX + BW / 2, BY + 58, "mean over", 14, fill=ACCENT[K])
    s += txt(BX + BW / 2, BY + 78, "per-member ranks", 14, fill=ACCENT[K])
    s += txt(BX + BW / 2, BY + 104, "at PLAN time", 12.5, fill=MUTED)
    s += txt(BX + BW / 2, BY + 124, "no term in the loss", 12.5, fill=MUTED)
    # CEM sits to the RIGHT of the vote, so the last hop is left to right like every
    # other one: the old route hooked back from x=846 and read as a return path.
    s += aw(BX + BW, cy, 744, cy, K, w=2.0)
    s += pill(792, cy, "CEM", K, rx=46, ry=22)
    s += txt(792, cy + 44, "argsort( loss )[ : topk ]", 12, fill=MUTED)

    # -- what the curve's SHAPE decides
    s += inset(60, 1210, 816, 216, "what the curve's SHAPE decides", K,
               note="solid = measured;  dashed = the two hypotheses")
    PX0, PX1, PY0, PY1 = 132, 600, 1258, 1382
    s += axes(PX0, PX1, PY0, PY1)
    s += txt(PX0 - 2, PY0 - 10, "CEM success", 12, anchor="start", fill=MUTED)

    def mx(m):
        return lin(m - 1, 0, 11, PX0 + 12, PX1 - 12)

    def my(v):
        # the floor is 0.20, not 0.30: at 0.30 the single-model point sat 16px above the
        # axis and its label had nowhere to go that was not on the axis itself.
        return lin(v, 0.20, 0.76, PY1, PY0 + 8)

    for m, lab in ((1, "1"), (2, "2"), (3, "3"), (5, "5"), (8, "8"), (12, "12")):
        s += (f'<path d="M{round(mx(m),1)},{PY1} L{round(mx(m),1)},{PY1 + 5}" '
              f'stroke="{MUTED}" stroke-width="1.1"/>\n')
        s += txt(mx(m), PY1 + 20, lab, 11.5, fill=MUTED)
    s += txt((PX0 + PX1) / 2, PY1 + 40, "M   (members)", 12, fill=MUTED)
    measured = [(1, 0.357), (3, 0.517), (5, 0.608)]
    s += curve([(mx(m), my(v)) for m, v in measured], K, w=2.4)
    s += curve([(mx(5), my(0.608)), (mx(8), my(0.700)), (mx(12), my(0.745))],
               K, w=2.2, dash="7,5")
    s += curve([(mx(5), my(0.608)), (mx(8), my(0.615)), (mx(12), my(0.618))],
               "crit", w=2.2, dash="7,5")
    for m, v in measured:
        s += mark(mx(m), my(v), K, r=5.5)
    for m, v in ((8, 0.700), (12, 0.745)):
        s += mark(mx(m), my(v), K, r=5.0, fill=WHITE)
    for m, v in ((8, 0.615), (12, 0.618)):
        s += mark(mx(m), my(v), "crit", r=5.0, fill=WHITE)
    s += txt(mx(1) + 8, my(0.357) + 20, "0.357", 11.5, anchor="start", fill=INK,
             weight="bold")
    s += txt(mx(3) - 6, my(0.517) - 14, "0.517", 11.5, anchor="end", fill=INK,
             weight="bold")
    s += txt(mx(5) + 12, my(0.608) + 24, "0.608", 11.5, anchor="start", fill=INK,
             weight="bold")
    # the two blocks are 70px apart on their own baselines: at my(0.700) and my(0.560)
    # the third line of one landed on the first line of the other.
    for y0, kk, head, l1, l2 in (
            (1276, K, "still rising", IMPLIES + "  independent errors,",
             "a real single-model ceiling"),
            (1342, "crit", "saturates early", IMPLIES + "  correlated errors,",
             "a common-mode defect")):
        s += txt(620, y0, head, 13, anchor="start", fill=ACCENT[kk], weight="bold")
        s += txt(620, y0 + 19, l1, 12.5, anchor="start", fill=ACCENT[kk])
        s += txt(620, y0 + 37, l2, 12.5, anchor="start", fill=ACCENT[kk])

    s += frow(1462, [
        (["rank" + sub("", "m", 12) + "( c )  =  the candidate's rank within the CEM "
          "batch, under member m"], INK, "normal"),
        (["borda( c )  =  mean" + sub("", "m", 12) + " rank" + sub("", "m", 12)
          + "( c )"], INK, "normal")])
    s += frow(1488, [
        ("M complete models, each in its OWN latent basis " + NDASH
         + " averaging their z would be meaningless", INK, "normal"),
        ("nothing here enters a training loss", MUTED, "normal", 0.95)])
    s += strip(PIN, 1504, [
        ("M swept", "2 , 3 , 5 , 8 , 12", False),
        ("vote5-borda", "0.608", False),
        ("vote3-median", "0.517", False),
        ("single model", "0.357", False),
        ("touches training", "eval only", False)], K, cw=150)
    return b + s, PY + PH + 40


FIGURES = (
    ("arch-r1-ksweep.svg", r1_ksweep),
    ("arch-r2-consistency.svg", r2_consistency),
    ("arch-r3-sam.svg", r3_sam),
    ("arch-r4-eps.svg", r4_eps),
    ("arch-r5-mcurve.svg", r5_mcurve),
    ("arch-r6-support.svg", r6_support),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in FIGURES:
        body, h = fn()
        print("  wrote", os.path.join(a.out, out_emit(name, body, a.out, h)))


if __name__ == "__main__":
    main()
