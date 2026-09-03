"""Architecture diagrams for round 5 (V1-V5, the PLANNER proposals), in the LpWM style.

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py:
the unmodified LpWM schematic from `base()` on top, and a MODULE PANEL below carrying the
proposed change as blocks, operator nodes and arrows, with the equation it implements and
a strip of measured values. No prose inside a diagram.

These five differ from rounds 4 and 5's model proposals in one way: nothing here changes
the trained model. V1, V2, V3 and V5 change what the PLANNER does with a model that is
already trained; V4 deletes the planner. So the dashed connector into the schematic lands
on the objective (the MSE gutter between the two code stacks) or on the predictor's flat
top edge -- never on the a_t circle, which is fenced in by the link/encoder/RDMReg wiring.

Layout rule that keeps the panels readable: every panel is a strict left-to-right dataflow
on fixed row baselines, and any wire that must change rows does so on its own vertical
corridor. No two wires share a segment.

Usage:  python analysis/round5_figs_planners.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, INK, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402
    aw, apoly, eq, mbox, nrm, opnode, pill, strip, sup,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-03")


# --- primitives copied from analysis/arch_figs_predictor.py -----------------------
def panel(x, y, w, h, key, title):
    """Accent-bordered module panel with a title rule. Explicit geometry so each figure
    can size its own flow area."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.4"/>\n')
    s += txt(x + 20, y + 32, title, 18, anchor="start", fill=c, weight="bold")
    s += (f'<path d="M{x + 18},{y + 44} L{x + w - 18},{y + 44}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def blockdiag(x, y, n=6, cell=17, key="green"):
    """Inset: an n x n grid with 2x2 blocks filled on the diagonal -- what R(.) is."""
    c = ACCENT[key]
    s = ""
    for i in range(n):
        for j in range(n):
            on = (i // 2) == (j // 2)
            s += (f'<rect x="{x + j * cell}" y="{y + i * cell}" width="{cell}" '
                  f'height="{cell}" fill="{ACCENT_FILL[key] if on else WHITE}" '
                  f'stroke="{c if on else "#cfcfca"}" stroke-width="{1.4 if on else 0.8}"/>\n')
    return s


# --- a few more, local to this round ----------------------------------------------
def hook(key, label, dy=0):
    """The dashed connector from the module panel up into the objective the planner
    minimises: the empty gutter BETWEEN the two code stacks (x 723..801), entered from
    BELOW their bottom edge (y 337) so it clears the MSE arrow and its label."""
    s = mbox(732, 312, 66, 36, label, key, size=15)
    s += apoly([(880, 898), (880, 366), (766, 366), (766, 352)], key, w=2.0, dash="6,4")
    s += apoly([(766, 312), (766, 278)], key, w=2.0, dash="6,4")
    return s


def topedge(key, label):
    """The other legal entry point: the predictor's flat top edge, via the two empty
    corridors x=880 and y=120."""
    s = apoly([(880, 898), (880, 120), (300, 120), (300, 190)], key, w=2.0, dash="6,4")
    s += txt(310, 106, label, 16, anchor="start", fill=ACCENT[key])
    return s


def card(x, y, w, h, title, formula, note, key, hot=False):
    """One of several alternatives shown side by side."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="{2.6 if hot else 1.8}"/>\n')
    s += txt(x + w / 2, y + 27, title, 18, fill=c, weight="bold")
    s += txt(x + w / 2, y + 52, formula, 15)
    s += txt(x + w / 2, y + 70, note, 12, fill="#555")
    return s


def inset(x, y, w, h, title, key, note=""):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="2"/>\n')
    s += txt(x + w / 2, y + 26, title, 17, fill=c, weight="bold")
    if note:
        s += txt(x + w / 2, y + 45, note, 12, fill="#555")
    return s


def acell(x, y, w, h, label, key, fill=WHITE, size=12):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
         f'stroke="{c}" stroke-width="1.6"/>\n')
    return s + txt(x + w / 2, y + h / 2 + 4, label, size)


def ghost(x, y, w, h, label, size=15):
    """A block the proposal DELETES: grey, dashed, struck through."""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#f2f2ef" '
         f'stroke="#9a9a94" stroke-width="1.8" stroke-dasharray="7,4"/>\n')
    s += txt(x + w / 2, y + h / 2 + 5, label, size, fill="#77776f")
    c = ACCENT["crit"]
    s += (f'<path d="M{x+22},{y+h/2} L{x+w-22},{y+h/2}" stroke="{c}" '
          f'stroke-width="2.4" stroke-linecap="round" stroke-opacity="0.8"/>\n')
    return s


def vdots(cx, cy, key="slate", n=3, gap=6, r=1.9):
    c = ACCENT[key]
    return "".join(f'<circle cx="{cx}" cy="{cy + i*gap}" r="{r}" fill="{c}"/>\n'
                   for i in range(n))


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# ------------------------------------------------------------------ V1: pessimism
def v1_pessimism():
    K = "green"
    b = base("(V1) PiWM-pessimism -- combine the M members' RANKS, pessimistically")
    b += hook(K, "vote")
    b += txt(864, 386, "M members vote", 14, anchor="end", fill=ACCENT[K],
             weight="bold")

    PY, PH = 898, 496
    s = panel(PX, PY, PW, PH, K,
              "module:  M member rollouts  ->  per-member ranks  ->  one score CEM sorts on")

    ys = (972, 1018, 1064)
    tags = ("1", "2", "M")
    s += pill(92, 1018, sub("a", "1:H", 10), K, rx=44)
    s += (f'<path d="M136,1018 L156,1018 M156,972 L156,1064" stroke="{ACCENT[K]}" '
          f'stroke-width="2" fill="none"/>\n')
    s += vdots(168, 1036, key=K, gap=5, r=1.7)
    for y, tg in zip(ys, tags):
        s += aw(156, y, 176, y, K)
        s += mbox(180, y - 20, 104, 40, sub("WM", tg, 11), K, size=16)
        s += aw(284, y, 306, y, K)
        s += mbox(310, y - 20, 180, 40, nrm("zhat_H - z_g"), K, size=12)
        s += aw(490, y, 512, y, K)
        s += pill(542, y, sub("L", tg, 11), K, rx=30, ry=17)
        s += aw(572, y, 594, y, K)
        s += mbox(598, y - 20, 108, 40, sub("rank", tg, 11), K, size=15)
    # three private corridors into the rank bus
    s += mbox(752, 994, 132, 48, "r_1  ...  r_M", K, size=15)
    s += apoly([(706, 972), (728, 972), (728, 1002), (748, 1002)], K)
    s += aw(706, 1018, 748, 1018, K)
    s += apoly([(706, 1064), (728, 1064), (728, 1034), (748, 1034)], K)
    # rank spine feeding the three combination rules
    s += (f'<path d="M818,1042 L818,1098 M188,1098 L818,1098" stroke="{ACCENT[K]}" '
          f'stroke-width="2" fill="none"/>\n')
    for cx in (188, 470, 752):
        s += aw(cx, 1098, cx, 1108, K)
    s += card(64, 1112, 248, 76, "borda", "s  =  mean_m  r_m",
              "already implemented (rule = borda)", K)
    s += card(346, 1112, 248, 76, "cvar", "s  =  mean  +  lam . std",
              "penalise DISAGREEMENT", K, hot=True)
    s += card(628, 1112, 248, 76, "minimax", "s  =  max_m  r_m",
              "the worst member decides", K, hot=True)
    # converge on CEM's argsort
    s += apoly([(188, 1188), (188, 1234), (316, 1234)], K)
    s += aw(470, 1188, 470, 1206, K)
    s += apoly([(752, 1188), (752, 1234), (624, 1234)], K)
    s += mbox(320, 1210, 300, 48, "CEM:   argsort( s )[ :topk ]", K, size=17,
              fill=ACCENT_FILL[K])
    s += eq(PX + 26, 1288, "r_m  =  rank of the candidate under member m", 16)
    s += eq(PX + 360, 1288,
            "cvar and minimax are two new branches at objectives.py:95", 15)
    s += eq(PX + 26, 1314, "ranks are scale-free, so columns with different links and "
            "different D can vote together; the mean rule cannot mix them", 15)
    s += strip(PX + 26, 1330, [
        ("M = 5 today", "+0.228,  p = 0.0005", False),
        ("rule in use", "median over ranks", False),
        ("what is new", "cvar,  minimax", False),
        ("cost", "M rollouts per step", False)], K, cw=198)
    return b + s


# ------------------------------------------------------------------- V2: gradient
def v2_gradient():
    K = "blue"
    b = base("(V2) PiWM-gradplan -- optimise the actions by GRADIENT, not by sampling")
    b += topedge(K, "dJ / da flows back through g and h")

    PY, PH = 898, 496
    s = panel(PX, PY, PW, PH, K,
              "module:  the actions are LEAVES;  the objective backprops to them")
    yA = 990
    s += pill(104, yA, sub("a", "1:H", 10), K, rx=48)
    s += txt(170, 1048, "leaf, requires_grad", 12, anchor="start", fill=ACCENT[K])
    s += aw(152, yA, 184, yA, K)
    s += mbox(188, 966, 176, 48, "rollout  ( H steps )", K, size=15,
              sub_="z_k+1 = h( g( z_k , a_k ) )")
    s += aw(364, yA, 392, yA, K)
    s += pill(432, yA, "zhat_H", K, rx=40, fill=WHITE)
    s += aw(472, yA, 500, yA, K)
    s += mbox(504, 966, 230, 48, nrm("zhat_H - z_g"), K, size=14)
    s += aw(734, yA, 762, yA, K)
    s += pill(806, yA, "J", K, rx=44, fill=ACCENT_FILL[K])
    # the backward pass: its own corridor, under everything
    s += apoly([(806, 1009), (806, 1062), (104, 1062), (104, 1013)], "crit", w=2.4)
    s += txt(470, 1052, "backward:    a  :=  a  -  eta . dJ / da", 16,
             fill=ACCENT["crit"])

    # -- inset A: gradient descent
    ix, iy, iw, ih = 60, 1096, 402, 156
    s += inset(ix, iy, iw, ih, "V2:  follow the gradient", K,
               note="1 rollout per step, exact direction")
    bx, byb, byt = ix + 201, iy + 130, iy + 58
    pts = []
    for k in range(41):
        t = -1.0 + 2.0 * k / 40.0
        pts.append((bx + t * 150.0, byb - (byb - byt) * t * t))
    d = " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    s += f'<path d="M{d}" stroke="{ACCENT[K]}" stroke-width="2" fill="none"/>\n'
    prev = None
    for t in (-0.95, -0.66, -0.38, -0.14):
        px, py = bx + t * 150.0, byb - (byb - byt) * t * t
        if prev is not None:
            s += arrow(prev[0], prev[1], px, py, color=ACCENT[K], w=1.6)
        s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{ACCENT[K]}" '
              f'stroke="{INK}" stroke-width="1.1"/>\n')
        prev = (px, py)
    s += txt(bx + 6, byb + 20, "J ( a )", 13, fill="#555")

    # -- inset B: CEM
    jx, jy, jw, jh = 486, 1096, 390, 156
    s += inset(jx, jy, jw, jh, "CEM (today):  sample and rank", "slate",
               note="300 samples per step, ranks only")
    cx, cy = jx + 195, jy + 104
    s += (f'<circle cx="{cx}" cy="{cy}" r="40" fill="none" stroke="{ACCENT["slate"]}" '
          f'stroke-width="1.6" stroke-dasharray="6,4"/>\n')
    scat = [(-.86, .30), (-.55, -.52), (-.30, .66), (-.05, -.18), (.18, .44),
            (.44, -.62), (.62, .22), (.86, -.10), (-.70, -.10), (.30, -.30),
            (-.18, .10), (.70, .58)]
    elite = {3, 9, 10}
    for i, (fx, fy) in enumerate(scat):
        px, py = cx + fx * 88, cy + fy * 38
        hot = i in elite
        s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{5.5 if hot else 4.2}" '
              f'fill="{ACCENT[K] if hot else WHITE}" stroke="'
              f'{ACCENT[K] if hot else ACCENT["slate"]}" stroke-width="1.6"/>\n')
    s += arrow(cx, cy, cx + 20, cy - 12, color=ACCENT[K], w=1.8)
    s += txt(cx + 4, cy + 50, "topk mean  ->  new mu", 13, fill="#555")

    s += eq(PX + 26, 1288, "J( a )  =  " + nrm("zhat_H( a ) - z_g"), 17)
    s += eq(PX + 330, 1288, "grad steps replace 300-sample resampling; planning/gd.py "
            "already does this", 15)
    s += eq(PX + 26, 1314, "the rollout carries NO no_grad on its path -- the file's "
            "three no_grad sites (243, 573, 698) are all off it", 15)
    s += strip(PX + 26, 1330, [
        ("CEM cost per step", "300 rollouts", False),
        ("GD cost per step", "1 rollout + 1 backward", False),
        ("rank rules cannot be used", "borda / median are flat", True)], K, cw=268)
    return b + s


# ------------------------------------------------------------------- V3: two-level
def v3_twolevel():
    K = "amber"
    b = base("(V3) PiWM-twolevel -- search over SUBGOALS, then a short push")
    b += topedge(K, "z_sub is a latent the predictor must reach")

    PY, PH = 898, 496
    s = panel(PX, PY, PW, PH, K,
              "module:  an upper level over LATENTS, a lower level over ACTIONS")
    yA, yB = 986, 1086
    s += pill(96, 1036, sub("z", "t", 11), K, rx=34)
    s += (f'<path d="M130,1036 L152,1036 M152,986 L152,1086" stroke="{ACCENT[K]}" '
          f'stroke-width="2" fill="none"/>\n')
    # -- upper level
    s += aw(152, yA, 176, yA, K)
    s += mbox(180, 962, 156, 48, "UPPER", K, size=17, sub_="CEM over z_sub")
    s += aw(336, yA, 362, yA, K)
    s += pill(410, yA, sub("z", "sub", 10), K, rx=46)
    s += txt(258, 950, "proposes a latent, not an action", 12, fill="#666")
    # -- lower level
    s += aw(152, yB, 448, yB, K)
    s += apoly([(410, 1005), (410, 1046), (534, 1046), (534, 1058)], K)
    s += mbox(452, 1060, 164, 52, "LOWER  pi", K, size=17, sub_="reach z_sub, k steps")
    s += aw(616, yB, 640, yB, K)
    s += pill(684, yB, sub("a", "1:k", 10), K, rx=42)
    s += aw(726, yB, 750, yB, K)
    s += mbox(754, 1062, 122, 48, "PUSH", K, size=17, sub_="H - k free steps")

    # -- inset: flat vs two-level
    ix, iy, iw, ih = 60, 1132, 816, 134
    s += inset(ix, iy, iw, ih, "flat 5-step action search   vs   subgoal + short push", K)
    yF, yT = 1194, 1240
    s += txt(104, yF + 5, "flat", 13, anchor="start", fill="#555")
    for i in range(5):
        s += acell(190 + i * 46, yF - 14, 38, 28, sub("a", f"{i+1}", 9), "slate")
    s += arrow(412, yF, 436, yF, color=ACCENT["slate"], w=1.6)
    s += mbox(440, yF - 14, 186, 28, nrm("zhat_5 - z_g"), "slate", size=12)
    s += txt(640, yF + 5, "one score, at H = 5", 13, anchor="start", fill="#555")
    s += txt(104, yT + 5, "two-level", 13, anchor="start", fill="#555")
    for i in range(2):
        s += acell(190 + i * 46, yT - 14, 38, 28, sub("a", f"{i+1}", 9), K,
                   fill=ACCENT_FILL[K])
    s += pill(312, yT, sub("z", "sub", 9), K, rx=34, ry=14)
    for i in range(3):
        s += acell(352 + i * 46, yT - 14, 38, 28, sub("a", f"{i+3}", 9), K,
                   fill=ACCENT_FILL[K])
    s += mbox(494, yT - 14, 186, 28, nrm("zhat_5 - z_g"), K, size=12)
    s += txt(694, yT + 5, "reach score + task score", 13, anchor="start", fill="#555")

    s += eq(PX + 26, 1298, "upper:   min over z_sub of  " + nrm("zhat_H - z_g"), 16)
    s += eq(PX + 400, 1298, "lower:   a_1:k  =  argmin  " + nrm("zhat_k - z_sub"), 16)
    s += eq(PX + 26, 1324, "a subgoal is a LATENT, so it is searchable without the env; "
            "the push is the only part that has to move the block", 15)
    s += strip(PX + 26, 1338, [
        ("block static in", "75% of transitions", True),
        ("top 1% of steps carry", "34.8% of motion", True),
        ("flat search", "H=5, 10-D, 300 samp", False),
        ("two-level", "k=2 reach + 3 push", False)], K, cw=198)
    return b + s


# ---------------------------------------------------------------------- V4: policy
def v4_policy():
    K = "magenta"
    b = base("(V4) PiWM-policy -- a goal-conditioned policy; no search at all")
    b += topedge(K, "pi( z_t , z_g ) replaces the CEM loop")

    PY, PH = 898, 460
    s = panel(PX, PY, PW, PH, K,
              "module:  a policy head on the frozen latents;  the evaluator is unchanged")
    yA = 996
    s += txt(470, 960, "PLAN TIME:  one forward pass per step", 13, fill="#555")
    s += pill(92, 972, sub("z", "t", 11), K, rx=32, ry=17)
    s += pill(92, 1020, sub("z", "g", 11), K, rx=32, ry=17)
    s += mbox(170, 972, 66, 48, "[ ; ]", K, size=17)
    s += aw(124, 972, 166, 984, K)
    s += aw(124, 1020, 166, 1008, K)
    s += aw(236, yA, 262, yA, K)
    s += mbox(266, 972, 150, 48, "pi", K, size=20, sub_="MLP ( 2 x 512 )")
    s += aw(416, yA, 442, yA, K)
    s += pill(486, yA, sub("a", "t", 11), K, rx=42, fill=WHITE)
    s += aw(528, yA, 556, yA, K)
    s += mbox(560, 972, 168, 48, "env step", K, size=17, sub_="same evaluator")
    s += txt(802, 1002, "1 forward pass", 15, fill=ACCENT[K], weight="bold")
    # what disappears
    s += ghost(200, 1070, 470, 56, "CEM:  300 samples x 30 opt steps x H rollouts")
    s += txt(790, 1104, "deleted", 16, fill=ACCENT["crit"], weight="bold")
    # training
    s += txt(470, 1150, "TRAINING:  behaviour cloning with hindsight goals, offline",
             13, fill="#555")
    yB = 1186
    s += mbox(60, 1162, 176, 48, "offline demos", K, size=16, sub_="18,685 trajectories")
    s += aw(236, yB, 262, yB, K)
    s += mbox(266, 1162, 204, 48, "hindsight goal", K, size=16,
              sub_="z_g = z_t+k ,  k ~ U( 1 , H )")
    s += aw(470, yB, 496, yB, K)
    s += mbox(500, 1162, 220, 48, "L_bc", K, size=18, sub_="regress the demo action")
    s += aw(720, yB, 750, yB, K)
    s += pill(812, yB, "pi", K, rx=56, fill=ACCENT_FILL[K])
    s += eq(PX + 26, 1252, "a_t  =  pi( z_t , z_g )", 17)
    s += eq(PX + 250, 1252, "L_bc  =  " + nrm("pi( z_t , z_g ) - a_t"), 17)
    s += eq(PX + 26, 1278, "same evaluator, same goals, same 50 episodes -- the only "
            "change is WHO produces the action", 15)
    s += strip(PX + 26, 1292, [
        ("what it removes", "300 x 30 rollouts", False),
        ("what it needs", "1 head, 1 BC loss", False),
        ("the risk", "no test-time search", True)], K, cw=268)
    return b + s


# --------------------------------------------------------------- V5: value-guided
def v5_valueguided():
    K = "crit"
    b = base("(V5) PiWM-value -- a learned V( z , g ) at the leaves of the CEM tree")
    b += hook(K, "V")
    b += txt(864, 386, "V replaces the MSE", 14, anchor="end", fill=ACCENT[K],
             weight="bold")

    PY, PH = 898, 440
    s = panel(PX, PY, PW, PH, K,
              "module:  the CEM tree is unchanged;  only what scores a LEAF changes")
    root = (96, 1076)
    l1 = ((232, 1020), (232, 1132))
    leaves = (992, 1048, 1104, 1160)
    s += circle(root[0], root[1], 24, sub("z", "t", 11), fill=ACCENT_FILL[K], size=15)
    for cx, cy in l1:
        s += circle(cx, cy, 21, "zhat", fill=WHITE, size=12)
    s += aw(118, 1062, 208, 1028, K, w=1.8)
    s += aw(118, 1090, 208, 1124, K, w=1.8)
    s += txt(160, 1026, sub("a", "1", 9), 13, fill="#555")
    s += txt(160, 1128, sub("a", "1", 9) + "'", 13, fill="#555")
    for k, ly in enumerate(leaves):
        px, py = l1[0] if k < 2 else l1[1]
        s += aw(px + 21, py + (-9 if k % 2 == 0 else 9), 352,
                ly + (6 if k % 2 == 0 else -6), K, w=1.8)
        s += circle(374, ly, 19, "zhat", fill=WHITE, size=12)
    # the value head replaces the terminal MSE at every leaf
    s += (f'<rect x="412" y="968" width="152" height="218" rx="8" fill="none" '
          f'stroke="{ACCENT[K]}" stroke-width="2" stroke-dasharray="7,4"/>\n')
    s += txt(488, 960, "requires T4 (the value head)", 13, fill=ACCENT[K], weight="bold")
    for ly in leaves:
        s += aw(394, ly, 416, ly, K)
        s += mbox(420, ly - 19, 136, 38, "V( z , g )", K, size=15)
        s += aw(556, ly, 596, ly, K)
    s += txt(232, 1204, "step 1", 12, fill="#555")
    s += txt(374, 1204, "step H", 12, fill="#555")
    s += mbox(600, 972, 150, 196, "argsort( s )", K, size=17, sub_="elites -> new mu")
    s += aw(750, 1070, 786, 1070, K)
    s += pill(842, 1070, "mu", K, rx=48, fill=ACCENT_FILL[K])
    s += eq(PX + 26, 1232, "today:    s  =  " + nrm("zhat_H - z_g"), 17)
    s += eq(PX + 320, 1232, "V5:    s  =  - V( zhat_H , z_g )", 17)
    s += eq(PX + 600, 1232, "cost-to-go PAST the horizon", 15)
    s += eq(PX + 26, 1260, "Sutton: search and a learned value are NOT additive -- if "
            "T4's V is good the CEM gain shrinks, and that is the prediction", 14)
    s += strip(PX + 26, 1272, [
        ("depends on", "T4 (the V head)", True),
        ("latent vs task dist", "Spearman +0.398", True),
        ("CEM today", "H = 5,  300 x 30", False),
        ("Sutton predicts", "not additive with T4", True)], K, cw=198)
    return b + s


FIGURES = (
    ("arch-v1-pessimism.svg", v1_pessimism, (940, 1420)),
    ("arch-v2-gradient.svg", v2_gradient, (940, 1420)),
    ("arch-v3-twolevel.svg", v3_twolevel, (940, 1420)),
    ("arch-v4-policy.svg", v4_policy, (940, 1384)),
    ("arch-v5-valueguided.svg", v5_valueguided, (940, 1364)),
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
