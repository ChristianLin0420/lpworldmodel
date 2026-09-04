"""Architecture diagrams for round 4 (P1-P6), in the LpWM paper's style.

Same construction as analysis/arch_figs_causal.py: the unmodified LpWM schematic on top
and a MODULE PANEL below carrying the proposed change as blocks, operator nodes, arrows
and the equation it implements. No prose inside a diagram.

Everything structural comes from arch_figs_causal -- the fitting primitives (mbox, pill,
strip, eqrow), the hop, and the three connector routes into base(). See that module's
header for what each rule is there to prevent.

Usage:  python analysis/arch_figs_predictor.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, BOXLINE, DIM, GRID, INK, MSERED, MUTED, NAVY, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, BAR, BETA, DOT, EPS, IMPLIES, LAMB, MDASH, MINUS, NDASH, PHI, RHO, SIGMA,
    TAU, THETA, TIMES, aw, apoly, badge, dot, eq, eqfit, eqmid, eqrow, eqseq, fit_size,
    hatrun, head, hpoly, mbox, nrm, opnode, pill, sch, statrow, strip, sup,
    wire_to_action,
    wire_to_loss, wire_to_pred,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
PIX = PX + PW - 18
PY = 848                              # panel top, matching arch_figs_causal
PH = 468                              # tall panels; PY + PH = 1316, canvas 1344
EQ1, EQ2 = 1176, 1206                 # the two equation baselines
SY = 1230                             # the value strip's top edge
H_TALL = 1344


def panel(x, y, w, h, key, title):
    """Accent-bordered module panel with a title rule. Explicit geometry so each figure
    can size its own flow area."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.2"/>\n')
    s += txt(x + 20, y + 32, title, fit_size(title, 18, w - 44, floor=14),
             anchor="start", fill=c, weight="bold")
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
                  f'stroke="{c if on else GRID}" stroke-width="{1.4 if on else 0.8}"/>\n')
    return s


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + _hdr(w, h) + body + "</svg>\n")
    return name


# --------------------------------------------------------------- P2: the 2x2 factorial
def pred_2x2():
    """The predictor factorial. Three cells implemented and never run; one is new."""
    X0, CW, GAPX = 168, 356, 28
    Y0, CH, GAPY = 152, 176, 28
    s = head(40, 52, "The 2x2 nobody ran: how the action enters the predictor",
             "36 of 39 campaign arms use one cell of this table, plus a ReLU readout and "
             "a low-rank gate")
    for j, lab in enumerate(("the action is a BIAS", "the action is an OPERATOR")):
        s += txt(X0 + j * (CW + GAPX) + CW / 2, Y0 - 18, lab, 14, fill=MUTED)
    cells = [
        (0, 0, "linear_wb", ["u = W z", sub("", "t"), "  +  B a"], "NEVER RUN", "slate"),
        (1, 0, "linear_pa", ["u = [ diag(d(a)) + U(a)V(a)", sup("", "T", 11), " ] z",
                             sub("", "t")], "NEVER RUN", "amber"),
        (0, 1, "linear_var", [f"u = {SIGMA}", sub("", "k"), " A", sub("", "k"), " z",
                              sub("", "t-k"), "  +  B a"], "NEVER RUN", "slate"),
        (1, 1, "lie   (P1)", [f"u = R({THETA}(a)) {SIGMA}", sub("", "k"), f" R({PHI}",
                              sub("", "k"), ") z", sub("", "t-k")], "NEW", "green"),
    ]
    for cx, cy, name, formula, tag, key in cells:
        x, y = X0 + cx * (CW + GAPX), Y0 + cy * (CH + GAPY)
        lags = "1 lag" if cy == 0 else "3 lags"
        kind = "additive" if cx == 0 else "multiplicative"
        s += (f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="10" '
              f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="2.2"/>\n')
        s += badge(x + CW - 16, y + 14, tag, key, anchor="end")
        s += txt(x + CW / 2, y + 68, name, 21, fill=ACCENT[key], weight="bold")
        s += txt(x + CW / 2, y + 90, f"{lags}, {kind}", 12.5, fill=MUTED)
        s += (f'<rect x="{x + 16}" y="{y + 108}" width="{CW - 32}" height="52" rx="6" '
              f'fill="{WHITE}" stroke="{ACCENT[key]}" stroke-width="1.4"/>\n')
        s += eqmid(x + CW / 2, y + 140, formula, 15)
    for i, lab in enumerate(("1 lag", "3 lags")):
        s += txt(X0 - 18, Y0 + i * (CH + GAPY) + CH / 2 + 6, lab, 16, anchor="end")
    yb = Y0 + 2 * CH + GAPY
    s += txt(40, yb + 50, f"ltv  =  the BOTTOM-LEFT cell  +  ReLU readout  +  low-rank "
             f"gate  {MDASH}  36 of 39 arms", 16, anchor="start")
    # two lines, because one ran off the right edge of the canvas
    s += txt(40, yb + 76, "The campaign varied the link, the target, the regulariser, "
             "the loss, sparsity, width, LR, the encoder, the conditioning and the head "
             f"count {MDASH}", 13, anchor="start", fill=MUTED)
    s += txt(40, yb + 96, "and held this table fixed.", 13, anchor="start", fill=MUTED)
    return s


# ---------------------------------------------------------------------------- P1: lie
def lie():
    K = "green"
    b = base("(P1) PiWM-lie -- the action is a GROUP ELEMENT acting on the code")
    b += wire_to_pred(K, f"u = R({THETA}(a)) v")
    b = sch(b)

    s = panel(PX, PY, PW, PH, K, "module:  block-diagonal rotations; the action supplies "
              "the angle")
    yA, yB = 950, 1090
    # -- row A: the control angle
    s += circle(92, yA, 22, sub("a", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += mbox(152, yA - 24, 148, 48, sub("W", THETA, 13), K, size=18,
              sub_="Linear(D, D/2)")
    s += aw(114, yA, 148, yA, K)
    s += pill(352, yA, THETA, K, rx=34)
    s += aw(300, yA, 314, yA, K)
    # -- inset: what R(.) is, in the panel's own top-right corner
    s += blockdiag(768, 910, n=6, cell=17, key=K)
    s += txt(819, 1036, f"R = blockdiag( 2{TIMES}2 )", 12.5, fill=ACCENT[K])
    # -- row B: the autonomous sum, then the group action
    for i, lab in enumerate((sub("z", "t", 11), sub("z", "t-1", 10), sub("z", "t-2", 10))):
        s += pill(88, yB - 40 + i * 40, lab, K, rx=34, ry=16)
        s += aw(122, yB - 40 + i * 40, 164, yB - 20 + i * 20, K)
    s += mbox(168, yB - 30, 230, 60, SIGMA + sub("", "k", 12) + f"  R({PHI}"
              + sub("", "k", 12) + ") z" + sub("", "t-k", 12), K, size=16)
    s += pill(440, yB, "v", K, rx=26)
    s += aw(398, yB, 410, yB, K)
    s += mbox(500, yB - 26, 130, 52, f"R({THETA})", K, size=19)
    s += aw(466, yB, 496, yB, K)
    s += apoly([(352, yA + 19), (352, 1006), (565, 1006), (565, yB - 30)], K)
    s += pill(716, yB, sub("u", "t+1", 12), K, rx=52, fill=WHITE)
    s += aw(630, yB, 660, yB, K)
    # -- equations + accounting
    s += eqrow(EQ1, [
        [f"R({THETA}) R({THETA}')  =  R({THETA} + {THETA}')"],
        ["R", sup("", "T", 11), " R  =  I"],
        [f"{IMPLIES}  H-step composition is EXACT"]], 17)
    s += eqfit(PX + 26, EQ2, "the rotation acts on the linked (non-negative) code and "
               "emits pre-link u; the ReLU then selects a NEW support", 15)
    s += strip(PX + 26, SY, [
        ("params vs ltv", f"75,072  (10.8{TIMES} fewer)", False),
        ("cost per patch", f"O(D), not O(D²)", False),
        ("+ lie_sim (P1b)", "scaled rotations, +192", False),
        ("d_action becomes", f"O( {BAR}z{BAR} {DOT} {BAR}a {MINUS} a'{BAR} )", False)],
        K, cw=198)
    return b + s


# --------------------------------------------------------------------------- P4: ctrb
def ctrb():
    K = "blue"
    b = base("(P4) PiWM-ctrb -- make the learned dynamics CONTROLLABLE, not just accurate")
    b += wire_to_loss(K, "plus", "L_ctrb is added to the loss")
    b = sch(b)

    s = panel(PX, PY, PW, PH, K, "module:  the H-step controllability Gramian of the "
              "linearised dynamics")
    yA, yB = 956, 1084
    # -- row A: build the augmented system from the weights
    s += pill(120, yA, "A" + sub("", "0", 12) + "  A" + sub("", "1", 12)
              + "  A" + sub("", "2", 12), K, rx=64)
    s += mbox(206, yA - 26, 178, 52, "A_aug", K, size=18,
              sub_=f"companion form, 3D {TIMES} 3D")
    s += aw(184, yA, 202, yA, K)
    s += pill(120, yB, "B", K, rx=34)
    s += mbox(206, yB - 26, 178, 52, "B_aug  =  [ B ; 0 ; 0 ]", K, size=15)
    s += aw(154, yB, 202, yB, K)
    # -- the controllability matrix, then the Gramian, then the objective
    s += mbox(420, yA - 26, 220, 52, "C_H  =  [ B, AB, ..., A⁴ B ]", K, size=15,
              sub_="H = 5, the horizon CEM actually uses")
    s += aw(384, yA, 416, yA, K)
    s += apoly([(384, yB), (402, yB), (402, yA + 14), (416, yA + 14)], K)
    s += mbox(420, yB - 26, 220, 52, "W_c  =  C_H " + sup("C_H", "T", 11), K,
              size=17)
    s += apoly([(530, yA + 26), (530, yB - 30)], K)
    s += mbox(690, yB - 26, 198, 52, f"{MINUS} logdet( W_c + {EPS} I )", K, size=16)
    s += aw(640, yB, 686, yB, K)
    s += txt(784, yB + 50, "Cholesky; once per step, on the weights", 12,
             fill=MUTED)
    # -- reachable-set inset, in the row-A gutter to the right of C_H
    cx, cy = 780, 982
    s += (f'<ellipse cx="{cx}" cy="{cy}" rx="84" ry="13" fill="{ACCENT_FILL[K]}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.8"/>\n')
    s += aw(cx, cy, cx + 76, cy, K, w=1.5)
    s += aw(cx, cy, cx, cy - 30, "crit", w=1.5)
    s += txt(cx, cy - 40, "unreachable by ANY action sequence", 12, fill=ACCENT["crit"])
    s += txt(cx, cy + 32, "reachable in H steps", 12, fill=ACCENT[K])
    # -- equations
    s += eqrow(EQ1, [
        ["effort to reach direction v  =  v", sup("", "T", 11), " W_c",
         sup("", "-1", 11), " v"],
        [f"L  =  L_z  +  {LAMB}_c {DOT} L_ctrb"]], 17)
    s += eqfit(PX + 26, EQ2, f"ill-conditioned W_c  {IMPLIES}  CEM is flat along whole "
               "latent directions, however good the 1-step prediction is", 15)
    s += strip(PX + 26, SY, [
        ("d_action measures", f"{BAR}B a{BAR}, 1 step", False),
        ("W_c measures", "does it SPAN over H", False),
        ("explains V3", "healthy 1-step, null CEM", True),
        ("run on", "linear_var (exact)", False)], K, cw=198)
    return b + s


# ------------------------------------------------------------------ P5: actinfo-cond
def actinfo_cond():
    K = "magenta"
    b = base("(P5) PiWM-actinfo-cond -- negatives from p(a|z), not p(a)")
    b += wire_to_action(K, "K negatives, drawn at the SAME state")
    b = sch(b)

    s = panel(PX, PY, PW, PH, K, f"module:  the proposal distribution the negatives come "
              f"from {MDASH} the objective is unchanged")
    BW, BH, BY = 360, 236, 908
    AX, AY = 152, 140               # the anchor, as an offset within a box
    # An anchor, three neighbours inside the K-NN radius and seven outside it. Placed as
    # OFFSETS from the anchor so no point can drift under the "K-NN" or "z_t" label.
    near = [(38, -22), (-34, 16), (6, 34)]
    far = [(-92, -55), (54, -64), (152, -48), (168, 37), (-88, 67), (46, 78),
           (126, 74)]
    for x0, title, key, note in ((78, "V2:  q = p(a)", "slate",
                                 "permute a within the batch"),
                                (502, f"P5:  q ~ p(a | z{sub('', 't', 10)})", K,
                                 "actions of the K nearest "
                                 + sub("z", "t", 10))):
        c = ACCENT[key]
        s += (f'<rect x="{x0}" y="{BY}" width="{BW}" height="{BH}" rx="8" fill="{WHITE}" '
              f'stroke="{c}" stroke-width="1.8"/>\n')
        s += txt(x0 + BW / 2, BY + 30, title, 17, fill=c, weight="bold")
        s += txt(x0 + BW / 2, BY + 52, note, 12.5, fill=MUTED)
        ax, ay = x0 + AX, BY + AY
        if key == K:
            s += (f'<circle cx="{ax}" cy="{ay}" r="52" fill="none" stroke="{c}" '
                  f'stroke-width="1.6" stroke-dasharray="6,4"/>\n')
        for dx, dy in near + far:
            hot = key != K or (dx, dy) in near
            s += (f'<circle cx="{ax + dx}" cy="{ay + dy}" r="{5.5 if hot else 4.5}" '
                  f'fill="{c if hot else WHITE}" stroke="{c}" stroke-width="1.5"/>\n')
        s += (f'<circle cx="{ax}" cy="{ay}" r="8" fill="{ACCENT_FILL[key]}" '
              f'stroke="{c}" stroke-width="2.4"/>\n')
        s += txt(ax + 16, ay + 5, sub("z", "t", 11), 15, anchor="start", fill=c)
        if key == K:
            s += txt(ax + 70, ay + 5, "K-NN", 13, anchor="start", fill=c, weight="bold")
    s += eqrow(EQ1, [
        ["E", sub("", "k"), f"  =  {BAR}h( g(z", sub("", "t"), " , a", sub("", "k"),
         f") ) {MINUS} z", sub("", "t+1"), BAR + "²"],
        ["a", sub("", "0"), " = a", sub("", "t")],
        [f"L_ctrl  =  {MINUS}log softmax( {MINUS}E / {TAU} )"]], 17, gap=36)
    s += eqrow(EQ2, [
        ["q = p(a) bounds I( a ; (z", sub("", "t"), " , z", sub("", "t+1"),
         f") )  {MDASH}  winnable by PLAUSIBILITY alone."],
        ["q = p(a|z", sub("", "t"), ") bounds I( a ; z", sub("", "t+1"), " | z",
         sub("", "t"), " )."]], 15, gap=36)
    s += strip(PX + 26, SY, [
        ("V2 measured", f"d_action 0.699, {RHO} 0.448 {ARROWC} 0.052", True),
        ("the shortcut it took", "encode action identity, drop the state", True),
        ("cost of the fix", f"one cdist on 64{TIMES}64, 0 new params", False)], K, cw=264)
    return b + s


# --------------------------------------------------------------------- P6: objframe
def objframe():
    K = "amber"
    b = base("(P6) PiWM-objframe -- integrate the OBJECT's frame, not the agent's")
    b += wire_to_action(K, "object frame joins the action embedding")
    b = sch(b)

    s = panel(PX, PY, PW, PH, K, f"module:  the same head V3 used, on a different "
              f"variable {MDASH} V3 is the matched control")
    yA, yB = 962, 1100
    # -- V3 (the matched control), drawn in neutral slate
    s += txt(PX + 26, 916, "V3   (already run, n = 8)", 14, anchor="start",
             fill=ACCENT["slate"], weight="bold")
    s += circle(92, yA - 22, 21, sub("p", "t", 11), fill="#eeeeea", size=14)
    s += circle(92, yA + 24, 21, sub("a", "t", 11), fill="#eeeeea", size=14)
    s += mbox(158, yA - 24, 150, 48, "f_path", "slate", size=17, sub_="[ p_t ; a_t ]")
    s += aw(113, yA - 22, 154, yA - 10, "slate", w=1.5)
    s += aw(113, yA + 24, 154, yA + 12, "slate", w=1.5)
    s += pill(370, yA, "p", "slate", rx=44, fill=WHITE, hat="t+1")
    s += aw(308, yA, 322, yA, "slate", w=1.5)
    s += txt(288, yA + 64, f"agent pose {MDASH} rel_actions determine it (95.2%)", 12,
             fill=MUTED)
    # -- P6 (this round), in amber
    s += txt(PX + 26, 1054, "P6   (this round)", 14, anchor="start", fill=ACCENT[K],
             weight="bold")
    s += circle(92, yB, 21, sub("b", "t", 11), fill=ACCENT_FILL[K], size=14)
    s += txt(92, yB + 44, "state[2:5]", 12, fill=ACCENT[K])
    s += mbox(158, yB - 24, 150, 48, "f_obj", K, size=17, sub_="[ b_t ; p_t ; a_t ]")
    s += aw(113, yB, 154, yB, K)
    s += pill(370, yB, "b", K, rx=44, fill=WHITE, hat="t+1")
    s += aw(308, yB, 322, yB, K)
    s += pill(492, yB - 60, sub("b", "t+1", 10), K, rx=40)
    s += opnode(492, yB, "minus", K)
    s += aw(418, yB, 476, yB, K)
    s += apoly([(492, yB - 41), (492, yB - 18)], K)
    s += mbox(538, yB - 24, 104, 48, nrm("."), K)
    s += aw(508, yB, 534, yB, K)
    s += mbox(690, yB - 24, 198, 48, "L_obj", K, size=18, fill=ACCENT_FILL[K])
    s += aw(642, yB, 686, yB, K)
    s += eqrow(EQ1, [
        [f"L  =  L_z  +  {LAMB}_o {DOT} {BAR}", ("hat", "b", "t+1"), f" {MINUS} b",
         sub("", "t+1"), BAR + "²"],
        ["rollout uses the model's own head"]], 16)
    s += eqfit(PX + 26, EQ2, f"identical head shape, identical {LAMB}, identical "
               f"self-rollout {MDASH} P6 vs V3 differs ONLY in which pose is integrated",
               15)
    s += strip(PX + 26, SY, [
        ("target", "b = state[..., 2:5]", False),
        ("why it matters", "the GOAL uses this frame", False),
        ("why it is hard", "contact-dependent", False),
        ("leak control", "V3, matched", False)], K, cw=188)
    return b + s


# --------------------------------------------------------------------- P3: actgain
def actgain():
    K = "crit"
    b = base("(P3) PiWM-actgain -- one scalar; the falsifier that exists to kill P1")
    b += wire_to_pred(K, f"{BETA} scales the action branch")
    b = sch(b)

    P3_PH, P3_EQ1, P3_EQ2, P3_SY = 392, 1100, 1130, 1154
    s = panel(PX, PY, PW, P3_PH, K, f"module:  make d_state / d_action a hyperparameter "
              f"instead of an accident")
    yA, yB = 936, 1020
    s += pill(100, yA, sub("z", "t-k", 10), K, rx=42)
    s += mbox(176, yA - 24, 178, 48, SIGMA + sub("", "k", 12) + " A"
              + sub("", "k", 12) + " z" + sub("", "t-k", 12), K, size=16)
    s += aw(142, yA, 172, yA, K)
    s += pill(100, yB, sub("a", "t", 11), K, rx=34)
    s += mbox(176, yB - 24, 178, 48, "B a" + sub("", "t", 12), K, size=17)
    s += aw(134, yB, 172, yB, K)
    s += mbox(398, yB - 24, 214, 48, f"{BETA} {DOT} ( {DOT} ) / {BAR}{DOT}{BAR}", K,
              size=16, sub_="RMS-normalise, then scale")
    s += aw(354, yB, 394, yB, K)
    # the state branch's scale is the normaliser's reference: its OWN corridor, off the
    # box's bottom edge, so it no longer starts on top of the row-A output wire.
    s += apoly([(265, yA + 24), (265, 978), (376, 978), (376, yB - 12), (394, yB - 12)],
               K, dash="5,3")
    # the two branches meet at the sum, each on its own vertical, each landing ON the node
    s += opnode(668, 978, "plus", K)
    s += apoly([(354, yA), (668, yA), (668, 960)], K)
    s += apoly([(612, yB), (668, yB), (668, 996)], K)
    s += mbox(718, 954, 170, 48, f"W ReLU( {DOT} )", K, size=17)
    s += aw(684, 978, 714, 978, K)
    s += eqfit(PX + 26, P3_EQ1, f"{BETA} = 1 with the normaliser is itself a new arm; "
               "the baseline has no normaliser at all", 15)
    s += eqfit(PX + 26, P3_EQ2, "if this recovers what P1 recovers, the group structure "
               "is UNNECESSARY and the write-up must say so", 15.5, weight="bold")
    s += strip(PX + 26, P3_SY, [
        ("measured today", f"{BAR}W{BAR} action 20.3 vs encoder 354.7", True),
        ("grad_norm", "0.0075 vs 0.0360", True),
        ("swept", f"{BETA} = 0.3, 3.0", False)], K, cw=264)
    return b + s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, body, wh in (
        ("arch-pred-2x2.svg", pred_2x2(), (940, 668)),
        ("arch-lie.svg", lie(), (940, H_TALL)),
        ("arch-ctrb.svg", ctrb(), (940, H_TALL)),
        ("arch-actinfo-cond.svg", actinfo_cond(), (940, H_TALL)),
        ("arch-objframe.svg", objframe(), (940, H_TALL)),
        ("arch-actgain.svg", actgain(), (940, 1268)),
    ):
        print("  wrote", a.out + "/" + emit(name, body, a.out, *wh))


if __name__ == "__main__":
    main()
