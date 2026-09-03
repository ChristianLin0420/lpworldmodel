"""Architecture diagrams for round 4 (P1-P6), in the LpWM paper's style.

Same construction as analysis/arch_figs_causal.py: the unmodified LpWM schematic on top
and a MODULE PANEL below carrying the proposed change as blocks, operator nodes, arrows
and the equation it implements. No prose inside a diagram.

Layout rule that keeps the panels readable: every panel is a strict left-to-right
dataflow on fixed row baselines, and any wire that must change rows does so on its own
vertical corridor. No two wires share a segment.

Usage:  python analysis/arch_figs_predictor.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, BOXLINE, INK, MSERED, NAVY, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402
    aw, apoly, eq, mbox, nrm, opnode, pill, strip, sup,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure


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


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --------------------------------------------------------------- P2: the 2x2 factorial
def pred_2x2():
    """The predictor factorial. Three cells implemented and never run; one is new."""
    s = txt(470, 56, "The 2x2 nobody ran: how the action enters the predictor", 24)
    s += txt(470, 86, "36 of 39 campaign arms use one cell of this table, plus a ReLU "
             "readout and a low-rank gate", 15, fill="#555")
    CW, CH, X0, Y0 = 356, 214, 146, 150
    cells = [
        (0, 0, "linear_wb", "u = W z_t  +  B a", "1 lag, additive", "NEVER RUN", "slate"),
        (1, 0, "linear_pa", "u = [ diag(d(a)) + U(a)V(a)^T ] z_t", "1 lag, multiplicative",
         "NEVER RUN", "amber"),
        (0, 1, "linear_var", "u = SUM_k A_k z_t-k  +  B a", "3 lags, additive",
         "NEVER RUN", "slate"),
        (1, 1, "lie   (P1)", "u = R(theta(a)) SUM_k R(phi_k) z_t-k", "3 lags, multiplicative",
         "NEW", "green"),
    ]
    for cx, cy, name, formula, kind, tag, key in cells:
        x, y = X0 + cx * (CW + 26), Y0 + cy * (CH + 30)
        s += (f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="10" '
              f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="2.6"/>\n')
        s += txt(x + CW / 2, y + 40, name, 21, fill=ACCENT[key], weight="bold")
        s += txt(x + CW / 2, y + 66, kind, 13, fill="#555")
        s += (f'<rect x="{x + 16}" y="{y + 84}" width="{CW - 32}" height="46" rx="5" '
              f'fill="{WHITE}" stroke="{ACCENT[key]}" stroke-width="1.6"/>\n')
        s += txt(x + CW / 2, y + 113, formula, 15)
        s += (f'<rect x="{x + CW / 2 - 62}" y="{y + 150}" width="124" height="30" rx="15" '
              f'fill="{ACCENT[key]}"/>\n')
        s += txt(x + CW / 2, y + 171, tag, 15, fill=WHITE, weight="bold")
    s += txt(X0 - 40, Y0 + CH / 2, "1 lag", 18, fill="#555")
    s += txt(X0 - 40, Y0 + CH + 30 + CH / 2, "3 lags", 18, fill="#555")
    s += txt(X0 + CW / 2, Y0 - 16, "action is a BIAS", 16, fill="#555")
    s += txt(X0 + CW + 26 + CW / 2, Y0 - 16, "action is an OPERATOR", 16, fill="#555")
    s += txt(470, 654, "ltv  =  the BOTTOM-LEFT cell  +  ReLU readout  +  low-rank gate  --  36 of 39 arms", 17)
    s += txt(470, 682, "The campaign varied the link, the target, the regulariser, the "
             "loss, sparsity, width, LR, the encoder,", 14, fill="#555")
    s += txt(470, 702, "the conditioning and the head count -- and held this table fixed.",
             14, fill="#555")
    return s


# ---------------------------------------------------------------------------- P1: lie
def lie():
    K = "green"
    b = base("(P1) PiWM-lie -- the action is a GROUP ELEMENT acting on the code")
    b += apoly([(880, 898), (880, 120), (300, 120), (300, 190)], K, w=2.0, dash="6,4")
    b += txt(310, 106, "u = R(theta(a)) v", 16, anchor="start", fill=ACCENT[K])

    PY, PH = 898, 446
    s = panel(PX, PY, PW, PH, K, "module:  block-diagonal rotations; the action supplies "
              "the angle")
    # -- row A: the control angle
    yA = 984
    s += circle(96, yA, 23, sub("a", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += mbox(154, yA - 24, 148, 48, "W_theta", K, size=18, sub_="Linear(D, D/2)")
    s += aw(119, yA, 150, yA, K)
    s += pill(360, yA, "theta", K, rx=36)
    s += aw(302, yA, 320, yA, K)
    # -- inset: what R(.) is
    s += blockdiag(690, yA - 52, n=6, cell=17, key=K)
    s += txt(741, yA + 74, "R = blockdiag( 2x2 )", 13, fill=ACCENT[K])
    # -- row B: the autonomous sum, then the group action
    yB = 1102
    for i, lab in enumerate((sub("z", "t", 11), sub("z", "t-1", 10), sub("z", "t-2", 10))):
        s += pill(88, yB - 40 + i * 40, lab, K, rx=32, ry=16)
        s += aw(120, yB - 40 + i * 40, 168, yB - 22 + i * 18, K)
    s += mbox(172, yB - 30, 224, 60, "SUM_k  R(phi_k) z_t-k", K, size=16)
    s += pill(444, yB, "v", K, rx=28)
    s += aw(396, yB, 412, yB, K)
    s += mbox(510, yB - 26, 132, 52, "R(theta)", K, size=19)
    s += aw(472, yB, 506, yB, K)
    s += apoly([(360, yA + 19), (360, yA + 52), (576, yA + 52), (576, yB - 30)], K)
    s += pill(724, yB, "u_t+1", K, rx=48, fill=WHITE)
    s += aw(642, yB, 672, yB, K)
    # -- equations + accounting
    s += eq(PX + 26, 1206, "R(th) R(th')  =  R(th + th')", 17)
    s += eq(PX + 320, 1206, "R^T R  =  I", 17)
    s += eq(PX + 560, 1206, "=>  H-step composition is EXACT", 16)
    s += eq(PX + 26, 1234, "the rotation acts on the linked (non-negative) code and emits "
            "pre-link u; the ReLU then selects a NEW support", 15)
    s += strip(PX + 26, 1250, [
        ("params vs ltv", "75,072  (10.8x fewer)", False),
        ("cost per patch", "O(D), not O(D^2)", False),
        ("+ lie_sim (P1b)", "scaled rotations, +192", False),
        ("d_action becomes", "O( ||z|| . ||a - a'|| )", False)], K, cw=198)
    return b + s


# --------------------------------------------------------------------------- P4: ctrb
def ctrb():
    K = "blue"
    b = base("(P4) PiWM-ctrb -- make the learned dynamics CONTROLLABLE, not just accurate")
    b += apoly([(880, 898), (880, 362), (766, 362), (766, 349)], K, w=2.0, dash="6,4")
    b += opnode(766, 330, "plus", K, r=15)
    b += apoly([(766, 315), (766, 278)], K, w=2.0, dash="6,4")
    b += txt(836, 350, "L_ctrb", 16, fill=ACCENT[K], weight="bold")

    PY, PH = 898, 446
    s = panel(PX, PY, PW, PH, K, "module:  the H-step controllability Gramian of the "
              "linearised dynamics")
    yA, yB = 992, 1096
    # -- row A: build the augmented system from the weights
    s += pill(94, yA, "A_0 A_1 A_2", K, rx=62)
    s += mbox(184, yA - 26, 176, 52, "A_aug", K, size=18,
              sub_="companion form, 3D x 3D")
    s += aw(156, yA, 180, yA, K)
    s += pill(94, yB, "B", K, rx=32)
    s += mbox(184, yB - 26, 176, 52, "B_aug  =  [ B ; 0 ; 0 ]", K, size=15)
    s += aw(126, yB, 180, yB, K)
    # -- the controllability matrix
    s += mbox(400, yA - 26, 244, 52, "C_H  =  [ B, AB, ..., A^4 B ]", K, size=15,
              sub_="H = 5, the horizon CEM actually uses")
    s += apoly([(360, yA), (400, yA)], K)
    s += apoly([(360, yB), (380, yB), (380, yA + 14), (400, yA + 14)], K)
    # -- Gramian and objective
    s += mbox(400, yB - 26, 244, 52, "W_c  =  C_H C_H^T", K, size=17)
    s += apoly([(522, yA + 26), (522, yB - 26)], K)
    s += mbox(688, yB - 26, 186, 52, "- logdet( W_c + eps I )", K, size=15,
              sub_="Cholesky; once per step, on the weights")
    s += aw(644, yB, 684, yB, K)
    # -- reachable-set inset
    cx, cy = 748, yA
    s += (f'<ellipse cx="{cx}" cy="{cy}" rx="76" ry="12" fill="{ACCENT_FILL[K]}" '
          f'stroke="{ACCENT[K]}" stroke-width="2"/>\n')
    s += aw(cx, cy, cx + 72, cy, K, w=1.6)
    s += aw(cx, cy, cx, cy - 34, "crit", w=1.6)
    s += txt(cx + 18, cy + 34, "reachable in H steps", 12, fill=ACCENT[K])
    s += txt(cx + 4, cy - 42, "unreachable by ANY action sequence", 12, fill=ACCENT["crit"])
    # -- equations
    s += eq(PX + 26, 1206, "effort to reach direction v  =  v^T W_c^-1 v", 17)
    s += eq(PX + 400, 1206, "L  =  L_z  +  lambda_c . L_ctrb", 17)
    s += eq(PX + 26, 1234, "ill-conditioned W_c  =>  CEM is flat along whole latent "
            "directions, however good the 1-step prediction is", 15)
    s += strip(PX + 26, 1250, [
        ("d_action measures", "|| B a ||, 1 step", False),
        ("W_c measures", "does it SPAN over H", False),
        ("explains V3", "healthy 1-step, null CEM", True),
        ("run on", "linear_var (exact)", False)], K, cw=198)
    return b + s


# ------------------------------------------------------------------ P5: actinfo-cond
def actinfo_cond():
    K = "magenta"
    b = base("(P5) PiWM-actinfo-cond -- negatives from p(a|z), not p(a)")
    b += apoly([(880, 898), (880, 486), (740, 486), (740, 430), (362, 430)], K,
               w=2.0, dash="6,4")
    b += txt(736, 402, "K negatives, drawn at the SAME state", 15, anchor="end",
             fill=ACCENT[K])

    PY, PH = 898, 446
    s = panel(PX, PY, PW, PH, K, "module:  the proposal distribution the negatives come "
              "from -- the objective is unchanged")
    # -- left: V2's marginal proposal.  right: P5's conditional proposal.
    for col, (x0, title, key, note) in enumerate((
            (76, "V2:  q = p(a)", "slate", "permute a within the batch"),
            (500, "P5:  q ~ p(a | z_t)", K, "actions of the K nearest z_t"))):
        c = ACCENT[key]
        s += (f'<rect x="{x0}" y="962" width="360" height="196" rx="8" fill="{WHITE}" '
              f'stroke="{c}" stroke-width="2"/>\n')
        s += txt(x0 + 180, 990, title, 18, fill=c, weight="bold")
        s += txt(x0 + 180, 1012, note, 13, fill="#555")
        # the batch as points in z_t space; the anchor is the filled one
        # normalised so every point stays inside the box (x0+34..x0+326, 1034..1146)
        pts = [(.08, .18), (.30, .06), (.24, .62), (.44, .30), (.79, .14), (.46, .90),
               (.92, .58), (.04, .96), (.68, .88), (.17, .74)]
        ax, ay = x0 + 34 + 0.34 * 292, 1034 + 0.46 * 112
        for fx, fy in pts:
            px, py = x0 + 34 + fx * 292, 1034 + fy * 112
            near = (px - ax) ** 2 + (py - ay) ** 2 < 48 ** 2
            hot = key == K and near
            s += (f'<circle cx="{px}" cy="{py}" r="{5.5 if hot else 4.5}" '
                  f'fill="{c if (hot or key != K) else WHITE}" stroke="{c}" '
                  f'stroke-width="1.6"/>\n')
        s += (f'<circle cx="{ax}" cy="{ay}" r="8" fill="{ACCENT_FILL[key]}" '
              f'stroke="{c}" stroke-width="2.6"/>\n')
        s += txt(ax + 20, ay + 5, sub("z", "t", 11), 15, anchor="start", fill=c)
        if key == K:
            s += (f'<circle cx="{ax}" cy="{ay}" r="48" fill="none" stroke="{c}" '
                  f'stroke-width="1.8" stroke-dasharray="6,4"/>\n')
            s += txt(ax + 60, ay - 40, "K-NN", 14, fill=c, weight="bold")
    s += eq(PX + 26, 1196, "E_k  =  " + nrm("h( g(z_t , a_k) ) - z_t+1"), 17)
    s += eq(PX + 330, 1196, "a_0 = a_t", 17)
    s += eq(PX + 480, 1196, "L_ctrl  =  -log softmax( -E / tau )", 17)
    s += eq(PX + 26, 1224, "q = p(a) bounds I( a ; (z_t , z_t+1) ) -- winnable by "
            "PLAUSIBILITY alone.   q = p(a|z_t) bounds I( a ; z_t+1 | z_t ).", 15)
    s += strip(PX + 26, 1242, [
        ("V2 measured", "d_action 0.699, rho 0.448 -> 0.052", True),
        ("the shortcut it took", "encode action identity, drop the state", True),
        ("cost of the fix", "one cdist on 64x64, 0 new params", False)], K, cw=268)
    return b + s


# --------------------------------------------------------------------- P6: objframe
def objframe():
    K = "amber"
    b = base("(P6) PiWM-objframe -- integrate the OBJECT's frame, not the agent's")
    b += apoly([(880, 898), (880, 486), (740, 486), (740, 430), (362, 430)], K,
               w=2.0, dash="6,4")
    b += txt(736, 402, "object frame joins the action embedding", 15, anchor="end",
             fill=ACCENT[K])

    PY, PH = 898, 446
    s = panel(PX, PY, PW, PH, K, "module:  the same head V3 used, on a different variable "
              "-- V3 is the matched control")
    yA, yB = 996, 1104
    # -- V3 (control), drawn in grey
    s += circle(92, yA, 22, sub("p", "t", 11), fill="#eeeeea", size=14)
    s += circle(92, yA + 54, 22, sub("a", "t", 11), fill="#eeeeea", size=14)
    s += mbox(160, yA + 2, 150, 48, "f_path", "slate", size=17, sub_="[ p_t ; a_t ]")
    s += aw(114, yA, 156, yA + 14, "slate", w=1.6)
    s += aw(114, yA + 54, 156, yA + 40, "slate", w=1.6)
    s += pill(374, yA + 26, "phat_t+1", "slate", rx=48, fill=WHITE)
    s += aw(310, yA + 26, 322, yA + 26, "slate", w=1.6)
    s += txt(236, yA - 26, "V3  (already run, n=8)", 14, fill=ACCENT["slate"],
             weight="bold")
    s += txt(236, 1074, "agent pose -- rel_actions determine it (95.2%)", 12, fill="#666")
    # -- P6, in amber
    s += circle(92, yB + 40, 22, sub("b", "t", 11), fill=ACCENT_FILL[K], size=14)
    s += txt(92, yB + 84, "state[2:5]", 12, fill=ACCENT[K])
    s += mbox(160, yB + 16, 150, 48, "f_obj", K, size=17, sub_="[ b_t ; p_t ; a_t ]")
    s += aw(114, yB + 40, 156, yB + 40, K)
    s += pill(374, yB + 40, "bhat_t+1", K, rx=48, fill=WHITE)
    s += aw(310, yB + 40, 322, yB + 40, K)
    s += pill(492, yB - 26, sub("b", "t+1", 10), K, rx=42)
    s += opnode(492, yB + 40, "minus", K)
    s += aw(422, yB + 40, 476, yB + 40, K)
    s += apoly([(492, yB - 7), (492, yB + 24)], K)
    s += mbox(538, yB + 16, 104, 48, nrm("."), K)
    s += aw(508, yB + 40, 534, yB + 40, K)
    s += mbox(680, yB + 16, 168, 48, "L_obj", K, size=18, fill=ACCENT_FILL[K])
    s += aw(642, yB + 40, 676, yB + 40, K)
    s += txt(236, yB + 4, "P6  (this round)", 14, fill=ACCENT[K], weight="bold")
    s += eq(PX + 26, 1246, "L  =  L_z  +  lambda_o . " + nrm("bhat_t+1 - b_t+1"), 16)
    s += eq(PX + 400, 1246, "rollout uses the model's own head", 16)
    s += eq(PX + 26, 1274, "identical head shape, identical lambda, identical self-rollout "
            "-- P6 vs V3 differs ONLY in which pose is integrated", 15)
    s += strip(PX + 26, 1290, [
        ("target", "b = state[..., 2:5]", False),
        ("why it matters", "the GOAL uses this frame", False),
        ("why it is hard", "contact-dependent", False),
        ("leak control", "V3, matched", False)], K, cw=198)
    return b + s


# --------------------------------------------------------------------- P3: actgain
def actgain():
    K = "crit"
    b = base("(P3) PiWM-actgain -- one scalar; the falsifier that exists to kill P1")
    b += apoly([(880, 898), (880, 120), (300, 120), (300, 190)], K, w=2.0, dash="6,4")
    b += txt(310, 106, "beta scales the action branch", 16, anchor="start", fill=ACCENT[K])

    PY, PH = 898, 400
    s = panel(PX, PY, PW, PH, K, "module:  make d_state / d_action a hyperparameter "
              "instead of an accident")
    yA, yB = 992, 1076
    s += pill(96, yA, sub("z", "t-k", 10), K, rx=38)
    s += mbox(170, yA - 24, 168, 48, "SUM_k A_k z_t-k", K, size=16)
    s += aw(134, yA, 166, yA, K)
    s += pill(96, yB, sub("a", "t", 11), K, rx=32)
    s += mbox(170, yB - 24, 168, 48, "B a_t", K, size=17)
    s += aw(128, yB, 166, yB, K)
    s += mbox(374, yB - 24, 214, 48, "beta . ( . ) / || . ||", K, size=15,
              sub_="RMS-normalise, then scale")
    s += aw(338, yB, 370, yB, K)
    s += apoly([(338, yA), (356, yA), (356, yB - 14), (370, yB - 14)], K, dash="5,3")
    s += opnode(628, yA + 42, "plus", K)
    s += apoly([(338, yA), (600, yA), (600, yA + 26)], K)
    s += apoly([(588, yB), (612, yB), (612, yA + 58)], K)
    s += mbox(682, yA + 18, 190, 48, "W ReLU( . )", K, size=17)
    s += aw(644, yA + 42, 678, yA + 42, K)
    s += eq(PX + 26, 1176, "beta = 1 with the normaliser is itself a new arm; the "
            "baseline has no normaliser at all", 15)
    s += eq(PX + 26, 1204, "if this recovers what P1 recovers, the group structure is "
            "UNNECESSARY and the write-up must say so", 16, weight="bold")
    s += strip(PX + 26, 1220, [
        ("measured today", "||W|| action 20.3 vs encoder 354.7", True),
        ("grad_norm", "0.0075 vs 0.0360", True),
        ("swept", "beta = 0.3, 3.0", False)], K, cw=268)
    return b + s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, body, wh in (
        ("arch-pred-2x2.svg", pred_2x2(), (940, 730)),
        ("arch-lie.svg", lie(), (940, 1372)),
        ("arch-ctrb.svg", ctrb(), (940, 1372)),
        ("arch-actinfo-cond.svg", actinfo_cond(), (940, 1372)),
        ("arch-objframe.svg", objframe(), (940, 1372)),
        ("arch-actgain.svg", actgain(), (940, 1320)),
    ):
        print("  wrote", a.out + "/" + emit(name, body, a.out, *wh))


if __name__ == "__main__":
    main()
