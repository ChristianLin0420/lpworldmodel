"""Architecture diagrams for the round-5 REPRESENTATION proposals (T1-T3).

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py:
the unmodified LpWM schematic from analysis/arch_figs.base() on top, and a MODULE PANEL
below carrying the proposed change as blocks, operator nodes and arrows, with the
equation it implements underneath and a short strip of measured values. No prose.

T1  decoder   -- attach the decoder that the code already has, and cut the stop-grad at
                 models/visual_world_model.py:728 (z_dec = z_emb.detach()), so a pixel
                 error reaches the encoder for the first time. The switch that keeps the
                 branch dark is conf/train_rdmreg.yaml:126  has_decoder: False.
T2  patchdec  -- the 2x2 {CLS, 256 patch tokens} x {no decoder, decoder}. Three cells are
                 on the record (baseline; PiWM-columns at +0.072, null; and a decoder
                 column that has never existed at all); the fourth is new.
T3  residual  -- a per-sample weight built from the visual change proprio CANNOT predict.
                 The residual is the block. No ground-truth state is used: obs["proprio"]
                 is the agent's own [x, y, vx, vy], and everything else is the model's
                 own code.

Layout rule that keeps the panels readable: every panel is a strict left-to-right
dataflow on fixed row baselines, and any wire that must change rows does so on its own
vertical corridor. No two wires share a segment. The a_t circle at (318, 430) is fenced
in by the link/encoder/RDMReg wiring, so no connector is routed into it -- the dashed
callouts land on the observation circle (T1, T2) or in the MSE gutter (T3).

Usage:  python analysis/round5_figs_repr.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, INK, PEACH, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    aw, apoly, eq, mbox, nrm, opnode, pill, strip, sup,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-03")
MUTED = "#555555"


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
ENC_F = "Enc <tspan font-style='italic'>f</tspan>"


def obshook(key, label):
    """Dashed callout into the o_t+1 observation circle (812, 742), reached along the
    empty right margin x = 880. The decoder is the only part of the model that ever
    touches a pixel, so this is where its branch belongs."""
    s = apoly([(880, 898), (880, 742), (852, 742)], key, w=2.0, dash="6,4")
    s += txt(858, 800, label, 15, anchor="end", fill=ACCENT[key])
    return s


def gutterhook(key, label):
    """Dashed callout into the gutter BETWEEN the two code stacks (x 723..801), entered
    from below their bottom edge (y 337) so it clears the MSE arrow and its label. This
    is where a per-sample weight on the prediction loss attaches."""
    s = opnode(766, 330, "times", key, r=15)
    s += apoly([(880, 898), (880, 362), (766, 362), (766, 349)], key, w=2.0, dash="6,4")
    s += apoly([(766, 315), (766, 278)], key, w=2.0, dash="6,4")
    s += txt(836, 350, label, 18, fill=ACCENT[key], weight="bold")
    return s


def scissors(cx, cy, key="crit"):
    """An explicit CUT on a wire: a white gap punched through it, then a drawn pair of
    scissors. The serif face has no scissor glyph, so this is geometry."""
    c = ACCENT[key]
    s = f'<circle cx="{cx}" cy="{cy}" r="15" fill="{WHITE}"/>\n'
    s += (f'<path d="M{cx - 9},{cy - 8} L{cx + 13},{cy + 7} M{cx - 9},{cy + 8} '
          f'L{cx + 13},{cy - 7}" stroke="{c}" stroke-width="2.6" '
          f'stroke-linecap="round" fill="none"/>\n')
    for dy in (-11, 11):
        s += (f'<circle cx="{cx - 14}" cy="{cy + dy}" r="4.6" fill="none" '
              f'stroke="{c}" stroke-width="2.2"/>\n')
    return s


def struck(cx, cy, label, key="crit", size=15, half=38):
    """A label with a rule drawn through it -- the thing the proposal deletes."""
    c = ACCENT[key]
    s = txt(cx, cy, label, size, fill=c)
    s += (f'<path d="M{cx - half},{cy - 4} L{cx + half},{cy - 4}" stroke="{c}" '
          f'stroke-width="2" stroke-linecap="round"/>\n')
    return s


def badge(x, y, w, h, line1, line2, key):
    """A small bordered claim: used for the 'no privileged state' guarantee."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="2"/>\n')
    s += txt(x + w / 2, y + 22, line1, 15, fill=c, weight="bold")
    s += txt(x + w / 2, y + 40, line2, 12, fill=MUTED)
    return s


def cell2x2(x, y, w, h, title, who, result, tag, key):
    """One cell of the T2 factorial: coordinates, provenance, result, verdict pill."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="2.6"/>\n')
    s += txt(x + w / 2, y + 27, title, 19, fill=c, weight="bold")
    s += txt(x + w / 2, y + 46, who, 12.5, fill=MUTED)
    s += (f'<rect x="{x + 16}" y="{y + 56}" width="{w - 32}" height="34" rx="5" '
          f'fill="{WHITE}" stroke="{c}" stroke-width="1.5"/>\n')
    s += txt(x + w / 2, y + 79, result, 14.5)
    s += (f'<rect x="{x + w / 2 - 62}" y="{y + 96}" width="124" height="22" rx="11" '
          f'fill="{c}"/>\n')
    s += txt(x + w / 2, y + 112, tag, 12.5, fill=WHITE, weight="bold")
    return s


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --------------------------------------------------------------------- T1: decoder
def t1_decoder():
    K = "green"
    b = base("(T1) PiWM-decode -- attach the decoder and CUT the stop-grad")
    b += obshook(K, "reconstruct o_t+1 -- the only pixel term")

    PY, PH = 898, 418
    s = panel(PX, PY, PW, PH, K, "module:  the decoder branch the code already has, "
              "and the one line that makes it inert")
    yA, yB = 996, 1114

    # -- row A: observation -> encoder -> code -> [CUT] -> decoder -> pixel error
    s += circle(76, yA, 24, sub("o", "t", 11), size=15)
    s += aw(100, yA, 136, yA, K)
    s += mbox(140, yA - 24, 132, 48, ENC_F, K, size=17, sub_="ViT-S, D = 384")
    s += aw(272, yA, 300, yA, K)
    s += pill(334, yA, "z", K, rx=30)
    s += aw(364, yA, 446, yA, K)
    s += scissors(405, yA)
    s += struck(405, yA - 30, ".detach()")
    s += mbox(450, yA - 24, 132, 48, "Dec", K, size=19, sub_="never built today")
    s += aw(582, yA, 608, yA, K)
    s += pill(648, yA, "ohat", K, rx=36, fill=WHITE)
    s += circle(752, yA + 58, 20, sub("o", "t", 10), size=13)
    s += aw(752, yA + 38, 752, yA + 22, K)
    s += aw(684, yA, 734, yA, K)
    s += opnode(752, yA, "minus", K)
    s += aw(768, yA, 788, yA, K)
    s += mbox(792, yA - 24, 100, 48, nrm("."), K)
    s += aw(842, yA + 24, 842, yB - 28, K)
    s += mbox(758, yB - 24, 132, 48, "L_pix", K, size=18, fill=ACCENT_FILL[K])

    # -- row B: the gradient the cut restores, on its own corridor back to the encoder
    s += apoly([(754, yB), (206, yB), (206, yA + 30)], K, w=2.4, dash="7,4")
    s += txt(470, yB - 18, "the gradient the cut restores", 14, fill=ACCENT[K],
             weight="bold")
    s += txt(470, yB + 22, "the encoder receives a pixel error for the first time", 12,
             fill=MUTED)

    # -- equations, citations, accounting
    s += eq(PX + 26, 1180, "today:   z_dec  =  z_emb.detach()", 17)
    s += eq(PX + 340, 1180, "T1:   z_dec  =  z_emb", 17)
    s += txt(PX + 620, 1180, "visual_world_model.py:728", 14, anchor="start", fill=MUTED)
    s += eq(PX + 26, 1208, "L  =  L_z  +  lambda_d . " + nrm("Dec(z) - o"), 17)
    s += eq(PX + 330, 1208, "the branch is dark at the config level:", 15)
    s += txt(PX + 620, 1208, "train_rdmreg.yaml:126  has_decoder: False", 14,
             anchor="start", fill=MUTED)
    s += eq(PX + 26, 1236, "with the detach in place a decoder is only a linear probe on "
            "a frozen code -- it measures the code, it cannot shape it", 15)
    s += strip(PX + 26, 1250, [
        ("pixel decoder today", "never built", True),
        ("gradient into the encoder", "zero -- detached", True),
        ("tokens to paint 150,528 px", "1  (CLS)", True),
        ("what T1 adds", "decoder, no detach", False)], K, cw=198)
    return b + s


# -------------------------------------------------------------------- T2: patchdec
def t2_patchdec():
    K = "blue"
    b = base("(T2) PiWM-patchdec -- 256 located tokens, and something that reads them")
    b += obshook(K, "256 patch tokens  +  a pixel decoder")

    PY, PH = 898, 484
    s = panel(PX, PY, PW, PH, K, "module:  the factorial the campaign never opened -- "
              "tokens x pixels")
    X0, CW, GAP = 100, 380, 22
    Y0, CH, VGAP = 982, 124, 16

    s += txt(X0 + CW / 2, 968, "no decoder", 16, fill=MUTED)
    s += txt(X0 + CW + GAP + CW / 2, 968, "+ pixel decoder  (T1)", 16, fill=MUTED)
    s += txt(68, Y0 + CH / 2 - 6, "cls", 14, fill=MUTED)
    s += txt(68, Y0 + CH / 2 + 12, "P = 1", 12, fill=MUTED)
    s += txt(68, Y0 + CH + VGAP + CH / 2 - 6, "patch", 14, fill=MUTED)
    s += txt(68, Y0 + CH + VGAP + CH / 2 + 12, "P = 256", 12, fill=MUTED)

    cells = [
        (0, 0, "cls  x  no decoder", "LpWM-ltv -- 36 of 39 campaign arms",
         "CEM 0.347,   rel_mse 0.0092,   rho 0.448", "THE BASELINE", "slate"),
        (1, 0, "cls  x  decoder", "has_decoder: False -- never built",
         "384 numbers must paint 150,528 pixels", "NEVER RUN", "crit"),
        (0, 1, "patch  x  no decoder", "PiWM-columns  (n = 12)",
         "delta = +0.072,   95% CI [-0.064, +0.207],   p = 0.269", "NULL", "amber"),
        (1, 1, "patch  x  decoder", "T2  PiWM-patchdec  (this round)",
         "each token owns one 14 x 14 patch", "NEW", "green"),
    ]
    for cx, cy, title, who, result, tag, key in cells:
        x = X0 + cx * (CW + GAP)
        y = Y0 + cy * (CH + VGAP)
        s += cell2x2(x, y, CW, CH, title, who, result, tag, key)

    s += eq(PX + 26, 1278, "the whole right column has never existed:", 15)
    s += txt(PX + 320, 1278, "conf/train_rdmreg.yaml:126  has_decoder: False", 14,
             anchor="start", fill=MUTED)
    s += eq(PX + 26, 1304, "patch tokens alone gave the encoder more to say and nothing "
            "that forced it to say anything -- the decoder is that force", 15)
    s += strip(PX + 26, 1318, [
        ("patch alone", "+0.072  (null, n=12)", True),
        ("tokens the predictor sees", "1  ->  256", True),
        ("predictor params", "unchanged (shared op)", False),
        ("cells now on record", "3 of 4", False)], K, cw=198)
    return b + s


# -------------------------------------------------------------------- T3: residual
def t3_residual():
    K = "amber"
    b = base("(T3) PiWM-residual -- weight each sample by what proprio CANNOT predict")
    b += gutterhook(K, "w")

    PY, PH = 898, 462
    s = panel(PX, PY, PW, PH, K, "module:  the visual change proprio explains, subtracted "
              "off -- what is left is the block")
    yA, yB, yC, yR = 982, 1068, 1156, 1025

    # -- row A: the change the agent's own pose accounts for
    s += circle(60, yA - 21, 17, sub("p", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += circle(60, yA + 21, 17, sub("a", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += aw(77, yA - 21, 100, yA - 10, K)
    s += aw(77, yA + 21, 100, yA + 10, K)
    s += mbox(104, yA - 24, 140, 48, "f_prop", K, size=17, sub_="[ p_t ; a_t ]")
    s += aw(244, yA, 258, yA, K)
    s += pill(316, yA, "dhat_prop", K, rx=54, fill=WHITE)
    s += txt(316, yA + 38, "predicted visual change", 12, fill=ACCENT[K])

    # -- row B: the change actually measured, in the model's own code
    s += pill(80, yB - 20, sub("z", "t+1", 10), K, rx=40, ry=15)
    s += pill(80, yB + 20, sub("z", "t", 10), K, rx=40, ry=15)
    s += aw(120, yB - 20, 152, yB - 8, K)
    s += aw(120, yB + 20, 152, yB + 8, K)
    s += opnode(168, yB, "minus", K)
    s += aw(184, yB, 254, yB, K)
    s += pill(316, yB, "d_meas", K, rx=54)
    s += txt(316, yB + 36, "measured visual change", 12, fill=ACCENT[K])

    # -- the subtraction that isolates the block, then the weight it becomes
    s += aw(370, yA, 396, yR - 15, K)
    s += aw(370, yB, 396, yR + 15, K)
    s += opnode(414, yR, "minus", K)
    s += aw(430, yR, 536, yR, K)
    s += txt(478, yR - 10, "r", 17, fill=ACCENT[K], weight="bold")
    s += txt(470, yR + 28, "residual = the block", 12, fill=ACCENT[K])
    s += mbox(540, yR - 24, 152, 48, nrm("r"), K, sub_="/ batch mean")
    s += aw(692, yR, 710, yR, K)
    s += pill(744, yR, "w", K, rx=34)
    s += badge(614, 952, 282, 44, "NO ground-truth state",
               "obs['proprio'] = agent [x, y, vx, vy] only", K)

    # -- row C: the prediction loss the weight multiplies
    s += pill(80, yC - 20, "zhat", K, rx=40, ry=15, fill=WHITE)
    s += pill(80, yC + 20, sub("z", "t+1", 10), K, rx=40, ry=15, fill=WHITE)
    s += aw(120, yC - 20, 152, yC - 8, K)
    s += aw(120, yC + 20, 152, yC + 8, K)
    s += opnode(168, yC, "minus", K)
    s += aw(184, yC, 206, yC, K)
    s += mbox(210, yC - 24, 96, 48, nrm("."), K)
    s += aw(306, yC, 724, yC, K)
    s += aw(744, yR + 19, 744, yC - 18, K)
    s += opnode(744, yC, "times", K)
    s += aw(760, yC, 778, yC, K)
    s += mbox(782, yC - 24, 116, 48, "L_z", K, size=18, fill=ACCENT_FILL[K])

    # -- equations and the motion accounting
    s += eq(PX + 26, 1226, "r  =  ( z_t+1 - z_t )  -  f_prop( [ p_t ; a_t ] )", 17)
    s += eq(PX + 400, 1226, "w  =  " + nrm("r"), 17)
    s += eq(PX + 516, 1226, "/  batch mean,   detached", 17)
    s += eq(PX + 26, 1254, "L_z  =  w  .  " + nrm("zhat - z_t+1"), 17)
    s += eq(PX + 330, 1254, "V1 DOWN-weighted motion and collapsed; T3 up-weights "
            "what proprio misses", 15)
    s += eq(PX + 26, 1282, "f_prop is fit on detached z with its own term, so it cannot "
            "lower L_z by shrinking the residual it is measuring", 14)
    s += strip(PX + 26, 1296, [
        ("block motion, top 5%", "77.9% of all of it", True),
        ("block motion, top 1%", "34.8%", True),
        ("fully static transitions", "48%", True),
        ("block median motion", "0.000 px", True)], K, cw=198)
    return b + s


FIGURES = (
    ("arch-t1-decoder.svg", t1_decoder, (940, 1348)),
    ("arch-t2-patchdec.svg", t2_patchdec, (940, 1414)),
    ("arch-t3-residual.svg", t3_residual, (940, 1392)),
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
