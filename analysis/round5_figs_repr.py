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

STYLE.  analysis/style.py, via analysis/arch_figs.py: sans throughout, hues by IDENTITY,
pale fills with a matching saturated border, recessive furniture.  Maths is typeset --
subscripts, a drawn circumflex, real Greek and a real norm -- because the ASCII stand-ins
("zhat", "dhat_prop", "lambda_d") were the loudest thing on the page.

LAYOUT RULES, and the defect each one exists to stop

  * Strict left-to-right dataflow on fixed row baselines; a wire that changes rows does so
    on its OWN vertical corridor. No two wires share a segment. The a_t circle at
    (318, 430) is fenced in by the link/encoder/RDMReg wiring, so no connector is routed
    into it -- the dashed callouts land on the observation circle (T1, T2) or in the MSE
    gutter (T3).
  * SHIFT reclaims the empty band above the schematic: base() draws nothing above y=158,
    which used to leave 158px of white at the top of every figure in this set.
  * Every footer line goes through frow(), which guarantees a gap between clauses and a
    hard right margin. Hand-placed footers used to run past the panel border and off the
    canvas: T1's `train_rdmreg.yaml:126 has_decoder: False` ended at x=934 on a 940-wide
    page, and its next line at x=945.
  * Nothing is sized by hand where it can be measured -- panel titles, factorial cells and
    verdict pills all fit their own text.

Usage:  python analysis/round5_figs_repr.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, INK, MUTED, PEACH, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, sup, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    BAR, DOT, MINUS, NDASH, SQ, TIMES, aw, apoly, eq, eqseq, fit_size, hatrun, mbox,
    nb, nrm, opnode, pill, seq_w, strip,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
PIN, PIX = PX + 26, PX + PW - 26      # the footer's inner margins
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-03")

# The whole figure is drawn in base()'s coordinates and then lifted. base() puts no ink
# above y=158; SHIFT turns that dead band into a normal 78px top margin.
SHIFT = -80


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


# --- the footer -------------------------------------------------------------------
def frow(y, cols, size=16, x0=PIN, x1=PIX, gap=34, floor=12.0):
    """One footer line of several clauses, justified between x0 and x1 with a GUARANTEED
    gap and a hard right margin.

    A clause is (text, fill, weight) or (text, fill, weight, scale); `text` may also be an
    eqseq() part list, so a clause can carry a hatted variable. The gap shrinks first and
    the type second -- a footer can never run past x1, which is what every overflow in
    this set was."""
    cols = [(c if isinstance(c, list) else [c], f, w, (p[3] if len(p) > 3 else 1.0))
            for p in cols for c, f, w in (p[:3],)]
    n, avail = len(cols), x1 - x0
    sz = float(size)
    while True:
        ws = [seq_w(c, sz * k, w) for c, _, w, k in cols]
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
        while sz > floor and seq_w(c, sz * k, w) > avail:
            sz -= 0.25
        ws = [seq_w(c, sz * k, w)]
    g = gap if n < 2 else max(gap, (avail - sum(ws)) / (n - 1))
    s, cx = "", float(x0)
    for (c, f, w, k), wd in zip(cols, ws):
        s += eqseq(round(cx, 1), y, c, round(sz * k, 2), f, w)
        cx += wd + g
    return s


# --- a few more, local to this round ----------------------------------------------
ENC_F = "Enc <tspan font-style='italic'>f</tspan>"


def rlabel(x, y, parts, size=15, fill=INK, weight="normal"):
    """An eqseq() run ENDING at x -- the right-aligned twin of eqseq()."""
    return eqseq(round(x - seq_w(parts, size, weight), 1), y, parts, size, fill, weight)


def obshook(key, parts):
    """Dashed callout into the o_t+1 observation circle (812, 742), reached along the
    empty right margin x = 880. The decoder is the only part of the model that ever
    touches a pixel, so this is where its branch belongs."""
    s = apoly([(880, 898), (880, 742), (852, 742)], key, w=1.8, dash="6,4")
    s += rlabel(866, 800, parts, 15, ACCENT[key])
    return s


def gutterhook(key, label):
    """Dashed callout into the gutter BETWEEN the two code stacks (x 723..801), entered
    from below their bottom edge (y 337) so it clears the MSE arrow and its label. This
    is where a per-sample weight on the prediction loss attaches."""
    s = opnode(766, 330, "times", key, r=15)
    s += apoly([(880, 898), (880, 362), (766, 362), (766, 349)], key, w=1.8, dash="6,4")
    s += apoly([(766, 315), (766, 272)], key, w=1.8, dash="6,4")
    s += txt(872, 352, label, 17, anchor="end", fill=ACCENT[key], weight="bold")
    return s


def scissors(cx, cy, key="crit"):
    """An explicit CUT on a wire: a white gap punched through it, then a drawn pair of
    scissors. No sans face has a scissor glyph, so this is geometry."""
    c = ACCENT[key]
    s = f'<circle cx="{cx}" cy="{cy}" r="15" fill="{WHITE}"/>\n'
    s += (f'<path d="M{cx - 9},{cy - 8} L{cx + 13},{cy + 7} M{cx - 9},{cy + 8} '
          f'L{cx + 13},{cy - 7}" stroke="{c}" stroke-width="2.4" '
          f'stroke-linecap="round" fill="none"/>\n')
    for dy in (-11, 11):
        s += (f'<circle cx="{cx - 14}" cy="{cy + dy}" r="4.6" fill="none" '
              f'stroke="{c}" stroke-width="2.0"/>\n')
    return s


def struck(cx, cy, label, key="crit", size=15):
    """A label with a rule drawn through it -- the thing the proposal deletes. The rule
    is measured off the label, so it can neither fall short nor overhang."""
    c = ACCENT[key]
    half = text_width(label, size) / 2 + 5
    s = txt(cx, cy, label, size, fill=c)
    s += (f'<path d="M{round(cx - half, 1)},{cy - 5} L{round(cx + half, 1)},{cy - 5}" '
          f'stroke="{c}" stroke-width="1.8" stroke-linecap="round"/>\n')
    return s


def badge(x, y, w, h, line1, line2, key):
    """A small bordered claim: used for the 'no privileged state' guarantee."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.6"/>\n')
    s += txt(x + w / 2, y + 22, line1, fit_size(line1, 15, w - 24, floor=12,
                                                weight="bold"), fill=c, weight="bold")
    s += txt(x + w / 2, y + 40, line2, fit_size(line2, 12, w - 20, floor=9.5),
             fill=MUTED)
    return s


def cell2x2(x, y, w, h, title, who, result, tag, key):
    """One cell of the T2 factorial: coordinates, provenance, result, verdict pill. Every
    line fits its own box and the pill is as wide as its own tag."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="2.0"/>\n')
    s += txt(x + w / 2, y + 27, title, fit_size(title, 19, w - 30, floor=15, weight="bold"),
             fill=c, weight="bold")
    s += txt(x + w / 2, y + 46, who, fit_size(who, 12.5, w - 26, floor=10.5), fill=MUTED)
    s += (f'<rect x="{x + 16}" y="{y + 56}" width="{w - 32}" height="34" rx="5" '
          f'fill="{WHITE}" stroke="{c}" stroke-width="1.3"/>\n')
    s += txt(x + w / 2, y + 79, result, fit_size(result, 14.5, w - 52, floor=11.5))
    pw = round(text_width(tag, 12.5, "bold") + 30, 1)
    s += (f'<rect x="{round(x + w / 2 - pw / 2, 1)}" y="{y + 96}" width="{pw}" '
          f'height="22" rx="11" fill="{c}"/>\n')
    s += txt(x + w / 2, y + 112, tag, 12.5, fill=WHITE, weight="bold")
    return s


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h + SHIFT) + lift(body) + "</svg>\n")
    return name


# --------------------------------------------------------------------- T1: decoder
def t1_decoder():
    K = "green"
    b = base("(T1) PiWM-decode -- attach the decoder and CUT the stop-grad")
    b += obshook(K, ["reconstruct ", ("hat", "o", "t+1"),
                     f" {NDASH} the only pixel term"])

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
    s += struck(405, yA - 32, ".detach()")
    s += mbox(450, yA - 24, 132, 48, "Dec", K, size=19, sub_="never built today")
    s += aw(582, yA, 608, yA, K)
    s += pill(648, yA, "o", K, rx=36, fill=WHITE, hat=True)
    s += circle(752, yA + 58, 20, sub("o", "t", 10), size=13)
    s += aw(752, yA + 38, 752, yA + 22, K)
    s += aw(684, yA, 734, yA, K)
    s += opnode(752, yA, "minus", K)
    s += aw(768, yA, 788, yA, K)
    s += mbox(792, yA - 24, 100, 48, nrm("."), K)
    s += aw(842, yA + 24, 842, yB - 28, K)
    s += mbox(758, yB - 24, 132, 48, sub("L", "pix", 12), K, size=18,
              fill=ACCENT_FILL[K])

    # -- row B: the gradient the cut restores, on its own corridor back to the encoder
    s += apoly([(756, yB), (206, yB), (206, yA + 30)], K, w=2.2, dash="7,4")
    s += txt(462, yB - 20, "the gradient the cut restores", 14, fill=ACCENT[K],
             weight="bold")
    s += txt(462, yB + 24, "the encoder receives a pixel error for the first time", 12.5,
             fill=MUTED)

    # -- equations, citations, accounting
    s += frow(1176, [
        ("today:   z_dec  =  z_emb.detach()", INK, "normal"),
        ("T1:   z_dec  =  z_emb", ACCENT[K], "bold"),
        ("visual_world_model.py:728", MUTED, "normal", 0.86)])
    s += frow(1204, [
        (["L  =  ", sub("L", "z", 12), f"  +  {sub('λ', 'd', 12)}  {DOT}  ",
          nrm("Dec(z) " + MINUS + " o")], INK, "normal"),
        ("the branch is dark at the config level:", INK, "normal"),
        ("train_rdmreg.yaml:126  has_decoder: False", MUTED, "normal", 0.86)])
    s += frow(1232, [
        (f"with the detach in place a decoder is only a linear probe on a frozen code "
         f"{NDASH} it measures the code, it cannot shape it", INK, "normal")], size=14.5)
    s += strip(PIN, 1248, [
        ("pixel decoder today", "never built", True),
        ("gradient into the encoder", "zero – detached", True),
        ("tokens to paint 150,528 px", "1  (CLS)", True),
        ("what T1 adds", "decoder, no detach", False)], K, cw=198)
    return b + s


# -------------------------------------------------------------------- T2: patchdec
def t2_patchdec():
    # The four cells carry the identities here (slate baseline, crimson never-run, amber
    # null, green new), so the panel frame is FURNITURE: drawing it in the module's own
    # green made the green "NEW" cell disappear into its own background.
    K, F = "slate", "blue"
    b = base("(T2) PiWM-patchdec -- 256 located tokens, and something that reads them")
    b += obshook(F, ["256 patch tokens  +  a pixel decoder"])

    PY, PH = 898, 484
    s = panel(PX, PY, PW, PH, K, "module:  the factorial the campaign never opened "
              f"{NDASH} tokens {TIMES} pixels")
    X0, CW, GAP = 100, 380, 22
    Y0, CH, VGAP = 982, 124, 16

    s += txt(X0 + CW / 2, 966, "no decoder", 15.5, fill=MUTED)
    s += txt(X0 + CW + GAP + CW / 2, 966, "+ pixel decoder  (T1)", 15.5, fill=MUTED)
    for row, (a, bb) in enumerate((("cls", "P = 1"), ("patch", "P = 256"))):
        cy = Y0 + row * (CH + VGAP) + CH / 2
        s += txt(90, cy - 5, a, 14, anchor="end", fill=MUTED)
        s += txt(90, cy + 13, bb, 12, anchor="end", fill=MUTED)

    cells = [
        (0, 0, f"cls  {TIMES}  no decoder", f"LpWM-ltv {NDASH} 36 of 39 campaign arms",
         "CEM 0.347,   rel_mse 0.0092,   ρ 0.448",
         "THE BASELINE", "slate"),
        (1, 0, f"cls  {TIMES}  decoder", f"has_decoder: False {NDASH} never built",
         "384 numbers must paint 150,528 pixels", "NEVER RUN", "crit"),
        (0, 1, f"patch  {TIMES}  no decoder", "PiWM-columns  (n = 12)",
         "Δ = +0.072,   95% CI [-0.064, +0.207],   p = 0.269", "NULL", "amber"),
        (1, 1, f"patch  {TIMES}  decoder", "T2  PiWM-patchdec  (this round)",
         f"each token owns one 14 {TIMES} 14 patch", "NEW", "blue"),
    ]
    for cx, cy, title, who, result, tag, key in cells:
        x = X0 + cx * (CW + GAP)
        y = Y0 + cy * (CH + VGAP)
        s += cell2x2(x, y, CW, CH, title, who, result, tag, key)

    s += frow(1278, [
        ("the whole right column has never existed:", INK, "normal"),
        ("conf/train_rdmreg.yaml:126  has_decoder: False", MUTED, "normal", 0.88)],
        size=15.5)
    s += frow(1304, [
        (f"patch tokens alone gave the encoder more to say and nothing that forced it to "
         f"say anything {NDASH} the decoder is that force", INK, "normal")], size=15)
    s += strip(PIN, 1318, [
        ("patch alone", "+0.072  (null, n=12)", True),
        ("tokens the predictor sees", "1  →  256", True),
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
              f"off {NDASH} what is left is the block")
    yA, yB, yC, yR = 982, 1068, 1156, 1030

    # -- row A: the change the agent's own pose accounts for
    s += circle(60, yA - 21, 17, sub("p", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += circle(60, yA + 21, 17, sub("a", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += aw(77, yA - 21, 100, yA - 10, K)
    s += aw(77, yA + 21, 100, yA + 10, K)
    s += mbox(104, yA - 24, 140, 48, sub("f", "prop", 11), K, size=17,
              sub_="[ " + sub("p", "t", 9) + " ; " + sub("a", "t", 9) + " ]")
    s += aw(244, yA, 258, yA, K)
    s += pill(316, yA, "d", K, rx=54, fill=WHITE, hat="prop")
    s += txt(316, yA + 38, "predicted visual change", 12, fill=ACCENT[K])

    # -- row B: the change actually measured, in the model's own code
    s += pill(80, yB - 20, sub("z", "t+1", 10), K, rx=40, ry=15)
    s += pill(80, yB + 20, sub("z", "t", 10), K, rx=40, ry=15)
    s += aw(120, yB - 20, 152, yB - 8, K)
    s += aw(120, yB + 20, 152, yB + 8, K)
    s += opnode(168, yB, "minus", K)
    s += aw(184, yB, 254, yB, K)
    s += pill(316, yB, sub("d", "meas", 11), K, rx=54)
    s += txt(316, yB + 36, "measured visual change", 12, fill=ACCENT[K])

    # -- the subtraction that isolates the block, then the weight it becomes
    s += aw(370, yA, 396, yR - 15, K)
    s += aw(370, yB, 396, yR + 15, K)
    s += opnode(414, yR, "minus", K)
    s += aw(430, yR, 536, yR, K)
    s += txt(483, yR - 11, "r", 17, fill=ACCENT[K], weight="bold")
    s += txt(500, yR + 31, "residual = the block", 12, fill=ACCENT[K])
    s += mbox(540, yR - 24, 152, 48, nrm("r"), K, sub_="/ batch mean")
    s += aw(692, yR, 710, yR, K)
    s += pill(744, yR, "w", K, rx=34)
    s += badge(626, 950, 270, 42, "NO ground-truth state",
               "obs['proprio'] = agent [x, y, vx, vy] only", K)

    # -- row C: the prediction loss the weight multiplies
    s += pill(80, yC - 20, "z", K, rx=40, ry=15, fill=WHITE, hat=True)
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
    s += mbox(782, yC - 24, 116, 48, sub("L", "z", 12), K, size=18,
              fill=ACCENT_FILL[K])

    # -- equations and the motion accounting
    zt1, zt = sub("z", "t+1", 12), sub("z", "t", 12)
    s += frow(1222, [
        ([f"r  =  ( {zt1} {MINUS} {zt} )  {MINUS}  {sub('f', 'prop', 12)}( [ "
          f"{sub('p', 't', 12)} ; {sub('a', 't', 12)} ] )"], INK, "normal"),
        (["w  =  " + nrm("r")], INK, "normal"),
        ("/  batch mean,   detached", MUTED, "normal", 0.92)])
    s += frow(1250, [
        ([sub("L", "z", 12) + f"  =  w  {DOT}  {BAR}", ("hat", "z", ""),
          f" {MINUS} {zt1}{BAR}{SQ}"], INK, "normal"),
        ("V1 DOWN-weighted motion and collapsed; T3 up-weights what proprio misses",
         INK, "normal", 0.94)])
    s += frow(1278, [
        (f"{sub('f', 'prop', 11)} is fit on detached z with its own term, so it cannot "
         f"lower {sub('L', 'z', 11)} by shrinking the residual it is measuring",
         INK, "normal")], size=14)
    s += strip(PIN, 1292, [
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
