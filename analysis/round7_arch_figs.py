"""Architecture diagrams for round 7 -- the round that changes the REPRESENTATION.

Round 7's premise is a measurement, not an argument: ridge-decoding the block's pose from
ONE FROZEN FRAME, the baseline's cls latent scores 15.72 deg against a 14.51 deg
best-constant bound (worse than the bound), five times the width lands exactly on it
(14.64), patch tokens beat it (9.58), and a patch arm with 95 % of its tokens dropped falls
straight back to it (16.01). So the latent every one of ~120 contrasts was built on cannot
see the variable PushT is about, and it is the TOKENS rather than the word "patch"
(`2026-09-04` §7.4, `2026-09-05` §1).

Three of round 7's four experiments need a diagram:

    S1  arch-s1-tokendrop.svg   the token dose-response, INSIDE the patch family.
    S3  arch-s3-patchvote.svg   an M=5 plan-time committee whose members are patch models.
        round7-design.svg       all four experiments on one canvas, as a plan.

STYLE.  analysis/style.py, via analysis/arch_figs.py, and every primitive comes from
analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py / round6_arch_figs.py --
nothing here re-implements one.  Hues by IDENTITY, never by rank:

    amber    S1, the intervention under test
    teal     S3, the consensus family (arch_figs._consensus, round6's R5)
    slate    round7-design, because a PLAN is not yet a claim
    crimson  a falsifying outcome, and an arm the record shows destroyed
    green    a measured value that beats the constant bound

LAYOUT RULES INHERITED FROM round6_arch_figs.py, and the defect each one stops

  * Strict left-to-right dataflow on fixed row baselines; no two wires share a segment; a
    signal that legitimately fans out gets a drawn dot().
  * base() fixes every anchor, so a connector uses ONE named route.  S3 reuses round6's
    `wire_from_pred` verbatim -- the route for "this proposal reads the model and changes
    no loss" -- so the reader learns it once.  S1's stage sits inside the encoder, where
    none of the four house routes lands, so its connector runs the other way: it STARTS on
    the drop box's right edge and ends with its arrowhead on the panel's top edge, after a
    395 px vertical in the empty right margin (x = 890, clear of the dome's 868 and the
    o_{t+1} circle's 846).  Starting short and ending long is what keeps an 11 px arrowhead
    off an elbow.
  * Nothing is sized by hand where it can be measured: mbox/pill/inset/frow/strip all fit
    their own text and none may cross the panel border.
  * A NULL is never drawn like a positive.  +0.072 [-0.064, +0.207] and +0.140
    [-0.047, +0.327] both span zero, so on round7-design's evidence panel they get slate,
    a HOLLOW circle and a "NULL" chip; only +0.232 [+0.130, +0.334], which does not span
    zero, gets the teal filled marker.

THE ROUND'S NUMBERS AND WHERE EACH COMES FROM (a few are cited here as the
provenance of a claim without themselves being drawn)

    9.58 / 14.64 / 15.72 / 16.01 deg, bound 14.51 deg   2026-09-04 §7.4 (the probe table)
    n = 13 / 6 / 16 / 7 on those                        the same table
    CEM 0.520 / 0.418 / 0.357 / 0.006                   2026-09-05 §2
    +0.072 [-0.064, +0.207] at n = 12                   2026-09-05 §2 (a NULL)
    +0.140 [-0.047, +0.327] at n = 8                    2026-09-05 §4 / 2026-09-04 §7.4
    0.602 (n = 10), +0.232 [+0.130, +0.334], p = 0.0006 2026-09-03 §16.9
    +0.058 [-0.008, +0.124] vs its best member, SAME    2026-09-03 §16.9, the off-diagonal
      block -- NOT the +0.014 that §7.5 still quotes:   matrix.  §16.9 states outright that
      that one is the diagonal artefact §16.9 retracts  '+0.014 was an artefact of the diagonal'
    rho-bar +0.0010 [-0.015, +0.016] at K=1, -0.0008    2026-09-04 §7.5
    0.2599 against 1/sqrt(16) = 0.2500                  2026-09-04 §7.5
    doses 0.25 / 0.50 / 0.75 / 0.90, 8 seeds each       scripts/run_campaign.sh wave26
    seeds 0-2 (S2), 11-14 (S4), blocks 3-12 (S3)        2026-09-05 §3, scripts/plan_patch_vote.sh
    P = 256, keep = max(1, round(P(1-d))), train only   models/vit_encoder.py:118-131,
                                                        conf/encoder/vit_scratch_patch.yaml
                                                        (224 / 14 -> a 16x16 grid)

ARITHMETIC THIS MODULE DOES ITSELF (nothing else is computed here):

    the tokens-kept row of the ladder, keep(d) = max(1, round(256 (1 - d))) evaluated at
    the six doses -> 256 / 192 / 128 / 64 / 26 / 13, which is the encoder's own formula;
    n = 8 + 4 = 12 for S4, whose control takes the same four seeds; for S2, n = 12 + 3 = 15
    is drawn as CONDITIONAL, because collect_evals(scheme='fixed') has LpWM-ltv at seeds
    3-15 only and its 0-2 evals are pre-49a3e55, which that module rules out of a pairing;
    32 + 3 + 8 = 43 training runs in the design canvas's subtitle.

Usage:  python analysis/round7_arch_figs.py --out diary/assets/2026-09-05
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MSERED, MUTED, WHITE,
    arrow, base, box, caret, circle, domeup, poly, sub, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, BAR, DOT, IMPLIES, MINUS, NDASH, RHO, SQ, TIMES,
    apoly, aw, badge, dot, emit, fit_size, hpoly, mbox, opnode, pill, strip,
)
from analysis.round5_figs_obj import (  # noqa: E402,F401
    frow, inset, mmid, mrun, panel, rpill, run_w,
)
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PIX, PW, PX, PY, SHIFT, W, axes, curve, ellipsis_v, lin, mark, out_emit,
    wire_from_pred,
)

DEG = "°"
IN_ = "∈"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-05")

# The encoder's own numbers: conf/encoder/vit_scratch_patch.yaml is patch_size 14 at
# img_size 224, so the grid is 16x16 and P = 256 exactly -- which is why the ladder can
# draw one cell per token instead of a stand-in for one.
P_TOKENS = 256
DOSES = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95)


def keep_for(d):
    """models/vit_encoder.py:129, verbatim: keep = max(1, int(round(P * (1 - drop))))."""
    return max(1, int(round(P_TOKENS * (1.0 - d))))


# --- local primitives --------------------------------------------------------------
def cbadge(cx, y, label, key, size=12.5):
    """badge(), centred on cx.  badge() anchors start or end only, and a ladder rung's
    chip has to sit on its own rung's centre line or the column stops reading as a column.
    """
    w = text_width(label, size, "bold") + 22
    return badge(round(cx - w / 2, 1), y, label, key, size)


def tokengrid(cx, y, keep, key, total=P_TOKENS, cell=3.9, gap=0.8, seed=11):
    """P patch tokens as a 16x16 grid with `keep` of them kept.

    One cell is one token, because 16*16 == 256 == the encoder's real num_patches: the
    picture IS the dose.  The kept set is a fixed pseudo-random draw because
    _drop_tokens draws a random keep-set (one per call, shared across the clip), so a
    contiguous block would misrepresent what the arm does.
    """
    n = int(round(math.sqrt(total)))
    step = cell + gap
    x0 = round(cx - (n * step - gap) / 2.0, 1)
    kept = set(random.Random(seed).sample(range(total), keep))
    s = (f'<rect x="{x0 - 3}" y="{y - 3}" width="{n * step - gap + 6:.1f}" '
         f'height="{n * step - gap + 6:.1f}" rx="2" fill="{WHITE}" stroke="{DIM}" '
         f'stroke-width="1"/>\n')
    for i in range(total):
        r, c = divmod(i, n)
        s += (f'<rect x="{x0 + c * step:.1f}" y="{y + r * step:.1f}" width="{cell}" '
              f'height="{cell}" fill="{ACCENT[key] if i in kept else GRID}"/>\n')
    return s


def slots(x, y, states, key, cell=14, gap=3):
    """A seed row: one square per seed, so "what this experiment buys" is countable.

      'f'  a seed on record on the instrument the pairing uses (filled slate)
      'n'  a seed this round launches (hollow, dashed, in the accent)
      'x'  a seed whose only eval predates 49a3e55 and so cannot enter a paired test
           (hollow, SOLID slate).  Without this state the S2 row drew LpWM-ltv as 16
           usable seeds and claimed a pairing that collect_evals(scheme="fixed") does
           not support -- the arm has fixed-scheme evals at 3-15 only.
    """
    s = ""
    for i, st in enumerate(states):
        cx = x + i * (cell + gap)
        if st == "f":
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{ACCENT["slate"]}" fill-opacity="0.55" '
                  f'stroke="{ACCENT["slate"]}" stroke-width="1"/>\n')
        elif st == "x":
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{WHITE}" stroke="{ACCENT["slate"]}" stroke-width="1.4"/>\n')
        else:
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{WHITE}" stroke="{ACCENT[key]}" stroke-width="1.6" '
                  f'stroke-dasharray="3,2"/>\n')
    return s


def interval(y, lo, mid, hi, key, hollow, xmap, h=7):
    """One confidence interval on a shared axis.

    The whole point of this primitive: a null and a positive are drawn DIFFERENTLY.  A
    positive gets its identity hue and a filled marker; an interval that spans zero gets
    slate and a hollow circle, whatever its point estimate looks like."""
    c = ACCENT[key]
    xl, xm, xh = xmap(lo), xmap(mid), xmap(hi)
    s = (f'<path d="M{xl:.1f},{y} L{xh:.1f},{y} M{xl:.1f},{y - h} L{xl:.1f},{y + h} '
         f'M{xh:.1f},{y - h} L{xh:.1f},{y + h}" stroke="{c}" stroke-width="2" '
         f'fill="none" stroke-linecap="round"/>\n')
    return s + mark(round(xm, 1), y, key, r=6.5, fill=(WHITE if hollow else None))


def chiprow(x, y, w, text, key, size=13.5):
    """The cost bar at the foot of a design card: a tinted rule, not a saturated fill."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="32" rx="5" fill="{c}" '
         f'fill-opacity="0.12" stroke="{c}" stroke-width="1.5"/>\n')
    return s + txt(x + w / 2, y + 21, text, size, fill=c, weight="bold")


# ============================================================ S1: the token dose-response
def s1_tokendrop():
    """feature=patch throughout, TOKEN_DROP swept.  The design point is what the contrast
    does NOT do: it never compares patch against cls, so the feature cannot confound it."""
    K = "amber"
    b = base("(S1) PiWM-tok -- the token dose-response, inside the patch family  [PLANNED]",
             enc_sub="feature = patch", skip=("enc",))
    # The encoder redrawn 20px lower with the drop stage straddling its output wire, and
    # both legs of that wire visible -- arch_figs.build()'s PiWM-drop95 construction, at
    # the ladder's own symbol d.  BOTH encoders carry it: in a JEPA the encoder's output
    # is also the target, which is the mechanism the 0.95 endpoint already demonstrated.
    for x in (64, 756):
        b += domeup(x, 540, 112, 104, "Enc <tspan font-style='italic'>f</tspan>",
                    sub="feature = patch")
    for x in (120, 812):
        b += arrow(x, 540, x, 524)
        b += box(x - 60, 488, 120, 30, "TOKEN_DROP", fill=ACCENT_FILL[K],
                 stroke=ACCENT[K], sw=2.0, size=15, dash="6,4")
        b += arrow(x, 488, x, 470)
    b += txt(712, 508, "d  =  0.25 , 0.50 , 0.75 , 0.90 " + NDASH + "  the four new rungs",
             14, anchor="end", fill=ACCENT[K], weight="bold")
    # the connector runs model -> panel, so the long leg carries the arrowhead.  It has no
    # label of its own: the caption above already names d, and a second label here would
    # sit in the only band the title's [PLANNED] chip occupies.
    b += poly([(872, 503), (890, 503), (890, PY)], color=ACCENT[K], w=1.8, dash="6,4")

    PH = 652
    s = panel(PX, PY, PW, PH, K,
              "module:  TOKEN_DROP = d  on the patch tokens " + NDASH
              + "  a dose ladder INSIDE the patch family")

    # -- row A: where the tokens go, and what survives the stage
    s += circle(88, 1010, 24, sub("o", "t", 11), size=15)
    s += aw(112, 1010, 134, 1010, K)
    s += mbox(138, 986, 132, 48, "patch embed", K, size=15, sub_="1 CLS + 256 patch")
    s += aw(270, 1010, 292, 1010, K)
    s += mbox(296, 986, 156, 48, "keep-set", K, size=16,
              sub_="one per call, train only")
    s += aw(452, 1010, 474, 1010, K)
    s += pill(516, 1010, "1 + keep", K, rx=42, ry=19, fill=WHITE)
    s += aw(558, 1010, 580, 1010, K)
    s += mbox(584, 986, 136, 48, "Transformer", K, size=15, sub_="12 blocks")
    s += aw(720, 1010, 742, 1010, K)
    s += pill(800, 1010, sub("z", "t", 11), K, rx=34, ry=19)
    s += txt(800, 1050, "keep tokens", 12, fill=MUTED)
    s += txt(470, 1068, "the ladder never compares patch against cls, so the FEATURE "
             "cannot confound it", 13.5, fill=ACCENT[K], weight="bold")
    s += txt(470, 1088, "and the encoder's output is ALSO the JEPA target " + NDASH
             + "  a dropped token leaves both sides of the loss", 13,
             fill=ACCENT["crit"], weight="bold")

    # -- row B: the ladder itself, as a table with its own row labels
    s += inset(60, 1104, 816, 292, "the dose ladder " + NDASH
               + "  six rungs, and the two endpoints are already on record", K,
               note="every rung is feature = patch and every rung's objective is "
                    "identical: only d changes")
    LAB = 166
    for lab, ly in (("TOKEN_DROP  d", 1182), ("tokens kept", 1286),
                    ("arm", 1314), ("status", 1338), ("angular error", 1360),
                    ("CEM mean", 1380)):
        s += txt(LAB, ly, lab, 11.5, anchor="end", fill=MUTED)
    # the two n's differ and are both carried: the probe is over CHECKPOINTS and the CEM
    # mean over evaluated seeds, and columns has 13 of the first and 12 of the second.
    rungs = (
        (0.00, "columns", "slate", "9.58" + DEG + "  n=13", "green", "0.418  n=12", INK,
         ("already run", MUTED)),
        (0.25, "tok25", K, None, None, None, None, ("8 new seeds", ACCENT[K])),
        (0.50, "tok50", K, None, None, None, None, ("8 new seeds", ACCENT[K])),
        (0.75, "tok75", K, None, None, None, None, ("8 new seeds", ACCENT[K])),
        (0.90, "tok90", K, None, None, None, None, ("8 new seeds", ACCENT[K])),
        (0.95, "drop95", "slate", "16.01" + DEG + "  n=7", "crit", "0.006  n=7",
         ACCENT["crit"], ("already run", MUTED)),
    )
    for i, (d, arm, gk, ang, angk, cem, cemc, (seeds, seedc)) in enumerate(rungs):
        cx = 233 + i * 114
        s += txt(cx, 1182, f"{d:.2f}", 17, weight="bold")
        s += tokengrid(cx, 1192, keep_for(d), gk, seed=11 + i)
        s += txt(cx, 1286, str(keep_for(d)), 14, weight="bold",
                 fill=INK if ang else ACCENT[K])
        s += cbadge(cx, 1298, arm, gk, size=11.5)
        s += txt(cx, 1338, seeds, 12, fill=seedc, weight="bold" if ang is None else
                 "normal")
        s += txt(cx, 1360, ang if ang else MINUS, 13 if ang else 14,
                 fill=ACCENT[angk] if ang else MUTED, weight="bold")
        s += txt(cx, 1380, cem if cem else MINUS, 13 if cem else 14,
                 fill=cemc if cem else MUTED, weight="bold")

    s += frow(1428, [
        (["keep  =  max( 1 , round( P " + DOT + " ( 1 " + MINUS + " d ) ) ) ,   P = 256"],
         INK, "normal"),
        (["one keep-set per call, shared across the clip"], INK, "normal"),
        (["models/vit_encoder.py:118" + NDASH + "131"], MUTED, "normal", 0.9)])
    s += frow(1452, [
        ("train only: at plan time every arm still encodes all 256 tokens", INK,
         "normal"),
        ("green = below the 14.51" + DEG + " constant bound, crimson = at or above it",
         MUTED, "normal", 0.92)])
    s += frow(1476, [
        ("falsifier:  CEM flat across the ladder while the angular error climbs",
         ACCENT["crit"], "bold"),
        ("then orientation decodability is a fourth screened quantity that dies under "
         "intervention", MUTED, "normal", 0.92)])
    s += strip(PIN, 1492, [
        ("d = 0.00   PiWM-columns", "9.58" + DEG + " , CEM 0.418", False),
        ("d = 0.95   PiWM-drop95", "16.01" + DEG + " , CEM 0.006", True),
        ("the constant bound", "14.51" + DEG, False),
        ("new", "4 doses " + TIMES + " 8 seeds = 32 runs", False)], K, cw=190)
    return b + s, PY + PH + 40


# ================================================================= S3: the patch committee
def s3_patchvote():
    """The campaign's two leads, composed: a committee whose members can see the variable.
    Drawn as a PREDICTION with its falsifier, because that is all it is until it runs."""
    K = "green"                       # teal -- the consensus family
    b = base("(S3) PiWM-patchvote5 -- a plan-time committee of PATCH models  [PLANNED]",
             enc_sub="feature = patch")
    b += wire_from_pred(K, "eval only " + NDASH + " no term in the loss")

    PH = 770
    s = panel(PX, PY, PW, PH, K,
              "module:  M = 5 PiWM-columns checkpoints, per-member rank, MEDIAN "
              + ARROWC + " CEM's argsort( loss )")

    # -- row A: five members, each in its OWN latent basis, each on its own corridor
    BX, BY, BW, BH = 552, 973, 170, 142
    cy = BY + BH / 2
    for i, (y, seed) in enumerate(((992, "3"), (1044, "4"), (1096, "5"))):
        s += mbox(64, y - 22, 138, 44, "member " + str(i + 1), K, size=15,
                  sub_="PiWM-columns  s" + seed)
        s += aw(202, y, 222, y, K)
        s += mbox(226, y - 22, 108, 44, sub("MSE", str(i + 1), 11), K, size=15)
        s += aw(334, y, 356, y, K)
        s += mbox(360, y - 22, 148, 44, "rank", K, size=15,
                  sub_="within the CEM batch")
        # ONE corridor, not three.  The staggered corridors (524/534/544) left member 3
        # with a 2-unit final leg carrying a 9-unit arrowhead: the head floated clear of
        # the MEDIAN box and read as a broken wire.  The three vertical legs occupy
        # disjoint y-spans (992-1003, none, 1096-1085), so they cannot cross anyway.
        ey = (BY + 30, cy, BY + BH - 30)[i]
        s += apoly([(508, y), (522, y), (522, ey), (BX - 6, ey)], K)
    s += ellipsis_v(133, 1132, "slate")
    s += txt(168, 1166, "M = 5 members " + NDASH + "  blocks 3" + NDASH
             + "7 take seeds 3" + NDASH + "7, blocks 8" + NDASH + "12 take seeds 8"
             + NDASH + "12, exactly as the cls committees did", 13.5, anchor="start",
             fill=MUTED)
    s += (f'<rect x="{BX}" y="{BY}" width="{BW}" height="{BH}" rx="9" '
          f'fill="{ACCENT_FILL[K]}" stroke="{ACCENT[K]}" stroke-width="2.0"/>\n')
    s += txt(BX + BW / 2, BY + 34, "MEDIAN", 19, fill=ACCENT[K], weight="bold")
    s += txt(BX + BW / 2, BY + 58, "of the per-member", 14, fill=ACCENT[K])
    s += txt(BX + BW / 2, BY + 78, "ranks", 14, fill=ACCENT[K])
    s += txt(BX + BW / 2, BY + 104, "at PLAN time", 12.5, fill=MUTED)
    s += txt(BX + BW / 2, BY + 124, "no training at all", 12.5, fill=MUTED)
    s += aw(BX + BW, cy, 744, cy, K, w=2.0)
    s += pill(792, cy, "CEM", K, rx=46, ry=22)
    s += txt(792, cy + 46, "argsort( loss )[ : topk ]", 12, fill=MUTED)

    # -- the two leads this composes, side by side on one rule
    s += inset(60, 1180, 816, 128, "S3 composes the campaign's only two leads", K)
    s += (f'<path d="M468,1214 L468,1296" stroke="{ACCENT[K]}" stroke-width="1.2" '
          f'stroke-opacity="0.45"/>\n')
    for x0, head, hk, lines in (
            (76, "LEAD 1 " + DOT + " consensus, the one positive", K,
             ("M = 5 median committee:  0.602   ( n = 10 )",
              "vs a compute-matched single model:  +0.232 [+0.130, +0.334]",
              # the implication is spelled out under the prediction plot below, so the
              # line carries only the number.  With it appended, this line ran past the
              # inset's mid-rule and into LEAD 2's third line.
              "vs its BEST member, same block:  +0.058 [" + MINUS
              + "0.008, +0.124], a null")),
            (486, "LEAD 2 " + DOT + " the representation", "amber",
             ("PiWM-columns (patch):  9.58" + DEG + ", the only arm below the bound",
              "LpWM-ltv (cls):  15.72" + DEG + ", worse than the 14.51" + DEG + " bound",
              "so a patch member SEES the variable a cls member cannot"))):
        s += txt(x0, 1234, head, 13, anchor="start", fill=ACCENT[hk], weight="bold")
        for j, ln in enumerate(lines):
            s += txt(x0, 1256 + j * 19, ln, fit_size(ln, 12, 384, floor=10.5),
                     anchor="start", fill=INK)

    # -- the prediction, and the single observation that refutes it
    s += inset(60, 1320, 816, 210, "the prediction, and what refutes it", K,
               note="solid = measured;  hollow = not yet run")
    AX0, AX1, AY0, AY1 = 170, 520, 1388, 1482
    s += axes(AX0, AX1, AY0, AY1)
    s += txt(AX0 - 4, AY0 - 8, "CEM success", 12, anchor="start", fill=MUTED)

    def vy(v):
        return lin(v, 0.30, 0.72, AY1, AY0)

    s += (f'<path d="M{AX0},{vy(0.602):.1f} L{AX1},{vy(0.602):.1f}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.3" stroke-dasharray="5,4" '
          f'stroke-opacity="0.8"/>\n')
    s += txt(AX0 + 8, vy(0.602) - 8, "the cls committee's 0.602", 11, anchor="start",
             fill=ACCENT[K], weight="bold")
    for cxx, val, lab, sub_lab in ((210, 0.357, "LpWM-ltv", "single, cls"),
                                   (288, 0.418, "PiWM-columns", "single, patch"),
                                   (366, 0.602, "vote5-median", "committee, cls")):
        s += mark(cxx, vy(val), "slate" if val < 0.5 else K, r=5.5)
        if val < 0.5:
            s += txt(cxx, vy(val) - 12, f"{val:.3f}", 11.5, weight="bold")
        s += txt(cxx, AY1 + 20, lab, fit_size(lab, 10.5, 76, floor=9), fill=MUTED)
        s += txt(cxx, AY1 + 34, sub_lab, fit_size(sub_lab, 10.5, 76, floor=9), fill=MUTED)
    s += (f'<path d="M414,{AY0} L414,{AY1}" stroke="{MUTED}" stroke-width="1.1" '
          f'stroke-dasharray="4,4" stroke-opacity="0.7"/>\n')
    s += (f'<rect x="436" y="{AY0 + 2}" width="48" height="{vy(0.602) - AY0 - 5:.1f}" '
          f'rx="3" fill="{ACCENT[K]}" fill-opacity="0.15" stroke="{ACCENT[K]}" '
          f'stroke-width="1.3" stroke-dasharray="5,4"/>\n')
    s += caret(460, vy(0.602) - 14, w=15, h=7, color=ACCENT[K])
    s += mark(460, vy(0.602), "crit", r=5.5, fill=WHITE)
    s += txt(460, AY1 + 20, "S3", 10.5, fill=MUTED)
    s += txt(460, AY1 + 34, "committee, patch", fit_size("committee, patch", 10.5, 86,
                                                         floor=9), fill=MUTED)
    for y0, kk, head, l1, l2 in (
            (1398, K, "lands ABOVE the cls committee",
             IMPLIES + "  the cap IS the best member's quality,",
             "and the representation is the lever"),
            (1456, "crit", "ties at 0.602",
             IMPLIES + "  the cap is NOT member quality, and the",
             "measured mechanism for the one positive is wrong")):
        s += txt(548, y0, head, 13, anchor="start", fill=ACCENT[kk], weight="bold")
        s += txt(548, y0 + 19, l1, 12.5, anchor="start", fill=ACCENT[kk])
        s += txt(548, y0 + 37, l2, 12.5, anchor="start", fill=ACCENT[kk])

    s += frow(1560, [
        (["Var" + sub("", "ens", 12) + " / Var" + sub("", "member", 12) + "  =  ",
          ("bar", "ρ"), "  +  ( 1 " + MINUS + " ", ("bar", "ρ"), " ) / M"],
         INK, "normal"),
        ([("bar", "ρ"), "  =  +0.0010 [" + MINUS + "0.015, +0.016] at K = 1 ,   "
          + MINUS + "0.0008 at K = 5"], INK, "normal"),
        (["so the cap is not correlation"], MUTED, "normal", 0.92)])
    s += frow(1586, [
        ("the median combines per-member RANKS, not MSEs " + NDASH
         + " scale-free, so members need not share a latent basis", INK, "normal"),
        ("nothing here enters a training loss", MUTED, "normal", 0.95)])
    s += strip(PIN, 1602, [
        ("cls committee, M = 5 median", "0.602   ( n = 10 )", False),
        ("vs a compute-matched single", "+0.232 [+0.130, +0.334]", False),
        ("vs best member, same block", "+0.058 [" + MINUS + "0.008, +0.124]  null", False),
        ("cost", "10 evals, no training", False)], K, cw=170)
    return b + s, PY + PH + 40


# ==================================================== the four experiments, as a PLAN
DW, DH = 940, 1164
QW, QH = 438, 372
QX = (26, 476)
QY = (104, 496)


def card(i, j, tag, title, spec):
    """One experiment's frame: a slate chip, the title, the arm spec that proves it, and
    a rule.  Slate for all four, because a plan is not yet a claim -- the only hue inside
    a card is crimson, on the outcome that would kill it."""
    x, y, c = QX[j], QY[i], ACCENT["slate"]
    s = (f'<rect x="{x}" y="{y}" width="{QW}" height="{QH}" rx="10" '
         f'fill="{ACCENT_FILL["slate"]}" fill-opacity="0.35" stroke="{c}" '
         f'stroke-width="2.2"/>\n')
    s += f'<rect x="{x + 14}" y="{y + 12}" width="32" height="24" rx="5" fill="{c}"/>\n'
    s += txt(x + 30, y + 30, tag, 15, fill=WHITE, weight="bold")
    s += txt(x + 54, y + 30, title, fit_size(title, 15, QW - 74, floor=13),
             anchor="start", fill=c, weight="bold")
    s += txt(x + 54, y + 48, spec, fit_size(spec, 11.5, QW - 74, floor=10),
             anchor="start", fill=MUTED, style="italic")
    s += (f'<path d="M{x + 14},{y + 58} L{x + QW - 14},{y + 58}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def card_text(i, j, tests, falsifies, cost):
    """The three fields every card carries, on identical baselines across all four so the
    cards can be read down a column instead of one at a time."""
    x, y = QX[j], QY[i]
    s = txt(x + 20, y + 176, "WHAT IT TESTS", 11, anchor="start", fill=ACCENT["slate"],
            weight="bold")
    for k, ln in enumerate(tests):
        s += txt(x + 20, y + 196 + k * 19, ln, fit_size(ln, 12.5, QW - 40, floor=10.5),
                 anchor="start", fill=INK)
    s += txt(x + 20, y + 264, "WHAT WOULD FALSIFY IT", 11, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    for k, ln in enumerate(falsifies):
        s += txt(x + 20, y + 284 + k * 19, ln, fit_size(ln, 12.5, QW - 40, floor=10.5),
                 anchor="start", fill=ACCENT["crit"])
    return s + chiprow(x + 18, y + QH - 46, QW - 36, cost, "slate")


def round7_design():
    s = txt(DW / 2, 46, "Round 7 " + NDASH + " four experiments on the REPRESENTATION",
            23, weight="bold")
    s += txt(DW / 2, 70, "a PLAN, stated before any of it has run: 43 training runs and "
             "10 evals; every arm is feature = patch and nothing changes the objective",
             13.5, fill=MUTED)
    s += (f'<path d="M26,86 L914,86" stroke="{ACCENT["slate"]}" stroke-width="1.2" '
          f'stroke-opacity="0.5"/>\n')

    # -- S1: the ladder, with its two endpoints already standing
    x, y = QX[0], QY[0]
    s += card(0, 0, "S1", "the token dose-response",
              "PiWM-tok25 / 50 / 75 / 90 " + DOT + " wave26 " + DOT + " 8 seeds each")
    for k, d in enumerate(DOSES):
        cxx = x + 44 + k * 60
        exists = d in (0.00, 0.95)
        kk = "slate" if exists else "amber"
        dash = "" if exists else ' stroke-dasharray="5,3"'
        s += (f'<rect x="{cxx - 26}" y="{y + 78}" width="52" height="26" rx="5" '
              f'fill="{ACCENT_FILL[kk]}" stroke="{ACCENT[kk]}" stroke-width="1.5"'
              f'{dash}/>\n')
        s += txt(cxx, y + 96, f"{d:.2f}", 13, fill=ACCENT[kk], weight="bold")
        s += txt(cxx, y + 120, "exists" if exists else "new", 11,
                 fill=MUTED if exists else ACCENT["amber"],
                 weight="normal" if exists else "bold")
    s += txt(x + 44, y + 142, "PiWM-columns", 10.5, fill=MUTED)
    s += txt(x + 344, y + 142, "PiWM-drop95", 10.5, fill=MUTED)
    s += card_text(0, 0,
                   ("does removing tokens degrade orientation AND planning together?",
                    "a ladder INSIDE the patch family: it never compares patch",
                    "against cls, so the feature cannot confound it"),
                   ("CEM flat across the ladder while the angular error climbs",
                    "then orientation decodability is a fourth screened quantity",
                    "that does not survive being intervened on"),
                   "COST " + DOT + "  32 training runs")

    # -- S2: three seeds, drawn as three seeds
    x, y = QX[1], QY[0]
    s += card(0, 1, "S2", "power the single-factor contrast",
              "PiWM-columns at seeds 0" + NDASH + "2 " + DOT
              + " more seeds of an EXISTING arm, not a new one")
    # LpWM-ltv's seeds 0-2 are drawn 'x', not 'f': collect_evals(scheme="fixed") has the
    # baseline at 3-15 only, and its 0-2 evals exist solely on the pre-49a3e55 instrument
    # that module calls invalid for a paired test.  Drawing them filled asserted a pairing
    # the archive cannot supply.
    for k, (lab, st) in enumerate((("LpWM-ltv", "x" * 3 + "f" * 13),
                                   ("PiWM-columns", "n" * 3 + "f" * 13))):
        s += txt(x + 118, y + 92 + k * 26, lab, 11.5, anchor="end", fill=MUTED)
        s += slots(x + 126, y + 80 + k * 26, st, "amber")
    for k, cap in enumerate((
            # "open square", not "hollow slate": the evidence panel at the foot of this
            # canvas already spends "hollow slate marker" on an interval that spans zero,
            # and one canvas may not give one phrase two meanings.
            "seeds 0" + NDASH + "15 " + DOT + "  dashed = trained by S2 " + DOT
            + "  open square = no fixed-scheme eval",
            "paired n = 12 " + ARROWC + " 15 only if the baseline's 0" + NDASH
            + "2 are re-evaluated too (3 evals)")):
        s += txt(x + 20, y + 140 + k * 16, cap, fit_size(cap, 10.5, QW - 40, floor=9.2),
                 anchor="start", fill=MUTED)
    s += card_text(0, 1,
                   ("PiWM-columns vs LpWM-ltv, arm specs identical but encoder.feature",
                    "today: +0.072 [" + MINUS + "0.064, +0.207] at n = 12, a NULL "
                    "that leans positive",
                    "and the largest n of any treated arm in the campaign"),
                   ("the interval still spans zero at n = 15",
                    "then patch tokens buy orientation without buying planning:",
                    "a latent that carries the variable, a planner that cannot use it"),
                   "COST " + DOT + "  3 training runs")

    # -- S3: five members, one median, no training
    x, y = QX[0], QY[1]
    s += card(1, 0, "S3", "patch consensus",
              "M = 5 PiWM-columns checkpoints " + DOT + " rule = median " + DOT
              + " NO training")
    for k in range(5):
        s += mark(x + 40 + k * 26, y + 100, "green", r=8)
    s += txt(x + 92, y + 130, "5 members", 11, fill=MUTED)
    s += aw(x + 178, y + 100, x + 204, y + 100, "green")
    s += mbox(x + 208, y + 78, 96, 44, "median", "green", size=15, sub_="of ranks")
    s += aw(x + 304, y + 100, x + 330, y + 100, "green")
    s += pill(x + 372, y + 100, "CEM", "green", rx=38, ry=20)
    cap = ("checkpoints that already exist " + NDASH + "  no GPU-hour of training")
    s += txt(x + QW / 2, y + 148, cap, fit_size(cap, 11.5, QW - 40, floor=10),
             fill=MUTED)
    s += card_text(1, 0,
                   ("is the consensus ceiling set by the quality of the BEST member?",
                    "composes the campaign's only two leads: the one positive",
                    "(+0.232 over a compute-matched single) and the representation"),
                   ("the patch committee lands at the same 0.602 as the cls one",
                    "then the ceiling is not member quality, and the measured",
                    "mechanism behind the campaign's one positive is wrong"),
                   "COST " + DOT + "  10 evals, 0 training runs")

    # -- S4: the arm and its control take the SAME seeds
    x, y = QX[1], QY[1]
    s += card(1, 1, "S4", "patchdecode at more seeds",
              "PiWM-patchdecode and its -detach control " + DOT + " seeds 11"
              + NDASH + "14")
    for k, lab in enumerate(("patchdecode", "-detach control")):
        s += txt(x + 148, y + 92 + k * 26, lab, 11.5, anchor="end", fill=MUTED)
        s += slots(x + 156, y + 80 + k * 26, "f" * 8 + "n" * 4, "amber", cell=16)
    cap = ("the control takes the SAME four seeds, or the pairing does not widen")
    s += txt(x + 20, y + 148, cap, fit_size(cap, 11.5, QW - 40, floor=10),
             anchor="start", fill=MUTED)
    s += card_text(1, 1,
                   ("the campaign's highest non-consensus arm mean: 0.520 vs 0.357",
                    "against its own detach control: +0.140 [" + MINUS
                    + "0.047, +0.327] at n = 8,",
                    "a NULL, and the second of the two patch arms that lean positive"),
                   ("the interval still spans zero at n = 12",
                    "then the highest arm mean in the campaign is seed noise",
                    "and patchdecode joins the other ~120 contrasts"),
                   "COST " + DOT + "  8 training runs")

    # -- the record as it stands, drawn so a null cannot be mistaken for a positive
    s += inset(26, 888, 888, 234, "what the record says TODAY " + NDASH
               + "  and how each result is drawn", "slate",
               note="filled marker = an interval clear of zero;  hollow slate marker = "
                    "an interval that spans zero")
    LO, HI, GX0, GX1 = -0.10, 0.36, 300, 820

    def xm(v):
        return lin(v, LO, HI, GX0, GX1)

    s += (f'<path d="M{GX0},966 L{GX1},966" stroke="{MUTED}" stroke-width="1.1"/>\n')
    for v in (-0.1, 0.0, 0.1, 0.2, 0.3):
        s += (f'<path d="M{xm(v):.1f},960 L{xm(v):.1f},966" stroke="{MUTED}" '
              f'stroke-width="1.1"/>\n')
        s += txt(xm(v), 956, ("+" if v > 0 else (MINUS if v < 0 else "")) + f"{abs(v):.1f}",
                 10.5, fill=MUTED)
    s += (f'<path d="M{xm(0):.1f},970 L{xm(0):.1f},1106" stroke="{ACCENT["slate"]}" '
          f'stroke-width="1.3" stroke-dasharray="5,4" stroke-opacity="0.8"/>\n')
    for k, (lab, val, lo, mid, hi, key, hollow, chip) in enumerate((
            ("M = 5 consensus vs a compute-matched single model   ( n = 10 )",
             "+0.232 [+0.130, +0.334]", 0.130, 0.232, 0.334, "green", False, "POSITIVE"),
            ("PiWM-patchdecode vs its own detach control   ( n = 8 )",
             "+0.140 [" + MINUS + "0.047, +0.327]", -0.047, 0.140, 0.327, "slate", True,
             "NULL"),
            ("PiWM-columns vs LpWM-ltv, the single-factor contrast   ( n = 12 )",
             "+0.072 [" + MINUS + "0.064, +0.207]", -0.064, 0.072, 0.207, "slate", True,
             "NULL"))):
        yl = 990 + k * 50
        s += txt(48, yl, lab, fit_size(lab, 11.5, 330, floor=10), anchor="start",
                 fill=INK)
        s += txt(880, yl, val, 12, anchor="end", fill=ACCENT[key], weight="bold")
        s += interval(yl + 16, lo, mid, hi, key, hollow, xm)
        s += badge(802, yl + 6, chip, key, size=11)
    s += txt(DW / 2, 1146, "round 7 changes the representation, and only the "
             "representation", 13.5, weight="bold")
    return s


FIGURES = (
    ("arch-s1-tokendrop.svg", s1_tokendrop, None),
    ("arch-s3-patchvote.svg", s3_patchvote, None),
    ("round7-design.svg", round7_design, (DW, DH)),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn, wh in FIGURES:
        if wh is None:
            body, h = fn()
            print("  wrote", os.path.join(a.out, out_emit(name, body, a.out, h)))
        else:
            print("  wrote", os.path.join(a.out, emit(name, fn(), a.out, *wh)))


if __name__ == "__main__":
    main()
