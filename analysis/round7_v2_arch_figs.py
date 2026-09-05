"""Round 7, REDESIGNED: the two design faults found in round 7 itself, and their fixes.

    python analysis/round7_v2_arch_figs.py --out diary/assets/2026-09-05
    -> arch-aligned-grid.svg    the dimension-matched cls-vs-patch grid
    -> arch-loo-consensus.svg   leave-one-out committees, with no member-count knob

Six rounds changed the training OBJECTIVE and produced exactly one positive (plan-time
consensus).  A success-rate reanalysis then reframed the whole campaign, and reading round 7's
own design against it turned up two faults, both real, both fixed here.  These are the two
figures that carry the fixes; `analysis/round7_arch_figs.py` still holds the ORIGINAL round-7
plan and is deliberately not edited, because that file is a dated statement of what was
believed on the day it was built.

    G1  arch-aligned-grid.svg    "PiWM-columns vs LpWM-ltv" changed the FEATURE and a 256x
                                 CAPACITY at the same time.  The fix is a grid matched on
                                 total latent dimension.  AMBER: the intervention under test.
    G2  arch-loo-consensus.svg   consensus had TWO hyperparameters (M and the rule) and the
                                 audit showed they measured one thing.  The fix is
                                 leave-one-out: M is set by the data, the rule is fixed.
                                 TEAL: the consensus family, as in round5/6/7's vote figures.

STYLE.  analysis/style.py, through analysis/arch_figs.py.  Every primitive is imported from
analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py / round6_arch_figs.py /
round7_arch_figs.py -- nothing here re-implements one, and no existing module is edited (eight
other figure modules import them).  Hues by IDENTITY, never by rank:

    amber    the intervention under test -- G1's grid, and every newly launched cell
    teal     the consensus family -- G2's committees (arch_figs._consensus, round6 R5,
             round7 S3 all use this key)
    crimson  the DEFECT: the 256x capacity gap, the two-knob sweep, the two committees
             that ten "seeds" really were
    slate    neutral -- a cell already on the record, an episode block, a NULL
    green    reserved here for nothing; no arm owns it in these two figures

LAYOUT RULES INHERITED FROM round6_arch_figs.py, and the defect each one stops

  * Strict left-to-right dataflow on fixed row baselines; no two wires share a segment; a
    signal that legitimately fans out gets a drawn dot().
  * base() fixes every anchor, so a connector uses ONE named route.  G2 reuses round6's
    `wire_from_pred` verbatim -- the route for "this proposal reads the model and changes no
    loss" -- so the reader learns it once.  G1's knobs (`encoder.feature`, `encoder.proj_dim`)
    live INSIDE the encoder, where none of the house routes lands, so it reuses round7 S1's
    encoder route instead: start on the dome's right edge, run the empty right margin at
    x = 890, and land the arrowhead on the panel's top edge.  Both routes pass the title's
    baseline at y = 872, so both titles are WIDTH-CHECKED against the corridor (`_check_title`)
    rather than trusted -- a full-length title with a status chip is 945 units wide and would
    have run straight through the wire.
  * Nothing is sized by hand where it can be measured: mbox / pill / inset / frow / strip all
    fit their own text and none may cross the panel border.
  * A NULL is never drawn like a positive.  +0.072 [-0.064, +0.207] spans zero, so it gets
    slate, a HOLLOW marker and the word NULL -- the same encoding round7-design.svg uses.
  * `frow()` shrinks its gap and then its type, but once it reaches its 12.0 pt floor it stops
    and OVERFLOWS SILENTLY.  G1's first footer line held three clauses and ran 19 units past
    PIX at the floor.  So a footer line here carries at most two clauses, and the audit checks
    the result rather than the intent.
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of every
    <text> element from the same Helvetica metrics the layout used (arch_figs.text_width), and
    reports any pair that intersects, anything that leaves the canvas, and anything that
    crosses the border of the panel or inset it sits in (every frame registers itself through
    the `frame()` / `pane()` wrappers).  The frame check is not redundant with the canvas
    check: G2's two crimson conclusions were centred on the right half of their inset and ran
    57 units past its border while still sitting comfortably inside the canvas.  It is the SVG
    counterpart of analysis/round7_figs.py's audit(), and it exists for the same reason -- at
    940 x 1700 units "nothing overlaps" is not checkable by eye, and it found four real
    defects in these two figures that a read of the rendered PNG had already missed.

Usage:  python analysis/round7_v2_arch_figs.py --out diary/assets/2026-09-05 [--png]
        `--png` rasterises beside each SVG, for the read-back; the audit runs either way and
        main() exits non-zero if it has anything to say.

THE NUMBERS, AND WHERE EACH ONE COMES FROM.  Nothing here is invented; nothing is computed
from data except the two seed counts and the two arithmetic identities noted at the end.

  G1
    num_patches x proj_dim vs proj_dim; 256 x 384 = 98304 against 384    2026-09-05 2c
    the 256x gap, and LpWM-ltv-d2048 at 5.3x as the only width control   2026-09-05 1, 2c
    +0.072 [-0.064, +0.207] at n = 12, a NULL                            2026-09-03 1 table
    9.58 deg / 15.72 deg / 14.51 deg best-constant bound                 2026-09-04 7.4
    the four rows, both existing cells, the five new arms and their
      patch_size -> token counts (224->1, 112->4, 56->16, 14->256)       scripts/run_campaign.sh
                                                                         wave27_arms / wave28_arms
    img_size 224, patch_size 14, proj_dim = embed_dim                    conf/encoder/vit_scratch*.yaml
    40 runs launched                                                     2026-09-05 2c
  G2
    vote{2,3,5,8,12} x {borda,median,cvar1,cvar2,max}                    scripts/plan_vote_loo.sh
    the member set is a step function of the episode block; blocks 3-7
      load {s3..s7} and blocks 8-12 load {s8..s12}; ten "seeds" of
      vote5-median are TWO committees; the rules correlate r = 0.87-0.98 2026-09-03 16.9 retraction
    leave-one-out: committee = every finished seed except block b,
      M = n-1, rule fixed to median-of-ranks, "first M deterministically" scripts/plan_vote_loo.sh
    median-of-ranks is scale-free and defined at every M                 planning/objectives.py:83-98
    M capped to min(n) over the arms compared = 11                       2026-09-05 2c
    then capped AGAIN, by GPU memory, to M = 5: members stack on the
      patch axis, so M=11 patch = 11 x 256 slots and every M=11 patch
      committee died with torch.cuda.OutOfMemoryError                    2026-09-05 2c wall 1
    16 cls and 12 patch committees, both at M = 5                        2026-09-05 2c

  ARITHMETIC DONE HERE, and nothing else:
    the two seed counts, read off the run directories that carry a DONE file --
      LpWM-ltv 16 (seeds 0-15) and PiWM-columns 12 (seeds 3-12, 14, 15).  They are the
      numbers the min(n) cap is derived from, so they are checked rather than quoted.
      The BINDING cap is smaller (memory, M = 5) and is not arithmetic on the seeds, so it
      is stated rather than computed.
    the capacity ratios between adjacent rows of the grid, 1536/384 = 4, 6144/1536 = 4 and
      98304/6144 = 16, and 5 new arms x 8 seeds = the 40 runs the round records.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    _marker, arrow, base, domeup, poly, sub, text_width, title_line, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, DOT, IMPLIES, MINUS, NDASH, TIMES,
    apoly, aw, badge, fit_size, mbox, pill, strip,
)
from analysis.round5_figs_obj import (  # noqa: E402,F401
    frow, inset, panel, strikeout,
)
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PIX, PW, PX, PY, SHIFT, W, ellipsis_v, mark, out_emit, wire_from_pred,
)
from analysis.round7_arch_figs import cbadge  # noqa: E402

DEG = "°"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-05")

# The encoder's own numbers.  conf/encoder/vit_scratch*.yaml is image_size 224 with
# proj_dim = embed_dim, and models/vit_encoder.py:67 sets num_patches = (img/patch)**2, so
# patch_size is the ONLY token knob and proj_dim is the ONLY width knob.
IMG = 224
PROJ_D = 384

# The two arms the min(n) cap comes from.  Counted from the run directories rather than
# quoted, because that cap is arithmetic on them (see _seed_counts).  It is NOT the binding
# cap: GPU memory binds first, at M = 5.
SEEDS_LTV = tuple(range(16))
SEEDS_COLS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15)


def tokens_for(patch_size):
    """models/vit_encoder.py:67, verbatim: grid = img // patch, num_patches = grid**2."""
    return (IMG // patch_size) ** 2


# --- local primitives --------------------------------------------------------------
def dblarrow(x1, x2, y, key, w=2.0):
    """A double-headed arrow: the two ends of a CONTRAST, neither of them the source.

    Every other wire in this set is directed, so a contrast drawn with one arrowhead would
    read as a dataflow from cls into patch, which is the opposite of the claim.
    """
    c = ACCENT[key]
    return (f'<path d="M{x1},{y} L{x2},{y}" stroke="{c}" stroke-width="{w}" fill="none" '
            f'stroke-linecap="round" marker-start="url(#{_marker(c, "start")})" '
            f'marker-end="url(#{_marker(c, "end")})"/>\n')


def tokenframe(cx, y, n, key, side=74.0, label="", pad=3.0):
    """n x n token slots inside a frame of FIXED outer size.

    Both latents are drawn at the same picture size and only the SLOT COUNT differs, because
    the quantity under discussion is num_patches and proj_dim is identical across the pair.
    Drawing the cls latent as one small square and the patch latent as a big grid would have
    made the picture say `capacity`, which is exactly the confound the figure is about.
    """
    gap = 0.8 if n > 4 else 0.0
    cell = (side - gap * (n - 1)) / n
    x0 = cx - side / 2
    s = (f'<rect x="{x0 - pad:.1f}" y="{y - pad:.1f}" width="{side + 2 * pad:.1f}" '
         f'height="{side + 2 * pad:.1f}" rx="3" fill="{WHITE}" stroke="{DIM}" '
         f'stroke-width="1.1"/>\n')
    for r in range(n):
        for c in range(n):
            s += (f'<rect x="{x0 + c * (cell + gap):.1f}" y="{y + r * (cell + gap):.1f}" '
                  f'width="{cell:.1f}" height="{cell:.1f}" fill="{ACCENT_FILL[key]}" '
                  f'stroke="{ACCENT[key]}" stroke-width="{0.5 if n > 4 else 1.4}"/>\n')
    if label:
        s += txt(cx, y + side / 2 + 5, label, 13, fill=ACCENT[key], weight="bold")
    return s


def seedrow(x, y, seeds, out_seed, key, cell=13, gap=3, drop=True):
    """One leave-one-out committee: a slot per finished seed, the left-out one struck.

    The excluded slot is HOLLOW AND CROSSED rather than simply absent, because the point of
    the design is that M is n-1 -- a missing slot would draw a committee of n.
    """
    s = ""
    for i, sd in enumerate(seeds):
        cx = x + i * (cell + gap)
        if drop and sd == out_seed:
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{WHITE}" stroke="{ACCENT["crit"]}" stroke-width="1.4"/>\n')
            s += strikeout(cx + cell / 2, y + cell / 2, "crit", r=cell / 2 - 1.0, w=1.7)
        else:
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{ACCENT[key]}" fill-opacity="0.72" stroke="{ACCENT[key]}" '
                  f'stroke-width="1"/>\n')
    return s


def vbrace(x, y0, y1, key, w=1.6, tick=9):
    """A vertical brace: `these rows, together, are one statement`."""
    c = ACCENT[key]
    return (f'<path d="M{x - tick},{y0} L{x},{y0} L{x},{y1} L{x - tick},{y1}" '
            f'stroke="{c}" stroke-width="{w}" fill="none" stroke-linejoin="round" '
            f'stroke-linecap="round"/>\n')


def hbrace(x0, x1, y, key, w=1.5, tick=7):
    """A horizontal brace under a run of chips."""
    c = ACCENT[key]
    return (f'<path d="M{x0},{y - tick} L{x0},{y} L{x1},{y} L{x1},{y - tick}" '
            f'stroke="{c}" stroke-width="{w}" fill="none" stroke-linejoin="round" '
            f'stroke-linecap="round"/>\n')


def gridcell(x, y, w, h, name, spec, detail, key, chip, dash=False, namecol=None):
    """One cell of G1's grid: arm name, what its two knobs are set to, and its status.

    The name is LEFT-aligned and the chip RIGHT-aligned on the same band: a centred name
    plus a corner chip is the arrangement that collides as soon as an arm is called
    `LpWM-ltv-d1536`.
    """
    c = ACCENT[key]
    d = ' stroke-dasharray="6,4"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
         f'fill="{ACCENT_FILL[key] if not dash else WHITE}" '
         f'fill-opacity="{0.55 if not dash else 1}" stroke="{c}" stroke-width="1.8"{d}/>\n')
    s += txt(x + 16, y + 24, name, fit_size(name, 16, w - 110, floor=12.5), anchor="start",
             fill=namecol or (c if not dash else MUTED), weight="bold")
    if chip:
        s += badge(x + w - 12, y + 8, chip, key, size=11, anchor="end")
    s += txt(x + 16, y + 43, spec, fit_size(spec, 12.5, w - 32, floor=10), anchor="start",
             fill=INK if not dash else MUTED)
    s += txt(x + 16, y + 59, detail, fit_size(detail, 11.5, w - 32, floor=9.5),
             anchor="start", fill=MUTED)
    return s


# Every frame a label must stay inside, registered as it is drawn.  The audit's canvas
# check is not enough on its own: G2's two crimson conclusions ran 57 units past the
# inset they belonged to and 27 past the panel, and both were still inside the canvas.
_FRAMES = []


def frame(x, y, w, h, title, key, note=""):
    """inset(), registered with the audit."""
    _FRAMES.append((x, y, x + w, y + h, title))
    return inset(x, y, w, h, title, key, note)


def pane(x, y, w, h, key, title):
    """panel(), registered with the audit."""
    _FRAMES.append((x, y, x + w, y + h, "panel"))
    return panel(x, y, w, h, key, title)


def _check_title(title, corridor_x, size=20, cx=470, clear=8):
    """The title is one centred run at y = 872 and both figures drop a connector past it, so
    its half-width is a HARD constraint, not a matter of taste."""
    body = re.sub(r"<[^>]*>", "", title)
    wide = text_width(body.replace(" -- ", " – "), size, "bold")
    right = cx + wide / 2
    assert right < corridor_x - clear, (
        f"title runs to x={right:.0f}, into the connector corridor at x={corridor_x}: "
        f"shorten it")
    return title


# ============================================== G1: the dimension-matched grid
G1_TITLE = "(G1) the aligned grid -- cls vs patch at matched capacity  [LAUNCHED]"

# rows: (total latent dims, ratio-to-the-row-above chip, cls cell, patch cell)
# Every cell is (name, spec, detail, key, chip, dashed).
G1_ROWS = (
    (384, None,
     ("LpWM-ltv", "proj_dim 384  " + DOT + "  1 token",
      "16 finished seeds  " + DOT + "  the baseline of ~120 contrasts", "slate", "EXISTS",
      False),
     ("LpWM-ltv-p1", "patch_size 224  " + ARROWC + "  1 token",
      "8 new seeds  " + DOT + "  patch tokens, without the capacity", "amber", "NEW",
      False)),
    (1536, TIMES + " 4",
     ("LpWM-ltv-d1536", "proj_dim 1536  " + DOT + "  1 token",
      "8 new seeds  " + DOT + "  width, without the tokens", "amber", "NEW", False),
     ("PiWM-cols-p4", "patch_size 112  " + ARROWC + "  4 tokens",
      "8 new seeds  " + DOT + "  4 " + TIMES + " 384 = 1536", "amber", "NEW", False)),
    (6144, TIMES + " 4",
     ("LpWM-ltv-d6144", "proj_dim 6144  " + DOT + "  1 token",
      "8 new seeds  " + DOT + "  width, without the tokens", "amber", "NEW", False),
     ("PiWM-cols-p16", "patch_size 56  " + ARROWC + "  16 tokens",
      "8 new seeds  " + DOT + "  16 " + TIMES + " 384 = 6144", "amber", "NEW", False)),
    (98304, TIMES + " 16",
     ("infeasible", "proj_dim 98 304 on a single token",
      "the cls column stops here", "slate", "", True),
     ("PiWM-columns", "patch_size 14  " + ARROWC + "  256 tokens",
      "12 finished seeds  " + DOT + "  9.58" + DEG + " on the probe", "slate", "EXISTS",
      False)),
)

GX_CLS, GX_PAT, GCW = 256, 588, 268
GCX_CLS, GCX_PAT = GX_CLS + GCW / 2, GX_PAT + GCW / 2
GROW_H, GROW_GAP, GRID_Y0 = 68, 12, 1242


def g1_aligned_grid():
    """The fix for a 256x confound: one grid, matched on TOTAL LATENT DIMENSION."""
    K = "amber"
    b = base(_check_title(G1_TITLE, 890), skip=("enc",))
    # Both encoders redrawn in the intervention's hue, because the two knobs this figure is
    # about -- encoder.feature and encoder.proj_dim -- are the encoder's own.
    for x in (64, 756):
        b += domeup(x, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                    fill=ACCENT_FILL[K], sub="feature · proj_dim")
    b += arrow(120, 520, 120, 470)
    b += arrow(812, 520, 812, 470)
    # x=712, not 736: base()'s RDMReg return runs vertically at x=726 from y=442 to 582, and
    # a label ending at 736 crossed it.
    b += txt(712, 502, "patch_size " + ARROWC + " num_patches   " + DOT
             + "   proj_dim " + ARROWC + " width", 14, anchor="end", fill=ACCENT[K],
             weight="bold")
    # round7 S1's encoder route: start short on the dome's right edge, run the empty right
    # margin (clear of the dome at 868 and o_{t+1} at 846), land long on the panel edge.
    b += poly([(868, 560), (890, 560), (890, PY)], color=ACCENT[K], w=1.8, dash="6,4")

    PH = 820
    s = pane(PX, PY, PW, PH, K,
              "module:  total latent dims  =  num_patches " + TIMES + " proj_dim  "
              + NDASH + "  matched across the two features")

    # -- row A: the confound itself, drawn.  Same frame size, different slot count.
    s += frame(60, 950, 816, 186, "the confound " + NDASH + "  one factor changed the "
               "FEATURE and the CAPACITY together", "crit",
               note="num_patches = ( img_size / patch_size )" + "²" + " ,   img_size = "
                    "224 ,   proj_dim = 384 on both sides")
    s += txt(172, 1014, "LpWM-ltv  " + DOT + "  cls", 12.5, fill=MUTED)
    s += txt(422, 1014, "PiWM-columns  " + DOT + "  patch", 12.5, fill=MUTED)
    s += tokenframe(172, 1026, 1, "slate", label="384")
    s += tokenframe(422, 1026, 16, "slate")
    s += mbox(240, 1040, 92, 46, TIMES + " 256", "crit", size=20, sub_="tokens")
    s += aw(214, 1063, 236, 1063, "crit")
    s += aw(336, 1063, 381, 1063, "crit")
    s += txt(172, 1122, "1 token " + TIMES + " 384  =  384", 12.5, weight="bold")
    s += txt(422, 1122, "256 tokens " + TIMES + " 384  =  98 304", 12.5, weight="bold")
    # the two readings the confound sits under, on the inset's right half
    s += (f'<path d="M508,1006 L508,1128" stroke="{ACCENT["crit"]}" stroke-width="1.2" '
          f'stroke-opacity="0.45"/>\n')
    s += txt(528, 1018, "so BOTH of round 7's readings are confounded", 12.5, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    # the NULL gets the null encoding: slate, a HOLLOW marker, and the word NULL.
    s += mark(536, 1046, "slate", r=6, fill=WHITE)
    s += txt(552, 1051, "the contrast   +0.072 [" + MINUS + "0.064, +0.207]", 12.5,
             anchor="start", fill=ACCENT["slate"], weight="bold")
    s += txt(552, 1068, "n = 12  " + NDASH + "  a NULL: the interval spans zero", 11.5,
             anchor="start", fill=MUTED)
    s += txt(530, 1098, "the probe   9.58" + DEG + "  vs  15.72" + DEG, 12.5,
             anchor="start", fill=ACCENT["slate"], weight="bold")
    s += txt(530, 1115, "against a 14.51" + DEG + " best-constant bound", 11.5,
             anchor="start", fill=MUTED)

    # -- row B: the grid.  The two readings above are what it exists to disentangle.
    s += txt(GCX_CLS + (GCX_PAT - GCX_CLS) / 2, 1170,
             "each ROW is a cls-vs-patch contrast at the SAME total latent dims", 13.5,
             fill=ACCENT[K], weight="bold")
    s += dblarrow(GCX_CLS, GCX_PAT, 1184, K)
    s += txt(GCX_CLS, 1212, "cls", 18, weight="bold")
    s += txt(GCX_CLS, 1230, "feature = cls  " + DOT + "  1 token", 11.5, fill=MUTED)
    s += txt(GCX_PAT, 1212, "patch", 18, weight="bold")
    s += txt(GCX_PAT, 1230, "feature = patch  " + DOT + "  num_patches tokens", 11.5,
             fill=MUTED)
    s += txt(236, 1230, "total latent dims", 11.5, anchor="end", fill=MUTED)

    ys = [GRID_Y0 + i * (GROW_H + GROW_GAP) for i in range(len(G1_ROWS))]
    # the capacity ladder: one arrow down the left gutter, with the ratio between adjacent
    # rows in its own column at x=120 (the row labels start no further left than x=170)
    s += apoly([(88, ys[0] + GROW_H / 2), (88, ys[-1] + GROW_H / 2)], K, w=1.8)
    for i, (dims, ratio, cls_cell, pat_cell) in enumerate(G1_ROWS):
        y = ys[i]
        s += txt(236, y + GROW_H / 2 + 7,
                 f"{dims:,}".replace(",", " ") if dims >= 10000 else str(dims), 19,
                 anchor="end", weight="bold")
        if ratio:
            s += cbadge(120, y - GROW_GAP - 11, ratio, K, size=12)
        s += gridcell(GX_CLS, y, GCW, GROW_H, *cls_cell[:4], cls_cell[4], dash=cls_cell[5])
        s += gridcell(GX_PAT, y, GCW, GROW_H, *pat_cell[:4], pat_cell[4], dash=pat_cell[5])
        if not cls_cell[5]:
            s += dblarrow(GX_CLS + GCW + 4, GX_PAT - 4, y + GROW_H / 2, K, w=1.8)
    s += txt(470, ys[-1] + GROW_H + 30,
             "each COLUMN is a capacity ladder at the SAME feature  " + NDASH + "  "
             + TIMES + " 4 ,  " + TIMES + " 4 ,  " + TIMES + " 16", 13.5,
             fill=ACCENT[K], weight="bold")

    s += frow(1616, [
        (["num_patches  =  ( img_size / patch_size )² ,   img_size = 224"], INK, "normal"),
        (["total latent dims  =  num_patches " + TIMES + " proj_dim"], INK, "bold")])
    s += frow(1640, [
        ("proj_dim is set independently of the ViT width", INK, "normal"),
        ("the arm name carries proj_dim, so collect_evals cannot pool two architectures",
         MUTED, "normal", 0.92)])
    s += strip(PIN, 1656, [
        ("the contrast, confounded", "+0.072 [" + MINUS + "0.064, +0.207]", False),
        ("the probe, confounded", "9.58" + DEG + " vs 15.72" + DEG, False),
        ("the capacity gap it hides", "256" + TIMES, True),
        ("newly launched", "5 arms " + TIMES + " 8 seeds = 40 runs", False)], K, cw=150)
    return b + s, PY + PH + 40


# =========================================== G2: leave-one-out consensus
G2_TITLE = "(G2) PiWM-loo5 -- consensus with no member-count knob  [LAUNCHED]"

VOTE_M = ("2", "3", "5", "8", "12")
VOTE_RULES = ("borda", "median", "cvar1", "cvar2", "max")


def g2_loo_consensus():
    """Consensus with both hyperparameters removed: M is data-set, the rule is fixed."""
    K = "green"                       # teal -- the consensus family
    b = base(_check_title(G2_TITLE, 880))
    b += wire_from_pred(K, "eval only " + NDASH + " no term in the loss")

    PH = 848
    s = pane(PX, PY, PW, PH, K,
              "module:  committee( b )  =  every finished seed EXCEPT b  " + NDASH
              + "  M = min ( n " + MINUS + " 1 , 5 )")

    # -- row A: the design being replaced, and the single fact that emptied it
    s += frame(60, 950, 816, 232, "rounds 5" + NDASH + "6 " + NDASH + "  two "
               "hyperparameters, and between them they measured one thing", "crit",
               note="vote { 2 , 3 , 5 , 8 , 12 }  " + TIMES + "  { borda , median , "
                    "cvar1 , cvar2 , max }")
    s += txt(78, 1020, "knob 1   the member count M", 13, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    for i, m in enumerate(VOTE_M):
        s += mbox(78 + i * 62, 1030, 52, 30, m, "crit", size=15)
    s += txt(78, 1086, "knob 2   the combination rule", 13, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    for i, r in enumerate(VOTE_RULES):
        s += mbox(78 + i * 78, 1096, 72, 30, r, "crit", size=13)

    s += (f'<path d="M478,1006 L478,1132" stroke="{ACCENT["crit"]}" stroke-width="1.2" '
          f'stroke-opacity="0.45"/>\n')
    s += txt(500, 1018, "what the audit found in the recorded runs", 12.5, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    # the ten episode blocks, and the TWO member sets a "seed" actually selected
    for i in range(10):
        s += mbox(500 + i * 34, 1030, 30, 26, str(3 + i), "slate", size=12)
    s += txt(500, 1070, "episode block", 11, anchor="start", fill=MUTED)
    s += hbrace(500, 666, 1086, "crit")
    s += hbrace(670, 836, 1086, "crit")
    s += txt(583, 1104, "members  s3 " + NDASH + " s7", 12, fill=ACCENT["crit"],
             weight="bold")
    s += txt(753, 1104, "members  s8 " + NDASH + " s12", 12, fill=ACCENT["crit"],
             weight="bold")
    # the two conclusions span the WHOLE inset, under both halves: centred on the right
    # half they ran 57 units past the inset border and through the `max` chip.
    s += txt(468, 1146, "the seed sets the BLOCK, not the models " + NDASH
             + " ten so-called seeds were TWO committees", 12.5, fill=ACCENT["crit"],
             weight="bold")
    s += txt(468, 1168, "and the five rules are order statistics of ONE rank matrix "
             + NDASH + " they correlate r = 0.87 " + NDASH + " 0.98", 12.5,
             fill=ACCENT["crit"], weight="bold")

    # -- row B: the replacement.  M is read off the data, one committee per block.
    s += frame(60, 1198, 816, 214, "leave-one-out " + NDASH + "  the committee is chosen "
               "by the data", K,
               note="for episode block b the members are every finished seed EXCEPT b, so "
                    "M = n " + MINUS + " 1 is not chosen at all")
    for i, (bl, lab) in enumerate(((0, "block b = 0"), (1, "block b = 1"),
                                   (15, "block b = 15"))):
        y = 1270 + i * 26 + (16 if i == 2 else 0)
        s += txt(176, y + 11, lab, 12, anchor="end", fill=MUTED)
        s += seedrow(186, y, SEEDS_LTV, bl, K)
    s += ellipsis_v(312, 1320, "slate", n=2, r=2.2, gap=9)
    # the arm label sits BELOW the inset's note, not beside it: the note is centred and
    # 584 units wide, so the band it occupies runs from x=176 to x=760.
    s += txt(186, 1258, "LpWM-ltv " + DOT + " 16 finished seeds", 11.5, anchor="start",
             fill=MUTED)
    s += vbrace(466, 1270, 1351, K)
    s += aw(466, 1310, 494, 1310, K)
    s += mbox(498, 1286, 152, 48, "median", K, size=18, sub_="of the per-member ranks")
    s += aw(650, 1310, 676, 1310, K)
    s += pill(722, 1310, "CEM", K, rx=44, ry=21)
    s += txt(722, 1352, "argsort( loss )[ : topk ]", 11.5, fill=MUTED)
    s += txt(574, 1274, "no rule knob", 12, fill=ACCENT[K], weight="bold")
    s += txt(468, 1378, "filled = a member of this block's committee     " + DOT
             + "     crossed = the one seed left out, a different one every block",
             11.5, fill=MUTED)
    s += txt(468, 1398, "the first M members are taken in order, and M is capped to the "
             "value below", 11.5, fill=MUTED)

    # -- row C: the cap, and why an uncapped M would repeat G1's mistake
    s += frame(60, 1428, 816, 176, "M is CAPPED " + NDASH + " first by the smaller arm, then by memory", K,
               note="an unmatched M would confound committee size with the feature "
                    + NDASH + "  the same error as the unmatched latent dimension")
    for i, (lab, seeds, note) in enumerate((
            ("LpWM-ltv  " + DOT + "  cls", SEEDS_LTV,
             "n = 16   " + ARROWC + "   n " + MINUS + " 1 = 15"),
            ("PiWM-columns  " + DOT + "  patch", SEEDS_COLS,
             "n = 12   " + ARROWC + "   n " + MINUS + " 1 = 11"))):
        y = 1500 + i * 34
        s += txt(212, y + 11, lab, 12.5, anchor="end", fill=MUTED)
        s += seedrow(222, y, seeds, None, K, drop=False)
        s += txt(492, y + 11, note, 12.5, anchor="start", weight="bold")
    s += vbrace(648, 1500, 1545, "crit")
    s += txt(664, 1516, "cap  =  min ( n ) " + MINUS + " 1  =  11", 13, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    s += txt(664, 1536, "but MEMORY binds first  " + ARROWC + "  M = 5", 12, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    s += txt(470, 1578, "16 cls committees and 12 patch committees, every one of them at "
             "M = 5", 13, fill=ACCENT[K], weight="bold")
    s += txt(470, 1598, "members stack on the PATCH axis: a patch member costs 256 slots "
             "where a cls member costs 1, so M = 11 patch OOMs and M = 5 fits", 11.5,
             fill=MUTED)

    s += frow(1640, [
        (["M  =  min ( n " + MINUS + " 1  over both arms ,  5 ) ,   the 5 from memory"],
         INK, "bold"),
        (["score( c )  =  median" + sub("", "m", 12) + "  rank" + sub("", "m", 12)
          + "( c )   ,   the rank within the CEM batch"], INK, "normal")])
    s += frow(1664, [
        ("median-of-ranks is scale-free, so members need not share a latent basis, and it "
         "is defined at every M", INK, "normal"),
        ("nothing here enters a training loss", MUTED, "normal", 0.95)])
    s += strip(PIN, 1680, [
        ("the design replaced", "5 M-values " + TIMES + " 5 rules", True),
        ("vote5-median's ten seeds", "really TWO committees", True),
        ("the five rules correlate", "r = 0.87 " + NDASH + " 0.98", True),
        ("launched", "16 cls + 12 patch , M = 5", False)], K, cw=190)
    return b + s, PY + PH + 40


# --- the overlap audit -------------------------------------------------------------
_TXT = re.compile(r'<text ([^>]*)>(.*?)</text>', re.S)
_ATTR = re.compile(r'(\S+)="([^"]*)"')


def _boxes(svg):
    """Every <text> element's box, from the same metrics the layout used."""
    dy = 0.0
    m = re.search(r'<g transform="translate\(0,(-?[\d.]+)\)"', svg)
    if m:
        dy = float(m.group(1))
    out = []
    for mt in _TXT.finditer(svg):
        a = dict(_ATTR.findall(mt.group(1)))
        body = mt.group(2)
        size = float(a.get("font-size", 16))
        weight = a.get("font-weight", "normal")
        w = text_width(body, size, weight)
        x, y = float(a.get("x", 0)), float(a.get("y", 0)) + dy
        anchor = a.get("text-anchor", "start")
        x0 = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
        plain = re.sub(r"<[^>]*>", "", body)
        out.append((plain, x0, y - 0.72 * size, x0 + w, y + 0.21 * size))
    return out


def audit(path, w, h, frames=(), slack=1.0, pad=9.0):
    """Every pair of text boxes that intersect, anything that leaves the canvas, and
    anything that crosses the border of the frame it sits in.

    The SVG counterpart of analysis/round7_figs.py's audit(). `NOTHING MAY OVERLAP` is not
    checkable by eye on a 940 x 1700 canvas; this reads the emitted file back and says so.
    A label is assigned to the SMALLEST registered frame whose interior contains its centre,
    which is what makes the check work for an inset nested inside a panel.
    """
    with open(path) as f:
        svg = f.read()
    dy = SHIFT if 'transform="translate(0,' in svg else 0
    bx, bad = _boxes(svg), []
    for i in range(len(bx)):
        ti, xi0, yi0, xi1, yi1 = bx[i]
        if xi0 < 4 or xi1 > w - 4 or yi0 < 0 or yi1 > h:
            bad.append(f"OUTSIDE  {ti[:44]!r}  box ({xi0:.0f},{yi0:.0f})-"
                       f"({xi1:.0f},{yi1:.0f})  canvas {w}x{h}")
        cx, cy = (xi0 + xi1) / 2, (yi0 + yi1) / 2
        best = None
        for fx0, fy0, fx1, fy1, nm in frames:
            fy0d, fy1d = fy0 + dy, fy1 + dy
            if fx0 < cx < fx1 and fy0d < cy < fy1d:
                if best is None or (fx1 - fx0) * (fy1d - fy0d) < best[0]:
                    best = ((fx1 - fx0) * (fy1d - fy0d), fx0, fy0d, fx1, fy1d, nm)
        if best and (xi0 < best[1] + pad or xi1 > best[3] - pad
                     or yi0 < best[2] or yi1 > best[4]):
            bad.append(f"CROSSES  {ti[:40]!r}  leaves {best[5][:34]!r}  "
                       f"(text {xi0:.0f}-{xi1:.0f}, frame {best[1]:.0f}-{best[3]:.0f})")
        for j in range(i + 1, len(bx)):
            tj, xj0, yj0, xj1, yj1 = bx[j]
            ow = min(xi1, xj1) - max(xi0, xj0)
            oh = min(yi1, yj1) - max(yi0, yj0)
            if ow > slack and oh > slack:
                bad.append(f"OVERLAP  {ti[:36]!r}  x  {tj[:36]!r}   "
                           f"({ow:.0f}x{oh:.0f} units)")
    return bad


FIGURES = (
    ("arch-aligned-grid.svg", g1_aligned_grid),
    ("arch-loo-consensus.svg", g2_loo_consensus),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--png", action="store_true", help="also rasterise, for the read-back")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rc = 0
    for name, fn in FIGURES:
        _FRAMES.clear()
        body, h = fn()
        frames = tuple(_FRAMES)
        out_emit(name, body, a.out, h)
        path = os.path.join(a.out, name)
        print("  wrote", path)
        bad = audit(path, W, h + SHIFT, frames)
        for line in bad:
            print("    ", line)
        rc |= bool(bad)
        if a.png:
            import cairosvg
            png = path.replace(".svg", ".png")
            cairosvg.svg2png(url=path, write_to=png, output_width=W * 2)
            print("  wrote", png)
    return rc


if __name__ == "__main__":
    sys.exit(main())
