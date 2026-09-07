"""Round 9 figures: the LpWM schematic with the new module drawn INTO it.

    python analysis/round9_arch_figs.py --out diary/assets/2026-09-07 [--png]

The house schematic (`arch_figs.base`) is the spine, exactly as every round-5..8 figure uses
it, so a reader who knows the campaign's other figures recognises this one immediately.  What
changes is that the proposed module is drawn **in the pipeline** rather than only described in
a panel underneath -- the modern-paper pattern: one architecture diagram with the new block in
place and highlighted, then a zoomed callout that expands it.

WHERE THE BLOCK GOES, AND WHY IT FITS.  base() fixes every anchor and its docstring says it
"may be re-COLOURED but not re-LAID-OUT", so the module has to occupy space the schematic
already leaves free:

    code stack bottom   y = 352
    link (ReLU) top     y = 420        -> 68 px of clear column, on both sides
    action circle       (318, 430) r = 38, so its top edge is y = 392
    action feed wire    x = 318, running from y = 392 up to y = 332

The state connector therefore runs at **y = 382** -- 10 px above the action circle -- and
crosses exactly one wire, the action feed at x = 318, which it HOPS.  That is the discipline
round6's `wire_to_pred` uses: one named route per connector, every crossing hopped rather than
drawn through.

NO OVERLAP IS ENFORCED, NOT CLAIMED.  analysis/fig_audit.audit_all checks text x text,
shape x shape and text x shape plus off-canvas, so it sees the rects and paths the old
text-only audit could not.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE, arrow, base, box, caret, circle,
    domeup, poly, sub, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    DOT, IMPLIES, MINUS, NDASH, SIGMA, TIMES, apoly, aw, badge, hpoly, mbox, opnode,
    pill, strip,
)
from analysis.round5_figs_obj import inset, panel  # noqa: E402
from analysis.round6_arch_figs import PIN, PW, PX, PY, SHIFT, W, out_emit  # noqa: E402
from analysis.fig_audit import audit_all  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-07")

_FRAMES = []


def frame(x, y, w, h, title, key, note=""):
    _FRAMES.append((x, y, x + w, y + h, title))
    return inset(x, y, w, h, title, key, note)


def pane(x, y, w, h, key, title):
    _FRAMES.append((x, y, x + w, y + h, "panel"))
    return panel(x, y, w, h, key, title)


def _fit(t, corridor_x, size=20, cx=470, clear=8):
    while size > 12 and cx + text_width(t, size) / 2 > corridor_x - clear:
        size -= 0.5
    return t


def ssm_inline(cx, y, key, label, w=84, h=36):
    """The new module, drawn in the column between the link and the code stack.

    One helper so the block reads identically in P1, P2 and P4 rather than being redrawn
    three slightly different ways.
    """
    return box(cx - w / 2, y, w, h, label, fill=ACCENT_FILL[key], stroke=ACCENT[key],
               size=14)


def pgrid(x, y, n=4, cell=26, gap=3, key="blue", cut=(), nums=False, size=10):
    """An n x n patch grid -- the image as the encoder actually sees it.

    Abstract circles do not tell a reader that the "token axis" is a raster scan of image
    patches, which is the whole reason a spatial scan needs to be BIDIRECTIONAL.
    """
    g = ""
    for r in range(n):
        for c in range(n):
            on = (r, c) in cut
            g += (f'<rect x="{x + c * (cell + gap)}" y="{y + r * (cell + gap)}" '
                  f'width="{cell}" height="{cell}" rx="2" '
                  f'fill="{ACCENT_FILL[key] if on else WHITE}" '
                  f'stroke="{ACCENT[key] if on else GRID}" stroke-width="1.1"/>\n')
            if nums:
                g += txt(x + c * (cell + gap) + cell / 2,
                         y + r * (cell + gap) + cell / 2 + size * 0.35,
                         str(r * n + c + 1), size, anchor="middle", fill=MUTED)
    return g


def recur(x0, y, key, in_lab, out_lab, a_lab="A", back=False):
    """x -> B -> (+) -> s -> C -> out, with s fed back one step through A."""
    g = txt(x0, y + 6, in_lab, 13, anchor="middle", fill=INK)
    g += arrow(x0 + 14, y, x0 + 38, y)
    g += box(x0 + 38, y - 17, 40, 34, "B", size=14)
    g += arrow(x0 + 78, y, x0 + 100, y)
    g += opnode(x0 + 115, y, "plus", key, r=14)
    g += arrow(x0 + 130, y, x0 + 152, y)
    g += box(x0 + 152, y - 17, 48, 34, "s", fill=ACCENT_FILL[key], stroke=ACCENT[key],
             size=14)
    g += arrow(x0 + 200, y, x0 + 222, y)
    g += box(x0 + 222, y - 17, 40, 34, "C", size=14)
    g += arrow(x0 + 262, y, x0 + 286, y)
    g += txt(x0 + 302, y + 6, out_lab, 13, anchor="middle", fill=ACCENT[key])
    g += box(x0 + 95, y + 54, 40, 34, a_lab, fill=WHITE, stroke=ACCENT[key], size=14)
    g += apoly([(x0 + 176, y + 17), (x0 + 176, y + 100), (x0 + 115, y + 100),
                (x0 + 115, y + 88)], key)
    g += apoly([(x0 + 115, y + 54), (x0 + 115, y + 14)], key)
    if not back:
        g += txt(x0 + 196, y + 96, "one token delay", 10, anchor="start", fill=MUTED)
    return g


def p1():
    """P1 -- a temporal state between the link and the code, on both encoder paths."""
    _FRAMES.clear()
    K = "amber"
    T = "(P1) PiWM-enc-ssm -- a temporal STATE inside the representation"
    b = base(_fit(T, 890), skip=("link",))

    # the link row is redrawn so the module can sit above it in the free column
    for cx, x0 in ((120, 78), (812, 770)):
        b += box(x0, 420, 84, 44, "ReLU", size=16)
        b += arrow(cx, 420, cx, 404)
        b += ssm_inline(cx, 364, K, "SSM")
        b += arrow(cx, 364, cx, 352)

    # the state, carried from t to t+1: the only new path in the schematic
    b += hpoly([(162, 382), (770, 382)], K, dash="6,4", hops=[(318, 382)])
    b += txt(466, 370, "s" + sub("", "t", 12) + "   carried forward", 13.5,
             anchor="middle", fill=ACCENT[K], weight="bold")

    # ---------------------------------------------------------------- the callout
    # Block diagrams, not prose: the module's own dataflow, the same dataflow with A = 0,
    # and the recurrence unrolled. A paper figure should be readable without its caption.
    PH = 626
    s = pane(PX, PY, PW, PH, K, "the module")

    def module(x0, key, a_label="A", dead=False):
        """z -> B -> (+) -> s -> C -> z', with s fed back through A. One helper, drawn
        twice: once live and once with A = 0, so the reduction is a picture."""
        g = ""
        yr = 1044                                     # the forward row
        g += txt(x0 + 16, yr + 6, "z" + sub("", "t", 11), 14, anchor="middle", fill=INK)
        g += arrow(x0 + 32, yr, x0 + 56, yr)
        g += box(x0 + 56, yr - 18, 44, 36, "B", size=15)
        g += arrow(x0 + 100, yr, x0 + 124, yr)
        g += opnode(x0 + 140, yr, "plus", key, r=15)
        g += arrow(x0 + 156, yr, x0 + 180, yr)
        g += box(x0 + 180, yr - 18, 52, 36, "s" + sub("", "t", 11),
                 fill=ACCENT_FILL[key], stroke=ACCENT[key], size=15)
        g += arrow(x0 + 232, yr, x0 + 256, yr)
        g += box(x0 + 256, yr - 18, 44, 36, "C", size=15)
        g += arrow(x0 + 300, yr, x0 + 324, yr)
        g += txt(x0 + 344, yr + 6, "z′" + sub("", "t", 11), 14, anchor="middle",
                 fill=ACCENT[key] if not dead else INK)
        # the feedback: s_t delayed one step and returned through A
        ak = "slate" if dead else key
        g += box(x0 + 118, yr + 68, 44, 36, a_label, fill=WHITE,
                 stroke=ACCENT[ak], size=15)
        g += apoly([(x0 + 206, yr + 18), (x0 + 206, yr + 118), (x0 + 140, yr + 118),
                    (x0 + 140, yr + 104)], ak)
        g += apoly([(x0 + 140, yr + 68), (x0 + 140, yr + 15)], ak)
        g += txt(x0 + 232, yr + 112, "one step", 10.5, anchor="start", fill=MUTED)
        if dead:
            g += txt(x0 + 140, yr + 48, TIMES, 20, anchor="middle",
                     fill=ACCENT["crit"], weight="bold")
        return g

    s += frame(60, 950, 400, 268, "THE MODULE", K, note="three matrices, one feedback path")
    s += module(96, K)

    s += frame(488, 950, 388, 268, "AT INITIALISATION  " + NDASH + "  A = 0", "blue",
               note="the state path is cut, so z′ = z exactly")
    s += module(516, "blue", a_label="A = 0", dead=True)

    s += frame(60, 1248, 816, 250, "UNROLLED OVER THE CLIP", K,
               note="T = num_hist = 3 at train time, T = 1 at the goal")
    xs = [186, 400, 614]
    for i2, cx in enumerate(xs):
        tag = "t" if i2 == 0 else f"t+{i2}"
        s += box(cx - 50, 1320, 100, 38, "z" + sub("", tag, 11), fill=WHITE, size=14)
        s += arrow(cx, 1358, cx, 1382)
        s += box(cx - 50, 1382, 100, 38, "s" + sub("", tag, 11),
                 fill=ACCENT_FILL[K], stroke=ACCENT[K], size=14)
        s += arrow(cx, 1420, cx, 1444)
        s += txt(cx, 1464, "z′" + sub("", tag, 11), 14, anchor="middle",
                 fill=ACCENT[K], weight="bold")
        if i2:
            s += apoly([(xs[i2 - 1] + 50, 1401), (cx - 50, 1401)], K)
            s += txt((xs[i2 - 1] + cx) / 2, 1393, "A", 12.5, anchor="middle",
                     fill=ACCENT[K], weight="bold")
    s += txt(792, 1343, "encoder code", 10.5, anchor="middle", fill=MUTED)
    s += txt(792, 1405, "carried state", 10.5, anchor="middle", fill=ACCENT[K])
    s += txt(792, 1464, "to the predictor", 10.5, anchor="middle", fill=MUTED)

    return b + s, PY + PH + 40


# ==================================================================== P2
def p2():
    """P2 -- the same state, but carried between ViT blocks inside the encoder."""
    _FRAMES.clear()
    K = "amber"
    T = "(P2) PiWM-enc-ssm-deep -- state carried BETWEEN the encoder's blocks"
    b = base(_fit(T, 890), skip=("enc",))

    # the encoder is the thing that changes, so it is redrawn in the intervention colour
    for x0, cx in ((64, 120), (756, 812)):
        b += domeup(x0, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                    sub="+ state", stroke=ACCENT[K])
        b += arrow(cx, 520, cx, 470)

    # the state now crosses INSIDE the encoder, so its route sits below the domes:
    # y = 676 clears RDMReg (ends 608) and both observation arrows (x = 120, x = 812).
    b += poly([(176, 676), (756, 676)], color=ACCENT[K], marker="a")
    b += txt(466, 666, "state at every insertion depth", 13, anchor="middle",
             fill=ACCENT[K], weight="bold")

    PH = 564
    s = pane(PX, PY, PW, PH, K, "inside the encoder")

    # -- the stack, with the scan inserted at depth
    s += frame(60, 950, 336, 486, "THE STACK", K, note="one scan after every k-th block")
    items = [("patch embed", "slate"), ("ViT block", "blue"), ("ViT block", "blue"),
             ("state scan", K), ("ViT block", "blue"), ("ViT block", "blue"),
             ("state scan", K), ("projector", "slate")]
    for i, (lab, key) in enumerate(items):
        y = 1024 + i * 50
        s += box(126, y, 204, 34, lab, size=13,
                 fill=ACCENT_FILL[key] if key == K else WHITE,
                 stroke=ACCENT[key] if key != "slate" else None)
        if i < len(items) - 1:
            s += arrow(228, y + 34, 228, y + 50)

    # -- one insertion point, expanded
    s += frame(424, 950, 452, 232, "ONE INSERTION, EXPANDED", K,
               note="the clip is folded so the scan runs over T, then folded back")
    for i, (lab, key) in enumerate((("(b, T" + DOT + "L, D)", "slate"),
                                    ("fold", K), ("scan over T", K),
                                    ("unfold", K), ("(b, T" + DOT + "L, D)", "slate"))):
        x = 456 + i * 84
        s += box(x, 1050, 72, 38, lab, size=11.5,
                 fill=ACCENT_FILL[key] if key == K else WHITE,
                 stroke=ACCENT[key] if key != "slate" else None)
        if i < 4:
            s += arrow(x + 72, 1069, x + 84, 1069)
    s += txt(650, 1130, "the ViT sees the clip as one long sequence;", 11.5,
             anchor="middle", fill=MUTED)
    s += txt(650, 1152, "the scan needs it back as frames", 11.5, anchor="middle",
             fill=ACCENT[K])

    # -- what changes versus blockcausal
    s += frame(424, 1206, 452, 230, "AGAINST blockcausal", "crit",
               note="same early mixing, opposite behaviour at T = 1")
    for i, (lab, key, mark) in enumerate((("blockcausal", "crit", "cut"),
                                          ("P2", K, "exact"))):
        y = 1284 + i * 84
        s += txt(468, y + 22, lab, 13, anchor="start", fill=ACCENT[key], weight="bold")
        s += box(596, y, 92, 40, "T = 3", size=12.5, fill=WHITE)
        s += arrow(688, y + 20, 712, y + 20)
        s += box(712, y, 92, 40, "T = 1", size=12.5,
                 fill=ACCENT_FILL[key] if mark == "exact" else WHITE,
                 stroke=ACCENT[key])
        if mark == "cut":
            s += txt(700, y + 26, TIMES, 19, anchor="middle", fill=ACCENT["crit"],
                     weight="bold")
    return b + s, PY + PH + 40


# ==================================================================== P3
def p3():
    """P3 -- a bidirectional scan replacing attention over the token axis."""
    _FRAMES.clear()
    K = "amber"
    T = "(P3) PiWM-enc-scan -- a bidirectional SCAN replaces attention over tokens"
    b = base(_fit(T, 890), skip=("enc",))
    for x0, cx in ((64, 120), (756, 812)):
        b += domeup(x0, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                    sub="scan", stroke=ACCENT[K])
        b += arrow(cx, 520, cx, 470)
    b += txt(466, 700, "the SPATIAL axis only " + NDASH
             + "  no path crosses between the two encoders", 12.5,
             anchor="middle", fill=MUTED)

    PH = 760
    s = pane(PX, PY, PW, PH, K, "inside one encoder block")

    # -- what "the token axis" actually is ------------------------------------------
    s += frame(60, 950, 400, 300, "THE TOKEN AXIS", "slate",
               note="the image is 16 x 16 patches, flattened in raster order")
    s += pgrid(146, 1022, n=4, cell=28, gap=4, nums=True)
    s += txt(376, 1060, "16 " + TIMES + " 16", 12, anchor="middle", fill=MUTED)
    s += txt(376, 1080, "patches", 12, anchor="middle", fill=MUTED)
    s += arrow(212, 1160, 212, 1186)
    for k in range(9):
        s += (f'<rect x="{104 + k * 28}" y="1192" width="22" height="22" rx="2" '
              f'fill="{WHITE}" stroke="{GRID}" stroke-width="1.1"/>\n')
    s += txt(372, 1208, "... 257", 11.5, anchor="start", fill=MUTED)
    s += txt(258, 1238, "one sequence, no natural order", 11.5, anchor="middle",
             fill=ACCENT[K])

    # -- what ONE token receives, under each operator --------------------------------
    s += frame(488, 950, 388, 300, "WHAT ONE TOKEN RECEIVES", K,
               note="the highlighted token is the one being updated")
    axs = [536, 584, 632, 680, 728, 776, 824]
    tgt = axs[3]
    s += txt(682, 1024, "attention", 12, anchor="middle", fill=ACCENT["blue"],
             weight="bold")
    for cx in axs:
        s += circle(cx, 1064, 13, "", fill=ACCENT_FILL["blue"] if cx == tgt else WHITE,
                    stroke=ACCENT["blue"])
    for k, cx in enumerate(axs):
        if cx == tgt:
            continue
        apex = 1044 - abs(cx - tgt) / 8.0          # farther source, higher arc
        s += apoly([(cx, 1051), ((cx + tgt) / 2, apex), (tgt, 1049)], "blue")
    s += txt(682, 1104, "every token, one step", 11, anchor="middle", fill=MUTED)

    s += txt(682, 1146, "scan", 12, anchor="middle", fill=ACCENT[K], weight="bold")
    for cx in axs:
        s += circle(cx, 1186, 13, "", fill=ACCENT_FILL[K] if cx == tgt else WHITE,
                    stroke=ACCENT[K])
    for i2 in range(len(axs) - 1):
        s += apoly([(axs[i2] + 13, 1186), (axs[i2 + 1] - 13, 1186)], K)
    for i2 in range(len(axs) - 1, 0, -1):
        s += apoly([(axs[i2] - 13, 1210), (axs[i2 - 1] + 13, 1210)], "magenta")
    s += txt(682, 1238, "its neighbours, through a state", 11, anchor="middle", fill=MUTED)

    # -- the scan itself -------------------------------------------------------------
    s += frame(60, 1280, 816, 352, "THE SCAN, EXPANDED", K,
               note="the same recurrence twice, in opposite directions, then merged")
    s += txt(96, 1362, "forward", 12, anchor="start", fill=ACCENT[K], weight="bold")
    s += recur(212, 1372, K, "x" + sub("", "i", 10), "f" + sub("", "i", 10))
    s += txt(96, 1478, "backward", 12, anchor="start", fill=ACCENT["magenta"],
             weight="bold")
    s += recur(212, 1488, "magenta", "x" + sub("", "i", 10), "b" + sub("", "i", 10),
               back=True)
    s += apoly([(530, 1372), (596, 1372), (596, 1420)], K)
    s += apoly([(530, 1488), (596, 1488), (596, 1440)], "magenta")
    s += box(560, 1412, 72, 36, "concat", size=12.5, fill=WHITE)
    s += arrow(632, 1430, 656, 1430)
    s += box(656, 1412, 80, 36, "Linear", size=12.5, fill=ACCENT_FILL[K],
             stroke=ACCENT[K])
    s += arrow(736, 1430, 760, 1430)
    s += txt(792, 1436, "out" + sub("", "i", 10), 13, anchor="middle", fill=ACCENT[K])
    s += txt(468, 1612, "bidirectional because a raster order is arbitrary "
             + NDASH + "  token i+1 is not \"later\" than token i",
             11.5, anchor="middle", fill=MUTED)
    return b + s, PY + PH + 40


# ==================================================================== P4
def p4():
    """P4 -- the spatial scan and the temporal state, composed."""
    _FRAMES.clear()
    K = "amber"
    T = "(P4) PiWM-enc-st-scan -- the spatial scan and the temporal state, composed"
    b = base(_fit(T, 890), skip=("enc", "link"))
    for x0, cx in ((64, 120), (756, 812)):
        b += domeup(x0, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                    sub="scan", stroke=ACCENT[K])
        b += arrow(cx, 520, cx, 470)
        b += box(x0 + 14, 420, 84, 44, "ReLU", size=16)
        b += arrow(cx, 420, cx, 404)
        b += ssm_inline(cx, 364, K, "SSM")
        b += arrow(cx, 364, cx, 352)
    b += hpoly([(162, 382), (770, 382)], K, dash="6,4", hops=[(318, 382)])
    b += txt(466, 370, "temporal state, across frames", 13, anchor="middle",
             fill=ACCENT[K], weight="bold")
    b += txt(466, 700, "spatial scan, within each frame", 12.5, anchor="middle",
             fill=ACCENT[K], weight="bold")

    PH = 672
    s = pane(PX, PY, PW, PH, K, "the two axes")

    # -- the two operators, on the object they actually act on ------------------------
    s += frame(60, 950, 816, 350, "ONE OPERATOR FAMILY, TWO AXES", K,
               note="the scan runs over patches inside a frame; the state runs between frames")
    for t in range(3):
        x0 = 132 + t * 262
        s += txt(x0 + 62, 1022, "frame t" + ("" if t == 0 else f"+{t}"), 12,
                 anchor="middle", fill=MUTED)
        # the patch grid, with the raster path drawn THROUGH it -- that path IS the scan
        s += pgrid(x0, 1036, n=4, cell=26, gap=3, key="blue")
        pts = []
        for r in range(4):
            cols = range(4) if r % 2 == 0 else range(3, -1, -1)
            for c in cols:
                pts.append((x0 + c * 29 + 13, 1036 + r * 29 + 13))
        s += apoly(pts, "blue")
        s += txt(x0 + 62, 1188, "scan over patches", 11, anchor="middle",
                 fill=ACCENT["blue"])
        s += arrow(x0 + 62, 1200, x0 + 62, 1224)
        s += box(x0 + 22, 1224, 80, 38, "s" + sub("", "t" + ("" if t == 0 else f"+{t}"), 10),
                 fill=ACCENT_FILL[K], stroke=ACCENT[K], size=13)
        if t:
            s += apoly([(x0 - 138, 1243), (x0 + 22, 1243)], K)
            s += txt(x0 - 58, 1235, "A", 12, anchor="middle", fill=ACCENT[K],
                     weight="bold")
    s += txt(468, 1284, "the scan has no memory across frames  " + DOT
             + "  the state has no memory across patches", 11.5,
             anchor="middle", fill=MUTED)

    # -- the ablation map --------------------------------------------------------------
    s += frame(60, 1330, 816, 214, "THE ABLATION MAP", "slate",
               note="every cell exists, or is an arm in this round")
    s += txt(316, 1402, "no scan", 12, anchor="middle", fill=MUTED)
    s += txt(646, 1402, "spatial scan", 12, anchor="middle", fill=MUTED)
    for r, (rl, c1, c2) in enumerate((("no state", "baseline", "P3"),
                                      ("temporal state", "P1", "P4"))):
        y = 1430 + r * 58
        s += txt(160, y + 24, rl, 12, anchor="middle", fill=MUTED)
        s += box(240, y, 152, 42, c1, size=13.5, fill=WHITE,
                 stroke=ACCENT["blue"] if c1 == "baseline" else None)
        s += box(570, y, 152, 42, c2, size=13.5,
                 fill=ACCENT_FILL[K] if c2 == "P4" else WHITE,
                 stroke=ACCENT[K] if c2 in ("P4", "P3") else None)
    return b + s, PY + PH + 40


FIGURES = (("arch-p1-enc-ssm.svg", p1),
           ("arch-p2-deep.svg", p2),
           ("arch-p3-scan.svg", p3),
           ("arch-p4-st-scan.svg", p4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rc = 0
    for name, fn in FIGURES:
        body, h = fn()
        frames = tuple(_FRAMES)
        out_emit(name, body, a.out, h)
        path = os.path.join(a.out, name)
        bad = audit_all(path, W, h + SHIFT, frames)
        print(f"  {name:<26} {'CLEAN' if not bad else str(len(bad)) + ' finding(s)'}")
        for line in bad[:10]:
            print("     ", line[:112])
        rc |= bool(bad)
        if a.png:
            import cairosvg
            cairosvg.svg2png(url=path, write_to=path.replace(".svg", ".png"),
                             output_width=W * 2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
