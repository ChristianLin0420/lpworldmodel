"""The four verified defects that sit underneath every model result in this campaign.

One quadrant each, drawn in the same block/flow style as arch_figs_causal.py: what the code
does, what it should do, and the file:line that establishes it. Nothing here is inferred --
every claim is a line of source.

STYLE.  Everything comes from analysis/style.py via analysis/arch_figs.py: sans-serif
throughout, hues assigned by IDENTITY (crimson = the metric that flatters, amber = the
latching, purple = the retraction, green = the missing decoder), pale fills with a matching
saturated border, recessive furniture.

LAYOUT.  Every quadrant is laid out down a single Y CURSOR from the header rule to the
verdict bar, and the verdict is pinned to the quadrant's bottom margin.  _check() asserts
that the cursor never reaches the verdict, so the overlap this figure used to have -- the
`loss = (path_int(x) - ...)` excerpt drawn straight through quadrant 3's verdict -- cannot
come back silently.  Nothing is sized by hand where it can be measured: every title,
caption and code line is fitted to its own box.

Usage:  python analysis/arch_figs_defects.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, DIM, INK, MUTED, WHITE, _hdr, circle, sub, txt,  # noqa: F401
)
from analysis.arch_figs_causal import (  # noqa: E402
    NDASH, aw, emit, fit_size, mbox, pill,
)

# --- canvas ------------------------------------------------------------------------
# The old canvas was 1180 wide with QX=(26, 616) and QW=566, so the right-hand column's
# border fell at x=1182: two pixels off the page, clipped by the viewBox.  The grid is
# derived from the margins now, so that cannot recur.
W, H = 1180, 980
MARGIN, GUTTER = 26, 24
QW = (W - 2 * MARGIN - GUTTER) // 2            # 552
QH = 412
QX = (MARGIN, MARGIN + QW + GUTTER)            # 26, 602
QY = (96, 96 + QH + 34)                        # 96, 542

PAD = 22                                       # a quadrant's inner margin
HEAD_RULE = 62                                 # header rule, below the quadrant's top
VERDICT_H = 34
VERDICT_Y = QH - 20 - VERDICT_H                # pinned to the bottom margin

MONO = "Liberation Mono, Nimbus Mono PS, monospace"
MONO_ADV = 0.6                                 # Courier metrics: one advance, all glyphs


def _check(name, cursor, y):
    """The cursor must stop clear of the verdict bar. This is the assertion the figure
    exists to keep true."""
    limit = y + VERDICT_Y - 12
    if cursor > limit:
        raise AssertionError(
            f"{name}: content reaches y={cursor:.0f}, verdict bar starts at "
            f"y={y + VERDICT_Y} (limit {limit:.0f}) -- {cursor - limit:.0f}px over")


# --- quadrant furniture -------------------------------------------------------------
def quad(i, j, key, tag, title, where):
    """Quadrant frame: number chip, title, and the file:line that proves it."""
    x, y, c = QX[j], QY[i], ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{QW}" height="{QH}" rx="10" '
         f'fill="{ACCENT_FILL[key]}" fill-opacity="0.38" stroke="{c}" '
         f'stroke-width="1.8"/>\n')
    s += (f'<rect x="{x + PAD}" y="{y + 16}" width="30" height="26" rx="6" fill="{c}"/>\n')
    s += txt(x + PAD + 15, y + 35, tag, 15.5, fill=WHITE, weight="bold")
    tx = x + PAD + 42
    s += txt(tx, y + 34, title, fit_size(title, 17, x + QW - PAD - tx, floor=14),
             anchor="start", fill=c, weight="bold")
    s += txt(tx, y + 53, where, fit_size(where, 12.5, x + QW - PAD - tx, floor=10.5),
             anchor="start", fill=MUTED, style="italic")
    s += (f'<path d="M{x + PAD},{y + HEAD_RULE} L{x + QW - PAD},{y + HEAD_RULE}" '
          f'stroke="{c}" stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def code(x, y, w, lines, key, size=13):
    """A monospace source excerpt. The type fits the box; it never runs past the border."""
    c = ACCENT[key]
    size = min([size] + [(w - 24) / (MONO_ADV * max(1, len(_plain(ln)))) for ln in lines])
    size = round(max(9.5, size), 2)
    h = 16 + round(size * 1.5) * len(lines)
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.2" stroke-opacity="0.6"/>\n')
    for k, ln in enumerate(lines):
        s += (f'<text x="{x + 12}" y="{y + 12 + size + round(size * 1.5) * k}" '
              f'font-size="{size}" font-family="{MONO}" fill="{INK}">{ln}</text>\n')
    return s, y + h


def _plain(s):
    """Character count of a code line with SVG entities counted as one glyph."""
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&#160;", " ")):
        s = s.replace(a, b)
    return s


def verdict(x, y, w, text, key):
    """The quadrant's conclusion: a pale fill with the matching saturated text."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{VERDICT_H}" rx="6" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="1.4"/>\n')
    return s + txt(x + w / 2, y + 22.5, text,
                   fit_size(text, 14.5, w - 28, floor=12, weight="bold"),
                   fill=c, weight="bold")


def lines(cx, y, rows, size=13.5, step=21, fill=INK, weight="normal", maxw=None):
    """A centred block of caption lines. Returns (svg, next_y)."""
    s = ""
    for k, ln in enumerate(rows):
        sz = fit_size(ln, size, maxw, floor=10.5, weight=weight) if maxw else size
        s += txt(cx, y + k * step, ln, sz, fill=fill, weight=weight)
    return s, y + (len(rows) - 1) * step


# --- 1: the metric scores the agent's own position ----------------------------------
def d1():
    K, x, y = "crit", QX[0], QY[0]
    inner = QW - 2 * PAD
    s = quad(0, 0, K, "1", "The success metric scores end-effector parking",
             "env/pusht/pusht_wrapper.py:62")
    frag, cur = code(x + PAD, y + HEAD_RULE + 28, inner, [
        "pos_diff = norm(goal_state[:4] - cur_state[:4])"], K)
    s += frag

    # the state vector, with the four scored dimensions ringed and the two that are the
    # agent's own position filled
    cw, gap, n = 68, 4, 7
    x0 = x + (QW - (n * cw + (n - 1) * gap)) / 2
    yb = cur + 36
    for k, (lab, agent) in enumerate((("agent x", True), ("agent y", True),
                                      ("T x", False), ("T y", False),
                                      ("T θ", False), ("vel", False), ("vel", False))):
        cx, scored = x0 + k * (cw + gap), k < 4
        s += (f'<rect x="{cx}" y="{yb}" width="{cw}" height="44" rx="5" '
              f'fill="{ACCENT[K] if agent else WHITE}" '
              f'stroke="{ACCENT[K] if scored else DIM}" '
              f'stroke-width="{1.8 if scored else 1.0}"/>\n')
        s += txt(cx + cw / 2, yb + 28, lab, 12.5,
                 fill=WHITE if agent else (INK if scored else MUTED))
    # a plain brace under the scored prefix -- no arrowhead: it delimits, it does not flow
    xl, xr = x0, x0 + 4 * (cw + gap) - gap
    s += (f'<path d="M{xl},{yb + 52} L{xl},{yb + 62} L{xr},{yb + 62} L{xr},{yb + 52}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.8" fill="none" '
          f'stroke-linejoin="round" stroke-linecap="round"/>\n')
    s += txt((xl + xr) / 2, yb + 88, f"state[:4] {NDASH} the scored ball", 14.5,
             fill=ACCENT[K], weight="bold")
    frag, cur = lines(x + QW / 2, yb + 130, [
        "2 of the 4 scored dimensions are the agent's own position:",
        "directly actuated, reached in one step, no contact, no physics"],
        size=13.5, step=22, maxw=inner)
    s += frag
    _check("d1", cur, y)
    return s + verdict(x + PAD, y + VERDICT_Y, inner,
                       "half the metric is reachable without ever touching the T", K)


# --- 2: success is latched over up to ten checkpoints --------------------------------
def d2():
    K, x, y = "amber", QX[1], QY[0]
    inner = QW - 2 * PAD
    s = quad(0, 1, K, "2", "Success is latched, not terminal",
             "planning/mpc.py:110-112, 60-70")
    frag, cur = code(x + PAD, y + HEAD_RULE + 18, inner, [
        "while not all(is_success) and iter &lt; max_iter:",
        "    self.is_success |= successes"], K)
    s += frag

    # ten replans, one of which lands in the ball -- and the episode is then frozen
    pitch, r, hit = 50, 15, 5
    x0 = x + (QW - (9 * pitch)) / 2
    yb = cur + 74
    s += txt(x0 + hit * pitch, yb - 30, "in the ball once", 13,
             fill=ACCENT[K], weight="bold")
    s += (f'<path d="M{x0 + hit * pitch},{yb - 24} L{x0 + hit * pitch},{yb - r - 3}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.6" stroke-linecap="round"/>\n')
    for k in range(10):
        cx, on = x0 + k * pitch, k == hit
        s += (f'<circle cx="{cx}" cy="{yb}" r="{r}" fill="{ACCENT[K] if on else WHITE}" '
              f'stroke="{ACCENT[K]}" stroke-width="{2.0 if on else 1.2}"/>\n')
        s += txt(cx, yb + 5, str(k + 1), 12, fill=WHITE if on else MUTED)
        if k < 9:
            s += aw(cx + r + 1, yb, cx + pitch - r - 2, yb, K, w=1.3)

    s += txt(x + QW / 2, yb + 50, "reported success  =  OR over all ten draws", 14.5,
             weight="bold")
    frag, cur = lines(x + QW / 2, yb + 76, [
        "a maximum over noisy samples, so the number",
        "rises with max_iter with the model held fixed"], size=13.5, step=22, maxw=inner)
    s += frag
    s += txt(x + QW / 2, cur + 30,
             "_apply_success_mask then freezes the episode at (+0.0431, -0.0340)",
             fit_size("_apply_success_mask then freezes the episode at "
                      "(+0.0431, -0.0340)", 12.5, inner, floor=10.5), fill=MUTED)
    _check("d2", cur + 30, y)
    return s + verdict(x + PAD, y + VERDICT_Y, inner,
                       'not "ended at the goal" but "passed through it at any point"', K)


# --- 3: path_int was never trained and never saved ------------------------------------
def d3():
    K, x, y = "magenta", QX[0], QY[1]
    inner = QW - 2 * PAD
    s = quad(1, 0, K, "3", "path_int has no optimizer and no checkpoint key",
             "train.py init_optimizers / _keys_to_save; visual_world_model.py:383-389")

    # what init_optimizers does build
    yb = y + HEAD_RULE + 24
    s += txt(x + QW / 2, yb, "optimizer groups built by init_optimizers", 13.5,
             fill=ACCENT[K], weight="bold")
    bw, pitch = 116, 130
    x0 = x + (QW - (3 * pitch + bw)) / 2
    for k, lab in enumerate(("encoder", "predictor", "decoder", "act / prop")):
        s += mbox(x0 + k * pitch, yb + 14, bw, 44, lab, K, size=14)

    # and what it does not: path_int sits outside, connected to nothing
    yc = yb + 82
    s += (f'<rect x="{x + PAD}" y="{yc}" width="164" height="44" rx="6" fill="{WHITE}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.6" stroke-dasharray="7,5"/>\n')
    s += txt(x + PAD + 82, yc + 28, "path_int", 16, fill=MUTED)
    for k, ln in enumerate(("no optimizer group", "no _keys_to_save entry",
                            "absent from every checkpoint on disk")):
        s += txt(x + PAD + 184, yc + 14 + k * 19, ln, 13, anchor="start", fill=ACCENT[K])

    frag, cur = code(x + PAD, yc + 78, inner, [
        "loss = (path_int(x) - proprio.detach()) ** 2",
        "# both leaves are data: it back-propagates into nothing"], K, size=12.5)
    s += frag
    s += txt(x + QW / 2, cur + 28,
             "_roll_pose then rolls a RANDOM nn.Linear at plan time", 13.5,
             fill=ACCENT[K], weight="bold")
    _check("d3", cur + 28, y)
    return s + verdict(x + PAD, y + VERDICT_Y, inner,
                       "V3 tested use_pose with a RANDOM rollout head, not path integration", K)


# --- 4: no decoder, so nothing anchors the latent to pixels ---------------------------
def d4():
    K, x, y = "blue", QX[1], QY[1]
    inner = QW - 2 * PAD
    s = quad(1, 1, K, "4", "No decoder has ever been attached",
             "conf/train_rdmreg.yaml:126, train.py:463, visual_world_model.py:728")

    # the encoder branch that exists, and the decoder branch that does not
    yb = y + HEAD_RULE + 62
    s += circle(x + 86, yb, 28, sub("o", "t", 12), size=16)
    s += mbox(x + 148, yb - 26, 106, 52, "encoder", K, size=15)
    s += pill(x + 312, yb, sub("z", "t", 12), K, rx=30, ry=19)
    s += aw(x + 116, yb, x + 144, yb, K)
    s += aw(x + 254, yb, x + 278, yb, K)
    dx = x + 392
    s += (f'<rect x="{dx}" y="{yb - 26}" width="106" height="52" rx="6" fill="{WHITE}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.6" stroke-dasharray="7,5" '
          f'stroke-opacity="0.6"/>\n')
    s += txt(dx + 53, yb + 5, "decoder", 15, fill=DIM)
    s += aw(x + 346, yb, dx - 4, yb, K, w=1.5, dash="7,5")
    s += txt(dx + 53, yb - 40, "has_decoder: False", 13, fill=ACCENT[K], weight="bold")
    s += txt(dx + 53, yb + 48, "and the branch that exists", 12, fill=MUTED)
    s += txt(dx + 53, yb + 65, "passes z_emb.detach()", 12, fill=MUTED)

    frag, cur = lines(x + QW / 2, yb + 96, [
        "the only pressure on the encoder is that its own",
        "output be predictable from its own past"], size=13.5, step=21, maxw=inner)
    s += frag
    s += txt(x + QW / 2, cur + 32, "RDMReg blocks DIMENSIONAL collapse", 13.5,
             fill=ACCENT[K], weight="bold")
    frag, cur = lines(x + QW / 2, cur + 54, [
        f"{NDASH} a full-rank latent can still encode only proprio,",
        "which is a free input to the predictor"], size=13, step=20, maxw=inner)
    s += frag
    _check("d4", cur, y)
    return s + verdict(x + PAD, y + VERDICT_Y, inner,
                       "nothing has ever required the latent to retain visual content", K)


# --- the figure ----------------------------------------------------------------------
def defects():
    title = "Four verified defects that precede every model question in this campaign"
    sub_ = (f"each read from the source, not inferred {NDASH} 3 retracts a published "
            f"conclusion, 4 explains four separate collapses")
    s = txt(W / 2, 44, title, fit_size(title, 22, W - 120, floor=18, weight="bold"),
            weight="bold")
    s += txt(W / 2, 70, sub_, fit_size(sub_, 14, W - 80, floor=11.5), fill=MUTED)
    return s + d1() + d2() + d3() + d4()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("  wrote", a.out + "/" + emit("arch-defects-4.svg", defects(), a.out, W, H))


if __name__ == "__main__":
    main()
