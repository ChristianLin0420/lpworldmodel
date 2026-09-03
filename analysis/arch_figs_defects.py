"""The four verified defects that sit underneath every model result in this campaign.

One quadrant each, drawn in the same block/flow style as arch_figs_causal.py: what the code
does on the left of the fault line, what it should do on the right, and the file:line that
establishes it. Nothing here is inferred -- every claim is a line of source.

Usage:  python analysis/arch_figs_defects.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, INK, WHITE, _hdr, box, circle, poly, sub, txt,
)
from analysis.arch_figs_causal import aw, apoly, emit, mbox, opnode, pill  # noqa: E402

W, H = 1180, 960
QW, QH = 566, 404                       # quadrant
QX = (26, 616)
QY = (86, 522)
MUTED = "#6b6b66"


def quad(i, j, key, tag, title, where):
    """Quadrant frame: number chip, title, and the file:line that proves it."""
    x, y, c = QX[j], QY[i], ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{QW}" height="{QH}" rx="10" '
         f'fill="{ACCENT_FILL[key]}" fill-opacity="0.4" stroke="{c}" stroke-width="2.4"/>\n')
    s += (f'<rect x="{x + 16}" y="{y + 14}" width="34" height="26" rx="5" fill="{c}"/>\n')
    s += txt(x + 33, y + 33, tag, 16, fill=WHITE, weight="bold")
    s += txt(x + 60, y + 33, title, 17, anchor="start", fill=c, weight="bold")
    s += txt(x + 60, y + 52, where, 12.5, anchor="start", fill=MUTED, style="italic")
    s += (f'<path d="M{x + 16},{y + 62} L{x + QW - 16},{y + 62}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def code(x, y, w, lines, key, size=13):
    """A monospace source excerpt."""
    c = ACCENT[key]
    h = 15 + 19 * len(lines)
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.3" stroke-opacity="0.65"/>\n')
    for k, ln in enumerate(lines):
        s += (f'<text x="{x + 12}" y="{y + 22 + 19 * k}" font-size="{size}" '
              f'font-family="monospace" fill="{INK}">{ln}</text>\n')
    return s


def verdict(x, y, w, text, key):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="5" fill="{c}" '
         f'fill-opacity="0.14" stroke="{c}" stroke-width="1.6"/>\n')
    return s + txt(x + w / 2, y + 23, text, 14.5, fill=c, weight="bold")


# --- 1: the metric scores the agent's own position --------------------------------
def d1():
    K, x, y = "crit", QX[0], QY[0]
    s = quad(0, 0, K, "1", "The success metric scores end-effector parking",
             "env/pusht/pusht_wrapper.py:62")
    s += code(x + 22, y + 76, QW - 44, [
        "pos_diff = norm(goal_state[:4] - cur_state[:4])"], K)
    # the state vector, with the scored prefix highlighted
    yb = y + 156
    cells = [("agent x", True), ("agent y", True), ("T x", False), ("T y", False),
             ("T theta", False), ("vel", False), ("vel", False)]
    cw = 70
    for k, (lab, agent) in enumerate(cells):
        cx = x + 26 + k * (cw + 4)
        scored = k < 4
        s += (f'<rect x="{cx}" y="{yb}" width="{cw}" height="44" rx="4" '
              f'fill="{ACCENT[K] if agent else WHITE}" '
              f'fill-opacity="{0.85 if agent else 1}" stroke="{ACCENT[K]}" '
              f'stroke-width="{2.2 if scored else 1.1}" '
              f'stroke-opacity="{1 if scored else 0.45}"/>\n')
        s += txt(cx + cw / 2, yb + 28, lab, 13,
                 fill=WHITE if agent else (INK if scored else MUTED))
    xl, xr = x + 26, x + 26 + 4 * (cw + 4) - 4
    s += apoly([(xl, yb + 58), (xl, yb + 68), (xr, yb + 68), (xr, yb + 58)], K, w=2.0)
    s += txt((xl + xr) / 2, yb + 90, "state[:4] -- the scored ball", 14.5,
             fill=ACCENT[K], weight="bold")
    s += txt(x + QW / 2, yb + 122,
             "2 of the 4 scored dimensions are the agent's own position:", 14.5)
    s += txt(x + QW / 2, yb + 143,
             "directly actuated, reached in one step, no contact and no physics", 14.5)
    s += verdict(x + 22, y + QH - 54, QW - 44,
                 "half the metric is reachable without ever touching the T", K)
    return s


# --- 2: success is latched over up to ten checkpoints ------------------------------
def d2():
    K, x, y = "amber", QX[1], QY[0]
    s = quad(0, 1, K, "2", "Success is latched, not terminal",
             "planning/mpc.py:110-112, 60-70")
    s += code(x + 22, y + 76, QW - 44, [
        "while not all(is_success) and iter &lt; max_iter:",
        "    self.is_success |= successes"], K)
    yb = y + 176
    # ten replans, one of which lands in the ball -- and the episode is then frozen
    hits = {5}
    for k in range(10):
        cx = x + 34 + k * 50
        hit = k in hits
        s += (f'<circle cx="{cx}" cy="{yb}" r="15" fill="{ACCENT[K] if hit else WHITE}" '
              f'stroke="{ACCENT[K]}" stroke-width="{2.4 if hit else 1.4}"/>\n')
        s += txt(cx, yb + 5, str(k + 1), 12, fill=WHITE if hit else MUTED)
        if k < 9:
            s += aw(cx + 16, yb, cx + 33, yb, K, w=1.4)
    s += txt(x + 34 + 5 * 50, yb - 30, "in the ball once", 13,
             fill=ACCENT[K], weight="bold")
    s += apoly([(x + 34 + 5 * 50, yb - 24), (x + 34 + 5 * 50, yb - 17)], K, w=1.8)
    s += txt(x + QW / 2, yb + 46, "reported success = OR over all ten draws", 15,
             weight="bold")
    s += txt(x + QW / 2, yb + 70, "a maximum over noisy samples, so the number", 14)
    s += txt(x + QW / 2, yb + 90, "rises with max_iter with the model held fixed", 14)
    s += txt(x + QW / 2, yb + 112,
             "then _apply_success_mask freezes the episode at (+0.0431, -0.0340)", 13,
             fill=MUTED)
    s += verdict(x + 22, y + QH - 54, QW - 44,
                 "not \"ended at the goal\" but \"passed through it at any point\"", K)
    return s


# --- 3: path_int was never trained and never saved ---------------------------------
def d3():
    K, x, y = "magenta", QX[0], QY[1]
    s = quad(1, 0, K, "3", "path_int has no optimizer and no checkpoint key",
             "train.py init_optimizers / _keys_to_save; visual_world_model.py:383-389")
    yb = y + 108
    s += mbox(x + 26, yb, 118, 46, "encoder", K)
    s += mbox(x + 158, yb, 118, 46, "predictor", K)
    s += mbox(x + 290, yb, 118, 46, "decoder", K)
    s += mbox(x + 422, yb, 118, 46, "act / prop", K)
    s += txt(x + QW / 2, yb - 16, "optimizer groups built by init_optimizers", 14,
             fill=ACCENT[K], weight="bold")
    # path_int sits outside, unconnected
    yc = yb + 92
    s += (f'<rect x="{x + 200}" y="{yc}" width="{166}" height="46" rx="5" fill="{WHITE}" '
          f'stroke="{ACCENT[K]}" stroke-width="2" stroke-dasharray="7,5"/>\n')
    s += txt(x + 283, yc + 29, "path_int", 16, fill=MUTED)
    s += txt(x + 283, yc + 64, "no optimizer group", 13.5, fill=ACCENT[K])
    s += txt(x + 283, yc + 83, "no _keys_to_save entry", 13.5, fill=ACCENT[K])
    s += txt(x + 283, yc + 102, "absent from every checkpoint on disk", 13.5, fill=ACCENT[K])
    s += code(x + 22, yc + 114, QW - 44, [
        "loss = (path_int(x) - proprio.detach()) ** 2", "# both leaves are data: it "
        "back-propagates into nothing"], K, size=12.5)
    s += verdict(x + 22, y + QH - 54, QW - 44,
                 "V3 tested use_pose with a RANDOM rollout head, not path integration", K)
    return s


# --- 4: no decoder, so nothing anchors the latent to pixels -------------------------
def d4():
    K, x, y = "blue", QX[1], QY[1]
    s = quad(1, 1, K, "4", "No decoder has ever been attached",
             "conf/train_rdmreg.yaml:126, train.py:463, visual_world_model.py:728")
    yb = y + 118
    s += circle(x + 66, yb, 30, sub("o", "t", 12), size=17)
    s += mbox(x + 128, yb - 26, 106, 52, "encoder", K)
    s += pill(x + 292, yb, sub("z", "t", 12), K, rx=32)
    s += aw(x + 98, yb, x + 124, yb, K)
    s += aw(x + 238, yb, x + 256, yb, K)
    # the decoder that is not there
    s += (f'<rect x="{x + 372}" y="{yb - 26}" width="106" height="52" rx="5" '
          f'fill="{WHITE}" stroke="{ACCENT[K]}" stroke-width="2" '
          f'stroke-dasharray="7,5" stroke-opacity="0.6"/>\n')
    s += txt(x + 425, yb + 5, "decoder", 16, fill=MUTED)
    s += aw(x + 328, yb, x + 368, yb, K, w=1.6, dash="7,5")
    s += txt(x + 425, yb - 40, "has_decoder: False", 13.5, fill=ACCENT[K], weight="bold")
    s += txt(x + 425, yb + 48, "and the branch that exists", 12.5, fill=MUTED)
    s += txt(x + 425, yb + 65, "passes z_emb.detach()", 12.5, fill=MUTED)
    yc = yb + 104
    s += txt(x + QW / 2, yc, "the only pressure on the encoder is that its own output", 14)
    s += txt(x + QW / 2, yc + 21, "be predictable from its own past", 14)
    s += txt(x + QW / 2, yc + 48, "RDMReg blocks DIMENSIONAL collapse", 14,
             fill=ACCENT[K], weight="bold")
    s += txt(x + QW / 2, yc + 69, "-- a full-rank latent can still encode only proprio,", 13.5)
    s += txt(x + QW / 2, yc + 88, "which is a free input to the predictor", 13.5)
    s += verdict(x + 22, y + QH - 54, QW - 44,
                 "nothing has ever required the latent to retain visual content", K)
    return s


def defects():
    s = txt(W / 2, 44, "Four verified defects that precede every model question in this "
            "campaign", 22, weight="bold")
    s += txt(W / 2, 70, "each read from the source, not inferred -- 3 retracts a published "
             "conclusion, 4 explains four separate collapses", 14.5, fill=MUTED)
    return s + d1() + d2() + d3() + d4()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("  wrote", a.out + "/" + emit("arch-defects-4.svg", defects(), a.out, W, H))


if __name__ == "__main__":
    main()
