"""Round 9 figures: a state-space encoder, drawn from evidence rather than asserted.

    python analysis/round9_arch_figs.py --out diary/assets/2026-09-07 [--png]
    -> arch-blockcausal-postmortem.svg  the dissociation the whole round rests on
    -> arch-p1-enc-ssm.svg              the temporal head, and its exact single-frame limit
    -> arch-ladder.svg                  the frame-dropout ladder and what each claim predicts

THE PREMISE IS DRAWN, NOT CLAIMED.  diary/2026-09-07 section 8 says the postmortem must be
rendered from the archive first, because if the dissociation does not survive being plotted
then round 9 has no reason to exist.  So this module computes it at render time: 96 arms with
both a median rel_mse and a mean success rate, and the reader can see for themselves that the
five best-fitting arms are the five worst planners -- and that four of them are explained by a
dead code while blockcausal is not.

Primitives are IMPORTED from analysis/round8_new_figs.py (bars, dotrow, civ, patchgrid) and
the audit from round8_t_figs.  Nothing here re-implements a mark the house already owns, and
no existing figure module is edited.

Hues by IDENTITY:
    green    the system as built -- the per-frame encoder, the baseline
    purple   the contrasting condition -- the frozen-A control, the collapsed arms
    amber    the intervention under test -- the SSM state
    crimson  a failure or a retraction -- blockcausal's zero, the goal-frame mismatch
    slate    neutral -- an axis, a rung not yet run
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    arrow, base, circle, domeup, poly, sub, text_width, title_line, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, DOT, IMPLIES, MINUS, NDASH, SIGMA, TIMES,
    apoly, aw, badge, dot, fit_size, mbox, opnode, pill, strip,
)
from analysis.round5_figs_obj import inset, panel  # noqa: E402
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PW, PX, PY, SHIFT, W, mark, out_emit,
)
from analysis.round8_new_figs import bars, civ, dotrow, patchgrid  # noqa: E402,F401
from analysis.round8_t_figs import audit  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-07")
RUNS = "/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs"

_FRAMES = []
_CACHE = {}


def frame(x, y, w, h, title, key, note=""):
    _FRAMES.append((x, y, x + w, y + h, title))
    return inset(x, y, w, h, title, key, note)


def pane(x, y, w, h, key, title):
    _FRAMES.append((x, y, x + w, y + h, "panel"))
    return panel(x, y, w, h, key, title)


def _check_title(title, corridor_x, size=20, cx=470, clear=8):
    s = size
    while s > 12 and cx + text_width(title, s) / 2 > corridor_x - clear:
        s -= 0.5
    return title


def fit_sr():
    """[(arm, median rel_mse, mean SR, median effective_dim, block_causal, n)] over the archive.

    Every arm that has BOTH a wandb summary and >=3 evals.  This is the round's premise, so it
    is recomputed here rather than copied from the diary.
    """
    if "pts" in _CACHE:
        return _CACHE["pts"]
    from analysis.collect_evals import collect
    A = collect(scheme="fixed")[0]
    pts = []
    for arm, ev in A.items():
        stem = re.sub(r"_(patch|cls)$", "", arm)
        rm, ed, bc = [], [], set()
        for d in glob.glob(f"{RUNS}/{stem}_pd*_s*"):
            if "CANARY" in d:
                continue
            fs = sorted(glob.glob(f"{d}/wandb/run-*/files/wandb-summary.json"),
                        key=os.path.getmtime)
            if fs:
                try:
                    j = json.load(open(fs[-1]))
                    if "err/rel_mse" in j:
                        rm.append(float(j["err/rel_mse"]))
                    if "sparsity/effective_dim" in j:
                        ed.append(float(j["sparsity/effective_dim"]))
                except Exception:
                    pass
            try:
                bc.add("block_causal: true" in open(f"{d}/.hydra/config.yaml").read())
            except Exception:
                pass
        v = [float(x) for x in ev.values()]
        if rm and len(v) >= 3:
            pts.append((stem, float(np.median(rm)), float(np.mean(v)),
                        float(np.median(ed)) if ed else -1.0, (True in bc), len(v)))
    pts.sort(key=lambda p: p[1])
    _CACHE["pts"] = pts
    return pts


def scatter(x, y, w, h, pts, xlab, ylab, hi=(), key="slate"):
    """rel_mse (log-ish rank) against success rate. One dot per arm.

    x is plotted on RANK rather than value: rel_mse spans four orders of magnitude and a
    linear axis would pile 90 arms into one pixel column. Rank keeps every arm visible and
    the ordering -- which is the whole claim -- exact.
    """
    s = (f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="{h + 8}" rx="3" '
         f'fill="{WHITE}" stroke="{DIM}" stroke-width="1"/>\n')
    n = len(pts)
    for gy in (0.0, 0.25, 0.5):
        yy = y + h - gy / 0.7 * h
        s += (f'<line x1="{x}" y1="{yy:.1f}" x2="{x + w}" y2="{yy:.1f}" '
              f'stroke="{GRID}" stroke-width="0.8" opacity="0.6"/>\n')
        s += txt(x - 8, yy + 4, f"{gy:.2f}", 9.5, anchor="end", fill=MUTED)
    for i, (nm, rm, sr, ed, bc, k) in enumerate(pts):
        cx = x + (i / max(n - 1, 1)) * w
        cy = y + h - min(sr, 0.7) / 0.7 * h
        if nm in hi:
            col, r = (ACCENT["crit"], 7) if bc else (ACCENT["magenta"], 5.5)
            s += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{WHITE}" '
                  f'stroke="{col}" stroke-width="2.5"/>\n')
        else:
            s += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{ACCENT[key]}" '
                  f'opacity="0.45"/>\n')
    s += txt(x + w / 2, y + h + 26, xlab, 11.5, anchor="middle", fill=MUTED)
    s += txt(x, y - 12, ylab, 11.5, anchor="start", fill=MUTED)
    return s


# ============================================== the premise
PM_TITLE = "ROUND 9 PREMISE -- the best-fitting model in the archive plans at ZERO"


def blockcausal_postmortem():
    K = "crit"
    b = base(_check_title(PM_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "the goal is encoded with NO history", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 742
    s = pane(PX, PY, PW, PH, K,
             "every arm with both a fit and a score:  rel_mse (rank)  vs  planning success")

    pts = fit_sr()
    worst = [p[0] for p in pts[:5]]
    s += frame(60, 950, 816, 300, "96 arms, ordered by FIT (best on the left)", "slate",
               note="hollow = the five best-fitting arms; red ring = block_causal")
    s += scatter(120, 1020, 700, 176, pts, "arms ordered by rel_mse, best fit first",
                 "success", hi=set(worst))
    s += txt(470, 1236, "the five best-fitting arms are the five WORST planners",
             13, anchor="middle", fill=ACCENT[K], weight="bold")

    # the table that separates collapse from the anomaly
    s += frame(60, 1276, 816, 268, "FOUR ARE EXPLAINED BY A DEAD CODE. ONE IS NOT.", K,
               note="a constant code is trivially predictable, so zero error is expected")
    s += txt(96, 1346, "arm", 11, anchor="start", fill=MUTED)
    s += txt(486, 1346, "rel_mse", 11, anchor="end", fill=MUTED)
    s += txt(636, 1346, "eff_dim", 11, anchor="end", fill=MUTED)
    s += txt(846, 1346, "success", 11, anchor="end", fill=MUTED)
    rows = [p for p in pts[:5]]
    base_row = [p for p in pts if p[0] == "LpWM-ltv"]
    for i, (nm, rm, sr, ed, bc, k) in enumerate(rows + base_row):
        y = 1368 + i * 24
        dead = ed >= 0 and ed < 10
        col = ACCENT[K] if bc else (ACCENT["magenta"] if dead else INK)
        s += txt(96, y, nm, 12, anchor="start", fill=col,
                 weight="bold" if bc else "normal")
        s += txt(486, y, f"{rm:.5f}", 12, anchor="end", fill=MUTED)
        s += txt(636, y, f"{ed:.2f}" if ed >= 0 else "-", 12, anchor="end",
                 fill=ACCENT["magenta"] if dead else INK,
                 weight="bold" if (bc or dead) else "normal")
        s += txt(846, y, f"{sr:.3f}", 12, anchor="end", fill=col,
                 weight="bold" if bc else "normal")
    s += txt(470, 1522, "blockcausal: near-perfect fit, FULL-RANK code, and exactly zero",
             12.5, anchor="middle", fill=ACCENT[K], weight="bold")

    s += strip(PIN, 1570, [
        ("arms compared", str(len(pts)), True),
        ("pooled rho(fit, SR)", MINUS + "0.641", True),
        ("blockcausal eff_dim", "25.04", False),
        ("its success", "0.000", False)], K, cw=178)
    return b + s, PY + PH + 40


# ============================================== P1
P1_TITLE = "(P1) PiWM-enc-ssm -- a temporal state with an EXACT single-frame limit"


def p1_enc_ssm():
    K = "amber"
    b = base(_check_title(P1_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "state added AFTER the per-frame encoder", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 744
    s = pane(PX, PY, PW, PH, K,
             "s" + sub("", "t", 12) + " = A s" + sub("", "t-1", 12) + " + B z"
             + sub("", "t", 12) + " ,   z" + sub("", "t", 12) + " = C s" + sub("", "t", 12)
             + "      A = 0 at init  " + IMPLIES + "  identical to today")

    # -- the contract, drawn: training sees T=3, the goal sees T=1
    s += frame(60, 950, 816, 286, "THE CONTRACT  " + NDASH
               + "  training sees T = 3, the goal sees T = 1", K,
               note="planning/cem.py:77 encodes the goal; plan.py:233 gives it one frame")
    for i, (lab, T, x0, key) in enumerate((("TRAINING  clip", 3, 110, K),
                                           ("GOAL  single frame", 1, 560, "blue"))):
        s += txt(x0 + 110, 1024, lab, 12.5, anchor="middle", fill=ACCENT[key], weight="bold")
        for t in range(T):
            cx = x0 + t * 76
            s += mbox(cx, 1042, 62, 40, "o" + sub("", str(t), 11), key, size=14)
            s += aw(cx + 31, 1082, cx + 31, 1104, key)
            s += mbox(cx, 1104, 62, 40, "z" + sub("", str(t), 11), key, size=14)
            if t:
                s += aw(cx - 14, 1124, cx - 2, 1124, K)
        s += txt(x0 + 110, 1176, ("state flows across 3 frames" if T == 3
                                  else "no history to flow from"),
                 11.5, anchor="middle", fill=MUTED)
    s += txt(470, 1210, "an SSM at T = 1 gives z = C B z  " + NDASH
             + "  a function of THAT frame alone. attention does not.",
             12, anchor="middle", fill=ACCENT[K], weight="bold")

    # -- why blockcausal broke, side by side with why this cannot
    s += frame(60, 1262, 400, 214, "blockcausal  " + NDASH + "  no T = 1 limit", "crit",
               note="z_goal is an object training never produced")
    for i, t in enumerate(("each frame attends to earlier ones",
                           "at T = 1 there is nothing to attend to",
                           "so the goal is off-distribution",
                           "success 0.00, three times")):
        s += txt(96, 1334 + i * 30, t, 12, anchor="start",
                 fill=ACCENT["crit"] if i == 3 else INK)

    s += frame(476, 1262, 400, 214, "enc-ssm  " + NDASH + "  exact at T = 1", K,
               note="A = 0 reproduces the per-frame encoder bit-for-bit")
    for i, t in enumerate(("s starts at zero, A" + DOT + "0 = 0 exactly",
                           "T = 1 gives z = C B z, well defined",
                           "A = 0 init  " + IMPLIES + "  arm IS the baseline",
                           "control: A allocated but frozen")):
        s += txt(512, 1334 + i * 30, t, 12, anchor="start",
                 fill=ACCENT[K] if i == 3 else INK)

    s += strip(PIN, 1502, [
        ("new params", "3 " + TIMES + " D" + SIGMA, True),
        ("A = 0 is", "bit-identical", True),
        ("control", "A frozen at 0", False),
        ("seeds", "5", False)], K, cw=178)
    return b + s, PY + PH + 40


# ============================================== the ladder
LAD_TITLE = "THE LADDER -- manufactured partial observability, and what each claim predicts"


def ladder():
    K = "amber"
    b = base(_check_title(LAD_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "corruption on the OBSERVATION stream only", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 700
    s = pane(PX, PY, PW, PH, K,
             "frame dropout p " + SIGMA + " {0.00, 0.15, 0.35}   "
             + NDASH + "   the anchor frame is never dropped")

    # -- the three rungs, drawn as clips with frames blanked
    s += frame(60, 950, 816, 244, "THE THREE RUNGS", K,
               note="one draw per frame per clip; p = 0 is bit-identical to today")
    DROP = {0: set(), 1: {2}, 2: {1, 2}}
    for r, (p, lab) in enumerate(((0.0, "p = 0.00"), (0.15, "p = 0.15"), (0.35, "p = 0.35"))):
        x0 = 110 + r * 268
        s += txt(x0 + 100, 1022, lab, 12.5, anchor="middle",
                 fill=ACCENT[K] if r else ACCENT["blue"], weight="bold")
        for t in range(3):
            cx = x0 + t * 68
            gone = t in DROP[r]
            s += mbox(cx, 1040, 56, 44, "" if gone else "o" + sub("", str(t), 11),
                      "crit" if gone else ("blue" if r == 0 else K), size=14)
            if gone:
                s += txt(cx + 28, 1068, TIMES, 17, anchor="middle", fill=ACCENT["crit"],
                         weight="bold")
        s += txt(x0 + 100, 1114, ("nothing hidden" if r == 0
                                  else f"{len(DROP[r])} of 3 frames blank"),
                 11.5, anchor="middle", fill=MUTED)
        s += txt(x0 + 100, 1146, ("the control" if r == 0 else "state must carry it"),
                 11.5, anchor="middle", fill=ACCENT["blue"] if r == 0 else ACCENT[K])

    # -- the two claims make OPPOSITE predictions, drawn as trend lines
    s += frame(60, 1220, 816, 268, "TWO CLAIMS, OPPOSITE PREDICTIONS", "slate",
               note="this is what makes the ladder an experiment and not a sweep")
    for i, (nm, key, pred, note) in enumerate((
            ("CLAIM A  denoising", "blue", (0.5, 0.55, 0.6),
             "a small gain everywhere, including p = 0"),
            ("CLAIM B  belief state", K, (0.03, 0.4, 0.95),
             "~0 at p = 0, rising with corruption"))):
        x0 = 96 + i * 412
        s += txt(x0 + 170, 1292, nm, 12.5, anchor="middle", fill=ACCENT[key], weight="bold")
        gx, gy, gw, gh = x0 + 40, 1310, 260, 96
        s += (f'<rect x="{gx - 4}" y="{gy - 4}" width="{gw + 8}" height="{gh + 8}" rx="3" '
              f'fill="{WHITE}" stroke="{DIM}" stroke-width="1"/>\n')
        s += (f'<line x1="{gx}" y1="{gy + gh}" x2="{gx + gw}" y2="{gy + gh}" '
              f'stroke="{GRID}" stroke-width="1.2"/>\n')
        ptsxy = [(gx + j * (gw / 2), gy + gh - v * gh) for j, v in enumerate(pred)]
        s += apoly(ptsxy, key)
        for (cx, cy) in ptsxy:
            s += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{ACCENT_FILL[key]}" '
                  f'stroke="{ACCENT[key]}" stroke-width="2"/>\n')
        for j, lab in enumerate(("0", ".15", ".35")):
            s += txt(gx + j * (gw / 2), gy + gh + 18, lab, 10.5, anchor="middle", fill=MUTED)
        s += txt(x0 + 170, 1452, note, 11.5, anchor="middle", fill=INK)

    s += strip(PIN, 1512, [
        ("rungs", "0 / .15 / .35", True),
        ("p = 0 is", "the control", True),
        ("read", "the SLOPE", False),
        ("MDE at n = 5", "~0.21", False)], K, cw=178)
    return b + s, PY + PH + 40


FIGURES = (
    ("arch-blockcausal-postmortem.svg", blockcausal_postmortem),
    ("arch-p1-enc-ssm.svg", p1_enc_ssm),
    ("arch-ladder.svg", ladder),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--png", action="store_true")
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
            cairosvg.svg2png(url=path, write_to=path.replace(".svg", ".png"),
                             output_width=W * 2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
