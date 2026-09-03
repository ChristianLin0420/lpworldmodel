"""Architecture diagrams for the round-5 MEASUREMENT proposals (M1-M4).

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py:
the unmodified LpWM schematic from analysis/arch_figs.base() on top, and a MODULE PANEL
below carrying the proposed measurement as blocks, operator nodes and arrows, with the
equation it implements underneath and a short strip of measured values. No prose.

M1  probe          -- is the block pose linearly decodable from the frozen code at all?
M2  metric         -- what the success criterion scores now vs what it should score.
M3  oracle ladder  -- {learned, oracle} objective x {learned, oracle} dynamics.
M4  error analysis -- the four counting buckets, as an ordered decision flow.

Layout rule that keeps the panels readable: every panel is a strict left-to-right
dataflow on fixed row baselines, and any wire that must change rows does so on its own
vertical corridor. No two wires share a segment. The a_t circle at (318, 430) is enclosed
by the link/encoder/RDMReg wiring, so no connector is routed into it.

Nothing drawn here puts privileged state into a training loss: every oracle quantity in
M1/M3/M4 is read from states.pth for MEASUREMENT only.

Usage:  python analysis/round5_figs_measure.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, INK, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402
    aw, apoly, eq, mbox, nrm, opnode, pill, strip, sup,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
MUTED = "#6b6b66"
DIM = "#eeeeea"


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


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --- local primitives for the measurement panels ----------------------------------
def frame(x, y, w, h, key, title, cite=""):
    """A sub-panel inside a module panel: used by M2 for the side-by-side criteria."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="2.2"/>\n')
    s += txt(x + w / 2, y + 30, title, 18, fill=c, weight="bold")
    if cite:
        s += txt(x + w / 2, y + 50, cite, 11.5, fill=MUTED, style="italic")
    return s


def cells(x, y, cw, ch, labels, key, filled, scored):
    """A state vector drawn as labelled cells. `filled` = agent half in the accent,
    `scored` = the slice the criterion actually reads."""
    c = ACCENT[key]
    s = ""
    for k, lab in enumerate(labels):
        cx = x + k * (cw + 4)
        hot, sc = k in filled, k in scored
        s += (f'<rect x="{cx}" y="{y}" width="{cw}" height="{ch}" rx="4" '
              f'fill="{c if hot else WHITE}" fill-opacity="{0.85 if hot else 1}" '
              f'stroke="{c if sc else "#b9b9b4"}" stroke-width="{2.2 if sc else 1.1}" '
              f'stroke-opacity="{1 if sc else 0.55}"/>\n')
        s += txt(cx + cw / 2, y + ch / 2 + 5, lab, 12.5,
                 fill=WHITE if hot else (INK if sc else MUTED))
    return s


def bracket(x1, x2, y, key, label="", tick=8):
    """A downward bracket under a span of cells, with a label beneath it."""
    c = ACCENT[key]
    s = (f'<path d="M{x1},{y - tick} L{x1},{y} L{x2},{y} L{x2},{y - tick}" '
         f'stroke="{c}" stroke-width="1.8" fill="none"/>\n')
    if label:
        s += txt((x1 + x2) / 2, y + 18, label, 14, fill=c)
    return s


def bits(x, y, n, size, gap, pattern, key, live):
    """Per-checkpoint success bits. `live` = the indices the aggregator actually reads."""
    c = ACCENT[key]
    s = ""
    for k in range(n):
        cx = x + k * (size + gap)
        on, rd = pattern[k], k in live
        s += (f'<rect x="{cx}" y="{y}" width="{size}" height="{size}" rx="3" '
              f'fill="{c if on else WHITE}" fill-opacity="{0.9 if on else 1}" '
              f'stroke="{c}" stroke-width="{2.0 if rd else 1.0}" '
              f'stroke-opacity="{1 if rd else 0.3}"/>\n')
        s += txt(cx + size / 2, y + size / 2 + 5, "1" if on else "0", 12,
                 fill=WHITE if on else (INK if rd else "#bcbcb7"))
    return s


def ladder_cell(x, y, w, h, key, title, note, hi, lo):
    """One cell of the M3 2x2: what it is, and what a high / low result would mean."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="2.6"/>\n')
    s += txt(x + w / 2, y + 30, title, 18, fill=c, weight="bold")
    s += txt(x + w / 2, y + 50, note, 12, fill=MUTED)
    s += (f'<rect x="{x + 16}" y="{y + 60}" width="{w - 32}" height="64" rx="5" '
          f'fill="{WHITE}" stroke="{c}" stroke-width="1.4" stroke-opacity="0.7"/>\n')
    s += txt(x + 28, y + 84, hi, 12.5, anchor="start")
    s += txt(x + 28, y + 108, lo, 12.5, anchor="start")
    return s


def bucket(x, y, w, h, key, name, lines):
    """One counting bucket of M4: a name and what a large count would imply."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="2.2"/>\n')
    s += txt(x + w / 2, y + 26, name, 16, fill=c, weight="bold")
    s += (f'<path d="M{x + 14},{y + 36} L{x + w - 14},{y + 36}" stroke="{c}" '
          f'stroke-width="1.1" stroke-opacity="0.5"/>\n')
    for i, ln in enumerate(lines):
        s += txt(x + w / 2, y + 58 + i * 19, ln, 12, fill=INK)
    return s


# ------------------------------------------------------------------ M1: the probe
def m1_probe():
    """Is the block pose in the frozen code at all? Ridge probe + two matched controls."""
    K, C = "blue", "slate"
    b = base("(M1) probe -- is the block pose linearly decodable from the frozen code?")
    # into the right encoder's FLAT BOTTOM edge; x=880 and y=676 are empty corridors
    b += apoly([(880, 898), (880, 676), (850, 676), (850, 648)], K, w=2.0, dash="6,4")
    b += txt(760, 700, "frozen -- the probe reads it, no gradient", 14, anchor="end",
             fill=ACCENT[K])

    PY, PH = 898, 460
    s = panel(PX, PY, PW, PH, K, "module:  a ridge probe on the frozen code, with two "
              "matched controls")
    # -- the probe itself: one ridge, two heads
    s += circle(99, 1010, 24, sub("z", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += mbox(166, 958, 180, 104, "ridge  W", K, size=19, sub_="closed form, alpha by CV")
    s += aw(123, 1010, 162, 1010, K)
    # head 1: position
    s += pill(424, 980, "( x , y )", K, rx=56)
    s += aw(346, 980, 364, 980, K)
    s += mbox(540, 958, 168, 44, "median position error", K, size=13)
    s += aw(480, 980, 536, 980, K)
    s += txt(724, 986, "reported, not gating", 12, anchor="start", fill=MUTED)
    # head 2: orientation -- this is the head the falsifier is written against
    s += pill(424, 1040, "( cos t , sin t )", K, rx=66)
    s += aw(346, 1040, 354, 1040, K)
    s += mbox(540, 1018, 168, 44, "median angular error", K, size=13)
    s += aw(490, 1040, 536, 1040, K)
    s += pill(806, 1040, "&gt; 20 deg  =  FAIL", "crit", rx=80, fill=WHITE)
    s += aw(708, 1040, 722, 1040, "crit")
    # -- control 1: the same pipeline on permuted targets
    for yc, head, note, res in ((1114, "control 1  --  shuffled labels",
                                 "targets PERMUTED", "chance floor"),
                                (1192, "control 2  --  block from agent",
                                 "input = proprio only", "leak floor")):
        s += txt(166, yc - 32, head, 13, anchor="start", fill=ACCENT[C], weight="bold")
        s += circle(99, yc, 24, sub("z" if res == "chance floor" else "p", "t", 11),
                    fill=DIM, size=15)
        s += mbox(166, yc - 22, 180, 44, "ridge  W", C, size=16, sub_=note)
        s += aw(123, yc, 162, yc, C, w=1.6)
        s += pill(424, yc, "bhat", C, rx=40, fill=WHITE)
        s += aw(346, yc, 380, yc, C, w=1.6)
        s += mbox(540, yc - 22, 168, 44, "median angular error", C, size=13)
        s += aw(464, yc, 536, yc, C, w=1.6)
        s += pill(806, yc, res, C, rx=80, fill=WHITE)
        s += aw(708, yc, 722, yc, C, w=1.6)
    # -- equations
    s += eq(PX + 26, 1250, "bhat  =  W z_t ,     W  =  argmin " + nrm("W Z - B"), 17)
    s += eq(PX + 396, 1250, "+  lambda " + nrm("W"), 17)
    s += eq(PX + 556, 1250, "B  =  [ x , y , cos t , sin t ]", 17)
    s += eq(PX + 26, 1278, "B comes from states.pth -- MEASUREMENT ONLY, never in a "
            "training loss. The encoder is frozen: the probe takes no gradient.", 15)
    s += strip(PX + 26, 1294, [
        ("falsifier", "&gt; 20 deg median error", True),
        ("probe", "ridge, closed form", False),
        ("control 1", "labels shuffled", False),
        ("control 2", "block from agent only", False)], K, cw=198)
    return b + s


# ---------------------------------------------------------------- M2: the metric
def m2_metric():
    """The success criterion as measured today, next to the repaired one."""
    K, L, R = "slate", "crit", "green"
    b = base("(M2) the success metric -- what it scores now, and what it should score")
    b += apoly([(880, 898), (880, 742), (852, 742)], L, w=2.0, dash="6,4")
    b += txt(760, 700, "success is scored on the STATE behind this frame", 14,
             anchor="end", fill=ACCENT[L])

    PY, PH = 898, 488
    s = panel(PX, PY, PW, PH, K, "module:  the criterion, side by side -- same episode, "
              "two verdicts")
    LX, RX, FW = 52, 478, 408
    s += frame(LX, 950, FW, 298, L, "CURRENT", "env/pusht/pusht_wrapper.py:62")
    s += frame(RX, 950, FW, 298, R, "REPAIRED", "block only, terminal")

    labels = ["agent x", "agent y", "T x", "T y", "T theta"]
    # -- LEFT: the agent half is scored, and the OR is latched
    s += cells(LX + 16, 1010, 72, 40, labels, L, filled=(0, 1), scored=(0, 1, 2, 3))
    s += bracket(LX + 16, LX + 16 + 4 * 76 - 4, 1058, L, "goal[0:4]  -  cur[0:4]")
    s += mbox(LX + 16, 1088, 184, 40, "pos_diff  &lt;  20", L, size=15)
    s += mbox(LX + 208, 1088, 184, 40, "angle_diff  &lt;  pi/9", L, size=15)
    s += pill(LX + 204, 1158, "success at step t", L, rx=80, fill=WHITE)
    s += aw(LX + 108, 1128, LX + 162, 1142, L)
    s += aw(LX + 300, 1128, LX + 246, 1142, L)
    s += aw(LX + 204, 1177, LX + 204, 1190, L)
    s += bits(LX + 22, 1194, 10, 22, 3, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], L,
              live=tuple(range(10)))
    s += aw(LX + 273, 1205, LX + 300, 1205, L)
    s += pill(LX + 352, 1205, "OR  =  1", L, rx=48, fill=ACCENT_FILL[L])
    s += txt(LX + 204, 1236, "latched OR over 10 checkpoints  --  planning/mpc.py:110",
             11.5, fill=MUTED, style="italic")
    # -- RIGHT: block only, and only the terminal checkpoint is read
    s += cells(RX + 16, 1010, 72, 40, labels, R, filled=(), scored=(2, 3, 4))
    s += bracket(RX + 16 + 2 * 76, RX + 16 + 5 * 76 - 4, 1058, R,
                 "goal[2:5]  -  cur[2:5]")
    s += mbox(RX + 16, 1088, 184, 40, "block dist  &lt;  20", R, size=15)
    s += mbox(RX + 208, 1088, 184, 40, "angle_diff  &lt;  pi/9", R, size=15)
    s += pill(RX + 204, 1158, "success at step t", R, rx=80, fill=WHITE)
    s += aw(RX + 108, 1128, RX + 162, 1142, R)
    s += aw(RX + 300, 1128, RX + 246, 1142, R)
    s += aw(RX + 204, 1177, RX + 204, 1190, R)
    s += bits(RX + 22, 1194, 10, 22, 3, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], R, live=(9,))
    s += aw(RX + 273, 1205, RX + 300, 1205, R)
    s += pill(RX + 352, 1205, "last  =  0", R, rx=48, fill=ACCENT_FILL[R])
    s += txt(RX + 204, 1236, "the terminal checkpoint only  --  no latch", 11.5,
             fill=MUTED, style="italic")
    # -- equations
    s += eq(PX + 26, 1282, "current:     success  =  OR over 10 checkpoints "
            "[  || goal[0:4] - cur[0:4] ||  &lt;  20   AND   |dtheta|  &lt;  pi/9  ]", 15)
    s += eq(PX + 26, 1308, "repaired:   success  =  at t = T only "
            "[  || goal[2:4] - cur[2:4] ||  &lt;  20   AND   |dtheta|  &lt;  pi/9  ]", 15)
    s += strip(PX + 26, 1322, [
        ("scored dims now", "0:4 = agent + block", True),
        ("checkpoints", "latched OR over 10", True),
        ("repaired", "2:5, terminal only", False),
        ("cost", "re-score saved logs", False)], K, cw=198)
    return b + s


# --------------------------------------------------------------- M3: oracle ladder
def m3_oracle():
    """The 2x2 that localises the fault: oracle objective x oracle dynamics."""
    K = "green"
    b = base("(M3) the oracle ladder -- which half of the planner is the fault?")
    b += apoly([(880, 898), (880, 120), (300, 120), (300, 190)], K, w=2.0, dash="6,4")
    b += txt(310, 106, "dynamics:  this predictor  OR  the simulator", 16,
             anchor="start", fill=ACCENT[K])
    b += txt(438, 382, "objective: latent MSE  OR  true task distance", 15,
             anchor="start", fill=ACCENT[K])
    # leader from that label up into the gutter between the two code stacks, where the
    # MSE comparison lives.  x=766 is the empty gutter (stacks end 723, start 801).
    b += apoly([(712, 378), (766, 378), (766, 344)], K, w=1.6, dash="5,3")

    PY, PH = 898, 488
    s = panel(PX, PY, PW, PH, K, "module:  swap one half at a time; each cell says what "
              "its result would mean")
    X0, CW, GX = 150, 356, 26
    Y0, CH, GY = 1016, 136, 22
    cx = (X0 + CW / 2, X0 + CW + GX + CW / 2)
    cy = (Y0 + CH / 2, Y0 + CH + GY + CH / 2)
    # -- headers
    s += txt(cx[0], 982, "OBJECTIVE  =  learned", 16, weight="bold")
    s += txt(cx[0], 1002, "latent distance to z_goal", 12.5, fill=MUTED)
    s += txt(cx[1], 982, "OBJECTIVE  =  oracle", 16, weight="bold")
    s += txt(cx[1], 1002, "true task distance, block pose", 12.5, fill=MUTED)
    for j, (lab, note) in enumerate((("=  learned", "predictor rolls z"),
                                     ("=  oracle", "simulator rolls state"))):
        s += txt(142, cy[j] - 22, "DYNAMICS", 13, anchor="end", weight="bold")
        s += txt(142, cy[j] - 2, lab, 13, anchor="end")
        s += txt(142, cy[j] + 18, note, 11, anchor="end", fill=MUTED)
    # -- the four cells
    s += ladder_cell(X0, Y0, CW, CH, "crit", "AS SHIPPED",
                     "CEM 0.357 ltv / 0.080 linvar  --  measured",
                     "high  =>  nothing below this is needed",
                     "low   =>  the fault is in one of the other 3")
    s += ladder_cell(X0 + CW + GX, Y0, CW, CH, "blue", "PROBE THE ROLLOUT",
                     "the M1 probe reads pose out of the model's roll",
                     "high  =>  the roll carries it; the METRIC fails",
                     "low   =>  the rollout loses the block entirely")
    s += ladder_cell(X0, Y0 + CH + GY, CW, CH, "amber", "SIM ROLLS, LATENT SCORES",
                     "the simulator rolls the true state, then encode",
                     "high  =>  the metric is fine; DYNAMICS at fault",
                     "low   =>  it cannot rank even PERFECT rollouts")
    s += ladder_cell(X0 + CW + GX, Y0 + CH + GY, CW, CH, "green", "THE CEILING",
                     "simulator + true task distance -- no learning",
                     "high  =>  the ceiling; the rest is model error",
                     "low   =>  H=5 / 300 / 30 -- the PLANNER fails")
    s += strip(PX + 26, 1326, [
        ("privileged state", "eval only, never in loss", True),
        ("oracle dynamics", "the simulator itself", False),
        ("oracle objective", "block pose distance", False),
        ("measured today", "rho(latent, true) 0.398", True)], K, cw=198)
    return b + s


# ------------------------------------------------------------- M4: error analysis
def m4_erroranalysis():
    """Four mutually exclusive buckets, reached by three ordered tests."""
    K = "amber"
    b = base("(M4) error analysis -- every failed episode into exactly one of four buckets")
    b += apoly([(880, 898), (880, 800), (812, 800), (812, 782)], K, w=2.0, dash="6,4")
    b += txt(770, 830, "the state trace of every FAILED episode", 14, anchor="end",
             fill=ACCENT[K])

    PY, PH = 898, 460
    s = panel(PX, PY, PW, PH, K, "module:  three ordered tests, four exhaustive buckets "
              "-- no new training")
    QY, QH = 980, 60                     # question row
    BY, BH = 1096, 118                   # bucket row
    QX = (164, 352, 540)
    QW, BW = 164, 164
    s += mbox(40, QY + 4, 108, 52, "failed", K, size=15, sub_="episodes")
    s += aw(148, 1010, 160, 1010, K)
    tests = (("contact?", "n_contacts &gt; 0"),
             ("toward the goal?", "d(block, goal) falls"),
             ("position solved?", "|| dxy ||  &lt;  20 px"))
    for i, (lab, note) in enumerate(tests):
        s += mbox(QX[i], QY, QW, QH, lab, K, size=16, sub_=note)
        if i < 2:                        # continue right on the row baseline
            s += aw(QX[i] + QW, 1010, QX[i + 1] - 4, 1010, K)
            s += txt(QX[i] + QW + 13, 998, "yes", 12, fill=MUTED)
        # the "no" answer drops on its own vertical corridor, straight down
        s += aw(QX[i] + QW / 2, QY + QH, QX[i] + QW / 2, BY - 4, "crit")
        s += txt(QX[i] + QW / 2 + 8, 1070, "no", 13, anchor="start", fill=ACCENT["crit"])
    # the last "yes" turns down into the fourth bucket, on its own corridor
    s += apoly([(QX[2] + QW, 1010), (816, 1010), (816, BY - 4)], K)
    s += txt(766, 1000, "yes", 12, fill=MUTED)
    s += bucket(QX[0], BY, BW, BH, "crit", "NO CONTACT",
                ["the plan never reaches", "the object at all --", "a REACHING failure"])
    s += bucket(QX[1], BY, BW, BH, "magenta", "WRONG WAY",
                ["contact, then motion", "AWAY from the goal --", "the sign is wrong"])
    s += bucket(QX[2], BY, BW, BH, "blue", "SHORTFALL",
                ["right direction, ran", "out of horizon --", "H = 5 is what binds"])
    s += bucket(730, BY, 172, BH, "green", "ANGLE ONLY",
                ["position solved, theta", "not -- the angular term", "is what fails"])
    s += eq(PX + 26, 1252, "the three tests are ordered and mutually exclusive, so the "
            "four counts partition the failures and sum to the failure rate.", 15)
    s += eq(PX + 26, 1278, "n_contacts (env/pusht/pusht_env.py:578) and the per-step block "
            "pose are already computed; pusht_wrapper.rollout:103-118 drops both.", 14)
    s += strip(PX + 26, 1294, [
        ("bucket source", "n_contacts, pusht_env", True),
        ("never logged", "wrapper.rollout:103", True),
        ("also dropped", "coverage, reward, pose", True),
        ("cost", "widen the return tuple", False)], K, cw=198)
    return b + s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "diary/assets/2026-09-03"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, body, wh in (
        ("arch-m1-probe.svg", m1_probe(), (940, 1386)),
        ("arch-m2-metric.svg", m2_metric(), (940, 1414)),
        ("arch-m3-oracle.svg", m3_oracle(), (940, 1414)),
        ("arch-m4-erroranalysis.svg", m4_erroranalysis(), (940, 1386)),
    ):
        print("  wrote", a.out + "/" + emit(name, body, a.out, *wh))


if __name__ == "__main__":
    main()
