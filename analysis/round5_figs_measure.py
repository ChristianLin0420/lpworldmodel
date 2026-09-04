"""Architecture diagrams for the round-5 MEASUREMENT proposals (M1-M4).

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py:
the unmodified LpWM schematic from analysis/arch_figs.base() on top -- nudged up by sch()
so the set does not carry a band of white above every figure -- and a MODULE PANEL below
carrying the proposed measurement as blocks, operator nodes and arrows, with the equation
it implements underneath and a short strip of measured values. No prose.

M1  probe          -- is the block pose linearly decodable from the frozen code at all?
M2  metric         -- what the success criterion scores now vs what it should score.
M3  oracle ladder  -- {learned, oracle} objective x {learned, oracle} dynamics.
M4  error analysis -- the four counting buckets, as an ordered decision flow.

WHAT THIS FILE OWES THE DESIGN SYSTEM (analysis/style.py, via arch_figs)

  * Hue by IDENTITY, never by rank. M3's four ladder cells and M4's four buckets each get
    four DIFFERENT hues (crimson / purple / amber / green and crimson / purple / slate /
    green): they used to be drawn with #18483C and #2E7D6F side by side, which read as one
    colour. Re-running these after more evals land cannot repaint anything.
  * Strict left-to-right dataflow on fixed row baselines; a wire that changes rows does so
    on its OWN vertical corridor, and the two connectors into the schematic (wire_to_enc,
    wire_to_obs) use corridors that cross NO model wire.
  * Nothing is sized by hand where it can be measured: mbox/pill/strip fit their own text
    and eqrow guarantees a gap between clauses. M1's `+ lambda ||W||` used to float 6px
    from the clause before it and read as a separate statement.
  * Text goes in free space and is owned by the mark beside it.

Nothing drawn here puts privileged state into a training loss: every oracle quantity in
M1/M3/M4 is read from states.pth for MEASUREMENT only.

Usage:  python analysis/round5_figs_measure.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, DIM, INK, MUTED, WHITE,
    _hdr, base, caret, circle, sub, sup, text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402
    BAR, IMPLIES, LAMB, MINUS, NDASH, PIX, PX, PW, PY, PYL, RHO, THETA,
    apoly, aw, eq, eqrow, fit_size, mbox, pill, sch, strip,
)

DEG = "°"
PI = "π"
HINT = 1.06     # cairo rounds every glyph advance UP to a whole pixel at these sizes, so
                # a 120-character line renders ~5% wider than the AFM table says. Without
                # this allowance M4's source line ran 17px past the panel border and M1's
                # ran into it -- fit_size() trusts the table.


def eqline(x, y, s, size=15, weight="normal"):
    """One equation line that cannot run past the panel, hinting included."""
    return eq(x, y, s, fit_size(s, size, (PIX - 8 - x) / HINT, floor=11.5,
                                weight=weight), weight)


def eqcols(y, cols, size=17, gap=44, x0=None):
    """eqrow() with the same allowance."""
    x0 = PX + 26 if x0 is None else x0
    return eqrow(y, cols, size, x0=x0, x1=x0 + (PIX - 8 - x0) / HINT, gap=gap)
_TALL = set("bdfhklt") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _nd(s):
    """The house dash. Every label in this file is written with ASCII `--`; it is set as
    an en dash exactly once, here, so no call site can forget."""
    return s.replace(" -- ", f"  {NDASH}  ")


def _hat_dy(ch, size):
    """How far over the baseline a drawn circumflex goes. arch_figs_causal.hatrun() sizes
    the offset for an x-height letter (z); over an ASCENDER it lands ON the stem, which is
    what b-hat used to look like here."""
    return round((1.02 if ch in _TALL else 0.66) * size, 1)


def hrun(x, y, main, size=17, fill=INK, weight="normal"):
    """`main` with a drawn circumflex over its first glyph, start-anchored at x.
    Returns (svg, advance) so a caller can chain runs on one baseline."""
    s = txt(x, y, main, size, anchor="start", fill=fill, weight=weight)
    w0 = text_width(main[0], size, weight)
    s += caret(round(x + w0 / 2, 1), round(y - _hat_dy(main[0], size), 1),
               w=max(7.0, 0.5 * size), h=max(4.0, 0.32 * size), color=fill)
    return s, text_width(main, size, weight)


def hpill(cx, cy, main, key, rx=34, ry=19, fill=None, size=16):
    """pill() with the hat placed by hrun()."""
    c, f = ACCENT[key], (fill or ACCENT_FILL[key])
    w = text_width(main, size)
    rx = max(rx, w / 2 + 12)
    s = (f'<rect x="{round(cx - rx, 1)}" y="{cy - ry}" width="{round(2 * rx, 1)}" '
         f'height="{2 * ry}" rx="{ry}" fill="{f}" stroke="{c}" stroke-width="1.8"/>\n')
    return s + hrun(round(cx - w / 2, 1), cy + 6, main, size)[0]


# --- the module panel -------------------------------------------------------------
def panel(x, y, w, h, key, title):
    """Accent-bordered module panel with a title rule. Explicit geometry, because these
    four panels are four different heights; everything else matches arch_figs_causal."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.2"/>\n')
    title = _nd(title)
    s += txt(x + 20, y + 32, title, fit_size(title, 18, w - 44, floor=14),
             anchor="start", fill=c, weight="bold")
    s += (f'<path d="M{x + 18},{y + 44} L{x + w - 18},{y + 44}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def emit(name, body, out, w, h):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --- the two connector routes this file needs -------------------------------------
# base()'s wires form a closed loop, but both measurement hooks attach OUTSIDE it -- at
# the encoder and at the observation -- so each reaches its anchor along empty corridors
# and crosses nothing at all.  The caption sits beside the vertical run, in the empty band
# under o_{t+1} and above the title, so it is owned by the wire and touches nothing.
def wire_to_enc(key, label):
    """Into the right encoder's flat bottom edge, via x=880 and the empty band y=676."""
    s = apoly([(880, PYL), (880, 676), (850, 676), (850, 648)], key, dash="6,4")
    return s + txt(866, 806, label, 15.5, anchor="end", fill=ACCENT[key])


def wire_to_obs(key, label, bottom=False):
    """Into o_{t+1}: its right edge (x=846), or its flat bottom (y=776)."""
    pts = ([(880, PYL), (880, 800), (812, 800), (812, 782)] if bottom else
           [(880, PYL), (880, 742), (852, 742)])
    s = apoly(pts, key, dash="6,4")
    return s + txt(866, 838 if bottom else 806, label, 15.5, anchor="end",
                   fill=ACCENT[key])


# --- local primitives for the measurement panels ----------------------------------
def frame(x, y, w, h, key, title, cite=""):
    """A sub-panel inside a module panel: used by M2 for the side-by-side criteria."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="2"/>\n')
    s += txt(x + w / 2, y + 30, title, 18, fill=c, weight="bold")
    if cite:
        s += txt(x + w / 2, y + 50, cite, fit_size(cite, 11.5, w - 24, floor=9.5),
                 fill=MUTED, style="italic")
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
              f'fill="{c if hot else WHITE}" fill-opacity="{0.9 if hot else 1}" '
              f'stroke="{c if sc else DIM}" stroke-width="{2.0 if sc else 1.1}" '
              f'stroke-opacity="{1 if sc else 0.7}"/>\n')
        s += txt(cx + cw / 2, y + ch / 2 + 5, lab, fit_size(lab, 12.5, cw - 8, floor=10),
                 fill=WHITE if hot else (INK if sc else MUTED))
    return s


def bracket(x1, x2, y, key, label="", tick=8):
    """A downward bracket under a span of cells, with a label beneath it."""
    c = ACCENT[key]
    s = (f'<path d="M{x1},{y - tick} L{x1},{y} L{x2},{y} L{x2},{y - tick}" '
         f'stroke="{c}" stroke-width="1.6" fill="none" stroke-linejoin="round"/>\n')
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
              f'stroke="{c}" stroke-width="{1.8 if rd else 1.0}" '
              f'stroke-opacity="{1 if rd else 0.32}"/>\n')
        s += txt(cx + size / 2, y + size / 2 + 5, "1" if on else "0", 12,
                 fill=WHITE if on else (INK if rd else DIM))
    return s


def ladder_cell(x, y, w, h, key, title, note, hi, lo):
    """One cell of the M3 2x2: what it is, and what a high / low result would mean."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="2.2"/>\n')
    s += txt(x + w / 2, y + 30, title, fit_size(title, 18, w - 30, floor=15),
             fill=c, weight="bold")
    note = _nd(note)
    s += txt(x + w / 2, y + 50, note, fit_size(note, 12, w - 24, floor=10), fill=MUTED)
    s += (f'<rect x="{x + 16}" y="{y + 62}" width="{w - 32}" height="62" rx="6" '
          f'fill="{WHITE}" stroke="{c}" stroke-width="1.3" stroke-opacity="0.7"/>\n')
    for i, ln in enumerate((_nd(hi), _nd(lo))):
        s += txt(x + 28, y + 85 + i * 23, ln,
                 fit_size(ln, 12.5, w - 56, floor=10.5), anchor="start")
    return s


def bucket(x, y, w, h, key, name, lines):
    """One counting bucket of M4: a name and what a large count would imply."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
         f'fill="{ACCENT_FILL[key]}" stroke="{c}" stroke-width="2"/>\n')
    s += txt(x + w / 2, y + 26, name, fit_size(name, 16, w - 20, floor=13),
             fill=c, weight="bold")
    s += (f'<path d="M{x + 14},{y + 36} L{x + w - 14},{y + 36}" stroke="{c}" '
          f'stroke-width="1.1" stroke-opacity="0.5"/>\n')
    for i, ln in enumerate(lines):
        s += txt(x + w / 2, y + 58 + i * 19, ln,
                 fit_size(ln, 12, w - 16, floor=10), fill=INK)
    return s


# ------------------------------------------------------------------ M1: the probe
def m1_probe():
    """Is the block pose in the frozen code at all? Ridge probe + two matched controls."""
    K, N = "blue", "slate"
    b = base("(M1) probe -- is the block pose linearly decodable from the frozen code?")
    b += wire_to_enc(K, f"frozen {NDASH} the probe reads it, no gradient")
    b = sch(b)

    PH = 480
    s = panel(PX, PY, PW, PH, K, "module:  a ridge probe on the frozen code, with two "
              "matched controls")
    # -- the probe itself: one ridge, two heads on two row baselines
    R1, R2 = 932, 992
    s += circle(96, 962, 24, sub("z", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += mbox(150, 910, 152, 104, "ridge  W", K, size=19, sub_="closed form, alpha by CV")
    s += aw(120, 962, 146, 962, K)
    for y, head, metric in ((R1, "( x , y )", "median position error"),
                            (R2, f"( cos {THETA} , sin {THETA} )", "median angular error")):
        s += pill(396, y, head, K, rx=72)
        s += aw(302, y, 320, y, K)
        s += mbox(500, y - 22, 176, 44, metric, K, size=13)
        s += aw(468, y, 496, y, K)
    # the orientation head is the one the falsifier is written against
    s += txt(692, R1 + 6, "reported, not gating", 12.5, anchor="start", fill=MUTED)
    s += pill(792, R2, f"&gt; 20{DEG}  =  FAIL", "crit", rx=86, fill=WHITE)
    s += aw(676, R2, 702, R2, "crit")
    # -- the two matched controls, on the same x grid, in the neutral hue
    for yc, head, note, res in ((1082, "control 1  --  shuffled labels",
                                 "targets PERMUTED", "chance floor"),
                                (1166, "control 2  --  block from agent",
                                 "input = proprio only", "leak floor")):
        s += txt(150, yc - 34, head.replace(" -- ", f"  {NDASH}  "), 13, anchor="start",
                 fill=ACCENT[N], weight="bold")
        s += circle(96, yc, 24, sub("z" if res == "chance floor" else "p", "t", 11),
                    fill=ACCENT_FILL[N], size=15)
        s += mbox(150, yc - 22, 152, 44, "ridge  W", N, size=16, sub_=note)
        s += aw(120, yc, 146, yc, N, w=1.5)
        s += hpill(396, yc, "b", N, rx=52, fill=WHITE)
        s += aw(302, yc, 340, yc, N, w=1.5)
        s += mbox(500, yc - 22, 176, 44, "median angular error", N, size=13)
        s += aw(448, yc, 496, yc, N, w=1.5)
        s += pill(792, yc, res, N, rx=86, fill=WHITE)
        s += aw(676, yc, 702, yc, N, w=1.5)
    # -- equations
    lhs, adv = hrun(PX + 26, 1222, "b", 17)
    s += lhs
    s += eqcols(1222, [
        ["  =  W z", sub("", "t"), " ,     W  =  argmin  ", BAR,
         f"W Z {MINUS} B", BAR, sup("", "2", 11), f"  +  {LAMB} ", BAR, "W", BAR,
         sup("", "2", 11)],
        [f"B  =  [ x , y , cos {THETA} , sin {THETA} ]"]], 17, x0=PX + 26 + adv)
    s += eqline(PX + 26, 1248, "B comes from states.pth " + NDASH + " MEASUREMENT ONLY, "
               "never in a training loss. The encoder is frozen: the probe takes no "
               "gradient.", 15)
    s += strip(PX + 26, 1264, [
        ("falsifier", f"&gt; 20{DEG} median error", True),
        ("probe", "ridge, closed form", False),
        ("control 1", "labels shuffled", False),
        ("control 2", "block from agent only", False)], K, cw=198)
    return b + s, PY + PH + 28


# ---------------------------------------------------------------- M2: the metric
def m2_metric():
    """The success criterion as measured today, next to the repaired one."""
    K, L, R = "slate", "crit", "blue"
    b = base("(M2) the success metric -- what it scores now, and what it should score")
    b += wire_to_obs(L, "success is scored on the STATE behind this frame")
    b = sch(b)

    PH = 496
    s = panel(PX, PY, PW, PH, K, "module:  the criterion, side by side  --  same episode, "
              "two verdicts")
    LX, RX, FW = 52, 478, 408
    s += frame(LX, 900, FW, 302, L, "CURRENT", "env/pusht/pusht_wrapper.py:62")
    s += frame(RX, 900, FW, 302, R, "REPAIRED", "block only, terminal")

    labels = ["agent x", "agent y", "T x", "T y", f"T {THETA}"]
    for X, key, filled, scored, span, tag, tests, pat, live, res, cite in (
        (LX, L, (0, 1), (0, 1, 2, 3), (0, 4), f"goal[0:4]  {MINUS}  cur[0:4]",
         ("pos_diff  &lt;  20", f"angle_diff  &lt;  {PI}/9"),
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], tuple(range(10)), "OR  =  1",
         "latched OR over 10 checkpoints  --  planning/mpc.py:110"),
        (RX, R, (), (2, 3, 4), (2, 5), f"goal[2:5]  {MINUS}  cur[2:5]",
         ("block dist  &lt;  20", f"angle_diff  &lt;  {PI}/9"),
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], (9,), "last  =  0",
         "the terminal checkpoint only  --  no latch")):
        s += cells(X + 16, 964, 72, 40, labels, key, filled=filled, scored=scored)
        s += bracket(X + 16 + span[0] * 76, X + 16 + span[1] * 76 - 4, 1012, key, tag)
        s += mbox(X + 16, 1044, 184, 40, tests[0], key, size=15)
        s += mbox(X + 208, 1044, 184, 40, tests[1], key, size=15)
        s += pill(X + 204, 1114, "success at step t", key, rx=82, fill=WHITE)
        s += aw(X + 108, 1084, X + 162, 1091, key)
        s += aw(X + 300, 1084, X + 246, 1091, key)
        s += aw(X + 204, 1133, X + 204, 1144, key)
        s += bits(X + 22, 1148, 10, 22, 3, pat, key, live=live)
        s += aw(X + 273, 1159, X + 296, 1159, key)
        s += pill(X + 346, 1159, res, key, rx=46, fill=ACCENT_FILL[key])
        s += txt(X + 204, 1190, cite.replace(" -- ", f"  {NDASH}  "), 11.5, fill=MUTED,
                 style="italic")
    # -- equations
    s += eqline(PX + 26, 1240, "current:     success  =  OR over 10 checkpoints  "
               f"[  {BAR}goal[0:4] {MINUS} cur[0:4]{BAR} &lt; 20   AND   "
               f"|d{THETA}| &lt; {PI}/9  ]", 15)
    s += eqline(PX + 26, 1266, "repaired:   success  =  at t = T only          "
               f"[  {BAR}goal[2:4] {MINUS} cur[2:4]{BAR} &lt; 20   AND   "
               f"|d{THETA}| &lt; {PI}/9  ]", 15)
    s += strip(PX + 26, 1280, [
        ("scored dims now", "0:4 = agent + block", True),
        ("checkpoints", "latched OR over 10", True),
        ("repaired", "2:5, terminal only", False),
        ("cost", "re-score saved logs", False)], K, cw=198)
    return b + s, PY + PH + 28


# --------------------------------------------------------------- M3: oracle ladder
def m3_oracle():
    """The 2x2 that localises the fault: oracle objective x oracle dynamics."""
    K = "green"
    b = base("(M3) the oracle ladder -- which half of the planner is the fault?")
    # the dynamics half enters the predictor's flat top, along two empty corridors
    b += apoly([(880, PYL), (880, 120), (300, 120), (300, 190)], K, dash="6,4")
    b += txt(310, 106, "dynamics:  this predictor  OR  the simulator", 15.5,
             anchor="start", fill=ACCENT[K])
    # the objective half enters the MSE gutter between the two code stacks (x 723..801)
    # from below.  The label is right-anchored at 700 so it stops clear of the leader.
    b += apoly([(712, 378), (766, 378), (766, 350)], K, w=1.6, dash="5,3")
    b += txt(700, 383, "objective:  latent MSE  OR  true task distance", 15.5,
             anchor="end", fill=ACCENT[K])
    b = sch(b)

    PH = 496
    s = panel(PX, PY, PW, PH, K, "module:  swap one half at a time;  each cell says what "
              "its result would mean")
    X0, CW, GX = 152, 350, 24
    Y0, CH, GY = 966, 136, 22
    cx = (X0 + CW / 2, X0 + CW + GX + CW / 2)
    cy = (Y0 + CH / 2, Y0 + CH + GY + CH / 2)
    # -- headers
    for j, (lab, note) in enumerate((("OBJECTIVE  =  learned", "latent distance to z_goal"),
                                     ("OBJECTIVE  =  oracle",
                                      "true task distance, block pose"))):
        s += txt(cx[j], 932, lab, 16, weight="bold")
        s += txt(cx[j], 952, note, 12.5, fill=MUTED)
    # the note wraps onto two short lines: "simulator rolls state" set on one line
    # right-anchored here reached x=35, seven px from the panel border
    for j, (lab, n1, n2) in enumerate((("=  learned", "predictor", "rolls z"),
                                       ("=  oracle", "simulator", "rolls state"))):
        s += txt(X0 - 12, cy[j] - 32, "DYNAMICS", 13, anchor="end", weight="bold")
        s += txt(X0 - 12, cy[j] - 12, lab, 13, anchor="end")
        s += txt(X0 - 12, cy[j] + 8, n1, 11, anchor="end", fill=MUTED)
        s += txt(X0 - 12, cy[j] + 24, n2, 11, anchor="end", fill=MUTED)
    # -- the four cells.  Four identities, four hues: the measured failure, the two
    # single-swap probes, and the ceiling.  (They used to be crimson / green / amber /
    # teal, and the last two read as the same colour.)
    s += ladder_cell(X0, Y0, CW, CH, "crit", "AS SHIPPED",
                     "CEM 0.357 ltv / 0.080 linvar  --  measured",
                     f"high  {IMPLIES}  nothing below this is needed",
                     f"low   {IMPLIES}  the fault is in one of the other 3")
    s += ladder_cell(X0 + CW + GX, Y0, CW, CH, "magenta", "PROBE THE ROLLOUT",
                     "the M1 probe reads pose out of the model's roll",
                     f"high  {IMPLIES}  the roll carries it; the METRIC fails",
                     f"low   {IMPLIES}  the rollout loses the block entirely")
    s += ladder_cell(X0, Y0 + CH + GY, CW, CH, "amber", "SIM ROLLS, LATENT SCORES",
                     "the simulator rolls the true state, then encode",
                     f"high  {IMPLIES}  the metric is fine; DYNAMICS at fault",
                     f"low   {IMPLIES}  it cannot rank even PERFECT rollouts")
    s += ladder_cell(X0 + CW + GX, Y0 + CH + GY, CW, CH, "blue", "THE CEILING",
                     "simulator + true task distance -- no learning",
                     f"high  {IMPLIES}  the ceiling; the rest is model error",
                     f"low   {IMPLIES}  H=5 / 300 / 30 -- the PLANNER fails")
    s += strip(PX + 26, 1276, [
        ("privileged state", "eval only, never in loss", True),
        ("oracle dynamics", "the simulator itself", False),
        ("oracle objective", "block pose distance", False),
        ("measured today", f"{RHO}(latent, true) 0.398", True)], K, cw=198)
    return b + s, PY + PH + 28


# ------------------------------------------------------------- M4: error analysis
def m4_erroranalysis():
    """Four mutually exclusive buckets, reached by three ordered tests."""
    K = "amber"
    b = base("(M4) error analysis -- every failed episode into exactly one of four buckets")
    b += wire_to_obs(K, "the state trace of every FAILED episode", bottom=True)
    b = sch(b)

    PH = 464
    s = panel(PX, PY, PW, PH, K, "module:  three ordered tests, four exhaustive buckets "
              " --  no new training")
    QY, QH, QW = 930, 60, 148            # question row
    BY, BH, BW = 1046, 118, 172          # bucket row
    CXS = (254, 438, 622, 806)           # one x grid for both rows
    LAST = CXS[3]                        # the corridor the final "yes" runs down
    s += mbox(48, QY + 4, 112, 52, "failed", K, size=15, sub_="episodes")
    s += aw(160, 960, 174, 960, K)
    tests = (("contact?", "n_contacts &gt; 0"),
             ("toward the goal?", "d(block, goal) falls"),
             (f"position solved?", f"{BAR}dxy{BAR} &lt; 20 px"))
    for i, (lab, note) in enumerate(tests):
        s += mbox(CXS[i] - QW / 2, QY, QW, QH, lab, K, size=16, sub_=note)
        if i < 2:                        # continue right on the row baseline
            s += aw(CXS[i] + QW / 2, 960, CXS[i + 1] - QW / 2 - 4, 960, K)
            s += txt((CXS[i] + CXS[i + 1]) / 2, 950, "yes", 12, fill=MUTED)
        # the "no" answer drops on its own vertical corridor, straight down
        s += aw(CXS[i], QY + QH, CXS[i], BY - 4, "crit")
        s += txt(CXS[i] + 8, 1022, "no", 13, anchor="start", fill=ACCENT["crit"])
    # the last "yes" turns down into the fourth bucket, on its own corridor
    s += apoly([(CXS[2] + QW / 2, 960), (LAST, 960), (LAST, BY - 4)], K)
    s += txt((CXS[2] + QW / 2 + LAST) / 2, 950, "yes", 12, fill=MUTED)
    # four buckets, four hues -- crimson through to the near-miss
    s += bucket(CXS[0] - BW / 2, BY, BW, BH, "crit", "NO CONTACT",
                ["the plan never reaches", f"the object at all {NDASH}",
                 "a REACHING failure"])
    s += bucket(CXS[1] - BW / 2, BY, BW, BH, "magenta", "WRONG WAY",
                ["contact, then motion", f"AWAY from the goal {NDASH}",
                 "the sign is wrong"])
    s += bucket(CXS[2] - BW / 2, BY, BW, BH, "slate", "SHORTFALL",
                ["right direction, ran", f"out of horizon {NDASH}",
                 "H = 5 is what binds"])
    s += bucket(CXS[3] - BW / 2, BY, BW, BH, "blue", "ANGLE ONLY",
                [f"position solved, {THETA}", f"not {NDASH} the angular term",
                 "is what fails"])
    s += eqline(PX + 26, 1206, "the three tests are ordered and mutually exclusive, so the "
               "four counts partition the failures and sum to the failure rate.", 15)
    s += eqline(PX + 26, 1232, "n_contacts (env/pusht/pusht_env.py:578) and the per-step "
               "block pose are already computed; pusht_wrapper.rollout:103-118 drops "
               "both.", 14)
    s += strip(PX + 26, 1248, [
        ("bucket source", "n_contacts, pusht_env", True),
        ("never logged", "wrapper.rollout:103", True),
        ("also dropped", "coverage, reward, pose", True),
        ("cost", "widen the return tuple", False)], K, cw=198)
    return b + s, PY + PH + 28


FIGURES = (
    ("arch-m1-probe.svg", m1_probe),
    ("arch-m2-metric.svg", m2_metric),
    ("arch-m3-oracle.svg", m3_oracle),
    ("arch-m4-erroranalysis.svg", m4_erroranalysis),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "diary/assets/2026-09-03"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in FIGURES:
        body, h = fn()
        print("  wrote", a.out + "/" + emit(name, body, a.out, 940, h))


if __name__ == "__main__":
    main()
