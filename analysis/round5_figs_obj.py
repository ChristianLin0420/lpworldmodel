"""Architecture diagrams for the round-5 OBJECTIVE proposals (T4-T7) and the four defects.

Same construction as analysis/arch_figs_causal.py and analysis/arch_figs_predictor.py: the
unmodified LpWM schematic from base() on top, and a MODULE PANEL below carrying the proposal
as blocks, operator nodes and arrows, with the equation it implements underneath and a short
strip of measured values. No prose paragraphs inside a diagram.

T4  value    -- hindsight relabelling gives a goal-conditioned V( z , g ) its labels for free.
T5  valueeq  -- the model loss becomes a matched Bellman backup, with V frozen.
T6  jumpy    -- one predictor call spans k frames, instead of composing the 1-step operator.
T7  energy   -- a frozen encoder and a scalar energy that RANKS whole action sequences.
    defects  -- the four verified defects, one panel each, with file:line and consequence.

Layout rule that keeps the panels readable: every panel is a strict left-to-right dataflow on
fixed row baselines, and any wire that must change rows does so on its own vertical corridor.
No two wires share a segment. The a_t circle at (318, 430) is fenced in by the link/encoder/
RDMReg wiring, so a connector enters the predictor's flat top edge or the MSE gutter instead.

Nothing drawn here puts privileged state into a training loss.

Usage:  python analysis/round5_figs_obj.py --out diary/assets/2026-09-03
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, INK, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    aw, apoly, eq, mbox, nrm, opnode, pill, strip, sup,
)

PX, PW = 34, 872                      # panel x-extent, shared by every figure
MUTED = "#6b6b66"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-03")


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


# --- connectors into the unmodified schematic -------------------------------------
def hook(key, label, note=""):
    """Dashed connector into the objective the model minimises: the empty gutter BETWEEN
    the two code stacks (x 723..801), entered from BELOW their bottom edge (y 337) so it
    clears the MSE arrow and its label."""
    s = mbox(732, 312, 66, 36, label, key, size=15)
    s += apoly([(880, 898), (880, 366), (766, 366), (766, 352)], key, w=2.0, dash="6,4")
    s += apoly([(766, 312), (766, 284)], key, w=2.0, dash="6,4")
    if note:
        s += txt(864, 386, note, 14, anchor="end", fill=ACCENT[key], weight="bold")
    return s


def topedge(key, label):
    """The other legal entry point: the predictor's flat top edge, via the two empty
    corridors x=880 and y=120."""
    s = apoly([(880, 898), (880, 120), (300, 120), (300, 190)], key, w=2.0, dash="6,4")
    s += txt(310, 106, label, 16, anchor="start", fill=ACCENT[key])
    return s


def strikeout(cx, cy, key="crit", r=13, w=2.6):
    """A drawn cross -- the font has no multiplication glyph worth trusting."""
    c = ACCENT[key]
    return (f'<path d="M{cx - r},{cy - r} L{cx + r},{cy + r} M{cx + r},{cy - r} '
            f'L{cx - r},{cy + r}" stroke="{c}" stroke-width="{w}" '
            f'stroke-linecap="round"/>\n')


# --- local primitives -------------------------------------------------------------
def inset(x, y, w, h, title, key, note=""):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="2"/>\n')
    s += txt(x + w / 2, y + 26, title, 17, fill=c, weight="bold")
    if note:
        s += txt(x + w / 2, y + 45, note, 12, fill="#555")
    return s


def sgbar(cx, cy, key="crit", h=11, gap=4):
    """The stop-gradient mark: two bars across a wire."""
    c = ACCENT[key]
    return (f'<path d="M{cx - gap},{cy - h} L{cx - gap},{cy + h} M{cx + gap},{cy - h} '
            f'L{cx + gap},{cy + h}" stroke="{c}" stroke-width="2.6" '
            f'stroke-linecap="round"/>\n')


def freeze(cx, cy, key, r=12, w=2.0):
    """A drawn snowflake: 'these weights do not move'. Geometry, not a glyph."""
    c = ACCENT[key]
    s = ""
    for deg in (90.0, 30.0, 150.0):
        a = math.radians(deg)
        dx, dy = r * math.cos(a), -r * math.sin(a)
        s += (f'<path d="M{cx - dx:.1f},{cy - dy:.1f} L{cx + dx:.1f},{cy + dy:.1f}" '
              f'stroke="{c}" stroke-width="{w}" stroke-linecap="round"/>\n')
        for sgn in (1.0, -1.0):
            ex, ey = cx + sgn * dx, cy + sgn * dy
            for off in (140.0, -140.0):
                b = math.radians(deg + off)
                bx = ex + sgn * 0.42 * r * math.cos(b)
                by = ey - sgn * 0.42 * r * math.sin(b)
                s += (f'<path d="M{ex:.1f},{ey:.1f} L{bx:.1f},{by:.1f}" stroke="{c}" '
                      f'stroke-width="{w * 0.8:.2f}" stroke-linecap="round"/>\n')
    return s


def cells(x, y, pattern, key, cell=18, gap=2, ghost=False):
    """A little code stack drawn horizontally: 1 = active."""
    c = ACCENT[key]
    s = ""
    for i, v in enumerate(pattern):
        f = (ACCENT[key] if v else WHITE) if not ghost else WHITE
        d = ' stroke-dasharray="4,3"' if ghost else ""
        s += (f'<rect x="{x + i * (cell + gap)}" y="{y}" width="{cell}" height="{cell}" '
              f'fill="{f}" fill-opacity="{0.85 if v and not ghost else 1}" '
              f'stroke="{"#9a9a94" if ghost else c}" stroke-width="1.4"{d}/>\n')
    return s


def tick(cx, cy, key, r=9):
    c = ACCENT[key]
    s = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{c}" '
         f'stroke-width="1.6"/>\n')
    s += (f'<path d="M{cx - 4.5},{cy} L{cx - 1},{cy + 3.8} L{cx + 4.8},{cy - 4.2}" '
          f'stroke="{c}" stroke-width="2" fill="none" stroke-linecap="round" '
          f'stroke-linejoin="round"/>\n')
    return s


def checklist(x, y, w, items, key, h=58):
    """The NEW HEAD CHECKLIST: the lines path_int was missing, one ticked cell each."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.8"/>\n')
    s += txt(x + 14, y + 24, "NEW HEAD CHECKLIST", 13, anchor="start", fill=c,
             weight="bold")
    s += txt(x + 14, y + 44, "path_int had NONE of these", 11.5, anchor="start",
             fill=ACCENT["crit"])
    x0, cw = x + 196, (w - 206) / len(items)
    for i, (lab, where) in enumerate(items):
        cx = x0 + i * cw
        s += tick(cx + 13, y + 22, key)
        s += txt(cx + 30, y + 27, lab, 13, anchor="start")
        s += txt(cx + 30, y + 46, where, 11, anchor="start", fill="#555")
    return s


# ------------------------------------------------------------------------ T4: value
def t4_value():
    """A goal-conditioned value head. Hindsight relabelling supplies the labels."""
    K = "blue"
    b = base("(T4) PiWM-value -- a goal-conditioned V( z , g ) trained by hindsight "
             "relabelling")
    b += hook(K, "V", "a NEW head on the codes")

    PY, PH = 898, 600
    s = panel(PX, PY, PW, PH, K,
              "module:  every later latent is a goal;  the step count is the label")
    s += txt(470, 962, "hindsight relabelling -- no reward, no extra data, no privileged "
             "state", 13, fill="#555")

    # -- row A: the trajectory, read as a set of labelled (state, goal) pairs
    xs = (96, 214, 332, 450, 568, 686)
    labs = (sub("z", "t", 11), sub("z", "t+1", 10), sub("z", "t+2", 10),
            sub("z", "t+3", 10), sub("z", "t+4", 10), sub("z", "t+5", 10))
    for i, (cx, lab) in enumerate(zip(xs, labs)):
        s += pill(cx, 1006, lab, K, rx=34, ry=17,
                  fill=ACCENT_FILL[K] if i in (0, 3) else WHITE)
        if i:
            s += aw(xs[i - 1] + 34, 1006, cx - 34, 1006, K)
            s += txt(cx, 1044, f"-{i}", 13, fill=ACCENT[K], weight="bold")
    s += txt(450, 984, "goal g", 13, fill=ACCENT[K], weight="bold")
    s += txt(792, 1044, "label = steps to the goal", 12, fill="#555")

    # -- z_t drops down its own corridor and fans to both rows off one spine
    s += (f'<path d="M96,1023 L96,1140 L136,1140 M136,1128 L136,1238" '
          f'stroke="{ACCENT[K]}" stroke-width="2" fill="none"/>\n')
    s += aw(136, 1128, 158, 1128, K)
    s += aw(136, 1238, 158, 1238, K)
    # -- the goal drops out of node 3's bottom edge on its own corridor (x = 470,
    # clear of the -3 label at x = 450) and runs in along the empty y = 1066 line
    s += apoly([(470, 1023), (470, 1066), (191, 1066), (191, 1100)], K)

    # -- row B: the prediction  V( z , g )
    s += mbox(162, 1104, 58, 48, "[ ; ]", K, size=17)
    s += mbox(244, 1104, 146, 48, "V", K, size=20, sub_="MLP ( 2 x 512 )")
    s += aw(220, 1128, 240, 1128, K)
    s += pill(466, 1128, "V( z , g )", K, rx=58, fill=WHITE)
    s += aw(390, 1128, 404, 1128, K)
    s += apoly([(524, 1128), (760, 1128), (760, 1167)], K)

    # -- row C: the TD target, with an expectile in place of the max
    s += mbox(162, 1214, 106, 48, "P", K, size=20, sub_="predictor")
    s += txt(215, 1204, "K sampled actions", 11.5, fill="#555")
    for k, yy in enumerate((1198, 1226, 1254)):
        s += mbox(292, yy, 108, 24, "V-bar", K, size=13)
        s += aw(268, 1238 + (-10 if k == 0 else (0 if k == 1 else 10)),
                288, yy + 12, K, w=1.6)
        s += aw(400, yy + 12, 420, 1238 + (-10 if k == 0 else (0 if k == 1 else 10)),
                K, w=1.6)
    s += mbox(424, 1214, 132, 48, "max_a", K, size=18, sub_="expectile, tau")
    s += pill(660, 1238, "y  =  -1 + gamma m", K, rx=86, fill=ACCENT_FILL[K])
    s += aw(556, 1238, 570, 1238, K)
    s += apoly([(746, 1238), (760, 1238), (760, 1199)], K)
    s += txt(346, 1292, "V-bar  =  a frozen copy of V", 11.5, fill="#555")

    # -- the TD error
    s += opnode(760, 1183, "minus", K)
    s += aw(776, 1183, 792, 1183, K)
    s += mbox(796, 1159, 92, 48, nrm("."), K)
    s += txt(842, 1228, "L_V", 16, fill=ACCENT[K], weight="bold")

    s += eq(PX + 26, 1326, "V( z , g )  =  MLP( [ z ; g ] )", 17)
    s += eq(PX + 340, 1326, "hindsight:   g  =  z_t+k ,    label  =  -k", 17)
    s += eq(PX + 26, 1352, "y  =  -1  +  gamma . max_a  V-bar( P( z , a ) , g )", 16)
    s += eq(PX + 430, 1352, "expectile:   L_tau( d )  =  | tau - 1[ d &lt; 0 ] | . "
            + sup("d", "2"), 16)
    s += checklist(PX + 26, 1366, PW - 52, [
        ("optimizer group", "train.py init_optimizers"),
        ("checkpoint key", "_keys_to_save"),
        ("lazy .to(device)", "built inside VWorldModel"),
        ("arm mirror", "tests/lpwm_build.py")], K)
    s += strip(PX + 26, 1432, [
        ("relabelled pairs", "free, from the demos", False),
        ("latent vs task dist", "Spearman +0.398", True),
        ("the V5 planner needs", "exactly this head", False),
        ("path_int precedent", "3 lines missing", True)], K, cw=198)
    return b + s


# ---------------------------------------------------------------------- T5: valueeq
def t5_valueeq():
    """The model loss becomes a matched Bellman backup instead of a latent MSE."""
    K = "green"
    b = base("(T5) PiWM-valueeq -- match Bellman backups, not next latents")
    b += hook(K, "V", "V REPLACES the MSE")
    b += strikeout(766, 260, "crit", r=13)

    PY, PH = 898, 580
    s = panel(PX, PY, PW, PH, K,
              "module:  the model is scored by the VALUE it implies, not by the latent "
              "it emits")

    # -- row A: the model's own backup
    s += pill(84, 1014, sub("z", "t", 11), K, rx=30, ry=16)
    s += circle(84, 1060, 20, sub("a", "t", 10), fill=ACCENT_FILL[K], size=13)
    s += mbox(136, 1008, 96, 48, "P", K, size=20, sub_="predictor")
    s += aw(114, 1014, 132, 1022, K)
    s += aw(106, 1054, 132, 1044, K)
    s += pill(296, 1032, "zhat_t+1", K, rx=46, fill=WHITE)
    s += aw(232, 1032, 246, 1032, K)
    s += mbox(400, 1008, 150, 48, "V-bar( . , g )", K, size=16)
    s += aw(342, 1032, 396, 1032, K)
    s += pill(620, 1032, "v_pred", K, rx=46, fill=WHITE)
    s += aw(550, 1032, 570, 1032, K)
    s += apoly([(666, 1032), (742, 1032), (742, 1071)], K)

    # -- row B: the environment's backup
    s += circle(84, 1150, 24, sub("o", "t+1", 10), size=14)
    s += mbox(136, 1126, 96, 48, "Enc", K, size=18, sub_="trained")
    s += aw(108, 1150, 132, 1150, K)
    s += pill(296, 1150, sub("z", "t+1", 11), K, rx=46, fill=WHITE)
    s += aw(232, 1150, 246, 1150, K)
    s += mbox(400, 1126, 150, 48, "V-bar( . , g )", K, size=16)
    s += aw(342, 1150, 396, 1150, K)
    s += sgbar(370, 1150)
    s += txt(360, 1190, "stop-grad", 11, fill=ACCENT["crit"], weight="bold")
    s += pill(620, 1150, "v_tgt", K, rx=46, fill=WHITE)
    s += aw(550, 1150, 570, 1150, K)
    s += apoly([(666, 1150), (742, 1150), (742, 1103)], K)

    # -- the residual
    s += opnode(742, 1087, "minus", K)
    s += aw(758, 1087, 774, 1087, K)
    s += mbox(778, 1063, 110, 48, nrm("."), K)
    s += txt(833, 1128, "L_veq", 16, fill=ACCENT[K], weight="bold")
    s += txt(475, 1094, "V-bar  =  V from T4, its weights FROZEN", 12.5,
             fill=ACCENT[K], weight="bold")

    # -- inset: what metric that induces
    s += inset(60, 1210, 440, 134, "only value-relevant error costs anything", K)
    s += eq(80, 1272, "L  =  " + sup("( v_pred  -  v_tgt )", "2"), 16)
    s += eq(80, 1302, "=  dz^T M dz  +  higher order", 16)
    s += eq(80, 1330, "M  =  grad V grad V^T   --   rank 1 in D = 384", 14)

    s += inset(520, 1210, 356, 134, "the metric it induces", K)
    s += txt(698, 1252, "level sets of V( . , g )", 11.5, fill="#555")
    for i in range(8):
        xx = 552 + i * 40
        s += (f'<path d="M{xx},1256 L{xx},1310" stroke="{ACCENT[K]}" stroke-width="1.3" '
              f'stroke-opacity="0.55"/>\n')
    s += aw(572, 1306, 572, 1264, "slate", w=1.8)
    s += txt(572, 1330, "no-op", 11, fill="#555")
    s += aw(612, 1282, 832, 1282, K, w=2.4)
    s += txt(730, 1330, "moves the block", 11, fill=ACCENT[K], weight="bold")

    s += eq(PX + 26, 1374, sup("L_veq  =  ( V-bar( P( z , a ) , g ) - V-bar( z_t+1 , g ) )",
                               "2"), 16)
    s += eq(PX + 566, 1374, "the target side is detached", 15)
    s += eq(PX + 26, 1400, "the latent MSE weights every transition equally; this one "
            "weights each by how much the VALUE moves -- so the 48% no-ops cost ~0", 14)
    s += strip(PX + 26, 1414, [
        ("fully static", "48% of transitions", True),
        ("block moves &gt; 0.5 px", "25.3%", True),
        ("top 1% carry", "34.8% of all motion", True),
        ("V comes from", "T4, frozen", False)], K, cw=198)
    return b + s


# ------------------------------------------------------------------------ T6: jumpy
def t6_jumpy():
    """One predictor call spanning k frames, beside the k-step compounded rollout."""
    K = "amber"
    b = base("(T6) PiWM-jumpy -- predict z_t+5 in ONE call, not five composed calls")
    b += topedge(K, "one predictor call spans 5 frames")

    PY, PH = 898, 560
    s = panel(PX, PY, PW, PH, K,
              "module:  the horizon CEM plans over is the horizon the loss is written at")

    # -- row A: today. the 1-step operator, composed five times.
    s += txt(64, 966, "TODAY", 14, anchor="start", fill="#555", weight="bold")
    s += pill(80, 1008, sub("z", "t", 11), K, rx=30, ry=17)
    s += aw(110, 1008, 124, 1008, K)
    bars = (4, 7, 11, 16, 22)
    for k in range(5):
        gx = 128 + k * 152
        s += circle(gx + 24, 952, 15, sub("a", str(k + 1), 9),
                    fill=ACCENT_FILL[K], size=12)
        s += aw(gx + 24, 967, gx + 24, 980, K, w=1.6)
        s += mbox(gx, 984, 48, 48, "g", K, size=19)
        s += aw(gx + 48, 1008, gx + 62, 1008, K)
        cx = gx + 100
        s += pill(cx, 1008, "zhat" + sub("", str(k + 1), 9), K, rx=34, ry=17, fill=WHITE)
        if k < 4:
            s += aw(cx + 34, 1008, gx + 152, 1008, K)
        s += (f'<rect x="{cx - 14}" y="1034" width="28" height="{bars[k]}" '
              f'fill="{ACCENT["crit"]}" fill-opacity="0.65" stroke="{ACCENT["crit"]}" '
              f'stroke-width="1"/>\n')
    s += txt(836, 1074, "error compounds", 11.5, fill=ACCENT["crit"], weight="bold")

    # -- row B: one call
    s += apoly([(80, 1025), (80, 1130), (124, 1130)], K)
    s += txt(64, 1156, "T6", 14, anchor="start", fill=ACCENT[K], weight="bold")
    for k in range(5):
        s += circle(146 + k * 34, 1078, 13, sub("a", str(k + 1), 8),
                    fill=ACCENT_FILL[K], size=11)
    s += aw(216, 1091, 216, 1102, K, w=1.6)
    s += mbox(128, 1106, 176, 48, "g", K, size=20, sub_="one call,  k = 5")
    s += pill(390, 1130, "zhat_t+5", K, rx=52, fill=WHITE)
    s += aw(304, 1130, 334, 1130, K)
    s += pill(560, 1078, sub("z", "t+5", 11), K, rx=44)
    s += opnode(560, 1130, "minus", K)
    s += aw(442, 1130, 544, 1130, K)
    s += aw(560, 1097, 560, 1114, K)
    s += mbox(600, 1106, 110, 48, nrm("."), K)
    s += aw(576, 1130, 596, 1130, K)
    s += pill(802, 1130, "L_jump", K, rx=58, fill=ACCENT_FILL[K])
    s += aw(710, 1130, 740, 1130, K)

    # -- what the config pins today
    s += inset(60, 1196, 816, 124, "what the config pins today", K)
    for i, (lab, kk) in enumerate((("num_pred = 1", "crit"), ("num_hist = 3", K),
                                   ("frameskip = 5", K), ("CEM horizon = 5", K))):
        s += mbox(108 + i * 184, 1238, 168, 44, lab, kk, size=15,
                  fill=ACCENT_FILL["crit"] if kk == "crit" else WHITE)
    s += txt(468, 1306, "the training loss has never evaluated a 5-step rollout", 13,
             fill=ACCENT[K], weight="bold")

    s += eq(PX + 26, 1354, "today:   zhat_t+H  =  ( h o g )^H ( z_t ; a_1:H )", 16)
    s += eq(PX + 400, 1354, "T6:   zhat_t+k  =  g_k ( z_t ; a_1:k )", 16)
    s += eq(PX + 26, 1380, "L  =  SUM_k  " + nrm("zhat_t+k - z_t+k"), 16)
    s += eq(PX + 300, 1380, "k in { 1 , 5 } -- one head per jump, or k as a "
            "conditioning input", 14)
    s += strip(PX + 26, 1394, [
        ("num_pred today", "1", True),
        ("CEM horizon", "5", False),
        ("frameskip", "5  (act dim 10)", False),
        ("what compounds", "5 x ( h o g )", True)], K, cw=198)
    return b + s


# ----------------------------------------------------------------------- T7: energy
def t7_energy():
    """A scalar energy over WHOLE action sequences, on a frozen encoder."""
    K = "magenta"
    b = base("(T7) PiWM-energy -- a frozen encoder and a scalar energy over action "
             "sequences")
    b += hook(K, "E", "E replaces the MSE")
    # the freeze, drawn on both encoder domes
    for cx in (120, 812):
        b += freeze(cx, 556, K, r=13)
        b += txt(cx, 588, "frozen", 12, fill=ACCENT[K], weight="bold")

    PY, PH = 898, 592
    s = panel(PX, PY, PW, PH, K,
              "module:  E ranks the EXECUTED sequence below the ones CEM would sample")

    # -- the conditioning both branches share
    s += pill(72, 1084, sub("z", "t", 11), K, rx=26, ry=15)
    s += pill(72, 1132, sub("z", "g", 11), K, rx=26, ry=15)
    s += mbox(126, 1084, 54, 48, "[ ; ]", K, size=17)
    s += aw(98, 1084, 122, 1094, K)
    s += aw(98, 1132, 122, 1122, K)
    s += (f'<path d="M180,1108 L206,1108 M206,1054 L206,1162" stroke="{ACCENT[K]}" '
          f'stroke-width="2" fill="none"/>\n')
    s += aw(206, 1054, 226, 1054, K)
    s += aw(206, 1162, 226, 1162, K)

    # -- positives: the sequence the demo actually executed
    s += txt(290, 972, "positives:  the EXECUTED sequence", 12.5, fill=ACCENT[K],
             weight="bold")
    for k in range(5):
        s += circle(238 + k * 26, 1000, 11, sub("a", str(k + 1), 8),
                    fill=ACCENT_FILL[K], size=10)
    s += aw(290, 1011, 290, 1026, K, w=1.6)
    s += mbox(230, 1030, 120, 48, "E", K, size=20, sub_="scalar head")
    s += pill(420, 1054, "E+", K, rx=38, fill=ACCENT_FILL[K])
    s += aw(350, 1054, 378, 1054, K)

    # -- negatives: CEM's own proposal
    for k in range(5):
        s += circle(238 + k * 26, 1106, 11, "", fill=WHITE, size=10)
    s += txt(366, 1110, "K sampled  a~_1:H", 11, anchor="start", fill="#555")
    s += aw(290, 1117, 290, 1134, K, w=1.6)
    s += mbox(230, 1138, 120, 48, "E", K, size=20, sub_="the SAME weights")
    for k, yy in enumerate((1136, 1162, 1188)):
        s += pill(420, yy, ["E-_1", "E-_2", "E-_K"][k], K, rx=34, ry=12, fill=WHITE)
        s += aw(350, 1162 + (-10 if k == 0 else (0 if k == 1 else 10)),
                384, yy, K, w=1.6)
    s += txt(290, 1216, "negatives:  the sequences CEM would sample", 11.5, fill="#555")

    # -- the ranking loss
    s += mbox(500, 1078, 180, 60, "softmax rank", K, size=16,
              sub_="E+ must be the lowest")
    s += apoly([(458, 1054), (478, 1054), (478, 1090), (496, 1090)], K)
    s += apoly([(454, 1162), (478, 1162), (478, 1126), (496, 1126)], K)
    s += pill(790, 1108, "L_rank", K, rx=60, fill=ACCENT_FILL[K])
    s += aw(680, 1108, 726, 1108, K)

    # -- why V2's collapse is structurally unavailable
    s += inset(60, 1240, 816, 116, "why the V2 collapse cannot recur", K)
    s += txt(250, 1290, "V2 / P5:   the encoder was TRAINABLE", 12.5,
             fill=ACCENT["crit"], weight="bold")
    s += cells(96, 1300, (1, 1, 0, 1, 1, 0), "crit")
    s += aw(214, 1309, 262, 1309, "crit", w=2.0)
    s += cells(272, 1300, (0, 0, 0, 1, 0, 0), "crit")
    s += txt(250, 1342, "rho  0.448  ->  0.052", 12, fill=ACCENT["crit"], weight="bold")
    s += (f'<path d="M468,1278 L468,1344" stroke="{ACCENT[K]}" stroke-width="1.2" '
          f'stroke-opacity="0.5"/>\n')
    s += txt(668, 1290, "T7:   the encoder is FROZEN", 12.5, fill=ACCENT[K],
             weight="bold")
    s += cells(520, 1300, (1, 1, 0, 1, 1, 0), K)
    s += freeze(648, 1309, K, r=9, w=1.6)
    s += aw(668, 1309, 716, 1309, "slate", w=1.8, dash="6,4")
    s += strikeout(692, 1309, "crit", r=9, w=2.2)
    s += cells(726, 1300, (0, 0, 0, 1, 0, 0), K, ghost=True)
    s += txt(668, 1342, "z is a CONSTANT of the optimisation", 12, fill=ACCENT[K],
             weight="bold")

    s += eq(PX + 26, 1386, "E :  ( z , a_1:H , z_g )   ->   scalar cost-to-go", 17)
    s += eq(PX + 400, 1386, "z = Enc( o ) with Enc frozen -- 0 trainable encoder params",
            15)
    s += eq(PX + 26, 1412, "L_rank  =  -log [ exp( -E+ )  /  ( exp( -E+ )  +  SUM_k "
            "exp( -E-_k ) ) ]", 16)
    s += eq(PX + 590, 1412, "negatives where CEM searches", 15)
    s += strip(PX + 26, 1426, [
        ("what trains", "E only", False),
        ("encoder", "frozen, 0 grads", False),
        ("V2 collapse mode", "rho 0.448 -> 0.052", True),
        ("negatives from", "the CEM proposal", False)], K, cw=198)
    return b + s


# ------------------------------------------------------------------- the four defects
DW, DH = 940, 900
QW, QH = 438, 384
QX = (26, 476)
QY = (78, 486)


def quad(i, j, key, tag, title, where):
    """Quadrant frame: number chip, title, and the file:line that proves it."""
    x, y, c = QX[j], QY[i], ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{QW}" height="{QH}" rx="10" '
         f'fill="{ACCENT_FILL[key]}" fill-opacity="0.4" stroke="{c}" '
         f'stroke-width="2.4"/>\n')
    s += (f'<rect x="{x + 14}" y="{y + 12}" width="30" height="24" rx="5" fill="{c}"/>\n')
    s += txt(x + 29, y + 30, tag, 15, fill=WHITE, weight="bold")
    s += txt(x + 52, y + 30, title, 15, anchor="start", fill=c, weight="bold")
    s += txt(x + 52, y + 48, where, 11.5, anchor="start", fill=MUTED, style="italic")
    s += (f'<path d="M{x + 14},{y + 58} L{x + QW - 14},{y + 58}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def code(x, y, w, lines, key, size=11.5):
    """A monospace source excerpt."""
    c = ACCENT[key]
    h = 13 + 18 * len(lines)
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{WHITE}" '
         f'stroke="{c}" stroke-width="1.3" stroke-opacity="0.65"/>\n')
    for k, ln in enumerate(lines):
        s += (f'<text x="{x + 11}" y="{y + 21 + 18 * k}" font-size="{size}" '
              f'font-family="monospace" fill="{INK}">{ln}</text>\n')
    return s


def verdict(x, y, w, text, key, size=13.5):
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="32" rx="5" fill="{c}" '
         f'fill-opacity="0.14" stroke="{c}" stroke-width="1.6"/>\n')
    return s + txt(x + w / 2, y + 21, text, size, fill=c, weight="bold")


def d1():
    """The success metric scores the agent's own position."""
    K, x, y = "crit", QX[0], QY[0]
    s = quad(0, 0, K, "1", "Success scores the agent's own position",
             "env/pusht/pusht_wrapper.py:62")
    s += code(x + 18, y + 72, QW - 36, [
        "pos_diff = norm(goal_state[:4] - cur_state[:4])"], K)
    yb, cw = 204, 52
    x0 = x + (QW - (7 * cw + 6 * 4)) / 2
    for k, (lab, agent) in enumerate((("agent x", True), ("agent y", True),
                                      ("T x", False), ("T y", False),
                                      ("T theta", False), ("vel", False),
                                      ("vel", False))):
        cx, scored = x0 + k * (cw + 4), k < 4
        s += (f'<rect x="{cx}" y="{yb}" width="{cw}" height="40" rx="4" '
              f'fill="{ACCENT[K] if agent else WHITE}" '
              f'fill-opacity="{0.85 if agent else 1}" stroke="{ACCENT[K]}" '
              f'stroke-width="{2.2 if scored else 1.1}" '
              f'stroke-opacity="{1 if scored else 0.45}"/>\n')
        s += txt(cx + cw / 2, yb + 25, lab, 10.5,
                 fill=WHITE if agent else (INK if scored else MUTED))
    xl, xr = x0, x0 + 4 * (cw + 4) - 4
    s += (f'<path d="M{xl},{yb + 50} L{xl},{yb + 60} L{xr},{yb + 60} L{xr},{yb + 50}" '
          f'stroke="{ACCENT[K]}" stroke-width="2" fill="none"/>\n')
    s += txt((xl + xr) / 2, yb + 84, "state[:4]  --  the scored ball", 13,
             fill=ACCENT[K], weight="bold")
    s += txt(x + QW / 2, yb + 112,
             "2 of the 4 scored dimensions are the agent's own", 12.5)
    s += txt(x + QW / 2, yb + 130, "position: directly actuated, reached in one step,",
             12.5)
    s += txt(x + QW / 2, yb + 148, "no contact and no physics", 12.5)
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "half the metric never requires touching the T", K)
    return s


def d2():
    """Success is latched over up to ten replans."""
    K, x, y = "amber", QX[1], QY[0]
    s = quad(0, 1, K, "2", "Success is latched, not terminal",
             "planning/mpc.py:110-112, 60-70")
    s += code(x + 18, y + 72, QW - 36, [
        "while not all(is_success) and iter &lt; max_iter:",
        "&#160;&#160;&#160;&#160;self.is_success |= successes"], K)
    yb, x0 = 252, x + 34
    for k in range(10):
        cx, hit = x0 + k * 38, k == 5
        s += (f'<circle cx="{cx}" cy="{yb}" r="12" '
              f'fill="{ACCENT[K] if hit else WHITE}" stroke="{ACCENT[K]}" '
              f'stroke-width="{2.4 if hit else 1.4}"/>\n')
        s += txt(cx, yb + 4, str(k + 1), 11, fill=WHITE if hit else MUTED)
        if k < 9:
            s += aw(cx + 13, yb, cx + 24, yb, K, w=1.3)
    s += txt(x0 + 5 * 38, yb - 26, "in the ball once", 12, fill=ACCENT[K], weight="bold")
    s += (f'<path d="M{x0 + 5 * 38},{yb - 20} L{x0 + 5 * 38},{yb - 14}" '
          f'stroke="{ACCENT[K]}" stroke-width="1.8"/>\n')
    s += txt(x + QW / 2, yb + 42, "reported success  =  OR over all ten replans", 13.5,
             weight="bold")
    s += txt(x + QW / 2, yb + 64, "a maximum over noisy draws, so the number rises", 12.5)
    s += txt(x + QW / 2, yb + 82, "with max_iter, with the model held fixed", 12.5)
    s += txt(x + QW / 2, yb + 106, "_apply_success_mask then freezes the episode", 11.5,
             fill=MUTED)
    s += txt(x + QW / 2, yb + 122, "at the normalised hold ( +0.0431 , -0.0340 )", 11.5,
             fill=MUTED)
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "not \"ended at the goal\" but \"passed through it\"", K)
    return s


def d3():
    """path_int has no optimizer, no checkpoint key, and is absent from every ckpt."""
    K, x, y = "magenta", QX[0], QY[1]
    s = quad(1, 0, K, "3", "path_int has no optimizer, no checkpoint",
             "train.py init_optimizers / _keys_to_save; visual_world_model.py:383-389")
    s += txt(x + QW / 2, 570, "optimizer groups built by init_optimizers", 13,
             fill=ACCENT[K], weight="bold")
    x0 = x + (QW - (4 * 92 + 3 * 8)) / 2
    for k, lab in enumerate(("encoder", "predictor", "decoder", "act / prop")):
        s += mbox(x0 + k * 100, 582, 92, 40, lab, K, size=12.5)
    s += (f'<rect x="{x + 18}" y="646" width="148" height="40" rx="5" fill="{WHITE}" '
          f'stroke="{ACCENT[K]}" stroke-width="2" stroke-dasharray="7,5"/>\n')
    s += txt(x + 92, 671, "path_int", 15, fill=MUTED)
    for k, ln in enumerate(("no optimizer group", "no _keys_to_save entry",
                            "absent from every checkpoint")):
        s += txt(x + 206, 660 + k * 18, ln, 12, anchor="start", fill=ACCENT[K])
    s += code(x + 18, 706, QW - 36, [
        "loss = (path_int(x) - proprio.detach()) ** 2",
        "# both leaves are data: it reaches no parameter"], K, size=11)
    s += txt(x + QW / 2, 790, "_roll_pose then rolls a RANDOM nn.Linear at plan time",
             12.5, fill=ACCENT[K], weight="bold")
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "V3 tested a random rollout head, not path integration", K, size=12.5)
    return s


def d4():
    """No decoder was ever attached, and the branch that exists detaches."""
    K, x, y = "blue", QX[1], QY[1]
    s = quad(1, 1, K, "4", "No decoder has ever been attached",
             "conf/train_rdmreg.yaml:126, train.py:463, visual_world_model.py:728")
    yb = 596
    s += circle(x + 52, yb, 26, sub("o", "t", 11), size=15)
    s += mbox(x + 106, yb - 26, 92, 52, "encoder", K, size=14)
    s += aw(x + 80, yb, x + 102, yb, K)
    s += pill(x + 250, yb, sub("z", "t", 11), K, rx=28, ry=17)
    s += aw(x + 198, yb, x + 218, yb, K)
    s += (f'<rect x="{x + 330}" y="{yb - 26}" width="92" height="52" rx="5" '
          f'fill="{WHITE}" stroke="{ACCENT[K]}" stroke-width="2" '
          f'stroke-dasharray="7,5" stroke-opacity="0.6"/>\n')
    s += txt(x + 376, yb + 5, "decoder", 14, fill=MUTED)
    s += aw(x + 278, yb, x + 326, yb, K, w=1.6, dash="7,5")
    s += txt(x + 376, yb - 40, "has_decoder: False", 11, fill=ACCENT[K], weight="bold")
    s += txt(x + 376, yb + 44, "and the adaln branch", 11, fill=MUTED)
    s += txt(x + 376, yb + 58, "passes z_emb.detach()", 11, fill=MUTED)
    s += txt(x + QW / 2, 690, "the only pressure on the encoder is that its own", 12.5)
    s += txt(x + QW / 2, 708, "output be predictable from its own past", 12.5)
    s += txt(x + QW / 2, 734, "RDMReg blocks DIMENSIONAL collapse", 13,
             fill=ACCENT[K], weight="bold")
    s += txt(x + QW / 2, 752, "-- a full-rank latent can still encode only proprio,",
             12)
    s += txt(x + QW / 2, 770, "which is a free input to the predictor", 12)
    s += verdict(x + 18, y + QH - 46, QW - 36,
                 "nothing has required the latent to keep visual content", K, size=12.5)
    return s


def defects_4():
    s = txt(DW / 2, 40, "Four verified defects that precede every model question", 22,
            weight="bold")
    s += txt(DW / 2, 62, "each read from the source, not inferred -- 3 retracts a "
             "published conclusion, 4 explains four separate collapses", 13, fill=MUTED)
    return s + d1() + d2() + d3() + d4()


FIGURES = (
    ("arch-t4-value.svg", t4_value, (940, 1524)),
    ("arch-t5-valueeq.svg", t5_valueeq, (940, 1504)),
    ("arch-t6-jumpy.svg", t6_jumpy, (940, 1484)),
    ("arch-t7-energy.svg", t7_energy, (940, 1516)),
    # arch-defects-4.svg is emitted by analysis/arch_figs_defects.py, which owns that
    # filename. Two modules writing one path is how a figure silently reverts; the
    # body below is kept only so the layout work is not lost.
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
