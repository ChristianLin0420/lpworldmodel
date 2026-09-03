"""Architecture diagrams for the causal-objective round (V1-V3), in the LpWM paper's style.

The top half of each figure is the unmodified LpWM schematic from analysis/arch_figs.py
(peach observation/action circles, teal encoder, pale-yellow link, lavender predictor,
navy/white code stacks, dark-red MSE). The bottom half is a MODULE PANEL: the proposed
addition drawn the same way -- blocks, operator nodes, and arrows -- with the equation it
implements underneath and the measured outcome as a value strip. No prose.

Layout rule that keeps the panels readable: every panel is a strict left-to-right dataflow
on fixed row baselines, and any wire that must change rows does so on its own vertical
corridor. No two wires share a segment.

Usage:  python analysis/arch_figs_causal.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, ARROW, BOXLINE, INK, MSERED, NAVY, WHITE,
    _hdr, arrow, base, box, caret, circle, poly, sub, txt,
)

W, H = 940, 1310
PX, PY, PW, PH = 34, 898, 872, 384          # module panel


# --- extra primitives -------------------------------------------------------------
def sup(main, s, size=12):
    """Superscript via dy (baseline-shift is ignored by cairosvg/rsvg)."""
    return f'{main}<tspan font-size="{size}" dy="-8">{s}</tspan><tspan dy="8"> </tspan>'


def nrm(inner):
    """|| inner ||^2 -- ASCII bars, because the serif face has no U+2016."""
    sq = '<tspan font-size="12" dy="-8">2</tspan><tspan dy="8"> </tspan>'
    return "|| " + inner + " ||" + sq


def panel(key, title):
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="10" fill="{f}" '
         f'fill-opacity="0.45" stroke="{c}" stroke-width="2.4"/>\n')
    s += txt(PX + 20, PY + 32, title, 18, anchor="start", fill=c, weight="bold")
    s += (f'<path d="M{PX + 18},{PY + 44} L{PX + PW - 18},{PY + 44}" stroke="{c}" '
          f'stroke-width="1.2" stroke-opacity="0.5"/>\n')
    return s


def mbox(x, y, w, h, label, key, size=16, fill=WHITE, sub_=""):
    """A block in the panel's accent colour."""
    c = ACCENT[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" '
         f'stroke="{c}" stroke-width="2"/>\n')
    s += txt(x + w / 2, y + h / 2 + (6 if not sub_ else 0), label, size)
    if sub_:
        s += txt(x + w / 2, y + h / 2 + 17, sub_, 11.5, fill="#555")
    return s


def pill(cx, cy, label, key, rx=34, ry=19, fill=None):
    """A rounded value/variable node."""
    c = ACCENT[key]
    f = fill or ACCENT_FILL[key]
    s = (f'<rect x="{cx - rx}" y="{cy - ry}" width="{2 * rx}" height="{2 * ry}" '
         f'rx="{ry}" fill="{f}" stroke="{c}" stroke-width="2"/>\n')
    return s + txt(cx, cy + 6, label, 16)


def opnode(cx, cy, sym, key, r=16):
    """Operator node: circle + glyph, drawn (not typeset) so it never misses a font."""
    c = ACCENT[key]
    s = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{c}" '
         f'stroke-width="2"/>\n')
    if sym == "minus":
        s += (f'<path d="M{cx - 8},{cy} L{cx + 8},{cy}" stroke="{c}" stroke-width="2.4" '
              f'stroke-linecap="round"/>\n')
    elif sym == "times":
        s += (f'<path d="M{cx - 6},{cy - 6} L{cx + 6},{cy + 6} M{cx + 6},{cy - 6} '
              f'L{cx - 6},{cy + 6}" stroke="{c}" stroke-width="2.4" '
              f'stroke-linecap="round"/>\n')
    elif sym == "plus":
        s += (f'<path d="M{cx - 8},{cy} L{cx + 8},{cy} M{cx},{cy - 8} L{cx},{cy + 8}" '
              f'stroke="{c}" stroke-width="2.4" stroke-linecap="round"/>\n')
    return s


def aw(x1, y1, x2, y2, key, w=2.0, dash=""):
    return arrow(x1, y1, x2, y2, color=ACCENT[key], w=w, dash=dash)


def apoly(pts, key, w=2.0, dash=""):
    return poly(pts, color=ACCENT[key], w=w, dash=dash)


def eq(x, y, s, size=17, weight="normal"):
    return txt(x, y, s, size, anchor="start", fill=INK, weight=weight)


def strip(x, y, items, key, cw=196, h=44):
    """Measured-outcome value strip: label above, number below, one cell per item."""
    c = ACCENT[key]
    s = ""
    for i, (lab, val, hot) in enumerate(items):
        cx = x + i * (cw + 8)
        s += (f'<rect x="{cx}" y="{y}" width="{cw}" height="{h}" rx="5" fill="{WHITE}" '
              f'stroke="{c}" stroke-width="1.4" stroke-opacity="0.7"/>\n')
        s += txt(cx + cw / 2, y + 17, lab, 12, fill="#555")
        s += txt(cx + cw / 2, y + 36, val, 16,
                 fill=ACCENT["crit"] if hot else INK, weight="bold")
    return s


# --- V1 ---------------------------------------------------------------------------
def v1_incr():
    K = "blue"
    b = base("(V1) PiWM-incr -- per-sample increment normalisation   [COLLAPSED]")
    # w scales the MSE. Connector runs up the right margin (x=880) and in along y=120,
    # both empty corridors, then down between the two code stacks.
    # the product sits in the empty gutter BETWEEN the two code stacks (x 723..801),
    # below their bottom edge (y 337), so it clears both the MSE arrow and its label
    b += opnode(766, 330, "times", K, r=15)
    b += apoly([(880, 898), (880, 362), (766, 362), (766, 349)], K, w=2.0, dash="6,4")
    b += apoly([(766, 315), (766, 278)], K, w=2.0, dash="6,4")
    b += txt(836, 350, "w", 18, fill=ACCENT[K], weight="bold")

    s = panel(K, "module:  per-sample weight on the prediction loss")
    yA, yB = 984, 1086
    # row A -- build the weight from the TRUE increment
    s += pill(96, yA - 25, sub("z", "t+1", 12), K)
    s += pill(96, yA + 25, sub("z", "t", 12), K)
    s += opnode(196, yA, "minus", K)
    s += aw(130, yA - 25, 178, yA - 8, K)
    s += aw(130, yA + 25, 178, yA + 8, K)
    s += mbox(232, yA - 24, 104, 48, nrm("."), K)
    s += aw(212, yA, 228, yA, K)
    s += mbox(372, yA - 24, 168, 48, "1 / ( . + eps )", K)
    s += aw(336, yA, 368, yA, K)
    s += pill(600, yA, "w", K, rx=30)
    s += aw(540, yA, 566, yA, K)
    # row B -- the prediction error it multiplies
    s += pill(96, yB - 25, "zhat", K, fill=WHITE)
    s += pill(96, yB + 25, sub("z", "t+1", 12), K, fill=WHITE)
    s += opnode(196, yB, "minus", K)
    s += aw(130, yB - 25, 178, yB - 8, K)
    s += aw(130, yB + 25, 178, yB + 8, K)
    s += mbox(232, yB - 24, 104, 48, nrm("."), K)
    s += aw(212, yB, 228, yB, K)
    s += opnode(600, yB, "times", K)
    s += aw(336, yB, 584, yB, K)
    s += aw(600, yA + 19, 600, yB - 16, K)
    s += mbox(668, yB - 24, 176, 48, sub("L", "z", 12), K, fill=ACCENT_FILL[K], size=18)
    s += aw(616, yB, 664, yB, K)
    # equation + measured
    s += eq(PX + 26, 1168, "L_z  =  w  .  " + nrm("zhat - z_t+1"), 17)
    s += eq(PX + 300, 1168, "w  =  1 / ( " + nrm("z_t+1 - z_t") + " + eps )", 17)
    s += eq(PX + 26, 1196, "w is formed PER SAMPLE. A batch-level w would only rescale "
            "L_z and leave every gradient direction unchanged.", 15)
    s += strip(PX + 26, 1212, [
        ("eps", "1e-4", False),
        ("median ||dz||^2", "4.1e-2", True),
        ("samples floored by eps", "0.1%", True),
        ("w_max / w_min", "3884x", True)], K, cw=198)
    return b + s


# --- V2 ---------------------------------------------------------------------------
def v2_actinfo():
    K = "magenta"
    b = base("(V2) PiWM-actinfo -- InfoNCE over actions   [NEGATIVE]")
    # connector into the action input, along two empty corridors (x=640, y=430)
    b += apoly([(880, 898), (880, 120), (300, 120), (300, 190)], K, w=2.0, dash="6,4")
    b += txt(310, 106, "K permuted copies of a_t", 16, anchor="start",
             fill=ACCENT[K])

    s = panel(K, "module:  the true action must beat K permuted ones at the SAME state")
    rows = [(986, sub("a", "t", 11), "E0", True),
            (1046, "a'", "E1", False), (1106, "a''", "E2", False)]
    SPINE = 112
    s += pill(64, 1046, sub("z", "t", 11), K, rx=28)
    s += (f'<path d="M92,1046 L{SPINE},1046 M{SPINE},1008 L{SPINE},1128" '
          f'stroke="{ACCENT[K]}" stroke-width="2" fill="none"/>\n')
    for y, lab, ek, is_true in rows:
        col = K if is_true else "slate"
        wl = 2.2 if is_true else 1.5
        s += circle(162, y, 15, lab, size=14,
                    fill=ACCENT_FILL[K] if is_true else "#eeeeea")
        s += aw(177, y, 202, y, col, w=wl)                      # a_k -> g
        s += apoly([(SPINE, y + 22), (196, y + 22), (196, y + 14), (202, y + 14)],
                   col, w=wl)                                    # z_t -> g (under a_k)
        s += mbox(206, y - 24, 72, 48, "g", col, size=20)
        s += mbox(312, y - 24, 200, 48, nrm("h(g) - z_t+1"), col, size=15)
        s += aw(278, y, 308, y, col, w=wl)
        s += pill(556, y, ek, col, rx=24, fill=WHITE)
        s += aw(512, y, 528, y, col, w=wl)
    # three private corridors at x=610 into the softmax
    s += mbox(626, 1000, 172, 108, "softmax( -E / tau )", K, size=16,
              sub_="positive = the TRUE action")
    s += apoly([(580, 986), (610, 986), (610, 1024), (622, 1024)], K, w=2.2)
    s += apoly([(580, 1046), (610, 1046), (610, 1054), (622, 1054)], "slate", w=1.5)
    s += apoly([(580, 1106), (610, 1106), (610, 1084), (622, 1084)], "slate", w=1.5)
    s += pill(858, 1054, "L_ctrl", K, rx=40)
    s += aw(798, 1054, 814, 1054, K)
    s += eq(PX + 26, 1168, "E_k  =  " + nrm("h( g(z_t , a_k) ) - z_t+1"), 17)
    s += eq(PX + 420, 1168, "a_0 = a_t ,    a_1..K = perm(a) within the batch", 17)
    s += eq(PX + 26, 1196, "L_ctrl  =  -log [ exp(-E_0/tau) / SUM_k exp(-E_k/tau) ]"
            "        =>   max  I( a_t ; z_t+1 | z_t ),    tau = detached mean E", 16)
    s += strip(PX + 26, 1212, [
        ("d_action", "0.699  (max of any arm)", False),
        ("code density rho", "0.448 -> 0.052", True),
        ("CEM vs control", "-0.280  (p = 0.017)", True)], K, cw=268)
    return b + s


# --- V3 ---------------------------------------------------------------------------
def v3_pathint():
    K = "amber"
    b = base("(V3) PiWM-pathint -- the reference frame is path-integrated by the action"
             "   [NULL]")
    b += apoly([(880, 898), (880, 120), (300, 120), (300, 190)], K, w=2.0, dash="6,4")
    b += txt(310, 106, "pose joins the action embedding", 16, anchor="start",
             fill=ACCENT[K])

    s = panel(K, "module:  a learned pose map, rolled forward by the model itself")
    yA, yB = 982, 1082
    s += circle(76, yA, 23, sub("p", "t", 11), fill=ACCENT_FILL[K], size=15)
    s += circle(76, yB, 23, sub("a", "t", 11), fill=ACCENT_FILL[K], size=15)
    # row A: the embedding the predictor consumes
    s += mbox(170, yA - 24, 168, 48, "W_a a_t  +  W_p p_t", K, size=16)
    s += aw(99, yA, 166, yA, K)
    s += apoly([(99, yB - 14), (136, yB - 14), (136, yA + 16), (166, yA + 16)], K)
    s += pill(410, yA, "to g", K, rx=42, fill=WHITE)
    s += aw(338, yA, 368, yA, K)
    # row B: the path-integration head and its supervision
    s += mbox(170, yB - 24, 168, 48, "f_path", K, size=19, sub_="[ p_t ; a_t ]")
    s += aw(99, yB, 166, yB, K)
    s += apoly([(99, yA + 14), (136, yA + 14), (136, yB - 16), (166, yB - 16)], K)
    s += pill(400, yB, "phat_t+1", K, rx=48, fill=WHITE)
    s += aw(338, yB, 352, yB, K)
    s += pill(516, yA + 30, sub("p", "t+1", 11), K, rx=36)
    s += opnode(516, yB, "minus", K)
    s += aw(448, yB, 500, yB, K)
    s += aw(516, yA + 49, 516, yB - 16, K)
    s += mbox(568, yB - 24, 104, 48, nrm("."), K)
    s += aw(532, yB, 564, yB, K)
    s += mbox(706, yB - 24, 150, 48, "L_path", K, fill=ACCENT_FILL[K], size=18)
    s += aw(672, yB, 702, yB, K)
    s += eq(PX + 26, 1168, "phat_t+1  =  f_path( [ p_t ; a_t ] )", 17)
    s += eq(PX + 330, 1168, "L  =  L_z  +  lambda . " + nrm("phat_t+1 - p_t+1"), 17)
    s += eq(PX + 26, 1196, "rollout uses the SAME head:   p_k+1 = f_path( [ p_k ; a_k ] )"
            "        warm start: the dataset fit (95.2% of the 1-step pose change)", 15)
    s += strip(PX + 26, 1212, [
        ("code density rho", "0.573", False), ("rel_mse", "0.0094", False),
        ("d_action", "0.460", False), ("CEM vs control", "-0.025  (null)", True)],
        K, cw=198)
    return b + s


def link_target_2x2():
    """The 2x2 the archive never ran: link x target_p. A data table, not a method."""
    s = txt(470, 58, "The 2x2 that had never been run: link x target_p", 23)
    s += txt(470, 88, "173 runs at (reprelu, p=1), 35 at (identity, p=2), ZERO off-diagonal",
             15, fill="#555")
    CW, CH, X0, Y0 = 320, 190, 130, 150
    cells = [
        (0, 0, "reprelu, p=1", "LpWM-ltv  (n=16)", "rho 0.448   eff_dim 24.1\nrel_mse 0.0092", "green"),
        (1, 0, "reprelu, p=2", "NEW this round (n=8)", "rho 0.509   eff_dim 26.1\nrel_mse 0.0082", "green"),
        (0, 1, "identity, p=1", "NEW this round (n=8)", "rho 1.000   eff_dim 2.9\nrel_mse 0.052", "crit"),
        (1, 1, "identity, p=2", "LeWM-ltv-p2  (n=8)", "rho 1.000   eff_dim 4.1\nrel_mse 0.83", "crit"),
    ]
    for cx, cy, title, who, body, key in cells:
        x, y = X0 + cx * (CW + 30), Y0 + cy * (CH + 34)
        s += (f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="10" '
              f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="2.6"/>\n')
        s += txt(x + CW / 2, y + 36, title, 20, fill=ACCENT[key], weight="bold")
        s += txt(x + CW / 2, y + 62, who, 13, fill="#555")
        for i, ln in enumerate(body.split("\n")):
            s += txt(x + CW / 2, y + 100 + i * 24, ln, 15)
    s += txt(X0 - 46, Y0 + CH / 2, "reprelu", 17)
    s += txt(X0 - 46, Y0 + CH + 34 + CH / 2, "identity", 17)
    s += txt(X0 + CW / 2, Y0 - 16, "target_p = 1  (Laplace)", 15, fill="#555")
    s += txt(X0 + CW + 30 + CW / 2, Y0 - 16, "target_p = 2  (Gaussian)", 15, fill="#555")
    s += txt(470, 646, "Rows differ (link: p = 2.7e-06).  Columns do not "
             "(target_p: p = 0.21 / 0.80).", 17, fill=ACCENT["slate"])
    s += txt(470, 676, "A 1-D projection of a D=384 Laplace sample IS Gaussian "
             "(Diaconis-Freedman), and RDMReg only sees 1-D projections.", 14, fill="#555")
    return s


def emit(name, body, out, w=W, h=H):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, body, wh in (
        ("arch-v1-incr.svg", v1_incr(), (W, H)),
        ("arch-v2-actinfo.svg", v2_actinfo(), (W, H)),
        ("arch-v3-pathint.svg", v3_pathint(), (W, H)),
        ("arch-link-target-2x2.svg", link_target_2x2(), (940, 720)),
    ):
        print("  wrote", a.out + "/" + emit(name, body, a.out, *wh))


if __name__ == "__main__":
    main()
