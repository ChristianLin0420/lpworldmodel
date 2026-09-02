"""SVG architecture diagrams for every PiWM variant, in the LpWM paper's own style.

The reference is assets/fig1_lpworldmodel.png (a): observation circles in peach, a
teal rounded-top encoder, pale-yellow (Rep)ReLU links, a lavender D-shaped predictor,
navy/white square stacks for the sparse code, a white regulariser box, and a dark-red
MSE link, all in serif type. Every diagram below reuses that base unchanged and adds
ONLY the proposed part, drawn in its family accent colour with a dashed callout, so a
reader can see at a glance what each variant changes relative to LpWM.

Family accents follow analysis/panels.py so the diagrams and the result plots agree:
blue = sparse codes, magenta = gating, green = union/consensus, and two further
accents for the LeVJEPA combinations.

Usage:  python analysis/arch_figs.py --out diary/assets/2026-09-02
"""
import argparse
import os

# --- palette lifted from the reference figure -------------------------------------
PEACH = "#FAD9BE"
TEAL = "#A9D5D1"
YELLOW = "#F9F2C4"
LAVENDER = "#E2CDE9"
NAVY = "#40566E"
WHITE = "#FFFFFF"
INK = "#1A1A1A"
ARROW = "#4A6274"
BOXLINE = "#55707F"
MSERED = "#9E3B3B"

ACCENT = {
    "blue": "#0055af",
    "magenta": "#912d59",
    "green": "#006e00",
    "amber": "#9a6700",
    "crit": "#b3261e",
    "slate": "#3d3d3a",
}
ACCENT_FILL = {
    "blue": "#d8e8fb",
    "magenta": "#f5dde7",
    "green": "#d9efd7",
    "amber": "#faeecb",
    "crit": "#f7dcd9",
    "slate": "#e6e6e3",
}

FONT = "Georgia, 'Times New Roman', serif"


def _hdr(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
        '<defs>\n'
        f'  <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker>\n'
        f'  <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{MSERED}"/></marker>\n'
        f'  <marker id="am" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto">'
        f'<path d="M10,0 L0,5 L10,10 z" fill="{MSERED}"/></marker>\n'
        '</defs>\n'
        f'<rect width="{w}" height="{h}" fill="white"/>\n'
    )


def txt(x, y, s, size=19, anchor="middle", fill=INK, style="", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}" font-style="{style}" font-weight="{weight}">{s}</text>\n')


def arrow(x1, y1, x2, y2, color=ARROW, marker="a", w=2.0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{w}" '
            f'fill="none" marker-end="url(#{marker})"{d}/>\n')


def poly(pts, color=ARROW, marker="a", w=2.0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " L".join(f"{x},{y}" for x, y in pts)
    return (f'<path d="M{p}" stroke="{color}" stroke-width="{w}" fill="none" '
            f'marker-end="url(#{marker})"{d}/>\n')


def circle(cx, cy, r, label, fill=PEACH, size=20):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{INK}" '
            f'stroke-width="1.6"/>\n') + txt(cx, cy + 7, label, size)


def domeup(x, y, w, h, label, fill=TEAL, sub=""):
    """Encoder shape: flat bottom, rounded top (as in the reference)."""
    r = w / 2
    s = (f'<path d="M{x},{y+h} L{x},{y+r} A{r},{r} 0 0 1 {x+w},{y+r} L{x+w},{y+h} Z" '
         f'fill="{fill}" stroke="{INK}" stroke-width="2.4"/>\n')
    s += txt(x + w / 2, y + h - 26, label, 21)
    if sub:
        s += txt(x + w / 2, y + h - 6, sub, 13, fill="#444")
    return s


def domeright(x, y, w, h, label, fill=LAVENDER, sub=""):
    """Predictor shape: flat left, rounded right (as in the reference)."""
    r = h / 2
    s = (f'<path d="M{x},{y} L{x+w-r},{y} A{r},{r} 0 0 1 {x+w-r},{y+h} L{x},{y+h} Z" '
         f'fill="{fill}" stroke="{INK}" stroke-width="2.4"/>\n')
    s += txt(x + w / 2 - 6, y + h / 2 + 1, label, 21)
    if sub:
        s += txt(x + w / 2 - 6, y + h / 2 + 22, sub, 13, fill="#444")
    return s


def box(x, y, w, h, label, fill=YELLOW, size=17, rx=4, stroke=INK, sw=1.4, sub=""):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="{sw}"/>\n')
    s += txt(x + w / 2, y + h / 2 + (0 if not sub else -5) + 6, label, size)
    if sub:
        s += txt(x + w / 2, y + h / 2 + 18, sub, 12, fill="#444")
    return s


def stack(x, y, pattern, cell=22, gap=3, label=None, lab_dx=0, accent=None, hat=False):
    """Vertical code stack. pattern: iterable of 0/1 (1 = active, navy)."""
    s = ""
    for i, v in enumerate(pattern):
        f = NAVY if v else WHITE
        if accent and v == 2:
            f = accent
        s += (f'<rect x="{x}" y="{y + i*(cell+gap)}" width="{cell}" height="{cell}" '
              f'fill="{f if v != 2 else accent}" stroke="{INK}" stroke-width="1.3"/>\n')
    if label:
        s += txt(x + cell / 2 + lab_dx, y - 12, label, 19)
        if hat:
            s += caret(x + cell / 2 + lab_dx - 7, y - 30)
    return s


def callout(x, y, w, h, text_lines, key="blue", size=13):
    """Dashed accent box naming the proposed change."""
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{f}" '
         f'fill-opacity="0.75" stroke="{c}" stroke-width="2" stroke-dasharray="7,4"/>\n')
    for i, ln in enumerate(text_lines):
        weight = "bold" if i == 0 else "normal"
        s += txt(x + w / 2, y + 21 + i * (size + 5), ln, size, fill=c, weight=weight)
    return s


def sub(main, s, size=13):
    """Subscript via dy, not baseline-shift: the latter is ignored by cairosvg/rsvg."""
    return f"{main}<tspan font-size=\"{size}\" dy=\"6\">{s}</tspan>"


def caret(cx, cy, w=9, h=6):
    """A drawn circumflex. The serif face used for these figures has neither the
    precomposed z-hat (U+1E91) nor the combining accent (U+0302), and no arrow
    glyphs either -- verified by rendering. So the hat is geometry, not text."""
    return (f'<path d="M{cx-w/2},{cy} L{cx},{cy-h} L{cx+w/2},{cy}" stroke="{INK}" '
            f'stroke-width="1.7" fill="none" stroke-linecap="round"/>\n')


# --- the shared LpWM base ----------------------------------------------------------
SP_L = [0, 1, 1, 0, 1, 0]      # z_t          (sparse: white = 0, navy = active)
SP_P = [0, 1, 0, 1, 1, 0]      # z_hat_{t+1}
SP_R = [0, 1, 1, 1, 0, 0]      # z_{t+1}


def base(title, reg_label="RDMReg", reg_sub="", link_label="ReLU",
         zl=SP_L, zp=SP_P, zr=SP_R, pred_label="Pred", pred_sub="g",
         enc_sub="", skip=()):
    """The LpWM schematic every variant starts from. `skip` omits pieces a variant redraws."""
    s = ""
    # observations
    if "obs" not in skip:
        s += circle(120, 742, 34, sub("o", "t", 13))
        s += circle(812, 742, 34, sub("o", "t+1", 13))
        s += arrow(120, 706, 120, 646)
        s += arrow(812, 706, 812, 646)
    # encoders
    if "enc" not in skip:
        s += domeup(64, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>", sub=enc_sub)
        s += domeup(756, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>", sub=enc_sub)
        s += arrow(120, 520, 120, 470)
        s += arrow(812, 520, 812, 470)
    # links
    if "link" not in skip:
        s += box(78, 420, 84, 44, link_label)
        s += box(770, 420, 84, 44, link_label)
        s += arrow(120, 420, 120, 352)
        s += arrow(812, 420, 812, 352)
    # code stacks
    if "codes" not in skip:
        s += stack(109, 190, zl, label=sub("z", "t", 13))
        s += stack(801, 190, zr, label=sub("z", "t+1", 13))
        s += stack(701, 190, zp, label=sub("z", "t+1", 13), hat=True)
        # MSE double arrow
        s += (f'<path d="M737,262 L795,262" stroke="{MSERED}" stroke-width="2" '
              f'marker-end="url(#ar)" marker-start="url(#am)"/>\n')
        s += txt(766, 240, "MSE (min)", 16, fill=MSERED)
    # predictor + action
    if "pred" not in skip:
        s += domeright(232, 196, 190, 132, pred_label, sub=pred_sub)
        s += circle(318, 430, 38, sub("a", "t", 13))
        s += arrow(318, 392, 318, 332)
        s += arrow(146, 262, 226, 262)
        s += txt(186, 248, sub("z", "t", 12), 16)
        s += arrow(424, 262, 697, 262)
    # regulariser
    if "reg" not in skip:
        s += box(430, 556, 160, 52, reg_label, fill=WHITE, stroke=BOXLINE, sw=1.6,
                 size=17, sub=reg_sub)
        s += poly([(162, 442), (206, 442), (206, 582), (426, 582)])
        s += poly([(770, 442), (726, 442), (726, 582), (594, 582)])
    s += txt(466, 872, title, 21)
    return s


def emit(name, body, out, w=940, h=900):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


# --- the variants -----------------------------------------------------------------
def build(out):
    made = []

    # 0. LpWM baseline (reference reproduction)
    made.append(emit("arch-00-lpwm.svg",
        base("(0) LpWM baseline -- sparse JEPA world model"), out))

    # 1. Step 2: k-WTA sparse codes
    b = base("(1) PiWM-sparse -- k-WTA imposed on the code  [REFUTED]",
             link_label="ReLU", skip=("link",))
    b += box(78, 420, 84, 44, "ReLU") + box(770, 420, 84, 44, "ReLU")
    b += arrow(120, 420, 120, 352) + arrow(812, 420, 812, 352)
    b += box(60, 356, 120, 40, "k-WTA", fill=ACCENT_FILL["blue"],
             stroke=ACCENT["blue"], sw=2.2, size=16)
    b += box(752, 356, 120, 40, "k-WTA", fill=ACCENT_FILL["blue"],
             stroke=ACCENT["blue"], sw=2.2, size=16)
    b += callout(286, 628, 330, 96, [
        "Proposed: hard top-k after the link",
        "Realised density = min(k, #positive)/D,",
        "so the arm never ran at its own k:",
        "0.19 of 8 units (D=384), 29 of 41 (D=2048)."], "blue")
    made.append(emit("arch-01-sparse-kwta.svg", b, out))

    # 2. Step 3: support gating
    b = base("(2) PiWM-gate -- gate driven by the code's SUPPORT  [NULL]",
             pred_label="Pred", pred_sub="g (LTV)")
    b += box(150, 352, 150, 40, "1[z &gt; 0]", fill=ACCENT_FILL["magenta"],
             stroke=ACCENT["magenta"], sw=2.2, size=16)
    b += poly([(146, 300), (140, 300), (140, 372), (146, 372)],
              color=ACCENT["magenta"], w=2.0, dash="6,4")
    b += poly([(300, 372), (392, 372), (392, 332)], color=ACCENT["magenta"],
              w=2.0, dash="6,4")
    b += callout(276, 628, 356, 96, [
        "Proposed: gate on the binary support only",
        "s = 1[z&gt;0] is a deterministic function of z,",
        "so I(s;Y) &lt;= I(z;Y): bounded above by magnitude.",
        "Measured: 76% of bits discarded, 63% of info kept."], "magenta")
    made.append(emit("arch-02-gate-support.svg", b, out))

    # 3. Step 4: union head
    b = base("(3) PiWM-union -- J parallel readouts, loss = min_j  [DEAD]",
             skip=("pred",))
    b += domeright(232, 150, 190, 108, "Pred", sub="g, head 1")
    b += (f'<path d="M244,272 L406,272 A54,54 0 0 1 406,380 L244,380 Z" '
          f'fill="{ACCENT_FILL["green"]}" stroke="{ACCENT["green"]}" '
          f'stroke-width="2.4" stroke-dasharray="7,4"/>\n')
    b += txt(318, 322, "heads 2 ... J", 19, fill=ACCENT["green"])
    b += circle(318, 466, 36, sub("a", "t", 13))
    b += arrow(318, 430, 318, 386)
    b += arrow(146, 204, 226, 204)
    b += arrow(424, 204, 697, 220)
    b += box(452, 300, 190, 46, "loss = min_j  L_j",
             fill=ACCENT_FILL["green"], stroke=ACCENT["green"], sw=2.2, size=17)
    b += callout(286, 700, 340, 82, [
        "Proposed: any head may be right (SDR union)",
        "min_j L_j admits z = 0 as a GLOBAL optimum;",
        "a variance floor prevents collapse and the arm",
        "still scores 0.000 on 6/6 seeds (p = 0.0115)."], "green")
    made.append(emit("arch-03-union-head.svg", b, out))

    # 4. consensus voting -- the positive result
    b = _consensus()
    made.append(emit("arch-04-consensus-vote.svg", b, out, w=980, h=880))

    # 5. reference frame (pose)
    b = base("(4) PiWM-refframe -- allocentric pose bound to the action")
    b += circle(318, 566, 34, sub("p", "t", 13),
                fill=ACCENT_FILL["amber"])
    b += box(258, 486, 120, 40, "Enc pose", fill=ACCENT_FILL["amber"],
             stroke=ACCENT["amber"], sw=2.2, size=15)
    b += arrow(318, 532, 318, 530, color=ACCENT["amber"])
    b += poly([(318, 532), (318, 528)], color=ACCENT["amber"])
    b += arrow(318, 486, 318, 470, color=ACCENT["amber"], w=2.2)
    b += txt(410, 468, "+", 24, fill=ACCENT["amber"], weight="bold")
    b += callout(556, 640, 350, 96, [
        "Proposed: bind location to the action embedding",
        "obs['proprio'] was loaded every batch and DROPPED",
        "(adaln path), so the pose encoder got zero gradient.",
        "Rollout must roll pose forward, not freeze it."], "amber")
    made.append(emit("arch-05-refframe-pose.svg", b, out))

    # 6. columns (patch tokens)
    b = base("(5) PiWM-columns -- P=256 patch tokens instead of one CLS",
             skip=("codes",), enc_sub="feature = patch")
    for k, xoff in enumerate((0, 26, 52)):
        b += stack(95 + xoff, 190 + k * 6, SP_L, cell=18, gap=3)
    b += txt(148, 170, sub("z", "t", 13) + " (P tokens)", 18)
    for k, xoff in enumerate((0, 26, 52)):
        b += stack(787 + xoff, 190 + k * 6, SP_R, cell=18, gap=3)
    b += txt(840, 170, sub("z", "t+1", 13), 18)
    for k, xoff in enumerate((0, 26, 52)):
        b += stack(660 + xoff, 190 + k * 6, SP_P, cell=18, gap=3)
    b += txt(700, 170, sub("z", "t+1", 13), 18) + caret(693, 152)
    b += (f'<path d="M745,258 L780,258" stroke="{MSERED}" stroke-width="2" '
          f'marker-end="url(#ar)" marker-start="url(#am)"/>\n')
    b += txt(762, 238, "MSE (min)", 15, fill=MSERED)
    b += arrow(160, 262, 226, 262)
    b += arrow(424, 262, 655, 262)
    b += callout(286, 700, 340, 82, [
        "Proposed: give the model columns at all",
        "feature=cls gives num_patches = 1, so the predictor",
        "saw ONE token and shared one operator across it.",
        "TBT needs a population of located columns."], "slate")
    made.append(emit("arch-06-columns-patch.svg", b, out))

    # 7. SIGReg / isotropic Gaussian target
    b = base("(6) PiWM-sigreg -- LeJEPA objective: isotropic-Gaussian target",
             reg_label="SIGReg", reg_sub="Epps-Pulley, 1024 proj.",
             link_label="Identity",
             zl=[1, 1, 1, 1, 1, 1], zp=[1, 1, 1, 1, 1, 1], zr=[1, 1, 1, 1, 1, 1],
             skip=("reg",))
    b += box(418, 552, 184, 60, "SIGReg", fill=ACCENT_FILL["crit"],
             stroke=ACCENT["crit"], sw=2.4, size=18,
             sub="Epps-Pulley -> N(0, I)")
    b += poly([(162, 442), (206, 442), (206, 582), (414, 582)])
    b += poly([(770, 442), (726, 442), (726, 582), (606, 582)])
    b += callout(276, 646, 356, 96, [
        "Proposed: replace the sparse target with N(0, I)",
        "LeJEPA claims the isotropic Gaussian minimises",
        "worst-case probing risk. Here the SAME target under",
        "RDMReg also dies -- the target, not the regulariser."], "crit")
    made.append(emit("arch-07-sigreg-gaussian.svg", b, out))

    # 8. token dropping
    b = base("(7) PiWM-drop95 -- 95% of patch tokens dropped  [COLLAPSED]",
             enc_sub="feature = patch")
    b += box(52, 470, 136, 40, "drop 95%", fill=ACCENT_FILL["amber"],
             stroke=ACCENT["amber"], sw=2.2, size=16)
    b += box(744, 470, 136, 40, "drop 95%", fill=ACCENT_FILL["amber"],
             stroke=ACCENT["amber"], sw=2.2, size=16)
    b += callout(276, 646, 356, 96, [
        "Proposed: LeVJEPA's strongest single gain",
        "In a JEPA the encoder output is ALSO the target, so",
        "dropping removes signal from both sides. Paired with",
        "RDMReg (dead code costs only 0.51) it collapsed: rho=0."], "amber")
    made.append(emit("arch-08-token-drop.svg", b, out))

    # 9. block-causal attention
    b = base("(8) PiWM-blockcausal -- temporal structure inside the encoder",
             skip=("enc",))
    b += domeup(64, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                fill=ACCENT_FILL["green"])
    b += domeup(756, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                fill=ACCENT_FILL["green"])
    b += arrow(120, 520, 120, 470)
    b += arrow(812, 520, 812, 470)
    b += (f'<path d="M176,582 L756,582" stroke="{ACCENT["green"]}" stroke-width="2.4" '
          f'stroke-dasharray="8,5" fill="none" marker-end="url(#a)"/>\n')
    b += txt(466, 572, "block-causal attention across frames", 15, fill=ACCENT["green"])
    b += callout(276, 646, 356, 96, [
        "Proposed: the encoder sees the clip, causally",
        "encode_obs folded t into the batch, so frames were",
        "encoded INDEPENDENTLY -- no temporal structure at all.",
        "Verified causal: frame 0 is unchanged by frame T."], "green")
    made.append(emit("arch-09-block-causal.svg", b, out))

    return made


def _consensus():
    """M independently trained columns voting at PLAN time (the positive result)."""
    s = ""
    ys = [96, 300, 504]
    BX, BY, BW, BH = 566, 246, 196, 120        # consensus box
    cy = BY + BH / 2
    for i, y in enumerate(ys):
        fill = TEAL if i < 2 else "#cfe3e1"
        s += domeup(70, y, 96, 104, sub("Enc <tspan font-style='italic'>f</tspan>", f"{i+1}", 12), fill=fill)
        s += domeright(232, y + 12, 150, 80, "Pred", sub=sub("g", f"{i+1}", 11))
        s += arrow(166, y + 52, 226, y + 52)
        s += stack(404, y + 6, [SP_P, SP_L, SP_R][i], cell=16, gap=3)
        s += arrow(384, y + 52, 398, y + 52)
        s += txt(118, y - 10, f"column {i+1}", 15, fill="#444")
        # converge into the vote box rather than stopping in mid-air
        if y + 52 == cy:
            s += arrow(452, cy, BX - 4, cy)
        else:
            s += poly([(452, y + 52), (508, y + 52), (508, cy), (BX - 4, cy)])
    # drawn ellipsis (the font has no U+22EE)
    for k in range(3):
        s += f'<circle cx="118" cy="{648 + k*11}" r="2.4" fill="#666"/>\n'
    s += txt(318, 672, "M independently trained models", 15, fill="#444")
    s += (f'<rect x="{BX}" y="{BY}" width="{BW}" height="{BH}" rx="10" '
          f'fill="{ACCENT_FILL["green"]}" stroke="{ACCENT["green"]}" stroke-width="2.6"/>\n')
    s += txt(BX + BW / 2, BY + 34, "CONSENSUS", 19, fill=ACCENT["green"], weight="bold")
    s += txt(BX + BW / 2, BY + 60, "median over", 15, fill=ACCENT["green"])
    s += txt(BX + BW / 2, BY + 82, "per-member ranks", 15, fill=ACCENT["green"])
    s += txt(BX + BW / 2, BY + 104, "(at PLAN time)", 13, fill="#555")
    s += arrow(BX + BW, cy, BX + BW + 52, cy, color=ACCENT["green"], w=2.4)
    s += circle(BX + BW + 96, cy, 42, "CEM", fill=PEACH, size=17)
    s += callout(250, 704, 470, 82, [
        "Proposed: vote among columns -- and it WORKS",
        "M=5: delta = +0.228 vs a single model (p = 0.0005), beats its",
        "members' mean 10/10, matches best-of-M, 0/10 catastrophes.",
        "On the OBJECTIVE, not in the loss: min_j L_j is degenerate."], "green")
    s += txt(490, 818, "(9) PiWM-vote -- plan-time consensus over independently trained columns", 20)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-02")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in build(a.out):
        print(f"  wrote {a.out}/{n}")


if __name__ == "__main__":
    main()
