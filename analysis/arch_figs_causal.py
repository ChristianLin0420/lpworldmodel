"""Architecture diagrams for the causal-objective round (V1-V3), in the LpWM paper's style.

Reuses every primitive from analysis/arch_figs.py so these sit beside the 2026-09-02 set
unchanged: peach observation/action circles, teal rounded-top encoder, pale-yellow link,
lavender D-shaped predictor, navy/white code stacks, dark-red MSE. Each diagram adds ONLY
the proposed part, in its accent colour with a dashed callout carrying the measured outcome.

Usage:  python analysis/arch_figs_causal.py --out diary/assets/2026-09-03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, INK, LAVENDER, MSERED, NAVY, PEACH, WHITE,
    _hdr, arrow, base, box, caret, circle, domeright, domeup, poly, stack, sub, txt,
    SP_L, SP_P, SP_R,
)


def emit(name, body, out, w=940, h=900):
    with open(os.path.join(out, name), "w") as f:
        f.write(_hdr(w, h) + body + "</svg>\n")
    return name


def v1_incr():
    """V1: per-sample increment normalisation of the prediction loss."""
    b = base("(V1) PiWM-incr -- per-sample increment normalisation  [COLLAPSED]")
    b += box(560, 300, 250, 66, "w = 1 / (||dz||^2 + eps)",
             fill=ACCENT_FILL["blue"], stroke=ACCENT["blue"], sw=2.4, size=15,
             sub="applied per sample, not per batch")
    b += poly([(737, 292), (737, 276)], color=ACCENT["blue"], w=2.2, dash="6,4")
    b += callout(186, 636, 570, 134, [
        "Proposed: stop large autonomous motions dominating the loss",
        "'Loss on the increment' is a NO-OP: (zhat - z_t) - (z_t+1 - z_t) = zhat - z_t+1 exactly.",
        "A BATCH-level normaliser only rescales. Only PER-SAMPLE weighting moves the gradient.",
        "MEASURED FAILURE (LpWM-ltv s3, 1536 real val samples): eps=1e-4 is 400x below the",
        "median ||dz||^2 = 4.1e-2, so it floors 0.1% of samples; weights span 3884x and the",
        "top 1% of samples take 29% of the batch loss.",
        "Result (n=8): rho 0.448 -> 0.004, eff_dim 24.1 -> 7.5, d_action = 0 on 7 of 8 seeds.",
        "An implementation flaw, not a verdict on the idea."], "blue")
    return b


def v2_actinfo():
    """V2: InfoNCE over actions -- max I(a_t ; z_{t+1} | z_t)."""
    s = ""
    # three predictor copies: one true action, two shuffled
    ys = [140, 300, 460]
    labs = [("a", "t", ACCENT["magenta"], "TRUE action"),
            ("a'", "", "#8a8a85", "shuffled"), ("a''", "", "#8a8a85", "shuffled")]
    for i, y in enumerate(ys):
        lab, sb, col, note = labs[i]
        fill = LAVENDER if i == 0 else "#efe6f3"
        s += domeright(214, y, 168, 92, "Pred", sub="g")
        s += circle(150, y + 46, 30, sub(lab, sb, 12) if sb else lab,
                    fill=PEACH if i == 0 else "#f0e6dd")
        s += arrow(180, y + 46, 208, y + 46)
        s += txt(150, y - 6, note, 12, fill=col)
        s += stack(404, y + 4, [SP_P, SP_R, SP_L][i], cell=15, gap=3)
        s += arrow(384, y + 46, 398, y + 46)
        s += arrow(452, y + 46, 556, y + 46, color=col, w=2.2 if i == 0 else 1.6)
        s += txt(504, y + 34, ["E(a_t)", "E(a')", "E(a'')"][i], 14, fill=col)
    # z_t feeds all three
    s += stack(64, 250, SP_L, label=sub("z", "t", 13))
    s += poly([(101, 300), (130, 300), (130, 186), (146, 186)])
    s += poly([(101, 320), (130, 320), (130, 506), (146, 506)])
    s += txt(83, 420, "same", 13, fill="#666")
    s += txt(83, 436, "state", 13, fill="#666")
    # softmax / InfoNCE box
    s += box(556, 262, 220, 108, "InfoNCE over actions",
             fill=ACCENT_FILL["magenta"], stroke=ACCENT["magenta"], sw=2.6, size=16,
             sub="-log softmax(-E / tau), true action = positive")
    s += arrow(776, 316, 828, 316, color=ACCENT["magenta"], w=2.4)
    s += txt(866, 306, "max", 17, fill=ACCENT["magenta"])
    s += txt(866, 328, "I(a;z'|z)", 15, fill=ACCENT["magenta"])
    s += callout(150, 622, 620, 116, [
        "Proposed: make the objective CAUSAL, not merely predictive",
        "The true z_t+1 must be better explained by the action TAKEN than by actions that were not.",
        "Negatives are free (permute a within the batch); the predictor is 0.81M params vs the",
        "encoder's 12.56M, so K=4 negatives cost almost nothing. tau = detached mean energy.",
        "MEASURED (n=8): d_action 0.699 -- the highest of any arm -- but rho 0.448 -> 0.052.",
        "It bought action-sensitivity by COLLAPSING the code, which is not the trade we wanted."], "magenta")
    s += txt(470, 872, "(V2) PiWM-actinfo -- InfoNCE over actions", 21)
    return s


def v3_pathint():
    """V3: learned path integration of the location signal."""
    b = base("(V3) PiWM-pathint -- the reference frame is UPDATED BY THE MOVEMENT  [HEALTHY]",
             enc_sub="")
    # pose -> path-integration head -> next pose, and into the action embedding
    b += circle(196, 566, 32, sub("p", "t", 12), fill=ACCENT_FILL["amber"])
    b += box(268, 542, 168, 52, "PathInt",
             fill=ACCENT_FILL["amber"], stroke=ACCENT["amber"], sw=2.6, size=17,
             sub="p_t+1 = f(p_t, a_t)")
    b += arrow(228, 566, 264, 566, color=ACCENT["amber"], w=2.2)
    b += circle(486, 566, 32, sub("p", "t+1", 12), fill=ACCENT_FILL["amber"])
    b += arrow(436, 566, 454, 566, color=ACCENT["amber"], w=2.2)
    b += txt(352, 616, "supervised by the true next pose", 12, fill=ACCENT["amber"])
    # pose joins the action embedding
    b += poly([(352, 542), (352, 470), (318, 470)], color=ACCENT["amber"], w=2.2, dash="6,4")
    b += txt(392, 462, "+ into the action embedding", 12, fill=ACCENT["amber"])
    b += callout(196, 646, 560, 116, [
        "Proposed: TBT's reference frame, done causally",
        "PiWM-refframe RECEIVED pose as a static side-channel and was a NULL (-0.063, p=0.169).",
        "TBT requires the frame to be PATH-INTEGRATED by the movement -- that is the causal half",
        "we omitted. Warm-started from the map fit on the dataset (95.2% of 1-step pose change).",
        "Rollout uses the model's OWN head, so training and planning cannot drift apart.",
        "MEASURED (n=8): rho 0.573, eff_dim 22.8, rel_mse 0.0094 -- indistinguishable from",
        "the baseline -- with d_action 0.460, the largest of any arm that stayed healthy.",
        "8/8 seeds pass the planning gate; CEM 0.260 (n=5) against the control's 0.357."], "amber")
    return b


def link_target_2x2():
    """The 2x2 the archive never ran: link x target_p."""
    s = ""
    s += txt(470, 58, "The 2x2 that had never been run: link x target_p", 23)
    s += txt(470, 88, "173 runs at (reprelu, p=1), 35 at (identity, p=2), ZERO off-diagonal",
             15, fill="#555")
    CW, CH, X0, Y0 = 320, 190, 130, 150
    cells = [
        (0, 0, "reprelu, p=1", "LpWM-ltv  (173 runs)", "rho 0.433   eff_dim 24.1\nrel_mse 0.0092", "green"),
        (1, 0, "reprelu, p=2", "NEW this round", "rho 0.501   eff_dim 26.1\nrel_mse 0.0083", "green"),
        (0, 1, "identity, p=1", "NEW this round", "rho 1.000   eff_dim 2.9\nrel_mse 0.051", "crit"),
        (1, 1, "identity, p=2", "LeWM-ltv-p2  (35 runs)", "rho 1.000   eff_dim 3.1\nrel_mse 0.53", "crit"),
    ]
    for cx, cy, title, who, body, key in cells:
        x, y = X0 + cx * (CW + 30), Y0 + cy * (CH + 34)
        s += (f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="10" '
              f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="2.6"/>\n')
        s += txt(x + CW / 2, y + 36, title, 20, fill=ACCENT[key], weight="bold")
        s += txt(x + CW / 2, y + 62, who, 13, fill="#555")
        for i, ln in enumerate(body.split("\n")):
            s += txt(x + CW / 2, y + 100 + i * 24, ln, 15)
    s += txt(X0 - 46, Y0 + CH / 2, "reprelu", 17, anchor="middle")
    s += txt(X0 - 46, Y0 + CH + 34 + CH / 2, "identity", 17, anchor="middle")
    s += txt(X0 + CW / 2, Y0 - 16, "target_p = 1  (Laplace)", 15, fill="#555")
    s += txt(X0 + CW + 30 + CW / 2, Y0 - 16, "target_p = 2  (Gaussian)", 15, fill="#555")
    s += callout(120, 590, 700, 118, [
        "The rows differ. The columns do not.",
        "Flipping target_p at a fixed link changes nothing measurable: (reprelu,p=1) and",
        "(reprelu,p=2) are 0.0092 vs 0.0083 rel_mse, rho 0.433 vs 0.501, eff_dim 24.1 vs 26.1.",
        "Flipping the LINK changes everything: eff_dim collapses 24 -> 3.",
        "Predicted by measurement: swd(Gaussian, Laplace) = 1.9947 +- 0.0170 against a null of",
        "1.9987 +- 0.0112 -- BELOW it. A 1-D projection of a D=384 Laplace sample IS Gaussian",
        "(Diaconis-Freedman), and RDMReg only ever looks at 1-D projections.",
        "So rho ~ 0.5 comes from the ReLU LINK, not the sparse prior."], "slate")
    return s


def callout(x, y, w, h, lines, key="blue", size=13):
    c, f = ACCENT[key], ACCENT_FILL[key]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{f}" '
         f'fill-opacity="0.75" stroke="{c}" stroke-width="2" stroke-dasharray="7,4"/>\n')
    for i, ln in enumerate(lines):
        s += txt(x + w / 2, y + 21 + i * (size + 5), ln, size, fill=c,
                 weight="bold" if i == 0 else "normal")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, body, wh in (
        ("arch-v1-incr.svg", v1_incr(), (940, 900)),
        ("arch-v2-actinfo.svg", v2_actinfo(), (940, 900)),
        ("arch-v3-pathint.svg", v3_pathint(), (940, 900)),
        ("arch-link-target-2x2.svg", link_target_2x2(), (940, 740)),
    ):
        print("  wrote", a.out + "/" + emit(name, body, a.out, *wh))


if __name__ == "__main__":
    main()
