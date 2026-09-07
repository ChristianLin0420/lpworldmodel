"""Round 9 architecture figures, paper-style: explicit dataflow, the innovation visible.

    python analysis/round9_arch_figs.py --out diary/assets/2026-09-07 [--png]

    arch-premise.svg      why a stateful encoder needs an exact single-frame limit
    arch-p1-enc-ssm.svg   P1  temporal state after the per-frame encoder
    arch-p2-deep.svg      P2  the same state carried between ViT blocks
    arch-p3-scan.svg      P3  a bidirectional scan replacing attention over tokens
    arch-p4-st-scan.svg   P4  the two composed
    arch-ladder.svg       the frame-dropout rungs, and the two claims they separate

DELIBERATELY NOT THE ROUND-8 STYLE.  Those figures spend their top 45% on one shared
encoder/predictor schematic and put the method in a panel underneath.  These are architecture
diagrams: the clip is unrolled left to right so the temporal state is a visible arrow between
frames, tensor shapes are annotated on the wires, and every equation sits in its own reserved
band beneath the block it defines rather than floating over a connector.

NO OVERLAP IS ENFORCED, NOT CLAIMED.  analysis/fig_audit.audit_all checks text x text,
shape x shape and text x shape, plus off-canvas and frame crossings.  The old text-only audit
could not see 1,632 rects, 489 paths and 75 circles; every "nothing overlaps" statement in
rounds 6-8 rested on it.

COLOUR IS ROLE, NEVER RANK
    amber    the new component under test
    green    the unchanged baseline path
    purple   the control arm -- same parameters, state disabled
    crimson  what breaks without the design property
    slate    neutral scaffolding: axes, shapes, captions
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE, caret, emit, sub,
    text_width, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402
    DOT, IMPLIES, MINUS, NDASH, SIGMA, TIMES, apoly, aw, badge, mbox, opnode, pill,
)
from analysis.round5_figs_obj import inset  # noqa: E402
from analysis.fig_audit import audit_all  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-07")
W = 940

_FRAMES = []


def frame(x, y, w, h, title, key, note=""):
    _FRAMES.append((x, y, x + w, y + h, title))
    return inset(x, y, w, h, title, key, note)


def title(t, sub_=""):
    s = txt(W / 2, 46, t, 20, anchor="middle", fill=INK, weight="bold")
    if sub_:
        s += txt(W / 2, 72, sub_, 13, anchor="middle", fill=MUTED)
    return s


def eqband(x, y, w, eq, key, note=""):
    """An equation in its own reserved box.

    Equations get a registered frame rather than floating text so the audit can police them:
    a formula drawn over a connector is the single most common way these figures go wrong,
    and it is invisible to a text-only check when the connector is a <path>.
    """
    h = 52 if not note else 70
    _FRAMES.append((x, y, x + w, y + h, "eq"))
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
         f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="1.2" '
         f'opacity="0.55"/>\n')
    s += txt(x + w / 2, y + 33, eq, 15.5, anchor="middle", fill=INK, weight="bold")
    if note:
        s += txt(x + w / 2, y + 57, note, 11.5, anchor="middle", fill=MUTED)
    return s


def wire(x1, y1, x2, y2, key, label="", side="right"):
    """A connector with its tensor shape annotated beside it, never on it."""
    s = aw(x1, y1, x2, y2, key)
    if label:
        if x1 == x2:                                    # vertical
            s += txt(x1 + (9 if side == "right" else -9), (y1 + y2) / 2 + 4, label, 10,
                     anchor="start" if side == "right" else "end", fill=MUTED)
        else:
            s += txt((x1 + x2) / 2, y1 - 9, label, 10, anchor="middle", fill=MUTED)
    return s


def node(cx, cy, label, key, r=26, size=15):
    """A circle with a centred label.

    NOT opnode(): its `sym` argument is a glyph NAME ("minus"/"times"/"plus") drawn as a
    path, and any other string silently yields an empty circle -- which is exactly what the
    first render of these figures produced for every observation node.
    """
    c = ACCENT[key]
    s = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{c}" '
         f'stroke-width="1.8"/>\n')
    return s + txt(cx, cy + size * 0.35, label, size, anchor="middle", fill=INK)


def clip_flow(x0, y_obs, cols, key, enc_key="blue", labels=None, obs_pfx="o",
              z_pfx="z", pitch=150, show_enc=True):
    """The unrolled clip: one column per frame, obs -> Enc -> code.

    Returns (svg, [column centre x], y of the code row). Shared by every figure so the
    reader learns the spine once.
    """
    s = ""
    xs = []
    for t in range(cols):
        cx = x0 + t * pitch
        xs.append(cx)
        lab = labels[t] if labels else f"t{'' if t == 0 else '+' + str(t)}"
        s += node(cx, y_obs, obs_pfx + sub("", lab, 10), "slate", r=26)
        if show_enc:
            s += wire(cx, y_obs + 26, cx, y_obs + 64, enc_key)
            s += mbox(cx - 46, y_obs + 64, 92, 44, "Enc", enc_key, size=15)
            s += wire(cx, y_obs + 108, cx, y_obs + 146, enc_key)
            s += txt(cx, y_obs + 168, z_pfx + sub("", lab, 10), 15, anchor="middle",
                     fill=ACCENT[enc_key], weight="bold")
    if show_enc and cols > 1:
        s += txt(x0 - 62, y_obs + 82, "shared", 10.5, anchor="end", fill=MUTED)
        s += txt(x0 - 62, y_obs + 96, "weights", 10.5, anchor="end", fill=MUTED)
    return s, xs, y_obs + 168


# ==================================================================== the premise
def premise():
    _FRAMES.clear()
    K = "crit"
    s = title("Why a stateful encoder needs an EXACT single-frame limit",
              "PiWM-blockcausal: best fit in the archive, full-rank code, and success 0.00")

    # left: what training sees
    s += frame(48, 104, 420, 400, "TRAINING  " + NDASH + "  the encoder sees T = 3", "blue",
               note="every frame attends to the earlier ones")
    f, xs, yz = clip_flow(162, 210, 3, K, enc_key="blue", pitch=104)
    s += f
    for i in range(2):
        s += apoly([(xs[i] + 30, 296), (xs[i + 1] - 30, 296)], "blue")
    s += txt(258, 452, "z depends on the whole clip", 12, anchor="middle",
             fill=ACCENT["blue"], weight="bold")

    # right: what the goal is
    s += frame(492, 104, 400, 400, "GOAL  " + NDASH + "  the encoder sees T = 1", K,
               note="planning/cem.py:77 with plan.py:233")
    f2, xs2, _ = clip_flow(692, 210, 1, K, enc_key=K, labels=["g"], obs_pfx="o", z_pfx="z")
    s += f2
    s += txt(692, 412, "nothing to attend to", 12.5, anchor="middle", fill=ACCENT[K],
             weight="bold")
    s += txt(692, 436, "z_goal is an object training never produced", 11, anchor="middle",
             fill=MUTED)
    s += txt(692, 470, "the planner aims at it anyway", 12, anchor="middle", fill=ACCENT[K])

    s += eqband(48, 528, 844, "attention over a length-1 sequence is a DIFFERENT function; "
                "an SSM with s" + sub("", "0", 11) + " = 0 is the same one", K,
                note="which is why round 9 uses a state-space operator and not attention")

    # the evidence: four collapses and one anomaly
    s += frame(48, 622, 844, 268, "IT IS NOT SIMPLY \"LOW ERROR IS BAD\"", "slate",
               note="the five lowest rel_mse in 96 arms -- four have a dead code, one does not")
    hdr = [("arm", 96, "start"), ("rel_mse", 470, "end"), ("eff_dim", 640, "end"),
           ("success", 852, "end")]
    for t, x, a in hdr:
        s += txt(x, 692, t, 11, anchor=a, fill=MUTED)
    rows = [("PiWM-drop95", "0.00000", "0.00", "0.006", True),
            ("LpWM-ltv-d2048-hilr", "0.00000", "0.00", "0.010", True),
            ("PiWM-white-dz", "0.00082", "14.31", "0.010", True),
            ("PiWM-white-zt", "0.00110", "12.33", "0.013", True),
            ("PiWM-blockcausal", "0.00155", "25.04", "0.000", False),
            ("LpWM-ltv  (baseline)", "0.00919", "24.10", "0.357", None)]
    for i, (nm, rm, ed, sr, dead) in enumerate(rows):
        y = 720 + i * 26
        col = ACCENT["magenta"] if dead else (ACCENT[K] if dead is False else INK)
        bold = "bold" if dead is False else "normal"
        s += txt(96, y, nm, 12, anchor="start", fill=col, weight=bold)
        s += txt(470, y, rm, 12, anchor="end", fill=MUTED)
        s += txt(640, y, ed, 12, anchor="end", fill=col, weight=bold)
        s += txt(852, y, sr, 12, anchor="end", fill=col, weight=bold)
    s += txt(470, 876, "a constant code is trivially predictable "
             + IMPLIES + "  four of these earned zero error by dying",
             12, anchor="middle", fill=MUTED)
    return s, 934


# ==================================================================== P1
def p1():
    _FRAMES.clear()
    K = "amber"
    s = title("(P1)  PiWM-enc-ssm " + NDASH + "  a temporal state after the per-frame encoder",
              "the ViT is untouched; three matrices are added on top of its output")

    f, xs, yz = clip_flow(200, 128, 3, K, enc_key="blue", pitch=180)
    s += f

    # the new component: one band spanning the three columns
    bx0, bx1 = xs[0] - 78, xs[-1] + 78
    s += frame(bx0, 330, bx1 - bx0, 120, "THE NEW COMPONENT", K,
               note="one recurrence, shared across frames and across tokens")
    for i, cx in enumerate(xs):
        s += wire(cx, yz + 12, cx, 330, K)
        s += mbox(cx - 34, 384, 68, 42, "s" + sub("", f"t{'' if i == 0 else '+' + str(i)}", 10),
                  K, size=15)
        if i:
            s += apoly([(xs[i - 1] + 34, 405), (cx - 34, 405)], K)
            s += txt((xs[i - 1] + cx) / 2, 396, "A", 12.5, anchor="middle",
                     fill=ACCENT[K], weight="bold")
    for i, cx in enumerate(xs):
        s += wire(cx, 450, cx, 494, K)
        s += txt(cx, 516, sub("z\u2032", f"t{'' if i == 0 else '+' + str(i)}", 10), 15,
                 anchor="middle", fill=ACCENT[K], weight="bold")
    s += apoly([(xs[-1] + 24, 510), (700, 510)], K)
    s += mbox(700, 488, 96, 44, "Pred", "magenta", size=15)
    s += wire(796, 510, 848, 510, "magenta")
    s += txt(872, 516, sub("z", "t+4", 10), 15, anchor="middle", fill=ACCENT["magenta"],
             weight="bold")
    s += caret(867, 500, color=ACCENT["magenta"])

    s += eqband(48, 556, 844,
                "s" + sub("", "t", 11) + " = A s" + sub("", "t-1", 11) + " + B z"
                + sub("", "t", 11) + "      " + DOT + "      z′" + sub("", "t", 11)
                + " = C s" + sub("", "t", 11), K,
                note="three nn.Linear(D, D, bias=False); D = 384; no change to the ViT")

    s += frame(48, 650, 412, 214, "THE CONTRACT  " + NDASH + "  A = 0 is the baseline", "blue",
               note="verified by loss_trace bit-identity, not by inspection")
    for i, t in enumerate(("A = 0, B = C = I at init", "s" + sub("", "0", 10) + " = 0, so A"
                           + DOT + "0 = 0 exactly",
                           "z\u2032" + sub("", "t", 10) + " = z" + sub("", "t", 10)
                           + " for every t, and for T = 1",
                           "so z_goal is the baseline's z_goal")):
        s += txt(84, 722 + i * 30, t, 12.5, anchor="start",
                 fill=ACCENT["blue"] if i == 3 else INK)

    s += frame(488, 650, 404, 214, "THE CONTROL  " + NDASH + "  enc-ssm-frozen", "magenta",
               note="same parameters, same ops, same RNG draw")
    for i, t in enumerate(("A allocated and zero-initialised", "A.requires_grad_(False)",
                           "state cannot flow, capacity identical",
                           "a win here is STATE, not parameters")):
        s += txt(524, 722 + i * 30, t, 12.5, anchor="start",
                 fill=ACCENT["magenta"] if i == 3 else INK)
    return s, 902


# ==================================================================== P2
def p2():
    _FRAMES.clear()
    K = "amber"
    s = title("(P2)  PiWM-enc-ssm-deep " + NDASH
              + "  the same state, carried between ViT blocks",
              "the direct test of the premise: early mixing WITH a single-frame limit")

    s += frame(48, 104, 420, 560, "THE STACK", "blue",
               note="state inserted after every k-th block, not once at the end")
    ys = 168
    items = [("patch embed", "blue", "(b" + DOT + "T, 257, 384)"),
             ("ViT block  1..k", "blue", ""),
             ("state scan over T", K, "the insertion"),
             ("ViT block  k+1..2k", "blue", ""),
             ("state scan over T", K, "the insertion"),
             ("ViT block  ...12", "blue", ""),
             ("projector", "blue", "(b" + DOT + "T, P, 384)")]
    for i, (lab, key, note) in enumerate(items):
        y = ys + i * 66
        s += mbox(148, y, 220, 44, lab, key, size=13.5)
        if note:
            s += txt(384, y + 27, note, 10, anchor="start", fill=MUTED)
        if i < len(items) - 1:
            s += aw(258, y + 44, 258, y + 66, "blue" if key == "blue" else K)

    s += frame(488, 104, 404, 264, "WHY IT MUST SUBCLASS Block", "crit",
               note="infojepa_modules.py:220 dispatches on isinstance")
    s += txt(524, 176, "the Transformer loop is:", 12, anchor="start", fill=MUTED)
    for i, t in enumerate(("if isinstance(block, Block):",
                           "    x = block(x, attn_mask=attn_mask)",
                           "else:",
                           "    x = block(x, c, attn_mask=attn_mask)")):
        s += txt(536, 204 + i * 24, t, 11.5, anchor="start",
                 fill=INK if i < 2 else ACCENT["crit"])
    s += txt(690, 322, "a non-subclass is called with 3 args", 12, anchor="middle",
             fill=ACCENT["crit"], weight="bold")
    s += txt(690, 346, "and crashes on arity", 11.5, anchor="middle", fill=MUTED)

    s += frame(488, 392, 404, 272, "WHAT IT TESTS", K,
               note="blockcausal had early mixing and NO single-frame limit")
    for i, t in enumerate(("blockcausal mixed early " + IMPLIES + " SR 0.00",
                           "P2 mixes early WITH the limit",
                           "",
                           "if P2 works  " + IMPLIES + "  the premise is right",
                           "if P2 is ~0  " + IMPLIES + "  early mixing is",
                           "        harmful for another reason")):
        if not t:
            continue
        s += txt(524, 462 + i * 30, t, 12.5, anchor="start",
                 fill=ACCENT[K] if i >= 3 else INK)

    s += eqband(48, 690, 844,
                "same recurrence as P1, applied to (b, T, L, D) at each insertion depth", K,
                note="StateBlock(Block) folds (b, T" + DOT + "L, D) -> (b, T, L, D), scans T, "
                     "folds back; T is num_hist, fixed per run")
    return s, 800


# ==================================================================== P3
def p3():
    _FRAMES.clear()
    K = "amber"
    s = title("(P3)  PiWM-enc-scan " + NDASH
              + "  a bidirectional scan replacing attention over tokens",
              "the SPATIAL axis; orthogonal to the temporal question P1 and P2 ask")

    s += frame(48, 104, 420, 330, "ATTENTION  " + NDASH + "  all pairs, O(L" + SIGMA + ")",
               "blue", note="every token sees every token, in one step")
    cxs = [128, 196, 264, 332, 400]
    for i, cx in enumerate(cxs):
        s += opnode(cx, 220, "", "blue", r=15)
    for i in range(len(cxs)):
        for j in range(i + 1, len(cxs)):
            s += (f'<line x1="{cxs[i]}" y1="{235}" x2="{cxs[j]}" y2="{235}" '
                  f'stroke="{ACCENT["blue"]}" stroke-width="0.8" opacity="0.35"/>\n')
    s += txt(258, 300, "L = 257 tokens  (256 patches + CLS)", 12, anchor="middle", fill=MUTED)
    s += txt(258, 336, "F.scaled_dot_product_attention", 11.5, anchor="middle",
             fill=ACCENT["blue"])
    s += txt(258, 396, "inner_dim = heads " + TIMES + " dim_head = 192", 11.5,
             anchor="middle", fill=MUTED)

    s += frame(488, 104, 404, 330, "SCAN  " + NDASH + "  two sweeps, O(L)", K,
               note="forward and backward; patch tokens have no causal order")
    sxs = [560, 628, 696, 764, 832]
    for cx in sxs:
        s += opnode(cx, 196, "", K, r=15)
    for i in range(len(sxs) - 1):
        s += apoly([(sxs[i] + 15, 196), (sxs[i + 1] - 15, 196)], K)
    for cx in sxs:
        s += opnode(cx, 276, "", "magenta", r=15)
    for i in range(len(sxs) - 1, 0, -1):
        s += apoly([(sxs[i] - 15, 276), (sxs[i - 1] + 15, 276)], "magenta")
    s += txt(690, 236, "forward sweep", 11, anchor="middle", fill=ACCENT[K])
    s += txt(690, 316, "backward sweep", 11, anchor="middle", fill=ACCENT["magenta"])
    s += txt(690, 396, "merge(concat)  " + IMPLIES + "  (B, L, D)", 11.5, anchor="middle",
             fill=INK)

    s += eqband(48, 460, 844,
                "s" + sub("", "i", 11) + " = A s" + sub("", "i-1", 11) + " + B x"
                + sub("", "i", 11) + " ,   out" + sub("", "i", 11) + " = C s"
                + sub("", "i", 11) + "     over the TOKEN axis, both directions", K,
                note="drop-in for Attention: forward(x, attn_mask=None) -> (B, L, D), "
                     "own LayerNorm, dropout gated on self.training")

    s += frame(48, 554, 412, 240, "REQUIRES FEATURE = patch", "crit",
               note="at cls there is one token and nothing to scan")
    for i, t in enumerate(("num_patches = 1 at cls",
                           "the scan would be a no-op",
                           "so the arm ASSERTS rather than",
                           "silently degenerating")):
        s += txt(84, 626 + i * 30, t, 12.5, anchor="start",
                 fill=ACCENT["crit"] if i >= 2 else INK)

    s += frame(488, 554, 404, 240, "COST, AND THE CONTROL", K,
               note="257 sequential steps per block, 12 blocks")
    for i, t in enumerate(("a Python loop, not a fused kernel",
                           "8 windows budgeted; the canary decides",
                           "control: scan_A frozen at zero",
                           "which degenerates to a per-token MLP")):
        s += txt(524, 626 + i * 30, t, 12.5, anchor="start",
                 fill=ACCENT[K] if i >= 2 else INK)
    return s, 832


# ==================================================================== P4
def p4():
    _FRAMES.clear()
    K = "amber"
    s = title("(P4)  PiWM-enc-st-scan " + NDASH + "  spatial scan and temporal state composed",
              "ablated by P1 (temporal only) and P3 (spatial only) -- both of which run")

    s += frame(48, 104, 844, 400, "THE TWO AXES, IN ONE ARM", K,
               note="within a frame the scan runs over tokens; across frames the state carries")
    for t in range(3):
        x0 = 132 + t * 264
        s += txt(x0 + 88, 168, "frame t" + ("" if t == 0 else "+" + str(t)), 12.5,
                 anchor="middle", fill=MUTED)
        s += frame(x0, 184, 176, 130, "", "blue")
        toks = [x0 + 32, x0 + 72, x0 + 112, x0 + 152]
        for cx in toks:
            s += opnode(cx, 250, "", "blue", r=13)
        for i in range(len(toks) - 1):
            s += apoly([(toks[i] + 13, 250), (toks[i + 1] - 13, 250)], "blue")
        s += txt(x0 + 88, 300, "scan over tokens", 10.5, anchor="middle",
                 fill=ACCENT["blue"])
        s += mbox(x0 + 52, 344, 72, 40, "s" + sub("", "t" + ("" if t == 0 else "+" + str(t)), 10),
                  K, size=14)
        s += aw(x0 + 88, 314, x0 + 88, 344, K)
        if t:
            s += apoly([(x0 - 88, 364), (x0 + 52, 364)], K)
            s += txt(x0 - 18, 356, "A", 12, anchor="middle", fill=ACCENT[K], weight="bold")
    s += txt(470, 434, "spatial within " + DOT + " temporal across  "
             + IMPLIES + "  one operator family on both axes",
             12.5, anchor="middle", fill=ACCENT[K], weight="bold")
    s += txt(470, 464, "a win is attributable because P1 and P3 isolate each half",
             11.5, anchor="middle", fill=MUTED)

    s += eqband(48, 530, 844,
                "within a frame:  s" + sub("", "i", 11) + " = A" + sub("", "s", 11)
                + " s" + sub("", "i-1", 11) + " + B" + sub("", "s", 11) + " x"
                + sub("", "i", 11) + "          |          across frames:  s"
                + sub("", "t", 11) + " = A" + sub("", "t", 11) + " s"
                + sub("", "t-1", 11) + " + B" + sub("", "t", 11) + " z"
                + sub("", "t", 11), K,
                note="both reduce to identity when their A is zero, so the arm nests both "
                     "controls")

    s += frame(48, 624, 844, 178, "THE ABLATION MAP", "slate",
               note="every cell already exists or is in this round")
    cols = [("", 200), ("no spatial", 420), ("spatial scan", 700)]
    for lab, x in cols:
        s += txt(x, 686, lab, 11.5, anchor="middle", fill=MUTED)
    grid = [("no temporal", "baseline", "P3"), ("temporal state", "P1", "P4")]
    for r, (rl, c1, c2) in enumerate(grid):
        y = 720 + r * 46
        s += txt(200, y + 6, rl, 12, anchor="middle", fill=MUTED)
        for c, cell in enumerate((c1, c2)):
            cx = 420 + c * 280
            key = K if cell == "P4" else ("blue" if cell == "baseline" else "slate")
            s += mbox(cx - 66, y - 16, 132, 40, cell, key, size=13.5)
    return s, 840


# ==================================================================== ladder
def ladder():
    _FRAMES.clear()
    K = "amber"
    s = title("THE LADDER " + NDASH + "  manufactured partial observability",
              "neither PushT nor Wall hides anything; the corruption creates what "
              "the claim needs")

    s += frame(48, 104, 844, 250, "THREE RUNGS, ON THE OBSERVATION STREAM ONLY", K,
               note="never actions, never proprio; the anchor frame is never dropped")
    DROP = {0: set(), 1: {2}, 2: {1, 2}}
    for r, lab in enumerate(("p = 0.00", "p = 0.15", "p = 0.35")):
        x0 = 132 + r * 268
        key = "blue" if r == 0 else K
        s += txt(x0 + 88, 168, lab, 13, anchor="middle", fill=ACCENT[key], weight="bold")
        for t in range(3):
            cx = x0 + t * 62
            gone = t in DROP[r]
            s += mbox(cx, 186, 52, 44, "" if gone else "o" + sub("", str(t), 10),
                      "crit" if gone else key, size=13)
            if gone:
                s += txt(cx + 26, 214, TIMES, 17, anchor="middle", fill=ACCENT["crit"],
                         weight="bold")
        s += txt(x0 + 88, 262, "the control" if r == 0 else
                 f"{len(DROP[r])} of 3 blank", 11.5, anchor="middle",
                 fill=ACCENT["blue"] if r == 0 else MUTED)
        s += txt(x0 + 88, 288, "bit-identical to today" if r == 0 else
                 "state must carry it", 11.5, anchor="middle", fill=ACCENT[key])

    s += eqband(48, 380, 844,
                "visual[m] = 0  where  m ~ Bernoulli(p) per frame,  m[0] = False", K,
                note="a dataset transform: no model code, and p = 0 is the whole ladder's "
                     "control")

    s += frame(48, 474, 844, 300, "THE TWO CLAIMS MAKE OPPOSITE PREDICTIONS", "slate",
               note="which is what makes this an experiment rather than a sweep")
    for i, (nm, key, pred, note) in enumerate((
            ("CLAIM A  denoising", "blue", (0.5, 0.55, 0.6),
             "a small gain everywhere, including p = 0"),
            ("CLAIM B  belief state", K, (0.03, 0.4, 0.95),
             "~0 at p = 0, rising with corruption"))):
        x0 = 96 + i * 424
        s += txt(x0 + 176, 546, nm, 13, anchor="middle", fill=ACCENT[key], weight="bold")
        gx, gy, gw, gh = x0 + 46, 566, 262, 112
        s += (f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="3" '
              f'fill="{WHITE}" stroke="{DIM}" stroke-width="1"/>\n')
        s += (f'<line x1="{gx}" y1="{gy + gh}" x2="{gx + gw}" y2="{gy + gh}" '
              f'stroke="{GRID}" stroke-width="1.2"/>\n')
        pts = [(gx + 8 + j * ((gw - 16) / 2), gy + gh - v * (gh - 10) - 5)
               for j, v in enumerate(pred)]
        s += apoly(pts, key)
        for (cx, cy) in pts:
            s += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{ACCENT_FILL[key]}" '
                  f'stroke="{ACCENT[key]}" stroke-width="2"/>\n')
        for j, l in enumerate(("0", ".15", ".35")):
            s += txt(gx + 8 + j * ((gw - 16) / 2), gy + gh + 20, l, 10.5,
                     anchor="middle", fill=MUTED)
        s += txt(x0 + 176, 728, note, 11.5, anchor="middle", fill=INK)
    s += txt(470, 758, "a flat response refutes B  " + DOT
             + "  a gain only at p = 0 refutes B and supports A  " + DOT
             + "  neither refutes both", 11.5, anchor="middle", fill=MUTED)
    return s, 812


FIGURES = (
    ("arch-premise.svg", premise),
    ("arch-p1-enc-ssm.svg", p1),
    ("arch-p2-deep.svg", p2),
    ("arch-p3-scan.svg", p3),
    ("arch-p4-st-scan.svg", p4),
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
        body, h = fn()
        frames = tuple(_FRAMES)
        emit(name, body, a.out, w=W, h=h)
        path = os.path.join(a.out, name)
        bad = audit_all(path, W, h, frames)
        print(f"  {name:<28} {'CLEAN' if not bad else str(len(bad)) + ' finding(s)'}")
        for line in bad[:8]:
            print("     ", line[:110])
        rc |= bool(bad)
        if a.png:
            import cairosvg
            cairosvg.svg2png(url=path, write_to=path.replace(".svg", ".png"),
                             output_width=W * 2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
