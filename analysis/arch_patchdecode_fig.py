"""T2 -- PiWM-patchdecode, an AUXILIARY PIXEL DECODER on the patch representation.

    python analysis/arch_patchdecode_fig.py --out diary/assets/2026-09-05
    -> arch-patchdecode.svg

PiWM-columns plus AUX_DECODER=true, DECODER=patch_head, DECODE_GRAD=true, LAMB_DECODE=0.1:
a per-patch head reconstructs pixels from the 256 patch tokens and its loss is added to the
training objective at weight 0.1.  The whole arm is one boolean wide.  With DECODE_GRAD=true
the reconstruction gradient REACHES THE ENCODER; the `-detach` control sets it false, so the
decoder still trains and the pixel loss is still added -- only the gradient path is cut
(models/visual_world_model.py:1630, `z_dec = z_emb if self.decode_grad else z_emb.detach()`).

WHY THIS FIGURE EXISTS.  patchdecode carries the campaign's highest arm mean outside the
consensus family, and the decoder is the only mechanism that RECURS among the arms with a
positive paired mean against the baseline.  Both facts are recomputed at render time
(`_rank_rows`, `_recurrence`) and asserted rather than quoted.

THE FACT THE FIGURE IS BUILT AROUND: the detach comparison INVERTS with the representation.

    98 304 dims (patch, 256 tokens)   PiWM-patchdecode      beats  PiWM-patchdecode-detach
       384 dims (cls,     1 token)    PiWM-decode-detach    beats  PiWM-decode

Same decoder plumbing, same lambda, same seeds, same eval instrument on both sides; only the
FEATURE and the one boolean differ, and the sign of the boolean's effect flips.  The figure
states the four means and the flip and says nothing about why: a claim about what a
reconstruction gradient does to a 256-token latent versus a 1-token latent is not supported
by two n = 8 contrasts, and BOTH of them are nulls.  So the inversion carries an UNEXPLAINED
badge, and each contrast carries the house's null encoding (slate, a hollow marker, the word
NULL) rather than being drawn as a result.

STYLE.  analysis/style.py, through analysis/arch_figs.py.  Every primitive is imported from
analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py / round6_arch_figs.py /
round7_arch_figs.py -- nothing here re-implements one, and no existing module is edited (nine
other figure modules import them).  Hues by IDENTITY, never by rank:

    amber    the intervention under test -- the decoder branch, and the CLOSED gradient
             switch (DECODE_GRAD=true).  The two inversion arrows are BOTH amber and point
             opposite ways: the hue says "this is the switch", the direction says the sign.
    purple   the contrasting condition -- the -detach control, and the OPEN blade
    slate    neutral -- the latent itself, the axes, both NULLs, the UNEXPLAINED mark
    crimson  reserved for the caveat that neither interval excludes zero
    green    the system as built; here it is carried by base()'s own schematic only, since
             no arm in this figure is the unmodified system
    teal     unused: no consensus arm appears here

LAYOUT RULES INHERITED FROM round6_arch_figs.py, and the defect each one stops

  * Strict left-to-right dataflow on fixed row baselines; no two wires share a segment.
    The forward chain runs left-to-right at y = 1064 and the GRADIENT runs right-to-left at
    y = 1152, below it, with the arrowhead landing back on the token stack -- opposite
    directions on their own baselines, which is the only way "this wire is the backward
    pass" can be read off the picture.
  * base() fixes every anchor, so a connector uses ONE named route.  This arm ADDS A TERM TO
    THE LOSS, so it takes round6's `wire_to_mse` verbatim, with a `plus` operator node: the
    house route for exactly that, used here at its own sign.  There is no second connector;
    the decoder branch, the switch and the inversion are all drawn inside the panel, because
    a second right-margin corridor at x = 890 would run parallel to wire_to_mse's at x = 880
    and two wires 10 units apart read as one.
  * Nothing is sized by hand where it can be measured: mbox / pill / inset / frow / strip
    all fit their own text and none may cross the panel border.
  * A NULL is never drawn like a positive.  +0.140 [-0.047, +0.327] and -0.060
    [-0.205, +0.085] both span zero, so both get slate, a HOLLOW marker and the word NULL --
    the encoding round7-design.svg and round7_v2 use.  The INVERSION is therefore stated as
    an inversion of the two MEANS, which is what the archive supports.
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of
    every <text> element from the same Helvetica metrics the layout used
    (arch_figs.text_width), and reports any pair that intersects, anything that leaves the
    canvas, and anything that crosses the border of the panel or inset it sits in (every
    frame registers itself through `frame()` / `pane()`).  Copied from
    analysis/round7_v2_arch_figs.py rather than imported, so this module's audit cannot
    drift when that one is edited.  At 940 x 1764 units "nothing overlaps" is not checkable
    by eye, and it found four real defects here that a read of the rendered PNG missed.

Usage:  python analysis/arch_patchdecode_fig.py --out diary/assets/2026-09-05 [--png]
        `--png` rasterises beside the SVG, for the read-back; the audit runs either way and
        main() exits non-zero if it has anything to say.

EVERY NUMBER DRAWN, AND WHERE IT IS READ FROM.  Nothing is typed as a literal.

  the arm specification
    AUX_DECODER / DECODER / DECODE_GRAD / LAMB_DECODE for all four arms, and
      FEATURE=patch on the two patchdecode arms                scripts/run_campaign.sh,
                                                               parsed by `_arm_specs`
    img_size 224, embed_dim 384                                conf/train_rdmreg.yaml
    patch_size 14, feature patch                               conf/encoder/vit_scratch_patch.yaml
    patch_size 14, grid 16, out_chans 3                        conf/decoder/patch_head.yaml
    z_dec = z_emb if decode_grad else z_emb.detach()           models/visual_world_model.py:1630

  the measurements, all from collect_evals.collect(scheme="fixed") through resolve_arm --
  the patch arms are keyed <name>_patch, which has silently zeroed three callers
    PiWM-patchdecode / -detach arm means and n                 plan_outputs/*/logs.json
    PiWM-decode / -detach arm means and n                      the same
    both paired contrasts, mean, 95 % CI and n                 figures.paired_effect, which
                                                               drops unmatched seeds rather
                                                               than mean-imputing them

  ARITHMETIC DONE HERE, and nothing else:
    num_patches = ( img_size // patch_size ) ** 2 = 256, verbatim from
      models/vit_encoder.py:65-67, and the two total latent dimensions 256 x 384 = 98 304
      and 1 x 384 = 384 that the two halves of the inversion are labelled with;
    PatchHead's output width 3 x 14**2 = 588 and its parameter count
      384 x 588 + 588 = 226 380, which is one nn.Linear (models/decoder/patch_head.py);
    the rank of patchdecode's arm mean among arms outside the consensus family that share
      at least 5 evaluated seeds with LpWM-ltv (asserted to be 1);
    the recurrence count: of the arms compared to LpWM-ltv at n >= 8, how many have a
      positive paired mean, and how many of those set AUX_DECODER=true.
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import figures as FG  # noqa: E402
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    _marker, arrow, base, domeup, poly, sub, text_width, title_line, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, DOT, IMPLIES, LAMB, MINUS, NDASH, TIMES,
    apoly, aw, badge, dot, fit_size, mbox, opnode, pill, strip,
)
from analysis.collect_evals import collect, resolve_arm  # noqa: E402
from analysis.round5_figs_obj import (  # noqa: E402,F401
    frow, inset, panel, sgbar, strikeout,
)
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PIX, PW, PX, PY, SHIFT, W, curve, ellipsis_v, mark, out_emit, wire_to_mse,
)
from analysis.round7_arch_figs import cbadge, tokengrid  # noqa: E402

DELTA = "Δ"                 # capital delta; its metrics are registered in arch_figs_causal
SQ2 = "²"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-05")

BASELINE = "LpWM-ltv"
#: The two contrasts the figure is: (feature, treated arm, its own -detach control).
PAIRS = (("patch", "PiWM-patchdecode", "PiWM-patchdecode-detach"),
         ("cls", "PiWM-decode", "PiWM-decode-detach"))
#: Consensus / oracle arms are excluded from the rank and the recurrence count: a committee
#: is not an objective, and an oracle is not an arm at all.
_CONSENSUS = re.compile(r"vote|solo|loo|Oracle", re.I)
#: A rank over arms that barely overlap the baseline's seeds is a rank over different
#: episodes.  5 is the floor used for the rank; the recurrence count uses the pairing's own
#: n >= 8, which is the campaign's seed floor.
MIN_SHARED, N_FLOOR = 5, 8


# --- the archive -------------------------------------------------------------------
def _text(rel):
    return open(os.path.join(REPO, rel)).read()


def _num(rel, key, cast=int):
    """One `key: value` line of a config, read at render time rather than transcribed."""
    m = re.search(rf"^{re.escape(key)}:\s*([-\d.]+)", _text(rel), re.M)
    assert m, f"{key} not found in {rel}"
    return cast(m.group(1))


IMG = _num("conf/train_rdmreg.yaml", "img_size")
EMB = _num("conf/train_rdmreg.yaml", "embed_dim")
PATCH = _num("conf/encoder/vit_scratch_patch.yaml", "patch_size")
DEC_P = _num("conf/decoder/patch_head.yaml", "patch_size")
DEC_C = _num("conf/decoder/patch_head.yaml", "out_chans")


def tokens_for(patch_size):
    """models/vit_encoder.py:65-67, verbatim: grid = image_size // patch_size,
    num_patches = grid * grid.  patch_size is the ONLY token knob."""
    return (IMG // patch_size) ** 2


P_TOK = tokens_for(PATCH)                    # 256
DIMS = {"patch": P_TOK * EMB, "cls": 1 * EMB}
TOKENS = {"patch": P_TOK, "cls": 1}
HEAD_OUT = DEC_C * DEC_P ** 2                # 3 * 14**2 = 588
HEAD_PARAMS = EMB * HEAD_OUT + HEAD_OUT      # one nn.Linear(emb_dim, c*p*p)

_SPEC = re.compile(r'^\s*ARMS\[([^\]]+)\]="([^"]*)"', re.M)
_FEAT = re.compile(r'^\s*ARM_FEAT\[([^\]]+)\]="([^"]*)"', re.M)


def _arm_specs():
    """{arm: {KEY: VALUE}} for every `ARMS[name]="... KEY=VALUE ..."` in the campaign
    script, plus a synthetic FEATURE from ARM_FEAT (absent => cls, which is train.sh's
    default).  The arm definition is the run script; quoting it here would let the figure
    and the launcher disagree silently."""
    src = _text("scripts/run_campaign.sh")
    out = {}
    for name, spec in _SPEC.findall(src):
        kv = dict(tok.split("=", 1) for tok in spec.split() if "=" in tok)
        out.setdefault(name, {}).update(kv)
    for name, feat in _FEAT.findall(src):
        out.setdefault(name, {})["FEATURE"] = feat
    for d in out.values():
        d.setdefault("FEATURE", "cls")
    return out


SPECS = _arm_specs()


def _archive():
    """collect_evals resolves plan_outputs/ and slurm_logs/ RELATIVE TO THE CWD, and the
    slurm table is what classifies a run as fixed- or buggy-instrument, so the read is done
    from REPO whatever directory the module was invoked from.  Getting this wrong does not
    raise: it silently falls back to the mtime heuristic, which misclassifies six runs."""
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        return collect(scheme="fixed")[0]
    finally:
        os.chdir(cwd)


def _mean_n(arms, name):
    v = arms[resolve_arm(arms, name)]
    return sum(v.values()) / len(v), len(v)


def _pe(arms, ctrl, arm):
    return FG.paired_effect(arms, resolve_arm(arms, ctrl), resolve_arm(arms, arm))


def _rank_rows(arms):
    """(mean, arm) over non-consensus arms, on the seeds they share with the baseline.

    Restricted to the shared seeds because an unrestricted arm mean is a mean over whatever
    episode blocks that arm happened to be evaluated on -- the confound that put
    LpWM-ltv-d2048's 0.587 at the top of the raw table on seeds the baseline has no
    fixed-instrument eval for at all (2026-09-05 section 2)."""
    seeds = set(arms[resolve_arm(arms, BASELINE)])
    rows = []
    for k, v in arms.items():
        if _CONSENSUS.search(k):
            continue
        shared = [s for s in v if s in seeds]
        if len(shared) >= MIN_SHARED:
            rows.append((sum(v[s] for s in shared) / len(shared), k))
    return sorted(rows, reverse=True)


def _recurrence(arms):
    """(n_compared, n_positive, n_positive_with_decoder, n_decoder_arms, recurring_knobs).

    "Beats its baseline" is the paired mean against LpWM-ltv at the campaign's n >= 8 floor,
    not the raw arm mean.  A decoder arm is one whose spec in the campaign script sets
    AUX_DECODER=true -- read from the launcher, so the family cannot be defined by whatever
    happens to be spelled "decode".

    `recurring_knobs` is what makes "the only mechanism that recurs" a computed statement
    rather than an impression: it is the set of TRAINING KNOBS (the KEY of every KEY=VALUE in
    an arm's spec) that appear in more than one of the positive arms.  FEATURE is excluded
    because it is synthesised from ARM_FEAT and is a representation, not a mechanism -- and
    it is shared by patchdecode and columns, so leaving it in would answer the question with
    the figure's own other axis.
    """
    dec = {n for n, kv in SPECS.items() if kv.get("AUX_DECODER") == "true"}
    compared, positive, pos_dec, knobs = 0, 0, 0, {}
    for k in arms:
        if _CONSENSUS.search(k) or k == BASELINE:
            continue
        bare = k[:-len("_patch")] if k.endswith("_patch") else k
        e = FG.paired_effect(arms, BASELINE, k)
        if e["n"] < N_FLOOR:
            continue
        compared += 1
        if e["mean"] > 0:
            positive += 1
            pos_dec += bare in dec
            for knob in SPECS.get(bare, {}):
                if knob != "FEATURE":
                    knobs[knob] = knobs.get(knob, 0) + 1
    return (compared, positive, pos_dec, len(dec),
            {k for k, c in knobs.items() if c > 1})


def f3(v):
    return f"{v:.3f}"


def signed(v, nd=3):
    """A drawn signed number with a real minus sign, so it matches every other label."""
    return format(v, f"+.{nd}f").replace("-", MINUS)


# --- local primitives --------------------------------------------------------------
def gswitch(x0, x1, y, key_on, key_off, lift=27.0, r=3.6):
    """A two-position switch drawn IN A GAP in a wire: the gradient path and its control.

    The -detach control is not another module and not another loss -- it is ONE boolean on
    ONE line (models/visual_world_model.py:1630), so it is drawn as one blade with two
    positions rather than as two diagrams.  Two diagrams would have implied two arms that
    differ in more than a boolean, which is exactly what the control exists to deny.
    The closed blade is solid in the intervention's hue and bridges the gap; the open blade
    is dashed in the control's and is LIFTED off the contact, so the open state reads as a
    severed path and not as a missing one.
    """
    on, off = ACCENT[key_on], ACCENT[key_off]
    a = math.radians(lift)
    L = x1 - x0
    s = (f'<circle cx="{x0}" cy="{y}" r="{r}" fill="{WHITE}" stroke="{DIM}" '
         f'stroke-width="1.6"/>\n')
    s += (f'<path d="M{x1},{y} L{x0 + r:.1f},{y}" stroke="{on}" stroke-width="2.8" '
          f'fill="none" stroke-linecap="round"/>\n')
    s += (f'<path d="M{x1},{y} L{x1 - L * math.cos(a):.1f},{y - L * math.sin(a):.1f}" '
          f'stroke="{off}" stroke-width="2.6" fill="none" stroke-linecap="round" '
          f'stroke-dasharray="6,4"/>\n')
    s += f'<circle cx="{x1}" cy="{y}" r="{r + 1.4}" fill="{on}"/>\n'
    return s


def axis(x0, x1, y, ticks, xmap, key="slate", size=11):
    """A bare success-rate axis: one rule, ticks below it, no box.

    Both halves of the inversion are drawn against the SAME axis limits and the same tick
    set, computed once from the four means, because the whole claim is a comparison of two
    displacements and two axes would let a smaller one look larger."""
    c = ACCENT[key]
    s = (f'<path d="M{x0},{y} L{x1},{y}" stroke="{c}" stroke-width="1.3" '
         f'stroke-opacity="0.8"/>\n')
    for v in ticks:
        tx = round(xmap(v), 1)
        s += (f'<path d="M{tx},{y} L{tx},{y + 5}" stroke="{c}" stroke-width="1.1" '
              f'stroke-opacity="0.8"/>\n')
        s += txt(tx, y + 18, f"{v:.2f}", size, fill=MUTED)
    return s


def vrule(x, y0, y1, key, w=1.2, opacity=0.45):
    """The divider between the two halves: these are two SEPARATE experiments that happen
    to share an axis, and without the rule the four markers read as one series of four."""
    return (f'<path d="M{x},{y0} L{x},{y1}" stroke="{ACCENT[key]}" stroke-width="{w}" '
            f'stroke-opacity="{opacity}"/>\n')


# Every frame a label must stay inside, registered as it is drawn.  The audit's canvas
# check is not enough on its own: a label centred on one half of an inset can run well past
# that inset's border and still sit comfortably inside the canvas.
_FRAMES = []


def frame(x, y, w, h, title, key, note=""):
    """inset(), registered with the audit."""
    _FRAMES.append((x, y, x + w, y + h, title))
    return inset(x, y, w, h, title, key, note)


def pane(x, y, w, h, key, title):
    """panel(), registered with the audit."""
    _FRAMES.append((x, y, x + w, y + h, "panel"))
    return panel(x, y, w, h, key, title)


def _check_title(title, corridor_x, size=20, cx=470, clear=8):
    """The title is one centred run at y = 872 and the figure drops a connector past it, so
    its half-width is a HARD constraint, not a matter of taste."""
    body = re.sub(r"<[^>]*>", "", title)
    wide = text_width(body.replace(" -- ", " – "), size, "bold")
    right = cx + wide / 2
    assert right < corridor_x - clear, (
        f"title runs to x={right:.0f}, into the connector corridor at x={corridor_x}: "
        f"shorten it")
    return title


# ============================================== the figure
TITLE = ("(T2) PiWM-patchdecode -- a pixel decoder on the patch tokens  [NULL]")

# the two halves of the inversion, and the shared axis they are both drawn on
HX = (76, 470)                       # left edge of the patch half, of the cls half
HW = 390                             # half width; the divider sits between them
AX0, AX1 = 152, 368                  # axis span, as offsets from a half's left edge
YMARK_ON, YMARK_MID, YMARK_OFF = 1374, 1396, 1418
YAXIS, YGLYPH = 1458, 1359


def patchdecode():
    """One arm, one boolean, and a sign that flips with the representation."""
    K = "amber"                      # the intervention under test
    C = "magenta"                    # the contrasting condition: the -detach control
    arms = _archive()

    spec = SPECS["PiWM-patchdecode"]
    lam = spec["LAMB_DECODE"]
    assert spec["DECODE_GRAD"] == "true" and spec["FEATURE"] == "patch"
    assert SPECS["PiWM-patchdecode-detach"]["DECODE_GRAD"] == "false"
    assert SPECS["PiWM-decode"]["DECODE_GRAD"] == "true"
    assert SPECS["PiWM-decode-detach"]["DECODE_GRAD"] == "false"
    assert spec["DECODER"] == "patch_head"

    # the four means, and the two paired contrasts they invert on
    M = {}
    for feat, on, off in PAIRS:
        M[feat] = {"on": _mean_n(arms, on), "off": _mean_n(arms, off),
                   "e": _pe(arms, off, on), "on_name": on, "off_name": off}
    assert M["patch"]["e"]["mean"] > 0 > M["cls"]["e"]["mean"], "the inversion is gone"

    rank = _rank_rows(arms)
    assert rank[0][1] == resolve_arm(arms, "PiWM-patchdecode"), \
        f"patchdecode is no longer the top non-consensus arm mean: {rank[:2]}"
    n_cmp, n_pos, n_pos_dec, n_dec, knobs = _recurrence(arms)
    assert n_pos_dec > 1, "the decoder no longer recurs among the positives"
    assert knobs and knobs <= set(SPECS["PiWM-patchdecode"]) - {"FEATURE"}, (
        f"a knob outside the decoder now recurs among the positives too: {knobs}")

    vals = [M[f][k][0] for f in ("patch", "cls") for k in ("on", "off")]
    lo = math.floor((min(vals) - 0.02) * 20) / 20.0
    hi = math.ceil((max(vals) + 0.02) * 20) / 20.0
    ticks = [lo + i * 0.05 for i in range(int(round((hi - lo) / 0.05)) + 1)]

    # ---------------------------------------------------------------- the schematic
    b = base(_check_title(TITLE, 880), enc_sub="feature = patch")
    # the house route for "this arm adds a term to the loss", at its own sign.
    b += wire_to_mse(K, "plus", LAMB + " " + DOT + " MSE( decode( z ) , o )")
    # the one boolean, stated where the encoder is: anchored at x=712 rather than 736
    # because base()'s RDMReg return runs vertically at x=726 from y=442 to 582.
    b += txt(712, 502, "DECODE_GRAD = true   " + IMPLIES
             + "   the pixel gradient reaches Enc " + "<tspan font-style='italic'>f</tspan>",
             13.5, anchor="end", fill=ACCENT[K], weight="bold")

    PH = 906
    s = pane(PX, PY, PW, PH, K,
             "module:  PatchHead on the " + f"{P_TOK}" + " patch tokens  " + NDASH
             + "  loss  +=  " + LAMB + " " + DOT + " MSE( decode( z ) , o ) ,   "
             + LAMB + " = " + lam)

    # -- row A: the branch, and the gradient as a switchable wire ------------------
    s += frame(60, 956, 816, 292, "the branch " + NDASH + "  one nn.Linear, and one "
               "boolean on the return path", K,
               note="the same z the JEPA loss already uses;  DECODER = " + spec["DECODER"]
                    + " ,  AUX_DECODER = " + spec["AUX_DECODER"]
                    + " ,  LAMB_DECODE = " + lam)
    # the token stack the head hangs off.  One cell is one token, so the picture IS the
    # 256 the arithmetic says.
    # start-anchored on the glyph's own left edge: centred on the glyph it ran to x=49,
    # eleven units past the inset's inner margin.
    s += txt(72, 1019, sub("z", "t", 11) + "  " + DOT + "  " + str(P_TOK)
             + " tokens " + TIMES + " " + str(EMB), 12, anchor="start", fill=MUTED)
    s += tokengrid(108, 1027, P_TOK, "slate", total=P_TOK, cell=3.6, gap=0.7)
    s += aw(149, 1064, 196, 1064, K)
    s += mbox(200, 1040, 132, 48, "PatchHead", K, size=15,
              sub_="Linear " + str(EMB) + " " + ARROWC + " " + str(HEAD_OUT))
    s += aw(332, 1064, 356, 1064, K)
    s += mbox(360, 1040, 136, 48, str(P_TOK) + " blocks", K, size=15,
              sub_=f"{DEC_P} {TIMES} {DEC_P} {TIMES} {DEC_C}, by position")
    s += aw(496, 1064, 516, 1064, K)
    s += mbox(520, 1040, 148, 48, "MSE  vs  " + sub("o", "t", 11), K, size=15,
              sub_="decoder_criterion")
    s += aw(668, 1064, 690, 1064, K)
    s += opnode(708, 1064, "times", K, r=15)
    s += txt(708, 1032, LAMB + " = " + lam, 12.5, fill=ACCENT[K], weight="bold")
    s += aw(726, 1064, 750, 1064, K)
    s += pill(800, 1064, "the loss", K, rx=46, ry=20)

    # the BACKWARD path, on its own baseline below the forward one and running the other
    # way, with the switch sitting in a gap in it.
    s += curve([(594, 1088), (594, 1152), (398, 1152)], K, w=2.2)
    s += gswitch(320, 398, 1152, K, C)
    s += apoly([(320, 1152), (108, 1152), (108, 1106)], K, w=2.2)
    # end-anchored at 586, not started at 420: the wire's own vertical runs at x=594, and a
    # start-anchored caption ran 136 units through it.
    s += txt(586, 1138, "the switch:  DECODE_GRAD", 11.5, anchor="end", fill=MUTED)
    s += txt(200, 1132, "gradient", 12, anchor="end", fill=ACCENT[K], weight="bold")
    # the two blade positions named where they are drawn; the annotation rows below repeat
    # the words, which is what makes the rows a key to the glyph rather than prose.
    s += txt(359, 1174, "closed", 11, fill=ACCENT[K], weight="bold")
    s += txt(322, 1108, "open", 11, anchor="end", fill=ACCENT[C], weight="bold")

    s += txt(80, 1194, "closed", 12.5, anchor="start", fill=ACCENT[K], weight="bold")
    s += txt(142, 1194, "DECODE_GRAD = true   " + ARROWC + "   " + sub("z", "dec", 11)
             + " = z  ,   the encoder is asked to keep pixel content", 12.5,
             anchor="start")
    s += badge(868, 1182, M["patch"]["on_name"], K, size=11, anchor="end")
    s += sgbar(88, 1222, C, h=7.5, gap=3.2)
    s += txt(104, 1226, "open", 12.5, anchor="start", fill=ACCENT[C], weight="bold")
    s += txt(142, 1226, "DECODE_GRAD = false   " + ARROWC + "   " + sub("z", "dec", 11)
             + " = z.detach()  ,   the decoder still trains, the encoder never sees it",
             12.5, anchor="start", fill=INK)
    s += badge(868, 1214, M["patch"]["off_name"], C, size=11, anchor="end")

    # -- row B: the inversion ------------------------------------------------------
    s += frame(60, 1262, 816, 306, "the same boolean, the two representations  " + NDASH
               + "  the comparison INVERTS", "slate",
               note="same head, same " + LAMB + ", same seeds, same eval instrument on both "
                    "sides " + NDASH + " only the feature and the boolean differ")
    s += mark(88, 1322, K, r=5.5)
    s += txt(100, 1327, "DECODE_GRAD = true", 11.5, anchor="start", fill=ACCENT[K],
             weight="bold")
    s += mark(268, 1322, C, r=5.5)
    s += txt(280, 1327, "DECODE_GRAD = false  ( " + MINUS + "detach )", 11.5,
             anchor="start", fill=ACCENT[C], weight="bold")
    s += txt(864, 1327, "arm mean, final_eval/success_rate", 11.5, anchor="end",
             fill=MUTED)
    s += vrule(468, 1338, 1552, "slate")

    for hx, (feat, _, _) in zip(HX, PAIRS):
        d = M[feat]
        a0, a1 = hx + AX0, hx + AX1

        def xmap(v, a0=a0, a1=a1):
            return a0 + (v - lo) / (hi - lo) * (a1 - a0)

        s += txt(hx + 195, 1344, "feature = " + feat + "   " + DOT + "   "
                 + f"{TOKENS[feat]:,}".replace(",", " ") + (" token" if TOKENS[feat] == 1
                                                            else " tokens"),
                 14, weight="bold")
        # both latents at the same CELL size, so the picture says tokens and not capacity
        n_side = int(round(math.sqrt(TOKENS[feat])))
        side = n_side * 4.3 - 0.7 + 6
        s += tokengrid(hx + 66, YGLYPH + (P_TOK ** 0.5 * 4.3 - 0.7 + 6 - side) / 2,
                       TOKENS[feat], "slate", total=TOKENS[feat], cell=3.6, gap=0.7)
        # a leader from the glyph's own bottom edge down to the shared label baseline:
        # the cls glyph is 9.6 units tall against the patch glyph's 74.1, so on a shared
        # baseline its label sat 42 units clear of it and the token read as a stray mark.
        s += curve([(hx + 66, YGLYPH - 3 + (74.1 - side) / 2 + side + 2),
                    (hx + 66, 1441)], "slate", w=1.0, opacity=0.35)
        s += txt(hx + 66, 1452, f"{TOKENS[feat]} " + TIMES + " " + str(EMB), 11.5,
                 fill=MUTED)
        s += txt(hx + 66, 1468, "=  " + f"{DIMS[feat]:,}".replace(",", " ") + " dims", 12,
                 weight="bold")

        x_on, x_off = xmap(d["on"][0]), xmap(d["off"][0])
        s += axis(a0, a1, YAXIS, ticks, xmap)
        s += mark(round(x_on, 1), YMARK_ON, K, r=7)
        s += txt(round(x_on, 1), YMARK_ON - 14, f3(d["on"][0]), 15, fill=ACCENT[K],
                 weight="bold")
        s += mark(round(x_off, 1), YMARK_OFF, C, r=7)
        s += txt(round(x_off, 1), YMARK_OFF + 26, f3(d["off"][0]), 15, fill=ACCENT[C],
                 weight="bold")
        # ONE hue for both arrows, opposite directions.  The hue says "this is the
        # boolean"; the direction says its sign.  Colouring them by sign would rank them.
        step = 11.0 if x_on > x_off else -11.0
        s += aw(round(x_off + step, 1), YMARK_MID, round(x_on - step, 1), YMARK_MID, K,
                w=2.4)

        e = d["e"]
        s += txt(hx + 195, 1496, DELTA + " = " + signed(e["mean"]) + "   [ "
                 + signed(e["lo"]) + " ,  " + signed(e["hi"]) + " ]   n = " + str(e["n"]),
                 14, weight="bold")
        s += mark(hx + 74, 1514, "slate", r=6, fill=WHITE)
        s += txt(hx + 88, 1519, "NULL  " + NDASH + "  the interval spans zero", 12,
                 anchor="start", fill=ACCENT["slate"], weight="bold")
        # amber on BOTH halves.  This sentence is about the intervention, so it takes the
        # intervention's hue; painting the cls one purple because the control wins there
        # would be a hue assigned by rank, which is the one thing style.py forbids.
        s += txt(hx + 195, 1546,
                 "closing the switch moves the mean "
                 + ("UP" if e["mean"] > 0 else "DOWN"), 12.5,
                 fill=ACCENT[K], weight="bold")

    # -- the statement, and its limit ----------------------------------------------
    s += txt(470, 1594, "the SAME boolean, opposite signs:  "
             + signed(M["patch"]["e"]["mean"]) + " on a "
             + f"{DIMS['patch']:,}".replace(",", " ") + "-dim latent,  "
             + signed(M["cls"]["e"]["mean"]) + " on a " + str(DIMS["cls"]) + "-dim one",
             14, weight="bold")
    s += badge(PIN, 1606, "UNEXPLAINED", "slate", size=12)
    s += txt(486, 1622, "no mechanism is claimed: the four means and the sign flip are the "
             "whole statement, and neither contrast excludes zero", 12,
             fill=ACCENT["crit"])

    s += frow(1656, [
        (["loss  =  L" + sub("", "JEPA", 12) + "   +   " + LAMB + " " + DOT
          + " MSE( PatchHead( " + sub("z", "dec", 12) + " ) , o )"], INK, "bold"),
        ([sub("z", "dec", 12) + "  =  z   if DECODE_GRAD else   z.detach()"], INK,
         "normal")])
    s += frow(1680, [
        ("PatchHead is one nn.Linear:  " + str(EMB) + " " + ARROWC + " " + str(DEC_C) + " "
         + DOT + " " + str(DEC_P) + SQ2 + " = " + str(HEAD_OUT) + " ,  "
         + f"{HEAD_PARAMS:,}".replace(",", " ") + " params", INK, "normal"),
        ("num_patches = ( img_size / patch_size )" + SQ2 + " = " + str(P_TOK), MUTED,
         "normal", 0.92)])
    s += frow(1704, [
        ("of " + str(n_cmp) + " arms paired against " + BASELINE + " at n " + "≥"
         + " " + str(N_FLOOR) + " , " + str(n_pos) + " have a positive mean", INK,
         "normal"),
        (str(n_pos_dec) + " of those " + str(n_pos) + " are among the " + str(n_dec)
         + " AUX_DECODER arms ;  no other knob recurs", MUTED, "normal", 0.92)])
    s += frow(1728, [
        ("and its " + f3(M["patch"]["on"][0]) + " is the highest arm mean of the "
         + str(len(rank)) + " outside the consensus family", INK, "normal"),
        ("over arms sharing " + "≥" + " " + str(MIN_SHARED) + " evaluated seeds with "
         + BASELINE, MUTED, "normal", 0.9)])
    s += strip(PIN, 1746, [
        ("patch  " + f"{DIMS['patch']:,}".replace(",", " ") + " dims",
         signed(M["patch"]["e"]["mean"]) + " [" + signed(M["patch"]["e"]["lo"]) + ", "
         + signed(M["patch"]["e"]["hi"]) + "]", False),
        ("cls  " + str(DIMS["cls"]) + " dims",
         signed(M["cls"]["e"]["mean"]) + " [" + signed(M["cls"]["e"]["lo"]) + ", "
         + signed(M["cls"]["e"]["hi"]) + "]", False),
        ("both contrasts at", "n = " + str(M["patch"]["e"]["n"]), False),
        ("the two intervals", "both span zero", True)], K, cw=170)
    return b + s, PY + PH + 40


# --- the overlap audit -------------------------------------------------------------
_TXT = re.compile(r'<text ([^>]*)>(.*?)</text>', re.S)
_ATTR = re.compile(r'(\S+)="([^"]*)"')


def _boxes(svg):
    """Every <text> element's box, from the same metrics the layout used."""
    dy = 0.0
    m = re.search(r'<g transform="translate\(0,(-?[\d.]+)\)"', svg)
    if m:
        dy = float(m.group(1))
    out = []
    for mt in _TXT.finditer(svg):
        a = dict(_ATTR.findall(mt.group(1)))
        body = mt.group(2)
        size = float(a.get("font-size", 16))
        weight = a.get("font-weight", "normal")
        w = text_width(body, size, weight)
        x, y = float(a.get("x", 0)), float(a.get("y", 0)) + dy
        anchor = a.get("text-anchor", "start")
        x0 = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
        plain = re.sub(r"<[^>]*>", "", body)
        out.append((plain, x0, y - 0.72 * size, x0 + w, y + 0.21 * size))
    return out


def audit(path, w, h, frames=(), slack=1.0, pad=9.0):
    """Every pair of text boxes that intersect, anything that leaves the canvas, and
    anything that crosses the border of the frame it sits in.

    The SVG counterpart of analysis/round7_figs.py's audit(). `NOTHING MAY OVERLAP` is not
    checkable by eye on a 940 x 1740 canvas; this reads the emitted file back and says so.
    A label is assigned to the SMALLEST registered frame whose interior contains its centre,
    which is what makes the check work for an inset nested inside a panel.
    """
    with open(path) as f:
        svg = f.read()
    dy = SHIFT if 'transform="translate(0,' in svg else 0
    bx, bad = _boxes(svg), []
    for i in range(len(bx)):
        ti, xi0, yi0, xi1, yi1 = bx[i]
        if xi0 < 4 or xi1 > w - 4 or yi0 < 0 or yi1 > h:
            bad.append(f"OUTSIDE  {ti[:44]!r}  box ({xi0:.0f},{yi0:.0f})-"
                       f"({xi1:.0f},{yi1:.0f})  canvas {w}x{h}")
        cx, cy = (xi0 + xi1) / 2, (yi0 + yi1) / 2
        best = None
        for fx0, fy0, fx1, fy1, nm in frames:
            fy0d, fy1d = fy0 + dy, fy1 + dy
            if fx0 < cx < fx1 and fy0d < cy < fy1d:
                if best is None or (fx1 - fx0) * (fy1d - fy0d) < best[0]:
                    best = ((fx1 - fx0) * (fy1d - fy0d), fx0, fy0d, fx1, fy1d, nm)
        if best and (xi0 < best[1] + pad or xi1 > best[3] - pad
                     or yi0 < best[2] or yi1 > best[4]):
            bad.append(f"CROSSES  {ti[:40]!r}  leaves {best[5][:34]!r}  "
                       f"(text {xi0:.0f}-{xi1:.0f}, frame {best[1]:.0f}-{best[3]:.0f})")
        for j in range(i + 1, len(bx)):
            tj, xj0, yj0, xj1, yj1 = bx[j]
            ow = min(xi1, xj1) - max(xi0, xj0)
            oh = min(yi1, yj1) - max(yi0, yj0)
            if ow > slack and oh > slack:
                bad.append(f"OVERLAP  {ti[:36]!r}  x  {tj[:36]!r}   "
                           f"({ow:.0f}x{oh:.0f} units)")
    return bad


FIGURES = (
    ("arch-patchdecode.svg", patchdecode),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--png", action="store_true", help="also rasterise, for the read-back")
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
            png = path.replace(".svg", ".png")
            cairosvg.svg2png(url=path, write_to=png, output_width=W * 2)
            print("  wrote", png)
    return rc


if __name__ == "__main__":
    sys.exit(main())
