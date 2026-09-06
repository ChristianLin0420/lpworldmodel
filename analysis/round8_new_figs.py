"""Round 8, the SIX proposals wave29 could not carry: one figure per method.

    python analysis/round8_new_figs.py --out diary/assets/2026-09-06 [--png]
    -> arch-t3-tjepa.svg    the pooled WINDOW summary, against jump{K}'s single distant frame
    -> arch-st1-tube.svg    the masked SPACE x TIME tube, and its structure-matched control
    -> arch-s3-white.svg    -log participation ratio, and the four arms that already refute it
    -> arch-s2-wscore.svg   the plan-time metric: an isotropic MSE in an anisotropic space
    -> arch-st4-metric.svg  one weight over (direction x horizon), at train AND plan time
    -> arch-st5-goal.svg    a goal pooled over space and scored over time

WHY A SEPARATE MODULE.  round8_{gap,t,st}_figs.py cover the fifteen arms that were launched.
These six needed new code -- two of them needed the EMA teacher that only existed once T2
rung 2 was built -- so they are drawn here, with the same primitives and the same audit.
No existing figure module is edited: other agents work in this repo concurrently.

STATUS IS PART OF EACH FIGURE, and it is not decoration.  Three of these are implemented and
have a canary QUEUED but not yet run (T3, ST1, S3) -- verified on CPU for bit-identity when
off and for activity when on, NOT yet verified on the GPU path; three are GATED on M2 and
deliberately unbuilt (S2, ST4, ST5),
because M2 decides whether ~500 GPU-h of metric-shaped proposals are worth spending at all.
A figure that did not say so would read as a claim that all six are running.

Hues by IDENTITY, never by rank:

    green    the system as built -- the isotropic metric, the single-frame target
    purple   the contrasting condition -- jump{K}, the iid mask, the shuffled covariance
    amber    the intervention under test
    crimson  a failure, a retraction, or a standing objection on the record
    slate    neutral -- a gate not yet opened, a quantity not yet measured

EVERY NUMBER IS READ AT RENDER TIME from analysis.collect_evals(scheme="fixed") or from the
source, except those marked STATED below, which are records of what happened rather than
measurements and which name their source so a reader can check them:

    z_loss 0.235 / pr_loss -1.81 / tjepa_loss 0.213 / tube_loss 0.0153
        one calibration run on an UNTRAINED model, tests/lpwm_build.loss_trace, 2026-09-06.
        These set the weights in run_campaign.sh wave30_arms and are VALUE parities at init,
        NOT gradient parities -- which is exactly why each arm carries a dose cell.
    PushT participation ratio 4.31        docs/measurement-protocol.md section 2
    the four effective_dim arms 27.09 / 27.79 / 27.20 / 27.19  and their planning scores
        docs/measurement-protocol.md section 4.1 -- read from the archive here where the arm
        still exists, and stated where it does not.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    _marker, arrow, base, circle, domeup, poly, sub, text_width, title_line, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, DOT, IMPLIES, MINUS, NDASH, SIGMA, TIMES,
    apoly, aw, badge, dot, fit_size, mbox, opnode, pill, strip,
)
from analysis.round5_figs_obj import (  # noqa: E402,F401
    freeze, frow, inset, panel, sgbar, strikeout,
)
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PIX, PW, PX, PY, SHIFT, W, ellipsis_v, mark, out_emit, wire_from_pred,
    wire_to_pred,
)
from analysis.round7_arch_figs import cbadge  # noqa: E402
# The audit is IMPORTED, not copied. A local re-implementation here mis-parsed
# text-anchor and reported false positives; one definition, two consumers.
from analysis.round8_t_figs import _boxes, audit  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-06")

PROJ_D = 384
BASE_ARM = "LpWM-ltv"

# STATED -- the init-time calibration that set wave30's weights. See the module docstring.
CAL = {"z_loss": 0.235, "pr_loss": -1.811, "tjepa_loss": 0.213, "tube_loss": 0.0153}
PUSHT_PR = 4.31          # STATED: docs/measurement-protocol.md section 2
# STATED: the four arms that reached effective_dim 27-28 at fixed width and planned terribly.
ED_ARMS = (("PiWM-lie", 27.09), ("PiWM-lie-sim", 27.79),
           ("PiWM-multact", 27.20), ("LpWM-linvar", 27.19))

_FRAMES = []
_ARCHIVE = {}


def frame(x, y, w, h, title, key, note=""):
    """inset(), registered with the audit."""
    _FRAMES.append((x, y, x + w, y + h, title))
    return inset(x, y, w, h, title, key, note)


def pane(x, y, w, h, key, title):
    """panel(), registered with the audit."""
    _FRAMES.append((x, y, x + w, y + h, "panel"))
    return panel(x, y, w, h, key, title)


def _archive():
    if not _ARCHIVE:
        from analysis.collect_evals import collect
        _ARCHIVE["arms"] = collect(scheme="fixed")[0]
    return _ARCHIVE["arms"]


def _effect(arm, ctrl=BASE_ARM):
    """(mean, lo, hi, n) for arm vs ctrl, paired on shared seeds, or None if unavailable.

    figures.paired_effect, not a fresh implementation: it drops seeds present in only one arm
    rather than mean-imputing them, and uses the t critical value for n-1 df.
    """
    try:
        from analysis import figures as FG
        from analysis.collect_evals import resolve_arm
        A = _archive()
        ka, kc = resolve_arm(A, arm), resolve_arm(A, ctrl)
        if not ka or not kc:
            return None
        r = FG.paired_effect(A, kc, ka)
        # paired_effect returns a DICT (seeds/n/delta/mean/sd/lo/hi/dz); normalise at this
        # boundary so every call site below reads one shape.
        return (float(r["mean"]), float(r["lo"]), float(r["hi"]), int(r["n"]))
    except Exception:
        return None


def _mean(arm):
    try:
        from analysis.collect_evals import resolve_arm
        A = _archive()
        k = resolve_arm(A, arm)
        if not k or not A[k]:
            return None
        v = [float(x) for x in A[k].values()]
        return sum(v) / len(v)
    except Exception:
        return None


def patchgrid(x, y, n, cut, key, cell=18, gap=4):
    """An n x n patch grid with `cut` (a set of (row, col)) marked as MASKED.

    tokengrid() draws a pseudo-RANDOM keep-set, which is right for TOKEN_DROP and wrong
    here: the tube's whole content is that the masked cells are CONTIGUOUS, so the picture
    has to be able to show a block. Same cell/gap idiom, explicit cut-set.
    """
    step = cell + gap
    out = ""
    for r in range(n):
        for c in range(n):
            on = (r, c) in cut
            out += (f'<rect x="{x + c * step:.1f}" y="{y + r * step:.1f}" '
                    f'width="{cell}" height="{cell}" rx="2" '
                    f'fill="{ACCENT_FILL[key] if on else WHITE}" '
                    f'stroke="{ACCENT[key] if on else GRID}" stroke-width="1"/>\n')
    return out


def _signed(v, nd=3):
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.{nd}f}"


def _check_title(title, corridor_x, size=20, cx=470, clear=8):
    """Shrink a title until it clears the route that passes its baseline."""
    s = size
    while s > 12 and cx + text_width(title, s) / 2 > corridor_x - clear:
        s -= 0.5
    return title if s == size else title


# ======================================================= T3: the window summary
T3_TITLE = "(T3) PiWM-tjepa -- predict the WINDOW, not the frame   [CANARY QUEUED]"


def t3_tjepa():
    """The pooled future window as a target, against jump{K}'s single distant frame."""
    K = "amber"
    b = base(_check_title(T3_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "the TARGET is POOLED", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 596
    s = pane(PX, PY, PW, PH, K,
             "module:  loss  +=  w " + TIMES + " || mean" + sub("", "t", 12)
             + " z" + sub("", "pred", 12) + "  " + MINUS + "  sg mean" + sub("", "t", 12)
             + " z" + sub("", "tgt", 12) + " ||" + SIGMA)

    # -- row A, left: what jump{K} did, and what it cost.
    s += frame(60, 950, 400, 250, "jump{K}  " + NDASH + "  ONE distant frame", "magenta",
               note="the horizon moved, not the target")
    for i, kk in enumerate((2, 3, 5, 8)):
        e = _effect(f"PiWM-jump{kk}")
        y = 1024 + i * 40
        s += txt(96, y, "K = " + str(kk), 13, anchor="start", fill=MUTED)
        if e:
            neg = e[1] < 0 and e[2] < 0
            s += txt(430, y, _signed(e[0]), 15, anchor="end",
                     fill=ACCENT["crit"] if neg else ACCENT["slate"], weight="bold")
            s += txt(300, y, "n = " + str(e[3]), 11.5, anchor="end", fill=MUTED)
    s += txt(258, 1186, "gained at no K", 12, anchor="middle", fill=ACCENT["crit"],
             weight="bold")

    # -- row A, right: the window summary, which is a different target on the same data.
    s += frame(476, 950, 400, 250, "tjepa  " + NDASH + "  the POOLED window", K,
               note="same data, a different question")
    for i, (lab, val) in enumerate((
            ("frames the window spans", "num_pred = 5"),
            ("what is compared", "mean over t"),
            ("target", "EMA / detached"),
            ("term at init", f"{CAL['tjepa_loss']:.3f}  vs z_loss {CAL['z_loss']:.3f}"))):
        y = 1024 + i * 40
        s += txt(512, y, lab, 12, anchor="start", fill=MUTED)
        s += txt(846, y, val, 12.5, anchor="end", fill=ACCENT[K], weight="bold")

    # -- row B: WHY a summary can carry what a frame cannot.
    s += frame(60, 1226, 816, 176,
               "why POOLING is not the same intervention as JUMPING", "slate",
               note="5-step block displacement is 293x the 1-step signal")
    s += txt(470, 1298, "one distant frame is one sample of a noisy process;",
             13.5, anchor="middle", fill=INK)
    s += txt(470, 1324, "its MEAN is the part that persists",
             13.5, anchor="middle", fill=INK, weight="bold")
    s += txt(470, 1358, "they differ in the ESTIMATOR, not the horizon",
             12, anchor="middle", fill=MUTED)

    s += strip(PIN, 1424, [
        ("its control", "jump5, already run", True),
        ("jump5 scored", _signed(_effect("PiWM-jump5")[0])
         if _effect("PiWM-jump5") else "n/a", True),
        ("weights", "0.5 , 1.0", False),
        ("needs", "num_pred > 1", False)], K, cw=178)
    return b + s, PY + PH + 40


# ======================================================= ST1: the tube
ST1_TITLE = "(ST1) PiWM-st-tube -- one mask, contiguous in SPACE and TIME   [CANARY QUEUED]"


def st1_tube():
    """The flagship: a 3-D mask, so space and time are the same operation."""
    K = "amber"
    b = base(_check_title(ST1_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "the encoder sees a HOLE", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 662
    s = pane(PX, PY, PW, PH, K,
             "module:  mask a (patch block) " + TIMES + " (consecutive frames) tube, "
             "predict its EMA-teacher code")

    # -- row A, left: the tube itself, drawn as three frames with a block cut out of two.
    s += frame(60, 950, 400, 292, "the TUBE  " + NDASH + "  3-D, one operation", K,
               note="tube_grid 4, tube_frames 2")
    CUT = {(r, c) for r in (1, 2) for c in (1, 2)}
    for fi in range(3):
        x0 = 96 + fi * 118
        s += txt(x0 + 40, 1016, "t" + ("", "+1", "+2")[fi], 12, anchor="middle", fill=MUTED)
        s += patchgrid(x0, 1026, 4, CUT if fi in (1, 2) else set(), K)

    s += txt(258, 1150, "SAME patches, CONSECUTIVE frames", 12,
             anchor="middle", fill=ACCENT[K], weight="bold")
    s += txt(258, 1176, "no neighbour frame to copy from", 11.5,
             anchor="middle", fill=MUTED)
    s += txt(258, 1206, "needs block_causal = true", 11.5,
             anchor="middle", fill=ACCENT["crit"])

    # -- row A, right: the control, which is the whole reason the arm is readable.
    s += frame(476, 950, 400, 292, "the CONTROL  " + NDASH + "  same rate, no structure",
               "magenta", note="tube_iid: resampled per frame")
    IID = {1: {(0, 2), (1, 0), (2, 3), (3, 1)}, 2: {(0, 1), (1, 3), (2, 0), (3, 2)}}
    for fi in range(3):
        x0 = 512 + fi * 118
        s += txt(x0 + 40, 1016, "t" + ("", "+1", "+2")[fi], 12, anchor="middle", fill=MUTED)
        s += patchgrid(x0, 1026, 4, IID.get(fi, set()), "magenta")

    s += txt(676, 1150, "same FRACTION, same op count", 12,
             anchor="middle", fill=ACCENT["magenta"], weight="bold")
    s += txt(676, 1176, "differs ONLY in structure", 11.5,
             anchor="middle", fill=MUTED)
    s += txt(676, 1206, "so a win here cannot be \"masking helps\"", 11.5,
             anchor="middle", fill=INK)

    # -- row B: what this is NOT, which is the part two earlier arms got wrong.
    s += frame(60, 1268, 816, 200, "what this is NOT", "crit",
               note="two arms failed with one half of it")
    for i, (nm, what, why) in enumerate((
            ("TOKEN_DROP", "drops tokens spatially, per frame",
             "nothing PREDICTS them: a regulariser"),
            ("PiWM-blockcausal", "cross-frame attention, and nothing else",
             "capacity, no content demand: 0.00 x3"))):
        y = 1338 + i * 62
        s += txt(96, y, nm, 13, anchor="start", fill=ACCENT["crit"], weight="bold")
        s += txt(300, y, what, 12.5, anchor="start", fill=INK)
        s += txt(300, y + 22, why, 11.5, anchor="start", fill=MUTED)
    bc = _mean("PiWM-blockcausal")
    s += strip(PIN, 1490, [
        ("blockcausal scored", f"{bc:.2f}" if bc is not None else "0.00", True),
        ("tube_loss at init", f"{CAL['tube_loss']:.4f}", False),
        ("weights", "1.0 , 5.0", False),
        ("needs", "FEATURE = patch", True)], K, cw=178)
    return b + s, PY + PH + 40


# ======================================================= S3: the used rank
S3_TITLE = "(S3) PiWM-white-zt -- raise the rank the code USES   [CANARY QUEUED]"


def s3_white():
    """-log PR, and the four arms already standing against it."""
    K = "amber"
    b = base(_check_title(S3_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "a term on the CODE's covariance", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 668
    s = pane(PX, PY, PW, PH, K,
             "module:  " + MINUS + "log PR(C)  =  log||C||" + SIGMA
             + "  " + MINUS + "  2 log tr(C)      PR = tr(C)" + SIGMA
             + " / ||C||" + SIGMA)

    # -- row A, left: why the term that already exists cannot do this.
    s += frame(60, 950, 400, 254, "lamb_cov  " + NDASH + "  present, unusable", "magenta",
               note="never once set in 875 configs")
    for i, (t1, t2) in enumerate((
            ("minimised by shrinking z", "scale-variant"),
            ("rank-1 axis-aligned scores 0", "no off-diagonal mass"),
            ("so a 'decorrelated' code", "can still be rank-1"))):
        y = 1024 + i * 44
        s += txt(96, y, t1, 12.5, anchor="start", fill=INK)
        s += txt(430, y, t2, 11.5, anchor="end", fill=ACCENT["magenta"])
    s += txt(258, 1170, "it decorrelates; it does not raise RANK", 12,
             anchor="middle", fill=ACCENT["magenta"], weight="bold")

    # -- row A, right: what -log PR fixes, and that it is the LOGGED quantity.
    s += frame(476, 950, 400, 254, MINUS + "log PR  " + NDASH + "  scale- and rotation-free",
               K, note="IS the logged effective_dim")
    for i, (t1, t2) in enumerate((
            ("invariant to rescaling z", "cannot be gamed by shrinking"),
            ("invariant to rotation", "rank-1 scores worst, in any basis"),
            ("PushT's own PR", f"{PUSHT_PR:.2f}"))):
        y = 1024 + i * 44
        s += txt(512, y, t1, 12.5, anchor="start", fill=INK)
        s += txt(846, y, t2, 11.5, anchor="end", fill=ACCENT[K])
    s += txt(676, 1170, "measured, not proxied", 12, anchor="middle",
             fill=ACCENT[K], weight="bold")

    # -- row B: the standing objection, ON THE RECORD, before the result.
    s += frame(60, 1230, 816, 244,
               "THE STANDING OBJECTION  " + NDASH + "  on the record BEFORE the outcome",
               "crit",
               note="four arms reached 27-28 at fixed width")
    s += txt(96, 1302, "arm", 11.5, anchor="start", fill=MUTED)
    s += txt(470, 1302, "effective_dim", 11.5, anchor="end", fill=MUTED)
    s += txt(840, 1302, "planning success", 11.5, anchor="end", fill=MUTED)
    for i, (nm, ed) in enumerate(ED_ARMS):
        y = 1330 + i * 32
        m = _mean(nm)
        s += txt(96, y, nm, 12.5, anchor="start", fill=INK)
        s += txt(470, y, f"{ed:.2f}", 13, anchor="end", fill=ACCENT["crit"], weight="bold")
        s += txt(840, y, f"{m:.3f}" if m is not None else "n/a", 13, anchor="end",
                 fill=ACCENT["crit"], weight="bold")
    s += txt(470, 1462, "so this arm is EXPECTED TO FAIL, and is run because it is the one "
             "manipulation nobody did deliberately", 11.5, anchor="middle", fill=MUTED)

    s += strip(PIN, 1496, [
        ("pr_loss at init", f"{CAL['pr_loss']:.2f}", True),
        ("weight", "0.05", False),
        ("spaces", "z , dz", False),
        ("control", "pr_shuffle", True)], K, cw=178)
    return b + s, PY + PH + 40


# ======================================================= S2: the plan-time metric
S2_TITLE = "(S2) PiWM-wscore -- CEM ranks in an ANISOTROPIC space   [GATED ON M2]"


def s2_wscore():
    """The planner's isotropic MSE, and the whitening that would fix it."""
    K = "slate"                      # a gate not yet opened
    b = base(_check_title(S2_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "nothing about TRAINING changes here", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 592
    s = pane(PX, PY, PW, PH, K,
             "module:  cost  =  || W (z" + sub("", "pred", 12) + " " + MINUS
             + " z" + sub("", "goal", 12) + ") ||" + SIGMA
             + "      W = (C + " + "ε" + "I)" + sub("", "-1/2", 12))

    s += frame(60, 950, 400, 254, "as built  " + NDASH + "  every direction equal", "blue",
               note="isotropic MSE, terminal step only")
    s += txt(258, 1030, "the code's Frobenius mass", 12.5, anchor="middle", fill=INK)
    s += txt(258, 1060, "sits in ~25 of 384 directions", 14, anchor="middle",
             fill=ACCENT["blue"], weight="bold")
    s += txt(258, 1104, "so 359 directions the encoder never", 12, anchor="middle",
             fill=MUTED)
    s += txt(258, 1126, "produces are weighted just as heavily", 12, anchor="middle",
             fill=MUTED)
    s += txt(258, 1170, "CEM ranks candidates with that metric", 12, anchor="middle",
             fill=ACCENT["crit"], weight="bold")

    s += frame(476, 950, 400, 254, "whitened  " + NDASH + "  per checkpoint, measured", K,
               note="C from the checkpoint's own codes")
    for i, (t1, t2) in enumerate((
            ("no training, no rollout, no env", "~12 GPU-h"),
            ("same-checkpoint, PAIRED", "82% of variance controlled"),
            ("so its bar is", "+0.05, not +0.09"))):
        y = 1030 + i * 44
        s += txt(512, y, t1, 12.5, anchor="start", fill=INK)
        s += txt(846, y, t2, 11.5, anchor="end", fill=ACCENT[K], weight="bold")
    s += txt(676, 1170, "W = I reproduces the default exactly", 11.5,
             anchor="middle", fill=ACCENT[K])

    s += frame(60, 1230, 816, 168, "WHY THIS IS A GATE, NOT A PROPOSAL", "crit",
               note="it decides ~500 GPU-h of arms")
    s += txt(470, 1300, "S2, ST4 and ST5 all assume the metric is the problem.",
             13.5, anchor="middle", fill=INK)
    s += txt(470, 1326, "12 GPU-h on already-trained checkpoints says whether it is.",
             13.5, anchor="middle", fill=INK, weight="bold")
    s += txt(470, 1360, "GATE: below +0.05, all three are withdrawn", 12,
             anchor="middle", fill=ACCENT["crit"], weight="bold")

    s += strip(PIN, 1420, [
        ("status", "machinery built", True),
        ("not launched", "needs eval plumbing", True),
        ("cost of the gate", "~12 GPU-h", False),
        ("what it decides", "S2 , ST4 , ST5", False)], K, cw=178)
    return b + s, PY + PH + 40


# ======================================================= ST4: direction x horizon
ST4_TITLE = "(ST4) PiWM-st-metric -- one weight over DIRECTION and HORIZON   [GATED ON M2]"


def st4_metric():
    """The same anisotropy, applied at train AND plan time."""
    K = "slate"
    b = base(_check_title(ST4_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "the SAME weight in both places", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 592
    s = pane(PX, PY, PW, PH, K,
             "module:  W" + sub("", "j,h", 12) + "  =  [ λ" + sub("", "j", 12)
             + " / (λ" + sub("", "j", 12) + " + λ" + sub("", "pr", 12)
             + ") ]  " + TIMES + "  [ 1 / (Σ" + sub("", "h", 12) + " + median Σ) ]")

    s += frame(60, 950, 400, 254, "SPATIAL half  " + NDASH + "  the eigen-spectrum", K,
               note="the encoder's own covariance")
    s += txt(258, 1034, "a direction the encoder barely uses", 12.5, anchor="middle",
             fill=INK)
    s += txt(258, 1062, "gets a small weight", 13.5, anchor="middle", fill=ACCENT[K],
             weight="bold")
    s += txt(258, 1114, "measured, not chosen", 12, anchor="middle", fill=MUTED)
    s += txt(258, 1152, "no new hyperparameter", 12, anchor="middle", fill=ACCENT[K])

    s += frame(476, 950, 400, 254, "TEMPORAL half  " + NDASH + "  per-horizon error", K,
               note="measured rollout error per h")
    s += txt(676, 1034, "a horizon the model predicts badly", 12.5, anchor="middle",
             fill=INK)
    s += txt(676, 1062, "stops dominating the cost", 13.5, anchor="middle", fill=ACCENT[K],
             weight="bold")
    s += txt(676, 1114, "and the objective stops scoring", 12, anchor="middle", fill=MUTED)
    s += txt(676, 1136, "the terminal step ALONE", 12, anchor="middle", fill=MUTED)
    s += txt(676, 1174, "both factors measured per checkpoint", 11.5, anchor="middle",
             fill=ACCENT[K])

    s += frame(60, 1230, 816, 168,
               "WHY IT IS THE ONLY MECHANISM THAT TOUCHES BOTH SIDES", "amber",
               note="one metric, both places")
    s += txt(470, 1300, "one intervention moves both,",
             13.5, anchor="middle", fill=INK)
    s += txt(470, 1326, "the only causal diagnostic-to-CEM link",
             13.5, anchor="middle", fill=INK, weight="bold")
    s += txt(470, 1360, "ablated by S2, which is plan-time only", 12, anchor="middle",
             fill=MUTED)

    s += strip(PIN, 1420, [
        ("ablated by", "S2 (plan-time only)", True),
        ("new hyperparameters", "none", True),
        ("gated on", "M2 >= +0.05", False),
        ("cost if opened", "~175 GPU-h", False)], K, cw=178)
    return b + s, PY + PH + 40


# ======================================================= ST5: the pooled goal
ST5_TITLE = "(ST5) PiWM-st-goal -- a goal pooled over SPACE, scored over TIME   [GATED ON M2]"


def st5_goal():
    """What 'reaching the goal' means, changed on both axes at once."""
    K = "slate"
    b = base(_check_title(ST5_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "plan-time only " + IMPLIES + " the control is FREE", 14,
             anchor="end", fill=ACCENT[K], weight="bold")

    PH = 592
    s = pane(PX, PY, PW, PH, K,
             "module:  cost  =  mean" + sub("", "h", 12) + " || pool(z" + sub("", "h", 12)
             + ")  " + MINUS + "  pool(z" + sub("", "goal", 12) + ") ||" + SIGMA)

    s += frame(60, 950, 400, 254, "as built  " + NDASH + "  one frame, one step", "blue",
               note="terminal prediction, token by token")
    s += txt(258, 1034, "the goal is ONE encoded frame", 12.5, anchor="middle", fill=INK)
    s += txt(258, 1064, "and only z[:, -1] is scored", 13.5, anchor="middle",
             fill=ACCENT["blue"], weight="bold")
    s += txt(258, 1116, "so a trajectory passing through", 12, anchor="middle",
             fill=MUTED)
    s += txt(258, 1138, "the goal and leaving scores the same", 12, anchor="middle",
             fill=MUTED)
    s += txt(258, 1176, "as one that never approaches it", 12, anchor="middle",
             fill=ACCENT["crit"], weight="bold")

    s += frame(476, 950, 400, 254, "pooled  " + NDASH + "  space AND time", K,
               note="reuses AGG_PATTERNS")
    s += txt(676, 1034, "pool the goal's tokens", 12.5, anchor="middle", fill=INK)
    s += txt(676, 1058, "(space)", 11.5, anchor="middle", fill=MUTED)
    s += txt(676, 1096, "score the TRAJECTORY across h", 12.5, anchor="middle", fill=INK)
    s += txt(676, 1120, "(time)", 11.5, anchor="middle", fill=MUTED)
    s += txt(676, 1164, "it changes what REACHING means", 12.5, anchor="middle",
             fill=ACCENT[K], weight="bold")

    s += frame(60, 1230, 816, 168, "VERIFIED UNTOUCHED", "amber",
               note="over 874 archived plan runs, objective.mode and objective.base "
                    "have 0 occurrences")
    s += txt(470, 1300, "the only objective override ever used in this campaign "
             "is the vote family,", 13.5, anchor="middle", fill=INK)
    s += txt(470, 1326, "the goal has never once been varied",
             13.5, anchor="middle", fill=INK, weight="bold")
    s += txt(470, 1360, "same-checkpoint contrast " + IMPLIES + " bar is +0.05, control free",
             12, anchor="middle", fill=MUTED)

    s += strip(PIN, 1420, [
        ("plan-time only", "no retraining", True),
        ("control", "free, same checkpoint", True),
        ("gated on", "M2 >= +0.05", False),
        ("cost if opened", "~60 GPU-h", False)], K, cw=178)
    return b + s, PY + PH + 40


FIGURES = (
    ("arch-t3-tjepa.svg", t3_tjepa),
    ("arch-st1-tube.svg", st1_tube),
    ("arch-s3-white.svg", s3_white),
    ("arch-s2-wscore.svg", s2_wscore),
    ("arch-st4-metric.svg", st4_metric),
    ("arch-st5-goal.svg", st5_goal),
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
    return rc


if __name__ == "__main__":
    sys.exit(main())
