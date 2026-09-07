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

STATUS IS PART OF EACH FIGURE, and it is not decoration.  ALL SIX are now implemented and
launched at 8 seeds.  T3, ST1 and S3 passed a GPU canary; S2 and ST5 are plan-time and run
against existing checkpoints; ST4 trains.  The M2 gate that once held S2/ST4/ST5 back was
overridden deliberately -- the campaign wanted every method measured rather than three of
them argued about,
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

import numpy as np

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


# --------------------------------------------------------------- ST4's drawn evidence
# The code's variance spectrum, recovered at render time from the saved whitening matrix:
# W = (C + eps I)^(-1/2), so the eigenvalues of C are the eigenvalues of W raised to -2.
# Nothing here is typed; if the matrices change the picture changes.
def _spectrum(run="LpWM-ltv_pd384_bf16_s3", nb=26):
    import torch
    p = os.path.join(REPO, "assets", "wscore", run + ".pt")
    if not os.path.exists(p):
        return None
    W = torch.load(p, map_location="cpu")["W"].double()
    ev = torch.linalg.eigvalsh(W).clamp_min(1e-12)
    lam = (ev ** -2).numpy()
    lam = np.sort(lam)[::-1]
    idx = np.unique(np.round(np.logspace(0, np.log10(len(lam) - 1), nb)).astype(int))
    raw = lam[idx] / lam[0]
    w = 1.0 / np.sqrt(lam + 1e-3 * lam.mean())
    wl = lam * w * w
    return raw, (wl[idx] / wl[0]), float(lam[0] / np.median(lam)), float(wl[0] / np.median(wl))


def bars(x, y, w, h, vals, key, base_key="slate"):
    """A bar per value, heights normalised to the first. One bar IS one latent direction."""
    n = len(vals)
    bw = (w - (n - 1) * 2.0) / n
    s = (f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="{h + 8}" rx="3" '
         f'fill="{WHITE}" stroke="{DIM}" stroke-width="1"/>\n')
    for i, v in enumerate(vals):
        bh = max(1.0, float(v) * h)
        s += (f'<rect x="{x + i * (bw + 2.0):.1f}" y="{y + h - bh:.1f}" '
              f'width="{bw:.1f}" height="{bh:.1f}" rx="1" fill="{ACCENT[key]}" '
              f'opacity="0.85"/>\n')
    return s


def dotrow(x, y, w, h, deltas, key):
    """Per-seed deltas on a zero line: above = the arm won that seed, below = it lost."""
    s = (f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="{h + 8}" rx="3" '
         f'fill="{WHITE}" stroke="{DIM}" stroke-width="1"/>\n')
    mx = max(0.42, max(abs(float(d)) for d in deltas) * 1.15)
    zy = y + h / 2.0
    s += (f'<line x1="{x}" y1="{zy:.1f}" x2="{x + w}" y2="{zy:.1f}" '
          f'stroke="{GRID}" stroke-width="1.2"/>\n')
    n = len(deltas)
    for i, d in enumerate(deltas):
        cx = x + (i + 0.5) * (w / n)
        cy = zy - (float(d) / mx) * (h / 2.0 - 6)
        col = (ACCENT[key] if float(d) > 0
               else (ACCENT["slate"] if abs(float(d)) < 1e-9 else ACCENT["crit"]))
        s += (f'<line x1="{cx:.1f}" y1="{zy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
              f'stroke="{col}" stroke-width="1.6" opacity="0.55"/>\n')
        s += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{ACCENT_FILL[key] if float(d)>0 else WHITE}" '
              f'stroke="{col}" stroke-width="2"/>\n')
    return s


def civ(x, y, w, mean, lo, hi, key, lim=0.30, bar=0.09):
    """One effect as an interval, drawn against zero and against the +0.09 noise bar."""
    s = (f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="34" rx="3" '
         f'fill="{WHITE}" stroke="{DIM}" stroke-width="1"/>\n')
    def px(v):
        return x + (float(v) + lim) / (2 * lim) * w
    cy = y + 13
    s += (f'<line x1="{px(0):.1f}" y1="{y - 2}" x2="{px(0):.1f}" y2="{y + 28}" '
          f'stroke="{GRID}" stroke-width="1.4"/>\n')
    s += (f'<line x1="{px(bar):.1f}" y1="{y - 2}" x2="{px(bar):.1f}" y2="{y + 28}" '
          f'stroke="{ACCENT["crit"]}" stroke-width="1.2" stroke-dasharray="3,3" '
          f'opacity="0.8"/>\n')
    s += (f'<line x1="{px(lo):.1f}" y1="{cy:.1f}" x2="{px(hi):.1f}" y2="{cy:.1f}" '
          f'stroke="{ACCENT[key]}" stroke-width="3" opacity="0.6"/>\n')
    for v in (lo, hi):
        s += (f'<line x1="{px(v):.1f}" y1="{cy - 6:.1f}" x2="{px(v):.1f}" '
              f'y2="{cy + 6:.1f}" stroke="{ACCENT[key]}" stroke-width="2"/>\n')
    s += (f'<circle cx="{px(mean):.1f}" cy="{cy:.1f}" r="6" fill="{ACCENT[key]}" '
          f'stroke="{WHITE}" stroke-width="1.5"/>\n')
    return s


def _deltas(arm, ctrl="LpWM-ltv"):
    try:
        from analysis.collect_evals import resolve_arm as _ra
        A = _archive()
        v, c = A[_ra(A, arm)], A[_ra(A, ctrl)]
    except Exception:
        return [], []
    sh = sorted(set(v) & set(c), key=int)
    return sh, [float(v[s]) - float(c[s]) for s in sh]


def _signed(v, nd=3):
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.{nd}f}"


def _check_title(title, corridor_x, size=20, cx=470, clear=8):
    """Shrink a title until it clears the route that passes its baseline."""
    s = size
    while s > 12 and cx + text_width(title, s) / 2 > corridor_x - clear:
        s -= 0.5
    return title if s == size else title


# ======================================================= T3: the window summary
T3_TITLE = "(T3) PiWM-tjepa -- predict the WINDOW, not the frame   [LAUNCHED]"


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
ST1_TITLE = "(ST1) PiWM-st-tube -- one mask, contiguous in SPACE and TIME   [LAUNCHED]"


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
S3_TITLE = "(S3) PiWM-white-zt -- raise the rank the code USES   [LAUNCHED]"


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
S2_TITLE = "(S2) PiWM-wscore -- CEM ranks in an ANISOTROPIC space   [LAUNCHED]"


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
ST4_TITLE = "(ST4) PiWM-st-metric -- weight the residual by the code's own anisotropy"


def st4_metric():
    """Drawn evidence: the spectrum it acts on, the per-seed deltas, the two intervals."""
    K = "amber"
    b = base(_check_title(ST4_TITLE, 890))
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "a WEIGHT on the prediction residual", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 848
    s = pane(PX, PY, PW, PH, K,
             "built:  w = 1 / sqrt(var(z) + eps)  , detached , normalised to mean 1")

    # ---- row A: the thing it acts on, measured, drawn twice -------------------------
    sp = _spectrum()
    s += frame(60, 950, 816, 258, "WHAT IT ACTS ON  " + NDASH
               + "  the code's variance spectrum, measured", K,
               note="one bar per latent direction, log-spaced, normalised to the largest")
    if sp:
        raw, wtd, r_ratio, w_ratio = sp
        s += txt(258, 1026, "AS TRAINED", 12.5, anchor="middle", fill=ACCENT["magenta"],
                 weight="bold")
        s += bars(96, 1040, 324, 92, raw, "magenta")
        s += txt(258, 1158, "max / median  =  " + f"{r_ratio:,.0f}" + TIMES, 13,
                 anchor="middle", fill=ACCENT["magenta"], weight="bold")
        s += txt(676, 1026, "AFTER THE WEIGHT", 12.5, anchor="middle", fill=ACCENT[K],
                 weight="bold")
        s += bars(514, 1040, 324, 92, wtd, K)
        s += txt(676, 1158, "max / median  =  " + f"{w_ratio:.1f}" + TIMES, 13,
                 anchor="middle", fill=ACCENT[K], weight="bold")
        s += aw(438, 1086, 496, 1086, K)
        s += txt(467, 1188, "top 10 of 384 directions carry 48.5% of the variance",
                 11.5, anchor="middle", fill=MUTED)

    # ---- row B: the per-seed evidence, drawn ----------------------------------------
    s += frame(60, 1232, 400, 250, "PER SEED  " + NDASH + "  above the line = a win", K,
               note="metric_w = 0.5 (top) and 1.0 (bottom)")
    sh05, d05 = _deltas("PiWM-st-metric")
    sh10, d10 = _deltas("PiWM-st-metric-w1")
    if d05:
        s += dotrow(96, 1300, 328, 62, d05, K)
        s += txt(258, 1382, f"w = 0.5   {sum(1 for x in d05 if x>0)} of {len(d05)} seeds won",
                 12, anchor="middle", fill=INK)
    if d10:
        s += dotrow(96, 1396, 328, 62, d10, K)
        s += txt(258, 1470, f"w = 1.0   {sum(1 for x in d10 if x>0)} of {len(d10)} seeds won",
                 12, anchor="middle", fill=INK)

    # ---- row B right: the two intervals against the bar -----------------------------
    s += frame(476, 1232, 400, 250, "THE TWO DOSES  " + NDASH + "  against the +0.09 bar",
               K, note="dashed line = what an 8-seed arm needs")
    e05, e10 = _effect("PiWM-st-metric"), _effect("PiWM-st-metric-w1")
    for i, (lab, e) in enumerate((("w = 0.5", e05), ("w = 1.0", e10))):
        y = 1306 + i * 76
        s += txt(512, y - 8, lab, 12, anchor="start", fill=MUTED)
        if e:
            s += civ(512, y + 4, 330, e[0], e[1], e[2], K)
            s += txt(677, y + 56, _signed(e[0]) + "   n = " + str(e[3]), 12.5,
                     anchor="middle", fill=ACCENT[K], weight="bold")
    s += txt(676, 1462, "both intervals cross zero", 12, anchor="middle",
             fill=ACCENT["crit"], weight="bold")

    # ---- row C: the two things that are NOT drawn, because they did not move --------
    s += frame(60, 1506, 816, 148, "NO MECHANISM SIGNATURE", "crit",
               note="the intervention does not move the diagnostics it should")
    for i, (nm, a, bl) in enumerate((("rel_mse (the one axis that survives)", "0.0086", "0.0092"),
                                     ("effective_dim", "24.16", "24.10"))):
        y = 1576 + i * 30
        s += txt(96, y, nm, 12.5, anchor="start", fill=INK)
        s += txt(700, y, "arm " + a, 12.5, anchor="end", fill=ACCENT["crit"])
        s += txt(846, y, "base " + bl, 12.5, anchor="end", fill=MUTED)

    s2 = _effect("PiWM-wscore")
    s += strip(PIN, 1676, [
        ("best dose", _signed(e10[0]) if e10 else "n/a", True),
        ("bar at n = 8", "+0.09", True),
        ("its plan-time half", _signed(s2[0]) if s2 else "n/a", False),
        ("verdict", "unresolved", False)], K, cw=178)
    return b + s, PY + PH + 40


# ======================================================= ST5: the pooled goal
ST5_TITLE = "(ST5) PiWM-st-goal -- a goal pooled over SPACE, scored over TIME   [LAUNCHED]"


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
