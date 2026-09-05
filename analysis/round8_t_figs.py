"""Round 8, the TEMPORAL group: one figure per T mechanism.

    python analysis/round8_t_figs.py --out diary/assets/2026-09-06 [--png]
    -> arch-t1-ssm.svg      the recurrent state that replaces the FIR tap sum
    -> arch-t2-target.svg   the three target regimes as one switch
    -> arch-t4-lag.svg      masking and skip on the lag axis, and the dilation that cannot exist
    -> arch-t5-hist.svg     the num_hist ladder, against jump{K}'s target horizon

Seven rounds moved the objective, the planner, the predictor FAMILY, the link, the
representation and the width, and never once moved the temporal axis: `num_hist` is 3 in all
400 sampled run configs, every predictor mode is FIR, `detach_target: False` everywhere, and
no state-space predictor existed at all.  These four figures are the four temporal mechanisms
round 8 adds, drawn the way every other architecture figure in this project is drawn.

    T1  arch-t1-ssm.svg     FIR window vs IIR state, the deliberate init, and the 384-parameter
                            control.  AMBER: the intervention.  PURPLE: mlpvar, the contrast.
    T2  arch-t2-target.svg  LIVE / STOP-GRAD / EMA TEACHER as three columns of one switch, and
                            the teacher's plumbing.  GREEN: the system as built (LIVE, what
                            ~160 runs did).  PURPLE: the ctor default.  AMBER: the teacher.
    T4  arch-t4-lag.svg     the three lag slots, the skip that is expressible, and in CRIMSON
                            the dilation that is not -- 45 jobs died proving it.
    T5  arch-t5-hist.svg    the num_hist ladder as the FIR ORDER, with H = 1 as a plumbing
                            test, against CRIMSON jump{K}, which moved the TARGET horizon.

STYLE.  analysis/style.py, through analysis/arch_figs.py.  Every primitive is imported from
analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py / round6_arch_figs.py /
round7_arch_figs.py -- nothing here re-implements one, and no existing file is edited (other
agents are working in this repo at the same time).  The import set is
analysis/round7_v2_arch_figs.py's, plus four names taken from the SAME modules because the
house already owns those marks and re-drawing them would fork them: `circle` and `opnode` (the
action node and the summing node of T1's recurrence), `sgbar` (T2's stop-gradient bars) and
`freeze` (T2's frozen teacher).

Hues by IDENTITY, never by rank:

    green    the system as built -- LIVE targets, H = 3, the uniform lag set, LpWM-ltv
    purple   the contrasting condition -- mlpvar's FIR core, the stop-grad ctor default
    amber    the intervention under test -- ssm, the EMA teacher, lag mask/skip, the ladder
    crimson  a failure or a retraction -- the dead [0,2,3] dilation and its 45 jobs, the
             jump{K} horizon that lost at every K, the target's own gradient loop
    slate    neutral -- a frame outside the window, the H = 1 plumbing identity
    teal     reserved here for nothing; no consensus arm appears in these four figures

LAYOUT RULES INHERITED FROM round6_arch_figs.py / round7_v2_arch_figs.py

  * Strict left-to-right dataflow on fixed row baselines; a signal that legitimately fans out
    gets a drawn dot().  The one loop drawn here -- T1's state feedback -- returns on its own
    private corridor, which is what makes it read as a loop.
  * base() fixes every anchor, so a connector uses ONE named route.  T1, T4 and T5 all change
    the PREDICTOR, so all three reuse round6's `wire_to_pred` verbatim: the reader learns that
    route once and it means "this proposal changes the predictor".  T2 changes what the RIGHT
    ENCODER's output is used for, where none of the house routes lands, so it reuses round7
    S1 / round7-v2 G1's encoder route instead: up the empty right margin at x = 890 and into
    the dome's right edge, with the knob label at x = 712 (not further right -- base()'s
    RDMReg return runs vertically at x = 726).
  * Every route passes the title's baseline at y = 872, so every title is WIDTH-CHECKED
    against its corridor (`_check_title`) rather than trusted.
  * Nothing is sized by hand where it can be measured: mbox / pill / inset / frow / strip all
    fit their own text and none may cross the panel border.
  * A footer line carries at most two clauses, and an inset title is short: `frow()` shrinks
    its gap and then its type, and `inset()` shrinks its title, but both stop at a floor and
    then OVERFLOW SILENTLY.  T5's jump{K} title did exactly that (382 units into a 360-unit
    box) and the audit is what said so; the verdict moved into the inset's body instead.
  * A NULL is never drawn like a negative.  Two of the four jump{K} intervals span zero, so
    those two get slate, a HOLLOW marker and the word NULL -- the encoding round7-design.svg
    uses -- and the figure claims "gained at no K" rather than "lost at every K".
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of every
    <text> element from the same Helvetica metrics the layout used (arch_figs.text_width), and
    reports any pair that intersects, anything that leaves the canvas, and anything that
    crosses the border of the frame it sits in (every frame registers itself through
    `frame()` / `pane()`).  It is the SVG counterpart of analysis/round7_figs.py's audit(),
    and it exists for the same reason: at 940 x 1750 units "nothing overlaps" is not
    checkable by eye.

EVERY NUMBER IS READ OR COMPUTED AT RENDER TIME.  Nothing numeric below is typed into the
figure body; the module measures it and the figure draws what the measurement said.

  computed in-module, by building the real objects (`_params`, `_h1_identity`, `_guard_ok`)
    738 048 / 738 432 / 811 840 / 590 208 / 295 296     LinearDynamicsPredictor, D = 384,
      and the whole num_hist ladder 480 032 ... 1 641 360   H = num_hist, num_patches = 1
    var(H=1) == additive to the last bit on matched weights, and 295 296 params each
    the T4 guard: lag_dilation [0,2,3] rejected at construction, [0,2,2] accepted
  read from the config / the ctor at render time (`_conf_scalar`, `_ctor_default`)
    num_hist = 3 and detach_target = False        conf/train_rdmreg.yaml
    the detach_target ctor default = True         models/visual_world_model.py
  read from the ARCHIVE at render time (`analysis.collect_evals.collect(scheme="fixed")`)
    jump2 / jump3 / jump5 / jump8 against LpWM-ltv, paired on shared seeds, with CIs
    the arm and run counts the "every run trained with a live target" claim ranges over

  STATED, NOT COMPUTED -- these are records of what happened, not measurements, and each one
  names its source so a reader can check it:
    45 jobs lost to the DDP unused-parameter error   scripts/run_campaign.sh wave29_arms,
                                                     diary/2026-09-06.md section 3
    "400 sampled run configs", "~160 runs"           diary/2026-09-06.md sections 0 and 2
    the two EMA momenta 0.99 / 0.999                 scripts/run_campaign.sh wave29_arms
    the two mask doses 0.25 / 0.50                   scripts/run_campaign.sh wave29_arms
    the block is static in 48.1% of single steps     2026-09-04 / round6 R1's own strip
"""
import argparse
import os
import re
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-06")

# The predictor's own shape, as train.py builds it on the adaln path (train.py:825-834):
# input_dim = output_dim = hidden_dim = encoder.emb_dim, num_frames = cfg.num_hist,
# num_patches = encoder.num_patches (1 for the cls feature every T arm uses).
PROJ_D = 384
LADDER = (1, 2, 3, 5, 8)          # the num_hist rungs: 3 is what exists, the rest are new
LAG_SETS = ((0, 1, 2), (0, 2, 2), (0, 2, 3))     # uniform / the skip / the dead dilation
JUMP_K = (2, 3, 5, 8)
CTRL_ARM = "LpWM-ltv"


# --- the numbers -------------------------------------------------------------------
def _fmt(n):
    """A thin space between thousands: 738 432 rather than 738,432 or 738432."""
    return f"{n:,}".replace(",", " ")


_PARAMS = {}


def _params(mode, H=None, D=PROJ_D):
    """The parameter count of a real LinearDynamicsPredictor, built here.

    The T1 control claim is `mlpvar 738 048 against ssm 738 432`, and a claim of that shape
    is only worth making if the two numbers come out of the constructor rather than out of a
    note.  Cached because the same mode is asked for by two figures.
    """
    H = _num_hist() if H is None else H
    key = (mode, H, D)
    if key not in _PARAMS:
        from models.infojepa_modules import LinearDynamicsPredictor
        p = LinearDynamicsPredictor(input_dim=D, output_dim=D, hidden_dim=D,
                                    num_frames=H, num_patches=1, mode=mode)
        _PARAMS[key] = sum(q.numel() for q in p.parameters())
    return _PARAMS[key]


def _h1_identity(D=8):
    """`var` at H = 1 IS `additive` -- the docstring's identity, exercised.

    Returns the largest absolute difference between the two modes' outputs on matched
    weights.  T5's H = 1 rung is claimed to be a plumbing test, and the only honest way to
    draw that claim is to run it: a non-zero here would mean the knob is not threaded.
    Deliberately at D = 8, because this checks an ALGEBRAIC identity and a small D makes the
    check instant; the parameter equality is checked separately at the real D = 384.
    """
    import torch
    from models.infojepa_modules import LinearDynamicsPredictor as LDP
    torch.manual_seed(0)
    kw = dict(input_dim=D, output_dim=D, hidden_dim=D, num_patches=1, num_frames=1)
    a = LDP(mode="additive", **kw)
    v = LDP(mode="var", **kw)
    with torch.no_grad():
        a.W.weight.normal_(); a.W.bias.normal_(); a.B.weight.normal_()
        v.lags[0].weight.copy_(a.W.weight)
        v.lags[0].bias.copy_(a.W.bias)
        v.B.weight.copy_(a.B.weight)
        x, c = torch.randn(2, 1, 1, D), torch.randn(2, 1, D)
        return float((a(x, c) - v(x, c)).abs().max())


def _guard_ok(offsets, H=None, D=32):
    """Does the construction-time guard ACCEPT this lag_dilation?

    T4's whole point is that [0,2,3] is not expressible, and the guard that says so was added
    after 45 jobs died without it.  The figure draws the guard's verdict, not a memory of it.
    """
    from models.infojepa_modules import LinearDynamicsPredictor
    try:
        LinearDynamicsPredictor(input_dim=D, output_dim=D, hidden_dim=D,
                                num_frames=_num_hist() if H is None else H,
                                num_patches=1, mode="ltv", lag_dilation=list(offsets))
        return True
    except AssertionError:
        return False


_SCALAR = {}


def _conf_scalar(key, conf="train_rdmreg.yaml"):
    """A top-level scalar from a hydra config, read from the file.

    num_hist and detach_target are the two knobs this whole group is about, so they are read
    from the config every campaign run actually loads (scripts/train.sh:156 passes
    --config-name train_rdmreg.yaml) rather than restated here.
    """
    if (conf, key) not in _SCALAR:
        path = os.path.join(REPO, "conf", conf)
        val = None
        with open(path) as f:
            for line in f:
                if line.startswith(key + ":"):
                    val = line.split(":", 1)[1].split("#")[0].strip()
                    break
        _SCALAR[(conf, key)] = val
    return _SCALAR[(conf, key)]


def _num_hist():
    return int(_conf_scalar("num_hist"))


TRAIN_CONFS = ("train_rdmreg.yaml", "train_lewm.yaml")


def _detach_conf():
    """The value of detach_target across EVERY training config in conf/.

    T2's claim is about all ~160 runs, not about one file, so the figure reads both configs
    a campaign run can be launched under and draws the value only when they agree.  If they
    ever stop agreeing the figure says so instead of quietly picking one.
    """
    vals = {_conf_scalar("detach_target", c) for c in TRAIN_CONFS}
    return vals.pop() if len(vals) == 1 else "differs by config"


def _ctor_default(name):
    """A keyword's default in VWorldModel.__init__.

    T2 rung 1 turns on a flag whose CONSTRUCTOR default is already the value it wants; that
    is the fact the figure carries, so it is read off the signature.
    """
    import inspect
    from models.visual_world_model import VWorldModel
    return inspect.signature(VWorldModel.__init__).parameters[name].default


_ARCHIVE = {}


def _archive():
    """The eval archive on the FIXED instrument -- the only one valid for a paired test."""
    if not _ARCHIVE:
        from analysis.collect_evals import collect, resolve_arm  # noqa: F401
        arms = collect(scheme="fixed")[0]
        _ARCHIVE["arms"] = arms
        _ARCHIVE["n_arms"] = len(arms)
        _ARCHIVE["n_runs"] = sum(len(v) for v in arms.values())
    return _ARCHIVE


def _jump_effects():
    """[(K, mean, lo, hi, n)] for jump{K} against LpWM-ltv, paired on shared seeds.

    figures.paired_effect, not a fresh implementation: it drops seeds present in only one
    arm rather than mean-imputing them, and uses the t critical value for n-1 df.
    """
    from analysis import figures as FG
    from analysis.collect_evals import resolve_arm
    arms = _archive()["arms"]
    ctrl = resolve_arm(arms, CTRL_ARM)
    out = []
    for k in JUMP_K:
        e = FG.paired_effect(arms, ctrl, resolve_arm(arms, f"PiWM-jump{k}"))
        out.append((k, e["mean"], e["lo"], e["hi"], e["n"]))
    return out


def _signed(v, nd=3):
    """A signed effect with a REAL minus sign, so it lines up with every other number."""
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.{nd}f}"


# --- local primitives --------------------------------------------------------------
def framerow(x, y, n, inside, key, cell=44, gap=6, h=40, labels=(), out_key="slate"):
    """A run of frame slots, `inside` of them (the RIGHT-most) within the model's window.

    The window is drawn as a property of the FRAMES, not of the taps, because that is what
    the codebase enforces: n_lags == num_frames == num_hist, so a frame outside the window
    is not merely unused -- it is not in the graph at all.  A slot outside is hollow with a
    DIM border; one inside carries the key's fill.
    """
    s = ""
    for i in range(n):
        cx = x + i * (cell + gap)
        on = i >= n - inside
        d = "" if on else ' stroke-dasharray="4,3"'
        s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{h}" rx="4" '
              f'fill="{ACCENT_FILL[key] if on else WHITE}" '
              f'fill-opacity="{0.85 if on else 1}" '
              f'stroke="{ACCENT[key] if on else DIM}" '
              f'stroke-width="{1.6 if on else 1.1}"{d}/>\n')
        if i < len(labels):
            s += txt(cx + cell / 2, y + h + 15, labels[i], 11,
                     fill=INK if on else MUTED)
    return s


def slotcell(x, y, w, h, body, key, hollow=False, note=""):
    """One cell of T4's (configuration x offset) matrix: which lag slots read this offset.

    A cell is drawn even when EMPTY (an en-dash), because "no slot reads offset 1" is the
    content of the skip row and an absent cell would read as a layout gap.
    """
    c = ACCENT[key]
    d = ' stroke-dasharray="5,4"' if hollow else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
         f'fill="{WHITE if hollow else ACCENT_FILL[key]}" '
         f'fill-opacity="{1 if hollow else 0.6}" stroke="{c}" '
         f'stroke-width="1.5"{d}/>\n')
    s += txt(x + w / 2, y + h / 2 + (5 if not note else 0), body,
             fit_size(body, 14, w - 16, floor=10.5), fill=c, weight="bold")
    if note:
        s += txt(x + w / 2, y + h / 2 + 16, note,
                 fit_size(note, 10.5, w - 12, floor=9), fill=MUTED)
    return s


def whisker(x0, x1, xm, y, key, r=5.0, w=1.8, fill=None):
    """A confidence interval as a bar with end caps and the point estimate on it.

    Caps, not a bare line: an interval drawn as a plain segment is indistinguishable from a
    wire in a figure that is otherwise all wires.  `fill=WHITE` hollows the point estimate,
    which is how this project draws a NULL.
    """
    c = ACCENT[key]
    s = (f'<path d="M{x0:.1f},{y} L{x1:.1f},{y} M{x0:.1f},{y - 6} L{x0:.1f},{y + 6} '
         f'M{x1:.1f},{y - 6} L{x1:.1f},{y + 6}" stroke="{c}" stroke-width="{w}" '
         f'stroke-linecap="round" fill="none"/>\n')
    return s + mark(round(xm, 1), y, key, r=r, fill=fill)


def vrule(x, y0, y1, key, w=1.2, op=0.45, dash=""):
    """A hairline divider inside an inset: `these two halves are one statement`."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x},{y0} L{x},{y1}" stroke="{ACCENT[key]}" stroke-width="{w}" '
            f'stroke-opacity="{op}"{d}/>\n')


# Every frame a label must stay inside, registered as it is drawn.  The canvas check is not
# enough on its own: a label centred on half of an inset can run past that inset's border by
# tens of units while still sitting comfortably inside the page.
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
    """The title is one centred run at y = 872 and every figure drops a connector past it, so
    its half-width is a HARD constraint, not a matter of taste."""
    body = re.sub(r"<[^>]*>", "", title)
    wide = text_width(body.replace(" -- ", " – "), size, "bold")
    right = cx + wide / 2
    assert right < corridor_x - clear, (
        f"title runs to x={right:.0f}, into the connector corridor at x={corridor_x}: "
        f"shorten it")
    return title


# ====================================================== T1: the recurrent state
T1_TITLE = "(T1) PiWM-ssm -- a predictor that carries STATE  [LAUNCHED]"


def t1_ssm():
    """The FIR tap sum replaced by an IIR state, with a 384-parameter control."""
    K, C = "amber", "magenta"          # the intervention, and the contrasting condition
    H = _num_hist()
    b = base(_check_title(T1_TITLE, 880))
    b += wire_to_pred(K, "a STATE, not a window")

    PH = 782
    s = pane(PX, PY, PW, PH, K,
             "module:  s" + sub("", "t", 12) + " = A s" + sub("", "t" + MINUS + "1", 12)
             + " + B" + sub("", "z", 12) + " z" + sub("", "t", 12) + " + B"
             + sub("", "a", 12) + " a" + sub("", "t", 12) + " ,   z"
             + sub("", "t+1", 12) + " = W ReLU( C s" + sub("", "t", 12) + " )")

    # -- row A, left: the window every other mode has.  The picture is of the FRAMES,
    # because the cutoff is a property of what the predictor is fed, not of the taps.
    s += frame(60, 950, 404, 254, "FIR  " + NDASH + "  a hard H-tap window", C,
               note="H = num_frames = num_hist = " + str(H)
                    + " , in all 400 sampled configs")
    s += txt(258, 1020, "the cutoff", 12, anchor="end", fill=ACCENT["crit"], weight="bold")
    s += txt(337, 1020, "the " + str(H) + " taps", 12, fill=ACCENT[C], weight="bold")
    s += framerow(115, 1034, 6, H, C,
                  labels=("t" + MINUS + "5", "t" + MINUS + "4", "t" + MINUS + "3",
                          "t" + MINUS + "2", "t" + MINUS + "1", "t"))
    s += (f'<path d="M262,1026 L262,1092" stroke="{ACCENT["crit"]}" stroke-width="1.8" '
          f'stroke-dasharray="5,4"/>\n')
    s += txt(262, 1132, SIGMA + sub("", "k", 11) + " A" + sub("", "k", 11) + " z"
             + sub("", "t" + MINUS + "k", 11) + "   +   B a" + sub("", "t", 11), 16,
             fill=ACCENT[C], weight="bold")
    s += txt(262, 1158, "the window IS the model's memory", 12, fill=MUTED)
    s += txt(262, 1180, "nothing older than z" + sub("", "t" + MINUS + "2", 10)
             + " is in the graph", 12, fill=ACCENT["crit"], weight="bold")

    # -- row A, right: the state.  The feedback returns on its own corridor at y = 1156,
    # which no other wire touches -- that is what makes it read as a loop and not a wire.
    s += frame(476, 950, 400, 254, "IIR  " + NDASH + "  one state, a learned decay", K,
               note="the past enters with weight A" + sub("", "k", 10)
                    + " , with no cutoff at all")
    s += pill(534, 1036, sub("z", "t", 11), K, rx=26, ry=15, fill=WHITE)
    s += circle(534, 1080, 17, sub("a", "t", 10), fill=ACCENT_FILL[K], size=12)
    s += pill(534, 1124, sub("s", "t" + MINUS + "1", 11), K, rx=32, ry=15, fill=WHITE)
    s += mbox(592, 1020, 56, 32, "B" + sub("", "z", 11), K, size=15)
    s += mbox(592, 1064, 56, 32, "B" + sub("", "a", 11), K, size=15)
    s += mbox(592, 1108, 56, 32, "A", K, size=16)
    s += aw(560, 1036, 588, 1036, K)
    s += aw(551, 1080, 588, 1080, K)
    s += aw(566, 1124, 588, 1124, K)
    s += opnode(694, 1080, "plus", K, r=16)
    s += aw(648, 1036, 678, 1068, K)
    s += aw(648, 1080, 674, 1080, K)
    s += aw(648, 1124, 678, 1092, K)
    s += aw(710, 1080, 728, 1080, K)
    s += pill(758, 1080, sub("s", "t", 11), K, rx=28, ry=17)
    s += apoly([(758, 1097), (758, 1156), (534, 1156), (534, 1139)], K, dash="5,4")
    s += txt(646, 1178, "the state, fed back", 12, fill=ACCENT[K], weight="bold")

    # -- row B: the init.  Drawn, because A = 0.9 I is the one choice with no precedent.
    s += frame(60, 1216, 816, 140, "INIT " + NDASH + "  near-identity and action-free, "
               "exactly as `additive` does it", K,
               note="a contraction is the only init decision here with no precedent "
                    "among the FIR modes")
    for i, (lab, note) in enumerate((("A  =  0.9 I", "a CONTRACTION"),
                                     ("B" + sub("", "z", 12) + "  =  I", "identity in"),
                                     ("B" + sub("", "a", 12) + "  =  0", "AdaLN-zero"),
                                     ("C  =  I", "identity out"))):
        s += mbox(84 + i * 196, 1268, 180, 56, lab, K, size=18, sub_=note)
    s += txt(468, 1344, "0.9 and not 1.0: the FIR modes carry no state, so none of them "
             "could ever diverge", 12, fill=ACCENT[K], weight="bold")

    # -- row C: the control, measured rather than asserted.  Both cores read out the SAME
    # way, so the two arrows meet on one readout box instead of running past each other.
    p_fir, p_iir = _params("mlp_var", H), _params("ssm", H)
    s += frame(60, 1370, 816, 152, "the CONTROL " + NDASH + "  matched to "
               + _fmt(abs(p_iir - p_fir)) + " parameters out of " + _fmt(p_fir), K,
               note="both counts built at render time: LinearDynamicsPredictor, D = "
                    + str(PROJ_D) + " , H = " + str(H) + " , num_patches = 1")
    s += mbox(96, 1428, 250, 60, _fmt(p_fir), C, size=22, sub_="PiWM-mlpvar  " + DOT
              + "  the FIR core")
    s += mbox(590, 1428, 250, 60, _fmt(p_iir), K, size=22, sub_="PiWM-ssm  " + DOT
              + "  the IIR state")
    s += mbox(390, 1428, 156, 60, "ReLU " + ARROWC + " W", "slate", size=17,
              sub_="the SAME readout")
    s += aw(346, 1458, 386, 1458, C)
    s += aw(586, 1458, 550, 1458, K)
    s += txt(468, 1508, "FIR against IIR is then the only real difference between the two "
             "arms", 12.5, fill=INK, weight="bold")

    s += frow(1568, [
        (["FIR:   z" + sub("", "t+1", 12) + "  =  W ReLU( " + SIGMA + sub("", "k", 12)
          + " A" + sub("", "k", 12) + " z" + sub("", "t" + MINUS + "k", 12) + "  +  B a"
          + sub("", "t", 12) + " )"], INK, "normal"),
        (["IIR:   z" + sub("", "t+1", 12) + "  =  W ReLU( C s" + sub("", "t", 12)
          + " ) ,   s" + sub("", "t", 12) + " = A s" + sub("", "t" + MINUS + "1", 12)
          + " + B" + sub("", "z", 12) + " z" + sub("", "t", 12) + " + B"
          + sub("", "a", 12) + " a" + sub("", "t", 12)], INK, "bold")])
    s += frow(1592, [
        ("the scan is T = num_hist steps long, the same length the FIR modes unroll",
         INK, "normal"),
        ("the link, the rollout and the planner all see the same interface", MUTED,
         "normal", 0.92)])
    s += strip(PIN, 1610, [
        ("every predictor before this", "FIR , " + str(H) + " taps", True),
        ("PiWM-mlpvar, the control", _fmt(p_fir), False),
        ("PiWM-ssm", _fmt(p_iir), False),
        ("the whole difference", _fmt(abs(p_iir - p_fir)) + " params", False)], K, cw=180)
    return b + s, PY + PH + 40


# ======================================================== T2: the target regime
T2_TITLE = "(T2) the target regime -- live, stop-grad, EMA  [LAUNCHED]"


def t2_target():
    """Three target regimes as one switch, and the teacher's checkpoint plumbing."""
    K = "amber"                        # the EMA teacher is the intervention under test
    b = base(_check_title(T2_TITLE, 890))
    # round7 S1 / round7-v2 G1's encoder route, reversed: the panel drives the RIGHT
    # encoder, whose output IS the target.  x = 890 is the empty right margin; the label
    # stops at x = 712 because base()'s RDMReg return runs vertically at x = 726.
    b += poly([(890, PY), (890, 560), (872, 560)], color=ACCENT[K], w=1.8, dash="6,4")
    b += txt(712, 502, "the TARGET is this encoder's own output", 14, anchor="end",
             fill=ACCENT[K], weight="bold")

    PH = 752
    s = pane(PX, PY, PW, PH, K,
             "module:  target  =  h( Enc( o" + sub("", "t+1", 12) + " ) )  "
             + NDASH + "  what differs is whether gradient returns to Enc")

    # -- row A: the three regimes, side by side, each drawn as the same three-block chain
    # so the ONLY thing that varies between columns is the return path.
    cfg = (
        (60, 188, "blue", "LIVE",
         "detach_target: " + _detach_conf(),
         "the gradient loops back", "into the encoder itself", "live"),
        (340, 468, "magenta", "STOP-GRAD",
         "detach_target: " + str(_ctor_default("detach_target")),
         "the target is a constant", "the constructor's own default", "sg"),
        (620, 748, K, "EMA TEACHER", "ema_m = 0.99  /  0.999",
         "a frozen copy of the encoder", "no path back into the student", "ema"),
    )
    for x0, cx, key, ttl, note, l1, l2, kind in cfg:
        s += frame(x0, 956, 256, 290, ttl, key, note=note)
        if kind == "ema":
            s += mbox(636, 1016, 96, 44, "Enc " + sub("", "f", 12), key, size=15,
                      sub_="student")
            s += mbox(764, 1016, 96, 44, "Enc " + sub("", "f", 12), key, size=15,
                      sub_="teacher")
            s += aw(732, 1038, 760, 1038, key)
            # centred on the arrow's own midpoint: under the student box it read as that
            # box's caption rather than as the update between the two encoders
            s += txt(748, 1078, "EMA", 11.5, fill=ACCENT[key], weight="bold")
            s += mbox(672, 1088, 152, 44, "target", key, size=16, sub_="from the TEACHER")
            s += aw(812, 1060, 812, 1084, key)
        else:
            s += mbox(cx - 76, 1016, 152, 44, "Enc " + sub("", "f", 12), key, size=16,
                      sub_="the student")
            s += aw(cx, 1060, cx, 1084, key)
            s += mbox(cx - 76, 1088, 152, 44, "target", key, size=16,
                      sub_=sub("z", "t+1", 11))
        s += aw(cx, 1132, cx, 1150, key)
        s += pill(cx, 1172, "MSE", "crit", rx=36, ry=18, fill=WHITE)
        if kind != "ema":
            # the return path is the whole content of the column, so it gets its own
            # corridor on the right of the chain rather than sharing the downward one
            rx = cx + 104
            s += apoly([(cx + 36, 1172), (rx, 1172), (rx, 1038), (cx + 80, 1038)],
                       "crit" if kind == "live" else key, dash="5,4")
            if kind == "sg":
                s += sgbar(rx, 1104, key=key)
        s += txt(cx, 1212, l1, 11.5, fill=ACCENT[key], weight="bold")
        s += txt(cx, 1230, l2, 11.5, fill=MUTED)

    # -- row B: the plumbing.  The teacher is drawn OUTSIDE the optimizer frame, because
    # "excluded from every optimizer" is a claim about where it is, not about what it is.
    s += frame(60, 1262, 816, 232, "the teacher lives OUTSIDE every optimizer " + NDASH
               + "  and INSIDE the checkpoint", K,
               note="requires_grad_(False) and handed to no optimizer, so mup.py's printed "
                    "LR schema never lists it")
    s += frame(84, 1324, 300, 138, "the optimizers", "slate",
               note="built from named sub-modules")
    for i, nm in enumerate(("encoder", "predictor", "action_encoder", "decoder")):
        s += mbox(100 + (i % 2) * 136, 1382 + (i // 2) * 38, 124, 32, nm, "slate", size=13)
    s += apoly([(384, 1398), (500, 1398)], K)
    s += txt(442, 1388, "the EMA update", 11.5, fill=ACCENT[K], weight="bold")
    s += txt(442, 1416, "after the step", 11, fill=MUTED)
    s += mbox(504, 1372, 190, 56, "encoder_ema", K, size=17, sub_="a frozen deepcopy")
    # the snowflake goes INSIDE the teacher's own box: standing between the box and the
    # checkpoint it read as a third stage of the chain rather than as a property of the box
    s += freeze(526, 1394, K, r=11)
    s += aw(694, 1400, 762, 1400, K)
    s += pill(818, 1400, "checkpoint", K, rx=52, ry=20)
    s += txt(818, 1440, "_keys_to_save", 11.5, fill=MUTED)
    s += txt(468, 1478, "without that entry the teacher RESETS to the student at every 4 h "
             "window boundary", 12.5, fill=ACCENT["crit"], weight="bold")

    arch = _archive()
    s += frow(1538, [
        ([THETA_EMA + "  " + ARROWC + "  m " + THETA_EMA + "  +  ( 1 " + MINUS + " m ) "
          + THETA + " ,   after the optimizer step"], INK, "bold"),
        (["target  =  h( Enc" + sub("", "ema", 12) + "( o" + sub("", "t+1", 12)
          + " ) ) ,   always detached"], INK, "normal")])
    s += frow(1562, [
        ("the constructor default is already "
         + str(_ctor_default("detach_target")) + "; both training configs are what turn "
         "it off", INK, "normal"),
        ("an EMA copy has no graph, so it is detached by construction", MUTED,
         "normal", 0.92)])
    s += strip(PIN, 1580, [
        ("the archive it ranges over", f"{arch['n_arms']} arms, {arch['n_runs']} evals",
         False),
        ("every training config", "detach_target: " + _detach_conf(), True),
        ("the ctor default", str(_ctor_default("detach_target")), False),
        ("EMA momenta", "0.99 , 0.999", False)], K, cw=180)
    return b + s, PY + PH + 40


THETA = "θ"
THETA_EMA = "θ" + '<tspan font-size="12" dy="5.0">ema</tspan><tspan dy="-5.0"></tspan>'


# =============================================================== T4: the lag axis
T4_TITLE = "(T4) PiWM-lagskip -- masking and skip on the LAG axis  [LAUNCHED]"


def t4_lag():
    """The lag slots, the skip that is expressible, and the dilation that is not."""
    K = "amber"
    H = _num_hist()
    b = base(_check_title(T4_TITLE, 880))
    b += wire_to_pred(K, "which past frame each lag slot reads")

    PH = 898
    s = pane(PX, PY, PW, PH, K,
             "module:  lag slot k reads z" + sub("", "t" + MINUS + "d", 12)
             + sub("", "k", 10) + "  " + NDASH + "  and every d"
             + sub("", "k", 12) + " must lie in [ 0 , num_frames )")

    # -- row A: the frames on top, the three configurations under them.  Both live on the
    # SAME four columns, so "this configuration puts a slot in the dead column" is visible
    # rather than argued.
    s += frame(60, 950, 816, 268, "the " + str(H) + " lag slots, and the offsets each "
               "configuration gives them", K,
               note="n_lags == num_frames == num_hist == " + str(H)
                    + " , and the predictor is fed z_src = z_emb[:, :num_hist]")
    CX = [355, 473, 591, 709]
    for i, cx in enumerate(CX):
        s += txt(cx, 1024, "offset " + str(i), 11.5,
                 fill=MUTED if i < H else ACCENT["crit"], weight="bold")
    s += txt(292, 1058, "the frames in the graph", 12, anchor="end", fill=MUTED)
    for i, cx in enumerate(CX):
        on = i < H
        s += slotcell(cx - 55, 1034, 110, 38,
                      "z" + sub("", "t", 10) if i == 0
                      else "z" + sub("", "t" + MINUS + str(i), 10),
                      "blue" if on else "crit", hollow=not on)
    s += txt(473, 1090, "T  =  num_hist  =  " + str(H), 11, fill=MUTED)
    s += txt(709, 1090, "outside the graph", 11, fill=ACCENT["crit"], weight="bold")
    for r, (offs, lab, key, chip) in enumerate((
            (LAG_SETS[0], "uniform", "blue", "AS BUILT"),
            (LAG_SETS[1], "SKIP", K, "LAUNCHED"),
            (LAG_SETS[2], "dilation", "crit", "DEAD"))):
        y = 1102 + r * 38
        s += txt(292, y + 20, lab + "   [ " + " , ".join(map(str, offs)) + " ]", 12.5,
                 anchor="end", fill=ACCENT[key], weight="bold")
        for i, cx in enumerate(CX):
            who = [f"k{j}" for j, d in enumerate(offs) if d == i]
            body = " ".join(who) if who else NDASH
            dead = who and i >= H
            s += slotcell(cx - 55, y, 110, 30, body, "crit" if dead else key,
                          hollow=not who)
        s += badge(776, y + 4, chip, key, size=11)

    # -- row B: the failure, as a chain.  Four blocks, because the reason the arm died is a
    # four-step consequence and a sentence cannot be pointed at.
    s += frame(60, 1234, 816, 214, "why offset " + str(H) + " cannot exist " + NDASH
               + "  and what it cost", "crit",
               note="before the guard a single-process CPU test did not raise here: it "
                    "silently dropped the slot")
    for i, (lab, note) in enumerate((
            ("T  =  " + str(H), "z_src = z_emb[:, :num_hist]"),
            ("d" + sub("", "2", 12) + " = " + str(H) + "  " + "≥" + "  T",
             "the dilated slot"),
            ("break", "lags[2] never runs"),
            ("DDP refuses", "unused parameter"))):
        s += mbox(86 + i * 196, 1300, 176, 52, lab, "crit", size=18, sub_=note)
        if i:
            s += aw(66 + i * 196, 1326, 82 + i * 196, 1326, "crit")
    s += txt(468, 1382, "45 jobs died proving it " + NDASH + " every one with the same DDP "
             "unused-parameter error", 13, fill=ACCENT["crit"], weight="bold")
    s += txt(468, 1404, "there is no spare frame to dilate INTO: a dilation here can only "
             "SKIP, by repeating an offset", 11.5, fill=MUTED)
    ok23, ok22 = _guard_ok((0, 2, 3)), _guard_ok((0, 2, 2))
    s += txt(468, 1428, "now a construction-time assert " + NDASH + "  checked at render:  "
             "[0,2,3] " + ("accepted" if ok23 else "REJECTED") + " ,   [0,2,2] "
             + ("accepted" if ok22 else "REJECTED"), 12.5, fill=ACCENT[K], weight="bold")

    # -- row C: the other half of T4, which needed no new machinery at all
    s += frame(60, 1464, 816, 176, "lag MASKING " + NDASH + "  independent Bernoulli over "
               "the slots k " + "≥" + " 1", K,
               note="lag 0 is never masked: dropping z" + sub("", "t", 10)
                    + " is an input ablation, not a temporal-redundancy test")
    for i, (lab, note, key) in enumerate((("k  =  0", "never masked", "blue"),
                                          ("k  =  1", "kept w.p. 1 " + MINUS + " p", K),
                                          ("k  =  2", "kept w.p. 1 " + MINUS + " p", K))):
        s += mbox(96 + i * 128, 1524, 116, 46, lab, key, size=16, sub_=note)
    s += vrule(496, 1518, 1576, "slate")
    s += mbox(524, 1524, 156, 46, "p  =  0.25", K, size=18, sub_="PiWM-lagmask25")
    s += mbox(696, 1524, 156, 46, "p  =  0.50", K, size=18, sub_="PiWM-lagmask50")
    s += txt(468, 1598, "structurally free: _trunk already drops unavailable lags at cold "
             "start, and A" + sub("", "k", 10) + " " + DOT + " 0 = 0 exactly", 12,
             fill=INK, weight="bold")
    s += txt(468, 1620, "eval is untouched by the self.training guard, so nothing about "
             "planning or the rollout changes", 11.5, fill=MUTED)

    s += frow(1684, [
        (["slot k reads z" + sub("", "t" + MINUS + "d", 12) + sub("", "k", 10)
          + " ,   d" + sub("", "k", 12) + " " + "∈" + " [ 0 , num_frames )"], INK,
         "bold"),
        (["masking:  A" + sub("", "k", 12) + " z" + sub("", "t" + MINUS + "k", 12)
          + "  " + ARROWC + "  m" + sub("", "k", 12) + " A" + sub("", "k", 12) + " z"
          + sub("", "t" + MINUS + "k", 12) + " ,   m" + sub("", "k", 12) + " ~ Bern( 1 "
          + MINUS + " p )"], INK, "normal")])
    s += frow(1708, [
        ("the block is static in 48.1% of single steps, so a tight window may spend two "
         "thirds of its history on nothing", INK, "normal"),
        ("same parameter count either way", MUTED, "normal", 0.92)])
    s += strip(PIN, 1726, [
        ("lag slots", str(H) + " = num_hist", False),
        ("dilation [0,2,3]", "45 jobs dead", True),
        ("what IS expressible", "skip [0,2,2]", False),
        ("mask doses", "0.25 , 0.50", False)], K, cw=180)
    return b + s, PY + PH + 40


# ========================================================= T5: the num_hist ladder
T5_TITLE = "(T5) the num_hist ladder -- the ORDER of the dynamics  [LAUNCHED]"


def t5_hist():
    """H as the FIR order, with H = 1 as a plumbing test, against jump{K}'s horizon."""
    K = "amber"
    H0 = _num_hist()
    b = base(_check_title(T5_TITLE, 880))
    b += wire_to_pred(K, "how much past the predictor integrates")

    PH = 750
    s = pane(PX, PY, PW, PH, K,
             "module:  n_lags = num_hist , so H is the FIR ORDER:  z"
             + sub("", "t+1", 12) + " = " + SIGMA + sub("", "k &lt; H", 12) + " A"
             + sub("", "k", 12) + " z" + sub("", "t" + MINUS + "k", 12) + " + B a"
             + sub("", "t", 12))

    # -- row A: the ladder.  The parameter count is drawn on every rung because H moves it
    # too -- a reader who is not shown that will read the ladder as a pure context sweep.
    s += frame(60, 950, 816, 302, "the ladder " + NDASH + "  H taps of past, with the "
               "TARGET held at one step", K,
               note="H has been " + str(H0) + " in all 400 sampled run configs across "
                    "seven rounds")
    s += txt(400, 1014, "H", 11.5, anchor="end", fill=MUTED, weight="bold")
    s += txt(548, 1014, "the frames its taps read", 11.5, fill=MUTED)
    s += txt(690, 1014, "predictor params", 11, anchor="start", fill=MUTED)
    for i, h in enumerate(LADDER):
        y = 1032 + i * 42
        base_arm = h == H0
        key = "blue" if base_arm else K
        s += txt(270, y + 5, CTRL_ARM if base_arm else f"PiWM-hist{h}", 12.5,
                 anchor="end", fill=ACCENT[key], weight="bold")
        s += txt(400, y + 6, str(h), 16, anchor="end", fill=ACCENT[key], weight="bold")
        s += framerow(430, y - 13, 8, h, key, cell=26, gap=4, h=26,
                      labels=("t" + MINUS + "7", "", "", "", "", "", "",
                              "t") if i == len(LADDER) - 1 else ())
        s += txt(690, y + 5, _fmt(_params("ltv", h)), 12, anchor="start", fill=INK)
        s += badge(788, y - 6, "AS BUILT" if base_arm else "NEW", key, size=11)
    s += txt(468, 1244, "H moves the parameter count too, from " + _fmt(_params("ltv", 1))
             + " to " + _fmt(_params("ltv", 8)) + " " + NDASH
             + " a ladder, not a matched pair", 11.5, fill=MUTED)

    # -- row B: H = 1 as a plumbing test, with the docstring's identity actually run
    s += frame(60, 1268, 400, 224, "H = 1 is a PLUMBING TEST", "slate",
               note="the docstring's own identity, exercised here")
    s += mbox(84, 1344, 140, 52, "var , H = 1", "slate", size=15, sub_="one tap")
    s += mbox(238, 1344, 40, 52, "=", "slate", size=24)
    s += mbox(292, 1344, 140, 52, "additive", "slate", size=16,
              sub_="W z  +  B a")
    s += txt(258, 1424, "checked at render, on matched weights:", 11.5, fill=MUTED)
    s += txt(258, 1444, "max | var(H=1) " + MINUS + " additive |  =  "
             + f"{_h1_identity():.1f}", 13, fill=ACCENT["slate"], weight="bold")
    s += txt(258, 1466, "and " + _fmt(_params("var", 1)) + " parameters each, at D = "
             + str(PROJ_D), 11.5, fill=MUTED)

    # -- row C: the axis this one is NOT.  jump{K} moved the target horizon and lost at
    # every K, and those four numbers are read from the archive at render time.
    eff = _jump_effects()
    n_null = sum(1 for _, _, l, h, _ in eff if l <= 0 <= h)
    # the title is what inset() fits to w - 40 and it stops at a 13.5 floor rather than
    # shrinking further, so the verdict lives in the crimson summary line below instead
    s += frame(476, 1268, 400, 224, "jump{K} moved the TARGET horizon", "crit",
               note="num_pred = K ; this ladder leaves the target at one step")
    lo = min(e[2] for e in eff)
    hi = max(e[3] for e in eff)
    span = max(hi - lo, 1e-6)
    X0, X1 = 606.0, 770.0

    def px(v):
        return X0 + (v - lo + 0.04 * span) / (1.08 * span) * (X1 - X0)

    zx = round(px(0.0), 1)
    s += (f'<path d="M{zx},1338 L{zx},1452" stroke="{MUTED}" stroke-width="1.2" '
          f'stroke-dasharray="4,4"/>\n')
    s += txt(zx, 1332, "0", 11, fill=MUTED)
    for i, (k, m, l, h, n) in enumerate(eff):
        # A NULL is never drawn like a negative: an interval that spans zero gets slate, a
        # HOLLOW marker and the word NULL -- the same encoding round7-design.svg uses. Two
        # of these four span zero, so "lost at every K" would have been an overclaim; what
        # the archive supports is that no K gained.
        y = 1356 + i * 26
        null = l <= 0 <= h
        key = "slate" if null else "crit"
        s += txt(530, y + 4, "K = " + str(k), 12.5, anchor="end", fill=INK, weight="bold")
        s += txt(594, y + 4, _signed(m), 12.5, anchor="end", fill=ACCENT[key],
                 weight="bold")
        s += whisker(px(l), px(h), px(m), y, key, fill=WHITE if null else None)
        if null:
            s += txt(806, y + 4, "NULL", 11, anchor="end", fill=ACCENT["slate"],
                     weight="bold")
        s += txt(858, y + 4, "n = " + str(n), 11.5, anchor="end", fill=MUTED)
    s += txt(676, 1462, "paired against " + CTRL_ARM + " on shared seeds, read at render",
             11, fill=MUTED)
    s += txt(676, 1478, str(len(eff) - n_null) + " negative, " + str(n_null)
             + " NULL " + NDASH + " and not one positive", 11.5,
             fill=ACCENT["crit"], weight="bold")

    s += frow(1536, [
        (["the ladder:  H " + "∈" + " { " + " , ".join(map(str, LADDER))
          + " } ,   target at t + 1"], INK, "bold"),
        (["jump{K}:  target at t + K ,   H fixed at " + str(H0)], INK, "normal")])
    s += frow(1560, [
        ("one axis changes what the model READS, the other changes what it is ASKED for",
         INK, "normal"),
        ("H = 1 doubles as the plumbing test for the knob itself", MUTED, "normal", 0.92)])
    s += strip(PIN, 1578, [
        ("H, in all 400 sampled configs", str(H0), True),
        ("the new rungs", " , ".join(str(h) for h in LADDER if h != H0), False),
        ("H = 1", "= additive", False),
        ("jump{K}, best to worst", _signed(max(e[1] for e in eff)) + " to "
         + _signed(min(e[1] for e in eff)), True)], K, cw=170)
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
    checkable by eye on a 940 x 1750 canvas; this reads the emitted file back and says so.
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
    ("arch-t1-ssm.svg", t1_ssm),
    ("arch-t2-target.svg", t2_target),
    ("arch-t4-lag.svg", t4_lag),
    ("arch-t5-hist.svg", t5_hist),
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
