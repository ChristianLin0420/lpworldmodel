"""Round 8, the SPATIAL group: the pixel gradient into the predictor, and the two ST arms.

    python analysis/round8_st_figs.py --out diary/assets/2026-09-06 [--png]
    -> arch-s1-pdpred.svg    the two decoder paths, and the one that reaches the predictor
    -> arch-st2-ssm.svg      one broadcast operator against 256 per-token states
    -> arch-st3-pdpred.svg   per-token next-frame pixels + the lag skip, and its two halves

Round 8's temporal arms are drawn in analysis/round8_t_figs.py.  These three are the SPATIAL
ones -- the arms whose mechanism is indexed by the patch token, and the one structural defect
that made them worth launching at all: for ~120 contrasts the pixel loss was wired to a frame
the model had ALREADY SEEN, so no pixel gradient ever reached the predictor.

    S1   arch-s1-pdpred.svg   the existing decoder term decodes z_emb and its gradient stops
                              at the encoder; DECODE_PRED_W decodes z_pred -- a frame NOT seen
                              -- onto that frame's real pixels, and the gradient runs into
                              lags.0/1/2.  GREEN: the path as built.  AMBER: the new term.
                              CRIMSON: the dead branch that made this look done.
    ST2  arch-st2-ssm.svg     the predictor is ONE operator broadcast over every token
                              (measured: identical parameter count at P = 1 and P = 256), so a
                              token's dynamics are not its own.  ST2 gives each token a state.
                              GREEN: as built.  AMBER: the intervention.  PURPLE: the two
                              ablations that make a win attributable.
    ST3  arch-st3-pdpred.svg  S1's pixel target and T4's lag skip in one arm, with the two
                              halves that each run separately, and the three single-factor
                              contrasts that charge a result to one of them.

STYLE.  analysis/style.py, through analysis/arch_figs.py.  Every primitive is imported from
analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py / round6_arch_figs.py /
round7_arch_figs.py -- nothing here re-implements one, and NO EXISTING FILE IS EDITED (other
agents are working in this repo at the same time).  The import set is
analysis/round7_v2_arch_figs.py's, plus four marks taken from the SAME modules because the
house already owns them and re-drawing them would fork them -- `dot` (the fan-out junction),
`cells` (S1's 17 parameter slots), and round 7's `slots` and `tokengrid` (the seed row, and
the 16 x 16 token grid that is P = 256 exactly, one cell per real token) -- and the two
connector routes these figures attach through, `wire_to_mse` and `wire_to_pred`.

Hues by IDENTITY, never by rank:

    green    the system as built -- the decoder term that stops at the encoder, the one
             broadcast operator, the uniform lag set [0,1,2], LpWM-ltv
    amber    the intervention under test -- pdpred, st-ssm, st-pdpred and its two halves
    purple   the contrasting condition -- ST2's two ablations, which hold one axis each
    crimson  a failure or a retraction -- the unreachable AND detached z_pred decode, and
             "the token axis buys no dynamics"
    slate    neutral -- an arm mean, a seed already on record, a NULL
    teal     reserved here for nothing; no consensus arm appears in these three figures

LAYOUT RULES INHERITED FROM round6_arch_figs.py / round7_v2_arch_figs.py

  * Strict left-to-right dataflow on fixed row baselines; a signal that legitimately fans out
    gets a drawn dot().  The two gradient returns in S1 run BACKWARDS along their own
    corridor under the forward chain, which is what makes them read as gradients.
  * base() fixes every anchor, so a connector uses ONE named route.  S1 and ST3 both ADD A
    TERM TO THE OBJECTIVE, so both reuse round6's `wire_to_mse` verbatim -- the reader learns
    that route once and it means "this proposal adds a term to the loss".  ST2 changes the
    PREDICTOR, so it reuses `wire_to_pred`, the same route round8_t_figs' T1 uses for the same
    mechanism.  ST3 does NOT draw its lag half as a second connector: wire_to_pred and
    wire_to_mse share the x = 880 corridor from y = PY up to y = 380, and two wires may not
    share a segment -- so the lag half is drawn inside the panel, where it belongs anyway.
  * Every route passes the title's baseline at y = 872, so every title is WIDTH-CHECKED
    against its corridor (`_check_title`) rather than trusted.
  * Nothing is sized by hand where it can be measured: mbox / pill / inset / frow / strip all
    fit their own text and none may cross the panel border.
  * A footer line carries at most two clauses and an inset title is short: `frow()` shrinks
    its gap and then its type, and `inset()` shrinks its title, but both stop at a floor and
    then OVERFLOW SILENTLY.
  * A NULL is never drawn like a positive.  BOTH archive contrasts these figures carry span
    zero -- patchdecode against its own detach control, and PiWM-columns against the baseline
    -- so both get slate, a HOLLOW marker and the word NULL (`effrow`), and the figures show
    ARM MEANS separately, labelled as descriptive and not as a contrast.
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of every
    <text> element from the same Helvetica metrics the layout used (arch_figs.text_width), and
    reports any pair that intersects, anything that leaves the canvas, and anything that
    crosses the border of the frame it sits in (every frame registers itself through
    `frame()` / `pane()`).  It is the SVG counterpart of analysis/round7_figs.py's audit(),
    and it exists for the same reason: at 940 x 1750 units "nothing overlaps" is not
    checkable by eye.

EVERY NUMBER IS READ OR COMPUTED AT RENDER TIME.  Nothing numeric below is typed into a
figure body; the module measures it and the figure draws what the measurement said.

  computed in-module, by building and running the real objects
    11 of 17 predictor parameter tensors reached by decode_pred_loss alone (`_reach`) --
      one backward pass through a real LinearDynamicsPredictor(mode="ltv", D = 384, P = 256)
      and a real PatchHead, counting tensors with a non-zero grad.  The same function also
      checks that the ORDINARY latent loss reaches exactly the same 11, which is what makes
      the 6 it misses ltv's zero-initialised low-rank factors rather than a defect of S1.
      S1's two name lists ("lags.0 / 1 / 2 , B , W , Ulag , UB" and "Vlag , VB , gate") are
      DERIVED from those measured lists, so the caption cannot drift from the measurement.
    the DDP-safety of the lag skip (`_grad_none`): the number of predictor parameters that
      come back with grad IS NONE under [0,2,2] and under the uniform set, run here because
      PiWM-lagdil lost 45 jobs to exactly that error and "zero" only means something beside
      the number the baseline already has.  ST3 also derives WHICH frame the skip drops, and
      which one takes the freed slot, from the two lag sets rather than naming them.
    811 840 predictor params at num_patches = 1 AND at num_patches = 256 (`_params`)
    738 432 ssm params at both, and the ssm state's SHAPE (B, P, D), captured at render time
      from a forward hook on the A module (`_state_shape`) -- so "one state per token" is a
      measurement of the running module, not a reading of its source
  read from the config / the source at render time
    img_size, patch_size -> num_patches = (img/patch)**2 = 256, the encoder's own rule
      (models/vit_encoder.py:67), and num_hist = 3          conf/train_rdmreg.yaml,
                                                            conf/encoder/vit_scratch_patch.yaml
    action_conditioning = adaln in BOTH training configs, which is what makes _forward_adaln
      the path every campaign run takes                     conf/train_*.yaml
    the four line numbers S1 points at, and the check that the detached decode sits BELOW the
      adaln dispatch (`_srcline`)                           models/visual_world_model.py
    every arm's spec and feature, and every spec DIFF drawn ("differs in exactly one flag"),
      parsed from the launcher itself                       scripts/run_campaign.sh
    the count of arms defined there, how many set DECODE_PRED_W, and (from the archive)
      how many finished evals those arms have between them -- which is the whole of S1's
      "no run ever did this", drawn as a number rather than asserted
  read from the ARCHIVE at render time (`analysis.collect_evals.collect(scheme="fixed")`)
    PiWM-patchdecode and LpWM-ltv arm means and seed sets; PiWM-columns' mean and n; the
      paired contrasts patchdecode-vs-detach and columns-vs-baseline, both NULL; the total
      arms and runs on the fixed instrument, which is the range of "no run ever did this";
      and the current n of every arm this round launched (0 while they are in flight)

  STATED, NOT COMPUTED -- one clause, a record of what happened rather than a measurement:
    "~120 contrasts" is the campaign's own phrasing (diary/2026-09-06.md section 2,
    scripts/run_campaign.sh wave29_arms).  The figures do not draw it: they draw the ARM and
    RUN counts on the fixed instrument instead, which are read from the archive.

  A SUBTLETY, and why the feature is not simply read from ARM_FEAT[ ]: submit_arm() uses
    "${ARM_FEAT[$arm]:-${FEATURE}}", so an arm with no ARM_FEAT line inherits the
    invocation-wide FEATURE -- PiWM-columns has no ARM_FEAT line and every one of its runs is
    a patch run.  `_arm_feat` therefore prefers the explicit line, falls back to the archive's
    own run-name tag (collect_evals keys a patch arm "<name>_patch"), and only then to cls.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    _marker, arrow, base, domeup, poly, sub, text_width, title_line, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, DOT, IMPLIES, MINUS, NDASH, TIMES,
    apoly, aw, badge, dot, fit_size, mbox, pill, strip,
)
from analysis.round5_figs_obj import (  # noqa: E402,F401
    cells, frow, inset, panel, strikeout,
)
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PIX, PW, PX, PY, SHIFT, W, ellipsis_v, mark, out_emit, wire_from_pred,
    wire_to_mse, wire_to_pred,
)
from analysis.round7_arch_figs import cbadge, slots, tokengrid  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-06")
VWM = os.path.join(REPO, "models", "visual_world_model.py")
CAMPAIGN = os.path.join(REPO, "scripts", "run_campaign.sh")
TRAIN_CONFS = ("train_rdmreg.yaml", "train_lewm.yaml")

SQ = "²"
BASE_ARM = "LpWM-ltv"


# --- the numbers -------------------------------------------------------------------
def _fmt(n):
    """A thin space between thousands: 811 840 rather than 811,840 or 811840."""
    return f"{n:,}".replace(",", " ")


def _signed(v, nd=3):
    """A signed effect with a REAL minus sign, so it lines up with every other number."""
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.{nd}f}"


_SRC = {}


def _src(path):
    if path not in _SRC:
        with open(path) as f:
            _SRC[path] = f.read()
    return _SRC[path]


def _srcline(pat, path=VWM, group=0):
    """The 1-based line number where `pat` matches, searched over the WHOLE file.

    S1's whole claim is about WHICH code runs, so the figure points at line numbers -- and a
    typed line number is stale the moment anything above it moves.  These are found in the
    file at render time and the module fails loudly if one stops existing.  The search is
    multi-line and takes a group, because the one line S1 has to point at (`z_pred.detach()`)
    occurs four times in this file and is only identified by the `self.decode(` above it.
    """
    m = re.search(pat, _src(path), re.S)
    assert m, f"{pat!r} no longer occurs in {os.path.basename(path)}"
    return _src(path).count("\n", 0, m.start(group)) + 1


_SCALAR = {}


def _conf_scalar(key, conf="train_rdmreg.yaml"):
    """A top-level scalar from a hydra config, read from the file (comments stripped)."""
    if (conf, key) not in _SCALAR:
        val = None
        with open(os.path.join(REPO, "conf", conf)) as f:
            for line in f:
                if line.startswith(key + ":"):
                    val = line.split(":", 1)[1].split("#")[0].strip()
                    break
        _SCALAR[(conf, key)] = val
    return _SCALAR[(conf, key)]


def _enc_scalar(key, enc="vit_scratch_patch.yaml"):
    with open(os.path.join(REPO, "conf", "encoder", enc)) as f:
        for line in f:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].split("#")[0].strip()
    raise AssertionError(f"{key} not in conf/encoder/{enc}")


def _num_hist():
    return int(_conf_scalar("num_hist"))


def _num_patches():
    """models/vit_encoder.py:67, verbatim: grid = img // patch, num_patches = grid**2.

    P is the axis all three figures are indexed by, so it is derived from the two knobs that
    set it rather than written down as 256.
    """
    return (int(_conf_scalar("img_size")) // int(_enc_scalar("patch_size"))) ** 2


def _adaln_everywhere():
    """action_conditioning across EVERY training config a campaign run can be launched under.

    This is the premise of S1's structural claim: _forward_adaln is the path every run takes
    only because both configs say adaln.  If they ever stop agreeing the figure says so
    instead of quietly picking one.
    """
    vals = {_conf_scalar("action_conditioning", c) for c in TRAIN_CONFS}
    return vals.pop() if len(vals) == 1 else "differs by config"


_PARAMS = {}


def _params(mode, P, H=None, D=None):
    """The parameter count of a real LinearDynamicsPredictor, built here.

    ST2's claim is `811 840 at P = 1 and at P = 256`, and a claim of that shape is only worth
    making if both numbers come out of the constructor rather than out of a note.
    """
    H = _num_hist() if H is None else H
    D = int(_conf_scalar("embed_dim")) if D is None else D
    key = (mode, P, H, D)
    if key not in _PARAMS:
        from models.infojepa_modules import LinearDynamicsPredictor
        p = LinearDynamicsPredictor(input_dim=D, output_dim=D, hidden_dim=D,
                                    num_frames=H, num_patches=P, mode=mode)
        _PARAMS[key] = sum(q.numel() for q in p.parameters())
    return _PARAMS[key]


_REACH = {}


def _reach():
    """Which predictor parameters a PIXEL loss on z_pred actually reaches, measured.

    Builds the real LinearDynamicsPredictor and the real PatchHead, runs one forward and one
    backward from `decode_pred_loss` alone, and counts the parameter tensors that come back
    with a non-zero gradient.  Then does it again from the ORDINARY latent loss, because the
    honest reading of "11 of 17" needs the other 6 named: they are ltv's zero-initialised
    low-rank factors, which no loss reaches at init, so the pixel term is not weaker than the
    term the campaign already runs -- it is the same reach, through a different signal.
    """
    if not _REACH:
        import torch
        from models.decoder.patch_head import PatchHead
        from models.infojepa_modules import LinearDynamicsPredictor as LDP
        D, H, P = int(_conf_scalar("embed_dim")), _num_hist(), _num_patches()
        grid = int(round(P ** 0.5))

        def hit(kind):
            torch.manual_seed(0)
            pred = LDP(input_dim=D, output_dim=D, hidden_dim=D, num_frames=H,
                       num_patches=P, mode="ltv")
            z_pred = pred(torch.randn(1, H, P, D), torch.randn(1, H, D))
            if kind == "pixel":
                head = PatchHead(emb_dim=D, patch_size=int(_enc_scalar("patch_size")),
                                 grid=grid)
                out, _ = head(z_pred)          # the head asserts p == grid**2 itself
            else:
                out = z_pred
            torch.nn.functional.mse_loss(out, torch.randn_like(out)).backward()
            got = [(n, q.grad is not None and bool(q.grad.abs().max() > 0))
                   for n, q in pred.named_parameters()]
            return [n for n, ok in got if ok], [n for n, ok in got if not ok]

        px, latent = hit("pixel"), hit("latent")
        _REACH["reached"], _REACH["missed"] = px
        _REACH["same_as_latent"] = px[0] == latent[0]
        _REACH["total"] = len(px[0]) + len(px[1])
        # the labels are DERIVED from the measured name lists: "lags.0 / 1 / 2" is a reading
        # of what came back with a gradient, not a caption written next to it
        lag_i = sorted({n.split(".")[1] for n in px[0] if n.startswith("lags.")})
        rest = [q for q in dict.fromkeys(n.split(".")[0] for n in px[0]) if q != "lags"]
        _REACH["reached_lab"] = ("lags." + " / ".join(lag_i) + " ,  " + " ,  ".join(rest)
                                 + "   " + DOT + "   the low-rank U factors")
        _REACH["missed_lab"] = " , ".join(
            dict.fromkeys(n.split(".")[0] for n in px[1]))
    return _REACH


_GRADNONE = {}


def _grad_none(offsets):
    """How many predictor parameters come back with grad IS NONE under this lag set.

    This is the DDP-safety check the lag arms live or die by: PiWM-lagdil lost 45 jobs to an
    unused-parameter error because an offset of num_frames leaves lags[2] outside the graph.
    ST3's footer claims the skip is safe, so the claim is run here rather than remembered --
    and it is run against the uniform baseline too, because "zero" only means something next
    to the number the baseline already has.
    """
    key = tuple(offsets)
    if key not in _GRADNONE:
        import torch
        from models.infojepa_modules import LinearDynamicsPredictor as LDP
        D, H, P = int(_conf_scalar("embed_dim")), _num_hist(), _num_patches()
        torch.manual_seed(0)
        pred = LDP(input_dim=D, output_dim=D, hidden_dim=D, num_frames=H, num_patches=P,
                   mode="ltv", lag_dilation=list(offsets))
        out = pred(torch.randn(1, H, P, D), torch.randn(1, H, D))
        torch.nn.functional.mse_loss(out, torch.randn_like(out)).backward()
        _GRADNONE[key] = sum(q.grad is None for q in pred.parameters())
    return _GRADNONE[key]


_STATE = {}


def _state_shape(P):
    """The shape of the ssm predictor's carried state, captured from the running module.

    A forward hook on `A` sees exactly the tensor the recurrence carries from one step to the
    next, so "a state per token" is read off the object rather than off the docstring.
    """
    if P not in _STATE:
        import torch
        from models.infojepa_modules import LinearDynamicsPredictor as LDP
        D, H = int(_conf_scalar("embed_dim")), _num_hist()
        pred = LDP(input_dim=D, output_dim=D, hidden_dim=D, num_frames=H,
                   num_patches=P, mode="ssm")
        seen = []
        h = pred.A.register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape)))
        with torch.no_grad():
            pred(torch.randn(1, H, P, D), torch.randn(1, H, D))
        h.remove()
        assert len({s for s in seen}) == 1, f"the state changed shape mid-scan: {seen}"
        _STATE[P] = seen[0]
    return _STATE[P]


_ARCHIVE = {}


def _archive():
    """The eval archive on the FIXED instrument -- the only one valid for a paired test."""
    if not _ARCHIVE:
        from analysis.collect_evals import collect
        arms = collect(scheme="fixed")[0]
        _ARCHIVE["arms"] = arms
        _ARCHIVE["n_arms"] = len(arms)
        _ARCHIVE["n_runs"] = sum(len(v) for v in arms.values())
    return _ARCHIVE


def _arm(name):
    """(n, mean, sorted seeds) for an arm, or (0, None, []) if it has no fixed-scheme eval.

    resolve_arm rather than arms.get: the patch arms are keyed "<name>_patch" and a bare
    .get() returns an empty dict that is indistinguishable from "not evaluated yet".  This
    has bitten three times in this codebase, always silently.
    """
    from analysis.collect_evals import ArmNameError, resolve_arm
    arms = _archive()["arms"]
    try:
        v = arms[resolve_arm(arms, name)]
    except ArmNameError:
        return 0, None, []
    return len(v), sum(v.values()) / len(v), sorted(int(k) for k in v)


def _effect(ctrl, arm):
    """The paired effect of `arm` against `ctrl`, or None if either side is missing.

    figures.paired_effect, not a fresh implementation: it drops seeds present in only one arm
    rather than mean-imputing them, and uses the t critical value for n-1 df.
    """
    from analysis import figures as FG
    from analysis.collect_evals import ArmNameError, resolve_arm
    arms = _archive()["arms"]
    try:
        e = FG.paired_effect(arms, resolve_arm(arms, ctrl), resolve_arm(arms, arm))
    except ArmNameError:
        return None
    return e if e["n"] else None


def _spec(name):
    """The ARMS[<name>] line from scripts/run_campaign.sh, asserted unique.

    Several waves redefine the same arm; every such redefinition in this file is
    character-identical, and if one ever stops being so the figure must not pick one.
    """
    v = {m.group(1) for m in re.finditer(
        r'^\s*ARMS\[%s\]="([^"]*)"' % re.escape(name), _src(CAMPAIGN), re.M)}
    assert len(v) == 1, f"ARMS[{name}] is defined {len(v)} different ways: {sorted(v)}"
    return v.pop()


def _flags(name):
    """(mode, reg_weight, lr, {FLAG: value}) -- submit_arm's own reading of the spec."""
    t = _spec(name).split()
    return t[0], t[1], t[2], dict(x.split("=", 1) for x in t[3:])


def _arm_feat(name):
    """The encoder feature an arm runs with. See the module docstring's SUBTLETY note."""
    m = re.search(r'ARM_FEAT\[%s\]="([^"]*)"' % re.escape(name), _src(CAMPAIGN))
    if m:
        return m.group(1)
    from analysis.collect_evals import ArmNameError, resolve_arm
    try:
        return "patch" if resolve_arm(_archive()["arms"], name).endswith("_patch") else "cls"
    except ArmNameError:
        return "cls"


def _diff(a, b):
    """The flags of `a` that `b` does not set to the same value, as "KEY=VALUE" strings.

    This is what makes "single-factor" a checked property rather than an intention: a
    contrast is single-factor iff this list has one entry, the first three fields agree and
    the feature agrees.
    """
    fa, fb = _flags(a)[3], _flags(b)[3]
    return [f"{k}={v}" for k, v in fa.items() if fb.get(k) != v]


def _single_factor(a, b):
    """(is it single-factor, the one flag that differs) for the pair (a, b)."""
    d, r = _diff(a, b), _diff(b, a)
    same = _flags(a)[:3] == _flags(b)[:3] and _arm_feat(a) == _arm_feat(b)
    return (len(d) == 1 and not r and same), (d[0] if d else NDASH)


def _defined_arms():
    """Every LITERAL arm name run_campaign.sh defines, and the ones setting DECODE_PRED_W.

    Two names in that file are built from shell variables (wave5's width sweep), so they name
    no single arm and are dropped rather than drawn as if they did.
    """
    src = _src(CAMPAIGN)
    all_ = {m.group(1) for m in re.finditer(r'^\s*ARMS\[([^\]]+)\]="', src, re.M)}
    pdw = {m.group(1) for m in re.finditer(
        r'^\s*ARMS\[([^\]]+)\]="[^"]*DECODE_PRED_W', src, re.M)}
    return sorted(a for a in all_ if "${" not in a), sorted(pdw)


# --- local primitives --------------------------------------------------------------
def effrow(x, y, label, e, key, size=12, gap=15):
    """One paired effect from the archive, with the house's NULL encoding applied by the
    INTERVAL and not by the point estimate.

    An interval that spans zero gets slate, a HOLLOW marker and the word NULL, whatever its
    mean looks like -- the encoding round7-design.svg uses.  Both effects these figures carry
    are NULLs, and drawing either of them in an identity hue would be a claim the archive
    does not support.
    """
    null = e["lo"] <= 0 <= e["hi"]
    k = "slate" if null else key
    s = mark(x, y - 4, k, r=5.5, fill=(WHITE if null else None))
    body = ((f"{label}   " if label else "")
            + f"{_signed(e['mean'])} [{_signed(e['lo'])}, {_signed(e['hi'])}]"
            + f"   {DOT}   n = {e['n']}" + (f"   {DOT}   NULL" if null else ""))
    return s + txt(x + gap, y, body, size, anchor="start", fill=ACCENT[k], weight="bold")


def gradback(x1, x2, y, drop, key, label="", lx=None):
    """A gradient, drawn returning RIGHT to LEFT under the forward chain it belongs to.

    Every other wire in these figures is a forward dataflow, so a gradient drawn forwards
    would read as one.  It gets its own corridor `drop` units below the chain and lands its
    arrowhead on the chain's left-hand block.
    """
    s = apoly([(x1, y), (x1, y + drop), (x2, y + drop), (x2, y + 2)], key, dash="5,4")
    if label:
        s += txt(lx if lx is not None else (x1 + x2) / 2, y + drop + 18, label, 11.5,
                 fill=MUTED)
    return s


def fan(x, y, tips, key, r=4):
    """One source applied to many targets: a drawn dot() and a spray of arrows.

    ST2's whole first claim is that ONE operator is applied to every token, and a single
    arrow into a token grid would say "the operator produces the grid" instead.
    """
    s = dot(x, y, key, r=r)
    for tx, ty in tips:
        s += aw(x - r, y, tx, ty, key, w=1.5)
    return s


def seedrow(x, y, states, key, span=146, gap=3, cell=14.0):
    """round 7's `slots`, sized so a row of n seeds never widens past `span`.

    Not a fixed cell: the campaign is actively adding seeds 11-14 to PiWM-patchdecode
    (run_campaign.sh, S4), and at the house's 14-unit cell an n = 12 row runs 51 units into
    the label beside it.  The row is read from the archive, so its WIDTH has to be too.
    """
    n = max(len(states), 1)
    return slots(x, y, states, key, cell=round(min(cell, (span - gap * (n - 1)) / n), 1),
                 gap=gap)


def lagrow(x, y, offsets, key, cell=94, gap=10, h=34, label="", chip=""):
    """One lag configuration: which frame each of the num_hist slots reads.

    The slots are drawn as a ROW OF FRAMES rather than as a formula because the skip's whole
    content is that two slots read the SAME frame -- a repeated label is visible, a repeated
    subscript inside a sum is not.
    """
    s = ""
    if label:
        s += txt(x - 12, y + h / 2 + 5, label, 12.5, anchor="end", fill=ACCENT[key],
                 weight="bold")
    for i, d in enumerate(offsets):
        cx = x + i * (cell + gap)
        s += mbox(cx, y, cell, h, "z" + sub("", "t" if not d else "t" + MINUS + str(d), 11),
                  key, size=15)
    if chip:
        s += badge(x + len(offsets) * (cell + gap) + 6, y + 6, chip, key, size=11)
    return s


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


# ================================================= S1: the pixel gradient into the predictor
S1_TITLE = "(S1) PiWM-pdpred -- a pixel gradient into the PREDICTOR  [LAUNCHED]"
S1_CTRL, S1_ARM = "PiWM-patchdecode", "PiWM-pdpred"


def s1_pdpred():
    """The two decoder paths, drawn distinctly: one decodes a frame the model has seen and
    stops at the encoder, the other decodes a frame it has NOT seen onto that frame's real
    pixels and runs into the dynamics."""
    K, SYS = "amber", "blue"           # the new term, and the path as built
    b = base(_check_title(S1_TITLE, 880))
    b += wire_to_mse(K, "plus", "+ w " + DOT + " decode_pred_loss")

    PH = 872
    s = pane(PX, PY, PW, PH, K,
             "module:  L  +=  decode_pred_w " + DOT + " || Dec( z"
             + sub("", "pred", 12) + " ) " + MINUS + " o" + sub("", "t+1", 12) + " ||"
             + SQ + "   " + NDASH + "   the target is the DATASET")

    # -- row A, left: the term that exists.  The chain is drawn forwards and the GRADIENT
    # comes back on its own corridor underneath it, so the two directions never share a wire.
    s += frame(60, 950, 404, 236, "the term that EXISTS  " + NDASH + "  decode z"
               + sub("", "emb", 11), SYS,
               note="a frame the model has already SEEN")
    s += mbox(80, 1012, 104, 46, "z" + sub("", "emb", 11), SYS, size=16,
              sub_="frame t  " + DOT + "  SEEN")
    s += aw(184, 1035, 206, 1035, SYS)
    s += mbox(206, 1012, 92, 46, "Dec", SYS, size=16, sub_="patch head")
    s += aw(298, 1035, 320, 1035, SYS)
    s += mbox(320, 1012, 124, 46, "pixels o" + sub("", "t", 11), SYS, size=15,
              sub_="the real frame")
    s += gradback(382, 132, 1058, 26, SYS, label="the gradient, backwards", lx=257)
    s += mbox(80, 1114, 150, 44, "Enc " + sub("", "f", 12), SYS, size=16, sub_="REACHED")
    s += aw(232, 1136, 288, 1136, "crit")
    s += strikeout(260, 1136, "crit", r=10, w=2.2, gap=True)
    s += mbox(292, 1114, 152, 44, "Pred " + sub("", "g", 12), "slate", size=16,
              sub_="NOT reached")
    s += txt(262, 1176, "the pixel gradient stops at the encoder", 12.5,
             fill=ACCENT[SYS], weight="bold")

    # -- row A, right: the term S1 adds.  Same chain, same head, one frame later -- so the
    # ONLY thing that differs between the two halves is WHICH code is decoded.
    s += frame(476, 950, 400, 236, "the term S1 ADDS  " + NDASH + "  decode z"
               + sub("", "pred", 11), K,
               note="a frame the model has NOT seen, on that frame's real pixels")
    s += mbox(496, 1012, 120, 46, "z" + sub("", "pred", 11), K, size=16,
              sub_="frame t+1  " + DOT + "  UNSEEN")
    s += aw(616, 1035, 638, 1035, K)
    s += mbox(638, 1012, 92, 46, "Dec", K, size=16, sub_="the SAME head")
    s += aw(730, 1035, 752, 1035, K)
    s += mbox(752, 1012, 106, 46, "pixels o" + sub("", "t+1", 10), K, size=13,
              sub_="frame t+1")
    s += gradback(804, 548, 1058, 26, K, label="the gradient, backwards", lx=676)
    s += mbox(496, 1114, 150, 44, "Pred " + sub("", "g", 12), K, size=16, sub_="REACHED")
    s += aw(648, 1136, 688, 1136, K)
    s += mbox(692, 1114, 166, 44, "lags.0 / 1 / 2", K, size=15,
              sub_="the dynamics themselves")
    s += txt(676, 1176, "the first term whose gradient enters the DYNAMICS", 12.5,
             fill=ACCENT[K], weight="bold")

    # -- row B: why it never happened.  Three blocks, because the reason is a three-step
    # consequence and a sentence cannot be pointed at.  Every line number is found at render
    # time, and the ORDER of the last two is asserted rather than assumed.
    ln_adaln, ln_disp = _srcline(r"def _forward_adaln"), _srcline(
        r"return self\._forward_adaln\(obs, act\)")
    ln_det = _srcline(r"self\.decode\(\s*(z_pred\.detach\(\))", group=1)
    assert ln_det > ln_disp, "the detached decode no longer sits below the adaln dispatch"
    n_all, n_pdw = _defined_arms()
    A = _archive()
    pdw_runs = sum(_arm(a)[0] for a in n_pdw)
    s += frame(60, 1198, 816, 144, "why no run ever did this  " + NDASH + "  the only other "
               "z_pred decode is unreachable AND detached", "crit",
               note="models/visual_world_model.py, every line number below found at "
                    "render time")
    s += mbox(78, 1258, 232, 54, "_forward_adaln  :" + str(ln_adaln), "crit", size=15,
              sub_="action_conditioning: " + _adaln_everywhere() + " in both configs")
    s += aw(310, 1285, 330, 1285, "crit")
    s += mbox(334, 1258, 232, 54, "return  :" + str(ln_disp), "crit", size=15,
              sub_="so forward() below it never runs")
    s += aw(566, 1285, 586, 1285, "crit")
    s += mbox(590, 1258, 268, 54, "decode( z_pred.detach() )  :" + str(ln_det), "crit",
              size=13, sub_="unreachable, and detached anyway")
    s += txt(468, 1330, "of the " + str(len(n_all)) + " arms defined in run_campaign.sh, "
             + str(len(n_pdw)) + " set DECODE_PRED_W  " + NDASH + "  and between them they "
             "have " + str(pdw_runs) + " finished evals", 12.5, fill=ACCENT["crit"],
             weight="bold")

    # -- row C: the verification.  17 slots, 11 filled, measured by one backward pass.
    R = _reach()
    s += frame(60, 1354, 816, 136, "VERIFIED at render time  " + NDASH + "  decode_pred_loss "
               "alone reaches " + str(len(R["reached"])) + " of " + str(R["total"])
               + " predictor parameters", K,
               note="one backward pass through the real predictor and the real "
                    "PatchHead")
    s += txt(84, 1412, "the predictor's " + str(R["total"]) + " parameter tensors", 11.5,
             anchor="start", fill=MUTED)
    s += cells(84, 1420, [1] * len(R["reached"]) + [0] * len(R["missed"]), K,
               cell=18, gap=2)
    s += mbox(600, 1412, 250, 46, str(len(R["reached"])) + " of " + str(R["total"]), K,
              size=21, sub_="reached by decode_pred_loss alone")
    s += txt(84, 1464, "reached:  " + R["reached_lab"], 12, anchor="start",
             fill=ACCENT[K], weight="bold")
    s += txt(84, 1482, "the " + str(len(R["missed"])) + " it misses are ltv's zero-"
             "initialised " + R["missed_lab"] + " " + NDASH + " the LATENT loss misses the "
             "same " + str(len(R["missed"])) + ("" if R["same_as_latent"] else " ?"), 11.5,
             anchor="start", fill=MUTED)

    # -- row D: the control, already on disk.  The seed rows are countable and the "single
    # factor" claim is the launcher's own diff, not a reading of it.
    n_c, m_c, sd_c = _arm(S1_CTRL)
    n_b, m_b, _ = _arm(BASE_ARM)
    ok, only = _single_factor(S1_ARM, S1_CTRL)
    s += frame(60, 1502, 816, 140, "the CONTROL is already on disk  " + NDASH
               + "  single-factor and free", SYS,
               note=("the two ARMS[ ] lines differ in exactly one flag:  " + only)
                    if ok else ("they differ in more than one flag: " + " , ".join(
                        _diff(S1_ARM, S1_CTRL))))
    s += txt(84, 1562, S1_CTRL + "  " + DOT + "  the control", 12.5, anchor="start",
             fill=ACCENT[SYS], weight="bold")
    s += seedrow(84, 1570, ["f"] * n_c, SYS)
    s += txt(238, 1582, "seeds " + str(sd_c[0]) + NDASH + str(sd_c[-1]) + "  " + DOT
             + "  n = " + str(n_c), 12, anchor="start", fill=INK)
    s += txt(84, 1602, S1_ARM + "  " + DOT + "  the new arm", 12.5, anchor="start",
             fill=ACCENT[K], weight="bold")
    s += seedrow(84, 1610, ["n"] * n_c, K)
    s += txt(238, 1622, "the SAME seeds, launched", 12, anchor="start", fill=MUTED)
    s += mbox(556, 1556, 148, 54, f"{m_c:.3f}", SYS, size=22,
              sub_=S1_CTRL + "  " + DOT + "  n = " + str(n_c))
    s += mbox(714, 1556, 144, 54, f"{m_b:.3f}", "slate", size=22,
              sub_=BASE_ARM + "  " + DOT + "  n = " + str(n_b))
    s += txt(707, 1630, "arm means on the fixed instrument, NOT a contrast", 11,
             fill=MUTED)

    s += frow(1666, [
        (["decode_pred_loss  =  || Dec( z" + sub("", "pred", 12) + " ) " + MINUS + " o"
          + sub("", "t+1", 12) + " ||" + SQ], INK, "bold"),
        (["z_pred[ : , i ] predicts frame num_pred + i , so obs[ : , num_pred : ] is the "
          "exact pixel target"], INK, "normal")])
    s += frow(1690, [
        ("both arms build the identical PatchHead, a bare nn.Linear with zero extra RNG "
         "draws", INK, "normal"),
        ("proj_dim stays " + _conf_scalar("embed_dim") + " , so no width confound", MUTED,
         "normal", 0.92)])
    s += strip(PIN, 1706, [
        ("the control on disk", "n = " + str(n_c) + " , seeds " + str(sd_c[0]) + NDASH
         + str(sd_c[-1]), False),
        ("arm means, not a contrast", f"{m_c:.3f} vs {m_b:.3f}", False),
        ("params the term reaches", str(len(R["reached"])) + " of " + str(R["total"]),
         False),
        ("runs with a pixel gradient in g", str(pdw_runs) + " of " + str(A["n_runs"]),
         True)], K, cw=170)
    return b + s, PY + PH + 40


# ================================================== ST2: the state carried PER TOKEN
ST2_TITLE = "(ST2) PiWM-st-ssm -- a state per TOKEN, not one operator for all  [LAUNCHED]"
ST2_ARM, ST2_ABL_T, ST2_ABL_S = "PiWM-st-ssm", "PiWM-ssm", "PiWM-columns"


def st2_ssm():
    """One broadcast operator against 256 per-token states, with both ablations drawn."""
    K, SYS, ABL = "amber", "blue", "magenta"
    P, D = _num_patches(), int(_conf_scalar("embed_dim"))
    b = base(_check_title(ST2_TITLE, 880))
    b += wire_to_pred(K, "a state per token")

    PH = 888
    s = pane(PX, PY, PW, PH, K,
             "module:  s" + sub("", "t", 12) + "(p)  =  A s" + sub("", "t" + MINUS + "1", 12)
             + "(p) + B" + sub("", "z", 12) + " z" + sub("", "t", 12) + "(p) + B"
             + sub("", "a", 12) + " a" + sub("", "t", 12) + "   for every one of the P = "
             + str(P) + " tokens")

    # -- row A: the deficit, measured.  The two parameter counts are the whole argument, so
    # they are built at render time and set side by side with the equals sign drawn between.
    p_fir_1, p_fir_p = _params("ltv", 1), _params("ltv", P)
    s += frame(60, 950, 816, 248, "the predictor as built  " + NDASH + "  ONE operator, "
               "broadcast over every token", SYS,
               note="the same LinearDynamicsPredictor at num_patches = 1 and at num_patches "
                    "= " + str(P))
    s += tokengrid(170, 1012, P, SYS, total=P)
    s += txt(170, 1104, "P = " + str(P) + " patch tokens", 11.5, fill=MUTED)
    s += mbox(268, 1022, 156, 56, "A" + sub("", "k", 12) + " ,  B ,  W", SYS, size=17,
              sub_="ONE shared operator")
    s += fan(264, 1050, [(212, 1024), (212, 1050), (212, 1076)], SYS)
    s += mbox(468, 1014, 176, 56, _fmt(p_fir_1), SYS, size=22, sub_="params at P = 1")
    s += txt(658, 1050, "=", 20, weight="bold")
    s += mbox(672, 1014, 186, 56, _fmt(p_fir_p), SYS, size=22,
              sub_="params at P = " + str(P))
    s += txt(663, 1104, "the token axis buys NO dynamics", 12.5, fill=ACCENT["crit"],
             weight="bold")
    s += txt(468, 1148, "every token is pushed forward by the SAME operator " + NDASH
             + " a token's dynamics are not its own", 13, weight="bold")
    s += txt(468, 1172, "and no state is carried at all: the memory is a "
             + str(_num_hist()) + "-frame slice of the INPUT", 11.5, fill=MUTED)

    # -- row B: the intervention.  The state's SHAPE is captured from the running module,
    # so "one state per token" is a measurement and not a reading of the docstring.
    shp = _state_shape(P)
    p_iir_1, p_iir_p = _params("ssm", 1), _params("ssm", P)
    s += frame(60, 1210, 816, 222, "ST2  " + NDASH + "  one state per TOKEN, carried across "
               "time", K,
               note="the state tensor's shape, captured at render time from a forward hook "
                    "on A")
    s += tokengrid(150, 1274, P, K, total=P)
    s += txt(150, 1366, "each token: its own s", 11.5, fill=MUTED)
    s += mbox(228, 1272, 196, 56, "( B , P , D )  =  ( " + " , ".join(map(str, shp)) + " )",
              K, size=13, sub_="the state, measured")
    s += mbox(438, 1272, 176, 56, _fmt(shp[1] * shp[2]), K, size=22,
              sub_="state numbers per sample")
    s += mbox(628, 1272, 230, 56, _fmt(p_iir_p), K, size=22,
              sub_="params at P = 1 AND at P = " + str(P))
    s += txt(540, 1352, "z" + sub("", "t+1", 11) + "(p)  =  W ReLU( C s" + sub("", "t", 11)
             + "(p) )   ,   one scan of " + str(_num_hist()) + " steps, per token", 12.5)
    s += txt(468, 1392, "the operator is STILL shared " + NDASH + " what becomes per-token "
             "is the STATE", 13, fill=ACCENT[K], weight="bold")
    s += txt(468, 1414, "no new code: _trunk's ssm branch already carries the P axis, so "
             "ST2 is T1 with FEATURE=patch", 11.5, fill=MUTED)

    # -- row C: the two ablations, each holding ONE axis.  A win is attributable only if
    # both of these run, which is exactly what patchdecode's win did not have.
    n_s, _, _ = _arm(ST2_ABL_T)
    n_c, m_c, _ = _arm(ST2_ABL_S)
    same_spec = _spec(ST2_ARM) == _spec(ST2_ABL_T)
    s += frame(60, 1444, 404, 196, "ABLATION  " + NDASH + "  no spatial axis", ABL,
               note=ST2_ABL_T + "  " + DOT + "  feature " + _arm_feat(ST2_ABL_T) + "  "
                    + DOT + "  P = 1")
    s += mbox(84, 1512, 150, 50, "P = 1", ABL, size=18,
              sub_="one state, " + str(D) + " numbers")
    s += mbox(248, 1512, 196, 50, _spec(ST2_ABL_T), ABL, size=15,
              sub_="ARMS[" + ST2_ABL_T + "] , verbatim")
    s += txt(84, 1586, ("the SAME spec as ST2's " + NDASH + " only ARM_FEAT differs")
             if same_spec else "the specs differ, not only ARM_FEAT", 12, anchor="start",
             fill=ACCENT[ABL], weight="bold")
    s += txt(84, 1606, "n = " + str(n_s) + " on the fixed instrument"
             + ("  " + DOT + "  in flight" if not n_s else ""), 11.5, anchor="start",
             fill=MUTED)

    e_c = _effect(BASE_ARM, ST2_ABL_S)
    s += frame(476, 1444, 400, 196, "ABLATION  " + NDASH + "  the same tokens, no state",
               ABL, note=ST2_ABL_S + "  " + DOT + "  " + str(P) + " patch tokens  " + DOT
                    + "  a " + str(_num_hist()) + "-tap FIR")
    s += mbox(500, 1512, 150, 50, "n = " + str(n_c), ABL, size=18,
              sub_="mean " + f"{m_c:.3f}")
    s += mbox(664, 1512, 192, 50, str(_num_hist()) + "-tap FIR", ABL, size=17,
              sub_="no state at all")
    s += txt(500, 1572, "the contrast against " + BASE_ARM, 11.5, anchor="start",
             fill=MUTED)
    s += effrow(508, 1592, "", e_c, ABL, size=11.5, gap=13)
    s += txt(500, 1612, "a NULL: the interval spans zero", 11.5, anchor="start", fill=MUTED)

    s += frow(1680, [
        (["s" + sub("", "t", 12) + "(p) = A s" + sub("", "t" + MINUS + "1", 12) + "(p) + B"
          + sub("", "z", 12) + " z" + sub("", "t", 12) + "(p) + B" + sub("", "a", 12)
          + " a" + sub("", "t", 12)], INK, "bold"),
        (["the FIR modes carry no state: their memory is num_hist frames of INPUT"], INK,
         "normal")])
    s += frow(1704, [
        ("removing either axis destroys the mechanism, which is what makes it ONE arm",
         INK, "normal"),
        ("patch_size stays at " + _enc_scalar("patch_size") + ", the only granularity "
         "round 7 leaves standing", MUTED, "normal", 0.92)])
    s += strip(PIN, 1720, [
        ("params at P = 1 and P = " + str(P), _fmt(p_fir_p), True),
        ("the state ST2 adds", _fmt(shp[1] * shp[2]) + " numbers", False),
        ("no spatial axis", ST2_ABL_T + " , P = 1", False),
        ("no state", ST2_ABL_S + " , n = " + str(n_c), False)], K, cw=168)
    return b + s, PY + PH + 40


# ============================================ ST3: per-token pixels, and the lag skip
ST3_TITLE = "(ST3) PiWM-st-pdpred -- next-frame pixels per token, and a lag skip  [LAUNCHED]"
ST3_ARM, ST3_S, ST3_T = "PiWM-st-pdpred", "PiWM-pdpred", "PiWM-lagskip"


def st3_pdpred():
    """The joint arm as two halves that each run on their own, and the three single-factor
    contrasts that let a result be charged to one of them."""
    K, SYS = "amber", "blue"
    P = _num_patches()
    lags = tuple(int(v) for v in _flags(ST3_ARM)[3]["LAG_DILATION"].split(","))
    unif = tuple(range(_num_hist()))
    b = base(_check_title(ST3_TITLE, 880))
    b += wire_to_mse(K, "plus", "+ w " + DOT + " decode_pred_loss")

    PH = 848
    s = pane(PX, PY, PW, PH, K,
             "module:  z" + sub("", "pred", 12) + "(p) from lags " + str(list(lags))
             + "  " + ARROWC + "  Dec  " + ARROWC + "  the NEXT frame's real pixels, patch "
             "for patch")

    # -- row A: the two halves, side by side and the same size, because each one RUNS as its
    # own arm.  Drawing the joint arm bigger would say the halves are illustrations.
    s += frame(60, 950, 404, 270, "half S  " + NDASH + "  the pixels, per token", K,
               note="z_pred(p) decoded onto the NEXT frame's patch p")
    s += tokengrid(126, 1020, P, K, total=P)
    s += txt(126, 1112, "z" + sub("", "pred", 10) + " , per token", 11.5, fill=MUTED)
    s += aw(172, 1056, 196, 1056, K)
    s += mbox(196, 1032, 88, 48, "Dec", K, size=15, sub_="patch head")
    s += aw(284, 1056, 308, 1056, K)
    s += mbox(308, 1032, 136, 48, "o" + sub("", "t+1", 11), K, size=16, sub_="real pixels")
    s += txt(262, 1142, "patch p  " + ARROWC + "  the same patch, one step later", 12,
             fill=ACCENT[K], weight="bold")
    s += mbox(84, 1148, 190, 50, ST3_S, K, size=15,
              sub_="uniform lags " + str(list(unif)))
    s += badge(284, 1156, "ABLATION", K, size=11)
    n_pc, m_pc, _ = _arm("PiWM-patchdecode")
    n_b, m_b, _ = _arm(BASE_ARM)
    s += txt(84, 1210, "control: PiWM-patchdecode  " + DOT + "  n = " + str(n_pc)
             + " , mean " + f"{m_pc:.3f}", 11.5, anchor="start", fill=MUTED)

    s += frame(476, 950, 400, 270, "half T  " + NDASH + "  the lag set", K,
               note="slot k reads z" + sub("", "t" + MINUS + "d", 10) + sub("", "k", 9)
                    + " :  " + str(list(lags)) + " instead of " + str(list(unif)))
    s += lagrow(572, 1024, unif, SYS, cell=88, gap=8, h=36, label="as built")
    s += lagrow(572, 1080, lags, K, cell=88, gap=8, h=36, label="the SKIP")
    gone = sorted(set(unif) - set(lags))
    kept = sorted({d for d in lags if lags.count(d) > 1})
    s += txt(676, 1142, " , ".join("z" + sub("", "t" + MINUS + str(d), 10) for d in gone)
             + " is dropped " + NDASH + " the freed slot goes to "
             + " , ".join("z" + sub("", "t" + MINUS + str(d), 10) for d in kept), 12,
             fill=ACCENT[K], weight="bold")
    s += mbox(500, 1148, 190, 50, ST3_T, K, size=15, sub_="no pixel gradient")
    s += badge(700, 1156, "ABLATION", K, size=11)
    s += txt(500, 1210, "control: " + BASE_ARM + "  " + DOT + "  n = " + str(n_b)
             + " , feature " + _arm_feat(ST3_T), 11.5, anchor="start", fill=MUTED)

    # -- row B: the joint arm is the UNION of the two halves' flags, checked here rather
    # than asserted: only the flags that differ between the three specs are drawn.
    VARY = ("DECODE_PRED_W", "LAG_DILATION")
    shared = sorted(set(_flags(ST3_S)[3]) & set(_flags(ST3_ARM)[3]) - set(VARY))
    s += frame(60, 1232, 816, 160, "the JOINT arm  " + NDASH + "  ST3 is exactly the two "
               "halves at once", K,
               note="the " + str(len(VARY)) + " flags that VARY across the three rows; "
                    "the other " + str(len(shared)) + " decoder flags are identical "
                    "wherever they appear")
    for i, (arm, note) in enumerate(((ST3_S, "pixels, uniform lags"),
                                     (ST3_T, "the skip, no pixels"),
                                     (ST3_ARM, "both halves at once"))):
        yb = 1288 + i * 32
        joint = arm == ST3_ARM
        s += txt(216, yb + 17, arm, 13, anchor="end", fill=ACCENT[K] if joint else INK,
                 weight="bold")
        for col, fl in enumerate(VARY):
            val = _flags(arm)[3].get(fl)
            if val is None:
                continue
            s += badge(230 + col * 172, yb + 2, fl + "=" + val,
                       K if joint else "slate", size=11)
        s += txt(560, yb + 17, note, 12, anchor="start",
                 fill=ACCENT[K] if joint else MUTED, weight="bold" if joint else "normal")

    # -- row C: attribution.  "Single factor" is a CHECKED property of the two launcher
    # lines, so each row says which flag it is and what a difference there charges.
    s += frame(60, 1404, 816, 196, "ATTRIBUTION  " + NDASH + "  three single-factor "
               "contrasts", K,
               note="checked here: the two ARMS[ ] lines must differ in exactly one flag, "
                    "at the same feature")
    for i, (arm, ctrl, charge) in enumerate((
            (ST3_S, "PiWM-patchdecode", "charges the PIXEL half"),
            (ST3_ARM, ST3_S, "charges the LAG half"),
            (ST3_T, BASE_ARM, "the skip alone, on cls"))):
        y = 1468 + i * 32
        ok, only = _single_factor(arm, ctrl)
        s += txt(84, y, arm + "   vs   " + ctrl, 12.5, anchor="start", weight="bold")
        s += txt(360, y, ("differs in 1 flag:  " + only) if ok
                 else ("differs in " + str(len(_diff(arm, ctrl))) + " flags"), 11.5,
                 anchor="start", fill=MUTED)
        s += txt(626, y, charge, 12, anchor="start", fill=ACCENT[K], weight="bold")
    e_pd = _effect("PiWM-patchdecode-detach", "PiWM-patchdecode")
    s += effrow(90, 1560, "the precedent " + NDASH + "  patchdecode vs its detach control",
                e_pd, K, size=11.5)
    s += txt(84, 1582, "a win with no ablation is uninterpretable " + NDASH + " which is "
             "why both halves run on their own", 11.5, anchor="start", fill=MUTED)

    s += frow(1640, [
        (["ST3  =  DECODE_PRED_W=" + _flags(ST3_ARM)[3]["DECODE_PRED_W"] + "   +   "
          "LAG_DILATION=" + _flags(ST3_ARM)[3]["LAG_DILATION"]], INK, "bold"),
        (["the pixel target is the DATASET, so it cannot be gamed by degrading it"], INK,
         "normal")])
    s += frow(1664, [
        ("either half can be charged for the result, because both of them run",
         INK, "normal"),
        ("the skip is DDP-safe: " + str(_grad_none(lags)) + " params with grad-is-None, "
         "the same as " + str(_grad_none(unif)) + " for the uniform set", MUTED,
         "normal", 0.92)])
    s += strip(PIN, 1680, [
        ("the joint arm", "pixels + lag skip", False),
        ("half S runs alone", ST3_S, False),
        ("half T runs alone", ST3_T, False),
        ("controls on disk", "n = " + str(n_pc) + " and n = " + str(n_b), False)],
        K, cw=170)
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
    ("arch-s1-pdpred.svg", s1_pdpred),
    ("arch-st2-ssm.svg", st2_ssm),
    ("arch-st3-pdpred.svg", st3_pdpred),
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
