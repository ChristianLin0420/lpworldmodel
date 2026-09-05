"""PiWM-loo5 -- leave-one-out plan-time consensus, and the knob it removes.

    python analysis/arch_loo_fig.py --out diary/assets/2026-09-05
    -> arch-loo.svg    the committee chosen by the data, and the bar its gain has to clear

WHAT THE ARM IS.  Purely an EVAL-time construction: `scripts/plan_vote_loo.sh` loads M already
trained checkpoints, `planning/ensemble.py` stacks them, and
`planning.objectives.create_vote_objective_fn` combines their opinions.  No term is added to
any training loss, so the connector into the schematic is the house route for exactly that --
`round6_arch_figs.wire_from_pred`, "this proposal reads the model and changes no loss".  For
episode block b the committee is every finished seed EXCEPT b, so the MODEL SET is chosen by
the data instead of by a hyperparameter, and the rule is fixed to median-of-ranks, which is
scale-free (members never share a latent basis -- every run trains its own encoder) and
defined at every M.

WHY IT EXISTS, AND WHY THAT IS THE PICTURE.  The vote{M}-{rule} grid it replaces had TWO
knobs and between them measured one thing.  The seed sets the episode BLOCK, not the models,
so the committee is a STEP FUNCTION of the block -- and at the committee level, which is the
unit of independence for a claim about consensus, the headline arm has k = 2.  Row A draws
that step function from the eval logs themselves rather than asserting it.

WHAT THE ARCHIVE SAYS THAT THE PLAN DID NOT.  The design says leave-one-out gives one
committee per block, so k = n.  The emitted runs do not: M was capped to 5 by GPU memory and
members are taken "first M in order", so every block from 5 upward draws the SAME five
members and 16 blocks emit 6 distinct committees, not 16.  That is read off the same log
headers as everything else in row A and is drawn in crimson next to the design it qualifies.
The figure would be a lie without it, and the cap is real: `PiWM-loo11-columns` (M = 11,
patch) died 12/12 with `torch.cuda.OutOfMemoryError`.

AND THE CONTROL THAT DECIDES IT.  `PiWM-solo{m}` is checkpoint s{m}, unchanged, planning on
its family's OTHER blocks -- the only place in the archive where model quality and block
difficulty are unconfounded.  Their per-model means span a wide band, and the top of that
band reaches the committees.  Row C draws the band as the bar the consensus gain has to
clear, because a committee that only beats the AVERAGE member is a way of not picking a bad
seed, not a method.

STYLE.  analysis/style.py, through analysis/arch_figs.py.  Every drawing primitive is
imported from analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py /
round6_arch_figs.py / round7_arch_figs.py -- and NO EXISTING MODULE IS EDITED, because every
other architecture-figure module in this directory imports the same primitives, so an edit
there is an edit to all of them.  Hues by IDENTITY, never by rank:

    teal     the consensus family -- loo5 and the vote arms, the same key
             arch_figs._consensus, round6 R5, round7 S3 and round7_v2 G2 all use
    crimson  the defect and the retraction: the two-knob sweep, the two committees that ten
             "seeds" really were, the 6 committees that 16 leave-one-out blocks really are,
             and the OOM that capped M
    purple   the CONTRASTING CONDITION -- PiWM-solo, one lone unchanged checkpoint
    slate    neutral -- an episode block, and the LpWM-ltv single-model baseline
    green    reserved here for nothing; the baseline is context, not "the system as built"
    amber    reserved here for nothing.  This figure tests no training intervention, so no
             element may claim amber's identity.

LAYOUT RULES INHERITED FROM round6_arch_figs.py / round7_v2_arch_figs.py, and the defect each
one stops

  * base() fixes every anchor, so a connector uses ONE named route.  This one attaches where
    round7_v2 G2 attaches -- out of the predicted code stack, down the right margin to the
    panel -- via `wire_from_pred`, which hops the two wires it crosses.  That route passes
    the title's baseline at y = 872, so the title is WIDTH-CHECKED against the corridor
    (`_check_title`) rather than trusted.
  * Nothing is sized by hand where it can be measured: mbox / pill / inset / frow / strip all
    fit their own text, and `_fitrun` shrinks a laid-out row of numbers to its column.
  * A NULL is never drawn like a positive.  An interval that spans zero gets a HOLLOW marker,
    whatever its point estimate looks like -- which is how row A's committee-level column
    reads at a glance, and it is the honest reading of loo5 at k = 3.
  * `frow()` stops at its 12.0 pt floor and OVERFLOWS SILENTLY, so a footer line here carries
    at most two clauses and the audit checks the result rather than the intent.
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of every
    <text> element from the same Helvetica metrics the layout used (arch_figs.text_width), and
    reports any pair that intersects, anything that leaves the canvas, and anything that
    crosses the border of the panel or inset it sits in (every frame registers itself through
    the `frame()` / `pane()` wrappers).  At 940 x 1930 units "nothing overlaps" is not
    checkable by eye, and the frame check is not redundant with the canvas check -- a label
    centred on half an inset can run past that inset's border while sitting comfortably
    inside the canvas.

Usage:  python analysis/arch_loo_fig.py --out diary/assets/2026-09-05 [--png]
        `--png` rasterises beside the SVG, for the read-back; the audit runs either way and
        main() exits non-zero if it has anything to say.

THE NUMBERS, AND WHERE EACH ONE COMES FROM.  Nothing here is a typed-in literal; every figure
in the drawing is read at render time or is arithmetic on something that was.

  read from the eval archive, analysis/collect_evals.collect(scheme="fixed")
    the per-arm mean and n of PiWM-loo5-ltv, PiWM-vote5-median, PiWM-vote3-median,
      PiWM-vote5-borda and the LpWM-ltv baseline
    PiWM-solo{m} for every solo arm on record: its mean over the other blocks of its
      family, and hence the band those means span
    ALWAYS through resolve_arm -- a patch arm is keyed `<name>_patch` and `arms.get(name)`
      returns an empty dict that is indistinguishable from "not evaluated yet"
  read from slurm_logs/, the `label=... rule=... seed=... members=...` header every
  vote / loo planning job prints as its first act
    the committee each episode block actually used, for all four consensus arms; hence the
      number of DISTINCT committees, and hence k
    the rule each arm ran, checked to be single-valued
    the member count M, checked to be constant within an arm
    the 12 PiWM-loo11-columns jobs and the OutOfMemoryError in every one of their .err files
  conf/encoder/vit_scratch*.yaml and conf/train.yaml
    patch_size 14, img_size 224, and feature cls / patch
  arithmetic done here, and nothing else
    num_patches = (img_size // patch_size) ** 2, models/vit_encoder.py:65-67 verbatim, and
      num_patches = 1 when feature == "cls" (line 72) -- so a patch member costs 256 stacked
      slots where a cls member costs 1, and M x 256 is what the OOM is measuring
    the paired contrast against LpWM-ltv at BLOCK level (one observation per block) and at
      COMMITTEE level (one observation per distinct member set), each a mean, an sd and a
      t_{df} interval from analysis.figures.t_crit -- the same statistic paired_effect uses,
      re-aggregated rather than re-derived
    the ratio of the two interval widths, and the count of committees the best lone model
      outscores
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    _marker, arrow, base, domeup, poly, sub, text_width, title_line, txt,
)
from analysis.arch_figs_causal import (  # noqa: E402,F401
    ARROWC, DOT, IMPLIES, MINUS, NDASH, TIMES,
    apoly, aw, badge, fit_size, mbox, pill, strip,
)
from analysis.round5_figs_obj import (  # noqa: E402,F401
    frow, inset, panel, strikeout,
)
from analysis.round6_arch_figs import (  # noqa: E402,F401
    PIN, PIX, PW, PX, PY, SHIFT, W, ellipsis_v, mark, out_emit, wire_from_pred,
)
from analysis.round7_arch_figs import cbadge  # noqa: E402,F401
from analysis.collect_evals import collect, resolve_arm  # noqa: E402
from analysis import figures as FG  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-05")
PLAN_OUT = os.path.join(REPO, "plan_outputs")
SLURM = os.path.join(REPO, "slurm_logs")
ENC_CLS = os.path.join(REPO, "conf", "encoder", "vit_scratch.yaml")
ENC_PATCH = os.path.join(REPO, "conf", "encoder", "vit_scratch_patch.yaml")
TRAIN_CFG = os.path.join(REPO, "conf", "train.yaml")

BASE_ARM = "LpWM-ltv"
LOO_ARM = "PiWM-loo5-ltv"
OOM_ARM = "PiWM-loo11-columns"
VOTE_ARMS = ("PiWM-vote5-median", "PiWM-vote3-median", "PiWM-vote5-borda")


# --- the archive, read at render time ----------------------------------------------
def yaml_field(path, key):
    """The value of a TOP-LEVEL `key: value` line, comments stripped.

    A line reader, not a YAML load, for the same reason arch_columns_fig gives: a loader
    resolves `${img_size}` into a value that is not what is written in the file, and the
    claim here is about what the config SAYS.
    """
    with open(path) as f:
        for line in f:
            if line[:1] in (" ", "\t", "-"):
                continue
            line = line.split("#", 1)[0].rstrip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if k.strip() == key:
                return v.strip()
    raise KeyError(f"no top-level {key!r} in {path}")


IMG = int(yaml_field(TRAIN_CFG, "img_size"))
PATCH_SIZE = int(yaml_field(ENC_PATCH, "patch_size"))
FEAT_CLS = yaml_field(ENC_CLS, "feature")
FEAT_PATCH = yaml_field(ENC_PATCH, "feature")
assert (FEAT_CLS, FEAT_PATCH) == ("cls", "patch"), (FEAT_CLS, FEAT_PATCH)


def tokens_for(patch_size, feature="patch"):
    """models/vit_encoder.py:65-72, verbatim.

        grid = image_size // patch_size ; num_patches = grid * grid
        self.num_patches = 1 if feature == "cls" else num_patches

    This is the whole of the memory cap: `planning/ensemble.py` stacks members on the PATCH
    axis, so one member costs num_patches slots.
    """
    return 1 if feature == "cls" else (IMG // patch_size) ** 2


PATCH_SLOTS = tokens_for(PATCH_SIZE, FEAT_PATCH)     # 256
CLS_SLOTS = tokens_for(PATCH_SIZE, FEAT_CLS)         # 1

#: the header every planning job in scripts/vote_slurm.sbatch prints before it loads a thing
_HDR = re.compile(r"label=(?P<label>\S+)\s+rule=(?P<rule>\w+)\s+seed=(?P<seed>\d+)\s+"
                  r"members=(?P<members>\S+)")


def committee_map(arm, head=8):
    """{block: (member seed, ...)} and the set of rules, from the eval logs.

    The membership is NOT recoverable from plan_outputs/ -- that directory holds logs.json
    and the rollout frames and nothing about which checkpoints were loaded -- so the only
    record of what each committee actually was is this header line.  Reading it is what
    turns "the seed sets the block, not the models" from a claim into a measurement.
    """
    out, rules, ms = {}, set(), set()
    for f in sorted(glob.glob(os.path.join(SLURM, f"eval_{arm}_pd*_s*.out"))):
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i >= head:
                    break
                m = _HDR.search(line)
                if not m:
                    continue
                seeds = tuple(sorted(int(x.rsplit("_s", 1)[-1])
                                     for x in m.group("members").split(",")))
                b = int(m.group("seed"))
                assert out.get(b, seeds) == seeds, (
                    f"{arm} block {b} has two different member sets on record: "
                    f"{out[b]} and {seeds}")
                out[b], _ = seeds, rules.add(m.group("rule"))
                ms.add(len(seeds))
                break
    assert out, f"no `members=` header for {arm} under {SLURM}"
    assert len(ms) == 1, f"{arm} did not run at one M: {sorted(ms)}"
    return out, sorted(rules), ms.pop()


def oom_jobs(arm):
    """(jobs launched, jobs whose stderr carries a CUDA OOM) for an arm."""
    errs = sorted(glob.glob(os.path.join(SLURM, f"eval_{arm}_pd*_s*.err")))
    bad = 0
    for f in errs:
        with open(f, errors="replace") as fh:
            if "torch.cuda.OutOfMemoryError" in fh.read():
                bad += 1
    return len(errs), bad


def ci(vals):
    """(mean, lo, hi, half-width, n) at 95%, on analysis.figures' own t table.

    Deliberately the same statistic figures.paired_effect uses -- t_{n-1} on the sd of the
    observations -- applied to a different UNIT.  Re-deriving it would eventually disagree
    with the panels.
    """
    x = np.asarray(list(vals), dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, np.nan, n
    se = float(x.std(ddof=1)) / np.sqrt(n)
    h = FG.t_crit(n - 1) * se
    return m, m - h, m + h, h, n


def two_levels(arms, arm, ctrl=BASE_ARM):
    """The paired contrast against the baseline at BLOCK level and at COMMITTEE level.

    The block-level row is what the campaign reports: one observation per episode block.
    The committee-level row averages the blocks that shared a member set and treats each
    distinct set as one observation, which is the unit of independence for a claim about
    consensus -- if two blocks were scored by the SAME five models, they are not two
    samples of "does a committee help".
    """
    comm, _, _ = committee_map(arm)
    a, c = arms[resolve_arm(arms, arm)], arms[resolve_arm(arms, ctrl)]
    blocks = sorted(int(b) for b in set(a) & set(c) if int(b) in comm)
    d = [float(a[str(b)]) - float(c[str(b)]) for b in blocks]
    g = {}
    for b, dd in zip(blocks, d):
        g.setdefault(comm[b], []).append(dd)
    # k over the PAIRED blocks, and k over all of them.  They differ, and the difference
    # is not cosmetic: paired_effect drops a block the baseline has no fixed-scheme eval
    # on, and dropping a block drops its committee with it -- loo5 emits 6 committees and
    # the contrast can only be computed on 3 of them.  Reporting one number would have
    # made the row disagree with the badge above it for no visible reason.
    return ci(d), ci([float(np.mean(v)) for v in g.values()]), len(g), len(set(comm.values()))


def arm_mean(arms, name):
    """(mean, n) over an arm's fixed-scheme seeds, through resolve_arm."""
    v = np.array([float(x) for x in arms[resolve_arm(arms, name)].values()])
    return float(v.mean()), len(v)


def solo_means(arms):
    """{m: (mean, n)} for every PiWM-solo<m> arm on record.

    scripts/plan_consensus_controls.sh section B: checkpoint s<m>, UNCHANGED, planned on
    its family's other blocks.  It is the only place in the archive where checkpoint
    quality and block difficulty are not perfectly confounded, so it is the only honest
    "best single member" this campaign has.
    """
    out = {}
    for k in arms:
        m = re.fullmatch(r"PiWM-solo(\d+)", k)
        if m:
            out[int(m.group(1))] = arm_mean(arms, k)
    assert out, "no PiWM-solo<m> arms on record"
    return out


# --- local primitives --------------------------------------------------------------
def hbrace(x0, x1, y, key, w=1.5, tick=7):
    """A horizontal brace under a run of block chips: `these blocks are ONE committee`."""
    c = ACCENT[key]
    return (f'<path d="M{x0},{y - tick} L{x0},{y} L{x1},{y} L{x1},{y - tick}" '
            f'stroke="{c}" stroke-width="{w}" fill="none" stroke-linejoin="round" '
            f'stroke-linecap="round"/>\n')


def vbrace(x, y0, y1, key, w=1.6, tick=9):
    """A vertical brace: `these rows, together, are one statement`."""
    c = ACCENT[key]
    return (f'<path d="M{x - tick},{y0} L{x},{y0} L{x},{y1} L{x - tick},{y1}" '
            f'stroke="{c}" stroke-width="{w}" fill="none" stroke-linejoin="round" '
            f'stroke-linecap="round"/>\n')


def seedrow(x, y, seeds, taken, left_out, key, cell=13, gap=3):
    """One leave-one-out committee, drawn as three DIFFERENT states, not two.

      solid    a member: one of the first M eligible seeds, in order
      hollow   eligible but not taken -- present in n - 1, cut by the M cap
      crossed  the seed left out, a different one for every block

    Two states would have drawn a committee of n - 1 = 15 and hidden the entire reason
    the realised design has 6 distinct committees instead of 16.  The cap IS the finding,
    so the cap has to be visible in the row.
    """
    s = ""
    for i, sd in enumerate(seeds):
        cx = x + i * (cell + gap)
        if sd == left_out:
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{WHITE}" stroke="{ACCENT["crit"]}" stroke-width="1.4"/>\n')
            s += strikeout(cx + cell / 2, y + cell / 2, "crit", r=cell / 2 - 1.0, w=1.7)
        elif sd in taken:
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{ACCENT[key]}" fill-opacity="0.78" stroke="{ACCENT[key]}" '
                  f'stroke-width="1"/>\n')
        else:
            s += (f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                  f'fill="{WHITE}" stroke="{DIM}" stroke-width="1.1" '
                  f'stroke-dasharray="3,2"/>\n')
    return s


def runs_of(blocks, comm):
    """[(i0, i1)] -- the maximal runs of consecutive blocks that share a member set.

    A run, not a group: the defect being drawn is that membership is a STEP FUNCTION of
    the block, and a step function is exactly a partition into consecutive runs.  Grouping
    non-adjacent blocks would draw a brace over blocks that are not under it.
    """
    out = []
    for i, b in enumerate(blocks):
        if out and comm[b] == comm[blocks[i - 1]] and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def members_label(seeds):
    """`s3 – s7`, or `s0 – s5 , not s1`, or a comma list -- whichever is true and shortest."""
    lo, hi = min(seeds), max(seeds)
    full = set(range(lo, hi + 1))
    got = set(seeds)
    if got == full:
        return f"s{lo} {NDASH} s{hi}"
    missing = sorted(full - got)
    if len(missing) == 1:
        return f"s{lo} {NDASH} s{hi} , not s{missing[0]}"
    return " , ".join(f"s{s}" for s in sorted(seeds))


def blockmap(x, y, blocks, comm, key, cell=30, h=24, gap=4, size=12):
    """One arm's blocks as chips, braced into the committees they actually selected.

    Returns (svg, right edge, [(cx, width, member seeds)] per run) so the caller can label
    the runs that are wide enough and account for the ones that are not.
    """
    pitch = cell + gap
    s = ""
    for i, b in enumerate(blocks):
        s += mbox(x + i * pitch, y, cell, h, str(b), "slate", size=size)
    spans = []
    for i0, i1 in runs_of(blocks, comm):
        x0, x1 = x + i0 * pitch, x + i1 * pitch + cell
        s += hbrace(x0, x1, y + h + 7, key)
        spans.append(((x0 + x1) / 2.0, x1 - x0, comm[blocks[i0]]))
    return s, x + (len(blocks) - 1) * pitch + cell, spans


def _fitrun(x, y, parts, size, gap, fill=INK, weight="normal", maxx=None):
    """Lay out a row of labelled numbers left to right, shrinking until it fits `maxx`.

    frow() justifies a FOOTER between two margins; this is the inner-row counterpart, and
    it exists because these rows carry an interval whose width depends on the data -- a
    hand-set column position is a defect waiting for the next eval to land.
    """
    sz = float(size)
    while maxx is not None and sz > 8.5:
        w = sum(text_width(p, sz, weight) for p in parts) + gap * (len(parts) - 1)
        if x + w <= maxx:
            break
        sz -= 0.25
    s, cx = "", float(x)
    for p in parts:
        s += txt(round(cx, 1), y, p, round(sz, 2), anchor="start", fill=fill, weight=weight)
        cx += text_width(p, sz, weight) + gap
    return s, round(cx - gap, 1)


def vaxis(x0, x1, y, v0, v1, ticks, key="slate", size=11):
    """A bare value axis -- one rule, ticks below it, no box.  Recessive furniture."""
    c = ACCENT[key]
    s = (f'<path d="M{x0},{y} L{x1},{y}" stroke="{c}" stroke-width="1.3" '
         f'stroke-opacity="0.8"/>\n')
    for v in ticks:
        tx = round(x0 + (v - v0) / (v1 - v0) * (x1 - x0), 1)
        s += (f'<path d="M{tx},{y} L{tx},{y + 5}" stroke="{c}" stroke-width="1.2" '
              f'stroke-opacity="0.8"/>\n')
        s += txt(tx, y + 18, f"{v:.1f}", size, fill=MUTED)
    return s


def ival(y, lo, hi, mid, key, hollow, xmap, h=6):
    """One interval on a shared axis.  A null and a positive are drawn DIFFERENTLY.

    Copied in spirit from round7_arch_figs.interval(): an interval that spans zero gets a
    HOLLOW marker whatever its point estimate looks like, so the eye cannot read a wide
    null as a narrow positive.
    """
    c = ACCENT[key]
    xl, xh, xm = xmap(lo), xmap(hi), xmap(mid)
    s = (f'<path d="M{xl:.1f},{y} L{xh:.1f},{y} M{xl:.1f},{y - h} L{xl:.1f},{y + h} '
         f'M{xh:.1f},{y - h} L{xh:.1f},{y + h}" stroke="{c}" stroke-width="2" '
         f'fill="none" stroke-linecap="round"/>\n')
    return s + mark(round(xm, 1), y, key, r=5.5, fill=(WHITE if hollow else None))


def band(x0, x1, y, h, key, opacity=0.30):
    """The span a set of single models covers -- a range, not an estimate, so no marker."""
    return (f'<rect x="{round(x0,1)}" y="{y}" width="{round(x1-x0,1)}" height="{h}" rx="3" '
            f'fill="{ACCENT[key]}" fill-opacity="{opacity}" stroke="{ACCENT[key]}" '
            f'stroke-width="1.4"/>\n')


# Every frame a label must stay inside, registered as it is drawn.  The audit's canvas
# check is not enough on its own: a label centred on half an inset can run past that
# inset's border and still sit comfortably inside the canvas.
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
    """The title is one centred run at y = 872 and the connector drops past it, so its
    half-width is a HARD constraint, not a matter of taste."""
    body = re.sub(r"<[^>]*>", "", title)
    wide = text_width(body.replace(" -- ", " – "), size, "bold")
    right = cx + wide / 2
    assert right < corridor_x - clear, (
        f"title runs to x={right:.0f}, into the connector corridor at x={corridor_x}: "
        f"shorten it")
    return title


def _check_pane(title, x=PX, w=PW, clear=9):
    """The PANEL title, checked at the weight it is actually drawn in.

    round5_figs_obj.panel() picks its size with `fit_size(title, 18, w - 44)` -- whose
    default weight is NORMAL -- and then draws the result BOLD, which is about 5.6% wider
    on these faces.  A title that "fits" by that measure can therefore cross the panel's
    own border, and the first draft of this figure did, by 21 units.  So the panel title
    is width-checked here for the same reason `_check_title` checks the figure title:
    the primitive's guarantee does not cover the case, and the audit is not the place to
    discover it.
    """
    size = fit_size(title, 18, w - 44, floor=14)
    right = x + 20 + text_width(title, size, "bold")
    assert right <= x + w - clear, (
        f"panel title is drawn bold at {size} and runs to x={right:.0f}, past the panel "
        f"border at x={x + w}: shorten it")
    return title


# ============================================================== the figure
TITLE = ("(loo) PiWM-loo5 " + NDASH + " the committee is chosen by the data  [EVALUATED]")
PANE_TITLE = ("module:  committee( b ) = every finished seed EXCEPT b  " + NDASH
              + "  M = min ( n " + MINUS + " 1 , 5 ) , median of ranks")

# The inset's inner text margins.  The audit rejects a label within 9 units of the frame it
# sits in, so these are the real edges and every fitted line is measured against them.
IL, IR = 78, 867
# row A: the block-chip maps.  Chips start clear of the arm labels on their left, and the
# committee-count badges sit in a fixed column so three rows of different length line up.
AX, ACELL, AGAP, ABADGE = 232, 26, 3, 790
# row C: the success-rate axis.  Left of CX0 is the arm-label column, right of CX1 the value.
CX0, CX1, CV0, CV1, CVAL = 250, 792, 0.0, 0.8, 862


def fitline(x, y, s, size=11.5, maxx=IR, fill=INK, weight="normal", floor=9.0):
    """A start-anchored line, shrunk until it fits its column. See `_fitrun`."""
    return txt(x, y, s, fit_size(s, size, maxx - x, floor=floor, weight=weight),
               anchor="start", fill=fill, weight=weight)


def fitmid(cx, y, s, size=12, maxw=IR - IL, fill=INK, weight="normal", floor=9.0):
    """The centred counterpart of fitline()."""
    return txt(cx, y, s, fit_size(s, size, maxw, floor=floor, weight=weight), fill=fill,
               weight=weight)


def loo_fig():
    """Leave-one-out consensus: the knob it removes, the cap it hit, the bar it must clear."""
    K = "green"                     # teal -- the consensus family, as in round5/6/7
    arms, _, _ = collect(plan_outputs=PLAN_OUT, scheme="fixed")

    loo_comm, loo_rules, loo_M = committee_map(LOO_ARM)
    _, _, oom_M = committee_map(OOM_ARM)
    oom_n, oom_bad = oom_jobs(OOM_ARM)
    blocks = sorted(loo_comm)
    pool = sorted({s for v in loo_comm.values() for s in v} | set(blocks))
    # scripts/plan_vote_loo.sh emits exactly one committee per FINISHED seed and labels it
    # with that seed, so the block set IS the finished-seed set.  The row calls the slots
    # "finished seeds", which is only true if that holds -- so it is checked, not assumed.
    assert set(pool) == set(blocks), \
        f"a member sits outside the block set: {sorted(set(pool) - set(blocks))}"
    n_loo, k_loo = len(blocks), len(set(loo_comm.values()))

    b = base(_check_title(TITLE, 880))
    b += wire_from_pred(K, "eval only " + NDASH + " no term in the loss")

    PH = 1072
    s = pane(PX, PY, PW, PH, K, _check_pane(PANE_TITLE))
    s += _row_a(arms, K)
    s += _row_b(loo_comm, blocks, pool, loo_M, loo_rules, k_loo, n_loo, oom_M, oom_n,
                oom_bad, K)
    s += _row_c(arms, K)

    s += frow(1856, [
        (["committee( b )  =  { finished seeds }  \\  { b } ,  first M"], INK, "bold"),
        (["score( c )  =  median" + sub("", "m", 12) + "  rank" + sub("", "m", 12)
          + "( c )  within the CEM batch"], INK, "normal")])
    s += frow(1880, [
        ("median-of-ranks is scale-free, so members need not share a latent basis",
         INK, "normal"),
        ("nothing here enters a training loss", MUTED, "normal", 0.95)])
    s += strip(PIN, 1896, [
        ("knobs removed", "M , and the rule", False),
        ("blocks " + ARROWC + " committees", f"{n_loo} " + ARROWC + f" {k_loo}", True),
        ("M = " + str(oom_M) + " patch", f"{oom_bad} / {oom_n} OOM", True),
        ("the bar to clear", "the best lone model", True)], K, cw=176)
    return b + s, PY + PH + 40


def _row_a(arms, K):
    """The knob that measured nothing: membership as a step function of the block."""
    s = frame(60, 950, 816, 330,
              "the knob that measured nothing  " + NDASH + "  the committee is a STEP "
              "FUNCTION of the episode block", "crit",
              note="vote { 2 , 3 , 5 , 8 , 12 }  " + TIMES + "  { borda , median , cvar1 , "
                   "cvar2 , max } :  two knobs, and the seed sets the BLOCK, not the models")
    s += fitline(IL, 1022, "what each seed actually selected, read back from the eval logs",
                 12.5, fill=ACCENT["crit"], weight="bold")

    rows = [(a, "crit") for a in VOTE_ARMS[:2]] + [(LOO_ARM, K)]
    loose = 0
    for i, (arm, key) in enumerate(rows):
        comm, _, _ = committee_map(arm)
        blocks = sorted(comm)
        y = 1042 + i * 48
        s += txt(AX - 12, y + 16, arm, 12.5, anchor="end", fill=ACCENT[key], weight="bold")
        s += txt(AX - 12, y + 32, f"{len(blocks)} episode blocks", 10.5, anchor="end",
                 fill=MUTED)
        gsvg, _, spans = blockmap(AX, y, blocks, comm, key, cell=ACELL, gap=AGAP)
        s += gsvg
        for cx, wid, mem in spans:
            lab = members_label(mem)
            if text_width(lab, 10.5) <= wid - 4:
                s += txt(cx, y + 42, lab, 10.5, fill=ACCENT[key])
            else:
                # a brace one block wide cannot carry `s0 – s5 , not s1`, and a label that
                # overhangs its own brace would attach to the neighbouring committee.  The
                # unlabelled ones are counted and named below instead of being ignored.
                assert arm == LOO_ARM, f"{arm} has a brace too narrow for its members"
                loose += 1
        s += cbadge(ABADGE, y + 1, f"{len(set(comm.values()))} distinct", key, size=11.5)
    s += txt(CVAL, 1042 + 2 * 48 + 42,
             f"the {loose} unlabelled braces are one block each", 10.5, anchor="end",
             fill=MUTED)

    # the consequence, at the two levels.  The marker carries the null, not the number.
    s += fitline(IL, 1210, "and what that does to the interval  " + NDASH + "  the same "
                 "paired contrast against " + BASE_ARM + " , re-aggregated onto its unit "
                 "of independence", 12.5, fill=ACCENT["crit"], weight="bold")
    for i, (arm, key) in enumerate(rows):
        (bm, blo, bhi, bh, bn), (cm, clo, chi, ch, _), k, ktot = two_levels(arms, arm)
        y = 1232 + i * 20
        s += txt(AX - 12, y, arm, 11.5, anchor="end", fill=ACCENT[key])
        s += mark(AX + 6, y - 4, key, r=4.5, fill=(WHITE if blo <= 0 <= bhi else None))
        run, _ = _fitrun(AX + 18, y, [
            "block  " + _iv(bm, blo, bhi) + f"  n = {bn}",
            ARROWC,
            "committee  " + _iv(cm, clo, chi)
            + (f"  k = {k}" if k == ktot else f"  k = {k} of {ktot}")], 11.5, 14, maxx=744)
        s += run
        s += mark(758, y - 4, key, r=4.5, fill=(WHITE if clo <= 0 <= chi else None))
        s += txt(CVAL, y, f"{TIMES} {ch / bh:.1f} width", 11.5, anchor="end",
                 fill=(ACCENT["crit"] if ch > bh else INK), weight="bold")
    return s


def _iv(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", MINUS)


def _row_b(comm, blocks, pool, M, rules, k, n, oom_M, oom_n, oom_bad, K):
    """The design, and the cap the emitted runs actually ran under."""
    s = frame(60, 1296, 816, 232,
              "leave-one-out  " + NDASH + "  the members are set by the data, the rule is "
              "fixed", K,
              note="filled = a member  " + DOT + "  dashed = eligible but past the M cap  "
                   + DOT + "  crossed = the seed left out, a different one every block")
    s += txt(220, 1360, f"{LOO_ARM}  " + DOT + f"  {len(pool)} finished seeds  " + DOT
             + f"  rule = {rules[0]}", 11.5, anchor="start", fill=MUTED)
    for i, bl in enumerate((blocks[0], blocks[1], blocks[-1])):
        y = 1370 + i * 26 + (18 if i == 2 else 0)
        s += txt(210, y + 11, f"block b = {bl}", 12, anchor="end", fill=MUTED)
        s += seedrow(220, y, pool, set(comm[bl]), bl, K)
    s += ellipsis_v(346, 1420, "slate", n=2, r=2.2, gap=9)
    s += vbrace(492, 1370, 1453, K)
    s += aw(492, 1410, 518, 1410, K)
    s += mbox(522, 1386, 150, 48, rules[0], K, size=18, sub_="of the per-member ranks")
    s += aw(672, 1410, 696, 1410, K)
    s += pill(740, 1410, "CEM", K, rx=42, ry=20)
    s += txt(740, 1452, "argsort( loss )[ : topk ]", 11, fill=MUTED)
    s += txt(596, 1370, "no rule knob", 12, fill=ACCENT[K], weight="bold")

    s += fitline(IL, 1476, "M = " + str(oom_M) + " ( = n " + MINUS + " 1 ) does not fit:  "
                 f"{oom_bad} of {oom_n} " + OOM_ARM + " jobs died with "
                 "torch.cuda.OutOfMemoryError", 11.5, fill=ACCENT["crit"], weight="bold")
    s += fitline(IL, 1496, "members stack on the PATCH axis, so a patch member costs "
                 f"{PATCH_SLOTS} slots where a cls member costs {CLS_SLOTS}  " + ARROWC
                 + f"  M = {oom_M} patch = {oom_M * PATCH_SLOTS:,} slots".replace(",", " "),
                 11.5, fill=MUTED)
    s += fitline(IL, 1518, f"so the emitted runs are M = {M} , and 'first M in order' gives "
                 f"every block from {sorted(comm)[M]} up the SAME members  " + ARROWC
                 + f"  {k} distinct committees, not {n}", 11.5, fill=ACCENT["crit"],
                 weight="bold")
    return s


def _row_c(arms, K):
    """The bar the gain has to clear: one unchanged checkpoint on somebody else's block."""
    solos = solo_means(arms)
    lo = min(v[0] for v in solos.values())
    hi = max(v[0] for v in solos.values())
    best = max(solos, key=lambda m: solos[m][0])
    n_solo = min(v[1] for v in solos.values())

    s = frame(60, 1544, 816, 290,
              "the bar the gain has to clear  " + NDASH + "  one unchanged checkpoint, on "
              "somebody else's block", K,
              note="PiWM-solo m :  checkpoint s m , nothing changed, planned on its "
                   "family's other blocks  " + DOT + "  final_eval/success_rate")

    def xm(v):
        return CX0 + (v - CV0) / (CV1 - CV0) * (CX1 - CX0)

    base_m, base_n = arm_mean(arms, BASE_ARM)
    s += txt(CX0 - 12, 1622, BASE_ARM, 12, anchor="end", fill=ACCENT["slate"],
             weight="bold")
    s += txt(CX0 - 12, 1637, "one model, no committee", 10.5, anchor="end", fill=MUTED)
    s += mark(round(xm(base_m), 1), 1618, "slate", r=6)
    s += txt(CVAL, 1622, f"{base_m:.3f}   n = {base_n}", 11.5, anchor="end", fill=MUTED)

    s += band(xm(lo), xm(hi), 1651, 18, "magenta")
    for m, (mu, _) in sorted(solos.items()):
        s += mark(round(xm(mu), 1), 1660, "magenta", r=4.5, fill=WHITE)
    s += txt(CX0 - 12, 1664, f"PiWM-solo {min(solos)} " + NDASH + f" {max(solos)}", 12,
             anchor="end", fill=ACCENT["magenta"], weight="bold")
    s += txt(CX0 - 12, 1679, f"{len(solos)} lone models  " + DOT + f"  n = {n_solo} each",
             10.5, anchor="end", fill=MUTED)
    s += txt(CVAL, 1664, f"{lo:.3f} " + NDASH + f" {hi:.3f}", 11.5, anchor="end",
             fill=ACCENT["magenta"], weight="bold")

    comm_arms = [LOO_ARM] + list(VOTE_ARMS)
    beaten = 0
    for i, a in enumerate(comm_arms):
        mu, n = arm_mean(arms, a)
        y = 1704 + i * 21
        s += txt(CX0 - 12, y + 4, a, 12, anchor="end", fill=ACCENT[K],
                 weight="bold" if a == LOO_ARM else "normal")
        # every committee mark is FILLED.  A hollow marker is this project's null encoding
        # and row A spends it on exactly that, so it cannot also mean "an arm other than
        # the figure's".  The subject arm is bold and one radius larger instead.
        s += mark(round(xm(mu), 1), y, K, r=6.5 if a == LOO_ARM else 5)
        s += txt(CVAL, y + 4, f"{mu:.3f}   n = {n}", 11.5, anchor="end",
                 fill=INK if a == LOO_ARM else MUTED)
        beaten += hi >= mu
    s += (f'<path d="M{xm(hi):.1f},1651 L{xm(hi):.1f},1775" '
          f'stroke="{ACCENT["magenta"]}" stroke-width="1.3" stroke-dasharray="5,4" '
          f'stroke-opacity="0.8"/>\n')
    s += vaxis(CX0, CX1, 1786, CV0, CV1, [0.0, 0.2, 0.4, 0.6, 0.8])
    s += fitmid(468, 1826, f"the best lone model (solo{best}, {hi:.3f}) outscores {beaten} "
                f"of the {len(comm_arms)} committees  " + NDASH + "  consensus buys "
                "insurance against a bad seed, not a better model", 12,
                fill=ACCENT["crit"], weight="bold")
    return s


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
    checkable by eye on a 940 x 1930 canvas; this reads the emitted file back and says so.
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
    ("arch-loo.svg", loo_fig),
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
