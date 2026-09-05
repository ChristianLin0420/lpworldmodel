"""PiWM-columns -- the patch-token representation, and the 256x it moves at the same time.

    python analysis/arch_columns_fig.py --out diary/assets/2026-09-05
    -> arch-columns.svg    one config field, and a latent 256 times larger

PiWM-columns is the campaign's only single-factor cls-vs-patch contrast, and it is the arm
the whole "cortical columns" premise rests on.  This figure draws what that one factor
actually does.  `conf/encoder/vit_scratch.yaml` and `conf/encoder/vit_scratch_patch.yaml`
are read at render time and differ on EXACTLY ONE field -- `feature` -- out of sixteen;
`scripts/run_campaign.sh` gives both arms the same spec string; every evaluated run of both
arms carries the same PROJ_D, the same precision and goal_H = 5.  And yet
`models/vit_encoder.py` turns that one field into num_patches = (224 // 14)**2 = 256, so
the latent the predictor and the CEM objective consume goes from 1 x 384 = 384 numbers to
256 x 384 = 98 304.

THE POINT.  A cls-vs-patch comparison built this way changes the FEATURE and the CAPACITY
together, so it cannot separate them -- which is the same defect
`analysis/round7_v2_arch_figs.py` G1 exists to repair, drawn here at its source rather than
at its fix.  The footnote box carries the part the campaign got backwards: columns does
decode block orientation at 9.58 deg, but that is 7th of the NINE arms that beat the
14.51 deg best-constant bound, and FIVE 384-dimensional CLS arms decode it better.  The
capacity argument does not run in the direction the campaign assumed.

STYLE.  analysis/style.py, through analysis/arch_figs.py.  Every drawing primitive is
imported from analysis/arch_figs.py / arch_figs_causal.py / round5_figs_obj.py /
round6_arch_figs.py / round7_arch_figs.py -- nothing here re-implements one, and NO
EXISTING MODULE IS EDITED (every other architecture-figure module in this directory
imports the same primitives, so an edit there is an edit to all of them).  Hues by
IDENTITY, never by rank:

    green    LpWM-ltv -- the system as built, the 384-d cls latent everything else was
             measured on.  The five arms in the footnote are green for the same reason:
             they are 384-d cls arms.
    purple   PiWM-columns -- the contrasting condition, and the patch tokens themselves
    crimson  the defect and the retraction: the 256x the one field also moved, and the
             probe result read in the direction the record actually supports
    slate    neutral -- the shared predictor, the shared CEM, and the NULL
    amber    reserved here for nothing.  This figure tests no intervention; it describes a
             contrast already on the record, so no element may claim amber's identity.
    teal     likewise unused -- no consensus object appears here.

LAYOUT RULES INHERITED FROM round6_arch_figs.py / round7_v2_arch_figs.py, and the defect
each one stops

  * Strict left-to-right dataflow on fixed row baselines; no two wires share a segment.
  * base() fixes every anchor, so a connector uses ONE named route.  `encoder.feature`
    lives INSIDE the encoder, where none of the house routes lands, so this reuses round7
    S1 / round7_v2 G1's encoder route verbatim: start short on the dome's right edge, run
    the empty right margin at x = 890 (clear of the dome at 868 and o_{t+1} at 846), and
    land the arrowhead long on the panel's top edge.  That route passes the title's
    baseline at y = 872, so the title is WIDTH-CHECKED against the corridor
    (`_check_title`) rather than trusted.
  * Nothing is sized by hand where it can be measured: mbox / inset / frow / strip all fit
    their own text and none may cross the panel border.
  * `tokenscale()` fixes the CELL and lets the outer size vary, which is the exact
    opposite of round7_v2_arch_figs.tokenframe().  That is deliberate and the two are not
    interchangeable: tokenframe() fixes the OUTER size because G1's subject is the feature
    and drawing the capacity would have been the confound; here the subject IS the
    capacity, so one square is one token and the 256x is the picture.
  * A NULL is never drawn like a positive.  +0.072 [-0.064, +0.207] spans zero, so it gets
    slate, a HOLLOW marker and the word NULL -- the encoding round7-design.svg and
    round7_v2's G1 both use.
  * A footer line carries at most two clauses: frow() shrinks its gap and then its type,
    but at its 12.0 pt floor it stops and OVERFLOWS SILENTLY.
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of
    every <text> element from the same Helvetica metrics the layout used
    (arch_figs.text_width), and reports any pair that intersects, anything that leaves the
    canvas, and anything that crosses the border of the panel or inset it sits in (every
    frame registers itself through `frame()` / `pane()`).  Copied from
    analysis/round7_v2_arch_figs.py rather than imported, so that module stays a dated
    statement and is not edited.

Usage:  python analysis/arch_columns_fig.py --out diary/assets/2026-09-05 [--png]
        `--png` rasterises beside the SVG, for the read-back; the audit runs either way
        and main() exits non-zero if it has anything to say.

EVERY NUMBER IN THE FIGURE, AND WHERE IT IS READ FROM AT RENDER TIME.  Nothing below is a
literal typed into a label; each is either parsed out of a file in this repo when the
figure is drawn, or arithmetic done here on such a value.

  read from the repo
    16 encoder fields, 1 of them differing, and that one is `feature: cls | patch`
                                       conf/encoder/vit_scratch{,_patch}.yaml, parsed
    the arm spec "ltv 1.0 5e-4", identical for both arms
                                       scripts/run_campaign.sh   ARMS[...] lines
    PROJ_D 384 and PRECISION bf16      scripts/run_campaign.sh   ${VAR:-default}
    img_size 224, patch_size 14        conf/encoder/vit_scratch_patch.yaml + conf/train.yaml
    goal_H = 5 on every evaluated run of both arms, and the two run-name stems differing
      by the single token `patch`      plan_outputs/, via collect_evals.collect(detail)
    CEM success: PiWM-columns 0.418 at n = 12, LpWM-ltv 0.357 at n = 13
                                       collect_evals.collect(scheme="fixed"), ALWAYS
                                       through resolve_arm -- the patch arm is keyed
                                       "PiWM-columns_patch"
    the paired contrast +0.072 [-0.064, +0.207] at n = 12
                                       figures.paired_effect, on the 12 shared seeds
    9.58 deg, the 14.51 deg constant bound, the rank 7-of-9, the 46 arms probed, and
      linvar 3.52 / lie-sim 4.91 / multact 4.97 / ctrb 5.13 / lie 5.64
                                       assets/latent_probe*.json, through
                                       round7_figs.probe_rows (which DEDUPLICATES by run --
                                       the shards overlap and a naive read doubles every n)
    n_feat 384 and 98 304, the probe's own record of the latent width
                                       the same rows
    811 840 predictor parameters at P = 1 and at P = 256
                                       models.infojepa_modules.LinearDynamicsPredictor,
                                       instantiated twice at the config's own num_hist and
                                       rank; equal because that class never reads
                                       num_patches (it appears once, in the signature)

  arithmetic done here, and nothing else
    num_patches = (img_size // patch_size)**2, models/vit_encoder.py:65-67 verbatim
    the two latent sizes 1 x 384 and 256 x 384, and their ratio 98304 / 384 = 256
    the rank of PiWM-columns among the arms whose median beats the constant bound, and the
      count of 384-dim cls arms ahead of it
"""
import argparse
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
    ARROWC, DOT, IMPLIES, MINUS, NDASH, SQ, TIMES,
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
from analysis.round7_figs import const_bound, probe_rows  # noqa: E402

DEG = "°"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-05")

ENC_CLS = os.path.join(REPO, "conf", "encoder", "vit_scratch.yaml")
ENC_PATCH = os.path.join(REPO, "conf", "encoder", "vit_scratch_patch.yaml")
TRAIN_CFG = os.path.join(REPO, "conf", "train.yaml")
LTV_CFG = os.path.join(REPO, "conf", "predictor", "ltv.yaml")
CAMPAIGN = os.path.join(REPO, "scripts", "run_campaign.sh")
PLAN_OUT = os.path.join(REPO, "plan_outputs")
PROBE_GLOB = os.path.join(REPO, "assets", "latent_probe*.json")

BASE_ARM, COLS_ARM = "LpWM-ltv", "PiWM-columns"


# --- the archive, read at render time ----------------------------------------------
def yaml_fields(path):
    """{key: value} for every TOP-LEVEL `key: value` line, comments stripped.

    Deliberately a line reader and not a YAML load: the question is how many FIELDS OF
    THE FILE differ, and a loader would resolve `${img_size}` and `${oc.select:...}` into
    values that are not what is written down.

    Top-level only, and that is not cosmetic: conf/train.yaml sets `num_hist: 3` at the
    root and then passes it down as `num_hist: ${num_hist}` inside the predictor block, so
    a reader that took every indentation level would return the interpolation string and
    silently mis-instantiate the predictor.
    """
    out = {}
    with open(path) as f:
        for line in f:
            if line[:1] in (" ", "\t", "-"):
                continue
            line = line.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def encoder_diff():
    """(n_fields, {field: (cls_value, patch_value)}) for the two encoder configs.

    The claim the figure makes -- "identical in every respect except encoder.feature" --
    is checked here rather than asserted in prose, so a future edit to either file breaks
    the render instead of quietly falsifying the caption.
    """
    a, b = yaml_fields(ENC_CLS), yaml_fields(ENC_PATCH)
    assert set(a) == set(b), f"the two encoder configs do not share a field set: {a} {b}"
    d = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    assert list(d) == ["feature"], f"expected only `feature` to differ, got {sorted(d)}"
    return len(a), d, a


def campaign_spec(arm):
    """The set of distinct `ARMS[<arm>]="..."` strings in scripts/run_campaign.sh."""
    with open(CAMPAIGN) as f:
        src = f.read()
    v = set(re.findall(r'ARMS\[' + re.escape(arm) + r'\]="([^"]*)"', src))
    assert len(v) == 1, f"ARMS[{arm}] is not one string in {CAMPAIGN}: {sorted(v)}"
    return next(iter(v))


def shell_default(var):
    """The default of a `VAR=${VAR:-default}` line in scripts/run_campaign.sh."""
    with open(CAMPAIGN) as f:
        src = f.read()
    m = re.search(r"^" + re.escape(var) + r"=\$\{" + re.escape(var) + r":-([^}]*)\}",
                  src, re.M)
    assert m, f"no {var}=${{{var}:-...}} line in {CAMPAIGN}"
    return m.group(1)


def tokens_for(patch_size, img_size):
    """models/vit_encoder.py:65-67, verbatim: grid = img // patch, num_patches = grid**2."""
    return (img_size // patch_size) ** 2


def cem_facts():
    """Per-arm CEM success, the paired contrast, goal_H, and the run-name difference."""
    arms, detail, _ = collect(plan_outputs=PLAN_OUT, scheme="fixed")
    keys = {a: resolve_arm(arms, a) for a in (BASE_ARM, COLS_ARM)}
    out = {}
    for a, k in keys.items():
        v = np.array([float(x) for x in arms[k].values()])
        runs = [r for r in detail if FG.run_arm(r) == k]
        gh = sorted({detail[r]["goal_H"] for r in runs})
        assert len(gh) == 1, f"{a}: goal_H is not constant over its runs: {gh}"
        out[a] = dict(key=k, n=len(v), mean=float(v.mean()), goal_H=gh[0],
                      runs=sorted(runs))
    # The run names differ by exactly one underscore-separated token.  collect_evals'
    # docstring records that this tag is what keys the patch arm separately, so the
    # figure's "one field" claim has to survive it -- and it does: the extra token is the
    # feature itself.
    def stem(run, k):
        return [t for t in run[len(k):].split("_") if t and not re.fullmatch(r"s\d+", t)]
    a0 = stem(out[BASE_ARM]["runs"][0], BASE_ARM)
    a1 = stem(out[COLS_ARM]["runs"][0], COLS_ARM)
    for r in out[BASE_ARM]["runs"]:
        assert stem(r, BASE_ARM) == a0, f"{BASE_ARM} runs are not one stem: {r}"
    for r in out[COLS_ARM]["runs"]:
        assert stem(r, COLS_ARM) == a1, f"{COLS_ARM} runs are not one stem: {r}"
    extra = [t for t in a1 if t not in a0]
    assert not [t for t in a0 if t not in a1] and len(extra) == 1, \
        f"run-name stems differ by more than one token: {a0} vs {a1}"
    e = FG.paired_effect(arms, keys[BASE_ARM], keys[COLS_ARM])
    return out, e, a0, a1, extra[0]


def probe_facts(cols_dims, base_dims):
    """The orientation probe, per arm: median angular error, n, latent width, and rank.

    probe_rows() deduplicates by run -- the shard files overlap, and concatenating them
    doubles every n without moving a single median.
    """
    rows = probe_rows(PROBE_GLOB)
    bound = const_bound(rows)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append((float(r["err_ang_deg"]), int(r["n_feat"])))
    tab = {}
    for a, vs in by.items():
        nf = sorted({f for _, f in vs})
        assert len(nf) == 1, f"{a}: n_feat is not constant within the arm: {nf}"
        tab[a] = dict(med=float(np.median([v for v, _ in vs])), n=len(vs), n_feat=nf[0])
    # the probe's own record of the latent width is an INDEPENDENT route to the same two
    # numbers the token arithmetic gives, so the two are cross-checked against each other
    assert tab[COLS_ARM]["n_feat"] == cols_dims, \
        f"probe n_feat {tab[COLS_ARM]['n_feat']} != {cols_dims} from num_patches x proj_dim"
    assert tab[BASE_ARM]["n_feat"] == base_dims
    beat = sorted((v["med"], a) for a, v in tab.items() if v["med"] < bound)
    rank = [a for _, a in beat].index(COLS_ARM) + 1
    ahead384 = [(m, a) for m, a in beat[:rank - 1] if tab[a]["n_feat"] == base_dims]
    ahead_other = [(m, a) for m, a in beat[:rank - 1] if tab[a]["n_feat"] != base_dims]
    return tab, bound, beat, rank, ahead384, ahead_other, len(tab)


def predictor_params(proj_dim):
    """Parameter count of the campaign's predictor at P = 1 and at P = 256.

    Instantiated rather than quoted: LinearDynamicsPredictor takes num_patches and never
    reads it (the name occurs once in the class, in the signature), so the operator is one
    set of weights broadcast over the patch axis.  That is the whole reason the figure may
    draw ONE predictor block for both arms.
    """
    from models.infojepa_modules import LinearDynamicsPredictor
    tr, lt = yaml_fields(TRAIN_CFG), yaml_fields(LTV_CFG)
    kw = dict(input_dim=proj_dim, num_frames=int(tr["num_hist"]), mode="ltv",
              rank=int(lt["rank"]))
    n = {p: sum(q.numel() for q in LinearDynamicsPredictor(num_patches=p, **kw).parameters())
         for p in (1, tokens_for(14, 224))}
    assert len(set(n.values())) == 1, f"the predictor is not P-invariant: {n}"
    return next(iter(n.values())), kw["num_frames"], kw["rank"]


def num(v):
    """98304 -> '98 304'.  A thin space group, as every other figure in this set uses."""
    return f"{v:,}".replace(",", " ")


# --- local primitives --------------------------------------------------------------
def tokenside(n, cell=4.6, gap=0.6):
    """The drawn side of an n x n token grid at a fixed cell size."""
    return n * (cell + gap) - gap


def tokenscale(cx, cy, n, key, cell=4.6, gap=0.6, pad=3.0):
    """n x n token slots at a FIXED CELL SIZE, centred on (cx, cy).

    The inverse of round7_v2_arch_figs.tokenframe(), on purpose.  That one fixes the OUTER
    size so a cls latent and a patch latent are the same picture and only the slot count
    differs -- correct there, because its subject is the FEATURE and drawing the capacity
    would have drawn the confound.  Here the subject IS the capacity, so the cell is fixed
    and the outer size carries the 256x: one square is one token, both grids to scale.
    """
    step = cell + gap
    side = tokenside(n, cell, gap)
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    s = (f'<rect x="{x0 - pad:.1f}" y="{y0 - pad:.1f}" width="{side + 2 * pad:.1f}" '
         f'height="{side + 2 * pad:.1f}" rx="2" fill="{WHITE}" stroke="{DIM}" '
         f'stroke-width="1"/>\n')
    for r in range(n):
        for c in range(n):
            s += (f'<rect x="{x0 + c * step:.1f}" y="{y0 + r * step:.1f}" '
                  f'width="{cell}" height="{cell}" fill="{ACCENT[key]}" '
                  f'fill-opacity="0.85" stroke="{ACCENT[key]}" stroke-width="0.4"/>\n')
    return s


def vrule(x, y0, y1, key="crit", w=1.2, opacity=0.45):
    """The divider between two readings that sit side by side inside one inset.  Same
    hairline round7_v2's G1 and G2 both use to split their row A."""
    return (f'<path d="M{x},{y0} L{x},{y1}" stroke="{ACCENT[key]}" stroke-width="{w}" '
            f'stroke-opacity="{opacity}"/>\n')


def leader(x, y0, y1, key="slate", w=0.9):
    """A hairline from a mark to its own caption.  The cls latent is ONE 4.6-unit square,
    which is the point, and without a leader its caption 100 units below reads as
    unattached."""
    return (f'<path d="M{x},{y0} L{x},{y1}" stroke="{ACCENT[key]}" stroke-width="{w}" '
            f'stroke-opacity="0.55" stroke-dasharray="3,3"/>\n')


# Every frame a label must stay inside, registered as it is drawn.  The audit's canvas
# check is not enough on its own: a label centred on the right half of an inset can run
# well past that inset's border and still sit comfortably inside the canvas.
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


# ============================================================ the figure
TITLE = "(C) PiWM-columns -- the patch-token representation  [NULL]"

# Row origins.  EVERY coordinate inside a row is written as `<origin> + offset`, never as
# an absolute number: row A's table gained a sixth line during the build, and with absolute
# coordinates that line landed 14 units BELOW its own inset -- where the audit's frame check
# could not see it, because a label whose centre is outside every inset is assigned to the
# panel, and the panel had room.  Offsets make a row's height and its content one edit.
AY, AH = 950, 214          # A  the config, side by side
BY, BH = 1178, 292         # B  the latent, to scale
CY, CH = 1484, 124         # C  the contrast it buys
DY, DH = 1622, 196         # D  the footnote: the orientation probe
F1, F2, SY = 1846, 1870, 1886


def columns_fig():
    """One config field, a 256x latent, and a contrast that cannot separate the two."""
    K = "magenta"          # purple -- PiWM-columns, the contrasting condition
    KB = "blue"            # green  -- LpWM-ltv, the system as built
    KN = "slate"           # neutral -- the shared predictor, the shared CEM, the NULL

    # ---- the archive
    n_fields, diff, enc = encoder_diff()
    img = int(yaml_fields(TRAIN_CFG)["img_size"])
    patch = int(enc["patch_size"])
    proj_d = int(shell_default("PROJ_D"))
    prec = shell_default("PRECISION")
    spec_b, spec_c = campaign_spec(BASE_ARM), campaign_spec(COLS_ARM)
    assert spec_b == spec_c, f"the two arm specs differ: {spec_b!r} vs {spec_c!r}"
    NP = tokens_for(patch, img)
    D_CLS, D_PAT = 1 * proj_d, NP * proj_d
    RATIO = D_PAT // D_CLS
    assert D_PAT % D_CLS == 0 and RATIO == NP
    cem, eff, stem_b, stem_c, tag = cem_facts()
    tab, bound, beat, rank, ahead384, ahead_other, n_probed = probe_facts(D_PAT, D_CLS)
    npar, nhist, prank = predictor_params(proj_d)
    cls_v, pat_v = diff["feature"]

    # ---- the schematic, with the two encoders redrawn in the feature's own hue
    b = base(_check_title(TITLE, 890), skip=("enc",))
    for x in (64, 756):
        b += domeup(x, 520, 112, 124, "Enc <tspan font-style='italic'>f</tspan>",
                    fill=ACCENT_FILL[K], sub=cls_v + "  |  " + pat_v)
    b += arrow(120, 520, 120, 470)
    b += arrow(812, 520, 812, 470)
    # x=712, not 736: base()'s RDMReg return runs vertically at x=726 from y=442 to 582,
    # and a label ending at 736 crossed it.
    b += txt(712, 502, "encoder.feature  " + cls_v + " " + ARROWC + " " + pat_v
             + "   " + DOT + "   num_patches  1 " + ARROWC + " " + str(NP), 14,
             anchor="end", fill=ACCENT[K], weight="bold")
    # round7 S1 / round7_v2 G1's encoder route: start short on the dome's right edge, run
    # the empty right margin (clear of the dome at 868 and o_{t+1} at 846), land long.
    b += poly([(868, 560), (890, 560), (890, PY)], color=ACCENT[K], w=1.8, dash="6,4")

    PH = SY + 44 + 22 - PY
    s = pane(PX, PY, PW, PH, K,
             "module:  encoder.feature = " + pat_v + "  " + NDASH + "  one field, and the "
             "latent goes " + num(D_CLS) + " " + ARROWC + " " + num(D_PAT) + " numbers")

    # -- row A: the two configs, field by field.  The claim is checkable, so it is checked.
    s += frame(60, AY, 816, AH,
               "the config, side by side  " + NDASH + "  " + str(len(diff)) + " of "
               + str(n_fields) + " encoder fields differs", "crit",
               note="conf/encoder/vit_scratch.yaml  vs  vit_scratch_patch.yaml ,   and "
                    "ARMS[" + BASE_ARM + "] == ARMS[" + COLS_ARM + "] in run_campaign.sh")
    LAB, CLS_X, PAT_X, VER_X = 268, 424, 636, 794
    s += txt(CLS_X, AY + 68, BASE_ARM + "  " + DOT + "  " + cls_v, 13.5,
             fill=ACCENT[KB], weight="bold")
    s += txt(PAT_X, AY + 68, COLS_ARM + "  " + DOT + "  " + pat_v, 13.5,
             fill=ACCENT[K], weight="bold")
    s += txt(VER_X, AY + 68, "same?", 12, fill=MUTED)
    # (label, cls value, patch value, verdict, differs).  The run-name row is the last one
    # because it is a CONSEQUENCE of the field above it: collect_evals keys the patch arm
    # "PiWM-columns_patch", which is why every lookup here goes through resolve_arm.
    ROWS = (
        ("run_campaign.sh spec", spec_b, spec_c, "identical", False),
        ("PROJ_D", str(proj_d), str(proj_d), "identical", False),
        ("PRECISION", prec, prec, "identical", False),
        ("goal_H, every eval", str(cem[BASE_ARM]["goal_H"]),
         str(cem[COLS_ARM]["goal_H"]), "identical", False),
        ("encoder.feature", cls_v, pat_v, "THE ONE DIFFERENCE", True),
        ("run name", "_".join(stem_b), "_".join(stem_c),
         "+ the _" + tag + " tag", True),
    )
    for i, (lab, lv, rv, verdict, differs) in enumerate(ROWS):
        y = AY + 94 + i * 22
        s += txt(LAB, y, lab, 12, anchor="end", fill=MUTED)
        s += txt(CLS_X, y, lv, 12.5, fill=INK if not differs else ACCENT[KB],
                 weight="normal" if not differs else "bold")
        s += txt(PAT_X, y, rv, 12.5, fill=INK if not differs else ACCENT[K],
                 weight="normal" if not differs else "bold")
        s += txt(VER_X, y, verdict, 11.5,
                 fill=MUTED if not differs else ACCENT["crit"],
                 weight="normal" if not differs else "bold")

    # -- row B: the latent, to scale.  One square is one token.
    s += frame(60, BY, 816, BH, "the latent, to scale  " + NDASH + "  one square is one "
               "token, both grids at the same cell size", K,
               note="models/vit_encoder.py:65" + NDASH + "67:   num_patches = "
                    "( img_size // patch_size )" + SQ + "  =  ( " + str(img) + " // "
                    + str(patch) + " )" + SQ + "  =  " + str(NP))
    GTOP, MID = BY + 172, BY + 213
    for cx, n, key, cap in ((170, 1, KB, "1 token"),
                            (330, NP, K, str(NP) + " tokens")):
        g = int(round(n ** 0.5))
        side = tokenside(g)
        s += domeup(cx - 42, BY + 66, 84, 76, "Enc <tspan font-style='italic'>f</tspan>",
                    fill=ACCENT_FILL[key], sub=cls_v if key == KB else pat_v)
        s += arrow(cx, BY + 142, cx, BY + 166)
        s += tokenscale(cx, GTOP + side / 2.0, g, key)
        # the cls latent is ONE 4.6-unit square; without a leader its caption 100 units
        # below reads as unattached to anything
        if side < 20:
            s += leader(cx, GTOP + side + 6, BY + 264)
        s += txt(cx, BY + 276, cap, 12, fill=ACCENT[key], weight="bold")
    # the left arrow starts ON the cls latent's leader, at x = 170: started clear
    # of it, a red arrowhead 39 units below a 4.6-unit square points at nothing
    s += aw(171, MID, 192, MID, "crit")
    s += mbox(196, BY + 192, 72, 42, TIMES + " " + str(RATIO), "crit", size=19,
              sub_="tokens")
    s += aw(272, MID, 282, MID, "crit")
    s += vrule(396, BY + 62, BY + 284)

    s += txt(414, BY + 80, "what the predictor and the CEM objective consume", 13,
             anchor="start", weight="bold")
    for i, (nm, key, tk, tot) in enumerate(((cls_v, KB, 1, D_CLS),
                                            (pat_v, K, NP, D_PAT))):
        y = BY + 108 + i * 22
        s += txt(414, y, nm, 12.5, anchor="start", fill=ACCENT[key], weight="bold")
        s += txt(540, y, str(tk) + " " + TIMES + " " + str(proj_d), 12.5, anchor="start")
        s += txt(690, y, "=   " + num(tot), 12.5, anchor="start", weight="bold")
    s += txt(414, BY + 154, "ratio", 12.5, anchor="start", fill=MUTED)
    s += txt(540, BY + 154, TIMES + " " + str(RATIO), 14, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    s += txt(414, BY + 178, "the probe archive records the same two widths:  n_feat = "
             + num(tab[BASE_ARM]["n_feat"]) + "  and  " + num(tab[COLS_ARM]["n_feat"]),
             11, anchor="start", fill=MUTED)
    # ONE predictor block and ONE CEM block for both arms, in the neutral hue, because
    # LinearDynamicsPredictor never reads num_patches: the operator is the same weights
    # broadcast over the patch axis, which predictor_params() checks by construction.
    s += mbox(414, BY + 208, 112, 48, "P " + TIMES + " " + str(proj_d), KN, size=16,
              sub_="the latent")
    s += aw(526, BY + 232, 550, BY + 232, KN)
    s += mbox(554, BY + 208, 140, 48, "Pred g", KN, size=16, sub_="broadcast over P")
    s += aw(694, BY + 232, 718, BY + 232, KN)
    s += mbox(722, BY + 208, 144, 48, "CEM", KN, size=16, sub_="scores all P tokens")
    s += txt(640, BY + 276, "P = 1 for " + cls_v + " ,   P = " + str(NP) + " for "
             + pat_v + "  " + NDASH + "  the only thing that changes", 12,
             fill=ACCENT["crit"], weight="bold")

    # -- row C: what the contrast bought.  A NULL, drawn as a NULL.
    s += frame(60, CY, 816, CH, "the contrast it buys  " + NDASH + "  a NULL", KN,
               note="paired on the " + str(eff["n"]) + " seeds both arms hold, fixed eval "
                    "scheme  (analysis/figures.paired_effect)")
    for cx, arm, key in ((200, BASE_ARM, KB), (430, COLS_ARM, K)):
        d = cem[arm]
        s += txt(cx, CY + 72, arm + "  " + DOT + "  "
                 + (cls_v if key == KB else pat_v), 12.5, fill=MUTED)
        s += txt(cx, CY + 100, f"{d['mean']:.3f}", 24, fill=ACCENT[key], weight="bold")
        s += txt(cx, CY + 118, "n = " + str(d["n"]), 11.5, fill=MUTED)
    s += vrule(612, CY + 62, CY + 120, KN)
    # slate, HOLLOW, and the word NULL in the strip below: the interval spans zero, and a
    # null is never drawn the way a positive is
    s += mark(636, CY + 86, KN, r=6, fill=WHITE)
    s += txt(652, CY + 91, f"{eff['mean']:+.3f}   [ {eff['lo']:+.3f} , {eff['hi']:+.3f} ]"
             .replace("-", MINUS), 15, anchor="start", fill=ACCENT[KN], weight="bold")
    s += txt(652, CY + 112, "n = " + str(eff["n"]) + "  " + NDASH
             + "  the interval spans zero", 11.5, anchor="start", fill=MUTED)

    # -- row D: the footnote.  The probe result, in the direction the record supports.
    s += frame(60, DY, 816, DH, "footnote  " + NDASH + "  the orientation probe, read the "
               "way the record actually runs", "crit",
               note="ridge-decode the block angle from ONE FROZEN FRAME, median over an "
                    "arm's checkpoints, against the best CONSTANT prediction  "
                    + DOT + "  " + str(n_probed) + " arms probed")
    s += txt(468, DY + 72, COLS_ARM + " decodes at " + f"{tab[COLS_ARM]['med']:.2f}" + DEG
             + "  " + NDASH + "  " + str(rank) + "th of the " + str(len(beat))
             + " arms that beat the " + f"{bound:.2f}" + DEG + " constant bound", 13,
             weight="bold")
    # green, because these are 384-d CLS arms -- the system's own representation, not a
    # rank.  They are drawn at all because the campaign read 9.58 deg as evidence FOR the
    # patch capacity, and five arms without it decode the same variable better.
    CW, CG = 124, 14
    x0 = 60 + (816 - (len(ahead384) * CW + (len(ahead384) - 1) * CG)) / 2.0
    for i, (m, a) in enumerate(ahead384):
        s += mbox(x0 + i * (CW + CG), DY + 84, CW, 44, f"{m:.2f}" + DEG, KB, size=16,
                  sub_=a)
    s += txt(468, DY + 148, str(len(ahead384)) + " CLS arms at " + num(D_CLS)
             + " dims decode the angle BETTER than the " + num(D_PAT)
             + "-dim patch latent", 12.5, fill=ACCENT["crit"], weight="bold")
    s += txt(468, DY + 168, "so the capacity argument does not run in the direction the "
             "campaign assumed", 12.5, fill=ACCENT["crit"], weight="bold")
    # rank 7 with five 384-d arms ahead leaves one arm unaccounted for, and an unexplained
    # gap in the arithmetic of a figure about honest reading is itself a defect.  It is
    # named, with its own width, rather than dropped for tidiness.
    if ahead_other:
        s += txt(468, DY + 186, "arms ahead of " + COLS_ARM + " that are not "
                 + num(D_CLS) + "-dim " + cls_v + ":   "
                 + " ,   ".join(f"{a} {m:.2f}{DEG} at {num(tab[a]['n_feat'])} dims"
                                for m, a in ahead_other), 11, fill=MUTED)

    # -- the footer
    s += frow(F1, [
        (["num_patches  =  ( img_size / patch_size )" + SQ + " ,   img_size = " + str(img)
          + " ,  patch_size = " + str(patch)], INK, "normal"),
        (["total latent  =  num_patches " + TIMES + " proj_dim"], INK, "bold")])
    s += frow(F2, [
        ("the predictor is one operator: " + num(npar) + " parameters at P = 1 and at P = "
         + str(NP), INK, "normal"),
        ("models/infojepa_modules.py, mode ltv, num_frames " + str(nhist) + ", rank "
         + str(prank), MUTED, "normal", 0.9)])
    s += strip(PIN, SY, [
        (COLS_ARM + "  " + pat_v, f"{cem[COLS_ARM]['mean']:.3f}   n = "
         + str(cem[COLS_ARM]["n"]), False),
        (BASE_ARM + "  " + cls_v, f"{cem[BASE_ARM]['mean']:.3f}   n = "
         + str(cem[BASE_ARM]["n"]), False),
        ("the paired contrast", f"{eff['mean']:+.3f}  NULL", False),
        ("the latent it also moved", TIMES + " " + str(RATIO), True)], K, cw=180)
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
    checkable by eye on a 940 x 1900 canvas; this reads the emitted file back and says so.
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
    ("arch-columns.svg", columns_fig),
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
