"""Round 8, the GAP: what the model's temporal structure actually is, and the S x T factorial.

    python analysis/round8_gap_figs.py --out diary/assets/2026-09-06
    -> arch-temporal-gap.svg   the temporal structure as BUILT -- the premise of the round
    -> arch-factorial.svg      which spatial and which temporal mechanism composes each ST arm

Seven rounds changed the objective, the planner, the predictor family, the link, the
representation and the width.  None of them changed how much past the model integrates, how
the frames are encoded, or whether the target is detached.  G1 is that premise, drawn from
the source; G2 is the design that follows from it.  Hues by IDENTITY, never by rank:

    green    G1's panel -- the system AS BUILT.  G1 is not a proposal, it is a reading of
             the code that is running, so it wears the system's own hue.
    crimson  the two defects G1 finds: the hard cutoff at H, and a target that is not
             detached (`detach_target: False`), which is what makes this not a JEPA.
    amber    G2's panel and its two ST cells -- the intervention under test.
    purple   G2's single-factor ablations, the CONTRASTING condition for each ST arm.
    slate    neutral -- an arm already on the record, a ghost tap, a NULL.
    teal     reserved here for nothing; no consensus arm appears in either figure.

LAYOUT RULES INHERITED FROM round6_arch_figs.py / round7_v2_arch_figs.py

  * Strict left-to-right dataflow on fixed row baselines; no two wires share a segment.
  * base() fixes every anchor, so a connector uses ONE named route.  G2 reuses round6's
    `wire_to_pred` verbatim -- the route into the predictor's flat top, which is where every
    temporal mechanism in this round lives.  G1's panel reads the ENCODER, where none of the
    house routes lands, so it reuses round7 S1's encoder route (start short on the dome's
    right edge, run the empty right margin at x = 890, land long on the panel's top edge),
    exactly as round7_v2's G1 does.
  * G1 needs a SECOND wire -- the target's gradient returning into the encoder -- and the
    right margin is already taken by the panel route.  So that one is drawn where the
    gradient actually flows, beside the target column at x = 838: two segments straddling
    the right-hand link box (y 420..464), because the gradient passes THROUGH the link, and
    the arrowhead lands on the encoder dome at (838, 522).  It crosses nothing, and it never
    shares a segment with the panel route at x = 890.
  * A caption belongs on the side of its wire that has room.  G1's `detach_target: False`
    was first anchored end at x = 826 and ran back across the target column's own forward
    wire at x = 812; it is anchored START at x = 850 now, in the empty right margin.
  * Nothing is sized by hand where it can be measured: mbox / pill / inset / frow / strip all
    fit their own text and none may cross the panel border.
  * A NULL is never drawn like a positive.  patchdecode's +0.085 spans zero at n = 8, so on
    G2's crimson row it gets slate, a HOLLOW marker and the word NULL -- the encoding
    round7-design.svg and round7_v2's G1 both use.
  * `audit()` runs on every render: it re-reads the emitted SVG, reconstructs the box of
    every <text> element from the same Helvetica metrics the layout used, and reports any
    pair that intersects, anything that leaves the canvas, and anything that crosses the
    border of the frame it sits in (every frame registers itself through `frame()`/`pane()`).
    It is the SVG counterpart of analysis/round7_figs.py's audit() and it exists for the same
    reason: at 940 x 1800 units "nothing overlaps" is not checkable by eye.

EVERY NUMBER IS READ FROM THE ARCHIVE AT RENDER TIME.  Nothing here is typed from a diary.

    the 875 archived run configs, and the num_hist / frameskip / block_causal /
      decode_pred_w histograms over them            runs/outputs/*/.hydra/config.yaml
    which arms hold the num_hist exceptions         the same scan, by directory name
    PiWM-blockcausal's three success rates          collect_evals.collect(scheme="fixed")
    n for LpWM-ltv / PiWM-columns / PiWM-patchdecode                    the same
    patchdecode - columns, its 95 % interval and n  figures.paired_effect on the same
    each grid cell's feature / predictor mode / lag_dilation / decode_pred_w
                                                    that ARM'S OWN archived config
    seeds launched per round-8 arm                  its run directories on disk
    811 840 predictor parameters at P = 1 and at P = 256, and mlpvar's 738 048 against
      ssm's 738 432                                 models.infojepa_modules, instantiated
    every source line number cited on either figure `srcline()`, by searching the file

  ARITHMETIC DONE HERE, and nothing else: the 843 / 875, 872 / 875 and 24 / 875 ratios and
  the count of exception arms; the ssm-vs-mlpvar parameter gap as a percentage; and G2's two
  value-strip counts, which are len() over ST_PAIRS -- the same table the grids are drawn
  from, so the strip cannot disagree with the picture above it.

  ONE CLAIM DELIBERATELY NOT MADE.  An earlier draft of G2's crimson row drew patchdecode's
  two single-factor cells as "never run".  That is false: PiWM-columns (n = 12) and
  PiWM-patchdecode-detach (n = 8) are both on the record.  What the archive does support is
  the pair now drawn -- the quoted +0.085 spans zero, and decode_pred_w, the only key that
  sends a pixel gradient into the PREDICTOR, is zero in every config outside this round.

Usage:  python analysis/round8_gap_figs.py --out diary/assets/2026-09-06 [--png]
        `--png` rasterises beside each SVG, for the read-back; the audit runs either way and
        main() exits non-zero if it has anything to say.
"""
import argparse
import functools
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.arch_figs import (  # noqa: E402,F401
    ACCENT, ACCENT_FILL, DIM, GRID, INK, MUTED, WHITE,
    _marker, arrow, base, circle, domeup, poly, sub, text_width, title_line, txt,
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
    wire_to_pred,
)
from analysis.round7_arch_figs import cbadge  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "diary", "assets", "2026-09-06")

SGBAR = "sg"
CFG_GLOB = os.path.join(REPO, "runs", "outputs", "*", ".hydra", "config.yaml")


# --- the archive, read at render time ----------------------------------------------
def srcline(rel, needle):
    """The line number of the first line of `rel` containing `needle`.

    A citation typed as a literal goes stale the first time anyone inserts a function above
    it -- the round-8 diary already cites :1234 for a comment that now sits at :1280.  So
    every file:line on these two figures is SEARCHED for, and a moved line renumbers itself
    while a DELETED line raises here rather than being drawn as a fact."""
    with open(os.path.join(REPO, rel)) as f:
        for i, line in enumerate(f, 1):
            if needle in line:
                return i
    raise AssertionError(f"{needle!r} no longer appears in {rel}: the figure cites it")


def _cfg_key(path, key):
    """One TOP-LEVEL scalar out of a hydra config, without paying for a yaml parse.

    875 configs x a full safe_load is 7 s; three line scans is 0.2 s, and the three keys
    this figure counts (num_hist, frameskip, block_causal) are all top level."""
    with open(path) as f:
        for line in f:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].split("#")[0].strip()
    return None


@functools.lru_cache(maxsize=None)
def cfg_scan(key):
    """{value: [arm, ...]} for one top-level key over every archived run config."""
    out = {}
    for p in sorted(glob.glob(CFG_GLOB)):
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        out.setdefault(_cfg_key(p, key), []).append(re.sub(r"_s\d+$", "", run))
    return out


@functools.lru_cache(maxsize=None)
def n_configs():
    return len(glob.glob(CFG_GLOB))


@functools.lru_cache(maxsize=None)
def arm_cfg(arm):
    """(config dict, seeds launched) for an arm, from its OWN archived run directories.

    The launcher's ARMS[] table states the intent; this reads what actually got written to
    disk, which is the thing the runs were built from.  The trailing underscore in the glob
    matters for the same reason resolve_arm() needs it: `PiWM-ssm_` must not swallow
    `PiWM-ssm-foo`, and `PiWM-pdpred_` must not swallow `PiWM-pdpred-w003`."""
    import yaml
    dirs = sorted(glob.glob(os.path.join(REPO, "runs", "outputs", arm + "_pd384*_s*")))
    assert dirs, f"no run directory for {arm}"
    with open(os.path.join(dirs[0], ".hydra", "config.yaml")) as f:
        return yaml.safe_load(f), len(dirs)


@functools.lru_cache(maxsize=None)
def pred_params(num_patches, mode="ltv"):
    """The predictor's parameter count, by BUILDING it -- the claim is about the module.

    'One operator broadcast over every token' is exactly the statement that this number does
    not move when num_patches goes 1 -> 256, so the honest way to draw it is to construct
    both and count.  D and H come from the baseline's own archived config."""
    import torch  # noqa: F401  (imported for its side effect on the module below)
    from models.infojepa_modules import LinearDynamicsPredictor
    cfg, _ = arm_cfg("LpWM-ltv")
    p = LinearDynamicsPredictor(input_dim=int(cfg["embed_dim"]),
                                num_frames=int(cfg["num_hist"]),
                                num_patches=num_patches, mode=mode,
                                rank=int(cfg["predictor"].get("rank", 16)))
    return sum(q.numel() for q in p.parameters())


@functools.lru_cache(maxsize=None)
def evals():
    """collect_evals' fixed-instrument table, plus the paired-effect function it is read
    with.  scheme='fixed' because a pre-49a3e55 eval is a different instrument."""
    from analysis.collect_evals import collect, resolve_arm
    from analysis import figures as FG
    return collect(scheme="fixed")[0], resolve_arm, FG


def arm_n(name):
    """How many seeds of `name` have a final eval on the fixed instrument.

    KeyError, not a bare except: collect_evals.ArmNameError subclasses KeyError and means
    'this arm has no evals yet', which is the true answer for every arm launched this
    round.  Anything else is a bug and should still raise."""
    A, resolve, _ = evals()
    try:
        return len(A[resolve(A, name)])
    except KeyError:
        return 0


@functools.lru_cache(maxsize=None)
def facts():
    """Everything either figure draws, in one read of the archive."""
    A, resolve, FG = evals()
    hist = cfg_scan("num_hist")
    fskip = cfg_scan("frameskip")
    bcz = cfg_scan("block_causal")
    n_cfg = n_configs()
    exc = sorted({a for v, arms in hist.items() if v != "3" for a in arms})
    dpw = cfg_scan("decode_pred_w")
    dpw_nz = {k: v for k, v in dpw.items() if k not in (None, "0.0")}
    bc = A[resolve(A, "PiWM-blockcausal")]
    e = FG.paired_effect(A, resolve(A, "PiWM-columns"), resolve(A, "PiWM-patchdecode"))
    ltv_cfg, _ = arm_cfg("LpWM-ltv")
    return {
        "n_cfg": n_cfg,
        "n_hist3": len(hist.get("3", ())),
        "hist_exc": exc,
        "n_hist_exc": sum(len(v) for k, v in hist.items() if k != "3"),
        "hist_exc_vals": sorted(int(k) for k in hist if k != "3"),
        "n_fskip5": len(fskip.get("5", ())),
        "frameskip": int(ltv_cfg["frameskip"]),
        "num_hist": int(ltv_cfg["num_hist"]),
        "embed_dim": int(ltv_cfg["embed_dim"]),
        "n_bc_on": len(bcz.get("true", ())),
        "n_bc_off": n_cfg - len(bcz.get("true", ())),
        "bc_scores": [bc[k] for k in sorted(bc, key=int)],
        "n_bc": len(bc),
        "n_dpw_nz": sum(len(v) for v in dpw_nz.values()),
        "dpw_nz_arms": sorted({re.sub(r"_pd384.*$", "", a)
                               for v in dpw_nz.values() for a in v}),
        "pd_effect": e,
        "params_p1": pred_params(1),
        "params_p256": pred_params(256),
        "params_ssm": pred_params(1, "ssm"),
        "params_mlpvar": pred_params(1, "mlp_var"),
    }


def dpw(cfg):
    """decode_pred_w for an arm, with the ABSENT key read as the constructor default.

    The key only appears in configs written after the flag existed; an older config has no
    `decode_pred_w` line and its runs took visual_world_model.py's default, which is the
    same 0.0.  Reading absence as 0.0 is therefore the archive's own answer, not a guess."""
    return float(cfg.get("decode_pred_w", 0.0))


def tokens_for(cfg):
    """models/vit_encoder.py, verbatim: grid = img_size // patch_size, num_patches = grid²."""
    return (int(cfg["img_size"]) // int(cfg["encoder"]["patch_size"])) ** 2


def thousands(n):
    """4 811 840, not 4,811,840: a comma inside a number reads as a list separator in a
    strip cell that is already comma-separated."""
    return f"{n:,}".replace(",", " ")


# --- local primitives --------------------------------------------------------------
def clip3(cx, cy, side=42, off=9, key="slate", n=3):
    """`n` frames as offset squares: one CLIP, not one observation.

    base() draws a single circle for o_t and every figure in the set inherits it, but this
    round's whole premise is that the model reads num_hist frames at once and treats them as
    independent -- so the left-hand input is drawn as the clip it actually is."""
    s = ""
    for i in range(n - 1, -1, -1):
        x, y = cx - side / 2 + i * off, cy - side / 2 - i * off
        s += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{side}" height="{side}" rx="4" '
              f'fill="{ACCENT_FILL[key]}" stroke="{ACCENT[key]}" stroke-width="1.6"/>\n')
    return s


def vline(x, y0, y1, key, w=1.2, op=0.45, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x},{y0} L{x},{y1}" stroke="{ACCENT[key]}" stroke-width="{w}" '
            f'stroke-opacity="{op}"{d}/>\n')


def hline(x0, x1, y, key, w=1.2, op=0.45, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x0},{y} L{x1},{y}" stroke="{ACCENT[key]}" stroke-width="{w}" '
            f'stroke-opacity="{op}"{d}/>\n')


def vbrace(x, y0, y1, key, w=1.6, tick=9):
    """A vertical brace: `these rows, together, are one statement`."""
    c = ACCENT[key]
    return (f'<path d="M{x - tick},{y0} L{x},{y0} L{x},{y1} L{x - tick},{y1}" '
            f'stroke="{c}" stroke-width="{w}" fill="none" stroke-linejoin="round" '
            f'stroke-linecap="round"/>\n')


def sgmark(cx, cy, key, h=11, gap=4, w=2.4):
    """The stop-gradient bars.  round5_figs_obj.sgbar draws these in a fixed crimson; here
    the JEPA reference gets them in slate and the campaign gets them struck through, so the
    colour has to be the caller's."""
    c = ACCENT[key]
    return (f'<path d="M{cx - gap},{cy - h} L{cx - gap},{cy + h} M{cx + gap},{cy - h} '
            f'L{cx + gap},{cy + h}" stroke="{c}" stroke-width="{w}" '
            f'stroke-linecap="round"/>\n')


def cell(x, y, w, h, name, spec, detail, key, chip="", dash=False, chipkey=None):
    """One cell of a factorial grid: the arm, its two settings, and its status.

    The name is LEFT-aligned and the chip RIGHT-aligned on the same band -- a centred name
    plus a corner chip collides the moment an arm is called `PiWM-st-pdpred`."""
    c = ACCENT[key]
    d = ' stroke-dasharray="6,4"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
         f'fill="{WHITE if dash else ACCENT_FILL[key]}" '
         f'fill-opacity="{1 if dash else 0.55}" stroke="{c}" stroke-width="1.8"{d}/>\n')
    chw = text_width(chip, 11, "bold") + 26 if chip else 0
    s += txt(x + 14, y + 24, name, fit_size(name, 15.5, w - 24 - chw, floor=11.5),
             anchor="start", fill=MUTED if dash else c, weight="bold")
    if chip:
        s += badge(x + w - 10, y + 8, chip, chipkey or key, size=11, anchor="end")
    s += txt(x + 14, y + 42, spec, fit_size(spec, 12, w - 28, floor=9.5), anchor="start",
             fill=MUTED if dash else INK)
    if detail:
        s += txt(x + 14, y + 57, detail, fit_size(detail, 11, w - 28, floor=9),
                 anchor="start", fill=MUTED)
    return s


# Every frame a label must stay inside, registered as it is drawn.  The canvas check is not
# enough on its own: a label centred on the right half of an inset can run past that inset's
# border while still sitting comfortably inside the page.
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
    """The title is one centred run at y = 872 and both figures drop a connector past it, so
    its half-width is a HARD constraint, not a matter of taste."""
    body = re.sub(r"<[^>]*>", "", title)
    wide = text_width(body.replace(" -- ", " – "), size, "bold")
    right = cx + wide / 2
    assert right < corridor_x - clear, (
        f"title runs to x={right:.0f}, into the connector corridor at x={corridor_x}: "
        f"shorten it")
    return title


# =========================================== G1: the temporal structure, as built
G1_TITLE = ("(G1) the temporal axis as built -- a {h}-tap FIR, and a live target"
            "  [UNTESTED]")


def zlab(t):
    return sub("z", t, 11)


def g1_temporal_gap():
    """The premise of the whole round, read off the source: three frames encoded
    independently, an FIR window with a hard cutoff at H, and a target that is not
    detached."""
    K = "blue"                                  # green -- the system AS BUILT
    F = facts()
    b = base(_check_title(G1_TITLE.format(h=F["num_hist"]), 890), skip=("obs",))
    # the left input is the CLIP the model actually reads; the right one is the single
    # frame it is asked to predict.  Drawing both as one circle, as base() does, is the
    # picture that hides the round's question.
    b += clip3(120, 742, n=F["num_hist"])
    b += arrow(120, 706, 120, 646)
    b += txt(120, 800, f"num_hist = {F['num_hist']}  " + DOT
             + f"  frameskip = {F['frameskip']}", 12.5, fill=ACCENT[K], weight="bold")
    b += circle(812, 742, 34, sub("o", "t+1", 13))
    b += arrow(812, 706, 812, 646)
    b += txt(812, 800, "one target frame", 12.5, fill=MUTED)
    b += txt(190, 502, "every frame encoded INDEPENDENTLY  " + DOT
             + "  t folded into the batch", 13.5, anchor="start", fill=ACCENT[K],
             weight="bold")
    b += txt(327, 176, "a " + str(F["num_hist"]) + "-tap FIR  " + DOT
             + "  n_lags = num_frames = num_hist", 13, fill=ACCENT[K], weight="bold")
    # THE SECOND WIRE: the target's own gradient, drawn where it actually flows -- down the
    # target column, straddling the link box, into the encoder it came from.
    b += apoly([(823, 300), (838, 300), (838, 412)], "crit", dash="5,4")
    b += aw(838, 470, 838, 522, "crit", dash="5,4")
    # the caption sits to the RIGHT of its own wire: anchored the other way it ran back
    # across the target column's forward wire at x = 812.
    b += txt(850, 488, "detach_target", 11.5, anchor="start", fill=ACCENT["crit"],
             weight="bold")
    b += txt(850, 504, "= False", 11.5, anchor="start", fill=ACCENT["crit"],
             weight="bold")
    # round7 S1's encoder route: start short on the dome's right edge, run the empty right
    # margin, land long on the panel's top edge.
    b += poly([(868, 560), (890, 560), (890, PY)], color=ACCENT[K], w=1.8, dash="6,4")

    PH = 950
    s = pane(PX, PY, PW, PH, K,
             "module:  the encoder, the predictor and the target " + NDASH
             + "  none of the three carries time")

    # -- row A: the encoder.  One set of weights, three times, nothing passed between.
    s += frame(60, 958, 816, 190, "the encoder " + NDASH + "  t is folded into the BATCH",
               K, note="models/visual_world_model.py:"
               + str(srcline("models/visual_world_model.py", '"b t ... -> (b t) ..."'))
               + "   rearrange( visual , \"b t ... " + ARROWC + " (b t) ...\" )")
    for i, t in enumerate(("t" + MINUS + "2", "t" + MINUS + "1", "t")):
        x = 86 + i * 46
        s += (f'<rect x="{x}" y="{1028}" width="36" height="36" rx="4" '
              f'fill="{ACCENT_FILL["slate"]}" stroke="{ACCENT["slate"]}" '
              f'stroke-width="1.4"/>\n')
        s += txt(x + 18, 1080, sub("o", t, 10), 11.5, fill=MUTED)
    s += aw(226, 1046, 244, 1046, K)
    s += mbox(248, 1024, 128, 44, "( b t )", K, size=15, sub_="one batch axis")
    s += aw(376, 1046, 394, 1046, K)
    s += mbox(398, 1024, 122, 44, "Enc " + "<tspan font-style='italic'>f</tspan>", K,
              size=15, sub_="one set of weights")
    s += aw(520, 1046, 538, 1046, K)
    for i, t in enumerate(("t" + MINUS + "2", "t" + MINUS + "1", "t")):
        s += mbox(542 + i * 74, 1028, 66, 36, zlab(t), K, size=14)
    s += txt(468, 1106, "“the encoder otherwise has no temporal structure "
             "whatsoever”   " + DOT + "   visual_world_model.py:"
             + str(srcline("models/visual_world_model.py",
                           "no temporal structure whatsoever")), 12.5,
             fill=ACCENT["crit"], weight="bold")
    s += txt(468, 1128, "no attention across t, no state, no positional link " + NDASH
             + " block_causal is the flag that would change it, and it is off in "
             + f"{F['n_bc_off']} of {F['n_cfg']} archived configs", 11.5, fill=MUTED)

    # -- row B: the predictor.  An FIR bank, and a wall at H.
    EQ = (zlab("t+1") + "  =  " + "Σ" + sub("", "k=0", 11) + " A" + sub("", "k", 11)
          + " " + zlab("t" + MINUS + "k") + "  +  B a" + sub("", "t", 11)
          + " ,   k  &lt;  H  =  num_frames  =  num_hist  =  " + str(F["num_hist"]))
    s += frame(60, 1164, 816, 296, "the predictor " + NDASH + "  a "
               + str(F["num_hist"]) + "-tap FIR bank, and a HARD cutoff at H", K, note=EQ)
    for i, t in enumerate(("t", "t" + MINUS + "1", "t" + MINUS + "2")):
        y = 1232 + i * 36
        s += txt(104, y + 19, f"k = {i}", 12, anchor="end", fill=MUTED)
        s += mbox(116, y, 104, 28, zlab(t), K, size=14)
        s += aw(220, y + 14, 242, y + 14, K)
        s += mbox(246, y, 76, 28, "A" + sub("", str(i), 10), K, size=14)
        s += aw(322, y + 14, 346, y + 14, K)
    s += vbrace(352, 1246, 1318, K, tick=-6)
    s += aw(352, 1282, 374, 1282, K)
    s += pill(400, 1282, "Σ", K, rx=24, ry=22)
    s += aw(426, 1282, 448, 1282, K)
    s += mbox(452, 1260, 104, 44, "+ B a" + sub("", "t", 10), K, size=15,
              sub_="the action term")
    s += aw(556, 1282, 578, 1282, K)
    s += pill(626, 1282, zlab("t+1"), K, rx=44, ry=22)
    s += txt(626, 1330, "one step, always", 11.5, fill=MUTED)
    # the wall, and the two taps that do not exist
    s += hline(80, 560, 1352, "crit", w=1.8, op=0.9, dash="7,5")
    s += txt(576, 1348, "H = " + str(F["num_hist"]) + " " + NDASH + " the cutoff", 13,
             anchor="start", fill=ACCENT["crit"], weight="bold")
    s += txt(576, 1366, "not a decay, a WALL", 11.5, anchor="start", fill=MUTED)
    for i in range(2):                        # the first two taps BEYOND the cutoff
        k = F["num_hist"] + i
        t = "t" + MINUS + str(k)
        y = 1366 + i * 36
        s += txt(104, y + 19, f"k = {k}", 12, anchor="end", fill=DIM)
        s += (f'<rect x="116" y="{y}" width="104" height="28" rx="6" fill="{WHITE}" '
              f'stroke="{DIM}" stroke-width="1.3" stroke-dasharray="5,4"/>\n')
        s += txt(168, y + 19, zlab(t), 14, fill=DIM)
        s += strikeout(283, y + 14, "crit", r=11, w=2.0)
        s += txt(310, y + 19, "no A" + sub("", str(k), 10)
                 + ("   " + DOT + "   nothing older than "
                    + zlab("t" + MINUS + str(F["num_hist"] - 1)) + " is in the graph"
                    if i == 0 else
                    "   " + DOT + "   and no state carries it forward"), 12,
                 anchor="start", fill=MUTED)
    s += txt(468, 1444, "models/infojepa_modules.py:"
             + str(srcline("models/infojepa_modules.py",
                           "z_{t-dk} unavailable (cold-start)"))
             + "   `if dk >= T: break`   " + DOT
             + "   T = num_hist, so a lag at or beyond H is dropped, exactly", 12,
             fill=MUTED)

    # -- row C: the target.  detach_target: False, and what that costs.
    YAML_L = srcline("conf/train_rdmreg.yaml", "detach_target: False")
    s += frame(60, 1476, 816, 216, "the target " + NDASH + "  it is NOT detached, so the "
               "encoder trains through its own prediction target", "crit",
               note="conf/train_rdmreg.yaml:" + str(YAML_L) + "   detach_target: False"
                    + "   " + DOT + "   visual_world_model.py:"
                    + str(srcline("models/visual_world_model.py",
                                  "tgt.detach() if self.detach_target else tgt"))
                    + " , :"
                    + str(srcline("models/visual_world_model.py",
                                  "z_tgt.detach() if self.detach_target else z_tgt")))
    s += vline(470, 1540, 1652, "crit", w=1.2, op=0.45)
    for j, (x0, head, note, kk) in enumerate((
            (78, "what a JEPA does", "the target branch is CUT", "slate"),
            (490, "what every run in this archive did", "the gradient goes round",
             "crit"))):
        s += txt(x0, 1552, head, 13, anchor="start", fill=ACCENT[kk], weight="bold")
        s += mbox(x0, 1568, 92, 40, "Enc " + "<tspan font-style='italic'>f</tspan>", kk,
                  size=14)
        s += aw(x0 + 92, 1588, x0 + 128, 1588, kk)
        s += mbox(x0 + 132, 1568, 100, 40, "target", kk, size=14)
        s += aw(x0 + 232, 1588, x0 + 264, 1588, kk)
        s += mbox(x0 + 268, 1568, 84, 40, "MSE", kk, size=14)
        if j == 0:
            s += sgmark(x0 + 110, 1588, "slate")
            s += txt(x0 + 110, 1630, "sg", 12.5, fill=ACCENT["slate"], weight="bold")
        else:
            s += sgmark(x0 + 110, 1588, "slate")
            s += strikeout(x0 + 110, 1588, "crit", r=13, w=2.4)
            s += txt(x0 + 110, 1630, "no sg", 12.5, fill=ACCENT["crit"], weight="bold")
            s += apoly([(x0 + 310, 1608), (x0 + 310, 1642), (x0 + 46, 1642),
                        (x0 + 46, 1610)], "crit", dash="5,4")
        s += txt(x0, 1662, note, 12, anchor="start", fill=MUTED)
    s += txt(468, 1682, "“the cheapest way to lower S is therefore not to predict "
             "better but to inflate z and z_pred together”   " + DOT
             + "   :" + str(srcline("models/visual_world_model.py",
                                    "the cheapest way to lower S is therefore not to "
                                    "predict")), 12, fill=ACCENT["crit"], weight="bold")

    s += frow(1734, [
        (["num_hist = " + str(F["num_hist"]) + " in " + str(F["n_hist3"]) + " of "
          + str(F["n_cfg"]) + " archived configs"], INK, "bold"),
        (["the " + str(F["n_hist_exc"]) + " exceptions are round 8's OWN ladder, launched "
          "this round"], MUTED, "normal", 0.94)])
    s += frow(1758, [
        (["frameskip = " + str(F["frameskip"]) + " in " + str(F["n_fskip5"]) + " of "
          + str(F["n_cfg"]) + " " + NDASH + " every arm, every round"], INK, "normal"),
        (["the predictor is ONE operator, broadcast over every token"], INK, "bold")])
    s += strip(PIN, 1776, [
        ("num_hist, all " + str(F["n_cfg"]) + " configs",
         str(F["n_hist3"]) + " at " + str(F["num_hist"]), False),
        ("block_causal ever ON", str(F["n_bc_on"]) + " runs " + DOT + " CEM "
         + " , ".join(f"{v:.2f}" for v in F["bc_scores"]), True),
        ("predictor at P = 1", thousands(F["params_p1"]) + " params", False),
        ("predictor at P = 256", thousands(F["params_p256"]) + " params", False)],
        K, cw=170)
    return b + s, PY + PH + 40


# ============================================== G2: the S x T factorial
G2_TITLE = "(G2) the S " + TIMES + " T factorial -- every ST arm with both ablations  [LAUNCHED]"

LAB_R, C1X, C2X, CW = 282, 296, 596, 250
CELL_H, ROW_GAP = 60, 36

#: (the ST arm, the two arms that ablate it) -- the two grids below, as a table, so the
#: value strip COUNTS what the figure drew instead of restating it.
ST_PAIRS = (("PiWM-st-ssm", ("PiWM-ssm", "PiWM-columns")),
            ("PiWM-st-pdpred", ("PiWM-pdpred", "PiWM-lagskip")))


def grid2x2(y0, cols, rows, cells, edge_lab):
    """One 2 x 2: a spatial mechanism across, a temporal mechanism down.

    The two ABLATION EDGES are the point of the picture, so they are drawn rather than
    implied: the ST cell sits bottom-right, and one arrow runs left to the arm that removes
    its spatial half and one runs up to the arm that removes its temporal half.  Every cell's
    settings are read from that arm's own archived config, so a cell cannot claim a
    mechanism the run did not have."""
    s = ""
    r1, r2 = y0 + 26, y0 + 26 + CELL_H + ROW_GAP
    for cx, (head, sub_) in zip((C1X + CW / 2, C2X + CW / 2), cols):
        s += txt(cx, y0, head, 13.5, weight="bold")
        s += txt(cx, y0 + 17, sub_, 11.5, fill=MUTED)
    for ry, (head, sub_) in zip((r1, r2), rows):
        s += txt(LAB_R, ry + 26, head, 13.5, anchor="end", weight="bold")
        s += txt(LAB_R, ry + 44, sub_, fit_size(sub_, 11.5, LAB_R - 78, floor=9),
                 anchor="end", fill=MUTED)
    for ((cxi, ryi), spec) in cells:
        s += cell(C1X if cxi == 0 else C2X, r1 if ryi == 0 else r2, CW, CELL_H, *spec)
    # the two edges out of the ST cell
    s += aw(C2X - 4, r2 + CELL_H / 2, C1X + CW + 4, r2 + CELL_H / 2, "slate")
    s += txt((C1X + CW + C2X) / 2, r2 + CELL_H / 2 - 10, edge_lab[0], 10.5,
             fill=ACCENT["slate"], weight="bold")
    s += aw(C2X + CW / 2, r2 - 4, C2X + CW / 2, r1 + CELL_H + 4, "slate")
    s += txt(C2X + CW / 2 + 46, r2 - ROW_GAP / 2 + 4, edge_lab[1], 10.5,
             fill=ACCENT["slate"], weight="bold")
    return s, r2 + CELL_H


def g2_factorial():
    """Which spatial and which temporal mechanism composes each ST arm, and which S and T
    arm ablates it -- the thing patchdecode's round-5 win did not have."""
    K = "amber"                                 # the intervention under test
    P = "magenta"                               # purple -- the contrasting condition
    F = facts()
    b = base(_check_title(G2_TITLE, 880))
    b += wire_to_pred(K, "mode  " + DOT + "  lag_dilation  " + DOT + "  feature")
    b += txt(712, 502, "feature = cls on every S/T arm  " + DOT
             + "  feature = patch on both ST arms", 14, anchor="end", fill=ACCENT[K],
             weight="bold")

    PH = 1002
    s = pane(PX, PY, PW, PH, K,
             "module:  ST  =  a SPATIAL mechanism  " + TIMES + "  a TEMPORAL mechanism ,  "
             "with BOTH ablations")

    # -- row A: st-ssm.  per-token state = the patch axis x an IIR state.
    ltv_cfg, _ = arm_cfg("LpWM-ltv")
    col_cfg, _ = arm_cfg("PiWM-columns")
    ssm_cfg, ssm_n = arm_cfg("PiWM-ssm")
    stssm_cfg, stssm_n = arm_cfg("PiWM-st-ssm")
    dp = 100.0 * (F["params_ssm"] - F["params_mlpvar"]) / F["params_mlpvar"]
    s += frame(60, 958, 816, 284, "ST1  PiWM-st-ssm " + NDASH + "  a state per TOKEN", K,
               note="its matched control PiWM-mlpvar carries "
                    + thousands(F["params_mlpvar"]) + " params against ssm's "
                    + thousands(F["params_ssm"]) + f"  ({dp:+.2f} %) , same ReLU"
                    + ARROWC + "W readout")
    gA, endA = grid2x2(
        1030,
        (("no spatial axis", "feature = cls  " + DOT + "  1 token"),
         ("the patch axis", "feature = patch  " + DOT + "  "
          + str(tokens_for(col_cfg)) + " tokens")),
        (("a 3-tap FIR", "mode = ltv  " + DOT + "  " + "Σ" + " A" + sub("", "k", 10)
          + " " + zlab("t" + MINUS + "k")),
         ("an IIR state", "mode = ssm  " + DOT + "  s" + sub("", "t", 10) + " = A s"
          + sub("", "t" + MINUS + "1", 10) + " + Bz " + zlab("t") + " + Ba a"
          + sub("", "t", 10))),
        (((0, 0), ("LpWM-ltv", "mode ltv  " + DOT + "  feature cls",
                   f"n = {arm_n('LpWM-ltv')} evaluated  " + DOT + "  the baseline of the "
                   "campaign", "blue", "ON RECORD")),
         ((1, 0), ("PiWM-columns", "mode ltv  " + DOT + "  feature patch",
                   f"n = {arm_n('PiWM-columns')} evaluated  " + DOT + "  the S arm: the "
                   "axis, no state", P, "ABLATES T")),
         ((0, 1), ("PiWM-ssm", "mode " + str(ssm_cfg["predictor"]["mode"]) + "  " + DOT
                   + "  feature " + str(ssm_cfg["encoder"]["feature"]),
                   f"{ssm_n} seeds launched  " + DOT + "  the T arm: the state, no axis",
                   P, "ABLATES S")),
         ((1, 1), ("PiWM-st-ssm", "mode " + str(stssm_cfg["predictor"]["mode"]) + "  "
                   + DOT + "  feature " + str(stssm_cfg["encoder"]["feature"]),
                   f"{stssm_n} seeds launched  " + DOT + "  each patch keeps its own "
                   "history", K, "ST"))),
        (MINUS + " patch", MINUS + " state"))
    s += gA
    s += txt(468, endA + 24, "PiWM-ssm removes the patch axis and PiWM-columns removes "
             "the state " + NDASH + "  so either half can be charged for the result",
             12.5, fill=ACCENT[K], weight="bold")

    # -- row B: st-pdpred.  per-token next-frame pixels x a lag skip.
    pdp_cfg, pdp_n = arm_cfg("PiWM-pdpred")
    lsk_cfg, lsk_n = arm_cfg("PiWM-lagskip")
    stpd_cfg, stpd_n = arm_cfg("PiWM-st-pdpred")
    s += frame(60, 1258, 816, 284, "ST2  PiWM-st-pdpred " + NDASH + "  next-frame pixels "
               "per TOKEN, on a skipped history", K,
               note="decode_pred_w decodes " + zlab("t+1") + " " + NDASH + " the "
                    "predictor's code for a frame the model has NOT seen " + NDASH
                    + " onto that frame's real pixels")
    gB, endB = grid2x2(
        1330,
        (("no pixel gradient", f"decode_pred_w = {dpw(ltv_cfg)}  " + DOT
          + "  feature cls"),
         ("pixels through the PREDICTOR", f"decode_pred_w = {dpw(pdp_cfg)}  " + DOT
          + "  feature patch")),
        (("uniform lags", "lag_dilation unset  " + ARROWC + "  { "
          + zlab("t") + " , " + zlab("t" + MINUS + "1") + " , " + zlab("t" + MINUS + "2")
          + " }"),
         ("a lag SKIP", "lag_dilation " + str(lsk_cfg["predictor"]["lag_dilation"])
          + "  " + ARROWC + "  { " + zlab("t") + " , " + zlab("t" + MINUS + "2") + " , "
          + zlab("t" + MINUS + "2") + " }")),
        (((0, 0), ("LpWM-ltv", f"decode_pred_w {dpw(ltv_cfg)}  " + DOT + "  uniform lags",
                   f"n = {arm_n('LpWM-ltv')} evaluated  " + DOT + "  the same baseline as "
                   "ST1", "blue", "ON RECORD")),
         ((1, 0), ("PiWM-pdpred", f"decode_pred_w {dpw(pdp_cfg)}  " + DOT
                   + "  uniform lags",
                   f"{pdp_n} seeds launched  " + DOT + "  the S arm: pixels, no skip",
                   P, "ABLATES T")),
         ((0, 1), ("PiWM-lagskip", f"decode_pred_w {dpw(lsk_cfg)}  " + DOT
                   + "  lag_dilation "
                   + str(lsk_cfg["predictor"]["lag_dilation"]),
                   f"{lsk_n} seeds launched  " + DOT + "  the T arm: skip, no pixels",
                   P, "ABLATES S")),
         ((1, 1), ("PiWM-st-pdpred", f"decode_pred_w {dpw(stpd_cfg)}  " + DOT
                   + "  lag_dilation "
                   + str(stpd_cfg["predictor"]["lag_dilation"]),
                   f"{stpd_n} seeds launched  " + DOT + "  feature "
                   + str(stpd_cfg["encoder"]["feature"]), K, "ST"))),
        (MINUS + " pixels", MINUS + " skip"))
    s += gB
    s += txt(468, endB + 24, "PiWM-pdpred removes the skip and PiWM-lagskip removes the "
             "pixel target " + NDASH + "  the same two edges, on the other pair", 12.5,
             fill=ACCENT[K], weight="bold")

    # -- row C: why the ablations exist at all.
    #
    # NOT drawn as "its ablations were never run": PiWM-columns (n = 12) and
    # PiWM-patchdecode-detach (n = 8) are both on the record, so that claim would be false.
    # What the archive DOES show is the pair that matters -- the quoted win is a NULL, and
    # the mechanism side of it was never reachable at all, because decode_pred_w is the
    # only key that sends a pixel gradient into the predictor and it is zero everywhere
    # except this round's own arms.
    e = F["pd_effect"]
    s += frame(60, 1558, 816, 200, "why the ablations exist " + NDASH + "  round 5's "
               "winner moved two mechanisms and named neither", "crit",
               note="PiWM-patchdecode  =  feature cls " + ARROWC + " patch   " + TIMES
                    + "   an auxiliary pixel decoder with decode_grad")
    s += mbox(90, 1620, 158, 40, "feature  cls " + ARROWC + " patch", "crit", size=12.5)
    s += mbox(90, 1674, 158, 40, "pixel decoder", "crit", size=12.5,
              sub_="decode_grad = true")
    s += vbrace(266, 1630, 1704, "crit", tick=-9)
    s += aw(266, 1667, 288, 1667, "crit")
    s += mbox(292, 1643, 158, 48, "PiWM-patchdecode", "crit", size=13,
              sub_="round 5  " + DOT + "  n = " + str(arm_n("PiWM-patchdecode")))
    s += vline(470, 1616, 1722, "crit", w=1.2, op=0.45)
    s += txt(490, 1632, "one arm, two mechanisms", 13.5, anchor="start",
             fill=ACCENT["crit"], weight="bold")
    s += mark(500, 1658, "slate", r=6, fill=WHITE)
    s += txt(516, 1663, f"{e['mean']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]  at n = "
             + str(e["n"]) + " vs PiWM-columns", 12.5, anchor="start",
             fill=ACCENT["slate"], weight="bold")
    s += txt(516, 1682, "a NULL " + NDASH + "  the interval spans zero", 11.5,
             anchor="start", fill=MUTED)
    s += txt(490, 1704, "and no pixel gradient had ever reached the PREDICTOR", 11,
             anchor="start", fill=MUTED)
    s += txt(490, 1720, "decode_pred_w > 0 in " + str(F["n_dpw_nz"]) + " of "
             + str(F["n_cfg"]) + " configs " + NDASH + " all this round's own arms", 11,
             anchor="start", fill=MUTED)
    s += txt(490, 1742, "round 8 launches the ablation BEFORE the result", 12,
             anchor="start", fill=ACCENT["crit"], weight="bold")

    s += frow(1784, [
        (["an ST arm is only interpretable if BOTH single-factor arms run"], INK, "bold"),
        (["every cell's settings are read from that arm's own archived config"], MUTED,
         "normal", 0.94)])
    s += frow(1808, [
        (["the S arms cost nothing extra: each is also a T arm's control"], INK, "normal"),
        (["hue: amber = ST , purple = its ablation , green = on the record"], MUTED,
         "normal", 0.92)])
    s += strip(PIN, 1826, [
        ("ST arms this round", str(len(ST_PAIRS)), False),
        ("their ablations, all launched",
         str(len({a for _, ab in ST_PAIRS for a in ab})), False),
        ("round 5's winner, n = " + str(e["n"]),
         f"{e['mean']:+.3f} , spans 0", False),
        ("decode_pred_w > 0, ever",
         str(F["n_dpw_nz"]) + " of " + str(F["n_cfg"]) + " configs", True)], K, cw=170)
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
    checkable by eye on a 940 x 1800 canvas; this reads the emitted file back and says so.
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
    ("arch-temporal-gap.svg", g1_temporal_gap),
    ("arch-factorial.svg", g2_factorial),
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
