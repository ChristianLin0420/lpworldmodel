"""Regenerate the 2026-09-01 and 2026-09-02 diary figures on the project design system.

WHY THIS MODULE EXISTS. The twelve PNGs under ``diary/assets/2026-09-01/`` and
``diary/assets/2026-09-02/`` were made ad hoc in the sessions that wrote those entries and
most of them had no surviving generator. They were the only figures in the project still off
``analysis/style.py``. This module rebuilds every one of them from the data that is still in
the repo, so from here on they refresh like every other figure instead of drifting.

WHERE EACH NUMBER COMES FROM

  * CEM success        ``analysis.collect_evals.collect()`` at call time -- NOT the stale
                       ``campaign.json`` at the repo root, which is the 2026-09-01 snapshot.
  * training metrics   each run's ``runs/outputs/<run>/wandb/latest-run/files/
                       wandb-summary.json`` (end of training) and, where a figure needs a
                       CURVE, the run's local wandb datastore ``wandb/run-*/*.wandb``, read
                       through ``wandb.sdk.internal.datastore``. ``runs/export/*/
                       wandb_history.csv`` is NOT used: that export was taken mid-campaign
                       and stops at 60-80% of training, which would truncate exactly the
                       collapse this module has to show.
  * d_action           ``assets/d_action_probe.json`` -- the checkpoint re-probe, one fixed
                       batch and one fixed permutation for every arm.
  * configs            ``runs/outputs/<run>/hydra.yaml`` (``link.kwta_k``, ``n_heads``,
                       ``use_pose``, ``mup_input_lr_fix``) to identify an arm's family
                       without pattern-matching its name.

CANARY-* runs are excluded everywhere, exactly as ``analysis.causal_figs.harvest`` does: they
are ~200-step liveness probes, not experiments.

TWO EVAL INSTRUMENTS, NEVER POOLED. ``plan.py:134`` degenerated seed 0 to one episode
repeated 50x until 49a3e55. ``collect_evals`` classifies every eval as ``buggy`` (pre-fix) or
``fixed`` (post-fix) from the ``eval_seed:`` line in ``slurm_logs/``.

  * the 2026-09-01 entry was written entirely on the PRE-FIX instrument. Its figures are
    therefore rebuilt on ``scheme="buggy"``, which reproduces that entry's tables exactly,
    and every one of them carries a banner saying so. Re-drawing them on the repaired
    instrument would not be a correction -- those arms were never re-evaluated -- it would be
    a silent substitution of one measurement for another.
  * the 2026-09-02 entry onwards is the repaired instrument, ``scheme="fixed"``.

WHAT COULD NOT BE REBUILT, AND IS SAID SO ON THE FIGURE

  * ``action-sensitivity.png``. The 2026-09-02 section 8 table (d_action ~ 1e-4, Spearman
    +0.81 over 9 arms) is not reproducible from anything in the repo: NONE of those nine arms
    has a logged ``causal/d_action``, because the diagnostic was added after they trained.
    2026-09-03 section 12b re-measured the quantity from the checkpoints and found the quoted
    baseline wrong by a factor of ~2900. The figure therefore shows BOTH: panel A is the
    section-8 table exactly as published, under a retraction banner, and panel B is the
    re-measurement from ``assets/d_action_probe.json``. Nothing is deleted and nothing is
    quietly rescaled.
  * ``consensus-scaling.png``. Section 3's "vs members' mean" and "vs members' best" rows
    cannot be rebuilt -- ``plan_outputs/`` does not record which checkpoints a vote's M
    columns were, and ``planning/ensemble.py`` takes the member list from the launcher. The
    panel shows the scaling, the paired effects and the catastrophe counts, which are all
    reconstructible, and says the rest is not.
  * ``root-cause.png`` panel A's entropy table (support bits/unit, magnitude bits|active) was
    measured on trained codes with a binning the repo does not record, so those two rows are
    carried over from the diary as published and marked as such. The information CURVE they
    sit on, and all of panel B, are computed here.

Usage:
    PYTHONPATH=. python analysis/early_figs.py                 # all twelve
    PYTHONPATH=. python analysis/early_figs.py --only power    # one, by file stem
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import sys

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import style as S  # noqa: E402
from analysis.style import C, FILL, INK, MUTED, GRID, EDGE, HEAD  # noqa: E402

S.use_style()

# scipy's noncentral-t evaluates at ncp=0 for the left tail of a two-sided
# power calculation and warns; the returned value is the correct 0.0.
warnings.filterwarnings("ignore", message=".*_nct_(sf|cdf).*")

OUT_01 = os.path.join(REPO, "diary/assets/2026-09-01")
OUT_02 = os.path.join(REPO, "diary/assets/2026-09-02")

# --------------------------------------------------------------------------------------
# the one arm -> hue map for this module, by IDENTITY and never by rank, so an arm keeps
# its colour across all twelve figures and across regenerations.
#
#   slate    the controls and the plain baselines           -- neutral, the reference
#   teal     the same baseline made WIDER (D=768..2048)     -- neutral family, also works
#   green    plan-time consensus, the campaign's one positive
#   amber    the gate family, the intervention under test
#   purple   the other representation-side interventions (pose, columns)
#   crimson  the resolved failures: k-WTA, the union head, the collapsed arms
# --------------------------------------------------------------------------------------
ARM_HUE = {
    "LpWM-base": "slate", "LpWM-ltv": "slate", "LpWM-ltv-vfloor": "slate",
    "LpWM-ltv-mupfix": "slate", "LpWM-ltv-relu-p2": "slate", "LpWM-linvar": "slate",
    "LpWM-ltv-d768": "teal", "LpWM-ltv-d1536": "teal", "LpWM-ltv-d2048": "teal",
    "PiWM-vote3-median": "green", "PiWM-vote5-median": "green", "PiWM-vote5-borda": "green",
    "PiWM-vote5-cvar1": "green", "PiWM-vote5-cvar2": "green", "PiWM-vote5-max": "green",
    "PiWMvoteM1": "green",
    "PiWM-gate-mag-softmax": "amber", "PiWM-gate-sup-sigmoid": "amber",
    "PiWM-gate-sup-softmax": "amber", "PiWM-gate-both": "amber",
    "PiWM-refframe": "purple", "PiWM-columns_patch": "purple", "PiWM-columns": "purple",
    "PiWM-pathint": "purple", "PiWM-actinfo": "purple",
    "PiWM-sparse-2pct": "crimson", "PiWM-sparse-matched": "crimson",
    "PiWM-kwta8-J1": "crimson", "PiWM-sdr-d2048-k41": "crimson",
    "PiWM-union2": "crimson", "PiWM-union4": "crimson", "PiWM-union4-entropy": "crimson",
    "PiWM-union4-kwta8": "crimson", "PiWM-union4-vfloor": "crimson",
    "PiWM-drop95_patch": "crimson", "PiWM-blockcausal": "crimson",
    "PiWM-sigreg": "crimson", "PiWM-sigreg-w0p5": "crimson", "PiWM-sigreg-arpred": "crimson",
}
OTHER = "#AEB8B4"           # arms outside a figure's design: one recessive grey
OTHER_FILL = "#DCE2E0"


def hue(arm):
    """The arm's hue key, or None for the background population.

    The PiWM-vote* prefix is matched as a family rather than enumerated: new voting rules
    keep landing, and an enumeration would silently drop each new one into the unlabelled
    grey the moment it appeared.
    """
    if arm in ARM_HUE:
        return ARM_HUE[arm]
    return "green" if arm.startswith("PiWM-vote") else None


def col(arm, default=OTHER):
    h = hue(arm)
    return C[h] if h else default


def fillc(arm, default=OTHER_FILL):
    h = hue(arm)
    return FILL[h] if h else default


# --------------------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------------------
RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")


def campaign(scheme):
    """{arm: {seed: success}} straight from plan_outputs, at the requested instrument.

    Called in-process rather than through the CLI so nothing writes over campaign.json.
    """
    from analysis.collect_evals import collect
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        arms, _detail, _pending = collect("plan_outputs", scheme=scheme)
    finally:
        os.chdir(cwd)
    return {a: {int(k): v for k, v in d.items()} for a, d in arms.items()
            if not a.startswith("CANARY-")}


_CFG_CACHE = {}


def run_cfg(run):
    """The run's hydra config, flattened to dotted keys. {} if it has none."""
    if run in _CFG_CACHE:
        return _CFG_CACHE[run]
    import yaml
    p = os.path.join(REPO, "runs/outputs", run, "hydra.yaml")
    out = {}
    if os.path.exists(p):
        def walk(d, pre=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    walk(v, pre + k + ".")
            else:
                out[pre[:-1]] = d
        try:
            walk(yaml.safe_load(open(p)))
        except Exception:
            out = {}
    _CFG_CACHE[run] = out
    return out


def summaries():
    """One row per completed run: arm, seed, the training diagnostics, and the config bits
    that identify its family. CANARY-* excluded, as in causal_figs.harvest."""
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        if name.startswith("CANARY-"):
            continue
        m = RUN_RE.match(name)
        f = os.path.join(d, "wandb/latest-run/files/wandb-summary.json")
        if not m or not os.path.exists(f):
            continue
        try:
            s = json.load(open(f))
        except Exception:
            continue
        cfg = run_cfg(name)
        rows.append(dict(
            run=name, arm=m.group(1), seed=int(m.group(2)),
            rho=s.get("sparsity/val_l0_frac"), rho_train=s.get("sparsity/train_l0_frac"),
            eff=s.get("sparsity/effective_dim"), rel_mse=s.get("err/rel_mse"),
            smodel=s.get("jacc/S_model"), epoch=s.get("progress/epoch_frac"),
            kwta=cfg.get("link.kwta_k"), n_heads=cfg.get("n_heads"),
            width=int(re.search(r"_pd(\d+)_", name).group(1)),
        ))
    return rows


def wandb_curve(run, keys):
    """A (len(keys),) column array of the run's full logged history, from its LOCAL wandb
    datastore. Sorted by the first key. Empty array if the run has no datastore.

    runs/export/*/wandb_history.csv is deliberately not used here -- see the module
    docstring: that export stops at 60-80% of training on the very arms whose late collapse
    these figures exist to show.
    """
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs/outputs", run, "wandb/run-*/*.wandb"))):
        ds = datastore.DataStore()
        try:
            ds.open_for_scan(f)
        except Exception:
            continue
        while True:
            try:
                raw = ds.scan_data()
            except Exception:
                break
            if raw is None:
                break
            rec = pb.Record()
            try:
                rec.ParseFromString(raw)
            except Exception:
                continue
            if rec.WhichOneof("record_type") != "history":
                continue
            d = {it.key: it.value_json for it in rec.history.item}
            if not all(k in d for k in keys):
                continue
            try:
                out.append(tuple(json.loads(d[k]) for k in keys))
            except Exception:
                continue
    if not out:
        return np.zeros((0, len(keys)))
    a = np.array(sorted(out), float)
    return a


def paired(arms, variant, control):
    """Paired effect on the seeds both arms have. Drops unmatched seeds rather than
    imputing them, and uses the t critical value at n-1 df -- a normal approximation
    badly understates the width at n=3."""
    ks = sorted(set(arms.get(variant, {})) & set(arms.get(control, {})))
    if len(ks) < 2:
        return None
    d = np.array([arms[variant][k] - arms[control][k] for k in ks], float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_1samp(d, 0.0)
    return dict(n=n, seeds=ks, d=d, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                se=float(se), lo=float(d.mean() - tc * se), hi=float(d.mean() + tc * se),
                t=float(t), p=float(p))


def power_paired(delta, sd, n, alpha=0.05):
    """Power of a two-sided paired t-test -- the exact noncentral-t expression, not the
    normal approximation, which at n=3 overstates power by a factor of two."""
    if n < 2 or sd <= 0:
        return float("nan")
    df, tc = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)
    ncp = delta / (sd / math.sqrt(n))
    return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))


def mde(sd, n, target=0.80):
    """Smallest true effect this design detects at `target` power. Bisection, because the
    power function has no closed-form inverse."""
    lo, hi = 0.0, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if power_paired(mid, sd, n) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# shared furniture
# --------------------------------------------------------------------------------------
def head(fig, title, sub=None, x=0.012, dy=0.30, dy_sub=0.62):
    """Figure header in INCHES down from the top edge, so it reads the same on a 4in figure
    and on a 17in one. Titles state the finding."""
    fh = fig.get_figheight()
    fig.text(x, 1 - dy / fh, title, fontsize=13, fontweight="bold", color=INK,
             ha="left", va="top")
    if sub:
        fig.text(x, 1 - dy_sub / fh, sub, fontsize=10, color=MUTED, ha="left", va="top",
                 linespacing=1.5)


def panel(fig, ax, letter, title, sub=None, dy_title=None, dy_sub=0.17, rule=True):
    """The reference's panel header, placed in inches above the axes.

    dy_title defaults to a value that CLEARS the subtitle: a two-line sub grows upward from
    dy_sub, so a fixed 0.40 collides with the title the moment a caption wraps. That is the
    single most common defect in this project's figures, so it is computed, not guessed.
    """
    bb = ax.get_position()
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    if dy_title is None:
        n_sub = 0 if not sub else sub.count("\n") + 1
        dy_title = 0.40 + 0.155 * max(n_sub - 1, 0)
    dx = 0.0
    if letter:
        fig.text(bb.x0, bb.y1 + dy_title / fh, letter, fontsize=11.5, fontweight="bold",
                 color=HEAD, ha="left", va="bottom")
        dx = 0.21 / fw
    fig.text(bb.x0 + dx, bb.y1 + dy_title / fh, title, fontsize=11.5, fontweight="bold",
             color=INK, ha="left", va="bottom")
    if sub:
        fig.text(bb.x0 + dx, bb.y1 + dy_sub / fh, sub, fontsize=9.5, color=MUTED,
                 ha="left", va="bottom")
    if rule:
        y = bb.y1 + 0.085 / fh
        fig.add_artist(Line2D([bb.x0, bb.x1], [y, y], color=EDGE, lw=1.0,
                              transform=fig.transFigure))


def note(fig, text, key="crimson", x=0.012, dy=0.0, fontsize=9.5):
    """A pale banner across the foot of a figure -- where an instrument caveat or a
    retraction lives, so it cannot be cropped off with the axes."""
    fig.text(x, dy, text, fontsize=fontsize, color=C[key], fontweight="bold",
             ha="left", va="bottom", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.5", fc=FILL[key], ec=C[key], lw=0.9))


def callout(ax, x, y, text, key="slate", fontsize=9.5, ha="left", va="top",
            coords="axes fraction", weight="bold", **kw):
    return ax.annotate(text, (x, y), xycoords=coords, ha=ha, va=va, fontsize=fontsize,
                       color=C[key], fontweight=weight, zorder=20, linespacing=1.4,
                       bbox=dict(boxstyle="round,pad=0.45", fc=FILL[key], ec=C[key],
                                 lw=0.9), **kw)


def save(fig, out, name, dpi=170):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print("wrote", os.path.relpath(p, REPO))
    return p


PREFIX_BANNER = (
    "PRE-FIX EVAL INSTRUMENT.  Every CEM number on this figure was measured before 49a3e55, "
    "when plan.py:134 collapsed a seed-0 eval to one episode repeated 50x.  These arms were "
    "never re-evaluated, so this is the\nmeasurement the 2026-09-01 entry was written on and "
    "the one reproduced here.  It is superseded by the 2026-09-02 campaign and must not be "
    "pooled with it."
)


# ======================================================================================
# 2026-09-01  --  the sparse / SDR era, on the pre-fix eval instrument
# ======================================================================================
ARMS_0901 = ["LpWM-base", "LpWM-ltv", "PiWM-gate-mag-softmax", "PiWM-gate-sup-sigmoid",
             "PiWM-gate-sup-softmax", "PiWM-sparse-2pct", "PiWM-union4",
             "PiWM-union4-entropy"]
LAB_0901 = {
    "LpWM-base": "LpWM-base\n(mlp_var, control)", "LpWM-ltv": "LpWM-ltv\n(LTV, control)",
    "PiWM-gate-mag-softmax": "gate\nmag x softmax",
    "PiWM-gate-sup-sigmoid": "gate\nsup x sigmoid",
    "PiWM-gate-sup-softmax": "gate\nsup x softmax",
    "PiWM-sparse-2pct": "k-WTA\n2%", "PiWM-union4": "union\nJ=4",
    "PiWM-union4-entropy": "union J=4\n+ entropy",
}


def fig_cem_effects(out=OUT_01):
    """Section 1. Every arm's seeds, the paired effect against the matched control, and the
    thing the entry actually learned: the SAME variant resolves or does not depending on
    which control it is anchored on, because LpWM-ltv has a dead seed and LpWM-base does not.
    """
    arms = campaign("buggy")
    present = [a for a in ARMS_0901 if a in arms]
    variants = [a for a in present if a not in ("LpWM-base", "LpWM-ltv")]

    FW, FH = 14.0, 9.4
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.052, 0.545, 0.930, 0.275])
    bx = fig.add_axes([0.052, 0.135, 0.330, 0.250])
    cx = fig.add_axes([0.655, 0.135, 0.250, 0.250])

    # --- A: the seeds -----------------------------------------------------------------
    ctrl_seeds = arms.get("LpWM-ltv", {})
    ticklabs = []
    for i, a in enumerate(present):
        v = arms[a]
        c = col(a)
        # matched-seed connectors from the LTV control, drawn under everything
        if a not in ("LpWM-base", "LpWM-ltv"):
            for s_, y in v.items():
                if s_ in ctrl_seeds:
                    ax.plot([1, i], [ctrl_seeds[s_], y], color="#EBEBEB", lw=0.9, zorder=1)
        ys = np.array([v[s_] for s_ in sorted(v)], float)
        jit = np.linspace(-0.13, 0.13, len(ys)) if len(ys) > 1 else np.array([0.0])
        is_ctrl = a in ("LpWM-base", "LpWM-ltv")
        ax.scatter(i + jit, ys, s=54, marker="s" if is_ctrl else "o",
                   facecolor="white" if is_ctrl else c, edgecolor=c, lw=1.5, zorder=4)
        ax.plot([i - 0.28, i + 0.28], [ys.mean()] * 2, color=c, lw=3.2, zorder=5,
                solid_capstyle="butt")
        # the mean sits ABOVE the arm's highest seed, never on top of a mark
        ax.text(i, ys.max() + 0.022, f"{ys.mean():.3f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=c, zorder=6)
        sd = f"sd {ys.std(ddof=1):.3f}" if len(ys) > 1 else "n=1"
        ticklabs.append(f"{LAB_0901[a]}\n{sd}")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(ticklabs, fontsize=9.5)
    for t, a in zip(ax.get_xticklabels(), present):
        t.set_color(col(a))
        t.set_fontweight("bold")
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylim(-0.03, 0.62)
    ax.set_ylabel("CEM planning success", fontsize=10.5, color=INK)
    S.ax_style(ax)
    ax.tick_params(axis="x", length=0, pad=9)
    if 1 in ctrl_seeds and ctrl_seeds[1] == 0.0:
        # ring the dead seed and put the note in the empty right half -- a leader line
        # from here would have to cross three arms' worth of marks to reach it
        ax.scatter([1], [0.0], s=330, facecolor="none", edgecolor=C["crimson"], lw=2.0,
                   zorder=7)
        sd_ltv = np.std(list(ctrl_seeds.values()), ddof=1)
        sd_base = np.std(list(arms["LpWM-base"].values()), ddof=1)
        callout(ax, 0.585, 0.94,
                "LpWM-ltv seed 1 = 0.000 (ringed).  That one dead seed gives the control\n"
                f"sd {sd_ltv:.3f} against LpWM-base's {sd_base:.3f}, and it is what destroys\n"
                "the power of every contrast anchored on it.",
                "crimson", fontsize=9.5)
    ax.scatter([], [], s=54, marker="s", facecolor="white", edgecolor=MUTED, lw=1.5,
               label="control (flags off)")
    ax.scatter([], [], s=54, marker="o", facecolor=MUTED, edgecolor=MUTED, lw=1.5,
               label="variant")
    ax.legend(loc="upper left", fontsize=9.5, handletextpad=0.4, ncol=1,
              bbox_to_anchor=(0.30, 1.03))
    panel(fig, ax, "A", "Three seeds per arm; the two controls differ in spread, not only "
                        "in configuration",
          "Thin lines join one training seed across arms.  Arm means run 0.013-0.240 for the "
          "PiWM arms against LpWM-base 0.340 and LpWM-ltv 0.300.")

    # --- B: paired effects vs the matched LTV control ---------------------------------
    eff = [(a, paired(arms, a, "LpWM-ltv")) for a in variants]
    eff = [(a, e) for a, e in eff if e]
    eff.sort(key=lambda t: t[1]["mean"])
    tr = bx.get_yaxis_transform()
    for i, (a, e) in enumerate(eff):
        floor = mde(e["sd"], e["n"])
        inside = abs(e["mean"]) < floor
        c = col(a)
        bx.plot([e["lo"], e["hi"]], [i, i], color=c, lw=2.2, solid_capstyle="round",
                zorder=3)
        bx.scatter([e["mean"]], [i], s=95, facecolor="white" if inside else c,
                   edgecolor=c, lw=2.0, zorder=4)
        # the numeric columns live OUTSIDE the axes, so a digit can never land on a mark
        for xf, txt, cc in ((1.04, f"{e['mean']:+.3f}", c),
                            (1.16, f"n={e['n']}", MUTED),
                            (1.27, f"p={e['p']:.3f}", MUTED)):
            bx.text(xf, i, txt, transform=tr, va="center", ha="left", fontsize=9.5,
                    color=cc, clip_on=False)
    bx.axvline(0, color=MUTED, lw=1.0, zorder=2)
    bx.set_yticks(range(len(eff)))
    bx.set_yticklabels([LAB_0901[a].replace("\n", " ") for a, _ in eff], fontsize=9.5)
    for t, (a, _) in zip(bx.get_yticklabels(), eff):
        t.set_color(col(a))
    bx.set_ylim(-0.6, len(eff) - 0.4)
    bx.set_xlim(-0.62, 0.62)
    bx.set_xlabel("paired difference in CEM success vs LpWM-ltv   (95% t CI)",
                  fontsize=10.5, color=INK)
    S.ax_style(bx, grid="x")
    panel(fig, bx, "B", "Anchored on LpWM-ltv, nothing resolves",
          "Hollow = the effect is smaller than this contrast's own 80%-power\n"
          "detection floor: underpowered, not null.")

    # --- C: the same arms re-anchored on the stable control ---------------------------
    rows = [("PiWM-union4", "LpWM-base"), ("PiWM-union4-entropy", "LpWM-base"),
            ("PiWM-gate-sup-softmax", "LpWM-ltv")]
    rows = [(v, c) for v, c in rows if paired(arms, v, c)]
    trc = cx.get_yaxis_transform()
    for i, (v, ctrl) in enumerate(rows):
        e = paired(arms, v, ctrl)
        res = e["p"] < 0.05
        c = col(v)
        cx.plot([e["lo"], e["hi"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
        cx.scatter([e["mean"]], [i], s=95, facecolor=c if res else "white", edgecolor=c,
                   lw=2.0, zorder=4)
        cx.text(0.03, i + 0.22, f"vs {ctrl}    t = {e['t']:+.2f}    p = {e['p']:.4f}"
                                 + ("    RESOLVED" if res else ""),
                transform=trc, ha="left", va="bottom",
                fontsize=9.5, color=C["green"] if res else MUTED,
                fontweight="bold" if res else "normal")
    cx.axvline(0, color=MUTED, lw=1.0)
    cx.set_yticks(range(len(rows)))
    cx.set_yticklabels([LAB_0901[v].replace("\n", " ") for v, _ in rows], fontsize=9.5)
    for t, (v, _) in zip(cx.get_yticklabels(), rows):
        t.set_color(col(v))
    cx.set_ylim(-0.6, len(rows) - 0.25)
    cx.set_xlim(-1.15, 0.65)
    cx.set_xticks([-1.0, -0.5, 0.0, 0.5])
    cx.set_xlabel("paired difference vs the control named on each row", fontsize=10.5,
                  color=INK)
    S.ax_style(cx, grid="x")
    panel(fig, cx, "C", "Anchored on LpWM-base, two do",
          "Same variants, same seeds, different control.\nChoose a control by its variance, "
          "not only by its config.")

    head(fig, "The campaign was not underpowered in general -- it was underpowered against a "
              "control with a dead seed",
         "Six PiWM arms, three seeds each, against two baselines that differ in seed spread.  "
         "Re-anchoring the two union contrasts on the stable control resolves both at n=3; "
         "the Step-3 gate deficit\nstays inside the detection floor either way.",
         dy=0.30, dy_sub=0.62)
    note(fig, PREFIX_BANNER, "crimson", x=0.012, dy=0.010)
    return save(fig, out, "cem-effects.png")


def fig_factorial(out=OUT_01):
    """Section 2. The Step-3 proposal is one cell of a 2x2, both factors are negative, they
    are additive, and the per-seed ordering is monotone on every usable pair."""
    arms = campaign("buggy")
    cells = {("magnitude", "sigmoid"): "LpWM-ltv",
             ("magnitude", "softmax"): "PiWM-gate-mag-softmax",
             ("support", "sigmoid"): "PiWM-gate-sup-sigmoid",
             ("support", "softmax"): "PiWM-gate-sup-softmax"}
    rows_, cols_ = ["magnitude", "support"], ["sigmoid", "softmax"]
    M = np.full((2, 2), np.nan)
    for (r, c), a in cells.items():
        if a in arms:
            M[rows_.index(r), cols_.index(c)] = np.mean(list(arms[a].values()))
    ctrl = M[0, 0]
    e_support = 0.5 * ((M[1, 0] - M[0, 0]) + (M[1, 1] - M[0, 1]))
    e_softmax = 0.5 * ((M[0, 1] - M[0, 0]) + (M[1, 1] - M[1, 0]))
    inter = (M[1, 1] - M[1, 0]) - (M[0, 1] - M[0, 0])

    FW, FH = 13.6, 6.6
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.052, 0.150, 0.335, 0.590])
    bx = fig.add_axes([0.485, 0.150, 0.230, 0.590])
    cx = fig.add_axes([0.815, 0.150, 0.150, 0.590])

    # --- A: the 2x2 itself, on the single-hue magnitude ramp --------------------------
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("g", S.SEQ_GREEN)
    lo, hi = np.nanmin(M), np.nanmax(M)
    ax.imshow(M, cmap=ramp, vmin=lo - 0.02, vmax=hi + 0.02, aspect="auto",
              extent=[-0.5, 1.5, 1.5, -0.5])
    for i in range(2):
        for j in range(2):
            a = cells[(rows_[i], cols_[j])]
            if a not in arms:
                continue
            v = M[i, j]
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
            tc = "white" if dark else INK
            ax.text(j, i - 0.19, f"{v:.3f}", ha="center", va="center", fontsize=20,
                    fontweight="bold", color=tc)
            tag = "control" if a == "LpWM-ltv" else f"{v - ctrl:+.3f}"
            ax.text(j, i + 0.10, tag, ha="center", va="center", fontsize=11, color=tc)
            trip = "  ".join(f"{arms[a][s]:.2f}" for s in sorted(arms[a]))
            ax.text(j, i + 0.30, trip, ha="center", va="center", fontsize=9.5,
                    color=tc, alpha=0.85)
    v11 = M[1, 1]
    ax.text(1, 1 - 0.42, "Step 3 proposal", ha="center", va="center", fontsize=9.5,
            fontweight="bold",
            color="white" if (v11 - lo) / max(hi - lo, 1e-9) > 0.55 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(cols_, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(rows_, fontsize=11)
    ax.set_xlabel("gate normalisation", fontsize=10.5, color=INK)
    ax.set_ylabel("gate input", fontsize=10.5, color=INK)
    ax.tick_params(length=0, labelcolor=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    panel(fig, ax, "A", "Both factors are negative",
          "cell = mean CEM success; small text = the three seeds.")

    # --- B: the per-seed ladders ------------------------------------------------------
    order = [cells[("magnitude", "sigmoid")], cells[("magnitude", "softmax")],
             cells[("support", "sigmoid")], cells[("support", "softmax")]]
    short = ["mag\nsigmoid", "mag\nsoftmax", "sup\nsigmoid", "sup\nsoftmax"]
    usable = [s for s in sorted(arms.get("LpWM-ltv", {})) if arms["LpWM-ltv"][s] > 0]
    for s in usable:
        ys = [arms[a][s] for a in order if s in arms.get(a, {})]
        if len(ys) < len(order):
            continue
        mono = all(ys[k] >= ys[k + 1] for k in range(len(ys) - 1))
        bx.plot(range(len(ys)), ys, "-o", color=C["amber"] if mono else MUTED,
                lw=2.0, ms=7, mfc="white", mew=1.8, zorder=4)
        bx.text(len(ys) - 1 + 0.12, ys[-1], f"seed {s}", va="center", ha="left",
                fontsize=9.5, color=C["amber"] if mono else MUTED, fontweight="bold")
    bx.set_xticks(range(4)); bx.set_xticklabels(short, fontsize=9.5)
    bx.set_xlim(-0.35, 3.75)
    bx.set_ylim(0, 0.70)
    bx.set_ylabel("CEM success", fontsize=10.5, color=INK)
    S.ax_style(bx)
    bx.tick_params(axis="x", length=0, pad=6)
    n_pairs = sum(1 for s in usable for a in order[1:]
                  if s in arms.get(a, {}) and arms[a][s] < arms["LpWM-ltv"][s])
    n_tot = sum(1 for s in usable for a in order[1:] if s in arms.get(a, {}))
    callout(bx, 0.02, 0.985, f"monotone decreasing on both usable seeds;\n"
                             f"variant below control on {n_pairs}/{n_tot} (seed, arm) pairs",
            "amber", fontsize=9.5)
    panel(fig, bx, "B", "and the ordering is monotone",
          "The dead control seed is excluded: it is 0.000 in every cell.")

    # --- C: the decomposition ---------------------------------------------------------
    labs = ["support\n(main)", "softmax\n(main)", "interaction"]
    vals = [e_support, e_softmax, inter]
    cs = [C["amber"], C["amber"], C["slate"]]
    cx.bar(range(3), vals, width=0.6, color=cs, zorder=3)
    for i, v in enumerate(vals):
        cx.text(i, v - 0.006, f"{v:+.3f}", ha="center", va="top", fontsize=10,
                fontweight="bold", color=cs[i])
    cx.axhline(0, color=MUTED, lw=1.0)
    cx.set_xticks(range(3)); cx.set_xticklabels(labs, fontsize=9.5)
    cx.set_ylim(-0.115, 0.028)
    cx.set_ylabel("effect on CEM success", fontsize=10.5, color=INK)
    S.ax_style(cx)
    cx.tick_params(axis="x", length=0, pad=6)
    panel(fig, cx, "C", "additive",
          "the interaction is a tenth\nof either main effect.")

    head(fig, "Step 3 is one cell of a 2x2 and it is the cell that takes both penalties",
         "Gate input (magnitude -> support) x normalisation (sigmoid -> softmax), three seeds "
         "per cell.  The CIs at n=3 all span zero; the SIGNS do not.")
    note(fig, PREFIX_BANNER, "crimson", x=0.012, dy=0.012)
    return save(fig, out, "factorial-2x2.png")


# --- section 3 information table -------------------------------------------------------
# Measured on trained codes (389 frames x 384 units) for the 2026-09-01 entry. The binning
# behind the two entropy columns is not recorded anywhere in the repo, so these two rows are
# carried over from the diary AS PUBLISHED rather than re-derived, and the figure says so.
# rho is the one column that IS re-derivable, and the run summaries agree with it to 0.001
# (LpWM-ltv val_l0_frac 0.553 vs 0.554; gate-sup-softmax 0.610 vs 0.610).
INFO_TABLE = [
    dict(arm="LpWM-ltv", rho=0.554, sup_bits=0.636, mag_bits=3.676, discarded=0.762,
         I_s=0.614, I_z=0.979, retained=0.628),
    dict(arm="PiWM-gate-sup-softmax", rho=0.610, sup_bits=0.519, mag_bits=3.271,
         discarded=0.794, I_s=0.652, I_z=1.136, retained=0.574),
]


def fig_root_cause(out=OUT_01):
    """Section 3. Two mechanisms, both of them arithmetic rather than empirical: binarising
    the code throws away most of its bits (a), and OR-ing J readouts of a rho~0.55 code
    saturates (b) -- except that what the union arms actually did was worse than saturate."""
    H2 = lambda r: -(r * np.log2(np.clip(r, 1e-12, 1)) +
                     (1 - r) * np.log2(np.clip(1 - r, 1e-12, 1)))
    rows = summaries()
    by = collections.defaultdict(list)
    for r in rows:
        if r["rho_train"] is not None:
            by[r["arm"]].append(r)

    FW, FH = 14.2, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.048, 0.170, 0.235, 0.560])
    bx = fig.add_axes([0.372, 0.170, 0.160, 0.560])
    cx = fig.add_axes([0.640, 0.170, 0.345, 0.560])

    # --- A: what binarising keeps and what it discards ---------------------------------
    xs = np.arange(len(INFO_TABLE))
    for i, d in enumerate(INFO_TABLE):
        kept, disc = d["sup_bits"], d["rho"] * d["mag_bits"]
        ax.bar(i, kept, width=0.52, color=C["green"], zorder=3,
               label="support: what the gate keeps" if i == 0 else None)
        ax.bar(i, disc, width=0.52, bottom=kept, color=C["crimson"], zorder=3,
               label="magnitude: what binarising discards" if i == 0 else None)
        ax.text(i, kept / 2, f"{kept:.3f}", ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold")
        ax.text(i, kept + disc / 2, f"{disc:.3f}", ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold")
        ax.text(i, kept + disc + 0.10, f"{d['discarded']:.1%} discarded", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold", color=C["crimson"])
        # the analytic ceiling: a binary support cannot carry more than H2(rho) per unit.
        # Drawn in the gutter BESIDE the bar, never across it.
        ax.plot([i + 0.30, i + 0.60], [H2(d["rho"])] * 2, color=C["slate"], lw=1.8,
                ls=(0, (4, 2)), zorder=5,
                label="ceiling: H2(rho), the most a binary support can hold"
                      if i == 0 else None)
        ax.text(i + 0.45, H2(d["rho"]) + 0.05, f"{H2(d['rho']):.2f}", va="bottom",
                ha="center", fontsize=9, color=C["slate"], fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d['arm']}\nrho = {d['rho']:.3f}" for d in INFO_TABLE],
                       fontsize=9.5)
    for t, d in zip(ax.get_xticklabels(), INFO_TABLE):
        t.set_color(col(d["arm"]))
        t.set_fontweight("bold")
    ax.set_xlim(-0.45, len(xs) - 0.28)
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("information per unit (bits)", fontsize=10.5, color=INK)
    S.ax_style(ax)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.legend(loc="upper left", fontsize=8.5, handlelength=1.4, handletextpad=0.5,
              labelspacing=0.35)
    panel(fig, ax, "A", "Binarising the code discards three quarters of its bits",
          "src = 1[z > 0] is a deterministic function of z, so I(s;Y) <= I(z;Y) for every\n"
          "target: support gating can only ever win as an inductive bias, never on bits.")

    # --- B: and what survives of the PREDICTIVE information ---------------------------
    w = 0.30
    for i, d in enumerate(INFO_TABLE):
        bx.bar(i - w / 2 - 0.02, d["I_z"], width=w, color=C["slate"], zorder=3,
               label="I(z_t ; z_t+1)" if i == 0 else None)
        bx.bar(i + w / 2 + 0.02, d["I_s"], width=w, color=C["green"], zorder=3,
               label="I(s_t ; s_t+1)" if i == 0 else None)
        bx.text(i, max(d["I_z"], d["I_s"]) + 0.05, f"{d['retained']:.1%}\nretained",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=C["green"],
                linespacing=1.3)
    bx.set_xticks(range(len(INFO_TABLE)))
    bx.set_xticklabels(["LpWM-ltv", "gate\nsup x softmax"], fontsize=9.5)
    for t, d in zip(bx.get_xticklabels(), INFO_TABLE):
        t.set_color(col(d["arm"]))
        t.set_fontweight("bold")
    bx.set_xlim(-0.55, len(INFO_TABLE) - 0.45)
    bx.set_ylim(0, 1.72)
    bx.set_ylabel("one-step predictive information (bits)", fontsize=10.5, color=INK)
    S.ax_style(bx)
    bx.tick_params(axis="x", length=0, pad=7)
    bx.legend(loc="upper left", fontsize=9, handlelength=1.2, handletextpad=0.5)
    panel(fig, bx, "B", "24% of the bits, 63% of the prediction",
          "The TBT premise is partly validated:\nthe support is a cheap carrier.")

    # --- C: the union saturates, and then it did worse than saturate -------------------
    J = np.arange(1, 17)
    for rho_, key, lab in ((0.55, "crimson", r"$\rho$ = 0.55  (our operating point)"),
                           (0.10, "slate", r"$\rho$ = 0.10"),
                           (0.02, "green", r"$\rho$ = 0.02  (a real SDR)")):
        y = 1 - (1 - rho_) ** J
        cx.plot(J, y, "-o", color=C[key], lw=2.0, ms=4.5, zorder=4)
        cx.text(16.4, y[-1], lab, va="center", ha="left", fontsize=9.5, color=C[key],
                fontweight="bold")
    cx.axhspan(0, 0.20, color=FILL["green"], lw=0, zorder=1)

    y4 = 1 - (1 - 0.55) ** 4
    cx.scatter([4], [y4], s=190, facecolor="none", edgecolor=C["crimson"], lw=2.2, zorder=6)
    cx.annotate("J = 4 at our density:\n" + f"{y4:.0%} of the code is ON",
                xy=(4.4, y4 - 0.01), xytext=(5.8, 0.735), fontsize=9.5, color=C["crimson"],
                fontweight="bold", va="center", ha="left", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.42", fc=FILL["crimson"], ec=C["crimson"],
                          lw=0.9),
                arrowprops=dict(arrowstyle="-", color=C["crimson"], lw=1.0))
    # what the arms MEASURED, on the same axes as the prediction
    meas = []
    for arm, Jv in (("PiWM-union2", 2), ("PiWM-union4", 4)):
        v = sorted(float(r["rho_train"]) for r in by.get(arm, []))
        if not v:
            continue
        txt = f"{v[0]:.4f}" if len(v) == 1 else f"{v[0]:.4f}-{v[-1]:.4f}"
        meas.append((Jv, txt, len(v)))
        cx.scatter([Jv] * len(v), v, s=115, marker="X", color=C["crimson"], zorder=7,
                   edgecolor="white", lw=0.8,
                   label="measured density after training (one X per seed)"
                         if Jv == 2 else None)
    cx.set_xlim(0.4, 21.9)
    cx.set_ylim(-0.03, 1.07)
    cx.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    cx.set_xlabel("number of union heads J", fontsize=10.5, color=INK)
    cx.set_ylabel("fraction of the code ON after the union", fontsize=10.5, color=INK)
    S.ax_style(cx)
    cx.legend(loc="lower right", fontsize=9, handletextpad=0.5,
              bbox_to_anchor=(1.0, 0.02))
    sub_c = (r"OR of J independent $\rho$-dense patterns is ON with probability "
             r"$1-(1-\rho)^J$." + "\n")
    sub_c += ("Measured after training: " +
              ",  ".join(f"J={j} -> {t} ({n} seed{'s' if n > 1 else ''})"
                         for j, t, n in meas) +
              ".  min_j L_j is minimised by a dead code." if meas else "")
    panel(fig, cx, "C", "The union saturates -- and the arms did worse than saturate", sub_c)

    head(fig, "Both Step-3 and Step-4 fail for arithmetic reasons, before any experiment is "
              "run",
         "We transplanted SDR operations onto a non-SDR substrate.  In a Numenta SDR "
         "(binary, 0.05-2% sparse) magnitudes do not exist, so discarding them is free; here "
         "they carry the other 37%.")
    note(fig, "Panel A/B's entropy and mutual-information columns are the 2026-09-01 "
              "measurement on trained codes (389 frames x 384 units) as published -- the "
              "binning is not recorded in the repo, so they are\ncarried over rather than "
              "re-derived.  Every curve, every bound, and panel C's measured densities are "
              "computed here from the current runs.", "slate", x=0.012, dy=0.010)
    return save(fig, out, "root-cause.png")


CURVE_ARMS_0901 = ["LpWM-ltv", "PiWM-gate-mag-softmax", "PiWM-gate-sup-sigmoid",
                   "PiWM-gate-sup-softmax", "PiWM-union4-entropy", "PiWM-union4"]
CURVE_LAB_0901 = {"LpWM-ltv": "LpWM-ltv (control)", "PiWM-gate-mag-softmax": "gate mag x softmax",
                  "PiWM-gate-sup-sigmoid": "gate sup x sigmoid",
                  "PiWM-gate-sup-softmax": "gate sup x softmax",
                  "PiWM-union4": "union J=4", "PiWM-union4-entropy": "union J=4 + entropy"}


def _band(runs, key, grid):
    """Mean, min and max of `key` across seeds on a shared epoch grid.

    Every run logs on its own step schedule and resumes at different points, so the curves
    are interpolated onto one grid before they are pooled -- averaging raw index-aligned
    arrays would mix epoch 0.4 of one seed with epoch 0.7 of another.
    """
    ys = []
    for run in runs:
        a = wandb_curve(run, ["progress/epoch_frac", key])
        if a.shape[0] < 8:
            continue
        m = np.isfinite(a[:, 1])
        a = a[m]
        y = np.interp(grid, a[:, 0], a[:, 1], left=np.nan, right=np.nan)
        ys.append(y)
    if not ys:
        return None
    Y = np.vstack(ys)
    with np.errstate(invalid="ignore"):
        return np.nanmean(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0), len(ys)


def fig_union_collapse(out=OUT_01):
    """Section 3(b). The union arms' code density over training, and the separate finding
    that k-WTA holds its floor EXACTLY everywhere except under the union head -- which is
    what makes the union a failure of the objective rather than of the operating point."""
    rows = summaries()
    runs_of = collections.defaultdict(list)
    for r in rows:
        if r["seed"] <= 2:                     # the 2026-09-01 round is seeds 0-2
            runs_of[r["arm"]].append(r["run"])
    grid = np.linspace(0.0, 1.0, 260)

    FW, FH = 15.2, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.042, 0.155, 0.245, 0.550])
    bx = fig.add_axes([0.452, 0.155, 0.180, 0.550])
    cx = fig.add_axes([0.800, 0.155, 0.180, 0.550])

    for axis, key, ylab, ylim in (
            (ax, "sparsity/train_l0_frac", r"code density  $\rho$", (-0.02, 0.72)),
            (bx, "sparsity/l0_std_across_samples",
             "std of per-sample L0 across the batch", (-2.0, 78.0))):
        axis.set_ylim(*ylim)
        ends = []
        for arm in CURVE_ARMS_0901:
            got = _band(sorted(runs_of.get(arm, [])), key, grid)
            if got is None:
                continue
            mu, lo, hi, n = got
            c = col(arm)
            union = arm.startswith("PiWM-union")
            axis.fill_between(grid, lo, hi, color=fillc(arm), alpha=0.55, lw=0, zorder=2)
            axis.plot(grid, mu, color=c, lw=2.4 if union else 1.8,
                      ls="-" if union else (0, (5, 2)), zorder=4)
            last = np.where(np.isfinite(mu))[0]
            if last.size:
                ends.append((mu[last[-1]], arm, c, n))
        # direct labels, de-collided by pushing apart in DATA units after sorting
        ends.sort(key=lambda t: t[0])
        span = axis.get_ylim()[1] - axis.get_ylim()[0]
        minsep = 0.055 * span
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < minsep:
                ys[i] = ys[i - 1] + minsep
        for (y0, arm, c, n), y in zip(ends, ys):
            axis.plot([1.005, 1.035], [y0, y], color=c, lw=0.9, clip_on=False, zorder=5)
            axis.text(1.045, y, f"{CURVE_LAB_0901[arm]}  n={n}", va="center", ha="left",
                      fontsize=9, color=c, fontweight="bold", clip_on=False)
        axis.set_xlim(0, 1.0)
        axis.set_ylim(*ylim)
        axis.set_xlabel("epoch (fractional)", fontsize=10.5, color=INK)
        axis.set_ylabel(ylab, fontsize=10.5, color=INK)
        S.ax_style(axis)
    callout(ax, 0.26, 0.56,
            r"PiWM-union4 finishes at $\rho$ = 0.0000," "\n"
            "effective_dim 0.0, S_model 1.0000.",
            "crimson", fontsize=9.5)
    panel(fig, ax, "A", "The union arms crash in the first quarter-epoch",
          "Line = mean over the round's three seeds, band = their min-max.\n"
          "Solid = union head, dashed = every other arm.")
    panel(fig, bx, "B", "and stop varying between samples",
          "A code that is the same for every input\nhas nothing left to plan with.")

    # --- C: k-WTA holds its floor everywhere except under the union -------------------
    by = collections.defaultdict(list)
    for r in rows:
        if r["kwta"] and r["rho_train"] is not None:
            by[r["arm"]].append(r)
    order = ["PiWM-sparse-2pct", "PiWM-sparse-matched", "PiWM-union4-kwta8"]
    order = [a for a in order if by.get(a)]
    for i, arm in enumerate(order):
        v = by[arm]
        target = v[0]["kwta"] / v[0]["width"]
        realised = np.array([x["rho_train"] for x in v], float)
        breach = realised.max() < target * 0.9
        c = C["crimson"] if breach else C["slate"]
        cx.plot([i - 0.30, i + 0.30], [target] * 2, color=C["green"], lw=2.4, zorder=3,
                label="configured k/D floor" if i == 0 else None)
        cx.scatter(np.full(realised.size, i), realised, s=70, color=c, zorder=5,
                   edgecolor="white", lw=0.8,
                   label="realised density (one dot per seed)" if i == 0 else None)
        cx.text(i, max(target, realised.max()) * 1.55 + 3e-4,
                "HOLDS exactly" if not breach else "BREACHED",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=c)
    cx.set_yscale("log")
    cx.set_ylim(4e-5, 1.0)
    cx.set_xlim(-0.6, len(order) - 0.4)
    cx.set_xticks(range(len(order)))
    cx.set_xticklabels(["k-WTA 2%\nk=8, J=1", "k-WTA matched\nk=113, J=1",
                        "k-WTA 2%\nk=8, J=4"][:len(order)], fontsize=9)
    for t, a in zip(cx.get_xticklabels(), order):
        t.set_color(col(a))
        t.set_fontweight("bold")
    cx.tick_params(axis="x", length=0, pad=7)
    cx.set_ylabel(r"end-of-training density  $\rho$  (log)", fontsize=10.5, color=INK)
    S.ax_style(cx)
    cx.legend(loc="lower left", fontsize=8.5, handletextpad=0.5, labelspacing=0.3)
    panel(fig, cx, "C", "The union breaks k-WTA's own floor",
          "k-WTA marks k positions whose values MAY BE ZERO.\nThe union drives pre-link "
          "activations negative, so top-k\nselects units holding zeros.")

    head(fig, "The union head does not saturate the code -- it kills it, and it is the "
              "objective that does the killing",
         "min_j L_j admits z = 0 as a global optimum.  Pinning sparsity with k-WTA was built "
         "to test whether the union survives at a fixed operating point; it refutes that, "
         "because the union breaches the\nfloor k-WTA holds exactly in every non-union arm.")
    note(fig, PREFIX_BANNER.split(".")[0] + ".  This figure's curves are TRAINING metrics and "
              "are unaffected by the eval bug; only the arms' CEM numbers, quoted in the "
              "2026-09-01 entry, are on the pre-fix instrument.",
         "slate", x=0.012, dy=0.010)
    return save(fig, out, "union-collapse.png")


def fig_power(out=OUT_01):
    """Section 4. Why almost nothing resolved at n=3, and why the fix is more SEEDS rather
    than more eval episodes: training variance, not eval variance, is what dominates."""
    arms = campaign("buggy")
    ref_v, ref_c = "PiWM-gate-sup-softmax", "LpWM-ltv"
    e = paired(arms, ref_v, ref_c)
    sd = e["sd"]
    p_ctrl = float(np.mean(list(arms[ref_c].values())))
    n_evals = 50
    sd_eval = math.sqrt(p_ctrl * (1 - p_ctrl) / n_evals)
    sd_train = math.sqrt(max(sd ** 2 - sd_eval ** 2, 0.0))

    variants = [a for a in ARMS_0901 if a not in ("LpWM-base", ref_c) and a in arms]
    obs = [(a, paired(arms, a, ref_c)) for a in variants]
    obs = [(a, x) for a, x in obs if x]

    FW, FH = 13.8, 6.6
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.050, 0.155, 0.520, 0.570])
    bx = fig.add_axes([0.688, 0.155, 0.290, 0.570])

    # --- A: power curves at the contrast's OWN paired sd ------------------------------
    xs = np.linspace(0, 0.62, 320)
    ramp = [S.SEQ_GREEN[5], S.SEQ_GREEN[4], S.SEQ_GREEN[3], S.SEQ_GREEN[2]]
    for (n, cc) in zip((3, 5, 8, 16), reversed(ramp)):
        y = [power_paired(x, sd, n) for x in xs]
        m = mde(sd, n)
        ax.plot(xs, y, color=cc, lw=2.4, zorder=4)
        ax.plot([m, m], [0, 0.80], color=cc, lw=1.1, ls=(0, (3, 2)), zorder=3)
        # the MDE labels run ALONG their own drop-lines: four of them side by side at the
        # top would otherwise overprint, and n=8 and n=16 are only 0.05 apart in x
        ax.text(m - 0.007, 0.30, f"n = {n}    MDE {m:.3f}", rotation=90, ha="center",
                va="bottom", fontsize=9.5, color=cc, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
    ax.axhline(0.80, color=MUTED, lw=1.0, zorder=2)
    ax.text(0.617, 0.80, "80% power", ha="right", va="bottom", fontsize=9, color=MUTED)
    for a, x in obs:
        ax.plot([abs(x["mean"])] * 2, [0.005, 0.055], color=col(a), lw=3.0,
                solid_capstyle="butt", zorder=6)
    n_below = sum(1 for _a, x in obs if abs(x["mean"]) < mde(sd, 3))
    ax.text(0.315, 0.075, f"ticks = the {len(obs)} measured |effects|;\n"
                          f"{n_below} fall below the n=3 floor",
            fontsize=9.5, color=MUTED, va="bottom", ha="left", linespacing=1.4)
    ax.set_xlim(0, 0.62)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("true paired effect on CEM success", fontsize=10.5, color=INK)
    ax.set_ylabel(r"power at $\alpha$ = 0.05 (two-sided)", fontsize=10.5, color=INK)
    S.ax_style(ax, grid="both")
    panel(fig, ax, "A", "At n=3 this design cannot detect anything smaller than "
                        f"{mde(sd, 3):.2f}",
          f"Exact noncentral-t power for the paired t-test at the observed paired sd = "
          f"{sd:.3f}\n({ref_v} vs {ref_c}).  A normal approximation would overstate n=3 "
          f"power by about 2x.")

    # --- B: where the variance actually is --------------------------------------------
    ns = np.array([25, 50, 100, 200, 400, 800])
    tot = np.sqrt(sd_train ** 2 + p_ctrl * (1 - p_ctrl) / ns)
    bx.plot(ns, tot, "-o", color=C["crimson"], lw=2.4, ms=6, zorder=5,
            label="total paired sd")
    bx.axhline(sd_train, color=C["slate"], lw=2.0, ls=(0, (4, 2)), zorder=4)
    bx.text(1500, sd_train - 0.0015, f"training-variance floor = {sd_train:.3f}",
            fontsize=9.5, color=C["slate"], va="top", ha="right", fontweight="bold")
    for n in (50, 200):
        i = int(np.where(ns == n)[0][0])
        bx.scatter([n], [tot[i]], s=150, facecolor="none", edgecolor=C["crimson"], lw=2.0,
                   zorder=6)
        bx.text(n * 1.12, tot[i] + 0.002, f"{n} episodes: {tot[i]:.3f}", fontsize=9.5,
                color=C["crimson"], fontweight="bold", va="bottom", ha="left")
    bx.set_xscale("log")
    bx.set_xlim(20, 1600)
    bx.set_ylim(0.112, 0.157)
    bx.set_xticks([25, 50, 100, 200, 400, 800])
    bx.set_xticklabels(["25", "50", "100", "200", "400", "800"])
    bx.set_xlabel("eval episodes per run", fontsize=10.5, color=INK)
    bx.set_ylabel("paired sd of the contrast", fontsize=10.5, color=INK)
    S.ax_style(bx, grid="both")
    i50, i200 = int(np.where(ns == 50)[0][0]), int(np.where(ns == 200)[0][0])
    callout(bx, 0.30, 0.985,
            f"Quadrupling the eval budget 50 -> 200\n"
            f"moves the total sd {tot[i50]:.3f} -> {tot[i200]:.3f}.",
            "crimson", fontsize=9.5)
    panel(fig, bx, "B", "and more eval episodes cannot fix it",
          f"Decomposition at the control's own success rate {p_ctrl:.3f}:\n"
          f"sd_eval = sqrt(p(1-p)/n_evals) = {sd_eval:.3f} at n_evals = {n_evals},\n"
          f"leaving sd_train = {sd_train:.3f}.")

    head(fig, "Training variance dominates eval variance, so the campaign needed more "
              "SEEDS, not longer evals",
         f"Every contrast in the 2026-09-01 round ran at n=3.  Reaching the Step-3 deficit "
         f"of {abs(paired(arms, ref_v, ref_c)['mean']):.3f} at 80% power needs "
         f"n = {min(n for n in range(3, 40) if mde(sd, n) <= abs(paired(arms, ref_v, ref_c)['mean']))}.")
    note(fig, PREFIX_BANNER, "crimson", x=0.012, dy=0.010)
    return save(fig, out, "power.png")


def fig_effdim_vs_D(out=OUT_01):
    """Section 5. effective_dim tracks the TASK, not the code width -- and the caveat that
    goes with it, which is that the metric is still rising when training stops."""
    rows = summaries()
    widths = collections.defaultdict(list)
    for r in rows:
        if r["arm"] in ("LpWM-ltv", "LpWM-ltv-d768", "LpWM-ltv-d1536", "LpWM-ltv-d2048"):
            widths[r["width"]].append(r["run"])

    # Read at a MATCHED epoch, as the last value logged at or before epoch_frac = 1.0 --
    # the 2026-09-01 entry's own convention, which this reproduces exactly at D=768 (23.9)
    # and D=1536 (24.1). The metric is a per-batch estimate with a wide batch-to-batch
    # spread, so the interquartile range over the last 5% of that epoch is drawn with it.
    pts = {}
    for D in sorted(widths):
        vals = []
        for run in sorted(widths[D]):
            a = wandb_curve(run, ["progress/epoch_frac", "sparsity/effective_dim"])
            if a.shape[0] < 8:
                continue
            a = a[np.isfinite(a[:, 1])]
            at = a[a[:, 0] <= 1.0]
            win = a[(a[:, 0] >= 0.95) & (a[:, 0] <= 1.0), 1]
            if at.shape[0] and win.size >= 4:
                vals.append((float(at[-1, 1]), float(np.percentile(win, 25)),
                             float(np.percentile(win, 75))))
        if vals:
            pts[D] = vals

    FW, FH = 13.6, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.052, 0.155, 0.410, 0.545])
    bx = fig.add_axes([0.600, 0.155, 0.380, 0.545])

    Ds = sorted(pts)
    med = np.array([np.median([v for v, _l, _h in pts[D]]) for D in Ds])
    PR_PUSHT = 4.31          # the 2026-09-01 measurement; see the note at the foot
    ref = med[0] * np.array(Ds, float) / Ds[0]
    ax.plot(Ds, ref, ls=(0, (3, 2)), color=OTHER, lw=1.8, zorder=3)
    ax.text(0.365, 0.955, f"if it scaled with D  ({med[0] / Ds[0]:.1%} of D)",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, ha="left", va="top")
    for D in Ds:
        v = np.array([x[0] for x in pts[D]], float)
        iq = np.array([[x[1], x[2]] for x in pts[D]], float)
        c = C["teal"] if D > 384 else C["slate"]
        xs = np.full(v.size, float(D)) * np.exp(np.linspace(-0.055, 0.055, v.size))
        for xx, (q1, q3) in zip(xs, iq):
            ax.plot([xx, xx], [q1, q3], color=c, lw=1.0, alpha=0.55, zorder=3)
        ax.scatter(xs, v, s=36, facecolor="none", edgecolor=c, lw=1.2, zorder=4)
    ax.plot(Ds, med, "-o", color=C["slate"], lw=2.6, ms=10, zorder=5)
    for D, m in zip(Ds, med):
        top = max([x[2] for x in pts[D]] + [m])
        ax.text(D, top + 1.4, f"{m:.1f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=C["slate"], zorder=6)
        ax.text(D, 0.9, f"n={len(pts[D])}", ha="center", va="bottom", fontsize=9,
                color=MUTED)
    ax.text(Ds[-1] * 1.28, med[-1], "measured at\nmatched epoch 1.0", fontsize=9.5,
            color=C["slate"], fontweight="bold", va="center", ha="left", linespacing=1.35)
    ax.axhline(PR_PUSHT, color=C["green"], lw=2.2, zorder=4)
    ax.text(320, PR_PUSHT + 1.0, f"PushT's own participation ratio = {PR_PUSHT}",
            va="bottom", ha="left", fontsize=9.5, color=C["green"], fontweight="bold")
    big = 1536 if 1536 in Ds else Ds[-1]
    gain = med[Ds.index(big)] / med[0] - 1
    callout(ax, 0.235, 0.425,
            f"384 -> {big} is {big // 384}x the width for {gain:+.0%} effective dimension.\n"
            f"It saturates near {med.max():.0f}, about {med.max() / PR_PUSHT:.1f}x the "
            f"task's own\ndimensionality -- D=384 is already {384 / PR_PUSHT:.0f}x the task.",
            "green", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(300, 3500)
    ax.set_ylim(0, 41)
    ax.set_xticks(Ds)
    ax.set_xticklabels([str(D) for D in Ds])
    ax.minorticks_off()
    ax.set_xlabel("code width D  (log)", fontsize=10.5, color=INK)
    ax.set_ylabel("effective_dim (participation ratio)", fontsize=10.5, color=INK)
    S.ax_style(ax, grid="both")
    panel(fig, ax, "A", "effective_dim tracks the TASK, not the width",
          "Ring = one seed at the matched epoch, whisker = its interquartile spread over the "
          "last 5% of that\nepoch; the line joins the per-width medians.  Same LpWM-ltv "
          "configuration throughout, changed only in D.")

    # --- B: the caveat that goes with panel A -----------------------------------------
    grid = np.linspace(0.0, 2.0, 320)
    curves = []
    for run in sorted(r["run"] for r in rows if r["arm"] == "LpWM-ltv"):
        a = wandb_curve(run, ["progress/epoch_frac", "sparsity/effective_dim"])
        if a.shape[0] < 8:
            continue
        a = a[np.isfinite(a[:, 1])]
        curves.append(np.interp(grid, a[:, 0], a[:, 1], left=np.nan, right=np.nan))
    if curves:
        Y = np.vstack(curves)
        for y in Y:
            bx.plot(grid, y, color=OTHER, lw=0.8, alpha=0.55, zorder=2)
        cnt = np.isfinite(Y).sum(0)
        mu = np.where(cnt >= max(2, len(curves) // 2), np.nanmean(Y, 0), np.nan)
        bx.plot(grid, mu, color=C["slate"], lw=2.8, zorder=5)
        ok = np.where(np.isfinite(mu))[0]
        a1 = int(np.argmin(np.abs(grid - 1.0)))
        bx.scatter([1.0, grid[ok[-1]]], [mu[a1], mu[ok[-1]]], s=115, zorder=6,
                   facecolor="white", edgecolor=C["crimson"], lw=2.2)
        bx.annotate(f"the mean is still rising when training stops:\n"
                    f"{mu[a1]:.1f} at epoch 1.0 -> {mu[ok[-1]]:.1f} at epoch "
                    f"{grid[ok[-1]]:.1f}, and the seeds\nspread over "
                    f"{np.nanmin(Y[:, ok[-1]]):.1f}-{np.nanmax(Y[:, ok[-1]]):.1f} at the end",
                    xy=(1.02, mu[a1]), xytext=(0.075, 37.0), fontsize=9.5,
                    color=C["crimson"], fontweight="bold", va="top", ha="left",
                    linespacing=1.4,
                    bbox=dict(boxstyle="round,pad=0.42", fc=FILL["crimson"],
                              ec=C["crimson"], lw=0.9),
                    arrowprops=dict(arrowstyle="-", color=C["crimson"], lw=1.0))
        bx.axhline(PR_PUSHT, color=C["green"], lw=2.2, zorder=3)
        bx.text(1.97, PR_PUSHT + 0.9, "PushT participation ratio", va="bottom", ha="right",
                fontsize=9.5, color=C["green"], fontweight="bold")
        bx.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=3)
        panel(fig, bx, "B", "so a mid-training comparison of it is not safe",
              f"Every LpWM-ltv seed at D=384 (n={len(curves)}), thin; their mean, bold.  "
              "Panel A is read at a MATCHED\nepoch for exactly this reason.")
    bx.set_xlim(0, 2.0)
    bx.set_ylim(0, 41)
    bx.set_xlabel("epoch (fractional)", fontsize=10.5, color=INK)
    bx.set_ylabel("effective_dim (participation ratio)", fontsize=10.5, color=INK)
    S.ax_style(bx)

    head(fig, "Benchmark effective_dim against the task's dimensionality, not against D",
         "Measured against D=384 the baselines look like a 95% collapse; measured against "
         "PushT's own participation ratio they are 4-6x over-complete, which is healthy.  "
         "The units matter more than the number.")
    note(fig, "The reference line 4.31 is the 2026-09-01 measurement of PushT's own "
              "participation ratio and is carried over unchanged.  Recomputed here on the "
              "dataset it is 4.36 (val split) and 4.45 (train split) for\nstate+velocity, "
              "so the line is right to within 3%; the exact subsample behind 4.31 is not "
              "recorded.  wall_single reproduces exactly at 2.00.",
              "slate", x=0.012, dy=0.010)
    return save(fig, out, "effdim-vs-D.png")


# ======================================================================================
# 2026-09-02  --  the architecture round, on the REPAIRED eval instrument
# ======================================================================================
#: (label, variant, control). Every variant is paired against ITS OWN matched control,
#: never against LpWM-ltv generically. The mapping is verified from the runs' configs:
#: PiWM-refframe differs from LpWM-ltv-mupfix in `use_pose` alone, and PiWM-union4-vfloor
#: and PiWM-sdr-d2048-k41 differ from LpWM-ltv-vfloor / -d2048 in one flag each.
CONTRASTS_0902 = [
    ("consensus M=5",        "PiWM-vote5-median",    "LpWM-ltv"),
    ("consensus M=3",        "PiWM-vote3-median",    "LpWM-ltv"),
    ("gate: magnitude",      "PiWM-gate-mag-softmax", "LpWM-ltv"),
    ("gate: both",           "PiWM-gate-both",       "LpWM-ltv"),
    ("variance floor alone", "LpWM-ltv-vfloor",      "LpWM-ltv"),
    ("gate: support",        "PiWM-gate-sup-softmax", "LpWM-ltv"),
    ("refframe (pose)",      "PiWM-refframe",        "LpWM-ltv-mupfix"),
    ("muP input-LR fix",     "LpWM-ltv-mupfix",      "LpWM-ltv"),
    ("union4 + floor",       "PiWM-union4-vfloor",   "LpWM-ltv-vfloor"),
    ("Gaussian target",      "LeWM-ltv-p2",          "LpWM-ltv"),
    ("k-WTA @ D=384",        "PiWM-kwta8-J1",        "LpWM-ltv"),
    ("k-WTA @ D=2048",       "PiWM-sdr-d2048-k41",   "LpWM-ltv-d2048"),
]
#: the 2026-09-02 entry's own table, kept so the figure can show where more seeds have
#: since landed instead of silently replacing a published number. {label: (n, delta)}
PUBLISHED_0902 = {
    "consensus M=5": (10, +0.228), "consensus M=3": (12, +0.165),
    "gate: magnitude": (13, +0.015), "gate: both": (13, -0.012),
    "variance floor alone": (3, -0.020), "gate: support": (13, -0.069),
    "refframe (pose)": (13, -0.063), "muP input-LR fix": (10, -0.136),
    "union4 + floor": (6, -0.387), "Gaussian target": (3, -0.407),
    "k-WTA @ D=384": (3, -0.427), "k-WTA @ D=2048": (6, -0.580),
}


def fig_contrasts(out=OUT_02):
    """Section 1. Every matched contrast, recomputed. Rows whose n has grown since the entry
    was written carry their published estimate beside the current one -- the record's value
    is that a number can be seen to move, not that it is quietly overwritten."""
    arms = campaign("fixed")
    got = []
    for lab, v, c in CONTRASTS_0902:
        e = paired(arms, v, c)
        if e:
            got.append((lab, v, c, e))
    got.sort(key=lambda t: -t[3]["mean"])

    FW = 15.0
    ROW, TOP, BOT = 0.46, 1.62, 1.05
    FH = TOP + BOT + ROW * len(got)
    fig = plt.figure(figsize=(FW, FH))
    L, PW = 1.95, 6.10                       # margins/widths in inches
    ax = fig.add_axes([L / FW, BOT / FH, PW / FW, (ROW * len(got)) / FH])
    tr = ax.get_yaxis_transform()

    for i, (lab, v, c, e) in enumerate(reversed(got)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F6F8F7", lw=0, zorder=0)
        pos = e["p"] < 0.05 and e["mean"] > 0
        neg = e["p"] < 0.05 and e["mean"] < 0
        key = "green" if pos else ("crimson" if neg else "slate")
        cc = C[key]
        ax.plot([e["lo"], e["hi"]], [i, i], color=cc, lw=2.6, solid_capstyle="round",
                zorder=4)
        ax.scatter([e["mean"]], [i], s=120, color=cc, zorder=6, edgecolor="white", lw=0.9)
        pub = PUBLISHED_0902.get(lab)
        if pub and (pub[0] != e["n"] or abs(pub[1] - e["mean"]) > 0.0015):
            ax.plot([pub[1], e["mean"]], [i, i], color=MUTED, lw=0.9, ls=(0, (2, 2)),
                    zorder=5)
            ax.scatter([pub[1]], [i], s=70, marker="D", facecolor="white",
                       edgecolor=MUTED, lw=1.2, zorder=6)
        # "null" is a claim about precision, not just about p: it needs a CI tight enough
        # to exclude an effect worth acting on. A wide CI centred on zero is unresolved.
        verdict = ("POSITIVE" if pos else "negative" if neg else
                   "unresolved" if e["p"] < 0.10 else
                   "null" if e["hi"] - e["lo"] < 0.30 else "unresolved")
        for xf, txt, col_, wt in (
                (1.035, f"{e['mean']:+.3f}", cc, "bold"),
                (1.135, f"[{e['lo']:+.3f}, {e['hi']:+.3f}]", MUTED, "normal"),
                (1.330, f"n={e['n']}", MUTED, "normal"),
                (1.420, f"p={e['p']:.4f}" if e["p"] >= 1e-4 else "p<0.0001", MUTED,
                 "normal"),
                (1.545, verdict, cc if verdict in ("POSITIVE", "negative") else MUTED,
                 "bold" if verdict in ("POSITIVE", "negative") else "normal")):
            ax.text(xf, i, txt, transform=tr, va="center", ha="left", fontsize=9.5,
                    color=col_, fontweight=wt, clip_on=False)
        ax.text(-0.018, i - 0.30, f"vs {c}", transform=tr, va="center", ha="right",
                fontsize=8.5, color=MUTED, clip_on=False)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=3)
    ax.set_yticks(range(len(got)))
    ax.set_yticklabels([lab for lab, _v, _c, _e in reversed(got)], fontsize=10.5)
    for t, (lab, v, _c, e) in zip(ax.get_yticklabels(), reversed(got)):
        t.set_color(col(v, MUTED))
        t.set_fontweight("bold")
    ax.set_ylim(-0.5, len(got) - 0.5)
    ax.set_xlim(-1.05, 0.55)
    ax.set_xlabel("paired difference in CEM success vs the matched control   "
                  "(95% t CI; negative = worse)", fontsize=10.5, color=INK, labelpad=8)
    S.ax_style(ax, grid="x")

    # column headers, in the axes' own y-fraction so they cannot drift onto row 12
    yh = 1 + 0.30 / (ROW * len(got))
    for xf, hd in ((1.035, "effect"), (1.135, "95% CI"), (1.330, "n"), (1.420, "p"),
                   (1.545, "verdict")):
        ax.text(xf, yh, hd, transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
    ax.plot([1.02, 1.72], [yh - 0.012] * 2, transform=ax.transAxes, color=EDGE, lw=1.0,
            clip_on=False)
    ax.scatter([1.028], [yh + 0.055], transform=ax.transAxes, s=70, marker="D",
               facecolor="white", edgecolor=MUTED, lw=1.2, clip_on=False)
    ax.text(1.048, yh + 0.055, "= the estimate as published on 2026-09-02, where more "
                               "seeds have since landed", transform=ax.transAxes,
            ha="left", va="center", fontsize=9.5, color=MUTED, clip_on=False)

    head(fig, "One intervention out of nine is positive; the rest are well-measured "
              "negatives and nulls",
         "Each variant against ITS OWN matched control, never against LpWM-ltv generically.  "
         "CIs are t-based: a normal approximation badly understates the width at n=3.  "
         "Recomputed from the current campaign,\nwhich reproduces nine of the twelve "
         "published rows exactly.", dy=0.32, dy_sub=0.64)
    return save(fig, out, "contrasts.png")


VOTE5 = ["PiWM-vote5-median", "PiWM-vote5-borda", "PiWM-vote5-cvar1", "PiWM-vote5-cvar2",
         "PiWM-vote5-max"]
VOTE_RULE = {"PiWM-vote5-median": "median", "PiWM-vote5-borda": "Borda",
             "PiWM-vote5-cvar1": "CVaR-1", "PiWM-vote5-cvar2": "CVaR-2",
             "PiWM-vote5-max": "max"}


def fig_consensus(out=OUT_02):
    """Section 3. The round's one positive, and the two things about it that make it a
    result rather than a lucky arm: it scales with M, and it does not depend on the voting
    rule. What CANNOT be rebuilt is section 3's 'vs members' mean / best' rows -- see the
    panel subtitle and the module docstring."""
    arms = campaign("fixed")
    steps = [(1, "LpWM-ltv", "1\n(single model)"), (3, "PiWM-vote3-median", "3"),
             (5, "PiWM-vote5-median", "5")]
    steps = [(m, a, lab) for m, a, lab in steps if a in arms]

    FW, FH = 13.8, 6.6
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.052, 0.165, 0.335, 0.545])
    bx = fig.add_axes([0.510, 0.165, 0.290, 0.545])

    # --- A: it scales -----------------------------------------------------------------
    means = []
    for i, (m, a, lab) in enumerate(steps):
        v = np.array([arms[a][s] for s in sorted(arms[a])], float)
        c = C["slate"] if m == 1 else C["green"]
        jit = np.linspace(-0.16, 0.16, v.size)
        ax.scatter(i + jit, v, s=52, facecolor=c, edgecolor="white", lw=0.8, alpha=0.85,
                   zorder=4)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=c, lw=3.4, zorder=5,
                solid_capstyle="butt")
        means.append(v.mean())
        nz = int((v == 0).sum())
        ax.text(i, v.max() + 0.025, f"{v.mean():.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=c, zorder=6)
        ax.text(i, -0.055, f"n={v.size}\ncatastrophes {nz}/{v.size}", ha="center",
                va="top", fontsize=9.5, color=C["crimson"] if nz else C["green"],
                fontweight="bold", linespacing=1.35)
    ax.plot(range(len(steps)), means, color=C["green"], lw=1.8, alpha=0.6, zorder=3)
    for i, (m, a, lab) in enumerate(steps[1:], start=1):
        e = paired(arms, a, "LpWM-ltv")
        if e:
            ax.text(i, 0.895, f"{e['mean']:+.3f}\np = {e['p']:.4f}", ha="center",
                    va="bottom", fontsize=10, fontweight="bold", color=C["green"],
                    linespacing=1.35)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([lab for _m, _a, lab in steps], fontsize=10.5)
    ax.tick_params(axis="x", length=0, pad=34)
    ax.set_xlim(-0.55, len(steps) - 0.45)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("number of columns M in the plan-time vote", fontsize=10.5, color=INK,
                  labelpad=10)
    ax.set_ylabel("CEM planning success", fontsize=10.5, color=INK)
    S.ax_style(ax)
    panel(fig, ax, "A", "Consensus scales with the number of columns",
          "Every seed shown; the bar is the arm's mean.  Noise-averaging saturates, a vote "
          "keeps improving\nas independent models are added -- and catastrophic failure "
          "disappears entirely.")

    # --- B: and it is not an artefact of one voting rule -------------------------------
    rules = [(a, paired(arms, a, "LpWM-ltv")) for a in VOTE5 if a in arms]
    rules = [(a, e) for a, e in rules if e]
    rules.sort(key=lambda t: t[1]["mean"])
    trb = bx.get_yaxis_transform()
    for i, (a, e) in enumerate(rules):
        bx.plot([e["lo"], e["hi"]], [i, i], color=C["green"], lw=2.6,
                solid_capstyle="round", zorder=4)
        bx.scatter([e["mean"]], [i], s=115, color=C["green"], edgecolor="white", lw=0.9,
                   zorder=5)
        for xf, txt in ((1.04, f"{e['mean']:+.3f}"), (1.16, f"n={e['n']}"),
                        (1.27, f"p={e['p']:.4f}" if e["p"] >= 1e-4 else "p<0.0001")):
            bx.text(xf, i, txt, transform=trb, va="center", ha="left", fontsize=9.5,
                    color=C["green"] if xf < 1.1 else MUTED,
                    fontweight="bold" if xf < 1.1 else "normal", clip_on=False)
    bx.axvline(0, color=MUTED, lw=1.2, zorder=3)
    bx.set_yticks(range(len(rules)))
    bx.set_yticklabels([VOTE_RULE.get(a, a) for a, _e in rules], fontsize=10.5)
    for t in bx.get_yticklabels():
        t.set_color(C["green"])
        t.set_fontweight("bold")
    bx.set_ylim(-0.6, len(rules) - 0.4)
    bx.set_xlim(-0.06, 0.42)
    bx.set_xlabel("paired difference vs the single-model control  (95% t CI)",
                  fontsize=10.5, color=INK)
    S.ax_style(bx, grid="x")
    panel(fig, bx, "B", "and it does not depend on the voting rule",
          "Five different M=5 combination rules, all against LpWM-ltv on shared seeds.  "
          "Section 3's other two\nchecks -- against the members' MEAN and their BEST -- "
          "cannot be rebuilt: plan_outputs does not\nrecord which checkpoints a vote's "
          "columns were.")

    head(fig, "The campaign's one positive is a plan-time vote, and it attacks the variance "
              "the arms could not",
         "A variance decomposition over four D=384 arms x 11 shared seeds gives arm 3.8% / "
         "seed 51.5% / arm x seed 44.7%.  Seed instability, not representation design, owns "
         "the variance -- and consensus is\nthe only thing tried that attacks it.  Its "
         "honest limit is that it spends M times the rollout compute at plan time.",
         dy=0.30, dy_sub=0.62)
    return save(fig, out, "consensus-scaling.png")


def fig_kwta_not_sparsity(out=OUT_02):
    """Section 4. Sparsity LEVEL does not predict planning; the SPARSIFIER does.

    The population is the WHOLE current campaign, not the nine arms the section was written
    on, and that matters in one direction: "every k-WTA arm plans at zero" still holds, but
    "every dense arm plans" does not -- block-causal is dense and plans 0.000, which is the
    retraction the same entry records in section 5. Both are stated on the figure.
    """
    rows = summaries()
    arms_f, arms_b = campaign("fixed"), campaign("buggy")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    def agg(arm, camp):
        v = by.get(arm, [])
        rho = [x["rho"] for x in v if x["rho"] is not None]
        cem = camp.get(arm)
        if not rho or not cem:
            return None
        return float(np.median(rho)), float(np.median(list(cem.values()))), len(cem)

    #: label -> (dx, dy) in points, chosen so no two labels and no label and mark collide
    LABELLED = {
        "LpWM-ltv-d2048": ("dense, D=2048", (12, 4), "left"),
        "PiWM-vote5-median": ("consensus M=5", (12, -4), "left"),
        "LpWM-ltv": ("LpWM-ltv", (12, -10), "left"),
        "LpWM-ltv-vfloor": ("dense + var floor", (12, 6), "left"),
        "PiWM-gate-sup-softmax": ("gate: support", (12, -3), "left"),
        "PiWM-blockcausal": ("block-causal", (0, 15), "center"),
        "PiWM-kwta8-J1": ("k-WTA k=8, D=384", (0, 15), "center"),
        "PiWM-sdr-d2048-k41": ("k-WTA w=41, D=2048", (0, 15), "center"),
        "PiWM-union4-vfloor": ("union head J=4 + floor", (13, -11), "left"),
    }

    FW, FH = 15.4, 6.8
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.046, 0.170, 0.372, 0.520])
    bx = fig.add_axes([0.492, 0.170, 0.322, 0.520])
    cx = fig.add_axes([0.858, 0.170, 0.127, 0.520])

    kw_pts, dense_pts = [], []
    for arm in sorted(arms_f):
        g = agg(arm, arms_f)
        if not g:
            continue
        rho, cem, n = g
        v0 = by[arm][0]
        kw, nh = v0["kwta"], (v0["n_heads"] or 1)
        if kw:
            mk, cc, sz = "X", C["crimson"], 200
            kw_pts.append((arm, rho, cem, n))
        elif nh > 1:
            mk, cc, sz = "s", C["crimson"], 110
        else:
            mk, cc, sz = "o", col(arm), 95
            if 0.25 <= rho <= 0.60:
                dense_pts.append((arm, rho, cem))
        ax.scatter([rho], [cem], s=sz, marker=mk, color=cc, zorder=5 if kw or nh > 1 else 4,
                   edgecolor="white", lw=0.9, alpha=1.0 if arm in LABELLED else 0.7)
        if arm in LABELLED:
            txt, off, ha = LABELLED[arm]
            ax.annotate(txt, (rho, cem), textcoords="offset points", xytext=off,
                        fontsize=9.5, color=cc, fontweight="bold", ha=ha, va="center",
                        linespacing=1.3)
    ax.set_xscale("log")
    ax.set_xlim(2.5e-4, 3.0)
    ax.set_ylim(-0.06, 0.90)
    ax.set_xlabel(r"code density  $\rho$  (fraction of units active, log)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("median CEM planning success", fontsize=10.5, color=INK)
    S.ax_style(ax, grid="both")
    ax.scatter([], [], s=95, marker="o", color=C["slate"], label="no k-WTA")
    ax.scatter([], [], s=200, marker="X", color=C["crimson"], label="k-WTA imposed")
    ax.scatter([], [], s=110, marker="s", color=C["crimson"], label="union head")
    ax.legend(loc="upper left", fontsize=9.5, handletextpad=0.5, labelspacing=0.35,
              bbox_to_anchor=(0.0, 1.0))
    n_kw_seeds = sum(n for _a, _r, _c, n in kw_pts)
    dcem = [c for _a, _r, c in dense_pts]
    callout(ax, 0.020, 0.735,
            f"k-WTA: {len(kw_pts)} arms, {n_kw_seeds} seeds, median CEM 0.000 for every "
            f"one of them,\n"
            r"across three decades of $\rho$ and both widths." "\n"
            r"Density on its own predicts nothing: at $\rho$ = 0.25-0.60 without k-WTA "
            f"the\ncampaign's {len(dense_pts)} arms run from {min(dcem):.2f} to "
            f"{max(dcem):.2f}.",
            "crimson", fontsize=9.5)
    panel(fig, ax, "A", "Sparsity LEVEL does not predict planning -- the SPARSIFIER does",
          f"One point per arm on the repaired eval instrument ({len(arms_f)} arms), median "
          "over its seeds.  Not gradient\nstarvation: kwta has a straight-through backward.")

    # --- B: realised density is min(k, #positive)/D, not k/D --------------------------
    kw_arms = [a for a in sorted(by) if by[a][0]["kwta"]
               and any(x["rho"] is not None for x in by[a])]
    ticks = []
    for i, arm in enumerate(kw_arms):
        v = by[arm]
        target = v[0]["kwta"] / v[0]["width"]
        real = np.array([x["rho"] for x in v if x["rho"] is not None], float)
        holds = abs(float(np.median(real)) - target) < 0.02 * target
        bx.plot([i - 0.32, i + 0.32], [target] * 2, color=C["green"], lw=2.8, zorder=4,
                label="configured k/D" if i == 0 else None)
        bx.scatter(np.full(real.size, i), np.clip(real, 9e-5, None), s=72,
                   color=C["slate"] if holds else C["crimson"], edgecolor="white", lw=0.8,
                   zorder=5, label="realised (one dot per seed)" if i == 0 else None)
        med = float(np.median(real))
        ticks.append(f"{arm.replace('PiWM-', '')}\n{med * v[0]['width']:.0f}/"
                     f"{v[0]['kwta']} units\n{'HOLDS' if holds else 'BREACHED'}")
    bx.set_yscale("log")
    bx.set_ylim(6e-5, 1.4)
    bx.set_xlim(-0.6, len(kw_arms) - 0.4)
    bx.set_xticks(range(len(kw_arms)))
    bx.set_xticklabels(ticks, fontsize=8, linespacing=1.45)
    for t, a in zip(bx.get_xticklabels(), kw_arms):
        v = by[a]
        med = float(np.median([x["rho"] for x in v if x["rho"] is not None]))
        holds = abs(med - v[0]["kwta"] / v[0]["width"]) < 0.02 * v[0]["kwta"] / v[0]["width"]
        t.set_color(C["slate"] if holds else C["crimson"])
        t.set_fontweight("bold")
    bx.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    bx.tick_params(axis="x", length=0, pad=6)
    bx.set_ylabel(r"end-of-training density  $\rho$  (log)", fontsize=10.5, color=INK)
    S.ax_style(bx)
    bx.legend(loc="upper left", fontsize=8.5, handletextpad=0.5, labelspacing=0.3, ncol=2)
    panel(fig, bx, "B", "Realised density is min(k, #positive)/D, not k/D",
          "Link.forward rectifies THEN applies k-WTA, and kwta marks exactly k positions\n"
          "whose values may be zero.  Three of the five arms therefore never ran at the\n"
          "operating point they were configured for.")

    # --- C: the same-density pair ------------------------------------------------------
    pair = [("LpWM-base", "LpWM-base\nno k-WTA"),
            ("PiWM-sparse-matched", "sparse-matched\nk-WTA k=113")]
    got = [(a, lab, agg(a, arms_b)) for a, lab in pair]
    got = [(a, lab, g) for a, lab, g in got if g]
    for i, (a, lab, (rho, cem, n)) in enumerate(got):
        c = C["slate"] if i == 0 else C["crimson"]
        cx.bar(i, max(cem, 0.0), width=0.55, color=c, zorder=3)
        cx.text(i, cem + 0.014, f"{cem:.3f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=c)
    cx.set_xticks(range(len(got)))
    cx.set_xticklabels([f"{lab}\n" + r"$\rho$ = " + f"{g[0]:.3f}"
                        for _a, lab, g in got], fontsize=7.5, linespacing=1.55)
    for t, (a, _l, _g) in zip(cx.get_xticklabels(), got):
        t.set_color(col(a))
        t.set_fontweight("bold")
    cx.tick_params(axis="x", length=0, pad=6)
    cx.set_xlim(-0.6, len(got) - 0.4)
    cx.set_ylim(0, 0.46)
    cx.set_ylabel("CEM success", fontsize=10.5, color=INK)
    S.ax_style(cx)
    panel(fig, cx, "C", "Same density,\nopposite outcome",
          "PRE-FIX instrument: this\npair was never re-evaluated.")

    head(fig, "k-WTA is the damaging operator, not sparsity -- and the SDR-regime "
              "hypothesis is refuted with it",
         "w=41 at n=2048 is squarely inside Numenta's sparse-distributed-representation band "
         "and still scores 0.000 across 6 seeds.  Note the converse does NOT hold on the "
         "full campaign: block-causal is dense\n(rho = 0.38) and plans 0.000, which is the "
         "temporal-collapse retraction the same entry records in section 5.")
    return save(fig, out, "kwta-not-sparsity.png")


def fig_all_arms(out=OUT_02):
    """Every arm in the campaign, every seed, on one axis.

    This file is not referenced by any diary section -- it is the campaign's index. It was
    drawn for the nine arms of the 2026-09-02 round; regenerated it covers every arm that
    now has three or more evals on the repaired instrument. The Oracle-* ladder is excluded:
    those runs replace the model's own rollout with privileged state, so their success rate
    is not on the same scale as a learned model's.

    Layout is explicit inch-space. At this many rows tight_layout has nothing useful to do,
    and the numeric columns get their own axes so a digit can never land on a mark.
    """
    arms = campaign("fixed")
    keep = {a: v for a, v in arms.items()
            if not a.startswith("Oracle-") and not a.startswith("PiWMvote") and len(v) >= 3}
    order = sorted(keep, key=lambda a: (np.median(list(keep[a].values())),
                                        np.mean(list(keep[a].values()))))
    n = len(order)
    base = float(np.median(list(keep["LpWM-ltv"].values()))) if "LpWM-ltv" in keep else None

    FW = 13.2
    ROW, L, STRIP, GAP, COLW, TOP, BOT = 0.30, 3.05, 6.30, 0.32, 0.78, 1.72, 0.95
    FH = TOP + BOT + ROW * n
    fig = plt.figure(figsize=(FW, FH))
    y0, h = BOT / FH, (ROW * n) / FH
    ax = fig.add_axes([L / FW, y0, STRIP / FW, h])
    cxs = [fig.add_axes([(L + STRIP + GAP + i * (COLW + 0.14)) / FW, y0, COLW / FW, h])
           for i in range(3)]
    for cx in cxs:
        cx.set_xlim(0, 1)
        cx.set_ylim(-0.6, n - 0.4)
        cx.set_xticks([])
        cx.set_yticks([])
        for sp in cx.spines.values():
            sp.set_visible(False)

    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F6F8F7", lw=0, zorder=0)
            for cx in cxs:
                cx.axhspan(i - 0.5, i + 0.5, color="#F6F8F7", lw=0, zorder=0)
    ax.axvspan(-0.02, 0.02, color=FILL["crimson"], lw=0, zorder=1)
    if base is not None:
        ax.axvline(base, color=C["slate"], lw=1.2, ls=(0, (2, 2)), zorder=2)

    for i, arm in enumerate(order):
        v = np.array([keep[arm][s] for s in sorted(keep[arm])], float)
        c = col(arm)
        named = hue(arm) is not None
        ax.scatter(v, np.full(v.size, i), s=34 if named else 26, color=c, lw=0,
                   alpha=0.9 if named else 0.6, zorder=4)
        med = float(np.median(v))
        ax.plot([med, med], [i - 0.34, i + 0.34], color=c, lw=2.6, zorder=5,
                solid_capstyle="butt")
        for cx, txt in zip(cxs, (f"{len(v)}", f"{med:.3f}", f"{v.mean():.3f}")):
            cx.text(0.92, i, txt, va="center", ha="right", fontsize=9.5,
                    color=c if named else MUTED,
                    fontweight="bold" if named else "normal")
    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=9.5)
    for t, a in zip(ax.get_yticklabels(), order):
        t.set_color(col(a) if hue(a) else MUTED)
        if hue(a):
            t.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.035, 0.90)
    ax.set_xticks(np.arange(0, 0.9, 0.1))
    S.ax_style(ax, grid="x")
    ax.set_xlabel("CEM planning success on PushT   (1.0 = solves every episode)",
                  fontsize=10.5, color=INK, labelpad=8)

    yh = 1 + 0.26 / (ROW * n)
    for cx, hd in zip(cxs, ("seeds", "median", "mean")):
        cx.text(0.92, yh, hd, transform=cx.transAxes, ha="right", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
        cx.plot([0.0, 1.0], [yh - 0.010] * 2, transform=cx.transAxes, color=EDGE, lw=1.0,
                clip_on=False)
    ax.text(0.0, yh, "floored at zero", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=C["crimson"], fontweight="bold")
    if base is not None:
        ax.text(base / 0.935 + 0.02, yh, f"LpWM-ltv baseline, median {base:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
                color=C["slate"], fontweight="bold")

    # the hue key, in its own band above the strip so it cannot sit on a row
    keys = [("slate", "baselines / controls"), ("teal", "the baseline, wider"),
            ("green", "plan-time consensus"), ("amber", "the gate family"),
            ("purple", "other representation work"),
            ("crimson", "k-WTA, union head, collapsed")]
    x = 0.0
    for k, lab in keys:
        fig.text(L / FW + x, 1 - 1.15 / FH, "●", fontsize=11, color=C[k],
                 ha="left", va="center")
        fig.text(L / FW + x + 0.012, 1 - 1.15 / FH, lab, fontsize=9.5, color=MUTED,
                 ha="left", va="center")
        x += 0.016 + 0.0058 * len(lab)

    head(fig, "Every arm in the campaign, every seed",
         f"{n} arms, {sum(len(v) for v in keep.values())} evaluated runs, all on the "
         "repaired eval instrument, sorted by median.  The whole campaign produced one arm "
         "family above the baseline and a long tail at zero.",
         x=L / FW - 0.225, dy=0.34, dy_sub=0.66)
    return save(fig, out, "all-arms.png", dpi=150)


#: Section 8's table, transcribed from the 2026-09-02 entry. These nine numbers are NOT
#: reproducible from anything in the repo: `causal/d_action` was added to the training loop
#: after every one of these arms had trained, so none of them has a logged value, and the
#: quantity was hand-measured from one checkpoint each. They are kept here verbatim because
#: the retraction in 2026-09-03 section 12b is only legible next to what it retracts.
SEC8_PUBLISHED = [
    ("LpWM-ltv-d2048",        6.93e-04, 1.59e-03, 0.70, 0.70),
    ("PiWM-gate-both",        4.15e-04, 1.15e-03, 0.69, 0.44),
    ("PiWM-refframe",         4.12e-04, 3.55e-04, 0.57, 0.34),
    ("PiWM-gate-sup-softmax", 3.77e-04, 6.70e-04, 0.73, 0.46),
    ("LpWM-ltv-mupfix",       2.63e-04, 2.02e-03, 0.77, 0.20),
    ("PiWM-kwta8-J1",         2.38e-04, 4.63e-04, 0.03, 0.00),
    ("LeWM-ltv-p2",           2.15e-04, 1.79e-03, 0.48, 0.06),
    ("LpWM-ltv",              1.29e-04, 4.74e-03, 0.68, 0.24),
    ("PiWM-union4-vfloor",    8.67e-05, 7.36e-05, 0.32, 0.00),
]
#: what train.py's docstring quoted for the baseline, and what 12b showed it should be
QUOTED_BASELINE_RATIO = 1.29e-04 / 0.68


def fig_action_sensitivity(out=OUT_02):
    """Section 8, and the correction that overtook it.

    Panel A is the section as published, under its retraction banner. Panel B is the same
    quantity re-measured from every checkpoint in the archive by analysis/d_action_probe.py.
    Panel C is the arithmetic of the error. Nothing here is rescaled quietly: the retracted
    numbers and the replacement are on the same figure.
    """
    probe = {x["run"]: x for x in
             json.load(open(os.path.join(REPO, "assets/d_action_probe.json")))}
    rows = summaries()
    arms = campaign("fixed")
    recs = []
    for r in rows:
        pr = probe.get(r["run"])
        cem = arms.get(r["arm"], {}).get(r["seed"])
        if pr is None or cem is None or r["rho"] is None or r["rel_mse"] is None:
            continue
        recs.append(dict(arm=r["arm"], x=pr["d_action_over_scale"], cem=cem,
                         rho=r["rho"], rel_mse=r["rel_mse"]))

    FW, FH = 14.8, 7.0
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.048, 0.190, 0.290, 0.500])
    bx = fig.add_axes([0.437, 0.190, 0.290, 0.500])
    cx = fig.add_axes([0.815, 0.190, 0.170, 0.500])

    # --- A: as published ---------------------------------------------------------------
    xs = np.array([d for _a, d, _ds, _z, _c in SEC8_PUBLISHED])
    ys = np.array([c for _a, _d, _ds, _z, c in SEC8_PUBLISHED])
    r8, p8 = stats.spearmanr(xs, ys)
    #: (dx, dy) in points and the anchor, chosen so no two labels touch
    OFF8 = {"PiWM-gate-sup-softmax": ((-11, 4), "right"),
            "PiWM-gate-both": ((11, -3), "left"),
            "PiWM-union4-vfloor": ((7, -14), "left"),
            "PiWM-kwta8-J1": ((11, 13), "left"),
            "LeWM-ltv-p2": ((11, 12), "left"),
            "LpWM-ltv": ((-11, 0), "right")}
    for arm, d, _ds, _z, c in SEC8_PUBLISHED:
        ax.scatter([d], [c], s=110, color=col(arm), edgecolor="white", lw=0.9, zorder=4)
        off, ha = OFF8.get(arm, ((11, -1), "left"))
        ax.annotate(arm, (d, c), textcoords="offset points", xytext=off, fontsize=9,
                    color=col(arm), ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_xlim(5.5e-5, 2.6e-3)
    ax.set_ylim(-0.11, 1.02)
    ax.set_xlabel(r"$d_{action}$ as reported on 2026-09-02", fontsize=10.5, color=INK)
    ax.set_ylabel("CEM planning success", fontsize=10.5, color=INK)
    S.ax_style(ax, grid="both")
    callout(ax, 0.035, 0.965,
            f"Spearman = {r8:+.2f}  (p = {p8:.4f}, n = {len(xs)}),\n"
            "reported as the strongest predictor in the project.",
            "crimson", fontsize=9.5)
    panel(fig, ax, "A", "What section 8 measured",
          "Nine arms, ONE seed each, hand-measured from a checkpoint.  None of them has a\n"
          "logged causal/d_action: the diagnostic was added after they all trained.")

    # --- B: the same quantity, re-measured over the archive ----------------------------
    X = np.array([r["x"] for r in recs])
    Y = np.array([r["cem"] for r in recs])
    RHO = np.array([r["rho"] for r in recs])
    RM = np.array([r["rel_mse"] for r in recs])
    for r in sorted(recs, key=lambda r: hue(r["arm"]) is not None):
        named = hue(r["arm"]) is not None
        bx.scatter([r["x"]], [r["cem"]], s=30 if named else 22, color=col(r["arm"]),
                   lw=0, alpha=0.85 if named else 0.45, zorder=4 if named else 3)
    ok = RM < 0.05
    o = np.argsort(X[ok])
    xs_, ys_ = X[ok][o], Y[ok][o]
    bins = np.array_split(np.arange(xs_.size), 5)
    bxm = [float(np.median(xs_[b])) for b in bins]
    bym = [float(np.mean(ys_[b])) for b in bins]
    bx.plot(bxm, bym, "-o", color=C["crimson"], lw=2.6, ms=9, zorder=6,
            markeredgecolor="white", markeredgewidth=1.0)
    top = int(np.argmax(bym))
    bx.annotate(f"optimum near {bxm[top]:.2f}", (bxm[top], bym[top]),
                textcoords="offset points", xytext=(26, 12), fontsize=9.5,
                color=C["crimson"], fontweight="bold", ha="left")
    from analysis.causal_figs import partial_spearman
    rp, _rx, _ry, pp = partial_spearman(X, Y, [RHO, RM], n_perm=5000)
    bx.axvline(float(np.median([r["x"] for r in recs if r["arm"] == "LpWM-ltv"])),
               color=C["slate"], lw=1.2, ls=(0, (2, 2)), zorder=2)
    bx.text(float(np.median([r["x"] for r in recs if r["arm"] == "LpWM-ltv"])) + 0.02,
            -0.045, "LpWM-ltv", fontsize=9, color=C["slate"], fontweight="bold",
            ha="left", va="bottom")
    bx.set_xlim(-0.05, 1.55)
    bx.set_ylim(-0.06, 1.02)
    bx.set_xlabel(r"$d_{action}\,/\,\|z\|$, re-probed from every checkpoint",
                  fontsize=10.5, color=INK)
    bx.set_ylabel("CEM planning success", fontsize=10.5, color=INK)
    S.ax_style(bx, grid="both")
    callout(bx, 0.035, 0.965,
            f"partial Spearman = {rp:+.3f} controlling " r"$\rho$ and rel_mse" "\n"
            f"(permutation p " + (f"< {1/5000:.0e}" if pp <= 0 else f"= {pp:.4f}") +
            f", n = {len(recs)} runs).\n"
            "The line is the mean CEM in five equal bins over the\n"
            "healthy predictors -- the relation is an inverted U, not a slope.",
            "crimson", fontsize=9.5)
    panel(fig, bx, "B", "What the re-probe measures",
          "analysis/d_action_probe.py, one fixed batch and one fixed permutation for every\n"
          "arm, so the values are comparable in a way the logged ones are not.")

    # --- C: the arithmetic of the error ------------------------------------------------
    # all 16 LpWM-ltv seeds, not only the 13 that also have a CEM number -- 12b's claim
    # is about the arm, and restricting to evaluated runs would change the statistic
    meas = float(np.median([v["d_action_over_scale"] for v in probe.values()
                            if v["arm"] == "LpWM-ltv"]))
    n_meas = sum(1 for v in probe.values() if v["arm"] == "LpWM-ltv")
    vals = [("quoted in train.py\nand in every section\nthat followed",
             QUOTED_BASELINE_RATIO, C["crimson"]),
            (f"measured over all\n{n_meas} LpWM-ltv seeds", meas, C["slate"])]
    for i, (lab, v, c) in enumerate(vals):
        cx.barh(i, v, height=0.42, color=c, zorder=3)
        cx.text(v * 1.35, i, f"{v:.5f}" if v < 0.01 else f"{v:.3f}", va="center",
                ha="left", fontsize=11, fontweight="bold", color=c)
    cx.set_xscale("log")
    cx.set_xlim(8e-5, 12.0)
    cx.set_ylim(-2.0, 1.6)
    cx.set_yticks([0, 1])
    cx.set_yticklabels([lab for lab, _v, _c in vals], fontsize=9, linespacing=1.4)
    for t, (_l, _v, c) in zip(cx.get_yticklabels(), vals):
        t.set_color(c)
        t.set_fontweight("bold")
    cx.tick_params(axis="y", length=0, pad=6)
    cx.set_xlabel(r"$d_{action}\,/\,\|z\|$  (log)", fontsize=10.5, color=INK)
    S.ax_style(cx, grid="x")
    callout(cx, 0.02, 0.335,
            f"wrong by a factor of {meas / QUOTED_BASELINE_RATIO:.0f}x.\n"
            "The baseline was never action-inert:\nat 0.55 it sits in the campaign's top "
            "quartile,\nABOVE every arm built to raise it.",
            "crimson", fontsize=9.5)
    panel(fig, cx, "C", "The number that\ndrove two rounds",
          "2026-09-03 section 12b.")

    head(fig, "Section 8's premise was a wrong number, and rounds 3 and 4 were spent chasing "
              "it",
         "The action is NOT causally inert.  Rounds 3 and 4 spent nine arms trying to raise a "
         "quantity that was already near its optimum, against a target value off by three "
         "orders of magnitude; every one of\nthose arms moved it DOWN and every one of them "
         "lost.  That is one measurement error propagated, not nine independent failures.")
    note(fig, "PANEL A IS SUPERSEDED -- see 2026-09-03 section 12b.  It is drawn from the "
              "2026-09-02 entry's own table because those nine values exist nowhere else: "
              "none of the nine arms has a logged\ncausal/d_action.  It is kept, with its "
              "correlation, because a retraction is only legible next to what it retracts.  "
              "Panels B and C are measured from the current archive.",
         "crimson", x=0.012, dy=0.010)
    return save(fig, out, "action-sensitivity.png")


def fig_target_p_inert(out=OUT_02):
    """Section 9. RDMReg's own test cannot tell a p=1 target from a p=2 one at D=384, the
    reason is Diaconis-Freedman, and the archive at the time could not separate the target
    from the link -- which is why that section's claim was withdrawn.

    The three panels are computed here: the first two are a fresh simulation at a fixed seed
    (the entry's own triple is one draw from these distributions), and the third is a
    cross-tabulation of every run's config.
    """
    import torch
    from models.infojepa_modules import swd, sample_generalized_gaussian

    D, NPROJ, R = 384, 8192, 400
    torch.manual_seed(0)
    conds = [("NULL", "Gaussian vs Gaussian", 2.0, 2.0, "slate"),
             ("ALT", "Gaussian vs Laplace", 2.0, 1.0, "crimson"),
             ("ALT'", "Laplace vs Gaussian", 1.0, 2.0, "crimson")]
    draws = {}
    for name, _lab, pz, pt, _k in conds:
        draws[name] = np.array([
            swd(sample_generalized_gaussian((1, D), pz),
                sample_generalized_gaussian((1, D), pt), NPROJ).item() for _ in range(R)])

    g = torch.Generator().manual_seed(1)
    N, NP = 5000, 1000
    lap = sample_generalized_gaussian((N, D), 1.0)
    w = torch.randn(NP, D, generator=g)
    w = w / w.norm(dim=1, keepdim=True)
    proj = (lap @ w.T).numpy()
    ek = stats.kurtosis(proj, axis=0, fisher=True)
    sh = np.array([stats.shapiro(proj[:4500, i]).pvalue for i in range(200)])
    theory = 9.0 / (D + 2)

    tab = collections.Counter()
    for d in sorted(glob.glob(os.path.join(REPO, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        if name.startswith("CANARY-") or not RUN_RE.match(name):
            continue
        cfg = run_cfg(name)
        if cfg.get("link.kind") and cfg.get("target_p") is not None:
            tab[(cfg["link.kind"], int(cfg["target_p"]))] += 1

    FW, FH = 14.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.050, 0.180, 0.290, 0.540])
    bx = fig.add_axes([0.435, 0.180, 0.250, 0.540])
    cx = fig.add_axes([0.780, 0.180, 0.200, 0.540])

    # --- A: RDMReg's own statistic, under the null and under the alternative ------------
    rng = np.random.default_rng(0)
    for i, (name, lab, _pz, _pt, key) in enumerate(conds):
        v = draws[name]
        c = C[key]
        ax.scatter(np.full(v.size, i) + rng.uniform(-0.19, 0.19, v.size), v, s=9,
                   color=c, alpha=0.30, lw=0, zorder=3)
        se = v.std(ddof=1) / math.sqrt(v.size)
        ax.plot([i - 0.30, i + 0.30], [v.mean()] * 2, color=c, lw=3.2, zorder=5,
                solid_capstyle="butt")
        ax.plot([i, i], [v.mean() - se, v.mean() + se], color=c, lw=3.2, zorder=5)
        ax.text(i, 2.62, f"{v.mean():.4f}\n" r"$\pm$ " f"{se:.4f}", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold", color=c, linespacing=1.35)
    ax.axhline(draws["NULL"].mean(), color=C["slate"], lw=1.1, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{n}\n{lab}" for n, lab, _a, _b, _k in conds], fontsize=9.5,
                       linespacing=1.5)
    for t, (_n, _l, _a, _b, k) in zip(ax.get_xticklabels(), conds):
        t.set_color(C[k])
        t.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylim(1.45, 3.05)
    ax.set_ylabel("sliced-Wasserstein   (RDMReg's own test)", fontsize=10.5, color=INK)
    S.ax_style(ax)
    u = stats.mannwhitneyu(draws["ALT"], draws["NULL"])
    auc = u.statistic / (R * R)
    callout(ax, 0.035, 0.195,
            f"The 'sparse' alternative is indistinguishable from the null:\n"
            f"|ALT - NULL| = {abs(draws['ALT'].mean() - draws['NULL'].mean()):.4f} against a "
            f"draw-to-draw sd of {draws['NULL'].std(ddof=1):.3f},\n"
            f"and the two distributions separate at AUC = {auc:.3f}.",
            "crimson", fontsize=9.5)
    panel(fig, ax, "A", "RDMReg cannot tell a p=1 target from a p=2 one at D=384",
          f"{R} independent draws per condition, {NPROJ} projections each, fixed seed.\n"
          "The entry's triple (1.9987 / 1.9947 / 2.0025) is one draw from these.")

    # --- B: why -- Diaconis-Freedman ---------------------------------------------------
    bx.hist(proj[:, 0], bins=70, density=True, color=FILL["slate"], edgecolor=C["slate"],
            lw=0.5, zorder=3)
    xs = np.linspace(proj[:, 0].min(), proj[:, 0].max(), 400)
    bx.plot(xs, stats.norm.pdf(xs, proj[:, 0].mean(), proj[:, 0].std()), color=C["green"],
            lw=2.6, zorder=5)
    bx.text(0.97, 0.965, "Gaussian fit", transform=bx.transAxes, ha="right", va="top",
            fontsize=9.5, color=C["green"], fontweight="bold")
    bx.set_xlim(-4.2, 4.2)
    bx.set_xlabel("one random 1-D projection of a D=384 Laplace sample", fontsize=10.5,
                  color=INK)
    bx.set_ylabel("density", fontsize=10.5, color=INK)
    S.ax_style(bx)
    callout(bx, 0.035, 0.795,
            f"Over {NP} random projections the excess kurtosis is\n"
            f"{ek.mean():+.3f}  (95% of them in [{np.percentile(ek, 2.5):+.2f}, "
            f"{np.percentile(ek, 97.5):+.2f}]), against a\n"
            f"prediction of 9/(D+2) = {theory:.3f}; a single Laplace\n"
            f"coordinate has excess kurtosis 3, about {3 / theory:.0f}x more.\n"
            f"Shapiro-Wilk fails to reject on {(sh > 0.05).mean():.0%} of projections.",
            "green", fontsize=9.5)
    panel(fig, bx, "B", "because a 1-D projection of it IS Gaussian",
          "Diaconis-Freedman: almost every low-dimensional projection of a\n"
          "high-dimensional sample is approximately normal, and RDMReg only\n"
          "ever looks at random 1-D projections (infojepa_modules.py:675-682).")

    # --- C: and the archive could not separate the target from the link ----------------
    links, ps = ["reprelu", "identity"], [1, 2]
    Mx = np.array([[tab.get((lk, pv), 0) for pv in ps] for lk in links], float)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("g", S.SEQ_GREEN)
    cx.imshow(np.sqrt(Mx), cmap=ramp, vmin=0, vmax=max(np.sqrt(Mx).max(), 1) * 1.15,
              aspect="auto", extent=[-0.5, 1.5, 1.5, -0.5])
    for i in range(2):
        for j in range(2):
            v = int(Mx[i, j])
            dark = np.sqrt(v) / max(np.sqrt(Mx).max(), 1) > 0.5
            cx.text(j, i - 0.08, f"{v}", ha="center", va="center", fontsize=21,
                    fontweight="bold", color="white" if dark else INK)
            cx.text(j, i + 0.20, "runs", ha="center", va="center", fontsize=10,
                    color="white" if dark else INK)
            if (links[i], ps[j]) in (("reprelu", 2), ("identity", 1)):
                cx.text(j, i + 0.36, "0 when the entry\nwas written", ha="center",
                        va="center", fontsize=8.5, color="white" if dark else C["crimson"],
                        fontweight="bold", linespacing=1.35)
    cx.set_xticks([0, 1])
    cx.set_xticklabels(["p = 1", "p = 2"], fontsize=11)
    cx.set_yticks([0, 1])
    cx.set_yticklabels(links, fontsize=11)
    cx.tick_params(length=0, labelcolor=INK)
    cx.set_xlabel("RDMReg target", fontsize=10.5, color=INK)
    cx.set_ylabel("link", fontsize=10.5, color=INK)
    for sp in cx.spines.values():
        sp.set_visible(False)
    cx.grid(False)
    panel(fig, cx, "C", "The archive could not\nseparate target from link",
          "Which is why section 5's claim --\nthat the TARGET, not the regulariser,\n"
          "was implicated -- was WITHDRAWN.")

    head(fig, "target_p is an inert knob: the 'sparse prior' does nothing, and rho ~ 0.5 "
              "comes from the ReLU link",
         "The whole sparse-vs-dense axis of this project, and of the LpWM line it inherits, "
         "is really rectified-vs-not-rectified.")
    return save(fig, out, "target-p-inert.png")


# ======================================================================================
FIGURES = {
    "cem-effects": fig_cem_effects,
    "factorial-2x2": fig_factorial,
    "root-cause": fig_root_cause,
    "union-collapse": fig_union_collapse,
    "power": fig_power,
    "effdim-vs-D": fig_effdim_vs_D,
    "contrasts": fig_contrasts,
    "consensus-scaling": fig_consensus,
    "kwta-not-sparsity": fig_kwta_not_sparsity,
    "all-arms": fig_all_arms,
    "action-sensitivity": fig_action_sensitivity,
    "target-p-inert": fig_target_p_inert,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                    help="regenerate only these file stems (default: all twelve)")
    a = ap.parse_args(argv)
    for name in (a.only or list(FIGURES)):
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
