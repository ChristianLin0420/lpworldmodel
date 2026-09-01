"""Paired-comparison figure suite. matplotlib only, no new dependencies.

Inputs, all optional -- a figure is skipped (never faked) when its input is absent,
and `main()` prints exactly which file or metric would unblock it. `--strict` turns
any skip into a non-zero exit, so CI can insist that a panel stays renderable.

  --step1  analysis_step1.json written by analysis/predictive_jaccard.py, with the
           sibling .npz holding per-episode S_world / S_model / onset / z / j_star.
  --campaign  campaign.json describing the run tallies:
           {"arms":   {"<arm>": {"<seed>": success, ...}, ...},
            "gates":  [{"name","observed","lo","hi","threshold","direction"}],
            "k_sweep": {"<k/D>": {"success": s, "rdmreg": r}},
            "scale":  {"<gate_input>": {"c": [...], "rel_change": [...]}},
            "ladder": {"<predictor>": {"sparse"|"dense": {"<D>": [per-seed, ...]}}},
            "gate_values":  {"<arm>": [gate value samples, ...]},
            "gate_heatmap": {"<arm>": [[t x r gate values]]}}
  --runs   glob of run dirs, each holding the artefacts written by
           analysis/export_wandb.py: wandb_history.csv (required), and optionally
           wandb_meta.json, resume_steps.json, wandb_hists.json.

Nothing in the repo wrote wandb_history.csv before analysis/export_wandb.py existed,
which is why the run-level panels could never render on real runs. Export first:

    set -a && . ./.env && set +a
    python analysis/export_wandb.py --project PiWM-pushT

Usage:
    python analysis/figures.py --step1 runs/outputs/<run>/analysis_step1.json \
        --campaign campaign.json --runs 'runs/outputs/*' --out figures/
    python analysis/figures.py --selftest --out /tmp/figs   # exercise every panel
    python analysis/figures.py --runs 'runs/outputs/*' --out figs --strict
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# The design system lives in exactly one module. This file decides WHICH panels are
# worth drawing and loads their inputs; `panels` decides what they look like. Before
# this import existed, figures.py carried a second, incompatible palette -- which is
# how the same arm ended up green in one panel and red in the next on one page.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import panels as P  # noqa: E402

#: The two Step-1 statistics. Deliberately OUTSIDE the arm palette (violet / orange,
#: validated all-pairs, dE 29.5 protan) because they are quantities, not arms, and no
#: panel shows both an arm identity and these at once.
STAT = {"S_world": "#4a3aa7", "S_model": "#eb6834"}
#: Status tokens. Reserved: never used for "series 2", and always shipped with a
#: glyph or a word so the state is never carried by colour alone.
STATUS = {"good": "#1a7f37", "warn": "#9a6700", "crit": "#b3261e"}
#: A single-series stroke, for panels whose subject is one quantity rather than a
#: comparison. One series needs no legend and no categorical slot.
SOLO = "#1c5cab"
GHOST, INK = P.GHOST, "0.15"
WINDOW = 5
SLURM_WINDOW_H = 3.917  # the 3h55m sbatch limit every run is chopped at


# --- skip bookkeeping -----------------------------------------------------------
# The user's complaint about the previous version was that panels vanished silently.
# A panel now records WHY it bailed and WHAT would unblock it; main() prints both.

_LAST_SKIP = []


def skip(reason, unblock):
    """Record why a panel bailed and return None so callers stay one-liners."""
    _LAST_SKIP.append((reason, unblock))
    return None


def take_skip():
    """Pop the most recent skip reason, or a generic one if none was recorded."""
    return _LAST_SKIP.pop() if _LAST_SKIP else ("returned no figure", "unknown")


# --- io -------------------------------------------------------------------------

def load_step1(path):
    """Returns (summary dict, list of per-episode dicts)."""
    path = Path(path)
    summary = json.loads(path.read_text())
    z = np.load(path.with_suffix(".npz"))
    n = int(z["n_episodes"]) if "n_episodes" in z else summary["n_episodes"]
    eps = []
    for i in range(n):
        e = {k: z[f"{k}_{i}"] for k in ("S_world", "S_model", "onset", "block_disp")
             if f"{k}_{i}" in z}
        for k in ("z", "j_star", "states", "agent_disp"):
            if f"{k}_{i}" in z:
                e[k] = z[f"{k}_{i}"]
        eps.append(e)
    return summary, eps


def flat_aliases(k):
    """Every flat name a sectioned wandb column can be addressed by, outermost first.

    train.py's `wandb_key()` prepends a section chosen from a table that the
    training-side code keeps growing, so hardcoding the known sections here would
    break the whole suite the next time one is added. Instead peel one section at a
    time using only the two structural rules `wandb_key()` actually follows:
    'train/loss' -> 'train_loss' (the phase is the section), and
    '<anything>/train_l0_frac' -> 'train_l0_frac' (the phase is inside the name).
    Any other prefix -- known or not, nested or not -- is simply dropped.
    """
    out = [k]
    cur = k
    while "/" in cur:
        head, rest = cur.split("/", 1)
        if rest.startswith(("train_", "val_")):
            cur = rest
        elif head in ("train", "val"):
            cur = f"{head}_{rest}"
        else:
            cur = rest
        if cur in out:
            break
        out.append(cur)
    return out


def unsection(k):
    """Undo train.py's wandb_key sectioning: 'train/loss' -> 'train_loss',
    'heads/train_head_usage_p0' -> 'train_head_usage_p0'. Unknown sections are
    dropped rather than rejected, and flat exports pass through untouched."""
    return flat_aliases(k)[-1]


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_csv(path):
    """wandb history export -> {column: float array with nan for blanks}.

    Columns are stored under their exported name AND every un-sectioned alias, so
    panels can address flat keys whatever scheme wrote the file. Two quirks of real
    exports are handled here rather than in every panel: a header-only CSV (a run
    that had not logged yet) yields empty arrays instead of an IndexError, and when
    two sectioned columns collapse to the same flat alias the one with more finite
    values wins, since the loser is usually a stale duplicate.
    """
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        rows = list(rd)
    raw = {k: np.array([_as_float(r.get(k)) for r in rows], dtype=float)
           for k in cols}
    if rows and "_step" in raw and not np.all(np.diff(raw["_step"]) >= 0):
        order = np.argsort(raw["_step"], kind="mergesort")
        raw = {k: v[order] for k, v in raw.items()}

    out = dict(raw)
    for k in cols:
        for alias in flat_aliases(k)[1:]:
            if alias in raw:  # a real column always beats an alias of another one
                continue
            cur = out.get(alias)
            if cur is None or np.isfinite(raw[k]).sum() > np.isfinite(cur).sum():
                out[alias] = raw[k]
    return out


# x-axis preference order. `epoch` is deliberately not first: at 30,965 batches per
# epoch every live run sits inside epoch 1 for hours, so an epoch axis collapses the
# whole run onto a single vertical line. epoch_frac is the same quantity, continuous.
X_PREFS = (("epoch_frac", "epoch (fractional)"), ("epoch", "epoch"),
           ("_step", "train batch"), ("step", "train batch"))


def x_axis(hist):
    """Pick the best available x-axis for a history. Returns (x, label, key).

    Falls through to the row index so a panel is never blocked purely on axes, and
    rejects a candidate that does not vary, which is what makes a partially-trained
    run readable instead of a single vertical stripe.
    """
    n = max((len(v) for v in hist.values()), default=0)
    for key, label in X_PREFS:
        v = hist.get(key)
        if v is None or len(v) != n:
            continue
        f = v[np.isfinite(v)]
        if f.size >= 2 and f.max() > f.min():
            return v, label, key
    return np.arange(n, dtype=float), "logged sample", "index"


_MARK_UNIT = {"epoch_frac": "epoch_frac", "epoch": "epoch", "_step": "step",
              "step": "step"}


def marks_for(run, xkey):
    """Resume markers converted to the plot's x unit, or [] if unavailable.

    A marker drawn in the wrong unit is worse than no marker at all -- a step index
    plotted on an epoch axis lands off-scale and reads as a broken run -- so this
    refuses to guess and returns nothing when the exporter could not pin the marker
    down in the unit being plotted.
    """
    r = run.get("resumes") or {}
    if isinstance(r, list):  # legacy exports: a bare list of epoch numbers
        r = {"epoch": [float(x) for x in r]}
    unit = _MARK_UNIT.get(xkey)
    return list(r.get(unit, [])) if unit else []


_ARM_STRIP = re.compile(r"_pd\d+|_(bf16|fp16|no)(?=_|$)|_s\d+$")


def run_arm(name):
    """Run dir name -> campaign arm, matching train.py's wandb group derivation."""
    return _ARM_STRIP.sub("", name)


def run_seed(name):
    m = re.search(r"_s(\d+)$", name or "")
    return int(m.group(1)) if m else None


def load_runs(pattern):
    """Run dirs holding an export -> list of run dicts.

    Each dict carries the history, the arm/seed the run belongs to (from
    wandb_meta.json when the exporter wrote one, else parsed from the dir name),
    resume markers, and any exported histogram columns.
    """
    runs = []
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        hist = d / "wandb_history.csv"
        if not hist.exists():
            continue
        try:
            h = load_csv(hist)
        except (OSError, csv.Error) as e:
            print(f"  ! {d.name}: unreadable history ({e})")
            continue
        if not h or max((len(v) for v in h.values()), default=0) == 0:
            print(f"  ! {d.name}: history export has no rows yet")
            continue

        def _read(name, default):
            p = d / name
            try:
                return json.loads(p.read_text()) if p.exists() else default
            except (OSError, ValueError):
                return default

        meta = _read("wandb_meta.json", {})
        runs.append({
            "name": d.name,
            "dir": d,
            "hist": h,
            "resumes": _read("resume_steps.json", {}),
            "hists": _read("wandb_hists.json", {}),
            "meta": meta,
            "arm": meta.get("group") or run_arm(d.name),
            "seed": meta.get("seed") if meta.get("seed") is not None
            else run_seed(d.name),
        })
    return runs


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# --- stats helpers --------------------------------------------------------------

def mean_ci(rows, axis=0, n_boot=1000, seed=0):
    """Mean with a bootstrap 95% CI over the resampled axis (episodes)."""
    rows = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.nanmean(rows, axis=axis)
    idx = rng.integers(0, rows.shape[axis], size=(n_boot, rows.shape[axis]))
    boots = np.nanmean(rows[idx], axis=axis + 1)
    return m, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def roc_curve(scores, labels):
    """(fpr, tpr) at every threshold. Ties handled by sorting on score only."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tpr = np.concatenate([[0.0], np.cumsum(y) / max(y.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~y) / max((~y).sum(), 1)])
    return fpr, tpr


def peri_event(series, onset, window=WINDOW):
    """Rows of `series` aligned so column `window` is the onset frame."""
    rows = []
    for t in np.flatnonzero(onset > 0.5):
        lo, hi = t - window, t + window + 1
        if lo < 0 or hi > len(series):
            continue
        rows.append(series[lo:hi])
    return rows


# Two-sided t critical values at alpha=0.05, keyed by degrees of freedom. Hardcoded
# because the whole suite is numpy+matplotlib only, and at n=3 seeds the difference
# between t and the normal quantile is a factor of 2.2 -- far too big to fudge.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        30: 2.042, 60: 2.000, 120: 1.980}


def t_crit(df):
    """Two-sided 95% t critical value, interpolated in 1/df beyond the table."""
    if df < 1:
        return np.inf
    if df in _T95:
        return _T95[df]
    ks = sorted(_T95)
    if df > ks[-1]:
        return 1.960
    lo = max(k for k in ks if k < df)
    hi = min(k for k in ks if k > df)
    w = (1.0 / df - 1.0 / lo) / (1.0 / hi - 1.0 / lo)
    return _T95[lo] + w * (_T95[hi] - _T95[lo])


def paired_effect(arms, ctrl, arm):
    """Paired per-seed effect of `arm` against `ctrl`.

    Paired because the gates are defined on matched seeds: the same seed carries the
    same data order and init, so differencing removes the seed variance that
    dominates an unpaired comparison at n=3. Seeds present in only one arm are
    dropped rather than mean-imputed.
    """
    seeds = sorted(set(arms.get(ctrl, {})) & set(arms.get(arm, {})),
                   key=lambda s: (len(str(s)), str(s)))
    d = np.array([float(arms[arm][s]) - float(arms[ctrl][s]) for s in seeds])
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    tc = t_crit(n - 1)
    return {"seeds": seeds, "n": n, "delta": d, "mean": float(d.mean()) if n else np.nan,
            "sd": sd, "lo": d.mean() - tc * se if n > 1 else np.nan,
            "hi": d.mean() + tc * se if n > 1 else np.nan,
            "dz": float(d.mean() / sd) if n > 1 and sd else np.nan}


def paired_power(delta, sd, n, n_sim=3000, seed=0):
    """Simulated power of a paired t-test at alpha=0.05 for a true effect `delta`.

    Simulated rather than looked up so the small-n behaviour is the real thing
    (the sd is re-estimated from each simulated experiment, which is exactly what
    makes n=3 so weak) without pulling in scipy.
    """
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        return np.nan
    rng = np.random.default_rng(seed)
    d = rng.normal(delta, sd, size=(n_sim, n))
    s = d.std(axis=1, ddof=1)
    t = np.where(s > 0, d.mean(axis=1) / (s / np.sqrt(n)), 0.0)
    return float((np.abs(t) > t_crit(n - 1)).mean())


def mde(sd, n, power=0.8, hi_mult=6.0, n_grid=60, seed=0):
    """Smallest true effect a paired n-seed test detects with the given power."""
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        return np.nan
    grid = np.linspace(0, hi_mult * sd, n_grid)
    for g in grid:
        if paired_power(g, sd, n, seed=seed) >= power:
            return float(g)
    return np.nan


def sign_flip_min_p(n):
    """Smallest two-sided p a paired sign-flip permutation test can return at n.

    With n matched seeds there are only 2**n sign assignments, so the minimum
    attainable two-sided p is 2/2**n. At n=3 that is 0.25: a nonparametric paired
    test cannot reject at 0.05 no matter how large the effect. Worth showing.
    """
    return min(1.0, 2.0 / (2 ** n)) if n >= 1 else 1.0


def variance_decomposition(arms):
    """Between-arm and within-arm (seed) variance of a per-(arm, seed) scalar.

    Returns (between, within, icc). icc near 0 means seed noise swamps the arm
    effect, which is the thing the campaign has to beat.
    """
    groups = [np.array([float(v) for v in d.values()]) for d in arms.values()
              if len(d) > 0]
    groups = [g for g in groups if g.size]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    grand = np.concatenate(groups).mean()
    between = float(np.mean([(g.mean() - grand) ** 2 for g in groups]))
    within = float(np.mean([g.var(ddof=1) if g.size > 1 else 0.0 for g in groups]))
    tot = between + within
    return between, within, (between / tot if tot > 0 else np.nan)


def unit_activity(z):
    """Per-unit activation frequency and mean active magnitude over frames."""
    z = np.asarray(z, float)
    act = z > 0
    n_act = act.sum(0)
    freq = n_act / max(len(z), 1)
    mag = np.where(n_act > 0, np.where(act, z, 0.0).sum(0) / np.maximum(n_act, 1), 0.0)
    return freq, mag


def participation_ratio(p):
    """(sum p)^2 / sum p^2: the effective number of dimensions carrying activity.

    Equals D for a perfectly flat code and 1 when a single unit carries everything,
    so it converts 'is the code actually using its width' into one number.
    """
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    s2 = float((p ** 2).sum())
    return float(p.sum() ** 2 / s2) if s2 > 0 else 0.0


def dead_fraction(z):
    """Fraction of units that never activate anywhere in the sample."""
    z = np.asarray(z, float)
    return float((z <= 0).all(axis=0).mean()) if z.size else np.nan


def xcorr_lag(sig, event, max_lag=WINDOW):
    """Correlation of `sig` shifted by each lag against `event`.

    r[lag] = corr(sig[t+lag], event[t]), so a peak LEFT of zero means the statistic
    rises before the event -- the same sign convention as the peri-event panel.
    """
    sig, event = np.asarray(sig, float), np.asarray(event, float)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        a = sig[max_lag + L: len(sig) - max_lag + L]
        b = event[max_lag: len(event) - max_lag]
        if a.size < 3 or a.std() == 0 or b.std() == 0:
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return lags, out


def hist_matrix(entries, n_bins=48):
    """Exported wandb histograms -> (steps, edges, density matrix).

    Each logged histogram can have its own bin edges, so source bins are treated as
    point masses at their centres and re-binned onto one shared grid; rows are
    normalised so a row is a distribution regardless of how many samples it saw.
    """
    entries = [e for e in entries if e.get("counts") and e.get("edges")]
    if not entries:
        return None
    lo = min(float(e["edges"][0]) for e in entries)
    hi = max(float(e["edges"][-1]) for e in entries)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    steps, rows = [], []
    for e in sorted(entries, key=lambda x: x.get("step") or 0):
        ed = np.asarray(e["edges"], float)
        mid = 0.5 * (ed[:-1] + ed[1:])
        w = np.asarray(e["counts"], float)[: len(mid)]
        h, _ = np.histogram(mid, bins=edges, weights=w)
        rows.append(h / max(h.sum(), 1e-9))
        steps.append(float(e.get("step") or len(steps)))
    return np.array(steps), edges, np.array(rows)


# --- run-collection helpers -----------------------------------------------------

def by_arm(runs):
    """{arm: [run, ...]} with seeds ordered, controls first."""
    out = {}
    for r in runs:
        out.setdefault(r["arm"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: (r["seed"] is None, r["seed"], r["name"]))
    ctrl = [a for a in out if "upstream" in a or "ctrl" in a]
    return {**{a: out[a] for a in sorted(ctrl)},
            **{a: out[a] for a in sorted(set(out) - set(ctrl))}}


def arm_palette(arms):
    """Stable arm -> colour, delegated to the design system.

    This used to be `tab10[i % 10]` keyed on ENUMERATION POSITION, so dropping one
    run from a panel silently repainted every arm after it, and the same arm carried
    different colours in different figures of the same contact sheet. Identity must
    follow the entity, never its row number.
    """
    return P.arm_palette(arms)


def align_series(runs, key, n_grid=140):
    """Interpolate `key` from every run onto one shared x grid.

    Runs sit at different points in the campaign, so plotting raw arrays against
    their own x makes a seed band meaningless. The grid spans the union of the runs'
    ranges and each run is nan outside its own, so a band shows the leading seed
    continuing alone rather than a fabricated plateau.
    """
    series = []
    for r in runs:
        h = r["hist"]
        if key not in h:
            continue
        x, label, xkey = x_axis(h)
        y = h[key]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue
        series.append((r, x[m], y[m], label, xkey))
    if not series:
        return None
    lo = min(s[1].min() for s in series)
    hi = max(s[1].max() for s in series)
    grid = np.linspace(lo, hi, n_grid) if hi > lo else np.array([lo])
    mat = np.full((len(series), len(grid)), np.nan)
    for i, (_, x, y, _, _) in enumerate(series):
        o = np.argsort(x, kind="mergesort")
        v = np.interp(grid, x[o], y[o])
        v[(grid < x.min()) | (grid > x.max())] = np.nan
        mat[i] = v
    return {"grid": grid, "mat": mat, "runs": [s[0] for s in series],
            "xlabel": series[0][3], "xkey": series[0][4]}


def plot_arm_bands(ax, runs, key, palette=None, seed_lines=True, legend=True):
    """One seed-mean line per arm with a min-max band and thin per-seed traces.

    27 runs drawn as 27 lines is noise (and silently truncates at whatever colour
    cycle length is in play). Collapsing to one band per arm is what makes a
    9-arm campaign legible, and the thin traces keep the seed spread visible.

    Above three arms the legend is replaced by DIRECT LABELS at each curve's right
    end. A legend box at nine arms is a second lookup the reader has to perform per
    curve, and -- more to the point -- nine arms are past the count colour alone can
    carry, so the label is doing the identifying work and the hue only groups.
    Dash and marker come from the design system, which is what keeps the two Step 3
    arms that share a lightness step apart.
    """
    groups = by_arm(runs)
    palette = palette or arm_palette(list(groups))
    drawn, xlabel, ends = 0, "logged sample", []
    for arm, rs in groups.items():
        al = align_series(rs, key)
        if al is None:
            continue
        xlabel = al["xlabel"]
        st = P.arm_style(arm)
        c = palette.get(arm, st["color"])
        if seed_lines:
            for row in al["mat"]:
                ax.plot(al["grid"], row, color=c, lw=0.7, alpha=0.30, zorder=1)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(al["mat"], axis=0)
            lo = np.nanmin(al["mat"], axis=0)
            hi = np.nanmax(al["mat"], axis=0)
        ax.plot(al["grid"], m, color=c, lw=st["lw"], dashes=st["dashes"],
                zorder=st["zorder"], label=f"{arm} (n={len(al['runs'])})")
        if len(al["runs"]) > 1:
            ax.fill_between(al["grid"], lo, hi, color=c, alpha=0.14, lw=0, zorder=2)
        ok = np.flatnonzero(np.isfinite(m))
        if ok.size:
            ends.append((al["grid"][ok[-1]], m[ok[-1]], arm, P.arm_ink(arm)))
        drawn += 1
    ax.set_xlabel(xlabel)
    if drawn and legend:
        if drawn > P.SERIES_LADDER["hue_alone"]:
            P.declutter(ax, ends, fontsize=7.6)
        else:
            ax.legend(fontsize=7.5, framealpha=0.82, facecolor="white",
                      edgecolor="none", borderpad=0.3, labelspacing=0.3,
                      handlelength=1.6)
    ax.grid(alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    return drawn



def facet_metric(runs, key, **kw):
    """`P.facet_grid` over the seed-mean of `key`, one facet per arm.

    Used wherever the overlay would exceed `P.FACET_ABOVE`: past six series colour
    stops separating them, so the panel must split rather than add another hue.
    """
    series, band = {}, {}
    for arm, rs in by_arm(runs).items():
        al = align_series(rs, key)
        if al is None:
            continue
        with np.errstate(invalid="ignore"):
            series[arm] = (al["grid"], np.nanmean(al["mat"], axis=0))
            if al["mat"].shape[0] > 1:
                band[arm] = (np.nanmin(al["mat"], axis=0), np.nanmax(al["mat"], axis=0))
        kw.setdefault("xlabel", al["xlabel"])
    return P.facet_grid(series, band=band, **kw)


def have_key(runs, key):
    return [r for r in runs if key in r["hist"]
            and np.isfinite(r["hist"][key]).any()]


# Flat metric each run-level panel needs. Used both by the coverage panel and by
# main()'s skip report, so the two can never disagree about what unblocks what.
PANEL_NEEDS = {
    "head_usage": ["train_head_usage_p0"],
    "engagement": ["train_head_usage_entropy", "train_ltv_correction_norm"],
    "training_curves": ["train_loss"],
    "training_health": ["train_loss"],
    "sparsity_trajectories": ["train_l0_frac"],
    "rdmreg_vs_l0": ["train_l0_frac", "train_reg_loss"],
    "loss_decomposition": ["train_loss"],
    "gradient_health": ["grad_norm"],
    "throughput": ["batches_per_sec"],
    "preemption_timeline": ["_timestamp"],
    "seed_variance": ["train_loss"],
    "head_specialisation": ["train_head_usage_p0"],
    "head_switch_burst": ["train_head_switch_rate"],
    "l0_distribution": ["dist/z_l0_per_sample"],
}


# --- load-bearing figures -------------------------------------------------------

def fig_peri_event(eps, out, window=WINDOW):
    """Step 1's money figure: when does each statistic fire relative to contact?

    S_world is built from observed frame-to-frame change, so it can only rise
    once the block has moved, i.e. after t=0. S_model should already be elevated
    at or before t=0. The lag between the two peaks is the claim.
    """
    lags = np.arange(-window, window + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for stat, color in (("S_world", STAT["S_world"]), ("S_model", STAT["S_model"])):
        rows = [r for e in eps for r in peri_event(e[stat], e["onset"], window)]
        if not rows:
            plt.close(fig)
            return skip(f"no onset event in any episode is {window} frames clear "
                        "of the episode edges",
                        "analysis_step1.npz with onset_<i> marking interior frames")
        m, lo, hi = mean_ci(rows)
        ax.plot(lags, m, color=color, lw=2, marker="o", ms=4, label=f"{stat} (n={len(rows)})")
        ax.fill_between(lags, lo, hi, color=color, alpha=0.20, lw=0)
        ax.axvline(lags[int(np.argmax(m))], color=color, ls=":", lw=1, alpha=0.7)
    ax.axvline(0, color="k", lw=1.2)
    ax.annotate("contact onset", (0, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8, va="top")
    ax.set(xlabel="frames relative to contact onset", ylabel="support change $S_t$",
           title="Peri-event average, 95% CI over onset events")
    ax.legend(frameon=False, fontsize=9)
    return save(fig, out, "01_peri_event")


def fig_roc_overlay(eps, out, n_boot=200, seed=0):
    """The Step 1 gate rendered directly, with episode-bootstrap bands."""
    if not eps:
        return skip("no episodes", "--step1 analysis_step1.json + .npz")
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=True)
    for ax, label in zip(axes, ("onset", "block_disp")):
        labs = [e[label] > (0.5 if label == "onset" else np.median(
            np.concatenate([x["block_disp"] for x in eps]))) for e in eps]
        for stat, color in (("S_world", STAT["S_world"]), ("S_model", STAT["S_model"])):
            bands = []
            for _ in range(n_boot):
                k = rng.integers(0, len(eps), len(eps))
                s = np.concatenate([eps[i][stat] for i in k])
                y = np.concatenate([labs[i] for i in k])
                if y.sum() == 0 or (~y.astype(bool)).sum() == 0:
                    continue
                f, t = roc_curve(s, y)
                bands.append(np.interp(grid, f, t))
            f, t = roc_curve(np.concatenate([e[stat] for e in eps]),
                             np.concatenate(labs))
            ax.plot(f, t, color=color, lw=2, label=stat)
            if bands:
                b = np.array(bands)
                ax.fill_between(grid, np.percentile(b, 2.5, axis=0),
                                np.percentile(b, 97.5, axis=0), color=color,
                                alpha=0.20, lw=0)
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        n_pos = int(sum(int(np.sum(l > 0.5)) for l in labs))
        ax.set(xlabel="false positive rate",
               title=f"label: {label}  ({n_pos} positives, {len(eps)} episodes)",
               aspect="equal")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    fig.suptitle("ROC with 95% bands bootstrapped over episodes", y=0.99)
    return save(fig, out, "02_roc_overlay")


def default_baseline(names):
    """The matched upstream control among a set of arms, else the first arm.

    Getting this wrong silently inverts every delta in the panel, so prefer the
    arms the campaign designates as flags-off controls over dict order.
    """
    for pref in ("LpWM-base", "LpWM-ltv", "upstream", "ctrl"):
        hit = [n for n in names if n.startswith(pref) or pref in n]
        if hit:
            return sorted(hit)[0]
    return list(names)[0]


def fig_paired_dumbbell(arms, out, baseline=None, ylabel="CEM success rate"):
    """Paired-seed dumbbells. With n=3 seeds, bar charts with error bars hide the
    paired structure the gates actually test; a line per seed shows it.

    Read it as: both dots joined by a line are ONE training seed. A consistent
    tilt is the effect; crossing lines mean seed spread swamps it. The control end
    of every dumbbell is neutral and the variant end carries the arm's colour, so
    the direction of the tilt is legible without reading the axis labels.
    """
    names = list(arms)
    if len(names) < 2:
        return skip("fewer than two arms", "campaign.json arms with >=2 entries")
    baseline = baseline or default_baseline(names)
    others = [a for a in names if a != baseline]
    fig, axes = plt.subplots(1, len(others), figsize=(2.25 * len(others) + 0.8, 4.4),
                             sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], others):
        eff = paired_effect(arms, baseline, arm)
        if not eff["n"]:
            ax.text(0.5, 0.5, "no shared seed", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
            ax.set_xticks([])
            continue
        cb, cv = P.arm_color(baseline), P.arm_color(arm)
        mb, mv = P.arm_style(baseline)["marker"], P.arm_style(arm)["marker"]
        for s in eff["seeds"]:
            b, v = float(arms[baseline][s]), float(arms[arm][s])
            ax.plot([0, 1], [b, v], color=GHOST, lw=1.4, zorder=1)
            ax.scatter([0], [b], color=cb, marker=mb, s=55, zorder=2,
                       edgecolor="white", linewidth=1.2)
            ax.scatter([1], [v], color=cv, marker=mv, s=55, zorder=2,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(f"s{s}", (1, v), xytext=(6, -3), textcoords="offset points",
                        fontsize=7.5, color="0.45")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([baseline, arm], rotation=22, ha="right", fontsize=7.5)
        for t, m in zip(ax.get_xticklabels(), (baseline, arm)):
            t.set_color(P.arm_ink(m))
        ax.set_xlim(-0.35, 1.5)
        ax.set_title(f"$\\Delta$ = {eff['mean']:+.3f} $\\pm$ {eff['sd']:.3f}\n"
                     f"n={eff['n']}  $d_z$={eff['dz']:+.2f}", fontsize=8.5,
                     color=P.arm_ink(arm))
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(f"Paired seeds: {baseline} (neutral) vs each variant (arm colour)",
                 y=1.02, fontsize=10.5)
    return save(fig, out, "03_paired_dumbbell")


#: (gate title, matched control, variant prefix).
#:
#: The control is NAMED rather than derived from a shared prefix. Under the old
#: "s2_/s3_/s4_" scheme one prefix happened to cover both a gate's control and its
#: variants, so the control came along for free; with mechanism names the baseline
#: is "LpWM-*" and the variants "PiWM-*", and a prefix rule would silently drop the
#: control -- which inverts nothing but leaves every contrast without its reference.
#: "LpWM-ltv" is shared by the gating and union gates, so it appears first in both.
GATE_OF = (("Sparse codes: k-WTA", "LpWM-base", "PiWM-sparse"),
           ("Support gating", "LpWM-ltv", "PiWM-gate"),
           ("Union head", "LpWM-ltv", "PiWM-union"))


def group_arms(arms, groups=None):
    """arm name -> gate, control first.

    "LpWM-ltv" is the shared control of the gating and union gates, so it is placed
    first in both panels rather than given a panel of its own.
    """
    if groups:
        return {g: [a for a in v if a in arms] for g, v in groups.items()}
    out = {}
    for gate, ctrl_name, pre in GATE_OF:
        members = [a for a in arms if a.startswith(pre)]
        ctrl = [a for a in arms if a == ctrl_name]
        if members:
            out[gate] = ctrl + members
    return out


def fig_campaign_overview(arms, out, groups=None, ylabel="CEM success rate"):
    """The side-by-side view: every arm of a gate on one axis, with each seed shown
    and matched seeds joined across arms.

    A bar chart of means would hide exactly what the gates test. The thin lines are
    one training seed carried across arms, so a consistent up or down tilt is the
    effect; crossing lines mean the seed-to-seed spread swamps it. Colour is the
    arm's own identity throughout, so the same arm is the same colour here as in
    every other panel -- the control reads neutral because it is the reference.
    """
    gates = group_arms(arms, groups)
    if not gates:
        return skip("no arm name matched a gate prefix (s2_/s3_/s4_)",
                    'campaign.json "arms" keyed by campaign arm names, or a '
                    '"groups" map naming the panels explicitly')
    fig, axes = plt.subplots(
        1, len(gates), figsize=(1.55 * sum(len(v) for v in gates.values()) + 2.2, 4.8),
        squeeze=False,
        gridspec_kw={"width_ratios": [len(v) for v in gates.values()]},
    )
    for ax, (gate, members) in zip(axes[0], gates.items()):
        seeds = sorted(set.intersection(*[set(arms[a]) for a in members]))
        if not seeds:
            ax.text(0.5, 0.5, "no seed shared by every arm", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8, color="grey")
            ax.set_axis_off()
            continue
        xs = np.arange(len(members))
        for s in seeds:  # paired seed trace
            ax.plot(xs, [arms[a][s] for a in members], color=GHOST, lw=0.9,
                    zorder=1, marker="o", ms=3.0)
        ctrl = np.array([arms[members[0]][s] for s in seeds], float)
        for x, a in zip(xs, members):
            v = np.array([arms[a][s] for s in seeds], float)
            st = P.arm_style(a)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) if v.size > 1 else 0,
                        color=st["color"], lw=2.4, capsize=5, marker=st["marker"],
                        ms=8, mec="white", mew=1.2, zorder=3)
            if x:  # paired delta vs the matched control
                d = v - ctrl
                ax.annotate(f"{d.mean():+.3f}\n$\\pm${d.std(ddof=1):.3f}"
                            if d.size > 1 else f"{d.mean():+.3f}",
                            (x, v.mean()), xytext=(0, -34),
                            textcoords="offset points", ha="center", fontsize=7.5,
                            color=P.arm_ink(a))
        ax.axhline(ctrl.mean(), color=P.arm_color(members[0]), ls=":", lw=1.1, zorder=1)
        ax.set_xticks(xs)
        labels = [m.replace("_pd384", "").replace("LpWM-ltv", "upstream")
                  .replace("s2_", "").replace("s3_", "").replace("s4_", "")
                  for m in members]
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        for t, m in zip(ax.get_xticklabels(), members):
            t.set_color(P.arm_ink(m))
        ax.set_xlim(-0.5, len(members) - 0.5)
        ax.set_title(f"{gate}  (n={len(seeds)} seeds)", fontsize=9.5)
        ax.grid(axis="y", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle("Campaign overview: every arm against its matched control "
                 "(dotted line), thin lines = one seed", y=1.02, fontsize=10.5)
    return save(fig, out, "00_campaign_overview")


def fig_training_health(runs, out):
    """Operational panel: loss and epoch cadence per arm.

    Earned its place the hard way -- packing 7 single-GPU jobs onto one node cut
    per-job throughput from ~4.5 to ~1.2 batch/s, and nothing in the science plots
    would have shown it. A run far off the pack here is contending, not diverging.
    """
    if not have_key(runs, "train_loss"):
        return skip("no run logged train_loss",
                    "runs/outputs/<run>/wandb_history.csv containing train/loss "
                    "(run analysis/export_wandb.py)")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    pal = arm_palette(list(by_arm(runs)))
    plot_arm_bands(a0, runs, "train_loss", pal)
    a0.set(ylabel="train loss", title="Loss per arm (band = seed min-max)")
    for r in runs:
        _, _, xkey = x_axis(r["hist"])
        for m in marks_for(r, xkey):
            a0.axvline(m, color="k", ls=":", lw=0.8, alpha=0.4)

    # hours per epoch: the epoch-level epoch_seconds only exists once an epoch has
    # completed, so fall back to the per-batch projection, which exists from minute
    # one and is the number that actually decides whether a run fits the window.
    names, hrs, src = [], [], ""
    for r in sorted(runs, key=lambda r: r["name"]):
        h = r["hist"]
        if "epoch_seconds" in h and np.isfinite(h["epoch_seconds"]).any():
            names.append(r["name"][:30])
            hrs.append(float(np.nanmean(h["epoch_seconds"]) / 3600.0))
            src = "measured epoch_seconds"
        elif "hours_per_epoch" in h and np.isfinite(h["hours_per_epoch"]).any():
            names.append(r["name"][:30])
            hrs.append(float(np.nanmedian(h["hours_per_epoch"])))
            src = src or "projected from perf/hours_per_epoch"
    if hrs:
        y = np.arange(len(names))
        a1.barh(y, hrs, color=[STATUS["good"] if v <= SLURM_WINDOW_H else STATUS["warn"]
                           for v in hrs])
        a1.axvline(SLURM_WINDOW_H, color="k", ls="--", lw=1.2)
        a1.annotate("3h55m window", (SLURM_WINDOW_H, y[-1]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
        a1.set_yticks(y)
        a1.set_yticklabels(names, fontsize=6.5)
        a1.invert_yaxis()
        a1.set(xlabel="hours per epoch", title=f"Epoch wall-clock ({src})")
        a1.grid(axis="x", alpha=0.25)
    else:
        a1.text(0.5, 0.5, "no epoch_seconds and no perf/hours_per_epoch logged",
                ha="center", va="center", transform=a1.transAxes, fontsize=9,
                color="grey")
        a1.set_axis_off()
    return save(fig, out, "14_training_health")


def fig_head_usage(runs, out, n_heads=4):
    """Step 4's precondition figure. A single band filling the plot means the
    heads collapsed and the run cannot falsify the multimodality claim."""
    runs = [r for r in runs if any(f"train_head_usage_p{j}" in r["hist"]
                                  for j in range(n_heads))]
    if not runs:
        return skip(f"no run logged train_head_usage_p0..p{n_heads-1}",
                    "a J>1 run in the export (train.py logs heads/train_head_usage_pj "
                    "only when n_heads>1)")
    runs = sorted(runs, key=lambda r: r["name"])[:9]
    ncol = min(len(runs), 3)
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.1 * nrow),
                             sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, r in zip(flat, runs):
        h = r["hist"]
        x, xlabel, xkey = x_axis(h)
        ps = [np.nan_to_num(h.get(f"train_head_usage_p{j}", np.zeros_like(x)))
              for j in range(n_heads)]
        ax.stackplot(x, *ps, labels=[f"head {j}" for j in range(n_heads)],
                     colors=P.head_colors(n_heads), alpha=0.95)
        ax.axhline(0.9, color="k", ls="--", lw=1)
        for m in marks_for(r, xkey):
            ax.axvline(m, color="k", ls=":", lw=0.9)
        ax.set(xlabel=xlabel, ylim=(0, 1), title=r["name"][:30])
        ax.title.set_fontsize(8.5)
    for ax in flat[len(runs):]:
        ax.set_axis_off()
    for i in range(nrow):
        axes[i][0].set_ylabel(r"head usage $\bar{p}_j$")
    flat[0].annotate("collapse threshold 0.9", (0.02, 0.9), xycoords=("axes fraction", "data"),
                     xytext=(2, 3), textcoords="offset points", fontsize=7.5)
    flat[ncol - 1].legend(frameon=False, fontsize=8, loc="center left",
                          bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Head usage over training (Step 4 collapse precondition)", y=1.0)
    return save(fig, out, "04_head_usage")



def fig_gate_scorecard(gates, out):
    """All gates on one page: observed effect with across-seed CI against its
    pre-registered threshold. Pass/fail is read off, not argued.

    The verdict is a glyph plus a word in a status colour and sits in its own column
    outside the axes; the interval keeps the gate's own identity. Status never
    becomes the mark's fill, so "which gate" and "did it pass" never compete.
    """
    if not gates:
        return skip("no gates listed", 'campaign.json "gates": [{name, observed, '
                                      'lo, hi, threshold, direction}]')
    rows = []
    for g in gates:
        above = g.get("direction", "above") == "above"
        ok = (g["lo"] > g["threshold"]) if above else (g["hi"] < g["threshold"])
        rows.append({"label": g["name"], "arm": g.get("arm", g["name"]),
                     "mean": g["observed"], "lo": g["lo"], "hi": g["hi"],
                     "threshold": g["threshold"], "verdict": "PASS" if ok else "FAIL",
                     "n": g.get("n")})
    fig = P.forest(rows, arms_for_colour=False,
                   xlabel="observed effect (95% CI across seeds)",
                   title="Gate scorecard",
                   subtitle="black bar = pre-registered threshold  ·  hollow dot = interval "
                            "spans zero  ·  verdict is a glyph, never a fill")
    if P.is_no_data(fig):
        return skip("no gate has a finite effect", "campaign.json gates with numeric lo/hi")
    return save(fig, out, "05_gate_scorecard")


# --- step-1 representation diagnostics ------------------------------------------

def fig_jaccard_decomposition(eps, out):
    """Separates 'support reorganized' from 'magnitudes wrong' as the driver of
    S_model: points off the diagonal moved support, points along it changed scale."""
    lo, hi, on = [], [], []
    for e in eps:
        if "z" not in e:
            return skip("no per-episode z in the npz",
                        "re-run analysis/predictive_jaccard.py so z_<i> is saved")
        z, zc = e["z"][:-1], e["z"][1:]
        lo.append(np.minimum(z, zc).sum(-1))
        hi.append(np.maximum(z, zc).sum(-1))
        on.append(e["onset"][:-1])
    if not lo:
        return skip("no episodes", "--step1 analysis_step1.json + .npz")
    lo, hi, on = np.concatenate(lo), np.concatenate(hi), np.concatenate(on) > 0.5
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(hi[~on], lo[~on], s=5, alpha=0.25, color="grey",
               label=f"no onset (n={int((~on).sum())})")
    ax.scatter(hi[on], lo[on], s=16, alpha=0.9, color=P.CONTACT,
               label=f"onset (n={int(on.sum())})")
    m = max(hi.max(), 1e-9)
    ax.plot([0, m], [0, m], color="k", ls="--", lw=1, label="$J_S=1$")
    ax.set(xlabel=r"$\sum \max(z_t, z_{t+1})$", ylabel=r"$\sum \min(z_t, z_{t+1})$",
           title="Jaccard decomposition")
    ax.legend(frameon=False, fontsize=9)
    return save(fig, out, "06_jaccard_decomposition")


def fig_support_selfsim(eps, out, ep=0, max_frames=120):
    """J_S(z_t, z_s) over all frame pairs. Block-diagonal structure is direct
    visual evidence of discrete modes, which is the Pi-WM thesis."""
    if not eps or ep >= len(eps) or "z" not in eps[ep]:
        return skip("no per-episode z in the npz",
                    "re-run analysis/predictive_jaccard.py so z_<i> is saved")
    z = eps[ep]["z"][:max_frames]
    m = np.minimum(z[:, None], z[None]).sum(-1) / (
        np.maximum(z[:, None], z[None]).sum(-1) + 1e-8)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(m, cmap=P.SEQ, origin="lower", vmin=0, vmax=1)
    for t in np.flatnonzero(eps[ep]["onset"][:max_frames] > 0.5):
        ax.axvline(t, color="w", lw=0.7, alpha=0.7)
    fig.colorbar(im, ax=ax, label="$J_S(z_t, z_s)$")
    ax.set(xlabel="frame $s$", ylabel="frame $t$",
           title=f"Support self-similarity, episode {ep} ({len(z)} frames; "
                 "white = contact onset)")
    return save(fig, out, "07_support_selfsim")


def fig_head_raster(eps, out, ep=0):
    """j* over time with onset markers, plus j* painted onto the (agent, block)
    state plane. If heads own contact regimes, this is the claim in one picture."""
    if not eps or ep >= len(eps):
        return skip("no episodes", "--step1 analysis_step1.json + .npz")
    e = eps[ep]
    if e.get("j_star") is None or "states" not in e:
        return skip("episode has no j_star or no states",
                    "a J>1 checkpoint analysed by analysis/predictive_jaccard.py "
                    "(j_star is only written when model.n_heads > 1)")
    j, st, on = e["j_star"], e["states"], e["onset"] > 0.5
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                 gridspec_kw={"width_ratios": [2, 1.15]})
    hc = matplotlib.colors.ListedColormap(P.head_colors(int(np.nanmax(j)) + 1))
    a0.imshow(j[None], aspect="auto", cmap=hc, vmin=0, vmax=int(np.nanmax(j)),
              extent=[0, len(j), 0, 1])
    for t in np.flatnonzero(on):
        a0.axvline(t, color="k", lw=1.4)
    a0.set(yticks=[], xlabel="frame",
           title=f"head assignment $j^*$ (black = onset, {int(on.sum())} events)")
    sc = a1.scatter(st[:, 0], st[:, 1], c=j, cmap=hc, vmin=0, vmax=int(np.nanmax(j)), s=14)
    a1.scatter(st[:, 2], st[:, 3], facecolors="none", edgecolors="k", s=14, lw=0.5)
    a1.set(xlabel="x", ylabel="y", title="agent (filled) / block (open)")
    fig.colorbar(sc, ax=a1, label="$j^*$")
    return save(fig, out, "08_head_raster")


def fig_scale_perturbation(scale, out):
    """The Step 3 property test as a figure: a support gate is invariant to a
    support-preserving rescale of z, a magnitude gate is not."""
    if not scale:
        return skip("no scale sweep", 'campaign.json "scale": {"<gate_input>": '
                                     '{"c": [...], "rel_change": [...]}}')
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for (name, d), color in zip(scale.items(), P.epoch_colors(max(len(scale), 2))):
        ax.plot(d["c"], d["rel_change"], marker="o", color=color, lw=2, label=name)
    ax.axvline(1.0, color="k", lw=1)
    ax.axhline(0.0, color="grey", ls="--", lw=1)
    ax.set(xlabel="support-preserving scale $c$",
           ylabel="relative change in predictor output",
           title="Scale perturbation by gate input")
    ax.legend(frameon=False, fontsize=9)
    return save(fig, out, "09_scale_perturbation")


def fig_engagement(runs, out):
    """Did the mechanism ever turn on? A flat LTV correction norm or a p_bar
    entropy pinned at 0 means the run is void under our preconditions.

    Missing keys are named on the panel rather than dropped, because 'the metric
    was never logged' and 'the mechanism never engaged' are different findings and
    only the second one is a result.
    """
    keys = [("train_ltv_correction_norm", "LTV correction norm"),
            ("train_head_usage_entropy", r"head usage entropy $H(\bar{p})$"),
            ("train_head_usage_max", r"max head usage $\max_j \bar{p}_j$"),
            ("train_gate_mean", "mean gate magnitude")]
    present = [(k, lab) for k, lab in keys if have_key(runs, k)]
    absent = [k for k, _ in keys if not have_key(runs, k)]
    if not present:
        return skip("none of " + ", ".join(k for k, _ in keys) + " was logged",
                    "an export from a run that logs at least one engagement metric")
    fig, axes = plt.subplots(1, len(present), figsize=(4.7 * len(present), 3.9),
                             squeeze=False)
    pal = arm_palette(list(by_arm(runs)))
    for ax, (k, lab) in zip(axes[0], present):
        plot_arm_bands(ax, runs, k, pal)
        ax.set_ylabel(lab)
        if k == "train_head_usage_entropy":
            ax.axhline(np.log(4), color="k", ls="--", lw=1)
            ax.annotate(r"$\ln J$ ceiling (J=4)", (0.02, np.log(4)),
                        xycoords=("axes fraction", "data"), xytext=(2, 3),
                        textcoords="offset points", fontsize=7.5)
    note = ("not logged by any run: " + ", ".join(absent)) if absent else ""
    fig.suptitle("Engagement diagnostics" + (f"\n{note}" if note else ""),
                 y=1.04, fontsize=10)
    return save(fig, out, "10_engagement")



def fig_success_vs_k(k_sweep, out):
    """Success and the regulariser floor against k/D, on two stacked axes.

    These were drawn on a TWIN Y AXIS, which is the one chart form that can
    manufacture a correlation that is not in the data: the vertical alignment of the
    two scales is arbitrary, so "the curves cross here" means nothing. Stacking them
    over a shared x keeps every real comparison (where each curve turns, whether the
    floor moves when success does) and discards only the false one.
    """
    if not k_sweep:
        return skip("no k sweep", 'campaign.json "k_sweep": {"<k/D>": '
                                 '{"success": s, "rdmreg": r}}')
    # keep the original string keys next to their float value: round-tripping
    # through str() loses "0.10" and used to raise KeyError on a valid sweep
    items = sorted(((float(k), v) for k, v in k_sweep.items()), key=lambda kv: kv[0])
    ks = [k for k, _ in items]
    succ = [v["success"] for _, v in items]
    reg = [v["rdmreg"] for _, v in items]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                                 gridspec_kw={"hspace": 0.12})
    a0.plot(ks, succ, marker="o", color=SOLO, lw=2)
    a0.set(ylabel="CEM success rate")
    a0.set_title(f"Success and regularizer floor vs sparsity ({len(ks)} cells)\n"
                 "one measure per axis -- a twin y-axis would invent a crossing point",
                 fontsize=10.5, loc="left")
    a1.plot(ks, reg, marker="s", color=STAT["S_model"], lw=2)
    a1.set(xscale="log", xlabel="$k/D$", ylabel="RDMReg loss")
    for ax, vals in ((a0, succ), (a1, reg)):
        j = int(np.argmax(vals)) if ax is a0 else int(np.argmin(vals))
        ax.annotate(f"{vals[j]:.3g} at k/D={ks[j]:g}", (ks[j], vals[j]), xytext=(6, 6),
                    textcoords="offset points", fontsize=7.8, color="0.35")
        ax.grid(alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    return save(fig, out, "11_success_vs_k")


def fig_burst_vs_error(eps, out):
    """Do bursts coincide with prediction error, or are they cosmetic?"""
    if not eps:
        return skip("no episodes", "--step1 analysis_step1.json + .npz")
    s = np.concatenate([e["S_model"] for e in eps])
    d = np.concatenate([e["block_disp"] for e in eps])
    on = np.concatenate([e["onset"] for e in eps]) > 0.5
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(d[~on], s[~on], s=5, alpha=0.25, color="grey",
               label=f"no onset (n={int((~on).sum())})")
    ax.scatter(d[on], s[on], s=18, alpha=0.9, color=P.CONTACT,
               label=f"onset (n={int(on.sum())})")
    r = np.corrcoef(d, s)[0, 1] if d.std() and s.std() else np.nan
    ax.set(xlabel="block displacement", ylabel="$S_{model}$",
           title=f"Burst vs prediction error (r = {r:.3f})")
    ax.legend(frameon=False, fontsize=9)
    return save(fig, out, "12_burst_vs_error")


def fig_training_curves(runs, out):
    """Every run is chopped every 4h; a step at a resume marker means the resume
    is lossy. Infra diagnostic this project specifically needs.

    Train and val loss are drawn on the same axes per arm so a widening gap reads
    as overfitting rather than as two unrelated panels.
    """
    if not have_key(runs, "train_loss"):
        return skip("no run logged train_loss",
                    "runs/outputs/<run>/wandb_history.csv containing train/loss "
                    "(run analysis/export_wandb.py)")
    has_val = bool(have_key(runs, "val_loss"))
    fig, axes = plt.subplots(1, 2 if has_val else 1,
                             figsize=(6.6 * (2 if has_val else 1), 4.2),
                             squeeze=False, sharex=True)
    pal = arm_palette(list(by_arm(runs)))
    for ax, key, title in zip(axes[0], ["train_loss", "val_loss"],
                              ["train loss", "val loss"]):
        plot_arm_bands(ax, runs, key, pal)
        ax.set(ylabel=title, title=f"{title}; dotted = preemption resume")
        for r in runs:
            _, _, xkey = x_axis(r["hist"])
            for m in marks_for(r, xkey):
                ax.axvline(m, color="k", ls=":", lw=0.9, alpha=0.5)
    return save(fig, out, "13_training_curves")


# --- run-history diagnostics (unblocked by analysis/export_wandb.py) ------------

def fig_metric_coverage(runs, out):
    """Which panel can render for which run, as a matrix.

    This is the panel to look at when something is missing: a column of grey cells
    says the metric was never logged (a training-side fix), a single grey cell says
    that one run is behind (re-export). The bottom strip counts runs per metric.
    """
    if not runs:
        return skip("no run dirs with an export",
                    "runs/outputs/<run>/wandb_history.csv "
                    "(run analysis/export_wandb.py)")
    keys, seen = [], set()
    for ks in PANEL_NEEDS.values():
        for k in ks:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    names = [r["name"] for r in runs]
    m = np.zeros((len(runs), len(keys)))
    for i, r in enumerate(runs):
        for j, k in enumerate(keys):
            if k.startswith("dist/"):
                m[i, j] = 1.0 if r["hists"].get(k) else 0.0
            else:
                v = r["hist"].get(k)
                m[i, j] = 1.0 if v is not None and np.isfinite(v).any() else 0.0
    fig, ax = plt.subplots(figsize=(0.42 * len(keys) + 4.4, 0.26 * len(runs) + 2.6))
    ax.imshow(m, cmap=matplotlib.colors.ListedColormap(["#e8e8e8", STATUS["good"]]),
              aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    # the run count goes in the tick label rather than an annotation inside the
    # axes, where it would be clipped by the bottom row of cells
    ax.set_xticklabels([f"{k}  [{int(m[:, j].sum())}/{len(runs)}]"
                        for j, k in enumerate(keys)], rotation=60, ha="right",
                       fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:34] for n in names], fontsize=6.5)
    ax.set(title=f"Metric coverage: {len(runs)} runs x {len(keys)} metrics "
                 "(green = present, grey = never logged)\n"
                 "[n/N] after each metric = runs carrying it")
    ax.title.set_fontsize(9.5)
    return save(fig, out, "15_metric_coverage")


def fig_sparsity_trajectories(runs, out):
    """Does each arm hold the code density it is supposed to?

    k-WTA pins density at k/D by construction, so a k-WTA arm whose density drifts
    is not running the intervention the gate names. The per-sample spread beside it
    is the other half: a correct mean with a wide spread is a code that is dense on
    some samples and empty on others, which is not what "k active units" means.

    Faceted rather than overlaid: nine arms is past the count colour can separate
    (`P.needs_facet`), and this panel overlaid all nine, which is why it was
    unreadable. Each facet keeps every other arm ghosted, so the comparison survives.
    """
    if not have_key(runs, "train_l0_frac"):
        return skip("no run logged train_l0_frac",
                    "export containing sparsity/l0_frac or train/l0_frac")
    arms = list(by_arm(runs))
    if P.needs_facet(arms):
        fig = facet_metric(runs, "train_l0_frac", ylabel=r"$\rho$ = l0 fraction",
                           title=r"Code density $\rho$ over training",
                           subtitle="a k-WTA arm whose density drifts off k/D is not running "
                                    "the intervention its gate names")
        if P.is_no_data(fig):
            return skip("no arm has an alignable train_l0_frac", "a run logging l0_frac")
        return save(fig, out, "16_sparsity_trajectories")
    has_std = bool(have_key(runs, "l0_std_across_samples"))
    fig, axes = plt.subplots(1, 2 if has_std else 1, figsize=(11.6 if has_std else 6.4, 4.2),
                             squeeze=False)
    pal = arm_palette(arms)
    plot_arm_bands(axes[0][0], runs, "train_l0_frac", pal)
    axes[0][0].set(ylabel=r"$\rho$ = l0 fraction", title=r"Code density $\rho$ over training")
    if has_std:
        plot_arm_bands(axes[0][1], runs, "l0_std_across_samples", pal, legend=False)
        axes[0][1].set(ylabel=r"std of per-sample $L_0$", title="Per-sample sparsity spread")
    for a in axes[0]:
        a.title.set_fontsize(9.5)
    return save(fig, out, "16_sparsity_trajectories")



def fig_rdmreg_vs_l0(runs, out):
    """Is there an irreducible RDMReg floor at high sparsity, and did mu-matching
    remove it?

    RDMReg pulls density toward its target while k-WTA pins it at k/D; if the two
    disagree the W2 term cannot reach zero and any degradation is objective
    conflict, not sparsity. A floor is a place a trajectory STOPS MOVING, so the
    panel is a phase plane -- density against regulariser loss, walked over training
    -- rather than the bar of "final loss per arm" this used to be. Time is the
    lightness of the arm's own hue, so the trace carries both identity and when.
    """
    need = [r for r in runs if "train_l0_frac" in r["hist"]
            and "train_reg_loss" in r["hist"]]
    if not need:
        return skip("no run logged both train_l0_frac and train_reg_loss",
                    "export containing sparsity/train_l0_frac and train/reg_loss")
    traj = {}
    for arm, rs in by_arm(need).items():
        # one representative seed per arm: overlaying three seeds' phase traces
        # turns the facet into a scribble and the floor is a per-arm property
        r = rs[0]
        x, y = r["hist"]["train_l0_frac"], r["hist"]["train_reg_loss"]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            traj[arm] = (x[m], y[m])
    # per-step logs oscillate hard enough to hide the path; smooth over ~2% of the
    # series so the WALK is visible, and let the panel disclose that in its subtitle
    win = max(1, int(0.02 * np.median([len(v[0]) for v in traj.values()]))) if traj else 1
    fig = P.phase_plane(traj, smooth=win, xlabel=r"code density $\rho$", ylabel="RDMReg loss",
                        title="Where each arm walks to, and where it stops",
                        subtitle="hollow = first logged step, filled = last;  an arm that "
                                 "stalls at high loss with density pinned is paying the floor")
    if P.is_no_data(fig):
        return skip("no arm has two aligned finite series",
                    "a run with >=2 finite train_l0_frac / train_reg_loss points")
    return save(fig, out, "17_rdmreg_vs_l0")


LOSS_TERMS = (("train_loss", "total"), ("train_z_loss", "z (prediction)"),
              ("train_reg_loss", "RDMReg"), ("train_z_visual_loss", "z visual"),
              ("train_diag_cov_loss", "diag cov"),
              ("train_diag_var_perdim", "diag var/dim"))


def fig_loss_decomposition(runs, out):
    """Which loss term does each intervention actually move?

    The total loss hides the trade the interventions make: k-WTA can cut the
    regularizer while worsening prediction, and a gate change can leave the total
    flat while shifting weight between terms. One axis per term, one band per arm.
    """
    terms = [(k, lab) for k, lab in LOSS_TERMS if have_key(runs, k)]
    if not terms:
        return skip("no loss component logged",
                    "export containing train/loss (and ideally train/z_loss, "
                    "train/reg_loss)")
    ncol = min(3, len(terms))
    nrow = int(np.ceil(len(terms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.3 * nrow),
                             squeeze=False)
    pal = arm_palette(list(by_arm(runs)))
    flat = axes.ravel()
    for i, (ax, (k, lab)) in enumerate(zip(flat, terms)):
        plot_arm_bands(ax, runs, k, pal, legend=(i == 0))
        ax.set(ylabel=lab, title=lab)
        ax.title.set_fontsize(9)
        pos = np.concatenate([r["hist"][k][np.isfinite(r["hist"][k])]
                              for r in have_key(runs, k)])
        if pos.size and pos.min() > 0 and pos.max() / max(pos.min(), 1e-12) > 50:
            ax.set_yscale("log")
    for ax in flat[len(terms):]:
        ax.set_axis_off()
    fig.suptitle("Loss decomposition per arm (band = seed min-max; log y where the "
                 "term spans >50x)", y=1.01, fontsize=10)
    return save(fig, out, "18_loss_decomposition")


def fig_gradient_health(runs, out):
    """Is any run silently exploding, vanishing, or on a different LR schedule?

    A bf16 campaign can produce a run whose loss curve looks ordinary while its
    predictor gradient norm has collapsed by three orders of magnitude. Log y, one
    band per arm, and the LR beside it so a schedule mismatch between arms (which
    would invalidate the matched-control comparison) is visible.
    """
    if not have_key(runs, "grad_norm"):
        return skip("no run logged grad_norm",
                    "export containing opt/grad_norm (train.py logs it per batch "
                    "when has_predictor is true)")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.1),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    pal = arm_palette(list(by_arm(runs)))
    plot_arm_bands(a0, runs, "grad_norm", pal)
    a0.set(ylabel="predictor grad norm", title="Gradient norm (log y)")
    a0.set_yscale("log")
    if plot_arm_bands(a1, runs, "lr", pal, legend=False):
        a1.set(ylabel="learning rate", title="LR schedule (must match across arms)")
    else:
        a1.text(0.5, 0.5, "opt/lr not logged", ha="center", va="center",
                transform=a1.transAxes, fontsize=9, color="grey")
        a1.set_axis_off()
    flagged = []
    for r in have_key(runs, "grad_norm"):
        g = r["hist"]["grad_norm"]
        g = g[np.isfinite(g)]
        if g.size and (g[-max(1, len(g) // 10):].mean() < 1e-6 or g.max() > 1e3):
            flagged.append(r["name"])
    if flagged:
        a0.set_title("Gradient norm (log y) -- CHECK: " + ", ".join(flagged[:3]),
                     fontsize=9, color=STATUS["crit"])
    return save(fig, out, "19_gradient_health")



def fig_throughput(runs, out):
    """Is a slow run contending for a node, or is something wrong with the run?

    Training here is dataloader-bound, so co-tenancy on a shared node -- not the
    science -- sets throughput. A median per run, which is what this drew before,
    hides exactly the thing that matters: a run is slow because it is INTERMITTENTLY
    slow, when a neighbour lands. The strip keeps every observation.
    """
    have = have_key(runs, "batches_per_sec")
    if not have:
        return skip("no run logged batches_per_sec",
                    "export containing perf/batches_per_sec")
    groups = {}
    for arm, rs in by_arm(have).items():
        v = np.concatenate([r["hist"]["batches_per_sec"] for r in rs])
        v = v[np.isfinite(v)]
        if v.size:
            groups[arm] = v
    med = float(np.median(np.concatenate(list(groups.values())))) if groups else np.nan
    fig = P.strip_plot(groups, xlabel="batches / s", ref=med,
                       ref_label=f"campaign median {med:.2f}",
                       title=f"Throughput, every logged window ({len(have)} runs)",
                       subtitle="bar = interquartile range, tick = median, dots = individual "
                                "windows -- a left tail is co-tenancy, a shifted bar is the run")
    if P.is_no_data(fig):
        return skip("no run has finite batches_per_sec", "a run logging perf/batches_per_sec")
    return save(fig, out, "20_throughput")


def fig_preemption_timeline(runs, out):
    """How much of the wall clock is actually training?

    One row per run on a shared wall-clock axis, built from the logged timestamps:
    filled bars are windows where the run was logging, gaps are queue time after a
    4h eviction. The utilisation figure per row is the honest ETA input; the loss
    curves cannot show it because they are plotted against steps, not time.
    """
    have = [r for r in runs if "_timestamp" in r["hist"]
            and np.isfinite(r["hist"]["_timestamp"]).sum() > 2]
    if not have:
        return skip("no run has _timestamp in its history",
                    "re-export with analysis/export_wandb.py, which keeps the "
                    "wandb _timestamp column")
    t0 = min(np.nanmin(r["hist"]["_timestamp"]) for r in have)
    have = sorted(have, key=lambda r: r["name"])
    fig, ax = plt.subplots(figsize=(11.0, 0.30 * len(have) + 2.4))
    pal = arm_palette(list(by_arm(have)))
    for i, r in enumerate(have):
        ts = r["hist"]["_timestamp"]
        ts = np.sort(ts[np.isfinite(ts)])
        h = (ts - t0) / 3600.0
        gaps = np.diff(h)
        med = np.median(gaps) if gaps.size else 0.0
        cut = np.flatnonzero(gaps > max(0.05, 8 * med))
        bounds = np.concatenate([[0], cut + 1, [len(h)]])
        active = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 1:
                continue
            seg = h[a:b]
            ax.barh(i, max(seg[-1] - seg[0], 0.01), left=seg[0], height=0.6,
                    color=pal.get(r["arm"], SOLO))
            active += seg[-1] - seg[0]
        span = h[-1] - h[0] if h.size > 1 else 0.0
        ax.annotate(f"{active:.1f}h / {span:.1f}h"
                    + (f"  {len(cut)} gap" + ("s" if len(cut) != 1 else "")
                       if len(cut) else ""),
                    (h[-1], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=6.5)
    for k in range(1, int(max(np.nanmax((r["hist"]["_timestamp"] - t0) / 3600.0)
                              for r in have) // SLURM_WINDOW_H) + 2):
        ax.axvline(k * SLURM_WINDOW_H, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([r["name"][:34] for r in have], fontsize=6.5)
    ax.invert_yaxis()
    ax.set(xlabel="hours since the first log line of the campaign",
           title="Preemption timeline: filled = logging, blank = evicted/queued; "
                 f"dotted = {SLURM_WINDOW_H:.2f}h window boundaries")
    ax.title.set_fontsize(9.5)
    ax.margins(x=0.12)
    return save(fig, out, "21_preemption_timeline")



def fig_seed_variance(runs, out, key="train_loss"):
    """Is the arm effect bigger than seed noise -- yet?

    Every gate is a between-arm comparison at n=3 seeds, so the honest question is
    whether the between-arm spread has separated from the within-arm spread. The
    answer is ONE NUMBER, the intraclass correlation, which this used to draw as two
    rectangles whose ratio the reader had to estimate by eye. It is printed instead,
    with the seed-noise trajectory that produced it beside it.
    """
    groups = {a: rs for a, rs in by_arm(runs).items()
              if len([r for r in rs if key in r["hist"]]) > 1}
    if not groups:
        return skip(f"no arm has >=2 seeds carrying {key}",
                    f"exports for at least two seeds of one arm with {key}")
    series, per_arm, xlabel = {}, {}, "logged sample"
    for arm, rs in groups.items():
        al = align_series(rs, key)
        if al is None or al["mat"].shape[0] < 2:
            continue
        xlabel = al["xlabel"]
        with np.errstate(invalid="ignore"):
            series[arm] = (al["grid"], np.nanstd(al["mat"], axis=0, ddof=1))
        ok = np.isfinite(al["mat"]).all(axis=0)
        if ok.any():
            per_arm[arm] = al["mat"][:, np.flatnonzero(ok)[-1]]
    b, w, icc = variance_decomposition(
        {a: {i: v for i, v in enumerate(vals)} for a, vals in per_arm.items()})

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.4, 6.2),
                                 gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.30})
    for arm, (g, sd) in series.items():
        st = P.arm_style(arm)
        a0.plot(g, sd, color=st["color"], lw=st["lw"], dashes=st["dashes"],
                zorder=st["zorder"])
        a0.annotate(arm, (g[-1], sd[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=7.8, color=P.arm_ink(arm), va="center")
    a0.set(xlabel=xlabel, ylabel=f"across-seed std of {key}")
    a0.set_title("Seed noise over training\n"
                 "direct-labelled rather than legended: at this series count a legend "
                 "box is a second lookup", fontsize=10.5, loc="left")
    a0.grid(alpha=0.35)
    a0.spines[["top", "right"]].set_visible(False)

    a1.set_axis_off()
    if np.isfinite(icc):
        verdict = ("crit" if icc < 0.1 else "warn" if icc < 0.5 else "good")
        note = {"crit": "seed noise dominates -- the campaign cannot resolve the arms",
                "warn": "arms are only marginally separable at this n",
                "good": "between-arm spread has separated from seed noise"}[verdict]
        tiles = [("Intraclass correlation", f"{icc:.2f}", note, verdict),
                 ("Between-arm variance", f"{b:.3g}", f"{len(per_arm)} arms at the last common step", None),
                 ("Within-arm variance", f"{w:.3g}", "across seeds of one arm", None)]
        P.stat_tiles(tiles, ax=a1,
                     title=f"Variance split of {key} at the last common step")
    else:
        a1.text(0.5, 0.5, "need >=2 arms reaching a common step", ha="center",
                va="center", transform=a1.transAxes, fontsize=9.5, color="grey")
    return save(fig, out, "22_seed_variance")


def fig_head_specialisation(runs, out, n_heads=4):
    """Do the union head's J readouts divide the work, and does the entropy bonus
    buy anything?

    Left: the final usage vector per run, sorted within the run, so collapse reads
    as one dark cell and a genuine split as an even row. Right: entropy against
    lambda_ent with the ln(J) ceiling; if the two entropy arms land on top of each
    other the coefficient is doing nothing and the Step 4 contrast is void.
    """
    have = [r for r in runs
            if any(f"train_head_usage_p{j}" in r["hist"] for j in range(n_heads))]
    if not have:
        return skip(f"no run logged train_head_usage_p0..p{n_heads-1}",
                    "an export from a J>1 run")
    have = sorted(have, key=lambda r: r["name"])
    m = np.full((len(have), n_heads), np.nan)
    for i, r in enumerate(have):
        vals = []
        for j in range(n_heads):
            v = r["hist"].get(f"train_head_usage_p{j}")
            v = v[np.isfinite(v)] if v is not None else np.array([])
            vals.append(float(v[-max(1, len(v) // 10):].mean()) if v.size else np.nan)
        m[i] = sorted(vals, reverse=True)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 0.30 * len(have) + 3.0),
                                 gridspec_kw={"width_ratios": [1.25, 1],
                                              "wspace": 0.42})
    im = a0.imshow(m, cmap=P.SEQ + "_r", aspect="auto", vmin=0, vmax=1)
    for i in range(len(have)):
        for j in range(n_heads):
            if np.isfinite(m[i, j]):
                a0.annotate(f"{m[i,j]:.2f}", (j, i), ha="center", va="center",
                            fontsize=6.5,
                            color="w" if m[i, j] > 0.55 else "k")
    a0.set_xticks(range(n_heads))
    a0.set_xticklabels([f"rank {j+1}" for j in range(n_heads)], fontsize=8)
    a0.set_yticks(range(len(have)))
    a0.set_yticklabels([r["name"][:32] for r in have], fontsize=6.5)
    a0.set_title(rf"Final $\bar{{p}}_j$, sorted within run "
                 rf"(uniform = {1.0/n_heads:.2f}; >0.9 in rank 1 = collapse)",
                 fontsize=9)
    fig.colorbar(im, ax=a0, label=r"$\bar{p}_j$", fraction=0.046)

    ents, lams, labels = [], [], []
    for r in have:
        e = r["hist"].get("train_head_usage_entropy")
        e = e[np.isfinite(e)] if e is not None else np.array([])
        if not e.size:
            continue
        ents.append(float(e[-max(1, len(e) // 10):].mean()))
        lams.append(float((r["meta"].get("config") or {})
                          .get("head_entropy_coef", np.nan)))
        labels.append(r["arm"])
    if ents and np.isfinite(lams).any():
        pal = arm_palette(sorted(set(labels)))
        jitter = np.random.default_rng(0).normal(0, 0.004, len(lams))
        for lam, e, lab, jj in zip(lams, ents, labels, jitter):
            a1.scatter(lam + jj, e, s=52, color=pal[lab], zorder=3,
                       label=lab if lab not in a1.get_legend_handles_labels()[1] else None)
        a1.axhline(np.log(n_heads), color="k", ls="--", lw=1.1)
        a1.annotate(rf"$\ln J = {np.log(n_heads):.2f}$ (uniform)",
                    (0.02, np.log(n_heads)), xycoords=("axes fraction", "data"),
                    xytext=(2, 3), textcoords="offset points", fontsize=7.5)
        a1.set(xlabel=r"$\lambda_{ent}$ (head_entropy_coef)",
               ylabel=r"final $H(\bar{p})$",
               title=f"Entropy vs its coefficient (n={len(ents)} runs)")
        a1.legend(frameon=False, fontsize=7)
        a1.grid(alpha=0.2)
    else:
        a1.text(0.5, 0.5, "train_head_usage_entropy or head_entropy_coef missing\n"
                          "(entropy needs the export; the coefficient needs "
                          "wandb_meta.json)",
                ha="center", va="center", transform=a1.transAxes, fontsize=8.5,
                color="grey")
        a1.set_axis_off()
    return save(fig, out, "23_head_specialisation")


def fig_head_switch_burst(runs, out):
    """Is head switching event-driven or just churn?

    The union-head claim is that heads own dynamics regimes, which predicts
    switching concentrated at bursts. A high switch rate that is flat and
    uncorrelated with the burst rate is the opposite: the argmin is thrashing and
    the heads are interchangeable.
    """
    if not have_key(runs, "train_head_switch_rate"):
        return skip("no run logged train_head_switch_rate",
                    "an export from a J>1 run (train.py logs "
                    "heads/train_head_switch_rate only when n_heads>1)")
    has_burst = bool(have_key(runs, "train_head_burst_rate"))
    fig, axes = plt.subplots(1, 3 if has_burst else 1,
                             figsize=(3.9 * (3 if has_burst else 1) + 0.6, 3.9),
                             squeeze=False)
    pal = arm_palette(list(by_arm(runs)))
    plot_arm_bands(axes[0][0], runs, "train_head_switch_rate", pal)
    axes[0][0].set(ylabel="head switch rate", title="Switch rate over training")
    if has_burst:
        plot_arm_bands(axes[0][1], runs, "train_head_burst_rate", pal, legend=False)
        axes[0][1].set(ylabel="burst rate", title="Burst rate over training")
        ax = axes[0][2]
        for arm, rs in by_arm(runs).items():
            xs, ys = [], []
            for r in rs:
                a = r["hist"].get("train_head_burst_rate")
                b = r["hist"].get("train_head_switch_rate")
                if a is None or b is None:
                    continue
                m = np.isfinite(a) & np.isfinite(b)
                xs.append(a[m])
                ys.append(b[m])
            if not xs:
                continue
            x, y = np.concatenate(xs), np.concatenate(ys)
            if not x.size:
                continue
            r_ = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan
            ax.scatter(x, y, s=8, alpha=0.4, color=pal[arm],
                       label=f"{arm} r={r_:+.2f}")
        ax.set(xlabel="burst rate", ylabel="switch rate",
               title="Switching vs bursts (r per arm)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.2)
    for a in axes[0]:
        a.title.set_fontsize(9)
    return save(fig, out, "24_head_switch_burst")


def fig_l0_distribution(runs, out, key="dist/z_l0_per_sample", max_runs=4):
    """Is the code's sparsity tight around its target, or two populations?

    l0_frac is a mean, and a mean of 0.29 is produced equally well by every sample
    at 0.29 and by half the samples at 0.05 and half at 0.53. Only the logged
    histogram can tell those apart, and the second one would invalidate the
    density-matching argument the k-WTA arm rests on. Each column is one logged
    histogram, normalised, so brightness is where the mass sits.
    """
    have = [r for r in runs if r["hists"].get(key)]
    if not have:
        return skip(f"no run has {key} in wandb_hists.json",
                    "re-export with analysis/export_wandb.py (train.py logs it "
                    "every diag_every_x_batch batches)")
    # most snapshots first: a run with one snapshot cannot show a trend, and early
    # in a campaign most runs have exactly one
    have = sorted(have, key=lambda r: (-len(r["hists"][key]), r["name"]))[:max_runs]
    fig, axes = plt.subplots(1, len(have), figsize=(3.6 * len(have) + 1.2, 4.0),
                             squeeze=False, sharey=True)
    im = None
    for ax, r in zip(axes[0], have):
        got = hist_matrix(r["hists"][key])
        if got is None:
            ax.set_axis_off()
            continue
        steps, edges, mat = got
        # scale to the panel's own mass, not to 1.0: spread over 48 bins the peak
        # density of a healthy unimodal snapshot is ~0.1 and a 0-1 scale renders
        # every column uniformly black
        im = ax.pcolormesh(np.arange(len(steps) + 1) - 0.5, edges, mat.T,
                           cmap=P.SEQ, shading="auto", vmin=0,
                           vmax=max(float(mat.max()), 1e-6))
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([f"{int(s)}" for s in steps], rotation=70, fontsize=6)
        note = "single snapshot -- no trend yet" if len(steps) < 2 else \
            f"{len(steps)} snapshots"
        ax.set(xlabel="train batch", title=f"{r['name'][:26]}\n{note}")
        ax.title.set_fontsize(8.5)
    axes[0][0].set_ylabel(key.split("/")[-1])
    if im is not None:
        fig.colorbar(im, ax=axes[0][-1], label="density within snapshot",
                     fraction=0.046)
    fig.suptitle(f"{key} over training (each column is one logged histogram, "
                 "scaled to its own peak)", y=1.04, fontsize=10)
    return save(fig, out, "25_l0_distribution")


# --- step-1 code-geometry and head-alignment diagnostics ------------------------

def fig_code_geometry(eps, out):
    """Is the sparse code actually using its width, or a handful of units?

    Three views of the same z. Left: units ranked by activation frequency on log
    axes -- a Zipf-like straight line means a few units carry almost everything,
    which is a degenerate code that would still report a healthy mean l0_frac.
    Middle: participation ratio per frame against D, the effective dimension.
    Right: frequency against magnitude, which separates 'rarely on but large' from
    'always on but small'.
    """
    zs = [e["z"] for e in eps if "z" in e]
    if not zs:
        return skip("no per-episode z in the npz",
                    "re-run analysis/predictive_jaccard.py so z_<i> is saved")
    z = np.concatenate(zs, axis=0)
    D = z.shape[1]
    freq, mag = unit_activity(z)
    order = np.argsort(-freq)
    pr_frames = np.array([participation_ratio(row) for row in z])
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.0, 3.9))
    a0.plot(np.arange(1, D + 1), np.maximum(freq[order], 1e-6), color=SOLO, lw=2)
    a0.set(xscale="log", yscale="log", xlabel="unit rank",
           ylabel="activation frequency",
           title=f"Rank-frequency curve (D={D})\n"
                 f"dead units: {dead_fraction(z)*100:.1f}%")
    a0.grid(alpha=0.2, which="both")
    a0.annotate(f"PR over units = {participation_ratio(freq):.1f} / {D}",
                (0.03, 0.06), xycoords="axes fraction", fontsize=8)
    a1.hist(pr_frames, bins=40, color=SOLO, alpha=0.55, lw=0)
    a1.axvline(float(np.median(pr_frames)), color=P.arm_ink("x", 0.45), lw=2)
    a1.annotate(f"median {np.median(pr_frames):.1f}",
                (np.median(pr_frames), a1.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8, va="top", color=STATUS["warn"])
    a1.set(xlabel="participation ratio per frame", ylabel="frames",
           title=f"Effective dimension ({len(z)} frames, D={D})")
    a2.scatter(np.maximum(freq, 1e-6), np.maximum(mag, 1e-9), s=9, alpha=0.5,
               color=SOLO)
    a2.set(xscale="log", yscale="log", xlabel="activation frequency",
           ylabel="mean magnitude when active",
           title="Frequency vs magnitude per unit")
    a2.grid(alpha=0.2, which="both")
    for a in (a0, a1, a2):
        a.title.set_fontsize(9)
    return save(fig, out, "26_code_geometry")


def fig_onset_lead_lag(eps, out, window=WINDOW, n_null=200, seed=0):
    """Does S_model lead contact onset, and by more than chance?

    The peri-event panel shows the shapes; this reduces them to the one number the
    Step 1 claim needs -- the lag at which each statistic best explains onset --
    and puts a null band under it by circularly shifting the onset train, which
    preserves each statistic's autocorrelation. A peak left of zero is a lead.
    """
    if not eps:
        return skip("no episodes", "--step1 analysis_step1.json + .npz")
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    lags = np.arange(-window, window + 1)
    null = []
    base_on = np.concatenate([e["onset"] for e in eps])
    base_sig = np.concatenate([e["S_model"] for e in eps])
    for _ in range(n_null):
        sh = np.roll(base_on, int(rng.integers(window + 1, max(len(base_on) - window, window + 2))))
        _, r = xcorr_lag(base_sig, sh, window)
        null.append(r)
    null = np.array(null, dtype=float)
    if np.isfinite(null).any():
        ax.fill_between(lags, np.nanpercentile(null, 2.5, axis=0),
                        np.nanpercentile(null, 97.5, axis=0), color="grey",
                        alpha=0.25, lw=0, label=f"shifted-onset null (n={n_null})")
    for stat, color in (("S_world", STAT["S_world"]), ("S_model", STAT["S_model"])):
        sig = np.concatenate([e[stat] for e in eps])
        _, r = xcorr_lag(sig, base_on, window)
        ax.plot(lags, r, color=color, lw=2, marker="o", ms=4, label=stat)
        if np.isfinite(r).any():
            k = int(np.nanargmax(r))
            ax.axvline(lags[k], color=color, ls=":", lw=1)
            ax.annotate(f"{stat} peak at {lags[k]:+d}", (lags[k], np.nanmax(r)),
                        xytext=(6, 2), textcoords="offset points", fontsize=8,
                        color=color)
    ax.axvline(0, color="k", lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="frames of the statistic relative to contact onset "
                  "(negative = statistic leads)",
           ylabel="Pearson r with the onset train",
           title=f"Lead-lag alignment with contact onset "
                 f"({int(base_on.sum())} onsets, {len(eps)} episodes)")
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.2)
    return save(fig, out, "27_onset_lead_lag")


def fig_head_onset_alignment(eps, out, window=WINDOW, n_heads=4):
    """Do the union head's readouts line up with contact, or with nothing?

    P(j* = j | lag from onset) as a heatmap: a head that owns contact shows a bright
    band in one row near lag 0 while the others dim. A flat map is the null result
    -- heads partition something, but not contact -- and that is worth reporting
    just as clearly.

    The marginal beside it used to be a bar of P(j), which encodes distance from
    zero; the question is distance from UNIFORM, so it is drawn as a deviation from
    the 1/J rule with the connector carrying the sign.
    """
    rows, marg = [], []
    for e in eps:
        j = e.get("j_star")
        if j is None or "onset" not in e:
            continue
        marg.append(np.asarray(j))
        for t in np.flatnonzero(e["onset"] > 0.5):
            if t - window < 0 or t + window + 1 > len(j):
                continue
            rows.append(np.asarray(j)[t - window: t + window + 1])
    if not rows:
        return skip("no episode has both j_star and an interior onset",
                    "a J>1 checkpoint analysed by analysis/predictive_jaccard.py")
    rows = np.array(rows)
    lags = np.arange(-window, window + 1)
    # NOT `P` -- that is the panels module in this file's namespace, and shadowing it
    # here silently broke every design-system call inside this function.
    pmat = np.zeros((n_heads, len(lags)))
    for j in range(n_heads):
        pmat[j] = (rows == j).mean(axis=0)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                 gridspec_kw={"width_ratios": [2.1, 1], "wspace": 0.30},
                                 sharey=True)
    im = a0.imshow(pmat, aspect="auto", cmap=P.SEQ, origin="lower",
                   extent=[lags[0] - 0.5, lags[-1] + 0.5, -0.5, n_heads - 0.5],
                   vmin=0, vmax=max(pmat.max(), 1.0 / n_heads * 1.5))
    a0.axvline(0, color=P.CONTACT, lw=2.0)
    a0.annotate("contact onset", (0, n_heads - 0.55), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=P.CONTACT)
    a0.set(xlabel="frames relative to contact onset", ylabel="head $j$",
           yticks=range(n_heads),
           title=f"P($j^*$ = j | lag)  ({len(rows)} onset events)")
    fig.colorbar(im, ax=a0, label="probability", fraction=0.046)

    allj = np.concatenate(marg) if marg else np.array([])
    p = np.array([(allj == j).mean() for j in range(n_heads)])
    unif = 1.0 / n_heads
    lo, hi = P.symmetric_limits(p - unif)
    cmap = plt.get_cmap(P.DIV)
    a1.axvline(unif, color="0.35", lw=1.2, zorder=1)
    for j in range(n_heads):
        c = cmap(0.5 + 0.5 * (p[j] - unif) / (hi if hi else 1.0))
        a1.plot([unif, p[j]], [j, j], color=c, lw=2.4, solid_capstyle="round", zorder=2)
        a1.scatter([p[j]], [j], s=64, color=c, edgecolor="white", lw=1.5, zorder=3)
        a1.annotate(f"{p[j] - unif:+.3f}", (p[j], j), xytext=(8, 0),
                    textcoords="offset points", fontsize=7.4, va="center", color="0.3")
    a1.annotate(f"uniform 1/{n_heads}", (unif, n_heads - 0.55), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color="0.35")
    a1.set(xlabel="P($j^*$ = j) over all frames",
           title=f"Marginal head usage vs uniform\n(n={allj.size:,} frames)")
    a1.grid(axis="x", alpha=0.35)
    a1.spines[["top", "right", "left"]].set_visible(False)
    a1.margins(x=0.22)
    for a in (a0, a1):
        a.title.set_fontsize(9)
    return save(fig, out, "28_head_onset_alignment")



def fig_per_head_dynamics(eps, out, n_heads=4, n_boot=400, seed=0):
    """Do the heads own distinguishable dynamics regimes?

    Mean per-frame agent and block displacement conditioned on the winning head,
    with a bootstrap CI over frames and the frame count per head printed. If the
    intervals all overlap, the heads are a partition of nothing in particular and
    the Step 4 story has to rest on the loss, not on interpretability.

    Dots rather than bars: the quantity is a mean displacement whose zero is not
    meaningful, so a bar encodes distance-from-zero -- information the reader does
    not want -- and buries the interval that decides whether the heads differ.
    """
    js, ag, bl = [], [], []
    for e in eps:
        j = e.get("j_star")
        if j is None:
            continue
        n = len(j)
        a = e.get("agent_disp")
        b = e.get("block_disp")
        if a is None or b is None:
            continue
        js.append(np.asarray(j)[:n])
        ag.append(np.asarray(a)[:n])
        bl.append(np.asarray(b)[:n])
    if not js:
        return skip("no episode has j_star with agent_disp/block_disp",
                    "a J>1 checkpoint analysed by analysis/predictive_jaccard.py")
    j, a, b = np.concatenate(js), np.concatenate(ag), np.concatenate(bl)
    rng = np.random.default_rng(seed)
    groups = {}
    for v, lab in ((a, r"$|\Delta$ agent$|$"), (b, r"$|\Delta$ block$|$")):
        cell = {}
        for h in range(n_heads):
            m = j == h
            if m.sum() < 2:
                continue
            vals = v[m]
            boots = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1)
            cell[h] = (float(vals.mean()), float(np.percentile(boots, 2.5)),
                       float(np.percentile(boots, 97.5)), int(m.sum()))
        if cell:
            groups[lab] = cell
    fig = P.dot_ci(groups, xlabel=r"winning head $j^*$", ylabel="mean displacement",
                   title=f"Dynamics by winning head  ({j.size:,} frames)",
                   subtitle="dot = mean, bar = 95% bootstrap CI over frames -- overlapping "
                            "intervals mean the heads partition nothing")
    if P.is_no_data(fig):
        return skip("no head has >=2 frames", "an episode set where every head wins somewhere")
    return save(fig, out, "29_per_head_dynamics")


# --- campaign-level statistics --------------------------------------------------

def fig_ladder(ladder, out, ylabel="CEM success rate"):
    """The paper's Fig 1b shape, over whatever cells we actually ran.

    Success against feature dimension, one line per predictor, sparse solid and
    dense dashed. The paper's headline is that sparse beats dense across predictors
    and dimensions; this panel says whether the same ordering shows up here, which
    is the sanity check that our controls reproduce the upstream result before any
    Pi-WM claim is made on top of them.
    """
    if not ladder:
        return skip("no ladder cells", 'campaign.json "ladder": {"<predictor>": '
                                      '{"sparse"/"dense": {"<D>": [per-seed, ...]}}}')
    preds = sorted(ladder)
    pal = arm_palette(preds)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    n_cells = 0
    for pred in preds:
        for kind, ls, mk in (("sparse", "-", "o"), ("dense", "--", "s")):
            cells = ladder[pred].get(kind) or {}
            if not cells:
                continue
            ds = sorted(float(d) for d in cells)
            m, e, ns = [], [], []
            for d in ds:
                key = next(k for k in cells if float(k) == d)
                v = np.asarray(cells[key], float).ravel()
                m.append(float(np.nanmean(v)))
                e.append(float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0)
                ns.append(v.size)
                n_cells += 1
            ax.errorbar(ds, m, yerr=e, ls=ls, marker=mk, ms=6, lw=1.9, capsize=3,
                        color=pal[pred], label=f"{pred} {kind}")
            ax.annotate(f"n={min(ns)}-{max(ns)}", (ds[-1], m[-1]), xytext=(5, -2),
                        textcoords="offset points", fontsize=7, color=pal[pred])
    ax.set(xscale="log", xlabel="feature dimension $D$", ylabel=ylabel,
           title=f"Success ladder: solid = sparse, dashed = dense ({n_cells} cells)")
    ax.set_xticks(sorted({float(d) for p in ladder.values()
                          for k in p.values() for d in k}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    return save(fig, out, "30_ladder")


def fig_effect_sizes(arms, out, groups=None):
    """Every contrast as a paired effect, with the raw seeds that produced it.

    The estimation plot rather than a forest of summary rows: at n=3 the interval is
    doing almost all of the work, and a reader who cannot see the three seeds has no
    way to tell a real shift from one seed dragging the mean. The bootstrap
    distribution beside each point estimate is the honest width.

    Status -- resolved vs inside the detection floor -- rides on marker FILL, so an
    arm keeps its colour and the verdict still reads at a glance.
    """
    gates = group_arms(arms, groups)
    contrasts = []
    for members in gates.values():
        for arm in members[1:]:
            if paired_effect(arms, members[0], arm)["n"]:
                contrasts.append((members[0], arm))
    if not contrasts:
        return skip("no contrast has a shared seed",
                    'campaign.json "arms" with a control and a variant sharing a seed')
    sds = [paired_effect(arms, c, v)["sd"] for c, v in contrasts]
    sds = [x for x in sds if np.isfinite(x)]
    ns = [paired_effect(arms, c, v)["n"] for c, v in contrasts]
    m_obs = mde(float(np.median(sds)), int(np.median(ns))) if sds else np.nan
    seeds = {a: np.array([float(v) for v in arms[a].values()], float) for a in arms}
    fig = P.estimation_plot(seeds, contrasts,
                            mde=m_obs if np.isfinite(m_obs) else None,
                            title="Every contrast: the seeds, and the paired difference",
                            subtitle="squares / diamonds = flags-off control  ·  each dot = one "
                                     "seed  ·  hollow = inside the detection floor, i.e. "
                                     "underpowered rather than null")
    if P.is_no_data(fig):
        return skip("no contrast survived", "campaign.json arms with matched seeds")
    return save(fig, out, "31_effect_sizes")



def fig_gate_values(gate_values, out):
    """Did the Step 3 normalisation change scale as well as shape?

    A bare softmax over r=16 modes has mean 1/16 against sigmoid's ~0.5, an 8x
    shrink of the gated gradient path that at 2 epochs is indistinguishable from
    'support gating is worse'. r*softmax is supposed to fix that by construction.
    The mean +- sd bar this used to carry cannot tell "centred on 1.0" from "bimodal
    with no mass at 1.0" -- which is the failure mode the fix could actually have --
    so the panel shows the distribution and puts the mean on it as a tick.
    """
    if not gate_values:
        return skip("no gate value samples", 'campaign.json "gate_values": '
                                            '{"<arm>": [values, ...]}')
    fig = P.arm_ridgeline({a: np.asarray(v, float).ravel() for a, v in sorted(gate_values.items())},
                          xlabel="gate value $g$",
                          refs=[(0.5, r"sigmoid $\approx$ 0.5"), (1.0, r"$r\cdot$softmax target 1.0")],
                          title="Gate value distribution by arm",
                          subtitle="the r*softmax check: the softmax arms must put MASS near 1.0, "
                                   "not merely a mean there")
    if P.is_no_data(fig):
        return skip("no arm has finite gate samples", "campaign.json gate_values with numbers")
    return save(fig, out, "32_gate_values")


def fig_gate_heatmap(gate_heatmap, out):
    """Does support gating sharpen mode selection, or just rescale it?

    The gate over (timestep x r modes) for one episode, shared colour scale across
    arms so the comparison is fair. Sharpening looks like a few bright columns that
    persist; rescaling looks like the same texture at a different brightness.
    """
    if not gate_heatmap:
        return skip("no gate heatmaps", 'campaign.json "gate_heatmap": '
                                       '{"<arm>": [[t x r]]}')
    arms = sorted(gate_heatmap)
    mats = {}
    for arm in arms:
        m = np.asarray(gate_heatmap[arm], float)
        if m.ndim == 2 and m.size:
            mats[arm] = m
    if not mats:
        return skip("gate heatmaps were not 2-D",
                    'campaign.json "gate_heatmap" values shaped (timesteps, r)')
    vmax = max(np.nanmax(m) for m in mats.values())
    fig, axes = plt.subplots(1, len(mats), figsize=(3.5 * len(mats) + 1.2, 3.9),
                             squeeze=False)
    for ax, (arm, m) in zip(axes[0], mats.items()):
        im = ax.imshow(m.T, aspect="auto", cmap=P.SEQ, origin="lower",
                       vmin=0, vmax=vmax)
        ax.set(xlabel="timestep", ylabel="gate mode $r$",
               title=f"{arm}\nmean {m.mean():.3f}, max {m.max():.3f}")
        ax.title.set_fontsize(8.5)
    fig.colorbar(im, ax=axes[0][-1], label="gate value", fraction=0.046)
    fig.suptitle("Gate over (timestep x r modes), shared colour scale", y=1.04,
                 fontsize=10)
    return save(fig, out, "33_gate_heatmap")


def fig_power_curve(arms, out, groups=None, seeds_grid=(3, 5, 10)):
    """What effect can n=3 seeds actually detect -- i.e. is a null result a result?

    The gates are pre-registered at n=3 matched seeds, so before reading any of them
    it is worth knowing the floor. Simulated power against true effect size at the
    campaign's own observed across-seed sd, with the 80%-power minimum detectable
    effect marked for each n, and the observed effects laid on the same axis as
    ticks so "we did not detect it" and "we could not have detected it" are
    distinguishable.

    The observed effects used to be drawn here as |effect| bars coloured green/grey.
    That destroyed the sign, hid the seeds, and spent a status colour on a boolean
    the threshold line already carried; they now live in 31_effect_sizes, which
    shows the seeds. This panel keeps only the question the bars could not answer.
    """
    gates = group_arms(arms, groups)
    effs = []
    for members in gates.values():
        for arm in members[1:]:
            e = paired_effect(arms, members[0], arm)
            if e["n"] > 1 and np.isfinite(e["sd"]):
                effs.append((arm, e))
    if not effs:
        return skip("no contrast has >=2 shared seeds to estimate a seed sd",
                    'campaign.json "arms" with a control and a variant sharing '
                    ">=2 seeds")
    sd = float(np.median([e["sd"] for _, e in effs]))
    n_obs = int(np.median([e["n"] for _, e in effs]))
    m_obs = mde(sd, n_obs)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    grid = np.linspace(0, 5 * sd, 60)
    for n, color in zip(seeds_grid, P.epoch_colors(len(seeds_grid))):
        ax.plot(grid, [paired_power(g, sd, n) for g in grid], color=color, lw=2.2,
                zorder=3)
        m = mde(sd, n)
        if np.isfinite(m):
            ax.plot([m, m], [0, 0.8], color=color, ls=":", lw=1.2, zorder=2)
            ax.annotate(f"n={n}\nMDE {m:.3f}", (m, 0.815), xytext=(3, 0),
                        textcoords="offset points", fontsize=7.6, color=color, va="bottom")
    ax.axhline(0.8, color="0.35", lw=1.0, zorder=1)
    ax.annotate("80% power", (0.015, 0.8), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=7.6, color="0.35")
    # the observed effects on the same axis, as rug ticks: this is what turns the
    # curve from a statistics lesson into a statement about THIS campaign
    # An effect can be far larger than 5*sd, and a rug tick drawn there with
    # clip_on=False drags the saved bbox out with it -- which is how this panel
    # rendered 4700px wide. Clamp to the axis and flag the clamped ones instead.
    hi_x, n_off = grid[-1], 0
    for arm, e in effs:
        x = abs(e["mean"])
        if not np.isfinite(x):
            continue
        n_off += x > hi_x
        ax.plot([min(x, hi_x)] * 2, [0.006, 0.042], color=P.arm_color(arm), lw=2.4,
                solid_capstyle="butt", zorder=4)
    below = sum(1 for _, e in effs if np.isfinite(m_obs) and abs(e["mean"]) < m_obs)
    note = (f"ticks = |observed effect| (n={len(effs)}); {below} fall below the "
            f"n={n_obs} MDE")
    if n_off:
        note += f"; {n_off} exceed the axis and are drawn at its edge"
    ax.annotate(note, (0.012, 0.055), xycoords="axes fraction", fontsize=7.6,
                color="0.35", va="bottom")
    ax.set(xlabel="true paired effect", ylabel=r"power at $\alpha$=0.05",
           ylim=(0, 1.02), xlim=(0, grid[-1]))
    ax.set_title(f"Paired t-test power at the observed seed sd = {sd:.3f}\n"
                 f"a paired sign-flip test at n={n_obs} cannot return p below "
                 f"{sign_flip_min_p(n_obs):.2f} at all -- it has ZERO power here",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, out, "34_power_curve")


# --- summary --------------------------------------------------------------------

CONTACT_SHEET = ("31_effect_sizes", "05_gate_scorecard", "34_power_curve",
                 "01_peri_event", "17_rdmreg_vs_l0", "32_gate_values",
                 "16_sparsity_trajectories", "04_head_usage", "21_preemption_timeline")


def fig_contact_sheet(paths, out, names=CONTACT_SHEET, ncol=3):
    """One page holding the load-bearing panels, for the first look.

    Re-reads the PNGs already written rather than re-plotting, so the sheet can
    never disagree with the individual panels. Panels that were skipped appear as
    a labelled blank, which keeps the absence visible instead of silently
    reflowing the grid.
    """
    have = {Path(p).stem: Path(p) for p in paths if p}
    picks = [n for n in names if n in have]
    if not picks:
        return skip("none of the contact-sheet panels rendered",
                    "any of: " + ", ".join(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.9 * nrow))
    for ax, name in zip(np.ravel(axes), names):
        ax.set_axis_off()
        if name in have:
            ax.imshow(plt.imread(have[name]))
            ax.set_title(name, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"{name}\nskipped", ha="center", va="center",
                    fontsize=10, color="grey", transform=ax.transAxes)
    for ax in np.ravel(axes)[len(names):]:
        ax.set_axis_off()
    fig.suptitle(f"LpWM / Pi-WM contact sheet -- {len(picks)}/{len(names)} "
                 "load-bearing panels rendered", y=1.0, fontsize=12)
    fig.tight_layout()
    return save(fig, out, "99_contact_sheet")


# --- selftest -------------------------------------------------------------------

SYNTH_RUNS = (
    ("LpWM-base_pd384_bf16_s0", 0.50, 1, 0.0, None),
    ("LpWM-base_pd384_bf16_s1", 0.50, 1, 0.0, None),
    ("LpWM-base_pd384_bf16_s2", 0.50, 1, 0.0, None),
    ("PiWM-sparse-2pct_pd384_bf16_s0", 0.02, 1, 0.0, 8),
    ("LpWM-ltv_pd384_bf16_s0", 0.50, 1, 0.0, None),
    ("PiWM-gate-sup-softmax_pd384_bf16_s0", 0.50, 1, 0.0, None),
    ("PiWM-union4_pd384_bf16_s0", 0.50, 4, 0.0, None),
    ("PiWM-union4-entropy_pd384_bf16_s0", 0.50, 4, 0.1, None),
    ("PiWM-union4-entropy_pd384_bf16_s1", 0.50, 4, 0.1, None),
)


def _synth_history(d, rho, n_heads, lam_ent, kwta_k, rng, n_rows=90, n_batches=30965):
    """One synthetic export with the real column names train.py emits.

    Deliberately written with SECTIONED keys ('sparsity/train_l0_frac'), all rows
    inside epoch 1, sparse diagnostic columns, and a wall-clock hole -- i.e. the
    shape a real partially-trained campaign run has, which is what the panels have
    to survive.
    """
    d.mkdir(parents=True, exist_ok=True)
    step = np.arange(n_rows) * 50
    frac = step / n_batches
    t0 = 1.788e9
    # a 4h eviction plus queue wait, so the timeline and resume panels have a gap
    ts = t0 + step * 1.6 + np.where(step > step[n_rows // 2], 5400.0, 0.0)
    cols = {
        "_step": step.astype(float),
        "_runtime": ts - t0,
        "_timestamp": ts,
        "epoch": np.ones(n_rows),
        "progress/epoch_frac": frac,
        "train/loss": 0.6 * np.exp(-step / 4000.0) + 0.05 + 0.01 * rng.normal(size=n_rows),
        "train/z_loss": 0.3 * np.exp(-step / 3000.0) + 0.02,
        "train/reg_loss": (0.5 * np.exp(-step / 2500.0)
                           + (0.22 if kwta_k else 0.03)),
        "train/z_visual_loss": 0.3 * np.exp(-step / 3000.0) + 0.02,
        "sparsity/train_l0_frac": rho + 0.02 * rng.normal(size=n_rows),
        "opt/grad_norm": 0.2 * np.exp(-step / 6000.0) + 0.01 * rng.random(n_rows),
        "opt/lr": np.full(n_rows, 5e-4),
        "perf/batches_per_sec": 2.2 + 0.4 * rng.normal(size=n_rows),
        "perf/hours_per_epoch": n_batches / (2.2 * 3600.0) + 0.3 * rng.random(n_rows),
    }
    # diagnostics land only every diag_every_x_batch, so most cells are blank
    sparse = np.full(n_rows, np.nan)
    sparse[::13] = 10.0 + rng.random(len(sparse[::13]))
    cols["sparsity/l0_std_across_samples"] = sparse
    diag = np.full(n_rows, np.nan)
    diag[::7] = 0.01 * rng.random(len(diag[::7]))
    cols["diag/train_diag_cov_loss"] = diag
    cols["diag/train_diag_var_perdim"] = diag * 8
    if n_heads > 1:
        share = 0.30 + (0.60 if lam_ent == 0 else 0.0)
        ps = [share if j == 0 else (1 - share) / (n_heads - 1) for j in range(n_heads)]
        for j, p in enumerate(ps):
            cols[f"heads/train_head_usage_p{j}"] = p + 0.03 * rng.normal(size=n_rows)
        p = np.array(ps)
        cols["heads/train_head_usage_entropy"] = np.full(
            n_rows, float(-(p * np.log(p + 1e-9)).sum())) + 0.02 * rng.random(n_rows)
        cols["heads/train_head_usage_max"] = np.full(n_rows, max(ps))
        cols["heads/train_head_switch_rate"] = 0.4 + 0.1 * rng.random(n_rows)
        cols["heads/train_head_burst_rate"] = 0.08 + 0.05 * rng.random(n_rows)
    if not d.name.startswith("s2_"):  # the ltv-predictor arms
        cols["train/ltv_correction_norm"] = 0.4 + 0.1 * rng.random(n_rows)
    with open(d / "wandb_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n_rows):
            w.writerow(["" if not np.isfinite(cols[k][i]) else cols[k][i]
                        for k in cols])

    arm = run_arm(d.name)
    (d / "wandb_meta.json").write_text(json.dumps({
        "id": f"synth{d.name}", "name": f"{arm}/s{run_seed(d.name)}", "group": arm,
        "state": "running", "seed": run_seed(d.name), "n_rows": n_rows,
        "batches_per_epoch": n_batches,
        "config": {"embed_dim": 384, "precision": "bf16", "kwta_k": kwta_k,
                   "n_heads": n_heads, "head_entropy_coef": lam_ent,
                   "gate_input": "support" if "sup_" in d.name else "magnitude"},
    }))
    mid = float(frac[n_rows // 2 + 1])
    (d / "resume_steps.json").write_text(json.dumps({
        "epoch": [1.0], "epoch_frac": [mid], "step": [float(step[n_rows // 2 + 1])],
        "markers": [{"epoch": 1, "batch": int(step[n_rows // 2 + 1]),
                     "epoch_frac": mid, "source": "train.log"}],
    }))
    # exported histograms: bimodal for the k-WTA arm so the panel has something to
    # separate from the unimodal control
    hists = []
    for i, s in enumerate(step[::30]):
        centre = rho * 384
        vals = rng.normal(centre, 12 + 3 * i, 4000)
        if kwta_k:
            vals = np.concatenate([rng.normal(centre * 0.4, 5, 2000),
                                   rng.normal(centre * 1.8, 5, 2000)])
        c, e = np.histogram(vals, bins=48)
        hists.append({"step": float(s), "edges": e.tolist(),
                      "counts": c.astype(float).tolist()})
    (d / "wandb_hists.json").write_text(json.dumps(
        {"dist/z_l0_per_sample": hists}))


def _synth(tmp):
    """Synthetic inputs with the real schema, so every panel is exercised before
    any run exists. Numbers are fabricated; only the plotting code is tested."""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_ep, T, D, J = 6, 90, 48, 4
    npz = {}
    for i in range(n_ep):
        onset = np.zeros(T)
        onset[rng.choice(np.arange(8, T - 8), 4, replace=False)] = 1.0
        kern = np.exp(-((np.arange(-5, 6)) ** 2) / 4.0)
        sm = 0.3 + 0.5 * np.convolve(onset, kern, "same") + 0.03 * rng.normal(size=T)
        sw = 0.3 + 0.5 * np.convolve(np.roll(onset, 2), kern, "same") + 0.03 * rng.normal(size=T)
        # heavy-tailed unit usage plus a block of dead units, so the code-geometry
        # panel is exercised on something with real structure
        p = 0.6 / (1.0 + np.arange(D)) ** 0.7
        p[-8:] = 0.0
        z = np.abs(rng.normal(size=(T, D))) * (rng.random((T, D)) < p)
        j_star = rng.integers(0, J, T)
        j_star[np.convolve(onset, np.ones(3), "same") > 0.5] = 1  # head 1 owns contact
        npz.update({
            f"S_world_{i}": sw, f"S_model_{i}": sm, f"onset_{i}": onset,
            f"block_disp_{i}": np.abs(np.roll(onset, 1)) * 3 + 0.1 * rng.random(T),
            f"agent_disp_{i}": rng.random(T), f"z_{i}": z,
            f"states_{i}": rng.normal(size=(T, 5)) * 50,
            f"j_star_{i}": j_star,
        })
    np.savez_compressed(tmp / "analysis_step1.npz", n_episodes=n_ep, **npz)
    (tmp / "analysis_step1.json").write_text(json.dumps({"n_episodes": n_ep}))

    # arm names match scripts/run_campaign.sh so the overview panel groups them
    campaign = {
        "arms": {
            "LpWM-base": {"0": 0.42, "1": 0.38, "2": 0.45},
            "PiWM-sparse-matched": {"0": 0.40, "1": 0.37, "2": 0.44},
            "PiWM-sparse-2pct": {"0": 0.29, "1": 0.26, "2": 0.33},
            "LpWM-ltv": {"0": 0.36, "1": 0.33, "2": 0.39},
            "PiWM-gate-sup-sigmoid": {"0": 0.34, "1": 0.30, "2": 0.36},
            "PiWM-gate-mag-softmax": {"0": 0.35, "1": 0.32, "2": 0.38},
            "PiWM-gate-sup-softmax": {"0": 0.30, "1": 0.27, "2": 0.33},
            "PiWM-union4": {"0": 0.36, "1": 0.34, "2": 0.40},
            "PiWM-union4-entropy": {"0": 0.39, "1": 0.36, "2": 0.43},
        },
        "gates": [
            {"name": "Step 1: AUROC(S_model) > AUROC(S_world)", "observed": 0.11,
             "lo": 0.04, "hi": 0.18, "threshold": 0.0, "direction": "above"},
            {"name": "Step 2: k-WTA within one seed-std", "observed": -0.02,
             "lo": -0.05, "hi": 0.01, "threshold": -0.04, "direction": "above"},
            {"name": "Step 3: support gate >= magnitude gate", "observed": -0.06,
             "lo": -0.10, "hi": -0.02, "threshold": 0.0, "direction": "above"},
            {"name": "Step 4: J=4 with entropy > J=1", "observed": 0.03,
             "lo": -0.01, "hi": 0.07, "threshold": 0.0, "direction": "above"},
        ],
        "k_sweep": {"0.02": {"success": 0.29, "rdmreg": 0.31},
                    "0.10": {"success": 0.36, "rdmreg": 0.28},
                    "0.5": {"success": 0.41, "rdmreg": 0.26}},
        "scale": {"magnitude": {"c": [0.5, 0.75, 1.0, 1.5, 2.0],
                                "rel_change": [0.38, 0.16, 0.0, 0.21, 0.47]},
                  "support": {"c": [0.5, 0.75, 1.0, 1.5, 2.0],
                              "rel_change": [0.02, 0.01, 0.0, 0.01, 0.02]}},
        "ladder": {
            "mlp_var": {"sparse": {"48": [0.21, 0.19], "96": [0.30, 0.28],
                                   "192": [0.38, 0.35], "384": [0.42, 0.38, 0.45]},
                        "dense": {"48": [0.15, 0.14], "96": [0.22, 0.20],
                                  "192": [0.28, 0.27], "384": [0.33, 0.31]}},
            "ltv": {"sparse": {"192": [0.31, 0.29], "384": [0.36, 0.33, 0.39]},
                    "dense": {"192": [0.25, 0.24], "384": [0.29, 0.27]}},
        },
        "gate_values": {
            "PiWM-gate-mag-softmax": list(np.clip(rng.normal(1.0, 0.55, 4000), 0, None)),
            "PiWM-gate-sup-sigmoid": list(1 / (1 + np.exp(-rng.normal(0.0, 1.4, 4000)))),
            "PiWM-gate-sup-softmax": list(np.clip(rng.normal(1.0, 0.9, 4000), 0, None)),
        },
        "gate_heatmap": {
            "PiWM-gate-mag-softmax": (0.5 + 0.4 * rng.random((24, 16))).tolist(),
            "PiWM-gate-sup-softmax": (np.eye(16)[np.arange(24) % 16] * 2.6
                               + 0.25 * rng.random((24, 16))).tolist(),
        },
    }
    (tmp / "campaign.json").write_text(json.dumps(campaign))

    for name, rho, nh, lam, k in SYNTH_RUNS:
        _synth_history(tmp / "runs" / name, rho, nh, lam, k, rng)
    return tmp


# --- driver ---------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1")
    ap.add_argument("--campaign")
    ap.add_argument("--runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--episode", type=int, default=0,
                    help="episode index for the single-episode panels")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest_dir", default="/tmp/pi_wm_selftest")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any panel was skipped")
    ap.add_argument("--no_contact_sheet", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        t = _synth(args.selftest_dir)
        args.step1 = args.step1 or str(t / "analysis_step1.json")
        args.campaign = args.campaign or str(t / "campaign.json")
        args.runs = args.runs or str(t / "runs" / "*")

    made, skipped = [], []

    def run(fn, name, *a, **kw):
        _LAST_SKIP.clear()
        try:
            p = fn(*a, **kw)
        except Exception as e:  # a broken panel must not take the whole suite down
            plt.close("all")
            skipped.append((name, f"raised {type(e).__name__}: {e}",
                            "a bug in this panel -- please report"))
            return
        if p:
            made.append(p)
        else:
            reason, unblock = take_skip()
            skipped.append((name, reason, unblock))

    if args.step1:
        _, eps = load_step1(args.step1)
        print(f"step 1 figures ({len(eps)} episodes):")
        run(fig_peri_event, "peri_event", eps, args.out)
        run(fig_roc_overlay, "roc_overlay", eps, args.out)
        run(fig_jaccard_decomposition, "jaccard_decomposition", eps, args.out)
        run(fig_support_selfsim, "support_selfsim", eps, args.out, args.episode)
        run(fig_head_raster, "head_raster", eps, args.out, args.episode)
        run(fig_burst_vs_error, "burst_vs_error", eps, args.out)
        run(fig_code_geometry, "code_geometry", eps, args.out)
        run(fig_onset_lead_lag, "onset_lead_lag", eps, args.out)
        run(fig_head_onset_alignment, "head_onset_alignment", eps, args.out,
            WINDOW, args.n_heads)
        run(fig_per_head_dynamics, "per_head_dynamics", eps, args.out, args.n_heads)
    else:
        skipped.append(("all 10 step-1 panels", "--step1 not given",
                        "python analysis/predictive_jaccard.py --run_dir "
                        "runs/outputs/<run>, then pass its analysis_step1.json"))

    if args.campaign:
        c = json.loads(Path(args.campaign).read_text())
        print("campaign figures:")
        run(fig_campaign_overview, "campaign_overview", c.get("arms") or {},
            args.out, c.get("groups"))
        run(fig_paired_dumbbell, "paired_dumbbell", c.get("arms") or {}, args.out)
        run(fig_gate_scorecard, "gate_scorecard", c.get("gates") or [], args.out)
        run(fig_success_vs_k, "success_vs_k", c.get("k_sweep") or {}, args.out)
        run(fig_scale_perturbation, "scale_perturbation", c.get("scale") or {},
            args.out)
        run(fig_ladder, "ladder", c.get("ladder") or {}, args.out)
        run(fig_effect_sizes, "effect_sizes", c.get("arms") or {}, args.out,
            c.get("groups"))
        run(fig_gate_values, "gate_values", c.get("gate_values") or {}, args.out)
        run(fig_gate_heatmap, "gate_heatmap", c.get("gate_heatmap") or {}, args.out)
        run(fig_power_curve, "power_curve", c.get("arms") or {}, args.out,
            c.get("groups"))
    else:
        skipped.append(("all 10 campaign panels", "--campaign not given",
                        "a campaign.json with at least an \"arms\" map "
                        "(see this module's docstring for the schema)"))

    if args.runs:
        runs = load_runs(args.runs)
        print(f"run figures ({len(runs)} run dirs with an export):")
        run(fig_metric_coverage, "metric_coverage", runs, args.out)
        run(fig_head_usage, "head_usage", runs, args.out, args.n_heads)
        run(fig_engagement, "engagement", runs, args.out)
        run(fig_training_curves, "training_curves", runs, args.out)
        run(fig_training_health, "training_health", runs, args.out)
        run(fig_sparsity_trajectories, "sparsity_trajectories", runs, args.out)
        run(fig_rdmreg_vs_l0, "rdmreg_vs_l0", runs, args.out)
        run(fig_loss_decomposition, "loss_decomposition", runs, args.out)
        run(fig_gradient_health, "gradient_health", runs, args.out)
        run(fig_throughput, "throughput", runs, args.out)
        run(fig_preemption_timeline, "preemption_timeline", runs, args.out)
        run(fig_seed_variance, "seed_variance", runs, args.out)
        run(fig_head_specialisation, "head_specialisation", runs, args.out,
            args.n_heads)
        run(fig_head_switch_burst, "head_switch_burst", runs, args.out)
        run(fig_l0_distribution, "l0_distribution", runs, args.out)
    else:
        skipped.append(("all 15 run panels", "--runs not given",
                        "python analysis/export_wandb.py, then "
                        "--runs 'runs/outputs/*'"))

    if not args.no_contact_sheet:
        run(fig_contact_sheet, "contact_sheet", made, args.out)

    print(f"\n{len(made)} figures written to {args.out}")
    if skipped:
        w = max(len(n) for n, _, _ in skipped)
        print(f"\n{len(skipped)} SKIPPED -- nothing was faked; each line says what "
              "would unblock it:")
        for name, reason, unblock in skipped:
            print(f"  {name:<{w}}  why: {reason}")
            print(f"  {'':<{w}}  fix: {unblock}")
    if args.strict and skipped:
        print(f"\n--strict: {len(skipped)} panel(s) skipped, exiting non-zero")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
