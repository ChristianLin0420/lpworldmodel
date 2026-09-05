"""Round 7 RESULTS: the dimension-matched grid, and the contrasts it makes readable.

`analysis/round7_figs.py` is the round-7 v1 module and predates the v2 redesign -- it knows
`LpWM-ltv`, `LpWM-ltv-d2048`, `PiWM-columns` and `PiWM-drop95`, and none of the five aligned-grid
arms. This module draws what the grid actually measures.

WHY THE GRID EXISTS
    The campaign's patch-vs-cls contrast was `PiWM-columns` (patch_size 14 -> 256 tokens) against
    `LpWM-ltv` (cls -> 1 token). At proj_dim 384 those carry 98304 and 384 latent dimensions, so
    the contrast confounds THE FEATURE with 256x THE CAPACITY. Both the CEM result (+0.072) and
    the orientation probe (9.58 deg vs 15.72 deg) inherit that confound.

    The grid separates them. `patch_size` is the only token knob and `proj_dim` the only width
    knob, and `num_patches = (img_size // patch_size) ** 2` at img_size 224:

        patch_size 224 ->   1 token  x 384 =   384     patch_size 112 ->  4 x 384 = 1536
        patch_size  56 ->  16 tokens x 384 =  6144     patch_size  14 -> 256 x 384 = 98304

    so each ROW is a cls-vs-patch contrast at EQUAL total latent, and each COLUMN is a capacity
    ladder at FIXED feature:

                   384          1536          6144
        cls        LpWM-ltv     -d1536        -d6144
        patch      -ltv-p1      cols-p4       cols-p16

    A row difference is the feature. A column difference is the capacity. The old contrast is
    neither, and is drawn greyed for reference rather than dropped.

NUMBERS ARE READ, NEVER QUOTED
    Every mean, interval and n comes from `collect_evals.collect()` at render time. The round-5
    headline figure once carried a hardcoded GROUPS literal and could not see new evaluations;
    that is the failure this module is written not to repeat.

GUARDS CARRIED FORWARD
    * arm names resolved with `resolve_arm`, because every patch arm is keyed `<name>_patch` and
      the natural `A[name]` spelling has silently reported n=0 with data on disk three times;
    * the n = 8 floor -- a contrast below it is drawn as PENDING, never as a point (T1 read
      -0.233 [-0.39, -0.08] at n = 3 and settled at a null by n = 8);
    * the degenerate-interval guard, compared scale-free with `np.ptp` rather than `se == 0`,
      because np.std on identical floats returns 6.8e-17 and an exact test misses its own case.

    python analysis/round7_results_figs.py --out diary/assets/2026-09-05
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.collect_evals import collect, resolve_arm, ArmNameError  # noqa: E402
from analysis.style import (C, INK, MUTED, use_style, ax_style, panel_title,  # noqa: E402
                            suptitle, fit)

FLOOR = 8
IMG = 224
PROJ_D = 384


def tokens_for(patch_size):
    """models/vit_encoder.py:67, verbatim: grid = img // patch, num_patches = grid ** 2."""
    return (IMG // patch_size) ** 2


#: (arm, feature, patch_size or None, proj_dim). latent = tokens * proj_dim, computed below --
#: never written down, so a figure cannot disagree with the arithmetic it claims.
GRID = [
    ("LpWM-ltv",        "cls",   None, 384),
    ("LpWM-ltv-d1536",  "cls",   None, 1536),
    ("LpWM-ltv-d6144",  "cls",   None, 6144),
    ("LpWM-ltv-p1",     "patch", 224,  384),
    ("PiWM-cols-p4",    "patch", 112,  384),
    ("PiWM-cols-p16",   "patch", 56,   384),
]
#: The confounded original, kept visible rather than quietly dropped.
LEGACY = ("PiWM-columns", "patch", 14, 384)


def latent_dim(feature, patch_size, proj_dim):
    return (tokens_for(patch_size) if feature == "patch" else 1) * proj_dim


def _arm(A, name):
    try:
        return A[resolve_arm(A, name)]
    except ArmNameError:
        return None


def _mean_ci(vals):
    """Mean and t-interval of an arm's per-seed success rates."""
    d = np.asarray(sorted(float(v) for v in vals), dtype=float)
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else np.nan), np.nan, np.nan, n
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return float(d.mean()), float(d.mean() - t * se), float(d.mean() + t * se), n


def _paired(A, ctrl, arm):
    """Paired contrast, with the degenerate-interval guard. Returns None if an arm is unknown."""
    X, Y = _arm(A, ctrl), _arm(A, arm)
    if X is None or Y is None:
        return None
    s = sorted(set(X) & set(Y), key=int)
    if not s:
        return {"n": 0}
    d = np.array([float(Y[k]) - float(X[k]) for k in s])
    n = len(d)
    if n < 2:
        return {"n": n}
    se = d.std(ddof=1) / np.sqrt(n)
    degenerate = bool(np.ptp(d) <= 1e-9 * max(1.0, abs(float(d.mean()))))
    t = stats.t.ppf(0.975, n - 1)
    lo, hi = ((d.mean(), d.mean()) if degenerate
              else (d.mean() - t * se, d.mean() + t * se))
    return {"n": n, "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "degenerate": degenerate}


# --- panel A: the grid ---------------------------------------------------------------------
def fig_grid(out):
    """Each cell an arm mean; each ROW a feature contrast at equal latent, each COLUMN a ladder."""
    use_style()
    A = collect()[0]

    dims = sorted({latent_dim(f, p, d) for _, f, p, d in GRID})
    feats = ["cls", "patch"]
    cell = {}
    for name, f, p, d in GRID:
        cell[(f, latent_dim(f, p, d))] = (name, _arm(A, name))

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(-0.5, len(dims) - 0.5)
    ax.set_ylim(-0.75, len(feats) - 0.5)

    for yi, f in enumerate(feats):
        for xi, dim in enumerate(dims):
            got = cell.get((f, dim))
            if got is None:
                continue
            name, vals = got
            ax.add_patch(plt.Rectangle((xi - 0.44, yi - 0.30), 0.88, 0.60, fill=True,
                                       facecolor="#FAFCFB", edgecolor="#DCE6E0", lw=1.0,
                                       zorder=1))
            ax.text(xi, yi + 0.21, name, ha="center", va="center", fontsize=9.4,
                    color=INK, fontweight="bold", zorder=3)
            if not vals:
                ax.text(xi, yi - 0.06, "not evaluated", ha="center", va="center",
                        fontsize=9, color=MUTED, style="italic", zorder=3)
                continue
            m, lo, hi, n = _mean_ci(vals.values())
            ready = n >= FLOOR
            col = C["green"] if f == "cls" else C["purple"]
            ax.text(xi, yi - 0.02, f"{m:.3f}" if np.isfinite(m) else "--",
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color=col if ready else MUTED, zorder=3)
            sub = (f"[{lo:.3f}, {hi:.3f}]  n = {n}" if ready and np.isfinite(lo)
                   else f"n = {n} of {FLOOR} — pending")
            ax.text(xi, yi - 0.21, sub, ha="center", va="center", fontsize=8.4,
                    color=MUTED, style=None if ready else "italic", zorder=3)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"{d:,}" for d in dims], fontsize=10.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(["cls\n(1 token)", "patch\n(patch_size knob)"], fontsize=10.5)
    ax.set_xlabel("total latent dimension  =  tokens x proj_dim", fontsize=10.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)

    # the confounded original, off to the side and greyed
    lname, lf, lp, ld = LEGACY
    lvals = _arm(A, lname)
    ldim = latent_dim(lf, lp, ld)
    txt = f"{lname} sits at {ldim:,} dims — {ldim // dims[0]}x the cls baseline"
    if lvals:
        m, lo, hi, n = _mean_ci(lvals.values())
        txt += f"\nmean {m:.3f}, n = {n}: the contrast this grid replaces"
    # LEFT-anchored at the axis start, not centred. Centred it landed directly over the
    # "1,536" column, and in a figure whose entire claim is which column a thing sits in,
    # that reads as an annotation ON that column -- when the whole point is that columns is
    # at 98,304 and therefore off this axis altogether.
    ax.text(-0.46, -0.62, txt, ha="left", va="center", fontsize=9,
            color=MUTED, style="italic", zorder=3)

    suptitle(fig, "Round 7 — the dimension-matched grid",
             "A ROW difference is the feature; a COLUMN difference is the capacity. "
             "The old PiWM-columns vs LpWM-ltv contrast was neither:\n"
             "it moved both at once, 98,304 latent dimensions against 384.")
    fit(fig)
    p = os.path.join(out, "round7-grid.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# --- panel B: the contrasts ----------------------------------------------------------------
def _contrasts():
    """(label, control, arm, key). Feature rows first, then the two capacity ladders."""
    rows = [("FEATURE at 384 dims:  ltv-p1 (patch) vs ltv (cls)",
             "LpWM-ltv", "LpWM-ltv-p1", "purple"),
            ("FEATURE at 1,536:  cols-p4 vs ltv-d1536",
             "LpWM-ltv-d1536", "PiWM-cols-p4", "purple"),
            ("FEATURE at 6,144:  cols-p16 vs ltv-d6144",
             "LpWM-ltv-d6144", "PiWM-cols-p16", "purple"),
            ("CAPACITY cls:  d1536 vs ltv (384)",
             "LpWM-ltv", "LpWM-ltv-d1536", "green"),
            ("CAPACITY cls:  d6144 vs d1536",
             "LpWM-ltv-d1536", "LpWM-ltv-d6144", "green"),
            ("CAPACITY patch:  cols-p4 vs ltv-p1",
             "LpWM-ltv-p1", "PiWM-cols-p4", "teal"),
            ("CAPACITY patch:  cols-p16 vs cols-p4",
             "PiWM-cols-p4", "PiWM-cols-p16", "teal"),
            ("CONFOUNDED (both at once):  columns vs ltv",
             "LpWM-ltv", "PiWM-columns", "slate")]
    return rows


def fig_contrasts(out, floor=FLOOR):
    """The forest. Below the floor a contrast is PENDING, never a point."""
    use_style()
    A = collect()[0]

    rows = []
    for label, ctrl, arm, key in _contrasts():
        rows.append((label, key, _paired(A, ctrl, arm)))

    fig, ax = plt.subplots(figsize=(11.8, 0.46 * len(rows) + 2.8))
    ys = np.arange(len(rows))[::-1]
    # suptitle() is drawn in FIGURE coords just above the axes with va="top", so a two-line
    # subtitle descends into the axes and hid the top row. Reserve one blank row of headroom.
    ax.set_ylim(-0.8, len(rows) - 0.5 + 1.0)
    for y, (label, key, r) in zip(ys, rows):
        if r is None:
            ax.annotate("arm not in the archive", (0.0, y), ha="center", va="center",
                        fontsize=9, color=MUTED, style="italic",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F4F4F4",
                                  ec="#E0E0E0", lw=0.8))
            continue
        n = r["n"]
        if n < floor:
            ax.annotate(f"pending — n = {n} of {floor}", (0.0, y), ha="center", va="center",
                        fontsize=9, color=MUTED, style="italic",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F4F4F4",
                                  ec="#E0E0E0", lw=0.8))
            continue
        if r["degenerate"]:
            ax.scatter([r["mean"]], [y], s=90, facecolor="white", edgecolor=C["crimson"],
                       lw=1.9, zorder=4)
            ax.annotate(f"{r['mean']:+.3f} on {n}/{n} seeds — identical, no interval",
                        (0.62, y), fontsize=9, color=C["crimson"], va="center", ha="left",
                        annotation_clip=False, style="italic")
            continue
        sig = r["lo"] > 0 or r["hi"] < 0
        col = C[key] if sig else MUTED
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([r["mean"]], [y], s=78, color=col, zorder=4, marker="D" if sig else "o")
        ax.annotate(f"{r['mean']:+.3f}  n={n}", (0.62, y), fontsize=9, color=col,
                    va="center", ha="left", annotation_clip=False,
                    fontweight="bold" if sig else "normal")

    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.3)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel("paired Δ CEM success", fontsize=10.5)
    ax_style(ax, grid="x")
    for y in ys:
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#FAFAFA", zorder=0)

    ready = sum(1 for _, _, r in rows if r and r["n"] >= floor)
    suptitle(fig, f"Round 7 contrasts — {ready} of {len(rows)} at the n = {floor} floor",
             "The three FEATURE rows hold capacity fixed; the four CAPACITY rows hold the "
             "feature fixed. Only together do they say\nwhat the confounded row at the bottom "
             "cannot: whether patch tokens buy anything that the same number of dimensions "
             "does not.")
    fit(fig)
    p = os.path.join(out, "round7-contrasts.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-05")
    ap.add_argument("--only", choices=["grid", "contrasts"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # The arithmetic the grid rests on, checked rather than asserted in prose.
    assert tokens_for(224) == 1 and tokens_for(112) == 4
    assert tokens_for(56) == 16 and tokens_for(14) == 256
    assert latent_dim("patch", 112, 384) == 1536 == latent_dim("cls", None, 1536)
    assert latent_dim("patch", 56, 384) == 6144 == latent_dim("cls", None, 6144)
    assert latent_dim("patch", 224, 384) == 384 == latent_dim("cls", None, 384)

    if a.only in (None, "grid"):
        print("  wrote", fig_grid(a.out))
    if a.only in (None, "contrasts"):
        print("  wrote", fig_contrasts(a.out))


if __name__ == "__main__":
    main()
