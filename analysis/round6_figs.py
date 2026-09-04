"""Round-6 figures: the objective screen, and what it did to my own proposals.

The round's first act was a screen, not an experiment. `analysis/screen_objective.py` asks of any
candidate quantity: does it order the models that ALREADY WORK, or does it merely separate dead
models from live ones? Two rounds of this campaign were built on quantities that fail that test.

    python analysis/round6_figs.py --out diary/assets/2026-09-04
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from analysis.style import (C, FILL, GRID, INK, MUTED, ax_style,  # noqa: E402
                            fit, panel_title, use_style)

SCREEN = "assets/objective_screen.json"


def _load(path=SCREEN):
    return json.load(open(path)) if os.path.exists(path) else None


def fig_screen(rows, out):
    """The screen: how many convincing raw correlations survive their own covariates."""
    use_style()
    rows = sorted(rows, key=lambda r: -abs(r["partial"]))
    fig = plt.figure(figsize=(15.2, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.62)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- A: raw -> partial, one connected pair per candidate -----------------------
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        survives = abs(r["partial"]) >= 0.15 and r["p"] <= 0.05
        key = "green" if (survives and r["mono"]) else ("amber" if survives else "crimson")
        ax.plot([r["raw"], r["partial"]], [y, y], color=C[key], lw=2.0, alpha=0.55,
                solid_capstyle="round", zorder=3)
        ax.scatter([r["raw"]], [y], s=54, facecolor="white", edgecolor=MUTED, lw=1.5, zorder=4)
        ax.scatter([r["partial"]], [y], s=96, color=C[key], zorder=5,
                   marker="D" if (survives and r["mono"]) else "o")
        tag = ("survives, monotone" if survives and r["mono"] else
               "survives, but has an interior optimum" if survives else
               "collapse detector")
        ax.annotate(f"{r['partial']:+.3f}   {tag}", (0.82, y), xycoords=("data", "data"),
                    ha="left", va="center", fontsize=9.5, color=C[key], annotation_clip=False,
                    fontweight="bold" if survives and r["mono"] else "normal")
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.axvspan(-0.15, 0.15, color=FILL["crimson"], alpha=0.5, zorder=1, lw=0)
    ax.annotate("|partial| < 0.15", (0, ys.max() + 0.62), ha="center", va="bottom",
                fontsize=9, color=C["crimson"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']}\n n = {r['n']}" for r in rows], fontsize=9.5)
    ax.set_xlim(-1.02, 0.80)
    ax.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    ax.spines["bottom"].set_bounds(-1.0, 0.75)
    ax.set_xlabel("Spearman against CEM success   (hollow = raw,  solid = partial)", fontsize=10.5)
    ax_style(ax, grid="x")
    panel_title(ax, "A", "Most of these correlations are not real",
                "the hollow point is what you see; the solid point is what survives removing "
                "code density and prediction error")

    # --- B: the two exemplars, binned ------------------------------------------------
    keep = {"jacc/S_model": ("green", "S_model — support dissimilarity"),
            "rollout/val_z_visual_err_rollout_h8": ("crimson", "h8/h1 — rollout error growth")}
    x = np.arange(4)
    w = 0.34
    for i, (key, (ck, lab)) in enumerate(keep.items()):
        r = next((q for q in rows if q["key"] == key), None)
        if r is None:
            continue
        bx.bar(x + (i - 0.5) * w, r["means"], width=w * 0.88, color=C[ck], alpha=0.9,
               label=lab, zorder=3)
        for xi, m in zip(x + (i - 0.5) * w, r["means"]):
            bx.annotate(f"{m:.3f}", (xi, m), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, color=C[ck], fontweight="bold")
    bx.set_xticks(x)
    bx.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"], fontsize=9.5)
    bx.set_ylabel("mean CEM success", fontsize=10.5)
    bx.set_ylim(0, 0.50)
    bx.legend(loc="upper right", fontsize=9.5)
    ax_style(bx)
    panel_title(bx, "B", "Monotone, or flat",
                "quartiles of each quantity, over predictors that actually predict "
                "(rel_mse < 0.05)")

    fig.text(0.0, 1.055, "The objective screen: two of this campaign's design targets were "
             "collapse detectors", fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.0, 1.012, "A quantity that separates dead models from live ones correlates with "
             "everything, because dead models plan at zero. rel_mse is included as a self-null "
             "control:\nit is one of the covariates, so it must screen to ~0, and it does "
             "(+0.003). Only S_model both survives and orders monotonically.",
             fontsize=10, color=MUTED, ha="left", va="top")
    fit(fig)
    p = os.path.join(out, "objective-screen.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-04")
    ap.add_argument("--screen", default=SCREEN)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = _load(a.screen)
    if not rows:
        raise SystemExit(f"{a.screen} missing -- run the screen first")
    print("  wrote", fig_screen(rows, a.out))


if __name__ == "__main__":
    main()
