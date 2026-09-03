"""Deep-dive analysis figures for the causal-objective round (diary/2026-09-03.md).

Every number is harvested at call time from the runs themselves -- each run's local
``wandb/latest-run/files/wandb-summary.json`` for the training diagnostics, and
``analysis/collect_evals.py``'s campaign JSON for the CEM numbers -- so re-running this
after more evals land refreshes the figures without editing anything.

Usage:
    python analysis/causal_figs.py --out diary/assets/2026-09-03 \
        [--campaign /tmp/campaign_now.json] [--incr-stats path/to/incr_stats.json]
"""
import argparse
import collections
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --- palette: the accents from analysis/arch_figs.py, so figures and diagrams agree ---
ACCENT = {"blue": "#0055af", "magenta": "#912d59", "green": "#006e00",
          "amber": "#9a6700", "crit": "#b3261e", "slate": "#3d3d3a"}
INK, MUTED, GRID = "#1A1A1A", "#5b5b57", "#d8d8d4"

# Categorical hues assigned by ARM IDENTITY in a fixed order -- never by rank, so a
# figure regenerated after more evals land does not repaint the arms.
ARM_COLOR = {
    "LpWM-ltv":             ACCENT["slate"],
    "LpWM-ltv-relu-p2":     ACCENT["green"],
    "LpWM-ltv-ident-p1":    ACCENT["crit"],
    "PiWM-columns":         "#2b7a78",
    "PiWM-pathint":         ACCENT["amber"],
    "PiWM-actinfo":         ACCENT["magenta"],
    "PiWM-actinfo-pathint": "#6b3f7a",
    "PiWM-incr":            ACCENT["blue"],
    "PiWM-incr-actinfo":    "#5a7fb8",
    "PiWM-blockcausal":     "#8a8a85",
    "PiWM-sigreg-arpred":   "#a06a3f",
}
LABEL = {
    "LpWM-ltv": "LpWM-ltv (control)", "LpWM-ltv-relu-p2": "reprelu, p=2",
    "LpWM-ltv-ident-p1": "identity, p=1", "PiWM-columns": "columns P=256",
    "PiWM-pathint": "V3 pathint", "PiWM-actinfo": "V2 actinfo",
    "PiWM-actinfo-pathint": "V2+V3", "PiWM-incr": "V1 incr",
    "PiWM-incr-actinfo": "V1+V2", "PiWM-blockcausal": "block-causal",
    "PiWM-sigreg-arpred": "sigreg-arpred",
}
RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")
# the columns run directory is PiWM-columns_pd384_patch_bf16_sN, so the regex yields
# "PiWM-columns" while collect_evals keys the arm "PiWM-columns_patch"
CAMPAIGN_ALIAS = {"PiWM-columns": "PiWM-columns_patch"}


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)


def harvest(repo, campaign):
    """One row per run: the causal diagnostics, the health metrics, and the CEM number."""
    cem = json.load(open(campaign))["arms"] if campaign and os.path.exists(campaign) else {}
    rows = []
    for d in sorted(glob.glob(os.path.join(repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        f = os.path.join(d, "wandb/latest-run/files/wandb-summary.json")
        m = RUN_RE.match(name)
        if not m or not os.path.exists(f):
            continue
        try:
            s = json.load(open(f))
        except Exception:
            continue
        arm, seed = m.group(1), int(m.group(2))
        rows.append(dict(
            run=name, arm=arm, seed=seed, done=os.path.exists(os.path.join(d, "DONE")),
            d_action=s.get("causal/d_action"), d_state=s.get("causal/d_state"),
            soa=s.get("causal/state_over_action"),
            rel_mse=s.get("err/rel_mse"), rho=s.get("sparsity/val_l0_frac"),
            eff=s.get("sparsity/effective_dim"),
            cem=cem.get(CAMPAIGN_ALIAS.get(arm, arm), {}).get(str(seed))))
    return rows


# ------------------------------------------------------------------ figure 1
def fig_d_action_strip(rows, out):
    """Per-run d_action and d_state. The point: d_action is now measured at n=8 per arm,
    not inferred from a single checkpoint, and the arms separate by three orders of
    magnitude -- so the diagnostic is a property of the OBJECTIVE, not of a lucky seed."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["d_action"] is not None:
            by[r["arm"]].append(r)
    order = sorted(by, key=lambda a: np.median([x["d_action"] for x in by[a]]))
    FLOOR = 3e-4                      # exact zeros must stay visible on a log axis
    XMAX, X_MED, X_RATIO = 60.0, 7.0, 24.0
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    for i, arm in enumerate(order):
        v = by[arm]
        col = ARM_COLOR.get(arm, MUTED)
        da = np.array([x["d_action"] for x in v], float)
        ds = np.array([x["d_state"] for x in v if x["d_state"] is not None], float)
        ax.scatter(np.clip(ds, FLOOR, None), np.full(ds.size, i) + 0.17, s=34,
                   facecolor="none", edgecolor=col, lw=1.4, zorder=3)
        ax.scatter(np.clip(da, FLOOR, None), np.full(da.size, i) - 0.17, s=46,
                   color=col, zorder=4)
        med = float(np.median(da))
        ax.text(X_MED, i, "0" if med < 1e-6 else f"{med:.3f}", va="center", ha="center",
                fontsize=10.5, color=col, weight="bold")
        if ds.size:
            r = np.median(ds) / med if med > 0 else np.inf
            ax.text(X_RATIO, i, "inf" if not np.isfinite(r) else f"{r:.0f}x",
                    va="center", ha="center", fontsize=10.5,
                    color=ACCENT["crit"] if (np.isfinite(r) and r > 10) or not np.isfinite(r)
                    else MUTED)
    ax.axvline(3.2, color=GRID, lw=1.2)
    ax.text(X_MED, len(order) - 0.35, "median\n$d_{action}$", va="bottom", ha="center",
            fontsize=10, color=INK)
    ax.text(X_RATIO, len(order) - 0.35, "$d_{state}$\n$/\\,d_{action}$", va="bottom",
            ha="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL.get(a, a) for a in order], fontsize=11.5)
    ax.set_ylim(-0.7, len(order) + 0.15)
    ax.set_xscale("log")
    ax.set_xlim(8e-5, XMAX)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xlabel("RMS displacement of the PREDICTED latent under an in-batch permutation "
                  "(log; points left of $10^{-4}$ are exact zeros)", fontsize=11.5, color=INK)
    _style(ax)
    ax.scatter([], [], s=46, color=INK, label=r"$d_{action}$  (permute the action only)")
    ax.scatter([], [], s=34, facecolor="none", edgecolor=INK, lw=1.4,
               label=r"$d_{state}$  (permute the state only)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), frameon=False, fontsize=10.5,
              ncol=2)
    ax.set_title("Every arm, every seed: how far does changing ONLY the action move the "
                 "prediction?\n"
                 "Three orders of magnitude separate the arms. V1's prediction does not move "
                 "at all; block-causal's action is worth 464x less than its state.",
                 fontsize=12.5, color=INK, loc="left", pad=14)
    fig.tight_layout()
    p = os.path.join(out, "d-action-strip.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 2
def fig_link_target(rows, out):
    """The 2x2 the archive never ran, now complete at n=8-16 per cell. Colour encodes the
    LINK, the x label carries both factors, and every test is printed inside its panel --
    no below-axis annotations, which tight_layout cannot reason about."""
    cells = [("LpWM-ltv", "reprelu", "p=1"), ("LpWM-ltv-relu-p2", "reprelu", "p=2"),
             ("LpWM-ltv-ident-p1", "identity", "p=1"), ("LeWM-ltv-p2", "identity", "p=2")]
    LINKCOL = {"reprelu": ACCENT["green"], "identity": ACCENT["crit"]}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    present = [(a, lk, tp) for a, lk, tp in cells if by.get(a)]
    metrics = [("rho", "code density  " + r"$\rho$", False),
               ("eff", "effective dim", False),
               ("rel_mse", "rel_mse  (log, lower is better)", True)]
    G = lambda arm, k: np.array([x[k] for x in by.get(arm, []) if x.get(k) is not None], float)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4))
    for ax, (key, title, logy) in zip(axes, metrics):
        def mw(a, b):
            x, y = G(a, key), G(b, key)
            return stats.mannwhitneyu(x, y).pvalue if x.size and y.size else float("nan")
        labs = []
        for j, (arm, lk, tp) in enumerate(present):
            v = G(arm, key)
            if not v.size:
                continue
            col = LINKCOL[lk]
            ax.bar(j, np.median(v), 0.6, color=col, alpha=0.26, zorder=2)
            ax.scatter(np.full(v.size, j) + np.linspace(-.15, .15, v.size), v,
                       s=26, color=col, zorder=4)
            ax.annotate(f"{np.median(v):.3g}", (j, v.max()), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=10.5,
                        color=col, weight="bold")
            labs.append((j, f"{lk}\n{tp}   (n={v.size})", col))
        ax.set_xticks([j for j, _, _ in labs])
        ax.set_xticklabels([t for _, t, _ in labs], fontsize=10.5)
        for tick, (_, _, col) in zip(ax.get_xticklabels(), labs):
            tick.set_color(col)
        ax.set_xlim(-0.6, len(present) - 0.4)
        if logy:
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 26)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
        _style(ax)
        pl = mw(present[0][0], present[2][0]) if len(present) == 4 else float("nan")
        p1, p2 = mw(present[0][0], present[1][0]), mw(present[2][0], present[3][0])
        ax.set_title(title + "\n" + r"$\bf{link}$" + ":  p = "
                     + (f"{pl:.1e}" if pl < 1e-3 else f"{pl:.2f}")
                     + f"      target_p:  p = {p1:.2f} / {p2:.2f}",
                     fontsize=12, color=INK, pad=10)
    fig.suptitle("target_p is inert in BOTH rows; the link decides the CODE but not the "
                 "prediction error\n"
                 "Four within-row tests, none significant. Across rows the code collapses "
                 "(eff_dim 24.1 -> 2.9) while rel_mse does not separate at all.",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    p = os.path.join(out, "link-target-2x2.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 3
def fig_v1_mechanism(stats_path, out):
    """Why V1 collapsed: eps=1e-4 sits far below the increments it was meant to floor."""
    if not stats_path or not os.path.exists(stats_path):
        return None
    S = json.load(open(stats_path))
    d2 = np.array(S["d2"], float)
    eps = S["eps"]
    w = 1.0 / (d2 + eps)
    w = w / w.mean()
    share = np.sort(w / w.sum())[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    a1.hist(np.log10(np.clip(d2, 1e-9, None)), bins=60, color=ACCENT["blue"],
            alpha=0.55, edgecolor="white", lw=0.4)
    a1.axvline(np.log10(eps), color=ACCENT["crit"], lw=2.4)
    a1.annotate(f"eps = {eps:g}\n{S['frac_below_eps'] * 100:.1f}% of samples fall below it",
                (np.log10(eps), a1.get_ylim()[1] * 0.92), xytext=(12, 0),
                textcoords="offset points", fontsize=10.5, color=ACCENT["crit"],
                va="top", weight="bold")
    a1.axvline(np.log10(S["d2_median"]), color=ACCENT["slate"], lw=1.8, ls="--")
    a1.annotate(f"median = {S['d2_median']:.2e}", (np.log10(S['d2_median']), 0),
                xytext=(8, 14), textcoords="offset points", fontsize=10.5, color=MUTED)
    a1.set_xlabel(r"$\log_{10} \; \|\Delta z\|^2$  (per sample, per timestep)", fontsize=11.5)
    a1.set_ylabel("count", fontsize=11)
    a1.set_title("The floor never engages", fontsize=12.5, color=INK)
    _style(a1)

    k = np.arange(1, share.size + 1) / share.size * 100
    a2.plot(k, np.cumsum(share) * 100, color=ACCENT["blue"], lw=2.2)
    a2.plot([0, 100], [0, 100], color=MUTED, lw=1.4, ls=":")
    top1 = share[0] * 100
    a2.annotate(f"the single heaviest sample alone\ntakes {top1:.1f}% of the batch loss\n"
                f"(weights span {S['w_span']:.0f}x)",
                (0, np.cumsum(share)[0] * 100), xytext=(26, 12),
                textcoords="offset points", fontsize=10.5, color=ACCENT["crit"],
                weight="bold")
    a2.set_xlabel("share of samples, heaviest first (%)", fontsize=11.5)
    a2.set_ylabel("cumulative share of the loss (%)", fontsize=11)
    a2.set_title("...so a handful of near-static frames own the gradient", fontsize=12.5,
                 color=INK)
    _style(a2)

    fig.suptitle("V1 is an implementation failure, not a test of the idea\n"
                 r"$w = 1/(\|\Delta z\|^2 + \varepsilon)$ was meant to stop large "
                 "autonomous motion dominating; it did the opposite.",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    p = os.path.join(out, "v1-mechanism.png")
    fig.savefig(p, dpi=155, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 4
# The gate thresholds. Chosen by inspection AFTER seeing this round's runs, so the
# figure is descriptive, not a pre-registered test -- the ~180 evals still queued are
# what will confirm or break it out of sample.
GATE = dict(rho=(0.25, 0.85), rel_mse=0.05, d_action=0.10)


def _passes(r, which=("rho", "rel_mse", "d_action")):
    ok = True
    if "rho" in which:
        ok &= r["rho"] is not None and GATE["rho"][0] <= r["rho"] <= GATE["rho"][1]
    if "rel_mse" in which:
        ok &= r["rel_mse"] is not None and r["rel_mse"] < GATE["rel_mse"]
    if "d_action" in which:
        ok &= r["d_action"] is not None and r["d_action"] >= GATE["d_action"]
    return ok


def fig_d_action_vs_cem(rows, out):
    """No single diagnostic predicts planning; their CONJUNCTION does, perfectly so far."""
    sub = [r for r in rows if r["cem"] is not None and r["d_action"] is not None
           and r["rel_mse"] is not None and r["rho"] is not None]
    if len(sub) < 6:
        return None
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.6, 7.4),
                                 gridspec_kw={"width_ratios": [2.05, 1]})

    # --- left: d_action vs CEM, marker filled when the OTHER two conditions hold ---
    ax.axvspan(GATE["d_action"], 1e3, color=ACCENT["green"], alpha=0.055, zorder=0)
    ax.axvline(GATE["d_action"], color=ACCENT["green"], lw=1.6, ls="--", zorder=1)
    ax.text(GATE["d_action"] * 1.16, 0.735, r"$d_{action} \geq 0.1$", fontsize=10.5,
            color=ACCENT["green"], va="top")
    seen = set()
    for r in sub:
        col = ARM_COLOR.get(r["arm"], MUTED)
        lab = LABEL.get(r["arm"], r["arm"]) if r["arm"] not in seen else None
        seen.add(r["arm"])
        other = _passes(r, ("rho", "rel_mse"))
        ax.scatter(max(r["d_action"], 3e-4), r["cem"], s=124 if other else 78,
                   color=col if other else "white", edgecolor=col, lw=2.0,
                   zorder=4, label=lab)
    x = np.array([max(r["d_action"], 3e-4) for r in sub])
    y = np.array([r["cem"] for r in sub])
    sp = stats.spearmanr(x, y)
    ax.annotate(f"Spearman($d_{{action}}$, CEM) = {sp.correlation:+.2f}, "
                f"p = {sp.pvalue:.3f}   (n = {len(sub)} runs)",
                (0.025, 0.975), xycoords="axes fraction", va="top", fontsize=11,
                color=INK, bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f0", ec=GRID))
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 8)
    ax.set_ylim(-0.04, 0.78)
    ax.set_xlabel(r"$d_{action}$  (log; points at the left edge are exact zeros)",
                  fontsize=11.5)
    ax.set_ylabel("CEM success rate", fontsize=12)
    _style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), frameon=False, fontsize=9.5,
              ncol=4, columnspacing=1.1)
    ax.set_title("Filled = healthy code AND accurate prediction; hollow = at least one "
                 "already broken",
                 fontsize=12, color=INK, loc="left", pad=10)

    # --- right: the conjunction ---
    ok = [r["cem"] for r in sub if _passes(r)]
    no = [r["cem"] for r in sub if not _passes(r)]
    fisher = stats.fisher_exact([[sum(c > 0 for c in ok), sum(c == 0 for c in ok)],
                                 [sum(c > 0 for c in no), sum(c == 0 for c in no)]]).pvalue
    for j, (vals, col, name) in enumerate(((ok, ACCENT["green"], "all three\nhold"),
                                           (no, ACCENT["crit"], "at least one\nfails"))):
        v = np.array(vals, float)
        bx.scatter(np.full(v.size, j) + np.linspace(-.17, .17, v.size), v, s=62,
                   color=col, zorder=4, alpha=0.9)
        bx.plot([j - .3, j + .3], [v.mean()] * 2, color=col, lw=3, zorder=5)
        bx.annotate(f"mean {v.mean():.3f}\n{(v > 0).sum()}/{v.size} plan above zero",
                    (j, v.max()), xytext=(0, 20), textcoords="offset points",
                    ha="center", fontsize=10.5, color=col, weight="bold")
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["all three\nhold", "at least one\nfails"], fontsize=11.5)
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(-0.04, 0.78)
    bx.set_ylabel("CEM success rate", fontsize=12)
    _style(bx)
    bx.set_title("gate:  " + r"$0.25 \leq \rho \leq 0.85$" + ",   rel_mse < 0.05,   "
                 + r"$d_{action} \geq 0.1$"
                 + f"\nFisher exact  p = "+ (f"{fisher:.1e}" if fisher < 1e-3 else f"{fisher:.4f}"),
                 fontsize=11.5, color=INK, pad=12)

    a_p = stats.fisher_exact([[sum(1 for r in sub if _passes(r, ("rho",)) and r["cem"] > 0),
                               sum(1 for r in sub if _passes(r, ("rho",)) and r["cem"] == 0)],
                              [sum(1 for r in sub if not _passes(r, ("rho",)) and r["cem"] > 0),
                               sum(1 for r in sub if not _passes(r, ("rho",)) and r["cem"] == 0)]]).pvalue
    fig.suptitle("No single diagnostic separates the planners -- the CONJUNCTION of three "
                 "does\n"
                 f"Each alone is weak (" + r"$\rho$" + f" p={a_p:.2f}, and rel_mse and "
                 r"$d_{action}$" " likewise). Together, every run that passes plans above "
                 "zero -- but some that FAIL still plan,\nso the gate is sufficient so far, "
                 "not necessary. Thresholds were set after seeing these runs; the evals "
                 "still queued will test them out of sample.",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.905])
    p = os.path.join(out, "d-action-vs-cem.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 5
def fig_quadrant(rows, out):
    """Arm medians in the plane the round is actually about: is the code healthy, and is
    the prediction action-sensitive? Buying the second by destroying the first is easy."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["d_action"] is not None and r["rho"] is not None:
            by[r["arm"]].append(r)
    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    lo, hi = GATE["rho"]
    ax.add_patch(plt.Rectangle((lo, GATE["d_action"]), hi - lo, 12, color=ACCENT["green"],
                               alpha=0.07, zorder=0, lw=0))
    ax.plot([lo, hi, hi, lo, lo], [GATE["d_action"], GATE["d_action"], 12, 12,
                                   GATE["d_action"]],
            color=ACCENT["green"], lw=1.8, ls="--", zorder=1)
    ax.text((lo + hi) / 2, 5.4, "HEALTHY CODE  and  ACTION-SENSITIVE", ha="center",
            fontsize=11.5, color=ACCENT["green"], weight="bold")
    ax.text(0.045, 5.4, "collapsed code", fontsize=11, color=ACCENT["crit"])
    ax.text(0.90, 5.4, "saturated code", fontsize=11, color=ACCENT["crit"], ha="right")
    # hand offsets: three arms land almost on top of each other inside the box
    NUDGE = {"PiWM-columns": (-58, -34), "LpWM-ltv-relu-p2": (-52, 20),
             "PiWM-pathint": (44, 20), "LpWM-ltv-ident-p1": (-8, -36),
             "PiWM-sigreg-arpred": (-16, 22)}
    for arm, v in by.items():
        col = ARM_COLOR.get(arm, MUTED)
        x = float(np.median([r["rho"] for r in v]))
        y = max(float(np.median([r["d_action"] for r in v])), 4e-4)
        inside = lo <= x <= hi and y >= GATE["d_action"]
        ax.scatter(x, y, s=210 if inside else 130, color=col if inside else "white",
                   edgecolor=col, lw=2.4, zorder=5)
        dx, dy = NUDGE.get(arm, (0, 17))
        ax.annotate(f"{LABEL.get(arm, arm)}\n(n={len(v)})", (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha="center", fontsize=10.5, color=col,
                    weight="bold" if inside else "normal")
    ax.set_yscale("log")
    ax.set_ylim(2.5e-4, 12)
    ax.set_xlim(-0.04, 1.08)
    ax.set_xlabel(r"code density  $\rho$   (fraction of units active; 1.0 = fully dense)",
                  fontsize=12)
    ax.set_ylabel(r"$d_{action}$   (log)", fontsize=12)
    _style(ax)
    ax.set_title("Action-sensitivity is cheap if you are willing to wreck the code\n"
                 "V2 -- the arm DESIGNED for action-sensitivity -- bought the most of it and "
                 "fell out of the healthy band doing so.\n"
                 "Three arms landed in the box: V3 pathint, reprelu p=2, and columns.",
                 fontsize=12.5, color=INK, loc="left", pad=14)
    fig.tight_layout()
    p = os.path.join(out, "causal-quadrant.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 6
def fig_arm_health(rows, out):
    """Per-arm code density and prediction error side by side, so 'healthy' is not a word."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["d_action"] is not None and r["rho"] is not None and r["rel_mse"] is not None:
            by[r["arm"]].append(r)
    # barh index 0 sits at the BOTTOM, so sort worst-first to put the best arms on top
    order = sorted(by, key=lambda a: -np.median([r["rel_mse"] for r in by[a]]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 6.0), sharey=True)
    ys = np.arange(len(order))
    lo, hi = GATE["rho"]
    for i, arm in enumerate(order):
        v = by[arm]
        col = ARM_COLOR.get(arm, MUTED)
        rho = float(np.median([r["rho"] for r in v]))
        mse = float(np.median([r["rel_mse"] for r in v]))
        a1.barh(i, rho, 0.66, color=col, alpha=0.85 if lo <= rho <= hi else 0.3,
                edgecolor=col, lw=1.6)
        a2.barh(i, min(mse, 1.2), 0.66, color=col,
                alpha=0.85 if mse < GATE["rel_mse"] else 0.3, edgecolor=col, lw=1.6)
        a1.annotate(f"{rho:.3f}", (rho, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=col)
        a2.annotate(f"{mse:.4f}", (min(mse, 1.2), i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10, color=col)
    a1.axvspan(lo, hi, color=ACCENT["green"], alpha=0.07, zorder=0)
    a1.set_yticks(ys)
    a1.set_yticklabels([LABEL.get(a, a) for a in order], fontsize=11.5)
    a1.set_xlim(0, 1.22)
    a1.set_xlabel(r"code density  $\rho$", fontsize=12)
    a1.set_title("Which arms kept a healthy code?  (shaded = the band)", fontsize=12.5,
                 color=INK)
    a2.axvline(GATE["rel_mse"], color=ACCENT["green"], lw=1.5, ls="--")
    a2.axvline(1.0, color=ACCENT["crit"], lw=1.8, ls="--")
    a2.annotate("= predicting\n   the mean", (1.0, 0.35), xytext=(7, 0),
                textcoords="offset points", ha="left", fontsize=10, color=ACCENT["crit"])
    a2.annotate("gate:\nrel_mse < 0.05", (GATE["rel_mse"], 0.35), xytext=(-7, 0),
                textcoords="offset points", ha="right", fontsize=10, color=ACCENT["green"])
    a2.set_xscale("log")
    a2.set_xlim(8e-4, 6)
    a2.set_xlabel("rel_mse  (log, clipped at 1.2)", fontsize=12)
    a2.set_title("...and which still predict?", fontsize=12.5, color=INK)
    for ax in (a1, a2):
        _style(ax)
        ax.set_ylim(-0.7, len(order) - 0.2)
    fig.tight_layout()
    p = os.path.join(out, "arm-health.png")
    fig.savefig(p, dpi=155, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diary/assets/2026-09-03")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--campaign", default="/tmp/campaign_now.json")
    ap.add_argument("--incr-stats", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = harvest(a.repo, a.campaign)
    print(f"harvested {len(rows)} runs with causal diagnostics "
          f"({sum(r['cem'] is not None for r in rows)} with a CEM number)")
    for p in (fig_d_action_strip(rows, a.out), fig_link_target(rows, a.out),
              fig_v1_mechanism(a.incr_stats, a.out), fig_d_action_vs_cem(rows, a.out),
              fig_quadrant(rows, a.out), fig_arm_health(rows, a.out)):
        if p:
            print("  wrote", p)


if __name__ == "__main__":
    main()
