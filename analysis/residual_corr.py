"""Are the models' prediction errors INDEPENDENT? The measurement round 7 turns on.

The campaign's only positive is plan-time consensus, and an adversarial audit narrowed it
sharply (`diary/2026-09-03.md` §16.9):

    vs the members' MEAN   +0.228   survives, 18/18 cells
    vs the members' BEST   +0.014   NULL

An ensemble that beats the average member but not the best one is doing VARIANCE REDUCTION,
not adding capability. Classical ensemble theory says exactly how far that can go: for M
members with per-member error variance s^2 and mean pairwise error correlation rho_bar, the
committee's error variance is

    Var_ens = s^2 * ( rho_bar + (1 - rho_bar)/M )          -> s^2 * rho_bar  as M -> inf

so the achievable gain is capped by rho_bar and NOTHING else. rho_bar = 0 means averaging M
models divides the error by M; rho_bar = 0.8 means even an infinite committee only removes
20% of it. The measured M-curve (M = 1,2,3,5 -> 0.393, 0.430, 0.583, 0.608) is already
flattening, which predicts a LARGE rho_bar.

This script measures rho_bar directly, from checkpoints, on one fixed batch. No training, no
planner, no GPU-hours beyond a forward pass per checkpoint. It is deliberately a
MEASUREMENT-BEFORE-MODEL step, per `diary/2026-09-03.md` §14 rule 4: measure the baseline's
value of a quantity before building arms to change it.

What the answer licenses:
  * rho_bar HIGH (say > 0.5) -- the members share a common-mode error. Ensembling is near its
    ceiling, more members will not help, and the defect is in something every seed inherits:
    the architecture, the objective, or the data. Round 7 must attack the shared cause.
  * rho_bar LOW -- errors really are independent, the M-curve should keep climbing, and the
    single-model ceiling is the thing to attack.

    python analysis/residual_corr.py --arm LpWM-ltv --out assets/residual_corr.json
"""
import argparse
import glob
import json
import os
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan import load_model                                    # noqa: E402

RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")


def _batch(cfg, n, device, num_pred=None):
    # The FIRST return is the window-sliced split, which is what training saw; plan.py takes
    # the second (ragged whole trajectories). Same choice as d_action_probe.py, for the same
    # reason -- this is a training-time quantity.
    # num_pred is overridable: a K-step residual needs K frames of ground-truth future,
    # and the baseline trains at num_pred=1 so its own config cannot supply them.
    dsets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                num_pred=(cfg.num_pred if num_pred is None else num_pred),
                                frameskip=cfg.frameskip)
    dl = torch.utils.data.DataLoader(dsets["valid"], batch_size=n, shuffle=False)
    obs, act, _ = next(iter(dl))
    return {k: v.to(device) for k, v in obs.items()}, act.to(device)


@torch.no_grad()
def residual(model, obs, act):
    """The model's 1-step latent prediction error, flattened. Same forward as training."""
    model.eval()
    model(obs, act)
    d = getattr(model, "_diag", None)
    if not d:
        return None
    # keys per models/visual_world_model.py:1536 -- the loss target is "target", not "z_tgt"
    zp, zt = d.get("z_pred"), d.get("target")
    if zp is None or zt is None:
        return None
    return (zp - zt).reshape(-1).float().cpu().numpy()


@torch.no_grad()
def residual_k(model, obs, act, k):
    """The model's K-step CHAINED rollout error against the true latent K steps ahead.

    Uses the model's own `_chain_rollout` -- the same object R2's consistency loss and T6's
    overshoot control use -- so this is not a re-implementation that could invent its own
    disagreement. The target is the ENCODER's latent at frame num_hist+k-1, i.e. the same
    quantity the 1-step loss regresses, just further out.

    Why this matters separately from the 1-step number: the planner rolls 5 steps, and
    errors that are independent at one step can become correlated once compounded through
    a shared dynamics map.
    """
    model.eval()
    u = model.encode_obs(obs)["visual"]
    z_true = model._link(u)                                  # (b, num_hist+k, p, d)
    nh = model.num_hist
    if z_true.shape[1] < nh + k:
        return None
    a_emb = model._act_emb_with_pose(act, obs.get("proprio"))
    if a_emb.shape[1] < nh + k - 1:
        return None
    z_chain = model._chain_rollout(z_true[:, :nh], a_emb, k)  # (b, k, p, d)
    return (z_chain[:, -1] - z_true[:, nh + k - 1]).reshape(-1).float().cpu().numpy()


def summarise(R, names):
    """R: (M, D) residual matrix. Returns rho_bar and what it implies for a committee."""
    # DO NOT centre across models. An earlier version did `R - R.mean(axis=0)`, which
    # subtracts, per residual dimension, the mean over models -- i.e. it removes precisely
    # the common-mode component this function exists to measure, and forces the rows to sum
    # to zero, which induces a correlation of exactly -1/(M-1) between them. With M = 16 that
    # is -0.0667; the run reported -0.0646. The measurement was of the centring, not of the
    # models. np.corrcoef already centres each ROW by its own mean, which is the correct
    # Pearson treatment: each residual vector is a series of observations.
    keep = [i for i in range(len(R)) if np.ptp(R[i]) > 0]
    if len(keep) < 2:
        return None
    C = np.corrcoef(R[keep])
    iu = np.triu_indices(len(keep), k=1)
    rho = C[iu]
    rho_bar = float(np.mean(rho))
    # Var_ens / Var_member for the measured committee sizes.
    ceil = {M: float(rho_bar + (1 - rho_bar) / M) for M in (2, 3, 5, 8, 12, 1000)}
    # --- BIAS vs VARIANCE, which the correlation ALONE cannot separate --------------
    # np.corrcoef centres each row by its own mean, so a component every model shares --
    # a systematic error in the SAME direction -- is invisible to rho_bar. That component
    # is exactly the one averaging cannot remove: a committee cancels variance, never bias.
    # So measure it directly. For M independent zero-mean errors the mean residual shrinks
    # as 1/sqrt(M); anything above that floor is shared bias.
    Rk = R[keep]
    mean_resid = Rk.mean(axis=0)                       # the committee's own residual
    nb = float(np.linalg.norm(mean_resid))
    nm = float(np.mean([np.linalg.norm(r) for r in Rk]))
    M = len(keep)
    shrink = nb / max(nm, 1e-12)                       # observed
    indep_floor = 1.0 / np.sqrt(M)                     # predicted if fully independent
    # fraction of a single member's squared error that survives averaging M of them
    bias_frac = float(shrink ** 2)

    return dict(
        n_models=M, models=[names[i] for i in keep],
        rho_bar=rho_bar, rho_min=float(rho.min()), rho_max=float(rho.max()),
        rho_median=float(np.median(rho)),
        var_ratio_vs_M=ceil,
        # the fraction of a member's error variance that NO committee can remove
        irreducible_fraction=rho_bar,
        # --- the bias probe ---
        committee_resid_norm=nb, mean_member_resid_norm=nm,
        shrink_observed=float(shrink), shrink_if_independent=float(indep_floor),
        shrink_ratio=float(shrink / indep_floor),
        bias_fraction_of_member_sq_error=bias_frac,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--arm", default="LpWM-ltv", help="exact arm name whose seeds form the committee")
    ap.add_argument("--out", default="assets/residual_corr.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=1,
                    help="K. 1 = the trained 1-step residual; >1 chains K steps, which is "
                         "the regime the planner actually uses (CEM horizon is 5).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows, names, cache = [], [], {}
    for d in sorted(glob.glob(os.path.join(a.repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        if name.startswith("CANARY-"):
            continue
        m = RUN_RE.match(name)
        ck = os.path.join(d, "checkpoints", "model_latest.pth")
        cfgf = os.path.join(d, "hydra.yaml")
        if not m or m.group(1) != a.arm or not os.path.exists(ck) or not os.path.exists(cfgf):
            continue
        try:
            cfg = OmegaConf.load(cfgf)
            npred = max(cfg.num_pred, a.horizon)
            key = (str(cfg.env.dataset), cfg.num_hist, npred, cfg.frameskip)
            if key not in cache:
                cache[key] = _batch(cfg, a.batch, a.device, num_pred=npred)
            obs, act = cache[key]
            model = load_model(ck, cfg, cfg.num_action_repeat, device=a.device)
            r = residual(model, obs, act) if a.horizon == 1 else residual_k(model, obs, act, a.horizon)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:40s} SKIP {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            print(f"  {name:40s} SKIP no z_pred/z_tgt in _diag", flush=True)
            continue
        rows.append(r); names.append(name)
        print(f"  {name:40s} |resid| rms = {float(np.sqrt((r**2).mean())):.5f}", flush=True)

    if len(rows) < 2:
        raise SystemExit(f"need >=2 checkpoints of {a.arm}; found {len(rows)}")
    R = np.stack(rows)
    out = summarise(R, names)
    out["arm"] = a.arm
    out["batch"] = a.batch
    out["horizon"] = a.horizon
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n=== {a.arm}: {out['n_models']} checkpoints, horizon K={a.horizon} ===")
    print(f"  mean pairwise error correlation  rho_bar = {out['rho_bar']:+.4f}")
    print(f"  range [{out['rho_min']:+.4f}, {out['rho_max']:+.4f}]  median {out['rho_median']:+.4f}")
    print("\n  committee error variance / single-member variance:")
    for M, v in out["var_ratio_vs_M"].items():
        lab = "M -> inf" if M == 1000 else f"M = {M}"
        print(f"    {lab:9s}  {v:.4f}")
    print(f"\n  irreducible fraction from correlation alone: {out['irreducible_fraction']:.4f}")
    print("\n  BIAS PROBE -- what averaging CANNOT remove:")
    print(f"    ||mean residual over {out['n_models']} models||   = {out['committee_resid_norm']:.5f}")
    print(f"    mean ||residual|| of one model         = {out['mean_member_resid_norm']:.5f}")
    print(f"    observed shrink                        = {out['shrink_observed']:.4f}")
    print(f"    shrink if perfectly independent (1/sqrt M) = {out['shrink_if_independent']:.4f}")
    print(f"    ratio observed/independent             = {out['shrink_ratio']:.3f}"
          "   (1.0 = pure variance, >>1 = shared bias)")
    print(f"    shared-bias share of a member's squared error = "
          f"{out['bias_fraction_of_member_sq_error']:.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
