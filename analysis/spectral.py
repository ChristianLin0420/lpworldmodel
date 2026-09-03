"""Spectral and controllability diagnostics of the learned latent dynamics.

The campaign has never measured a property of the predictor's WEIGHTS. This reads them
straight out of each checkpoint -- no dataset, no model construction, no GPU -- and asks
the question CEM actually cares about: over the planning horizon, can the action reach
the latent directions the planner needs?

For the state-augmented system  s_k = [z_k ; z_{k-1} ; z_{k-2}] :

    s_{k+1} = A_aug s_k + B_aug a_k
    A_aug   = [[A_0, A_1, A_2], [I, 0, 0], [0, I, 0]]      (companion form)
    B_aug   = [B ; 0 ; 0]

    C_H = [B_aug, A_aug B_aug, ..., A_aug^{H-1} B_aug]      (controllability matrix)
    W_c = C_H C_H^T                                          (controllability Gramian)

`W_c` ill-conditioned  =>  whole latent directions are unreachable by ANY action sequence,
so the CEM objective is flat along them however good the 1-step prediction is.

WHICH MATRICES.  `forward` returns W(relu(trunk)) and VWorldModel applies the link after,
so the effective one-step map, linearised with ReLU treated as active, is
    z_{t+1} = W ( sum_k A_k z_{t-k} + B a_t )
i.e. A_eff_k = W @ lags[k].weight and B_eff = W @ B.weight for mlp_var / ltv. For `var`
and `additive` there is no readout and the system IS linear -- the quantity is exact there
and an approximation everywhere else. `ltv`'s low-rank correction is data-dependent and is
therefore NOT included; what is measured is its static core.

Usage:
    python analysis/spectral.py --out spectral.json [--campaign campaign.json] [--arm ...]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")
CAMPAIGN_ALIAS = {"PiWM-columns": "PiWM-columns_patch"}


def _predictor_state(ckpt_path):
    """The predictor sub-state-dict. Checkpoints store each module separately
    (train.py:606 / plan.py:435), so no model needs to be built."""
    payload = torch.load(ckpt_path, map_location="cpu")
    v = payload.get("predictor")
    if isinstance(v, torch.nn.Module):
        v = v.state_dict()
    return v if isinstance(v, dict) else None


def system_matrices(sd):
    """(A_eff list, B_eff) in float64, or None if this predictor has no linear core."""
    lags = sorted(k for k in sd if re.fullmatch(r"lags\.\d+\.weight", k))
    if not lags or "B.weight" not in sd:
        return None                      # action_linear / ar_adaln / vit: no A_k, B
    A = [sd[k].double().numpy() for k in lags]
    B = sd["B.weight"].double().numpy()
    if "W.weight" in sd:                 # mlp_var / ltv: fold the readout in
        W = sd["W.weight"].double().numpy()
        A = [W @ Ak for Ak in A]
        B = W @ B
    return A, B


def companion(A):
    """A_aug for the state-augmented system, shape (H*D, H*D)."""
    H, D = len(A), A[0].shape[0]
    M = np.zeros((H * D, H * D))
    M[:D] = np.concatenate(A, axis=1)
    for k in range(1, H):
        M[k * D:(k + 1) * D, (k - 1) * D:k * D] = np.eye(D)
    return M


def spectral_radius(M):
    """max |eigenvalue|, by a full eigendecomposition.

    NOT power iteration: the companion matrix is real but NON-SYMMETRIC, and when its
    dominant eigenvalue belongs to a complex-conjugate pair the power method does not
    converge to |lambda|. Verified on a random 12x12 companion system, power iteration
    returned 1.030 against a true 1.148. O(n^3) at n=1152 is ~1s, which is affordable.
    """
    return float(np.abs(np.linalg.eigvals(M)).max())


def gramian(A, B, H=5, ridge=1e-6):
    """W_c = C_H C_H^T and its conditioning. H defaults to the CEM planning horizon."""
    M, Baug = companion(A), None
    D, n = A[0].shape[0], companion(A).shape[0]
    Baug = np.zeros((n, B.shape[1]))
    Baug[:D] = B
    blocks, cur = [Baug], Baug
    for _ in range(H - 1):
        cur = M @ cur
        blocks.append(cur)
    C = np.concatenate(blocks, axis=1)                  # (n, H * a_dim)
    Wc = C @ C.T
    ev = np.clip(np.linalg.eigvalsh(0.5 * (Wc + Wc.T)), 0.0, None)
    lam_max = float(ev.max())
    eps = ridge * max(lam_max, 1e-300)      # relative ridge: scale-free and stable
    rank = int((ev > lam_max * 1e-12).sum()) if lam_max > 0 else 0
    pos = ev[ev > lam_max * 1e-12]
    return dict(
        rank=rank, dim=int(n), lam_max=lam_max,
        lam_min_pos=float(pos.min()) if pos.size else 0.0,
        # how many orders of magnitude separate the easiest reachable direction from the
        # hardest one that is reachable at all
        log10_cond=float(np.log10(lam_max / pos.min())) if pos.size else float("inf"),
        # -logdet/dim with the SAME ridge the objective would use. W_c is singular
        # whenever H * a_dim < n, so an unridged logdet is -inf by construction.
        neg_logdet_per_dim=float(-np.log(ev + eps).mean()),
        ridge_rel=ridge,   # matches the objective's ridge in visual_world_model._ctrb_loss
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default="spectral.json")
    ap.add_argument("--campaign", default=None, help="campaign JSON, to join CEM in")
    ap.add_argument("--arm", default=None, help="substring filter on the arm name")
    ap.add_argument("--horizon", type=int, default=5, help="CEM's planning horizon")
    a = ap.parse_args()

    cem = json.load(open(a.campaign))["arms"] if a.campaign else {}
    rows, skipped = [], 0
    for d in sorted(glob.glob(os.path.join(a.repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        m = RUN_RE.match(name)
        ck = os.path.join(d, "checkpoints", "model_latest.pth")
        if not m or not os.path.exists(ck):
            continue
        arm, seed = m.group(1), int(m.group(2))
        if a.arm and a.arm not in arm:
            continue
        try:
            sd = _predictor_state(ck)
            sys_ = system_matrices(sd) if sd else None
        except Exception:
            sys_ = None
        if sys_ is None:
            skipped += 1
            continue
        A, B = sys_
        g = gramian(A, B, H=a.horizon)
        rows.append(dict(
            run=name, arm=arm, seed=seed, n_lags=len(A), D=int(A[0].shape[0]),
            rho_companion=spectral_radius(companion(A)),
            rho_A0=spectral_radius(A[0]),
            **g,
            cem=cem.get(CAMPAIGN_ALIAS.get(arm, arm), {}).get(str(seed))))
        print(f"  {name:44s} rho={rows[-1]['rho_companion']:.3f} "
              f"log10_cond={g['log10_cond']:.2f} rank={g['rank']}/{g['dim']}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  ({len(rows)} runs; {skipped} had no linear core)")
    if rows:
        r = np.array([x["rho_companion"] for x in rows])
        c = np.array([x["log10_cond"] for x in rows])
        print(f"  spectral radius : median {np.median(r):.3f}  "
              f"[{r.min():.3f}, {r.max():.3f}]   frac >= 1: {(r >= 1).mean():.2f}")
        print(f"  log10 cond(W_c) : median {np.median(c):.2f}  "
              f"[{c.min():.2f}, {c.max():.2f}]")


if __name__ == "__main__":
    main()
