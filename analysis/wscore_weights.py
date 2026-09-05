"""ROUND 8 / M2 -- per-checkpoint whitening matrices for the plan-time metric.

`planning/objectives.py` ranks CEM candidates by an isotropic MSE averaged over every latent
coordinate. The code's Frobenius mass sits in ~25 of 384 directions (the measured participation
ratio), so directions the encoder never produces are weighted exactly as heavily as the ones that
carry the task. This computes

    W = (C + eps I)^(-1/2),    eps = 1e-3 * tr(C) / D

with C the encoder's own code covariance for THAT checkpoint, measured on real frames from
`runs/probe_cache.pt` (15,227 of them, already on disk -- no dataset, no rollout, no env).

WHY THIS IS A GATE AND NOT A PROPOSAL. It runs on checkpoints that are already trained, so the
contrast is SAME-CHECKPOINT paired: the 82 % training-seed variance that dominates every retrained
comparison is fully controlled, and only the ~16 % episode-block term is live. That is what lets its
bar be +0.05 rather than the +0.09 a retrained arm needs, and it is why 12 GPU-h can decide whether
~500 are worth spending on S2 / ST4 / ST5.

MEASURED IN THE LINKED SPACE, deliberately. The planner compares `encode_obs_linked` outputs against
rollout outputs (see the docstring on that method), so the covariance has to be of the linked code
`z`, not the pre-link `u`. Using the wrong one would whiten a space the objective never sees.

    python analysis/wscore_weights.py --runs LpWM-ltv_pd384_bf16_s3 ... --out assets/wscore
    python analysis/wscore_weights.py --arm LpWM-ltv --seeds 3 4 5 6 7 8 9 10

Writes assets/wscore/<run>.pt holding {"W": (D,D), "pr": float, "n": int, "run": str}, and prints
the participation ratio before and after whitening -- if PR does not rise, the matrix is not doing
what it claims and the run should not be scored with it.
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan import load_model  # noqa: E402

PROBE = "/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/probe_cache.pt"


def participation_ratio(C):
    """tr(C)^2 / ||C||_F^2 -- the same statistic train.py:178 logs as effective_dim."""
    tr = torch.diagonal(C).sum()
    return float(tr * tr / (C * C).sum().clamp_min(1e-20))


@torch.no_grad()
def code_covariance(model, frames, batch=64, device="cuda"):
    """Covariance of the LINKED code over `frames`, as (D, D) plus the sample count."""
    outs = []
    for i in range(0, len(frames), batch):
        v = frames[i : i + batch].to(device)
        z = model.encode_obs_linked({"visual": v.unsqueeze(1), "proprio": None})["visual"]
        outs.append(z.reshape(-1, z.shape[-1]).float().cpu())
    Z = torch.cat(outs, 0)
    Z = Z - Z.mean(0, keepdim=True)
    return (Z.T @ Z) / max(len(Z) - 1, 1), len(Z)


def whiten_from(C, eps_scale=1e-3):
    """(C + eps I)^(-1/2) by symmetric eigendecomposition.

    eps is scaled to tr(C)/D rather than fixed, so the ridge means the same thing on a code
    whose scale RDMReg pins and on one it does not.
    """
    D = C.shape[0]
    eps = eps_scale * float(torch.diagonal(C).sum()) / D
    evals, evecs = torch.linalg.eigh(C + eps * torch.eye(D, dtype=C.dtype))
    evals = evals.clamp_min(eps * 1e-3)
    return (evecs * evals.rsqrt().unsqueeze(0)) @ evecs.T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run names")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", default="assets/wscore")
    ap.add_argument("--n-frames", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    runs = list(a.runs or [])
    if a.arm and a.seeds:
        R = "/lustre/fsw/portfolios/edgeai/users/chrislin/projects/lpworldmodel/runs/outputs"
        for s in a.seeds:
            hit = sorted(glob.glob(f"{R}/{a.arm}_pd*_s{s}"))
            if hit:
                runs.append(os.path.basename(hit[0]))
    if not runs:
        ap.error("give --runs, or --arm with --seeds")

    os.makedirs(a.out, exist_ok=True)
    cache = torch.load(PROBE, map_location="cpu")
    frames = cache["visual"][: a.n_frames]
    print(f"  probe frames: {len(frames)}")

    for run in runs:
        try:
            model = load_model(run, "latest", device=a.device)
        except Exception as e:  # a missing or half-written checkpoint must not kill the sweep
            print(f"  {run:<44} SKIP ({type(e).__name__}: {str(e)[:60]})")
            continue
        model.eval()
        C, n = code_covariance(model, frames, device=a.device)
        W = whiten_from(C)
        pr_before = participation_ratio(C)
        pr_after = participation_ratio(W.T @ C @ W)
        path = os.path.join(a.out, f"{run}.pt")
        torch.save({"W": W, "pr": pr_before, "n": n, "run": run}, path)
        print(f"  {run:<44} PR {pr_before:6.2f} -> {pr_after:6.2f}  n={n}  -> {path}")


if __name__ == "__main__":
    main()
