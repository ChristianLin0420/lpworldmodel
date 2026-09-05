"""Do the models make the SAME mistake where the planner actually looks?

The campaign's loudest unexplained fact is M3's oracle ladder: replace the learned dynamics
with the simulator and success goes 0.427 -> 0.913. Tonight ruled out four explanations —

  * 1-step latent error   (`residual_corr.py`: members' errors are INDEPENDENT, rho_bar
                           +0.001, averaging removes 93% of them, and planning does not move)
  * compounding           (T6 at n=8: jump vs chain is +0.005 [-0.092, +0.102])
  * planner search budget (60 vs 300 CEM samples: -0.020 [-0.086, +0.046])
  * the leaf objective    (M3: learned objective + ORACLE dynamics already scores 0.913)

— and every one of those was measured ON THE DATASET's action distribution. CEM is not on it.
It proposes actions from `randn*sigma + mu`, most of which no demonstrator ever took.

So the gap this script tests: **errors independent on-distribution can still be common-mode
off-distribution.** If the members converge to the same wrong answer exactly where the planner
queries, then (a) the ensemble's variance reduction buys nothing there, which is what the
M-curve's flattening and the "does not beat its best member" result look like from the inside,
and (b) reducing average latent error was never going to help, which is six rounds of nulls.

WHAT IS MEASURED. Off-distribution there is no ground truth — no simulator rolled those
actions — so a residual against truth is not available. The ensemble-relevant quantity is
available and is the right one anyway: the ACROSS-MODEL SPREAD of the prediction. For M models
predicting z_i from the same state and action,

    disagreement = mean_i || z_i - mean_j z_j ||^2      (across-model variance)
    scale        = mean_i || z_i - mean_over_batch ||^2 (so the ratio is dimensionless)

Independent errors keep disagreement high; a common-mode error collapses it. Reporting the
RATIO of that quantity off-distribution to on-distribution is what isolates the effect: a
ratio near 1 means the planner's region is no different, and a ratio well below 1 means the
models agree precisely where they should not be trusted to.

    python analysis/offdist_agreement.py --arm LpWM-ltv --horizon 5
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


def _batch(cfg, n, device, num_pred):
    dsets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                num_pred=num_pred, frameskip=cfg.frameskip)
    dl = torch.utils.data.DataLoader(dsets["valid"], batch_size=n, shuffle=False)
    obs, act, _ = next(iter(dl))
    return {k: v.to(device) for k, v in obs.items()}, act.to(device)


@torch.no_grad()
def rollout_end(model, obs, act, k):
    """The model's K-step chained prediction, via the model's OWN _chain_rollout."""
    model.eval()
    z_true = model._link(model.encode_obs(obs)["visual"])
    nh = model.num_hist
    a_emb = model._act_emb_with_pose(act, obs.get("proprio"))
    if z_true.shape[1] < nh or a_emb.shape[1] < nh + k - 1:
        return None
    return model._chain_rollout(z_true[:, :nh], a_emb, k)[:, -1].float().cpu()


def cem_like_actions(act, sigma, gen):
    """CEM's proposal is randn*sigma + mu (planning/cem.py). The planner's mu starts at zero
    and its sigma is the action scale, so an unbiased draw at that scale is the honest
    stand-in for 'an action the planner would try'. Same shape and dtype as the real actions
    so nothing downstream can behave differently for a reason other than the values."""
    return torch.randn(act.shape, generator=gen, device="cpu").to(act.device).to(act.dtype) * sigma


def spread(Z):
    """(M, b, ...) -> across-model variance, and the batch variance that makes it scale-free."""
    Zf = Z.reshape(Z.shape[0], Z.shape[1], -1).numpy().astype(np.float64)
    across = float(np.mean(np.var(Zf, axis=0)))            # disagreement between models
    batch = float(np.mean(np.var(Zf.reshape(-1, Zf.shape[-1]), axis=0)))  # signal scale
    return across, batch, across / max(batch, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--arm", default="LpWM-ltv")
    ap.add_argument("--horizon", type=int, default=5, help="CEM's planning horizon")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="assets/offdist_agreement.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    gen = torch.Generator().manual_seed(0)                  # one action draw for every model
    on, off, names, cache, rnd = [], [], [], {}, {}
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
                cache[key] = _batch(cfg, a.batch, a.device, npred)
                obs0, act0 = cache[key]
                # ONE off-distribution action tensor, shared by every model, at the DATA's own
                # scale -- so the only thing differing between the two conditions is whether
                # the actions are the ones a demonstrator took.
                rnd[key] = cem_like_actions(act0, float(act0.float().std()), gen)
            obs, act = cache[key]
            model = load_model(ck, cfg, cfg.num_action_repeat, device=a.device)
            z_on = rollout_end(model, obs, act, a.horizon)
            z_off = rollout_end(model, obs, rnd[key], a.horizon)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:40s} SKIP {type(e).__name__}: {e}", flush=True)
            continue
        if z_on is None or z_off is None:
            continue
        on.append(z_on); off.append(z_off); names.append(name)
        print(f"  {name:40s} ok", flush=True)

    if len(on) < 2:
        raise SystemExit(f"need >=2 checkpoints of {a.arm}; found {len(on)}")
    A_on, B_on, R_on = spread(torch.stack(on))
    A_off, B_off, R_off = spread(torch.stack(off))
    out = dict(arm=a.arm, horizon=a.horizon, n_models=len(on), models=names,
               on_across=A_on, on_batch=B_on, on_ratio=R_on,
               off_across=A_off, off_batch=B_off, off_ratio=R_off,
               off_over_on=R_off / max(R_on, 1e-12))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n=== {a.arm}: {len(on)} checkpoints, K = {a.horizon} ===")
    print(f"  ON-distribution  (dataset actions)  disagreement/scale = {R_on:.4f}")
    print(f"  OFF-distribution (CEM-like actions) disagreement/scale = {R_off:.4f}")
    print(f"  ratio off/on = {out['off_over_on']:.3f}"
          "    (~1 = the planner's region is no different;  <<1 = COMMON-MODE off-distribution)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
