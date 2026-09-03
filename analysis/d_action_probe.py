"""Measure d_action on ANY checkpoint, including arms trained before the metric existed.

`causal/d_action` is logged from inside the training loop, so the arms that predate the
diagnostic -- LpWM-ltv, the mupfix / d2048 / vfloor variants, the gate family -- have no
value on record. The round-4 headline ("actgain raised d_action ~2000x and CEM did not
move") therefore rested on a single-seed number quoted in a train.py docstring. This
re-measures it from the checkpoints, on the same batch, for every arm.

The computation is character-for-character the one in train.py `_causal_diagnostics`:
run the model forward so `_diag` is populated, then permute the action across the batch
and take the RMS displacement of the linked prediction.

    python analysis/d_action_probe.py --out assets/d_action_probe.json [--arm LpWM-ltv]
"""
import argparse
import glob
import json
import os
import re
import sys

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan import load_model                                    # noqa: E402

RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")


@torch.no_grad()
def d_action(model, obs, act):
    """train.py:_causal_diagnostics, verbatim, on one batch."""
    model.eval()
    model(obs, act)                                # populates model._diag
    d = getattr(model, "_diag", None)
    if not d or d.get("act_src") is None:
        return None
    z_src, act_src = d["z_src"], d["act_src"]
    if z_src.shape[0] < 2:
        return None
    g = torch.Generator(device="cpu").manual_seed(0)            # same perm for every arm
    perm = torch.randperm(z_src.shape[0], generator=g).to(z_src.device)
    base = model._link(model.predict(z_src, act_src))
    z_a = model._link(model.predict(z_src, act_src[perm]))
    z_s = model._link(model.predict(z_src[perm], act_src))
    d_act = float((base - z_a).pow(2).mean().sqrt())
    d_st = float((base - z_s).pow(2).mean().sqrt())
    scale = float(base.pow(2).mean().sqrt())
    return dict(d_action=d_act, d_state=d_st, scale=scale,
                d_action_over_scale=d_act / max(scale, 1e-12),
                state_over_action=d_st / max(d_act, 1e-12))


def _batch(cfg, n, device):
    # train.py:393 takes the FIRST return (the window-sliced datasets); plan.py takes the
    # second (whole trajectories, ragged). d_action is a training-time quantity, so it is
    # the sliced split that reproduces the logged number.
    dsets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist,
                                num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dl = torch.utils.data.DataLoader(dsets["valid"], batch_size=n, shuffle=False)
    obs, act, _ = next(iter(dl))
    obs = {k: v.to(device) for k, v in obs.items()}
    return obs, act.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default="assets/d_action_probe.json")
    ap.add_argument("--arm", default=None, help="substring filter on the arm name")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows, cache = [], {}
    for d in sorted(glob.glob(os.path.join(a.repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        m, ck = RUN_RE.match(name), os.path.join(d, "checkpoints", "model_latest.pth")
        cfgf = os.path.join(d, "hydra.yaml")
        if not m or not os.path.exists(ck) or not os.path.exists(cfgf):
            continue
        arm, seed = m.group(1), int(m.group(2))
        if a.arm and a.arm not in arm:
            continue
        try:
            cfg = OmegaConf.load(cfgf)
            key = (str(cfg.env.dataset), cfg.num_hist, cfg.num_pred, cfg.frameskip)
            if key not in cache:
                cache[key] = _batch(cfg, a.batch, a.device)
            obs, act = cache[key]
            model = load_model(ck, cfg, cfg.num_action_repeat, device=a.device)
            r = d_action(model, obs, act)
            del model
            torch.cuda.empty_cache()
        except Exception as e:                       # a broken run must not stop the sweep
            print(f"  {name:44s} SKIP {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            print(f"  {name:44s} SKIP no act_src", flush=True)
            continue
        rows.append(dict(run=name, arm=arm, seed=seed, **r))
        print(f"  {name:44s} d_action={r['d_action']:.5f}  "
              f"/|z|={r['d_action_over_scale']:.5f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
