"""Step 1 -- predictive Jaccard on PushT. Measurement only: no training, no model edits.

For each (subsampled) frame t of a held-out episode we compute two support-change
statistics on the LINKED representation z (the space the predictor works in):

    S_world_t = 1 - J_S(z_t,     z_{t+1})    observed support change
    S_model_t = 1 - J_S(z_hat_t, z_t)        predicted-vs-actual support mismatch

with z_hat_t = link(g(z_{t-H..t-1}, a_{t-H..t-1})), i.e. the one-step-ahead
prediction from true history.

The claim under test is that S_model detects contact ONSET, which S_world cannot:
onset is the first frame the agent touches the T-block, and the block has not moved
yet at that frame, so any statistic built from observed frame-to-frame change is
blind to it. Onset is therefore the discriminating label, and block displacement is
reported alongside as a validity check (if the two are collinear on PushT, the test
is void and we say so).

Gate: AUROC(S_model) > AUROC(S_world) on onset with non-overlapping 95% CIs,
bootstrapped over EPISODES (frame-level resampling would be fraudulently tight
under temporal autocorrelation).

Usage:
    python analysis/predictive_jaccard.py --run_dir runs/outputs/<run> --epoch latest
"""
import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.pusht_dset import PushTDataset  # noqa: E402
from plan import load_model  # noqa: E402  (reuse the canonical checkpoint loader)

# --- PushT geometry (env/pusht/pusht_env.py) -------------------------------------
# state = [agent_x, agent_y, block_x, block_y, block_angle, (vel_x, vel_y)]
AGENT_RADIUS = 15.0
# add_tee(scale=30, length=4): two body-local rectangles, as (cx, cy, hx, hy)
TEE_RECTS = ((0.0, 15.0, 60.0, 15.0), (0.0, 75.0, 15.0, 45.0))


def soft_jaccard(a, b, eps=1e-8):
    """J_S(a,b) = sum(min(a,b)) / sum(max(a,b)), over the last axis. Needs a,b >= 0."""
    return np.minimum(a, b).sum(-1) / (np.maximum(a, b).sum(-1) + eps)


def tee_surface_distance(states):
    """Signed-ish clearance between the agent circle and the T-block surface.

    Exact rather than a centroid proxy: the T is two axis-aligned rectangles in the
    body frame, so we rotate the agent into that frame and take the point-to-rect
    distance. pymunk's local_to_world is position + rotate(local, angle), so the
    inverse is rotate(world - position, -angle).
    """
    agent, block, ang = states[:, 0:2], states[:, 2:4], states[:, 4]
    d = agent - block
    c, s = np.cos(-ang), np.sin(-ang)
    local = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=-1)

    best = None
    for cx, cy, hx, hy in TEE_RECTS:
        q = np.abs(local - np.array([cx, cy])) - np.array([hx, hy])
        dist = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(
            np.maximum(q[:, 0], q[:, 1]), 0.0
        )
        best = dist if best is None else np.minimum(best, dist)
    return best - AGENT_RADIUS


def contact_signals(states, tau):
    """Per-frame signals aligned to z_t. Returns dict of length T (index 0 undefined)."""
    clearance = tee_surface_distance(states)
    touching = clearance < tau
    onset = np.zeros_like(touching)
    onset[1:] = touching[1:] & ~touching[:-1]  # c_t = 1[d_t < tau and d_{t-1} >= tau]
    offset = np.zeros_like(touching)
    offset[1:] = ~touching[1:] & touching[:-1]

    agent_disp = np.zeros(len(states))
    agent_disp[1:] = np.linalg.norm(np.diff(states[:, 0:2], axis=0), axis=-1)
    block_disp = np.zeros(len(states))
    block_disp[1:] = np.linalg.norm(np.diff(states[:, 2:4], axis=0), axis=-1)
    return {
        "clearance": clearance,
        "touching": touching.astype(float),
        "onset": onset.astype(float),
        "offset": offset.astype(float),
        "agent_disp": agent_disp,
        "block_disp": block_disp,
    }


def _rankdata(x):
    """Average ranks (1-based), ties averaged. Avoids a scipy dependency."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores, labels):
    """Mann-Whitney AUROC, tie-corrected."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = _rankdata(np.asarray(scores, dtype=float))
    return (r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def episode_bootstrap(stat_fn, per_ep_scores, per_ep_labels, n_boot=2000, seed=0):
    """Resample EPISODES with replacement; return (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(per_ep_scores)
    point = stat_fn(np.concatenate(per_ep_scores), np.concatenate(per_ep_labels))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = np.concatenate([per_ep_scores[i] for i in idx])
        l = np.concatenate([per_ep_labels[i] for i in idx])
        v = stat_fn(s, l)
        if not np.isnan(v):
            boots.append(v)
    if not boots:
        return point, np.nan, np.nan
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@torch.no_grad()
def episode_z(model, dset, ep_idx, frameskip, num_hist, device):
    """Encode one episode and produce one-step-ahead predictions.

    Subsampling mirrors TrajSlicerDataset: frames at stride `frameskip`, actions
    concatenated in blocks of `frameskip`, so z_t here matches training-time z.
    Returns (z, z_hat, states_sub); z_hat[t] is defined for t >= num_hist.
    """
    T_full = dset.get_seq_length(ep_idx)
    n = T_full // frameskip
    if n < num_hist + 2:
        return None
    frames = list(range(n * frameskip))
    obs, act, state, _ = dset.get_frames(ep_idx, frames)

    visual = obs["visual"][::frameskip].to(device)          # (n, 3, H, W)
    proprio = obs["proprio"][::frameskip].to(device)        # (n, proprio_dim)
    acts = act.reshape(n, -1).to(device)                    # (n, frameskip*action_dim)
    states_sub = state[::frameskip].cpu().numpy()

    z = model.encode_obs_linked(
        {"visual": visual[None], "proprio": proprio[None]}
    )["visual"][0]                                          # (n, p, d)
    act_emb = model.encode_act(acts[None])[0]               # (n, act_emb_dim)

    # every window of num_hist frames; window ending at e predicts frame e+1
    starts = np.arange(0, n - num_hist)
    zs = torch.stack([z[s : s + num_hist] for s in starts])            # (W, H, p, d)
    as_ = torch.stack([act_emb[s : s + num_hist] for s in starts])     # (W, H, a)
    u_pred = model.predict(zs, as_)
    z_pred = model._link(u_pred)[:, -1]                                # (W, p, d)

    z_np = z.float().cpu().numpy().reshape(n, -1)
    z_hat = np.full_like(z_np, np.nan)
    z_hat[num_hist:] = z_pred.float().cpu().numpy().reshape(len(starts), -1)

    # union-head runs: recover the winning head per frame so the assignment
    # raster plots real j*, using the same argmin the training loss uses.
    j_star = None
    if getattr(model, "n_heads", 1) > 1:
        z_all = model._link(model.predictor.forward_heads(zs, as_))[:, :, -1]  # (J,W,p,d)
        tgt = z[num_hist : num_hist + len(starts)].unsqueeze(0)
        per_head = ((z_all - tgt) ** 2).mean(dim=(-1, -2))              # (J, W)
        j_star = np.full(n, -1, dtype=int)
        j_star[num_hist:] = per_head.argmin(dim=0).cpu().numpy()
    return z_np, z_hat, states_sub, j_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--n_episodes", type=int, default=21)
    ap.add_argument("--tau", type=float, default=2.0, help="contact clearance threshold")
    ap.add_argument("--tau_sweep", default="0.0,1.0,2.0,5.0,10.0")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / "hydra.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = run_dir / "checkpoints" / f"model_{args.epoch}.pth"
    model = load_model(ckpt, cfg, cfg.num_action_repeat, device)
    model.eval()

    frameskip, num_hist = cfg.frameskip, cfg.num_hist
    transform = hydra.utils.instantiate(cfg.env.dataset.transform)
    data_path = f"{os.environ['DATASET_DIR']}/pusht_noise/{args.split}"
    dset = PushTDataset(
        n_rollout=args.n_episodes,
        transform=transform,
        data_path=data_path,
        normalize_action=cfg.normalize_action,
        with_velocity=True,
    )

    eps = []
    for i in range(len(dset)):
        got = episode_z(model, dset, i, frameskip, num_hist, device)
        if got is None:
            continue
        z, z_hat, states, j_star = got
        sig = contact_signals(states, args.tau)
        # valid t: z_hat defined and z_{t+1} exists, so both statistics share a support
        t = np.arange(num_hist, len(z) - 1)
        eps.append(
            {
                "S_world": 1.0 - soft_jaccard(z[t], z[t + 1]),
                "S_model": 1.0 - soft_jaccard(z_hat[t], z[t]),
                "signals": {k: v[t] for k, v in sig.items()},
                "rho": float((z[t] != 0).mean()),
                "z": z[t],
                "states": states[t],
                "j_star": None if j_star is None else j_star[t],
            }
        )
        print(f"  episode {i}: n={len(t)} onsets={int(sig['onset'][t].sum())}")
    if not eps:
        raise RuntimeError("no usable episodes")

    rho = float(np.mean([e["rho"] for e in eps]))
    # chance floor: exact for binary supports, an UPPER BOUND for continuous J_S
    # (sum(min)/sum(max) < 1 even on the intersection), so label it as bounding.
    j_chance_bound = rho / (2.0 - rho)
    rng = np.random.default_rng(0)
    allz = np.concatenate([e["z"] for e in eps])
    pi = rng.permutation(len(allz))
    j_rand = float(np.mean(soft_jaccard(allz, allz[pi])))

    results = {
        "run_dir": str(run_dir),
        "epoch": args.epoch,
        "tau": args.tau,
        "rho_l0_frac": rho,
        "J_chance_upper_bound": j_chance_bound,
        "J_random_pairs_empirical": j_rand,
        "n_episodes": len(eps),
        "n_frames": int(sum(len(e["S_world"]) for e in eps)),
        "auroc": {},
        "pearson": {},
        "base_rates": {},
        "validity": {},
    }

    def labels_of(key):
        """Binary labels per episode. Event signals are already 0/1; continuous
        ones (displacements) are split at their global median so AUROC is defined."""
        if key in ("onset", "offset", "touching"):
            return [(e["signals"][key] > 0.5).astype(float) for e in eps]
        thr = np.median(np.concatenate([e["signals"][key] for e in eps]))
        return [(e["signals"][key] > thr).astype(float) for e in eps]

    for key in ("onset", "offset", "block_disp", "agent_disp"):
        labs = labels_of(key)
        results["base_rates"][key] = float(np.concatenate(labs).mean())
        for stat in ("S_world", "S_model"):
            scores = [e[stat] for e in eps]
            pt, lo, hi = episode_bootstrap(auroc, scores, labs, args.n_boot)
            results["auroc"][f"{stat}|{key}"] = {"point": pt, "lo95": lo, "hi95": hi}
            results["pearson"][f"{stat}|{key}"] = pearson(
                np.concatenate(scores), np.concatenate(labs)
            )

    # validity: is contact onset merely collinear with block displacement?
    results["validity"]["r_blockdisp_onset"] = pearson(
        np.concatenate([e["signals"]["block_disp"] for e in eps]),
        np.concatenate([e["signals"]["onset"] for e in eps]),
    )
    results["validity"]["onset_base_rate"] = results["base_rates"]["onset"]

    # tau calibration: pick the threshold whose onset base rate looks like a
    # discrete event (well under ~0.20). Reported so tau is chosen from data.
    results["tau_sweep"] = {}
    for tv in [float(x) for x in args.tau_sweep.split(",")]:
        onsets, touch = [], []
        for e in eps:
            s = contact_signals(e["states"], tv)
            onsets.append(s["onset"])
            touch.append(s["touching"])
        results["tau_sweep"][str(tv)] = {
            "onset_rate": float(np.concatenate(onsets).mean()),
            "touch_rate": float(np.concatenate(touch).mean()),
        }

    w = results["auroc"]["S_world|onset"]
    m = results["auroc"]["S_model|onset"]
    gate = (m["point"] > w["point"]) and (m["lo95"] > w["hi95"])
    results["gate_step1_pass"] = bool(gate)

    print("\n" + "=" * 74)
    print(f"rho (l0_frac) = {rho:.4f}   J_chance upper bound = {j_chance_bound:.4f}")
    print(f"empirical random-pair J_S = {j_rand:.4f}  (bounding, not validating)")
    print(f"episodes = {len(eps)}  frames = {results['n_frames']}")
    print("-" * 74)
    print(f"{'label':<12} {'base':>6}  {'AUROC S_world':>22}  {'AUROC S_model':>22}")
    for key in ("onset", "offset", "block_disp", "agent_disp"):
        a = results["auroc"][f"S_world|{key}"]
        b = results["auroc"][f"S_model|{key}"]
        print(
            f"{key:<12} {results['base_rates'][key]:>6.3f}  "
            f"{a['point']:>6.3f} [{a['lo95']:.3f},{a['hi95']:.3f}]  "
            f"{b['point']:>6.3f} [{b['lo95']:.3f},{b['hi95']:.3f}]"
        )
    print("-" * 74)
    print(f"validity r(block_disp, onset) = {results['validity']['r_blockdisp_onset']:.3f}"
          "   (near 1 => collinear => test void)")
    print(f"onset base rate = {results['validity']['onset_base_rate']:.3f}"
          "   (>~0.20 => contact is not a discrete event on PushT)")
    print("\ntau calibration:")
    for tv, d in results["tau_sweep"].items():
        print(f"  tau={tv:>6}  onset_rate={d['onset_rate']:.3f}  touch_rate={d['touch_rate']:.3f}")
    print(f"\nSTEP 1 GATE: {'PASS' if gate else 'FAIL'}")
    print("=" * 74)

    out = Path(args.out or (run_dir / "analysis_step1.json"))
    out.write_text(json.dumps(results, indent=2, default=float))
    npz = out.with_suffix(".npz")
    np.savez_compressed(
        npz,
        n_episodes=len(eps),
        **{f"S_world_{i}": e["S_world"] for i, e in enumerate(eps)},
        **{f"S_model_{i}": e["S_model"] for i, e in enumerate(eps)},
        **{f"onset_{i}": e["signals"]["onset"] for i, e in enumerate(eps)},
        **{f"block_disp_{i}": e["signals"]["block_disp"] for i, e in enumerate(eps)},
        **{f"agent_disp_{i}": e["signals"]["agent_disp"] for i, e in enumerate(eps)},
        **{f"states_{i}": e["states"] for i, e in enumerate(eps)},
        **{f"z_{i}": e["z"] for i, e in enumerate(eps)},
        **{f"j_star_{i}": e["j_star"] for i, e in enumerate(eps) if e["j_star"] is not None},
    )
    print(f"wrote {out} and {npz}")


if __name__ == "__main__":
    main()
