"""T7 - train E(z_0, a_{1:H}, z_g) on a FROZEN LpWM checkpoint to RANK action sequences.

    python train_energy.py --run LpWM-ltv_pd384_bf16_s3 --mode both \
        --episodes 3000 --steps 20000 --out assets/energy/s3

Two arms, trained here from the SAME cache, the SAME triples and the SAME negatives, so
the only difference between them is the objective:

  rank     (PiWM-energy)          L = -log softmax(-E)[0] over {a+, 63 negatives}
  distill  (PiWM-energy-distill)  L = (E - ||z_H(a) - z_g||^2 / y_scale)^2, the quantity
                                  CEM minimises TODAY, taken from this very checkpoint's
                                  own rollout and regressed into the same architecture.

If the two arms plan the same, T7's gain was capacity and smoothing, not the ranking
objective. That is the control the spec asks for and it is why `--mode both` exists.

WHY THE ENCODER IS FROZEN, AND WHAT THAT BUYS
    Both previous InfoNCE-over-actions arms collapsed the code (rho 0.448 -> 0.052 with
    batch-permuted negatives; rel_mse exactly 1.0000 with kNN negatives). In both the
    ENCODER was free and zeroing z was the cheapest way to satisfy the contrastive term.
    Here `requires_grad_(False)` is asserted over every world-model parameter, so the
    representation is a constant function of the data and that escape route does not
    exist. It also makes the frozen latent CACHEABLE: `encode_obs` folds t into the batch
    (`visual_world_model.py:571-599`), so a frame's code does not depend on its window,
    and one pass over whole episodes replaces re-encoding 40 decoded frames per window.
    The script REFUSES to cache if the encoder is block-causal, where that is false.

NEGATIVES ARE THE PLANNER'S OWN PROPOSAL (planning/cem.py:99-106)
    CEM samples `randn * sigma + mu` with `var_scale=1`. At opt step 0 that is exactly
    N(0, 1); at later steps mu is the elite mean and sigma has shrunk toward it. So half
    the negatives are N(0,1) and half are `a+ + s*eps, s ~ U(0.1, 1.0)`. A head trained
    against any other negative distribution would be scored on sequences the planner
    never queries.

WHAT IS MEASURED (rank agreement only -- diary sec 12b withdrew the d_action motivation)
    val_top1       fraction of held-out anchors where the head ranks the EXECUTED
                   sequence first out of 64. Chance = 1/64 = 1.56%.
    val_top1_mse   the same 64-way contest scored by today's leaf metric,
                   ||z_H(a) - z_g||^2 from the checkpoint's own rollout. This is the
                   number the head has to beat for the arm to mean anything.
    val_rho_mse    Spearman(E, today's leaf metric) per anchor. ~1 means the head merely
                   re-learned the rollout distance; low means it induces a different
                   ordering, which is the object under test.

NEW-HEAD CHECKLIST (round 5 mandatory; `path_int` had none of it -- diary sec 13.3)
    a. optimizer group ................ `torch.optim.AdamW(head.parameters())` below
    b. checkpoint key + shown present . `save_energy_head` writes `state_dict` and this
                                        script re-loads it with `load_energy_head`
                                        (strict) and prints the checksum both times
    c. lazy .to(device) ............... `EnergyCEMPlanner.__init__`
    d. tests mirror ................... `tests/test_energy_planner.py`
    e. gradient reaches the head ...... `--assert-grad` (on by default) fails the run if
                                        any parameter's grad is None or the total norm is
                                        0 after the first backward
"""

import argparse
import json
import os
import sys
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plan import load_model                                          # noqa: E402
from planning.energy import (                                        # noqa: E402
    SeqEnergyHead,
    load_energy_head,
    save_energy_head,
    state_dict_checksum,
)

# The window geometry, fixed by the planner: goal_H = 5 model steps, frameskip = 5 raw
# rows per step, num_hist = 3 observed frames. An 8-frame window therefore has the
# planner's anchor at index 2 and its goal at index 7, exactly 5 model steps apart, and
# the actions in between are rows 2..6.
NUM_HIST, NUM_PRED, FRAMESKIP = 3, 5, 5
I0, IG = 2, 7                       # window indices of the anchor and the goal
RAW0, RAWG = I0 * FRAMESKIP, IG * FRAMESKIP          # 10, 35
WIN_RAW = (NUM_HIST + NUM_PRED) * FRAMESKIP          # 40 raw rows per window


# ------------------------------------------------------------------ model loading


def load_frozen(run, epoch, ckpt_base, device):
    run_dir = os.path.join(ckpt_base, "outputs", run)
    cfg = OmegaConf.load(os.path.join(run_dir, "hydra.yaml"))
    ckpt = os.path.join(run_dir, "checkpoints", f"model_{epoch}.pth")
    model = load_model(ckpt, cfg, cfg.num_action_repeat, device=device)
    model.eval()
    model.requires_grad_(False)
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not live, f"world model is not frozen: {live[:4]}"
    assert model.action_conditioning == "adaln", (
        "train_energy.py targets the LpWM/LeWM adaln path (encode_obs returns visual "
        f"only); got action_conditioning={model.action_conditioning}"
    )
    assert not getattr(model, "use_pose", False), (
        "use_pose binds the agent pose into the action embedding, so the cached-latent "
        "rollout would not be the plan-time rollout. Not supported here."
    )
    enc = getattr(model.encoder, "module", model.encoder)
    assert not getattr(enc, "block_causal", False), (
        "a block-causal encoder makes a frame's code depend on its window, so the "
        "per-frame latent cache would not be the plan-time latent."
    )
    print(
        f"frozen world model: {run} epoch={epoch} emb_dim={model.encoder.emb_dim} "
        f"num_patches={model.encoder.num_patches} num_hist={model.num_hist} "
        f"num_pred={model.num_pred} n_heads={getattr(model, 'n_heads', 1)} "
        f"overshoot={getattr(model, 'overshoot', False)} link={type(model.link).__name__}",
        flush=True,
    )
    return model, cfg


@torch.no_grad()
def rollout_terminal_from_z(model, z0, act):
    """||z_H|| after H = act.shape[1] model steps, started from a CACHED latent.

    A transcription of `_rollout_adaln`'s K = 1 branch (`visual_world_model.py:1104-1109`)
    with `encode_obs` replaced by the cache. The plan-time rollout starts from ONE
    observed frame (`plan.py` expand_dims), so `_predict_next_adaln` widens its history
    window 1 -> 2 -> 3 exactly as it does here. `--check` asserts this equals
    `model.rollout` on real images to within float error; if it ever stops matching, the
    distill control is regressing the wrong quantity.
    """
    assert (1 if getattr(model, "overshoot", False) else model.num_pred) == 1, (
        "this rollout mirrors the K=1 branch; a K>1 option model rolls differently"
    )
    assert getattr(model, "n_heads", 1) == 1, "union-head rollout needs a goal latent"
    emb = z0[:, None, None, :]                                  # (B,1,1,D) linked
    act_emb_all = model._act_emb_with_pose(act, None)           # (B,H,a)
    while emb.shape[1] < act.shape[1]:
        emb = torch.cat([emb, model._predict_next_adaln(emb, act_emb_all)], dim=1)
    emb = torch.cat([emb, model._predict_next_adaln(emb, act_emb_all)], dim=1)
    return emb[:, -1, 0]                                        # (B,D)


@torch.no_grad()
def leaf_mse(model, z0, acts, zg):
    """Today's CEM leaf score for (B,K) candidates: mean_d (z_H(a) - z_g)^2.

    Same reduction as `planning.objectives.objective_fn_last` on the adaln path (visual
    only, last predicted frame, `nn.MSELoss` mean over the latent dims).
    """
    B, K = acts.shape[:2]
    z0f = z0[:, None].expand(B, K, z0.shape[-1]).reshape(B * K, -1)
    zgf = zg[:, None].expand(B, K, zg.shape[-1]).reshape(B * K, -1)
    zH = rollout_terminal_from_z(model, z0f, acts.reshape(B * K, *acts.shape[2:]))
    return (zH - zgf).pow(2).mean(dim=-1).view(B, K)


# ------------------------------------------------------------------ data


class _FrameChunks(torch.utils.data.Dataset):
    """One item = `chunk` CONSECUTIVE frames of one episode, decoded and transformed.

    Not one item per episode: a 125-frame PushT episode at 224x224 float32 is 75 MB, and
    a DataLoader worker returns tensors through /dev/shm, which is 64 MB on this
    cluster's login node -- one whole episode is a guaranteed `Bus error`. Chunking caps
    the in-flight payload at workers * prefetch * chunk * 0.6 MB.

    Actions never go through a worker at all: `PushTDataset.actions` is an in-memory
    tensor, already normalised in its `__init__`, and `get_frames` only indexes it.
    """

    def __init__(self, traj, eps, chunk):
        self.traj = traj
        self.items = []
        for ep in eps:
            T = int(traj.get_seq_length(int(ep)))
            for s in range(0, T, chunk):
                self.items.append((int(ep), s, min(s + chunk, T)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, s, e = self.items[i]
        obs, _, _, _ = self.traj.get_frames(ep, list(range(s, e)))
        return ep, s, obs["visual"].float()


class LatentCache:
    """Frozen per-frame latents + raw normalised actions for a set of episodes.

    Windows are enumerated exactly as `datasets.traj_dset.TrajSlicerDataset` does --
    `start in range(T - num_frames*frameskip + 1)` -- so the count is comparable to the
    dataset's own `len()`, which `--check` asserts.
    """

    def __init__(self, Z, A, base):
        self.Z, self.A, self.base = Z, A, base

    @property
    def n(self):
        return self.base.shape[0]

    def batch(self, idx, device):
        b = self.base[idx]                                   # (B,)
        z0 = self.Z[b + RAW0].to(device)
        zg = self.Z[b + RAWG].to(device)
        rows = b[:, None] + torch.arange(RAW0, RAWG)[None, :]     # (B,25)
        a = self.A[rows].reshape(len(idx), NUM_PRED, -1).to(device)  # (B,5,10)
        return z0, a, zg


def build_cache(model, traj, eps, device, workers=8, chunk=32, keep_first_visual=False):
    """Encode every frame of `eps` ONCE and index the windows into the result.

    Exact because the encoder is frozen and per-frame: `encode_obs` folds t into the
    batch, so a frame's code does not depend on the window it is read in. `check_cache`
    asserts that against the dataset's own items.
    """
    eps = [int(e) for e in eps]
    per_ep = {}
    dl = torch.utils.data.DataLoader(
        _FrameChunks(traj, eps, chunk),
        batch_size=None,
        num_workers=workers,
    )
    t0, nframes, keep = time.time(), 0, None
    for ep, s, visual in dl:
        with torch.no_grad():
            z = model.encode_obs_linked(
                {"visual": visual.to(device, non_blocking=True)[None]}
            )["visual"]                                        # (1, t, p, D)
        assert z.shape[2] == 1, f"expected one token per frame, got p={z.shape[2]}"
        per_ep.setdefault(int(ep), []).append((int(s), z[0, :, 0].float().cpu()))
        if keep_first_visual and int(ep) == eps[0] and int(s) == 0:
            keep = (int(ep), 0)              # (episode, its offset into the cache)
        nframes += visual.shape[0]
        if nframes % (200 * chunk) < chunk:
            print(
                f"  cached {nframes} frames  {nframes/(time.time()-t0):.0f} frame/s",
                flush=True,
            )
    Zs, As, bases, off = [], [], [], 0
    for ep in eps:
        parts = sorted(per_ep[ep], key=lambda t: t[0])
        Z = torch.cat([z for _, z in parts], 0)
        T = int(traj.get_seq_length(ep))
        assert Z.shape[0] == T, (Z.shape, T)
        Zs.append(Z)
        As.append(traj.actions[ep, :T].float())      # already normalised in the dataset
        n_win = T - WIN_RAW + 1
        if n_win > 0:
            bases.append(off + torch.arange(n_win))
        if keep is not None and keep[0] == ep:
            keep = (ep, off)
        off += T
    cache = LatentCache(torch.cat(Zs, 0), torch.cat(As, 0), torch.cat(bases, 0))
    print(
        f"cache: {len(Zs)} episodes  {nframes} frames  {cache.n} windows  "
        f"|z| RMS {float(cache.Z.pow(2).mean().sqrt()):.4f}  "
        f"{time.time()-t0:.0f}s",
        flush=True,
    )
    return cache, keep


def count_windows(traj):
    """sum_i max(0, T_i - 40 + 1): what TrajSlicerDataset would build."""
    return int(
        sum(max(0, int(traj.get_seq_length(i)) - WIN_RAW + 1) for i in range(len(traj)))
    )


# ------------------------------------------------------------------ negatives


def sample_negatives(a_pos, M, gen):
    """CEM's own proposal, `randn * sigma + mu` (planning/cem.py:99-106), var_scale=1.

    half  N(0, 1)                      -- opt step 0, mu = 0, sigma = var_scale = 1
    half  a+ + s*eps, s ~ U(0.1, 1.0)  -- later steps: mu near an elite, sigma shrunk
    """
    B, H, A = a_pos.shape
    n_far = (M + 1) // 2
    n_near = M - n_far
    far = torch.randn(B, n_far, H, A, generator=gen)
    s = torch.rand(B, n_near, 1, 1, generator=gen) * 0.9 + 0.1
    near = a_pos.detach().cpu()[:, None] + s * torch.randn(
        B, n_near, H, A, generator=gen
    )
    return torch.cat([far, near], dim=1).to(a_pos.device)


def candidates(a_pos, M, gen):
    """(B, 1+M, H, A) with the EXECUTED sequence at index 0."""
    return torch.cat([a_pos[:, None], sample_negatives(a_pos, M, gen)], dim=1)


def head_scores(head, z0, acts, zg):
    B, K = acts.shape[:2]
    z0f = z0[:, None].expand(B, K, z0.shape[-1]).reshape(B * K, -1)
    zgf = zg[:, None].expand(B, K, zg.shape[-1]).reshape(B * K, -1)
    return head(z0f, acts.reshape(B * K, *acts.shape[2:]), zgf).view(B, K)


def _spearman_rows(x, y):
    """mean over rows of Spearman(x_row, y_row); both (B,K)."""
    def rank(t):
        return t.argsort(dim=1).argsort(dim=1).float()
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(dim=1, keepdim=True)
    ry = ry - ry.mean(dim=1, keepdim=True)
    num = (rx * ry).sum(1)
    den = rx.pow(2).sum(1).sqrt() * ry.pow(2).sum(1).sqrt()
    return float((num / den.clamp_min(1e-12)).mean())


# ------------------------------------------------------------------ eval


@torch.no_grad()
def evaluate(head, model, cache, device, M, batches, batch, seed=1234, with_mse=True):
    head.eval()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(cache.n, generator=torch.Generator().manual_seed(seed))
    top1, top1_mse, rho, loss, n = [], [], [], [], 0
    for b in range(batches):
        idx = perm[b * batch : (b + 1) * batch]
        if len(idx) == 0:
            break
        z0, a_pos, zg = cache.batch(idx, device)
        acts = candidates(a_pos, M, g)
        E = head_scores(head, z0, acts, zg)
        top1.append(float((E.argmin(dim=1) == 0).float().mean()))
        loss.append(float(F.cross_entropy(-E, torch.zeros(len(idx), dtype=torch.long, device=device))))
        if with_mse:
            y = leaf_mse(model, z0, acts, zg)
            top1_mse.append(float((y.argmin(dim=1) == 0).float().mean()))
            rho.append(_spearman_rows(E, y))
        n += len(idx)
    head.train()
    out = dict(
        val_top1=float(np.mean(top1)), val_ce=float(np.mean(loss)), val_n=n
    )
    if with_mse:
        out["val_top1_mse"] = float(np.mean(top1_mse))
        out["val_rho_mse"] = float(np.mean(rho))
    return out


# ------------------------------------------------------------------ training


def train_one(
    mode, model, train_cache, val_cache, device, args, log
):
    torch.manual_seed(args.seed)
    head = SeqEnergyHead(
        emb_dim=int(model.encoder.emb_dim),
        horizon=NUM_PRED,
        act_dim=train_cache.A.shape[-1] * FRAMESKIP,
        hidden=args.hidden,
    ).to(device)
    with torch.no_grad():
        head.z_scale.fill_(float(train_cache.Z.pow(2).mean().sqrt()))
    print(
        f"[{mode}] head params {sum(p.numel() for p in head.parameters())} "
        f"z_scale {float(head.z_scale):.4f} init checksum {head.checksum():.6f}",
        flush=True,
    )
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1
    )
    gen = torch.Generator().manual_seed(args.seed + 1)          # negatives
    sampler = torch.Generator().manual_seed(args.seed + 2)      # window order
    trace, grad_ok = [], None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, train_cache.n, (args.batch,), generator=sampler)
        z0, a_pos, zg = train_cache.batch(idx, device)
        acts = candidates(a_pos, args.neg, gen)
        E = head_scores(head, z0, acts, zg)
        if mode == "rank":
            tgt = torch.zeros(len(idx), dtype=torch.long, device=device)
            loss = F.cross_entropy(-E, tgt)
        else:
            y = leaf_mse(model, z0, acts, zg)
            if step == 1:
                with torch.no_grad():
                    head.y_scale.fill_(max(float(y.pow(2).mean().sqrt()), 1e-12))
            loss = F.mse_loss(E, y / head.y_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            rep = head.grad_report()
            # The rank objective is a softmax over the 64 candidates of ONE anchor, and
            # the last bias adds the SAME constant to all of them: dL/db is exactly 0 by
            # shift-invariance. That is an identifiability fact about the objective, not
            # a wiring failure, and it is harmless because CEM only ever argsorts. It is
            # exempted EXPLICITLY rather than by loosening the threshold -- seed s6 hit
            # exactly 0.0 while the other seven got float dust ~4e-8 and slipped through,
            # so a tolerance here would have been a coin flip. The distill objective is a
            # regression, where the offset IS identified, so nothing is exempt there.
            exempt = {"mlp.%d.bias" % (len(head.mlp) - 1)} if mode == "rank" else set()
            dead = [
                n for n, gnorm in rep
                if (gnorm is None or gnorm == 0.0) and n not in exempt
            ]
            total = float(
                torch.norm(
                    torch.stack([p.grad.norm() for p in head.parameters() if p.grad is not None])
                )
            )
            print(
                f"[{mode}] GRAD LIVENESS after 1 backward: total_norm={total:.6e}"
                + (f"  (exempt by shift-invariance: {sorted(exempt)})" if exempt else "")
            )
            for n, gnorm in rep:
                print(f"    {n:28s} grad_norm={gnorm}")
            grad_ok = (not dead) and total > 0
            if args.assert_grad:
                assert grad_ok, f"gradient does not reach {dead} (total_norm={total})"
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            trace.append((step, float(loss)))
            print(
                f"[{mode}] step {step:6d}/{args.steps}  loss {float(loss):.5f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  {step/(time.time()-t0):.1f} it/s",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            m = evaluate(
                head, model, val_cache, device, args.neg, args.val_batches,
                args.val_batch, with_mse=not args.no_mse_eval,
            )
            print(f"[{mode}] step {step:6d} " + " ".join(f"{k}={v}" for k, v in m.items()), flush=True)
            log.setdefault(f"{mode}_val", []).append(dict(step=step, **m))
    final = evaluate(
        head, model, val_cache, device, args.neg, args.val_batches, args.val_batch,
        with_mse=not args.no_mse_eval,
    )
    log[f"{mode}_final"] = final
    log[f"{mode}_trace"] = trace
    log[f"{mode}_grad_live"] = bool(grad_ok)
    return head, final, trace


# ------------------------------------------------------------------ planner-level analysis


@torch.no_grad()
def elite_analysis(heads, model, cache, device, n_anchors=64, num_samples=300, topk=30,
                   seed=4242):
    """Reproduce ONE CEM optimisation step offline and compare the ELITES it selects.

    `planning/cem.py:99-119` draws `num_samples` candidates, scores them, and keeps
    `argsort(loss)[:topk]`. That set -- not the loss value -- is the whole content of the
    leaf metric. So the honest "does this arm differ from the control" question is: how
    much do the elite SETS overlap? 30/30 would mean the arm is the baseline under a
    different name; the V1-V3 silent-identity bug is exactly that failure.

    Reported per head: overlap with today's leaf metric (terminal latent MSE from this
    checkpoint's own rollout) and Spearman over all `num_samples` candidates; plus the
    head-vs-head overlap when both arms are given.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(cache.n, generator=torch.Generator().manual_seed(seed))[:n_anchors]
    z0, _, zg = cache.batch(idx, device)
    names = list(heads)
    ov = {n: [] for n in names}
    rho = {n: [] for n in names}
    pair, ov_mse_self = [], []
    for i in range(n_anchors):
        act = torch.randn(num_samples, NUM_PRED, cache.A.shape[-1] * FRAMESKIP, generator=g)
        act[0] = 0.0                       # cem.py: action[0] = mu, and mu = 0 at step 0
        act = act.to(device)
        z0i = z0[i][None].expand(num_samples, -1)
        zgi = zg[i][None].expand(num_samples, -1)
        y = leaf_mse(model, z0[i][None], act[None], zg[i][None])[0]      # (S,)
        e_mse = set(torch.argsort(y)[:topk].tolist())
        Es = {}
        for n in names:
            E = heads[n](z0i, act, zgi)
            Es[n] = E
            ov[n].append(len(e_mse & set(torch.argsort(E)[:topk].tolist())) / topk)
            rho[n].append(_spearman_rows(E[None], y[None]))
        if len(names) == 2:
            a = set(torch.argsort(Es[names[0]])[:topk].tolist())
            b = set(torch.argsort(Es[names[1]])[:topk].tolist())
            pair.append(len(a & b) / topk)
    out = {f"elite_overlap_with_mse[{n}]": float(np.mean(ov[n])) for n in names}
    out.update({f"spearman_with_mse[{n}]": float(np.mean(rho[n])) for n in names})
    if pair:
        out[f"elite_overlap[{names[0]} vs {names[1]}]"] = float(np.mean(pair))
    out["n_anchors"] = n_anchors
    out["num_samples"] = num_samples
    out["topk"] = topk
    return out


# ------------------------------------------------------------------ checks


@torch.no_grad()
def check_cache(model, traj, slices, cache, keep, device, k=4):
    """Assert the cache and the window arithmetic reproduce the DATASET's own items.

    Two things could silently be wrong and both would be invisible in the loss: the
    action block could be misaligned (so the head would rank a sequence the demo never
    executed), and the cached latent could belong to another frame. Both are checked
    against `TrajSlicerDataset.__getitem__` and against `model.rollout` on real images.
    """
    ep, off = keep
    T = int(traj.get_seq_length(ep))
    n_win = T - WIN_RAW + 1
    assert n_win > 0, f"check episode is shorter than one window (T={T})"
    starts = sorted({int(round(x)) for x in np.linspace(0, n_win - 1, k)})
    # (1) actions: the dataset's own concat for the same (episode, start)
    worst_a, worst_z, worst_r = 0.0, 0.0, 0.0
    for s in starts:
        obs, act, _, _ = traj.get_frames(ep, list(range(s, s + WIN_RAW)))
        a_ds = rearrange(act.float(), "(n f) d -> n (f d)", n=NUM_HIST + NUM_PRED)[I0:IG]
        b = torch.tensor([off + s])
        z0, a_mine, zg = cache.batch(b, device)
        worst_a = max(worst_a, float((a_ds.to(device) - a_mine[0]).abs().max()))
        # (2) latents: the dataset's own frames, encoded here
        v = obs["visual"].float()[:: FRAMESKIP].to(device)
        z_ds = model.encode_obs_linked({"visual": v[None]})["visual"][0, :, 0]
        worst_z = max(
            worst_z,
            float((z_ds[I0] - z0[0]).abs().max()),
            float((z_ds[IG] - zg[0]).abs().max()),
        )
        # (3) the cached-latent rollout equals the image rollout the planner runs
        obs_0 = {"visual": v[I0][None, None]}
        z_obses, _ = model.rollout(obs_0=obs_0, act=a_mine, z_goal=None)
        z_ref = z_obses["visual"][:, -1, 0]
        z_mine = rollout_terminal_from_z(model, z0, a_mine)
        worst_r = max(worst_r, float((z_ref - z_mine).abs().max()))
    zrms = float(cache.Z.pow(2).mean().sqrt())
    print(
        f"CHECK  max|a_dataset - a_cache| = {worst_a:.3e}   "
        f"max|z_dataset - z_cache| = {worst_z:.3e} ({worst_z/zrms:.2e} of |z| RMS "
        f"{zrms:.4f})   max|rollout(images) - rollout(cache)| = {worst_r:.3e} "
        f"({worst_r/zrms:.2e})",
        flush=True,
    )
    assert worst_a < 1e-6, "action block is misaligned with TrajSlicerDataset"
    # RELATIVE, not absolute: the cache encodes 32-frame chunks and this check encodes 8,
    # and cuBLAS picks a different GEMM for a different batch shape, so two float32 GPU
    # encodings of the SAME frame differ in the 4th decimal. On CPU this is exactly 0.
    assert worst_z / zrms < 5e-3, "cached latent is not the frame's latent"
    assert worst_r / zrms < 5e-3, "cached-latent rollout != image rollout"
    print(
        f"CHECK  windows: enumerated {count_windows(traj)} == len(TrajSlicerDataset) "
        f"{len(slices)} -> {count_windows(traj) == len(slices)}",
        flush=True,
    )
    assert count_windows(traj) == len(slices)


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir under $CKPT_BASE/outputs")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--ckpt-base", default=os.environ.get("CKPT_BASE", os.path.join(REPO, "runs")))
    ap.add_argument("--out", required=True, help="output .pt, or a PREFIX with --mode both")
    ap.add_argument("--mode", default="both", choices=["rank", "distill", "both"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--neg", type=int, default=63)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=3000, help="train episodes to cache")
    ap.add_argument("--val-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--enc-chunk", type=int, default=32,
                    help="frames per dataloader item AND per encoder forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=6)
    ap.add_argument("--val-batch", type=int, default=128)
    ap.add_argument("--no-mse-eval", action="store_true", help="skip the rollout-based val metrics")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--analyze", default=None,
                    help="comma-separated head .pt files: skip training, report the "
                         "CEM elite-set overlap of each head against today's leaf "
                         "metric and against each other")
    ap.add_argument("--no-assert-grad", dest="assert_grad", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"=== train_energy: {args.run} mode={args.mode} device={device} ===", flush=True)
    model, cfg = load_frozen(args.run, args.epoch, args.ckpt_base, device)

    # The dataset the head is trained on is the PLANNER's geometry (num_pred=5), not the
    # checkpoint's training geometry (num_pred=1): the head must rank 5-step sequences.
    slices, traj = hydra.utils.call(
        cfg.env.dataset, num_hist=NUM_HIST, num_pred=NUM_PRED, frameskip=FRAMESKIP
    )
    print(
        f"windows (num_hist={NUM_HIST} num_pred={NUM_PRED} frameskip={FRAMESKIP}): "
        f"train {len(slices['train'])}  val {len(slices['valid'])}",
        flush=True,
    )

    if args.analyze:
        heads = {}
        for f in args.analyze.split(","):
            h = load_energy_head(f.strip(), device=device, expect_horizon=NUM_PRED)
            heads[torch.load(f.strip(), map_location="cpu").get("mode", os.path.basename(f))] = h
        vcache, keep = build_cache(
            model, traj["valid"], list(range(len(traj["valid"]))), device,
            args.workers, args.enc_chunk, keep_first_visual=True,
        )
        if not args.no_check:
            check_cache(model, traj["valid"], slices["valid"], vcache, keep, device)
        res = elite_analysis(heads, model, vcache, device)
        for k, v in res.items():
            print(f"ELITE {k} = {v}", flush=True)
        for name, h in heads.items():
            m = evaluate(h, model, vcache, device, args.neg, args.val_batches,
                         args.val_batch, with_mse=not args.no_mse_eval)
            res[f"val[{name}]"] = m
            print(f"VAL   {name} " + " ".join(f"{k}={v}" for k, v in m.items()), flush=True)
        out = args.out if args.out.endswith(".json") else args.out + "_elite.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        print(f"wrote {out}", flush=True)
        return

    n_tr = len(traj["train"])
    g = torch.Generator().manual_seed(0)          # SAME episode subset for every seed
    eps_tr = torch.randperm(n_tr, generator=g)[: args.episodes or n_tr].tolist()
    n_va = len(traj["valid"])
    eps_va = list(range(args.val_episodes or n_va))
    print(f"caching {len(eps_tr)}/{n_tr} train episodes, {len(eps_va)}/{n_va} val episodes", flush=True)

    val_cache, keep = build_cache(
        model, traj["valid"], eps_va, device, args.workers, args.enc_chunk,
        keep_first_visual=True,
    )
    if not args.no_check:
        check_cache(model, traj["valid"], slices["valid"], val_cache, keep, device)
    del keep
    train_cache, _ = build_cache(
        model, traj["train"], eps_tr, device, args.workers, args.enc_chunk
    )

    log = dict(
        run=args.run, epoch=args.epoch, mode=args.mode, steps=args.steps,
        neg=args.neg, batch=args.batch, lr=args.lr, seed=args.seed,
        episodes=len(eps_tr), train_windows_cached=train_cache.n,
        val_windows_cached=val_cache.n,
        dataset_train_windows=len(slices["train"]),
        dataset_val_windows=len(slices["valid"]),
    )
    modes = ["rank", "distill"] if args.mode == "both" else [args.mode]
    heads, outs = {}, {}
    for mode in modes:
        head, final, trace = train_one(mode, model, train_cache, val_cache, device, args, log)
        out = args.out if args.mode != "both" else f"{args.out}_{mode}.pt"
        if not out.endswith(".pt"):
            out = out + ".pt"
        save_energy_head(
            head, out, run=args.run, epoch=str(args.epoch), step=args.steps,
            mode=mode, val_top1=final["val_top1"], meta=json.dumps(
                {k: v for k, v in log.items() if not k.endswith("_trace")}
            ),
        )
        # (b) of the new-head checklist: the parameter is SHOWN PRESENT in the file that
        # was just written, by loading it back strictly and matching the checksum.
        back = load_energy_head(out, device=device)
        assert state_dict_checksum(back.state_dict()) == state_dict_checksum(head.state_dict())
        heads[mode], outs[mode] = back, out

    if len(heads) == 2:
        # ARMS DIFFER, at the level the planner consumes: the two heads must not induce
        # the same ORDER on the same CEM-distributed candidates.
        g = torch.Generator().manual_seed(999)
        idx = torch.randperm(val_cache.n, generator=torch.Generator().manual_seed(7))[:256]
        z0, a_pos, zg = val_cache.batch(idx, device)
        acts = candidates(a_pos, args.neg, g)
        with torch.no_grad():
            Er = head_scores(heads["rank"], z0, acts, zg)
            Ed = head_scores(heads["distill"], z0, acts, zg)
        agree = float((Er.argmin(1) == Ed.argmin(1)).float().mean())
        rho = _spearman_rows(Er, Ed)
        log["arms_differ"] = dict(
            spearman_rank_vs_distill=rho, top1_agreement=agree,
            checksum_rank=heads["rank"].checksum(),
            checksum_distill=heads["distill"].checksum(),
        )
        print(
            f"ARMS DIFFER: spearman(E_rank, E_distill) = {rho:.4f}   "
            f"top-1 agreement = {agree:.4f}   "
            f"checksums {heads['rank'].checksum():.6f} vs {heads['distill'].checksum():.6f}",
            flush=True,
        )
        assert rho < 0.999 and heads["rank"].checksum() != heads["distill"].checksum()

    logf = (args.out if args.mode != "both" else args.out + "_log") + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(logf)) or ".", exist_ok=True)
    json.dump(log, open(logf, "w"), indent=1)
    print(f"wrote {logf}", flush=True)
    for mode in modes:
        print(f"RESULT {args.run} {mode} -> {outs[mode]}  " +
              " ".join(f"{k}={v}" for k, v in log[f"{mode}_final"].items()), flush=True)


if __name__ == "__main__":
    main()
