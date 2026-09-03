"""M1 -- ridge-decode the block pose from the FROZEN latent of every archived checkpoint.

THE QUESTION.  Spearman(the latent distance CEM minimises, the TRUE task distance) = +0.398.
That is produced either by bad dynamics or by a latent that never encoded the block, and the
one-step contrast cannot separate them (`LpWM-ltv` rel_mse 0.0092 / CEM 0.357 vs
`LpWM-linvar` 0.0095 / CEM 0.080 -- indistinguishable prediction error, 4.5x planning).
Decodability of the block pose from a SINGLE FROZEN FRAME needs no dynamics at all, so it is
the missing axis.  If the held-out median angular error is above 20 deg on every checkpoint,
no predictor can plan this task and the representation is the whole story.

WHAT IS REPORTED, AND WHAT IS NOT.  The headline is the MEDIAN ABSOLUTE ANGULAR ERROR IN
DEGREES on held-out episodes.  Never R^2 on raw theta: theta is circular, so an R^2 on it is
dominated by the wrap discontinuity and is meaningless.  The probe regresses
(block_x, block_y, cos theta, sin theta) and the angle is recovered with atan2.

THE THREE CONTROLS, AND WHY THE THIRD IS THE ONE THAT MATTERS.
  ctrl_shuf   labels permuted inside the fit set.  READ THIS AGAINST `const_err_ang_deg`,
              NOT against 90 deg.  90 deg is chance for a UNIFORM wrapped angle and the
              spec's LIVE criterion assumes it, but the block angle in this dataset is
              strongly non-uniform -- the T settles into a canonical orientation, 314 of
              636 val frames sit in one 30-deg bin -- so the best possible CONSTANT
              prediction already scores 15.3 deg on val and 19.0 deg on the held-out train
              episodes.  `const_err_ang_deg` reports that bound, and `ctrl_shuf` should
              land on it (a shuffled-label ridge picks the largest lambda and degenerates
              to the mean).  It also means the spec's 20-deg falsifier threshold is BELOW
              the trivial baseline and cannot be read on its own.
  ctrl_rand   a freshly built, never-loaded encoder from the SAME `cfg.encoder` (and a fresh
              link).  Capacity-matched: a random ViT is already a rich nonlinear feature map,
              so "a linear probe decodes it" is not evidence of learning by itself.
  ctrl_agent  block pose from the AGENT's own state [ax, ay, vx, vy] alone.  The agent
              carries 90.9% of the initial pos_diff^2 and is usually touching the block, so
              without this control "z knows the block" is unfalsifiable.  Reported once (it
              does not depend on any checkpoint) and mirrored onto every row.
  ctrl_resid  z residualised on [ax, ay, vx, vy, 1] -- what the latent knows about the block
              OVER AND ABOVE the agent configuration.

PRE-LINK.  `encode_obs` is probed as well as `encode_obs_linked`, which separates "the link
destroys it" from "the encoder never had it".  The planner and every rollout live in the
LINKED space, so `err_ang_deg` (post-link) is the number that bears on planning; the pre-link
number is diagnostic.

SPLIT UNIT IS THE EPISODE.  Grouping is on `(split, ep)` for the lambda cross-validation and
for the bootstrap.  Frames within an episode are near-duplicates.

PRIVILEGED STATE.  This file reads the block pose out of `states.pth`.  That is legal here --
M1 is measurement.  Nothing it produces may enter a training loss.

Usage
    python analysis/latent_probe.py --cache runs/probe_cache.pt --out probe.json \
        --campaign campaign_fixed.json [--shard 0 --nshards 6]
    python analysis/latent_probe.py --cache runs/probe_cache.pt --out probe_ens.json \
        --ensemble LpWM-ltv_pd384_bf16_s3,...,LpWM-ltv_pd384_bf16_s7
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra                                                   # noqa: E402
from plan import load_model                                    # noqa: E402

RUN_RE = re.compile(r"(.+)_pd\d+_\w+_s(\d+)$")                 # analysis/spectral.py:42
CAMPAIGN_ALIAS = {"PiWM-columns": "PiWM-columns_patch"}        # analysis/spectral.py:43

# state columns, from datasets/pusht_dset.py (states.pth ++ velocities.pth)
AX, AY, BX, BY, TH, VX, VY = range(7)


# --------------------------------------------------------------------------------------
# ridge
# --------------------------------------------------------------------------------------
def group_kfold(groups, k=5, seed=0):
    """Folds whose unit is the GROUP (an episode), not the row.

    sklearn's GroupKFold is deterministic-by-size; a seeded round robin over shuffled
    groups is the same idea with an explicit seed, and keeps this file dependency-light.
    """
    uniq = np.unique(groups)
    rs = np.random.RandomState(seed)
    order = rs.permutation(len(uniq))
    assign = {uniq[g]: i % k for i, g in enumerate(order)}
    fold_of = np.array([assign[g] for g in groups])
    return [(fold_of != f, fold_of == f) for f in range(k)]


def _eig_solver(Xtr, Ytr):
    """One eigendecomposition that serves EVERY ridge lambda.

    Primal when p <= n (G = X'X, w = V diag(1/(s+lam)) V' X'Y), dual otherwise
    (K = XX', a = V diag(1/(s+lam)) V' Y, and the primal weights are X'a).  Both give the
    identical estimator; the switch is purely about which Gram matrix is smaller.  A patch
    checkpoint has p = 98688 features against n ~ 8200 fit frames, which is why the dual
    path exists at all -- and why its Gram is accumulated in float32: a float64 copy of that
    feature matrix is 6.5 GB and the naive form materialises one per lambda.
    """
    n, p = Xtr.shape
    dual = p > n
    X = Xtr.double() if Xtr.numel() <= 10 ** 8 else Xtr      # exact when it is affordable
    G = ((X @ X.T) if dual else (X.T @ X)).double()
    s, V = torch.linalg.eigh(0.5 * (G + G.T))
    s = torch.clamp(s, min=0.0)
    if dual:
        return dict(dual=True, s=s, V=V, rhs=V.T @ Ytr.double(), Xtr=X)
    return dict(dual=False, s=s, V=V, rhs=V.T @ (X.T.double() @ Ytr.double()), Xtr=X)


def _coef(sol, lam):
    return sol["V"] @ (sol["rhs"] / (sol["s"] + lam).unsqueeze(1))


def _predict_many(sol, lams, Xte):
    """Ridge predictions on Xte for every lambda, from one cached eigendecomposition.

    The dual cross-Gram Xte X' is the expensive term and does not depend on lambda, so it is
    formed once; that is what makes a 13-point lambda grid free on the patch checkpoints.
    """
    if sol["dual"]:
        K = (Xte.to(sol["Xtr"].dtype) @ sol["Xtr"].T).double()
        return [K @ _coef(sol, lam) for lam in lams]
    Xd = Xte.double()
    return [Xd @ _coef(sol, lam) for lam in lams]


def _predict(sol, lam, Xte):
    return _predict_many(sol, [lam], Xte)[0]


def _weights(sol, lam):
    """Primal weight matrix (p, k) at `lam` -- what M3's oracle objective needs."""
    c = _coef(sol, lam)
    if sol["dual"]:
        return (sol["Xtr"].T @ c.to(sol["Xtr"].dtype)).double()
    return c


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def errors(pred, state):
    """(per-frame position error in px, per-frame angular error in deg).

    pred columns are [block_x, block_y, cos theta, sin theta].
    """
    pos = np.linalg.norm(pred[:, :2] - state[:, [BX, BY]], axis=1)
    ang = np.degrees(np.abs(_wrap(np.arctan2(pred[:, 3], pred[:, 2]) - state[:, TH])))
    return pos, ang


def _median_se(values, groups, n_boot=200, seed=0):
    med = float(np.median(values))
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return med, float("nan")
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rs = np.random.RandomState(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([idx[g] for g in rs.choice(uniq, len(uniq), replace=True)])
        draws[b] = np.median(values[sel])
    return med, float(np.std(draws))


# --------------------------------------------------------------------------------------
# one probe
# --------------------------------------------------------------------------------------
class ProbeData:
    """Targets, groups and role masks -- everything that does not depend on a checkpoint."""

    def __init__(self, cache, device):
        st = cache["state"].numpy().astype(np.float64)
        self.state = st
        self.Y = np.stack([st[:, BX], st[:, BY], np.cos(st[:, TH]), np.sin(st[:, TH])], 1)
        self.agent = st[:, [AX, AY, VX, VY]]
        self.groups = np.array([f"{s}:{e}" for s, e in zip(cache["split"], cache["ep"])])
        self.role = np.asarray(cache["role"])
        self.fit = self.role == "fit"
        self.eval_roles = [r for r in ("val", "heldout") if (self.role == r).any()]
        self.device = device
        self.Yt = torch.as_tensor(self.Y, device=device, dtype=torch.float64)


def run_probe(F, pd_, lams, shuffle=False, seed=0, want_weights=False):
    """Fit ridge F -> (bx, by, cos, sin) on the `fit` rows; score every eval role.

    F: (N, p) float32 tensor on the probe device.  Standardisation uses the FIT rows only.
    Returns (metrics dict, extras dict).
    """
    dev = F.device
    tr = torch.as_tensor(pd_.fit, device=dev)
    mu = F[tr].mean(0, keepdim=True)
    sd_raw = F[tr].std(0, keepdim=True)
    # A `reprelu` link emits exact zeros, so a unit can be CONSTANT on the fit rows and
    # nonzero on an eval row.  Dividing by a clamped 1e-6 would then hand the probe a
    # feature of magnitude 1e6 that it never saw in training -- a silent blow-up that looks
    # like the representation failing.  Zero those columns out instead (x / inf == 0).
    dead = sd_raw < 1e-6
    Fc = (F - mu) / sd_raw.masked_fill(dead, float("inf"))

    Y = pd_.Yt
    ymu = Y[tr].mean(0, keepdim=True)
    Yc = Y - ymu

    Ytr = Yc[tr]
    if shuffle:                                          # ctrl_shuf: kill the z <-> y link
        g = torch.Generator(device="cpu").manual_seed(seed)
        Ytr = Ytr[torch.randperm(Ytr.shape[0], generator=g).to(dev)]

    # ---- lambda by GroupKFold(5) grouped on episode, scored on standardised targets
    gtr = pd_.groups[pd_.fit]
    Xtr = Fc[tr]
    ysd = Ytr.std(0, keepdim=True).clamp_min(1e-12)
    cv = np.zeros(len(lams))
    for tr_m, te_m in group_kfold(gtr, k=5, seed=seed):
        a = torch.as_tensor(tr_m, device=dev)
        b = torch.as_tensor(te_m, device=dev)
        sol = _eig_solver(Xtr[a], Ytr[a])
        for j, p in enumerate(_predict_many(sol, lams, Xtr[b])):
            cv[j] += float((((p - Ytr[b]) / ysd) ** 2).mean())
    j = int(np.argmin(cv))
    lam = float(lams[j])

    sol = _eig_solver(Xtr, Ytr)
    out = {"lam": lam, "lam_at_boundary": bool(j in (0, len(lams) - 1)),
           "n_feat": int(F.shape[1]), "n_dead": int(dead.sum()), "n_fit": int(tr.sum())}
    for r in pd_.eval_roles:
        m = torch.as_tensor(pd_.role == r, device=dev)
        pred = (_predict(sol, lam, Fc[m]) + ymu).cpu().numpy()
        pos, ang = errors(pred, pd_.state[pd_.role == r])
        g = pd_.groups[pd_.role == r]
        sfx = "" if r == "val" else f"_{r}"
        out[f"err_pos_px{sfx}"], out[f"err_pos_se{sfx}"] = _median_se(pos, g, seed=seed)
        out[f"err_ang_deg{sfx}"], out[f"err_ang_se{sfx}"] = _median_se(ang, g, seed=seed)
    extras = {}
    if want_weights:
        extras = dict(W=_weights(sol, lam).cpu(), mu=mu.cpu(), sd=sd.cpu(), ymu=ymu.cpu())
    return out, extras


def const_baseline(pd_):
    """Best achievable by a CONSTANT prediction, per eval role.  Model-free.

    The floor any probe number has to be read against: see the note on ctrl_shuf above.
    """
    out = {}
    grid = np.linspace(-np.pi, np.pi, 721)
    for r in pd_.eval_roles:
        st = pd_.state[pd_.role == r]
        sfx = "" if r == "val" else f"_{r}"
        out[f"const_err_ang_deg{sfx}"] = float(np.degrees(min(
            np.median(np.abs(_wrap(g - st[:, TH]))) for g in grid)))
        c = np.median(st[:, [BX, BY]], axis=0)
        out[f"const_err_pos_px{sfx}"] = float(np.median(
            np.linalg.norm(st[:, [BX, BY]] - c, axis=1)))
    return out


def probe_agent_only(pd_, lams):
    """ctrl_agent: block pose from [ax, ay, vx, vy] alone.  No checkpoint involved.

    Reported twice.  `ctrl_agent_*` is the linear probe the spec asks for.
    `ctrl_agent2_*` adds squares and the ax*ay cross term, because the latent is a
    NONLINEAR function of the frame and a purely linear agent control would hand the
    latent an unfair functional-form advantage.  The stronger control is the honest null.
    """
    out = {}
    lin = torch.as_tensor(pd_.agent, device=pd_.device, dtype=torch.float32)
    quad = torch.as_tensor(np.concatenate([pd_.agent, pd_.agent ** 2,
                                           pd_.agent[:, :1] * pd_.agent[:, 1:2]], 1),
                           device=pd_.device, dtype=torch.float32)
    for tag, F in (("ctrl_agent", lin), ("ctrl_agent2", quad)):
        m, _ = run_probe(F, pd_, lams)
        out.update({f"{tag}_{k}": v for k, v in m.items()
                    if k.startswith("err_") or k == "lam"})
    return out


# --------------------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encode(model, cache, device, batch=128):
    """(pre-link u, post-link z) for every cached frame, each (N, p*d) float32.

    Goes through `VWorldModel.encode_obs` so the encoder_transform, the block-causal
    branch and the adaln branch are all exercised exactly as at plan time; the link is
    then applied by `_link`, which is what `encode_obs_linked` does.
    """
    from datasets.pusht_dset import PROPRIO_MEAN, PROPRIO_STD
    vis = cache["visual"]
    st = cache["state"].float()
    prop = (st[:, [AX, AY, VX, VY]] - PROPRIO_MEAN) / PROPRIO_STD
    us, zs = [], []
    for i in range(0, vis.shape[0], batch):
        xb = vis[i:i + batch].to(device).float().unsqueeze(1)            # (b, 1, 3, H, W)
        pb = prop[i:i + batch].to(device).unsqueeze(1)                   # (b, 1, 4)
        u = model.encode_obs({"visual": xb, "proprio": pb})["visual"]    # (b, 1, p, d)
        z = model._link(u)
        us.append(u[:, 0].flatten(1).float())
        zs.append(z[:, 0].flatten(1).float())
    return torch.cat(us), torch.cat(zs)


class _FreshModel(torch.nn.Module):
    """A never-loaded encoder + link of the same config -- the capacity-matched control."""

    def __init__(self, cfg, device):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
        lc = cfg.get("link", None)
        self.link = (hydra.utils.instantiate(lc).to(device).eval()
                     if lc is not None and lc.get("_target_", None) is not None else None)
        self.action_conditioning = cfg.get("action_conditioning", "concat")
        self.encoder_transform = torch.nn.Identity()

    def _link(self, x):
        return self.link(x) if self.link is not None else x

    def encode_obs(self, obs):
        from einops import rearrange
        v = obs["visual"]
        b, t = v.shape[0], v.shape[1]
        v = rearrange(v, "b t ... -> (b t) ...")
        enc = getattr(self.encoder, "module", self.encoder)
        e = (enc.forward_temporal(v, t) if getattr(enc, "block_causal", False)
             else self.encoder.forward(v))
        return {"visual": rearrange(e, "(b t) p d -> b t p d", b=b)}


# --------------------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------------------
def _dump(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)


def _lam_grid(spec):
    lo, hi = (int(x) for x in spec.split(":"))
    return list(10.0 ** np.arange(lo, hi))


def _runs(repo, arm, limit, shard, nshards, exclude=None, only=None):
    out = []
    for d in sorted(glob.glob(os.path.join(repo, "runs/outputs/*/"))):
        name = os.path.basename(d.rstrip("/"))
        m = RUN_RE.match(name)
        ck = os.path.join(d, "checkpoints", "model_latest.pth")
        cf = os.path.join(d, "hydra.yaml")
        if not m or not os.path.exists(ck) or not os.path.exists(cf):
            continue
        if only and name not in only:
            continue
        if arm and arm not in m.group(1):
            continue
        if exclude and any(x and x in name for x in exclude):
            continue
        out.append((name, m.group(1), int(m.group(2)), ck, cf))
    out = [r for i, r in enumerate(out) if i % nshards == shard]
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache", default="runs/probe_cache.pt")
    ap.add_argument("--out", default="assets/latent_probe.json")
    ap.add_argument("--campaign", default=None, help="campaign JSON, to join CEM in")
    ap.add_argument("--arm", default=None, help="substring filter on the arm name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run", default=None,
                    help="comma list of EXACT run dir names (--arm is a substring match, "
                         "so `--arm LpWM-ltv` also picks up LpWM-ltv-d2048 etc.)")
    ap.add_argument("--exclude", default=None,
                    help="comma list of substrings; matching run dirs are skipped. The "
                         "20 `patch` runs cost ~15x a cls run (98688 features -> the dual "
                         "ridge path), so they are normally given their own job.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lam-grid", default="-3:10",
                    help="10**arange(lo,hi).  Wider than the spec's -3:6 because a "
                         "98688-feature patch run selects lambda at the top of that grid, "
                         "and a boundary-selected lambda is a silent failure "
                         "(lam_at_boundary is reported for exactly this reason).")
    ap.add_argument("--ensemble", default=None,
                    help="comma list of run dirs; probes their CONCATENATED standardised z")
    ap.add_argument("--probe-out", default=None,
                    help="also write the fitted ridge (W, mu, sd, ymu) here, for M3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--merge", default=None,
                    help="glob of shard jsons to concatenate and summarise; no GPU needed")
    a = ap.parse_args()

    if a.merge:
        rows, seen = [], set()
        for f in sorted(glob.glob(a.merge)):
            for r in json.load(open(f)):
                if r["run"] in seen:
                    continue
                seen.add(r["run"])
                rows.append(r)
        _dump(rows, a.out)
        print(f"merged {len(rows)} rows -> {a.out}")
        summarise(rows)
        return

    lams = _lam_grid(a.lam_grid)
    cache = torch.load(a.cache)
    pd_ = ProbeData(cache, a.device)
    print(f"cache {a.cache}: {cache['visual'].shape[0]} frames, "
          f"fit={int(pd_.fit.sum())} roles={pd_.eval_roles} "
          f"episodes={len(np.unique(pd_.groups))}", flush=True)

    const = const_baseline(pd_)
    print(f"const baseline (model-free floor): err_ang={const['const_err_ang_deg']:.2f} deg  "
          f"err_pos={const['const_err_pos_px']:.2f} px   "
          f"(uniform-angle chance would be 90.00 deg -- this angle is NOT uniform)",
          flush=True)
    agent = probe_agent_only(pd_, lams)
    print(f"ctrl_agent (no checkpoint): err_ang={agent['ctrl_agent_err_ang_deg']:.2f} deg  "
          f"err_pos={agent['ctrl_agent_err_pos_px']:.2f} px", flush=True)

    cem = json.load(open(a.campaign))["arms"] if a.campaign else {}
    rand_cache = {}
    rows = []

    def one(name, arm, seed, ck, cf, extra_members=()):
        cfg = OmegaConf.load(cf)
        # NB VWorldModel.eval() (models/visual_world_model.py:164) returns None instead of
        # self, so the idiomatic `load_model(...).eval()` silently yields None.
        model = load_model(ck, cfg, cfg.num_action_repeat, device=a.device)
        model.eval()
        U, Z = encode(model, cache, a.device, a.batch)
        if extra_members:                              # ensemble: standardise then concat
            parts = [Z]
            for mck, mcf in extra_members:
                mcfg = OmegaConf.load(mcf)
                mm = load_model(mck, mcfg, mcfg.num_action_repeat, device=a.device)
                mm.eval()
                parts.append(encode(mm, cache, a.device, a.batch)[1])
                del mm
                torch.cuda.empty_cache()
            tr = torch.as_tensor(pd_.fit, device=a.device)
            parts = [(p - p[tr].mean(0, keepdim=True)) / p[tr].std(0, keepdim=True).clamp_min(1e-6)
                     for p in parts]
            Z = torch.cat(parts, dim=1)

        row = dict(run=name, arm=arm, seed=seed,
                   cem=cem.get(CAMPAIGN_ALIAS.get(arm, arm), {}).get(str(seed)))
        main_m, extras = run_probe(Z, pd_, lams, want_weights=a.probe_out is not None)
        row.update(main_m)
        pre, _ = run_probe(U, pd_, lams)
        row.update({f"prelink_{k}": v for k, v in pre.items()})
        del U                       # 5.4 GB on a patch run; the fresh control is next
        torch.cuda.empty_cache()
        shuf, _ = run_probe(Z, pd_, lams, shuffle=True)
        row.update({f"ctrl_shuf_{k}": v for k, v in shuf.items()})

        # ctrl_resid: what z adds ON TOP of the agent configuration
        A = torch.as_tensor(np.concatenate([pd_.agent, np.ones((len(pd_.agent), 1))], 1),
                            device=a.device, dtype=torch.float64)
        tr = torch.as_tensor(pd_.fit, device=a.device)
        beta = torch.linalg.lstsq(A[tr], Z[tr].double()).solution      # (5, p), tiny
        resid = Z - (A.float() @ beta.float())        # float32: a f64 copy of Z is 10.8 GB
        res, _ = run_probe(resid, pd_, lams)
        row.update({f"ctrl_resid_{k}": v for k, v in res.items()})
        del resid
        torch.cuda.empty_cache()

        key = (str(cfg.encoder), str(cfg.get("link", None)))
        if key not in rand_cache:
            fm = _FreshModel(cfg, a.device)
            Ur, Zr = encode(fm, cache, a.device, a.batch)
            rnd, _ = run_probe(Zr, pd_, lams)
            rnd_pre, _ = run_probe(Ur, pd_, lams)
            rand_cache[key] = ({f"ctrl_rand_{k}": v for k, v in rnd.items()},
                               {f"ctrl_rand_prelink_{k}": v for k, v in rnd_pre.items()})
            del fm, Ur, Zr
            torch.cuda.empty_cache()
        row.update(rand_cache[key][0])
        row.update(rand_cache[key][1])
        row.update(agent)
        row.update(const)
        del model, Z
        torch.cuda.empty_cache()
        return row, extras

    if a.ensemble:
        names = [n for n in a.ensemble.split(",") if n]
        specs = [(n, os.path.join(a.repo, "runs/outputs", n, "checkpoints", "model_latest.pth"),
                  os.path.join(a.repo, "runs/outputs", n, "hydra.yaml")) for n in names]
        m = RUN_RE.match(names[0])
        row, extras = one("ENSEMBLE:" + ",".join(names), "ENSEMBLE-" + m.group(1),
                          int(m.group(2)), specs[0][1], specs[0][2],
                          extra_members=[(c, f) for _, c, f in specs[1:]])
        rows.append(row)
        print(f"  ENSEMBLE({len(names)}) err_ang={row['err_ang_deg']:.2f} "
              f"err_pos={row['err_pos_px']:.2f} n_feat={row['n_feat']}", flush=True)
    else:
        todo = _runs(a.repo, a.arm, a.limit, a.shard, a.nshards,
                     exclude=(a.exclude.split(",") if a.exclude else None),
                     only=(set(a.run.split(",")) if a.run else None))
        print(f"shard {a.shard}/{a.nshards}: {len(todo)} runs", flush=True)
        for name, arm, seed, ck, cf in todo:
            try:
                row, extras = one(name, arm, seed, ck, cf)
            except Exception as e:                     # one broken run must not stop a sweep
                print(f"  {name:46s} SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(row)
            _dump(rows, a.out)          # a walltime kill must not lose completed rows
            print(f"  {name:46s} ang={row['err_ang_deg']:6.2f}+-{row['err_ang_se']:.2f} "
                  f"pos={row['err_pos_px']:6.2f}  pre={row['prelink_err_ang_deg']:6.2f}  "
                  f"rand={row['ctrl_rand_err_ang_deg']:6.2f}  "
                  f"shuf={row['ctrl_shuf_err_ang_deg']:6.2f}  "
                  f"resid={row['ctrl_resid_err_ang_deg']:6.2f}  lam={row['lam']:.0e}",
                  flush=True)
            if a.probe_out and extras:
                torch.save(dict(run=name, **{k: v for k, v in extras.items()}),
                           a.probe_out)
                print(f"  wrote probe weights -> {a.probe_out}", flush=True)

    _dump(rows, a.out)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")

    summarise(rows)


def summarise(rows):
    if rows:
        ang = np.array([r["err_ang_deg"] for r in rows])
        print(f"  err_ang_deg   median {np.median(ang):.2f}  "
              f"[{ang.min():.2f}, {ang.max():.2f}]   frac > 20 deg: {(ang > 20).mean():.2f}")
        j = [(r["err_ang_deg"], r["err_pos_px"], r["cem"]) for r in rows if r["cem"] is not None]
        if len(j) >= 8:
            from scipy.stats import spearmanr
            A_, P_, C_ = map(np.array, zip(*j))
            ra, pa = spearmanr(A_, C_)
            rp, pp = spearmanr(P_, C_)
            print(f"  spearman(err_ang, CEM) = {ra:+.3f} (p={pa:.3g}, n={len(j)})")
            print(f"  spearman(err_pos, CEM) = {rp:+.3f} (p={pp:.3g}, n={len(j)})")
        for k in ("prelink_err_ang_deg", "ctrl_rand_err_ang_deg", "ctrl_shuf_err_ang_deg",
                  "ctrl_resid_err_ang_deg", "ctrl_agent_err_ang_deg",
                  "ctrl_agent2_err_ang_deg"):
            v = np.array([r[k] for r in rows if k in r])
            if v.size:
                print(f"  {k:28s} median {np.median(v):6.2f} deg")
        best = min(rows, key=lambda r: r["err_ang_deg"])
        floor = best.get("const_err_ang_deg", float("nan"))
        print(f"  FALSIFIER: min err_ang_deg over {len(rows)} checkpoints = "
              f"{best['err_ang_deg']:.2f} deg ({best['run']}); "
              f"{'NOT ' if best['err_ang_deg'] <= 20 else ''}above the spec's 20 deg "
              f"threshold -- but the model-free constant floor is {floor:.2f} deg, so read "
              f"the trained probe against ctrl_shuf/ctrl_rand/ctrl_agent, not against 20.")
        n_beat = sum(r["err_ang_deg"] < r.get("ctrl_shuf_err_ang_deg", np.inf) for r in rows)
        print(f"  checkpoints beating their own shuffled-label control: {n_beat}/{len(rows)}")


if __name__ == "__main__":
    main()
