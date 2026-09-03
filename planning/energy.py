"""T7 - a frozen-encoder ranking head over action sequences, and the CEM leaf that uses it.

WHAT IS HERE
    `SeqEnergyHead`   E_theta(z_0, a_{1:H}, z_g) -> (B,) scalar, lower = better sequence.
    `load_energy_head`  strict load + checksum verification (the anti-`path_int` gate).
    `_ScoredCEMPlanner` `planning.cem.CEMPlanner.plan` verbatim, with the leaf score
                        factored out into `_score()`. Its `_score` IS the inline path.
    `EnergyCEMPlanner`  overrides `_score` with the head. It never calls `wm.rollout`.

WHY THE ENCODER IS FROZEN
    InfoNCE over actions has collapsed the code under both negative distributions tried:
    batch-permutation drove rho 0.448 -> 0.052, and kNN-from-p(a|z) reached rel_mse
    exactly 1.0000 with rho 0.13 and a temporally frozen latent (diary 2026-09-03 sec 5,
    12). In both cases the ENCODER was the free parameter and zeroing the code was the
    cheapest way to satisfy the contrastive term. Here z is a constant function of the
    data, so that degree of freedom does not exist -- the failure mode is structurally
    unavailable rather than merely discouraged. Nothing in this file can change the world
    model: `EnergyCEMPlanner` only reads `self.wm.encode_obs_linked`.

WHY IT IS A RANKING AND NOT A DISTANCE
    CEM only ever computes `argsort(loss)[:topk]` (`planning/cem.py:115`), so a leaf
    metric is consumed as an ORDER. The head is trained (`train_energy.py`) by
    cross-entropy to put the executed demonstration sequence above CEM-distributed
    samples, and it is judged by rank agreement (`val_top1` over 64 candidates, chance
    1/64 = 1.56%) alone. It is NOT motivated by action-sensitivity: diary sec 12b
    withdrew the claim that d_action fails to predict CEM.

FILE-OWNERSHIP NOTE (round 5, parallel agents)
    The spec put the module in `models/energy_head.py`, the CEM `_score` refactor in
    `planning/cem.py`, an `energy_head=` kwarg in `planning/mpc.py` and the checkpoint
    load in `plan.py`. This wave those four files belong to other agents, so all of it
    lives here instead and the checkpoint path arrives WITHOUT touching them:

      * `planning/mpc.py:45-54` instantiates `sub_planner` with hydra, so every extra key
        under `planner.sub_planner` is forwarded as a kwarg -- hence `energy_ckpt` is a
        constructor argument, settable with
        `+planner.sub_planner.energy_ckpt=/abs/path.pt` (see `scripts/plan_energy.sh`);
      * failing that, the env var `ENERGY_CKPT` is read here.

    A missing head is a hard error, never a fresh init: `path_int` was silently randomly
    initialised at plan time (`visual_world_model.py:383-389`) and produced a conclusion
    that had to be retracted (diary sec 13.3).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

from utils import move_to_device

from .cem import CEMPlanner


# ---------------------------------------------------------------- the head


class SeqEnergyHead(nn.Module):
    """E(z_0, a_{1:H}, z_g): one scalar per candidate action sequence, lower = better.

    The action trunk is FRESH on purpose. The checkpoint's `action_encoder` is the module
    suspected of carrying no action information, so reusing it would bottleneck the head
    with the defect under test; a `Linear(H*act_dim, hidden)` also gets a fan_in of
    H*act_dim = 50 rather than the per-row 10.

    `z_scale` / `y_scale` are BUFFERS, so they travel inside `state_dict()` and are
    covered by the load-time checksum. `z_scale` is the RMS of the frozen code on the
    training cache: the reprelu link's scale is a property of the world-model checkpoint,
    not of the head, and dividing it out keeps one architecture usable across seeds whose
    |z| differ. `y_scale` is only read by the distill control, which regresses the
    rollout's own terminal latent MSE and therefore needs the target in O(1) units.
    """

    def __init__(self, emb_dim=384, horizon=5, act_dim=10, hidden=512):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.horizon = int(horizon)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)
        self.act_mlp = nn.Sequential(
            nn.Linear(self.horizon * self.act_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.emb_dim + self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.register_buffer("z_scale", torch.ones(()))
        self.register_buffer("y_scale", torch.ones(()))

    @property
    def arch(self):
        return dict(
            emb_dim=self.emb_dim,
            horizon=self.horizon,
            act_dim=self.act_dim,
            hidden=self.hidden,
        )

    def forward(self, z0, ag, zg):
        """z0 (B,D), ag (B,H,act_dim), zg (B,D) -> E (B,)."""
        assert z0.shape == zg.shape and z0.ndim == 2, (z0.shape, zg.shape)
        assert z0.shape[-1] == self.emb_dim, (z0.shape, self.emb_dim)
        assert ag.ndim == 3 and ag.shape[1:] == (self.horizon, self.act_dim), ag.shape
        s = self.z_scale.to(z0.dtype).clamp_min(1e-8)
        z0, zg = z0 / s, zg / s
        a = self.act_mlp(ag.flatten(1).to(z0.dtype))
        return self.mlp(torch.cat([z0, zg, z0 - zg, a], dim=-1)).squeeze(-1)

    # -- checkpoint plumbing -------------------------------------------------

    def checksum(self):
        return state_dict_checksum(self.state_dict())

    def grad_report(self):
        """[(name, grad-norm)] for every parameter. Used by the liveness assertion:
        a head whose gradient never arrives is exactly the `path_int` failure."""
        out = []
        for n, p in self.named_parameters():
            out.append((n, None if p.grad is None else float(p.grad.norm())))
        return out


def state_dict_checksum(sd):
    """float64 sum over every tensor in a state_dict. Printed at save and at load; a
    fresh init cannot match it (diary sec 13.3: `path_int` was randomly initialised at
    plan time and nothing noticed)."""
    return float(sum(v.detach().double().sum() for v in sd.values()))


def save_energy_head(head, path, **meta):
    sd = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    payload = dict(state_dict=sd, arch=head.arch, checksum=state_dict_checksum(sd), **meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    print(
        f"energy head saved to {path} | param checksum {payload['checksum']:.6f} | "
        f"arch {payload['arch']} | " + " ".join(f"{k}={v}" for k, v in meta.items()),
        flush=True,
    )
    return payload["checksum"]


def load_energy_head(path, device=None, expect_horizon=None):
    """Load a TRAINED head, or refuse.

    Three refusals, all of them the `path_int` lesson: the file must exist, the
    state_dict must load STRICTLY into the architecture the file itself declares, and
    the recomputed checksum must equal the one written at save time. The checksum is
    printed so a plan log can be diffed against the training log.
    """
    if path is None or str(path) in ("", "null", "None"):
        raise ValueError(
            "EnergyCEMPlanner needs a TRAINED head: pass energy_ckpt=<path> "
            "(+planner.sub_planner.energy_ckpt=...) or set ENERGY_CKPT. Refusing to "
            "plan with a fresh init -- that is exactly how path_int produced a "
            "retracted result (diary 2026-09-03 sec 13.3)."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"energy head not found: {path}")
    payload = torch.load(path, map_location="cpu")
    for k in ("state_dict", "arch"):
        if k not in payload:
            raise KeyError(f"{path} has no '{k}'; not an energy-head checkpoint")
    head = SeqEnergyHead(**payload["arch"])
    head.load_state_dict(payload["state_dict"], strict=True)
    got = head.checksum()
    want = payload.get("checksum")
    if want is not None and abs(got - float(want)) > 1e-6 * max(1.0, abs(float(want))):
        raise ValueError(f"energy head checksum mismatch: file {want} != loaded {got}")
    if expect_horizon is not None and int(expect_horizon) != head.horizon:
        raise ValueError(
            f"energy head was trained for horizon {head.horizon} but the planner runs "
            f"horizon {expect_horizon}; the action block would be a different object."
        )
    head = head.to(device or "cpu").eval().requires_grad_(False)
    print(
        f"energy head loaded from {path} | param checksum {got:.6f} | "
        f"arch {payload['arch']} | run={payload.get('run')} step={payload.get('step')} "
        f"mode={payload.get('mode')} val_top1={payload.get('val_top1')}",
        flush=True,
    )
    return head


def flatten_z(z):
    """(b, t, p, d) or (b, p, d) linked latent -> (b, d), taking the LAST frame.

    p == 1 for the whole LpWM/LeWM line (`feature: cls`), and the M=1 ensemble wrapper
    used to relabel an eval (`planning/ensemble.py`, `scripts/plan_arm.sh`) stacks
    members on that same axis, so the two are the same tensor here.
    """
    if z.ndim == 4:
        z = z[:, -1]
    assert z.ndim == 3, f"expected (b,t,p,d) or (b,p,d), got {tuple(z.shape)}"
    assert z.shape[1] == 1, (
        f"the energy head consumes one token per frame, got p={z.shape[1]}. A patch-token "
        f"or multi-member latent needs a head trained on it."
    )
    return z[:, 0]


# ---------------------------------------------------------------- planners


class _ScoredCEMPlanner(CEMPlanner):
    """`CEMPlanner.plan` with the leaf score factored into `_score`, and nothing else.

    The body below is a verbatim copy of `planning/cem.py:64-135` (that file belongs to
    another agent this wave, so the hook could not be added there). `_score` here IS the
    inline `rollout` + `objective_fn` path, so this class is a drop-in for `CEMPlanner`;
    `tests/test_energy_planner.py` asserts the two produce bit-identical `mu` from the
    same RNG state, which is what makes the subclass's numbers comparable to the archive.
    """

    def _score(self, cur_trans_obs_0, action, cur_z_obs_g):
        with torch.no_grad():
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=cur_trans_obs_0,
                act=action,
                z_goal=cur_z_obs_g["visual"],
            )
        return self.objective_fn(i_z_obses, cur_z_obs_g)

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs_linked(trans_obs_g)  # match the linked rollout space

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself

                loss = self._score(cur_trans_obs_0, action, cur_z_obs_g)

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid


class EnergyCEMPlanner(_ScoredCEMPlanner):
    """CEM whose leaf score is the learned ranking head instead of a rollout.

    `_score` calls the head once per CEM batch, so the arm holds the PROPOSAL
    distribution, the elite fraction, the budget and the evaluator fixed and changes
    only the ordering the elites are selected by. There is no rollout at all, which is
    the point: no compounding, no `_roll_pose`, and the M3 finding (the rollout, not the
    leaf metric, is what the planner is limited by -- diary sec 16.4) is bypassed rather
    than inherited.

    Refuses to construct without a head on disk. `energy_head=` (an already-loaded
    module) exists for tests; production passes `energy_ckpt=`.
    """

    def __init__(self, *args, energy_head=None, energy_ckpt=None, **kwargs):
        super().__init__(*args, **kwargs)
        if energy_head is None:
            path = energy_ckpt if energy_ckpt else os.environ.get("ENERGY_CKPT")
            energy_head = load_energy_head(
                path, device=self.device, expect_horizon=None
            )
        self.energy_head = energy_head.to(self.device).eval().requires_grad_(False)
        self.energy_ckpt = energy_ckpt
        print(
            f"[energy] leaf objective = SeqEnergyHead (no rollout) | horizon="
            f"{self.energy_head.horizon} act_dim={self.energy_head.act_dim} "
            f"emb_dim={self.energy_head.emb_dim} | param checksum "
            f"{self.energy_head.checksum():.6f}",
            flush=True,
        )

    def _score(self, cur_trans_obs_0, action, cur_z_obs_g):
        """(B,) energies for B candidate sequences. `wm.rollout` is never called."""
        head = self.energy_head
        if action.shape[1] != head.horizon:
            raise ValueError(
                f"energy head trained for horizon {head.horizon}, planner is proposing "
                f"{action.shape[1]}-step sequences"
            )
        with torch.no_grad():
            z0 = flatten_z(self.wm.encode_obs_linked(cur_trans_obs_0)["visual"])
            zg = flatten_z(cur_z_obs_g["visual"])
            return head(z0, action.to(z0.dtype), zg)
