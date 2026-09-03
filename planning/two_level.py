"""V3 - PiWM-2lvl: a two-level contact planner (subgoal x short controller).

Why this exists. Over 915,565 transitions at the model's own frameskip the PushT
block's median displacement is EXACTLY 0.000 px, 48% of transitions move it not at
all, it moves >0.5 px in only 25.3%, and the top 1% of transitions carry 34.8% of all
its motion. Under a zero-mean Gaussian action proposal that means roughly three
quarters of a sampled 5-step sequence's terminal latent is determined by the agent
alone, so ||z_H - z_g||^2 is near-constant across most of a 300-sample cloud, the
top-30 elites of `planning/cem.py:115` are close to a uniform subsample, and the
refit at `cem.py:118-119` contracts toward noise. The fix proposed here is a better
PROPOSAL, not more samples.

The factorisation. A candidate is (approach, push):

    * level 1 - a subgoal: ONE 2-D displacement command `p` HELD for k steps. PushT
      actions are relative displacement targets (`env/pusht/pusht_env.py:370
      relative=True`, not overridden in `conf/env/pusht.yaml`) tracked by a PD
      controller, so a held `p` is literally "approach in a straight line at a
      constant commanded speed". The subgoal LATENT is free: it is whatever
      `z[:, k]` the model rolls to.
    * level 2 - a short push: the remaining H-k steps, free per step.

    A subgoal is scored by the BEST push it admits (`min` over its C controllers),
    which is the standard optimistic value of a state under a lower-level policy.

Budget. S*C rollouts per opt step with S*C == the flat planner's `num_samples`, and
the same `opt_steps`, `horizon`, objective, checkpoints and episodes. The comparison
against flat CEM is therefore compute-matched by construction; the only difference is
the proposal factorisation plus the min-over-push elite rule.

Contract (planning/base_planner.py, plan.py:187-204, planning/mpc.py:44-54):
`__init__` must swallow `name=`/`env=` through **kwargs; `plan(obs_0, obs_g, actions)`
returns (Tensor (B, T, action_dim), np.ndarray (B,)); `.horizon` and
`.logging_prefix` must be settable -- plan.py:201 OVERWRITES `horizon` with `goal_H`
AFTER construction, which is why `k` is resolved inside `plan()` and not in `__init__`.
"""

import numpy as np
import torch
from einops import repeat

from utils import move_to_device

from .base_planner import BasePlanner


class TwoLevelPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        k_reach,
        n_subgoals,
        n_ctrl,
        topk_sub,
        topk_ctrl,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        frameskip=None,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.k_reach = int(k_reach)
        self.n_subgoals = int(n_subgoals)
        self.n_ctrl = int(n_ctrl)
        self.topk_sub = int(topk_sub)
        self.topk_ctrl = int(topk_ctrl)
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self._frameskip = frameskip
        # `.std(0)` over a single elite is nan, which would silently poison mu/sigma
        # for every later opt step; refuse at construction instead.
        assert self.topk_sub >= 2, "topk_sub must be >= 2 (std over 1 elite is nan)"
        assert self.topk_ctrl >= 2, "topk_ctrl must be >= 2 (std over 1 elite is nan)"
        assert self.topk_sub <= self.n_subgoals, "topk_sub > n_subgoals"
        assert (
            self.topk_ctrl <= self.topk_sub * self.n_ctrl
        ), "topk_ctrl exceeds the elite controller pool"

    @property
    def frameskip(self):
        """f, the number of env sub-steps packed into one model action.

        `evaluator.frameskip` is the live source (plan.py:186); the constructor kwarg
        exists only so the planner can be unit-tested with `evaluator=None`.
        """
        if self.evaluator is not None:
            return int(self.evaluator.frameskip)
        assert self._frameskip is not None, "no evaluator: pass frameskip= explicitly"
        return int(self._frameskip)

    def _macro(self, p, k):
        """(..., d) held command -> (..., k, action_dim).

        `p.repeat(..., f)` TILES to [px, py, px, py, ...], which is the layout
        `planning/evaluator.py` unpacks with "b t (f d) -> b (t f) d" (f is the OUTER
        axis). `repeat_interleave` would give [px]*f + [py]*f -- a different, wrong
        command that no longer holds anything constant.
        """
        f = self.frameskip
        tiled = p.repeat(*([1] * (p.dim() - 1)), f)  # (..., d*f) == (..., action_dim)
        return tiled.unsqueeze(-2).expand(*tiled.shape[:-1], k, tiled.shape[-1])

    def init_mu_sigma(self, obs_0, k, actions=None):
        """Per-episode (mu_p, sig_p, mu_u, sig_u).

        Under the shipped config MPC hands back an EMPTY warm start
        (`n_taken_actions == goal_H == horizon`, mpc.py:94), so the `actions` branch is
        only exercised if someone shortens `n_taken_actions`.
        """
        n_evals = obs_0["visual"].shape[0]
        H, A, f = self.horizon, self.action_dim, self.frameskip
        d = A // f
        assert d * f == A, f"action_dim {A} is not divisible by frameskip {f}"
        dev = self.device
        mu_p = torch.zeros(n_evals, d, device=dev)
        sig_p = self.var_scale * torch.ones(n_evals, d, device=dev)
        mu_u = torch.zeros(n_evals, H - k, A, device=dev)
        sig_u = self.var_scale * torch.ones(n_evals, H - k, A, device=dev)
        if actions is not None and actions.shape[1] > 0:
            a = actions.to(dev)
            t = a.shape[1]
            # the held command warm-starts from the mean sub-step of the first
            # min(k, t) model steps; the push segment copies whatever is left.
            head = a[:, : min(k, t)].reshape(n_evals, -1, f, d).mean(dim=(1, 2))
            mu_p = head
            tail = a[:, k:] if t > k else a[:, t:]
            if tail.shape[1] > 0:
                n = min(tail.shape[1], H - k)
                mu_u[:, :n] = tail[:, :n]
        return mu_p, sig_p, mu_u, sig_u

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            obs_0, obs_g: raw obs dicts (B, ...); actions: optional warm start.
        Returns:
            (B, horizon, action_dim) torch.Tensor, np.ndarray (B,)
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs_linked(trans_obs_g)

        H, A = self.horizon, self.action_dim
        # plan.py:201 overwrites `horizon` with goal_H AFTER __init__, so k -- which
        # is a function of the horizon -- can only be resolved here.
        k = max(1, min(self.k_reach, H - 1))
        S, C = self.n_subgoals, self.n_ctrl
        print(
            f"[2lvl] H={H} k={k} S={S} C={C} rollouts/opt_step={S * C} "
            f"(flat CEM: 300)",
            flush=True,
        )

        mu_p, sig_p, mu_u, sig_u = self.init_mu_sigma(obs_0, k, actions)
        n_evals = mu_p.shape[0]
        dev = self.device

        for i in range(self.opt_steps):
            losses = []
            sub_disp = []
            best_actions = []
            for traj in range(n_evals):
                cur_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=S * C)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=S * C)
                    for key, arr in z_obs_g.items()
                }

                # ---- level 1: S held approach commands -------------------------
                p = torch.randn(S, mu_p.shape[1], device=dev) * sig_p[traj] + mu_p[traj]
                p[0] = mu_p[traj]                       # keep the incumbent
                macro = self._macro(p, k)               # (S, k, A)

                # ---- level 2: C free pushes per subgoal -------------------------
                u = torch.randn(S, C, H - k, A, device=dev) * sig_u[traj] + mu_u[traj]
                u[:, 0] = mu_u[traj]                    # keep the incumbent push
                act = torch.cat(
                    [macro.unsqueeze(1).expand(S, C, k, A), u], dim=2
                ).reshape(S * C, H, A)

                with torch.no_grad():
                    z_obses, _ = self.wm.rollout(
                        obs_0=cur_obs_0, act=act, z_goal=cur_z_g["visual"]
                    )
                loss = self.objective_fn(z_obses, cur_z_g).view(S, C)

                # a subgoal is worth the BEST push it admits
                v_sub = loss.min(dim=1).values                       # (S,)
                e_sub = v_sub.argsort()[: self.topk_sub]
                elite_p = p[e_sub]
                mu_p[traj], sig_p[traj] = elite_p.mean(0), elite_p.std(0)

                flat = loss[e_sub].reshape(-1)                       # (topk_sub*C,)
                order = flat.argsort()[: self.topk_ctrl]
                ue = u[e_sub].reshape(-1, H - k, A)[order]
                mu_u[traj], sig_u[traj] = ue.mean(0), ue.std(0)

                losses.append(float(v_sub.min()))
                best_actions.append(
                    torch.cat([self._macro(mu_p[traj], k), mu_u[traj]], dim=0)
                )
                # diagnostic: does level 1 resolve anything? dispersion of the elite
                # SUBGOAL LATENTS z[:, k], averaged over the elite subgoals' best push.
                zs = z_obses["visual"].reshape(S, C, *z_obses["visual"].shape[1:])
                z_sub = zs[e_sub, loss[e_sub].argmin(dim=1)][:, k]
                sub_disp.append(float(z_sub.std(dim=0).mean()))

            mu_actions = torch.stack(best_actions, dim=0)            # (B, H, A)
            # wandb is disabled in every eval job (arm_slurm.sbatch) and sub-planners get
            # log_filename=None (mpc.py:53), so stdout is the only place the objective
            # trace and the level-1 diagnostic survive.
            print(
                f"[2lvl] {self.logging_prefix} step {i + 1}/{self.opt_steps} "
                f"obj={np.mean(losses):.6f} elite_z_disp={np.mean(sub_disp):.6f}",
                flush=True,
            )
            self.wandb_run.log(
                {
                    f"{self.logging_prefix}/loss": np.mean(losses),
                    f"{self.logging_prefix}/elite_subgoal_z_dispersion": float(
                        np.mean(sub_disp)
                    ),
                    "step": i + 1,
                }
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu_actions, filename=f"{self.logging_prefix}_output_{i + 1}"
                )
                logs = {f"{self.logging_prefix}/{key}": v for key, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu_actions, np.full(n_evals, np.inf)  # all actions are valid
