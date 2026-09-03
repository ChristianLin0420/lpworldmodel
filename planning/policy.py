"""V4 - PiWM-policy: a reactive goal-conditioned policy, no search, same evaluator.

`a_t = pi(z_t, z_g)` from the BC head trained alongside T4's value head
(`VWorldModel._policy_loss`), driven through the SAME `planning.mpc.MPCPlanner` the CEM
baseline uses. The MPC wrapper is mandatory, not optional: it gives the policy exactly the
same 10 rounds of real-env feedback, the same latched-success accounting
(`planning/mpc.py:110-112`) and the same `n_taken_actions = 5`, so the only difference
from the baseline is *how the five actions were chosen*.

Why this is worth a planner slot. CEM's advantage over a reactive policy can only come
from its leaf metric being informative about the goal, and that metric -- the terminal
latent MSE -- is ranked against the TRUE task distance at Spearman **+0.398** (n = 296).
A search whose objective is that weakly ordered is, at the margin, a draw from the action
prior with 300 x 30 x 10 = 90,000 model rollouts of overhead. This planner spends
H(H+1)/2 = 15 predictor calls per episode instead. If the two match, every ranking in this
campaign was ranking representations rather than planners.

(The earlier version of this argument also cited `d_action` ~ 2e-4 for the baseline.
That is **withdrawn**: the true value is 0.549 and `d_action` orders CEM success at
partial rho = +0.70 among healthy predictors. V4's case rests entirely on the leaf metric.)

Two guards exist because of the `path_int` defect (diary 2026-09-03 s13.3), where a plan
-time module was silently rebuilt from a fresh `nn.Linear` and produced a conclusion that
had to be retracted:

  * the head must EXIST -- `plan.py:load_model` restores a fixed list of submodules and
    knows nothing about it, so `VWorldModel` restores it itself from `LPWM_HEAD_CKPT`;
  * the head must have been RESTORED FROM DISK -- `heads_restored_from` is set only by
    `VWorldModel.load_head_state`, and planning on a fresh init raises here instead of
    quietly measuring `torch.manual_seed(20260903)`.
"""

import numpy as np
import torch

from utils import move_to_device

from .base_planner import BasePlanner


def _head_owner(wm):
    """The module that owns `policy_head`: `wm` itself, or its single ensemble member.

    `scripts/plan_arm.sh` routes a planner-variant eval through `ensemble_members=[<one
    run>]` so the output label decouples from the checkpoint dir (otherwise a V4 eval of
    `PiWM-vp_..._s3` would overwrite that seed's CEM number in `collect_evals.py`). That
    wraps the model in an `EnsembleWorldModel`, whose M=1 path is verified inert for
    rollout and encoding but which does not forward attribute access.
    """
    if getattr(wm, "policy_head", None) is not None:
        return wm
    members = getattr(wm, "members", None)
    if members is not None:
        if len(members) != 1:
            raise ValueError(
                f"PolicyPlanner needs exactly one model, got an ensemble of {len(members)}"
            )
        return members[0]
    return wm


class PolicyPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="policy_0",
        log_filename="logs.json",
        shuffle_goal=False,
        require_restored=True,
        **kwargs,          # swallows name=/env=/topk=/num_samples=... : plan.py passes
    ):                     # env= to every planner and conf/plan_lewm.yaml's sub_planner
        super().__init__(  # node carries the CEM-only keys
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.logging_prefix = logging_prefix
        self.shuffle_goal = bool(shuffle_goal)
        owner = _head_owner(wm)
        self.policy = getattr(owner, "policy_head", None)
        if self.policy is None:
            raise ValueError(
                "PolicyPlanner needs a TRAINED policy_head. The checkpoint's train_cfg "
                "must carry model.policy_w > 0 and model.act_dim_raw, and the run must "
                "have been trained with POLICY_W set (scripts/train.sh)."
            )
        restored = getattr(owner, "heads_restored_from", None)
        if require_restored and not restored:
            raise ValueError(
                "policy_head is a FRESH INIT: plan.py:load_model restores a fixed list of "
                "submodules and never touches it. Set LPWM_HEAD_CKPT=<run>/checkpoints/"
                "model_latest.pth so VWorldModel.load_head_state restores it. Planning on "
                "a fresh init is the path_int defect (diary 2026-09-03 s13.3)."
            )
        print(f"policy_head restored from {restored}")
        out = getattr(self.policy, "d_out", None)
        if out is not None and int(out) != int(action_dim):
            raise ValueError(
                f"policy_head emits {out}-d actions but the env wants {action_dim} "
                "(2 * frameskip). The head was trained at a different frameskip."
            )
        self.device = next(owner.parameters()).device

    @torch.no_grad()
    def plan(self, obs_0, obs_g, actions=None):
        """`actions` is ignored: there is no search here, so nothing to warm-start.

        The latent is advanced by re-calling `wm.rollout` with the growing action prefix.
        That is H(H+1)/2 = 15 predictor calls against CEM's 90,000, and it reuses
        `_rollout_adaln` exactly -- including T6's option-model re-indexing and the
        `n_heads > 1` optimistic step -- so no window or pose-rolling logic is duplicated
        here and no rollout path can silently diverge from the one the baseline used.
        """
        t0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        tg = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        zg_full = self.wm.encode_obs_linked(tg)["visual"]          # (B, 1, p, D)
        if self.shuffle_goal:
            # The control. PushT demos have a strong action prior, so a policy that
            # ignores z_g entirely can still score; if shuffled-goal success equals
            # real-goal success, the policy is replaying the prior and the arm says
            # nothing about planning. Permutes goals ACROSS episodes and nothing else.
            zg_full = zg_full[torch.randperm(zg_full.shape[0], device=zg_full.device)]
        z_g = zg_full[:, 0].mean(dim=1)                            # (B, D); p == 1 exact
        z_t = self.wm.encode_obs_linked(t0)["visual"][:, -1].mean(dim=1)
        pdev = next(self.policy.parameters()).device
        acts = []
        for step in range(int(self.horizon)):
            acts.append(self.policy(z_t.to(pdev), z_g.to(pdev)).to(z_t.device))
            if step + 1 >= int(self.horizon):
                break                                   # last action needs no successor
            z_obses, _ = self.wm.rollout(
                obs_0=t0, act=torch.stack(acts, dim=1), z_goal=zg_full
            )
            z_t = z_obses["visual"][:, -1].mean(dim=1)
        mu = torch.stack(acts, dim=1)                              # (B, H, action_dim)
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    f"{self.logging_prefix}/pi_rms": float(mu.pow(2).mean().sqrt()),
                    f"{self.logging_prefix}/rollout_calls": max(int(self.horizon) - 1, 0),
                    "step": 1,
                }
            )
        # np.inf everywhere: "no episode has terminated inside this sub-plan", the same
        # value CEMPlanner returns (cem.py:127). MPCPlanner owns action_len.
        return mu, np.full(mu.shape[0], np.inf)
