import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
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
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        # V2. GD differentiates the objective w.r.t. the action tensor. The vote
        # objectives (planning/objectives.py:106-142) return ARGSORT RANKS, which are
        # piecewise constant in the actions, so their gradient is identically zero
        # almost everywhere: a GD-over-vote arm would silently plan with the random
        # init and look like a baseline. This incompatibility is documented at
        # planning/objectives.py; refuse rather than bodge it.
        assert getattr(objective_fn, "__name__", "") != "objective_fn_vote", (
            "rank-based vote objectives are piecewise constant (argsort); "
            "GD gets zero gradient"
        )

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs_linked(trans_obs_g)  # match the linked rollout space
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        actions = self.init_actions(obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_evals = actions.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
                z_goal=z_obs_g_detached["visual"],
            )
            loss = self.objective_fn(i_z_obses, z_obs_g_detached)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += (
                    torch.randn_like(actions_new) * self.action_noise
                )  # Add Gaussian noise
                actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            # The per-step objective is the quantity V2's informative sub-case turns on
            # ("GD reaches a LOWER objective than CEM at equal or worse success"), and
            # wandb is disabled in every eval job (arm_slurm.sbatch), so mirror it to
            # stdout -- slurm_logs/ is then the only place it survives.
            if i % self.eval_every == 0 or i == self.opt_steps - 1:
                print(
                    f"[gd] {self.logging_prefix} step {i + 1}/{self.opt_steps} "
                    f"obj={loss.mean().item():.6f} "
                    f"|grad|={float(actions.grad.norm()):.3e}",
                    flush=True,
                )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return actions, np.full(n_evals, np.inf)  # all actions are valid
