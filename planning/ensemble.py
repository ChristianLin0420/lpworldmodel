"""Plan-time consensus over independently trained world models ("columns").

TBT's mechanism is a VOTE among columns that each model the whole object, not a
`min` over readouts of one shared trunk (`VWorldModel._union_head_loss`). Two facts
about this codebase force the vote to happen at PLAN time, on the objective, rather
than in the loss or on the latents:

  * every run trains its own encoder from scratch (`conf/encoder/vit_scratch.yaml`,
    `model.train_encoder: True`), so two seeds have unrelated latent bases -- averaging
    their z is meaningless;
  * a loss-side ensemble of the existing heads is provably vacuous: heads share
    `_trunk` and differ only by a final Linear (`LinearDynamicsPredictor.forward_heads`),
    and for linear readouts
        mean_j ||W_j c - t||^2 == ||W_bar c - t||^2 + mean_j ||(W_j - W_bar) c||^2,
    i.e. "min -> mean" would just be one head plus a shrinkage penalty.

So: M complete models roll out independently, each in its own space, and their
per-candidate opinions are combined by `planning.objectives.create_vote_objective_fn`.
There is no degenerate optimum to admit, because there is no optimisation here at all.

Member latents are stacked on the PATCH axis (CLS => p == 1 per member), which keeps
every downstream consumer (`planning/cem.py`, `planning/evaluator.py`) tensor-shaped
and untouched. Members with different code widths D are right-zero-padded to max D;
that is a per-member monotone rescale of the MSE, so it leaves rank-based votes exactly
invariant (and only reweights the mean vote).
"""

import torch
import torch.nn as nn


class EnsembleWorldModel(nn.Module):
    """M independently trained VWorldModels voting at plan time.

    Exposes the subset of the VWorldModel API the planner and evaluator use:
    `encode_obs_linked`, `rollout`, `.decoder`. Every returned "visual" tensor is
    (b, t, M, D_max): member m occupies patch slot m.
    """

    def __init__(self, members, names=None):
        super().__init__()
        assert len(members) >= 1
        self.members = nn.ModuleList(members)
        self.names = list(names) if names is not None else [f"m{i}" for i in range(len(members))]
        self.n_members = len(members)
        for m in self.members:
            assert m.action_conditioning == "adaln", "ensemble supports the adaln/JEPA path only"
            assert m.encoder.num_patches == 1, "ensemble stacks members on the patch axis (CLS only)"
        self.dims = [int(m.encoder.emb_dim) for m in self.members]
        self.d_max = max(self.dims)
        self.decoder = None  # evaluator gates its rollout plots on this

    def _stack(self, per_member):
        """[(b, t, 1, D_m)] -> (b, t, M, D_max), right-zero-padded."""
        out = []
        for z in per_member:
            d = z.shape[-1]
            if d < self.d_max:
                z = torch.nn.functional.pad(z, (0, self.d_max - d))
            out.append(z)
        return torch.cat(out, dim=2)

    def _split_goal(self, z_goal, m):
        """Undo the stacking for member m (only used by n_heads>1 members)."""
        if z_goal is None:
            return None
        g = z_goal[..., m : m + 1, : self.dims[m]] if z_goal.dim() >= 3 else None
        return g

    def encode_obs_linked(self, obs):
        return {"visual": self._stack([m.encode_obs_linked(obs)["visual"] for m in self.members])}

    def rollout(self, obs_0, act, z_goal=None):
        outs = []
        for i, m in enumerate(self.members):
            z_obses, _ = m.rollout(obs_0, act, z_goal=self._split_goal(z_goal, i))
            outs.append(z_obses["visual"])
        z = self._stack(outs)
        return {"visual": z}, z
