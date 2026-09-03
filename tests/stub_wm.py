"""A minimal stand-in for VWorldModel, for planner unit tests.

The real model is a from-scratch ViT-small on 224x224 images; a two-level opt step
rolls out S*C = 300 candidates, which is not a CPU unit test. This stub implements the
exact three-item surface the planners touch -- `.parameters()` (BasePlanner reads the
device from it), `encode_obs_linked`, `rollout` -- with the shapes the adaln/JEPA path
produces: `rollout` over T actions returns T+1 latent frames because obs_0 carries one
frame (`plan.py:231` expand_dims) and `_rollout_adaln` appends one extra prediction.

`rollout` is a real differentiable function of `act`, so it also exercises GDPlanner.
"""
import torch
import torch.nn as nn


class StubWorldModel(nn.Module):
    def __init__(self, emb_dim=8, num_patches=1, action_dim=10):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = num_patches
        self.enc = nn.Linear(3, emb_dim)          # gives .parameters() a device
        self.act_proj = nn.Linear(action_dim, emb_dim)
        self.rollout_batches = []                 # every call's batch size

    def encode_obs_linked(self, obs):
        v = obs["visual"]                          # (b, t, c, h, w)
        pooled = v.mean(dim=(3, 4))                # (b, t, c)
        z = self.enc(pooled).unsqueeze(2)          # (b, t, 1, D)
        return {"visual": z.expand(-1, -1, self.num_patches, -1)}

    def rollout(self, obs_0, act, z_goal=None):
        self.rollout_batches.append(int(act.shape[0]))
        z0 = self.encode_obs_linked(obs_0)["visual"]        # (b, 1, p, D)
        steps = [z0[:, -1]]
        for t in range(act.shape[1]):
            steps.append(steps[-1] + self.act_proj(act[:, t]).unsqueeze(1).tanh())
        z = torch.stack(steps, dim=1)                       # (b, T+1, p, D)
        return {"visual": z}, z


class StubPreprocessor:
    """`transform_obs` on already-tensor observations (plan.py hands numpy)."""

    def transform_obs(self, obs):
        return {k: v for k, v in obs.items()}

    def normalize_actions(self, a):
        return a


class StubWandb:
    def __init__(self):
        self.logs = []

    def log(self, d):
        self.logs.append(d)


def make_obs(b=2, c=3, h=4, w=4, proprio_dim=4):
    return {
        "visual": torch.randn(b, 1, c, h, w),
        "proprio": torch.randn(b, 1, proprio_dim),
    }
