"""M3 guards.

Three things have to hold before any ladder cell means anything:

  1. the CONTROL cell is not a new planner.  `LadderCEMPlanner` at its defaults must produce
     bit-identical actions to `planning.cem.CEMPlanner` -- otherwise (learned, learned) is
     measuring the refactor, not the model.  This is the check that would have caught the
     V1-V3 silent-identity class of bug from the other side: there the arms were accidentally
     identical, here the control must be DELIBERATELY identical.
  2. oracle dynamics must be the same dynamical system the evaluator will score.  The pool's
     replay-from-init has to reproduce `PushTWrapper.rollout` exactly.
  3. the spec's stated precondition -- that `prepare(seed, state)` is an exact restart -- is
     FALSE, and the test pins the measured numbers so nobody re-derives the design from it.
"""
import numpy as np
import pytest
import torch

from analysis.oracle_ladder import LadderCEMPlanner, SimPool
from planning.cem import CEMPlanner


class _StubWM(torch.nn.Module):
    """Deterministic stand-in: the rollout is a fixed linear function of the actions."""

    def __init__(self, d=6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.d = d
        self.M = torch.arange(1.0, d + 1).reshape(1, 1, d) / d

    def encode_obs_linked(self, obs):
        v = obs["visual"]
        return {"visual": v[..., :self.d].reshape(v.shape[0], 1, 1, self.d)}

    def rollout(self, obs_0, act, z_goal=None):
        s = act.sum(dim=(1, 2)).reshape(-1, 1, 1, 1)                     # (n, 1, 1, 1)
        z = s * self.M.reshape(1, 1, 1, self.d).to(act.device)           # (n, 1, 1, d)
        z = torch.cat([torch.zeros_like(z), z], dim=1)                   # (n, 2, 1, d)
        return {"visual": z}, z


class _StubPre:
    def transform_obs(self, obs):
        return {k: torch.as_tensor(v) for k, v in obs.items()}


class _StubWandb:
    def log(self, *a, **k):
        pass


def _objective(pred, tgt):
    return ((pred["visual"][:, -1:] - tgt["visual"]) ** 2).mean(dim=(1, 2, 3))


def _mk(cls, **extra):
    wm = _StubWM()
    return cls(horizon=3, topk=4, num_samples=16, var_scale=1.0, opt_steps=3,
               eval_every=1, wm=wm, action_dim=4, objective_fn=_objective,
               preprocessor=_StubPre(), evaluator=None, wandb_run=_StubWandb(),
               log_filename=None, **extra)


def _run(planner, seed=0):
    obs_0 = {"visual": torch.zeros(2, 1, 8)}
    obs_g = {"visual": torch.ones(2, 1, 8)}
    torch.manual_seed(seed)
    mu, alen = planner.plan(obs_0=obs_0, obs_g=obs_g)
    return mu.detach().clone(), alen


def test_control_cell_is_bit_identical_to_cem():
    """(learned, learned) through LadderCEMPlanner == planning/cem.py, bit for bit."""
    a, la = _run(_mk(CEMPlanner))
    b, lb = _run(_mk(LadderCEMPlanner))          # defaults: dynamics/objective = learned
    assert torch.equal(a, b), (a - b).abs().max().item()
    assert np.array_equal(la, lb)


def test_defaults_build_nothing():
    """The control must not construct a simulator pool or a pose decoder."""
    p = _mk(LadderCEMPlanner)
    _run(p)
    assert p._pool is None and p._dec is None
    assert p._sim_calls == 0


def test_oracle_objective_changes_the_ranking():
    """A swapped objective must actually change what CEM selects -- the arms-differ check."""
    p = _mk(LadderCEMPlanner, objective="oracle")

    class _Ev:                                    # privileged goal, evaluation only
        state_g = np.tile(np.array([100., 100., 250., 250., 0.5, 0., 0.], np.float32), (2, 1))
        state_0 = state_g.copy()
        seed = [1, 2]
        frameskip = 2

        def get_init_cond(self):
            return None, self.state_0

        def eval_actions(self, *a, **k):        # never reached with log_filename=None
            return {}, np.zeros(2, bool), None, None
    p.evaluator = _Ev()
    p._dec = dict(W=torch.zeros(6, 6, dtype=torch.float64),
                  mu=torch.zeros(1, 6), sd=torch.ones(1, 6),
                  ymu=torch.arange(6.0, dtype=torch.float64).reshape(1, 6), lam=1.0, diag={})
    a, _ = _run(_mk(CEMPlanner))
    b, _ = _run(p)
    assert not torch.equal(a, b), "oracle objective produced the learned objective's plan"


def test_pool_replay_equals_env_rollout():
    """Oracle dynamics is the evaluator's own rollout, not an approximation of it."""
    from env.pusht.pusht_wrapper import PushTWrapper
    e = PushTWrapper()
    rs = np.random.RandomState(1)
    pool = SimPool(2)
    try:
        worst = 0.0
        for _ in range(3):
            bx, by, th = rs.randint(150, 350), rs.randint(150, 350), rs.randn()
            s = np.array([bx - 40., by - 40., bx, by, th % (2 * np.pi), 0, 0], np.float32)
            pre = rs.randn(6, 2) * 1.5
            cand = rs.randn(2, 10, 2) * 1.5
            full = np.concatenate([np.repeat(pre[None], 2, 0), cand], axis=1)
            ref = np.stack([e.rollout(7, s, full[i])[1][-1] for i in range(2)])
            got, _ = pool.terminal(7, s.astype(np.float64), pre, cand)
            worst = max(worst, float(np.abs(ref - got).max()))
        assert worst == 0.0, f"pool diverges from env.rollout by {worst}"
    finally:
        pool.close()


def test_spec_precondition_is_false():
    """docs/round5-specs.md M3 step 0 expects 0.0.  It is not 0.0; pin the real numbers."""
    from env.pusht.pusht_wrapper import PushTWrapper
    e = PushTWrapper()
    s0 = np.array([278., 412., 235., 306., .84, 0, 0], np.float32)
    a = np.random.RandomState(0).randn(20, 2) * .2
    _, S = e.rollout(7, s0, a)
    _, S2 = e.rollout(7, S[10], a[10:])
    err = float(np.abs(S[10:] - S2).max())
    assert err > 0.5, err                     # 0.638 px: one extra space.step of agent velocity
    assert float(np.abs(S[10:, 2:5] - S2[:, 2:5]).max()) == 0.0   # block idle in this example

    rs = np.random.RandomState(0)
    worst_block = 0.0
    for _ in range(8):                        # now with the agent ON the block
        bx, by, th = rs.randint(150, 350), rs.randint(150, 350), rs.randn()
        s = np.array([bx - 40., by - 40., bx, by, th % (2 * np.pi), 0, 0], np.float32)
        acts = rs.randn(20, 2) * 1.5
        _, R = e.rollout(7, s, acts)
        for k in (5, 10, 15):
            _, R2 = e.rollout(7, R[k], acts[k:])
            worst_block = max(worst_block, float(np.abs(R[k:, 2:5] - R2[:, 2:5]).max()))
    assert worst_block > 1.0, worst_block     # a restart is NOT an exact restart under contact
