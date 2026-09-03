"""V2 - GDPlanner as a first-class arm: contract, gradient liveness, vote guard.

`planning/gd.py` already existed but was never run as an arm. Three things are pinned:
  * the planner contract (returns (Tensor (B,H,A), np.ndarray (B,)), `.horizon` and
    `.logging_prefix` settable, `**kwargs` swallows `name=`/`env=`);
  * gradient actually reaches the action tensor -- the whole arm is vacuous otherwise,
    and a silently-zero gradient is exactly how `path_int` produced a retracted result;
  * GD must REFUSE a rank-based vote objective. `objective_fn_vote` returns argsort
    ranks, which are piecewise constant in the actions, so its gradient is zero almost
    everywhere; without the guard a `gd x vote` arm would plan with its random init and
    be reported as a null.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from planning.gd import GDPlanner  # noqa: E402
from planning.objectives import (  # noqa: E402
    create_objective_fn,
    create_vote_objective_fn,
)
from stub_wm import (  # noqa: E402
    StubPreprocessor,
    StubWandb,
    StubWorldModel,
    make_obs,
)

ACTION_DIM = 10


def build(objective_fn=None, **over):
    kw = dict(
        horizon=5,
        action_noise=0.003,
        sample_type="randn",
        lr=1,
        opt_steps=3,
        eval_every=100,
    )
    kw.update(over)
    wm = StubWorldModel(action_dim=ACTION_DIM)
    p = GDPlanner(
        wm=wm,
        action_dim=ACTION_DIM,
        objective_fn=objective_fn or create_objective_fn(alpha=0, base=2, mode="last"),
        preprocessor=StubPreprocessor(),
        evaluator=None,
        wandb_run=StubWandb(),
        log_filename=None,
        name="mpc_gd",
        env=None,
        **kw,
    )
    return p, wm


def test_is_a_base_planner_and_swallows_name_and_env():
    from planning.base_planner import BasePlanner

    p, _ = build()
    assert isinstance(p, BasePlanner)


def test_plan_returns_contract_shapes_and_dtypes():
    p, _ = build()
    actions, lens = p.plan(make_obs(b=2), make_obs(b=2))
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (2, 5, ACTION_DIM)
    assert actions.dtype == torch.float32
    assert isinstance(lens, np.ndarray) and lens.shape == (2,)


def test_horizon_and_logging_prefix_are_settable_after_construction():
    p, _ = build()
    p.horizon = 3            # plan.py:201
    p.logging_prefix = "plan_4"
    actions, _ = p.plan(make_obs(b=2), make_obs(b=2))
    assert actions.shape == (2, 3, ACTION_DIM)
    assert any(k.startswith("plan_4/") for log in p.wandb_run.logs for k in log)


def test_gradient_reaches_the_action_tensor():
    """The arm is vacuous if d(objective)/d(actions) is zero or None."""
    p, _ = build(opt_steps=1, action_noise=0.0)
    obs0, obsg = make_obs(b=2), make_obs(b=2)
    trans0 = p.preprocessor.transform_obs(obs0)
    zg = p.wm.encode_obs_linked(p.preprocessor.transform_obs(obsg))
    a = torch.randn(2, 5, ACTION_DIM, requires_grad=True)
    z, _ = p.wm.rollout(obs_0=trans0, act=a, z_goal=zg["visual"])
    p.objective_fn(z, zg).mean().backward()
    assert a.grad is not None
    assert float(a.grad.norm()) > 0.0


def test_actions_actually_move_over_opt_steps():
    p, _ = build(opt_steps=5, action_noise=0.0)
    torch.manual_seed(0)
    before = None

    orig = p.wm.rollout

    def spy(**kw):
        nonlocal before
        if before is None:
            before = kw["act"].detach().clone()
        return orig(**kw)

    p.wm.rollout = spy
    actions, _ = p.plan(make_obs(b=2), make_obs(b=2))
    assert not torch.allclose(before, actions.detach())


def test_refuses_a_rank_based_vote_objective():
    with pytest.raises(AssertionError, match="zero gradient"):
        build(objective_fn=create_vote_objective_fn(5, "cvar", lam=1.0))


@pytest.mark.parametrize("rule", ["mean", "borda", "median", "cvar", "max"])
def test_refuses_every_vote_rule(rule):
    with pytest.raises(AssertionError, match="zero gradient"):
        build(objective_fn=create_vote_objective_fn(3, rule))


def test_plain_objective_is_accepted():
    p, _ = build(objective_fn=create_objective_fn(alpha=0, base=2, mode="last"))
    assert p.objective_fn.__name__ == "objective_fn_last"


def test_config_matches_the_spec_budget():
    import yaml

    c = yaml.safe_load(open(REPO / "conf" / "plan_gd.yaml"))
    s = c["planner"]["sub_planner"]
    assert s["target"] == "planning.gd.GDPlanner"
    # NOT conf/planner/mpc_gd.yaml's 1000/10: that fires the real-env evaluator 100x
    # per plan() x 10 MPC iters and does not fit the 3h55 allocation.
    assert s["opt_steps"] == 300 and s["eval_every"] == 100
    assert c["objective"]["_target_"] == "planning.objectives.create_objective_fn"
    assert c["planner"]["max_iter"] == 10
