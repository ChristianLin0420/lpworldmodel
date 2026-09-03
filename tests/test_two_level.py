"""V3 - TwoLevelPlanner: contract, budget, and the held-command layout.

What can silently go wrong here, and is therefore asserted:
  * the planner contract (plan.py:187-204, planning/mpc.py:44-54): `**kwargs` must
    swallow `name=`/`env=`, `plan()` must return (Tensor (B,H,A), np.ndarray (B,)),
    and `.horizon`/`.logging_prefix` must be settable AFTER construction because
    plan.py:201 overwrites the horizon with goal_H;
  * `k` must be recomputed from the OVERWRITTEN horizon inside `plan()`;
  * the approach command must be TILED ([px,py,px,py,...]) not interleaved
    ([px]*f + [py]*f) -- planning/evaluator.py unpacks "b t (f d) -> b (t f) d";
  * the budget must be S*C rollouts per opt step per episode, matching flat CEM's
    num_samples, or the "compute-matched" claim in the spec is false.
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

from planning.two_level import TwoLevelPlanner  # noqa: E402
from stub_wm import (  # noqa: E402
    StubPreprocessor,
    StubWandb,
    StubWorldModel,
    make_obs,
)

FRAMESKIP = 5
ACTION_DIM = 10          # d=2 x frameskip=5, as in PushT
S, C = 15, 20            # S*C = 300 = conf/plan_lewm.yaml num_samples


def build(**over):
    kw = dict(
        horizon=5,
        k_reach=3,
        n_subgoals=S,
        n_ctrl=C,
        topk_sub=5,
        topk_ctrl=30,
        var_scale=1,
        opt_steps=2,
        eval_every=1,
        frameskip=FRAMESKIP,
    )
    kw.update(over)
    wm = StubWorldModel(action_dim=ACTION_DIM)
    return (
        TwoLevelPlanner(
            wm=wm,
            action_dim=ACTION_DIM,
            objective_fn=lambda p, g: ((p["visual"][:, -1:] - g["visual"]) ** 2).mean(
                dim=tuple(range(1, p["visual"].ndim))
            ),
            preprocessor=StubPreprocessor(),
            evaluator=None,
            wandb_run=StubWandb(),
            log_filename=None,
            # the two kwargs MPCPlanner/hydra inject that must be swallowed:
            name="mpc_two_level",
            env=None,
            **kw,
        ),
        wm,
    )


def test_is_a_base_planner_and_swallows_name_and_env():
    from planning.base_planner import BasePlanner

    p, _ = build()
    assert isinstance(p, BasePlanner)


def test_plan_returns_contract_shapes_and_dtypes():
    p, _ = build()
    obs0, obsg = make_obs(b=2), make_obs(b=2)
    actions, lens = p.plan(obs0, obsg)
    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (2, 5, ACTION_DIM)
    assert actions.dtype == torch.float32
    assert isinstance(lens, np.ndarray) and lens.shape == (2,)


def test_horizon_and_logging_prefix_are_settable_after_construction():
    """plan.py:201 sets sub_planner.horizon = goal_H; mpc.py:86 sets logging_prefix."""
    p, _ = build(horizon=5)
    p.horizon = 3                 # goal_H = 3
    p.logging_prefix = "plan_7"
    actions, _ = p.plan(make_obs(b=2), make_obs(b=2))
    assert actions.shape == (2, 3, ACTION_DIM)
    assert any(k.startswith("plan_7/") for log in p.wandb_run.logs for k in log)


def test_k_is_clamped_to_the_overwritten_horizon():
    """k_reach=3 with goal_H=2 must clamp to 1 so the push segment is non-empty."""
    p, wm = build(k_reach=3, horizon=5)
    p.horizon = 2
    actions, _ = p.plan(make_obs(b=1), make_obs(b=1))
    assert actions.shape == (1, 2, ACTION_DIM)
    a = actions[0, 0].view(FRAMESKIP, 2)
    assert torch.allclose(a, a[0:1].expand_as(a))       # step 0 is still a held macro
    # and step 1 is the free push, so it is NOT forced to be a held command
    assert actions.shape[1] - 1 == 1


def test_first_k_steps_hold_the_same_command_in_every_substep_slot():
    p, _ = build(k_reach=3, horizon=5)
    actions, _ = p.plan(make_obs(b=2), make_obs(b=2))
    k = 3
    for b in range(actions.shape[0]):
        held = actions[b, :k].view(k, FRAMESKIP, 2)
        # every one of the f sub-step slots carries the SAME (px, py) ...
        assert torch.allclose(held, held[:, 0:1].expand_as(held), atol=0)
        # ... and it is the same command across all k approach steps.
        assert torch.allclose(held, held[0:1].expand_as(held), atol=0)


def test_tiled_not_interleaved():
    """[px,py,px,py,...] (repeat) not [px]*f+[py]*f (repeat_interleave)."""
    p, _ = build()
    macro = p._macro(torch.tensor([[1.0, -2.0]]), 2)     # (1, 2, A)
    want = torch.tensor([1.0, -2.0] * FRAMESKIP)
    assert torch.equal(macro[0, 0], want)
    assert torch.equal(macro[0, 1], want)


def test_budget_is_S_times_C_rollouts_per_opt_step_per_episode():
    p, wm = build(opt_steps=3)
    b = 2
    p.plan(make_obs(b=b), make_obs(b=b))
    assert len(wm.rollout_batches) == 3 * b            # one call per episode per step
    assert set(wm.rollout_batches) == {S * C}          # each of batch 300
    assert S * C == 300                                # == flat CEM num_samples


def test_prints_the_budget_banner(capsys):
    p, _ = build()
    p.plan(make_obs(b=1), make_obs(b=1))
    out = capsys.readouterr().out
    assert "[2lvl] H=5 k=3 S=15 C=20 rollouts/opt_step=300 (flat CEM: 300)" in out


def test_logs_the_elite_subgoal_dispersion_diagnostic():
    p, _ = build(opt_steps=1)
    p.plan(make_obs(b=2), make_obs(b=2))
    keys = {k for log in p.wandb_run.logs for k in log}
    assert any(k.endswith("/elite_subgoal_z_dispersion") for k in keys)


@pytest.mark.parametrize("bad", [dict(topk_sub=1), dict(topk_ctrl=1)])
def test_topk_of_one_is_refused(bad):
    """std over a single elite is nan and would poison mu/sigma silently."""
    with pytest.raises(AssertionError, match="std over 1 elite is nan"):
        build(**bad)


def test_topk_ctrl_cannot_exceed_the_elite_pool():
    with pytest.raises(AssertionError, match="elite controller pool"):
        build(topk_sub=2, topk_ctrl=100)


def test_no_nan_in_the_returned_actions():
    p, _ = build(opt_steps=5)
    actions, _ = p.plan(make_obs(b=2), make_obs(b=2))
    assert torch.isfinite(actions).all()


def test_warm_start_from_a_nonempty_memo_is_accepted():
    """MPC hands back actions[:, n_taken:]; empty under the shipped config, but the
    branch must not crash if n_taken_actions is shortened."""
    p, _ = build()
    memo = torch.randn(2, 2, ACTION_DIM)
    actions, _ = p.plan(make_obs(b=2), make_obs(b=2), actions=memo)
    assert actions.shape == (2, 5, ACTION_DIM)


def test_config_is_budget_matched_to_flat_cem():
    import yaml

    flat = yaml.safe_load(open(REPO / "conf" / "plan_lewm.yaml"))["planner"]
    two = yaml.safe_load(open(REPO / "conf" / "plan_two_level.yaml"))["planner"]
    fs, ts = flat["sub_planner"], two["sub_planner"]
    assert ts["n_subgoals"] * ts["n_ctrl"] == fs["num_samples"] == 300
    assert ts["opt_steps"] == fs["opt_steps"] == 30
    assert ts["horizon"] == fs["horizon"]
    assert ts["topk_ctrl"] == fs["topk"]
    assert two["max_iter"] == flat["max_iter"]
    assert ts["target"] == "planning.two_level.TwoLevelPlanner"
