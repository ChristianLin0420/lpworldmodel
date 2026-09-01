"""Union-head rollout: which head advances the state when there is no target.

CEM never sees the next observation, so it cannot run the training argmin. The
rule here is a min over heads against the GOAL latent. The tests that matter are
(a) J=1 is untouched, and (b) at J>1 the chosen head is really the goal-nearest
one -- if it silently fell back to head 0, a J-head arm would be evaluated with
the head that owns ~1/J of transitions and would lose for the wrong reason.
"""
import pytest
import torch

from lpwm_build import build, load_cfg, seed_all, synthetic_batch

LTV = ["predictor=ltv"]


def _model(overrides, seed=0):
    cfg = load_cfg(overrides)
    seed_all(seed)
    model, _ = build(cfg)
    model.eval()
    return cfg, model


def _rollout_inputs(cfg, model, b=2, t=5, gen_seed=5):
    gen = torch.Generator().manual_seed(gen_seed)
    obs, _ = synthetic_batch(cfg, b, gen)
    obs_0 = {k: v[:, : cfg.num_hist] for k, v in obs.items()}
    act = torch.rand(b, t, 2 * cfg.frameskip, generator=gen)
    with torch.no_grad():
        z_goal = model.encode_obs_linked(
            {k: v[:, -1:] for k, v in obs.items()}
        )["visual"]
    return obs_0, act, z_goal


# --- J = 1 must be untouched -----------------------------------------------------

def test_j1_ignores_z_goal_exactly():
    """The default arm has to stay bit-identical: matched controls are only
    comparable if passing z_goal cannot perturb them."""
    cfg, model = _model(LTV + ["n_heads=1"])
    obs_0, act, z_goal = _rollout_inputs(cfg, model)
    with torch.no_grad():
        a = model.rollout(obs_0, act)[1]
        b = model.rollout(obs_0, act, z_goal=z_goal)[1]
    assert torch.equal(a, b)


def test_j1_rollout_shape_is_unchanged():
    cfg, model = _model(LTV + ["n_heads=1"])
    obs_0, act, _ = _rollout_inputs(cfg, model, t=5)
    with torch.no_grad():
        _, z = model.rollout(obs_0, act)
    assert z.shape[1] == act.shape[1] + 1


# --- J > 1 behaviour -------------------------------------------------------------

def test_union_rollout_runs_and_keeps_the_shape():
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, z_goal = _rollout_inputs(cfg, model)
    with torch.no_grad():
        _, z = model.rollout(obs_0, act, z_goal=z_goal)
    assert z.shape[1] == act.shape[1] + 1
    assert torch.isfinite(z).all()


def _heads_and_goal(model, obs_0, act):
    """The pieces of one rollout step: every head's linked prediction, plus the
    action window _predict_next_adaln would have used."""
    emb = model._link(model.encode_obs(obs_0)["visual"])
    act_emb = model.encode_act(act)[:, : emb.shape[1]]
    z_all = model._link(model.predictor.forward_heads(emb, act_emb)[:, :, -1:])
    return emb, act_emb, z_all                                   # (J,b,1,p,d)


def test_rollout_depends_on_the_goal():
    """Aiming at two different heads' own predictions must give two different
    rollouts. Note head 0 winning for some particular goal is legitimate -- with
    muP init its output norm can simply be closest -- so a plain 'union differs
    from head 0' check would be flaky."""
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, _ = _rollout_inputs(cfg, model)
    with torch.no_grad():
        _, _, z_all = _heads_and_goal(model, obs_0, act)
        a = model.rollout(obs_0, act, z_goal=z_all[0, :, 0])[1]
        b = model.rollout(obs_0, act, z_goal=z_all[3, :, 0])[1]
    assert not torch.allclose(a, b), "rollout ignores the goal"


def test_chosen_head_is_the_goal_nearest_one():
    """Directly checks the rule on a single step, against a brute-force argmin."""
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, z_goal = _rollout_inputs(cfg, model)
    with torch.no_grad():
        emb, act_emb, z_all = _heads_and_goal(model, obs_0, act)
        got = model._optimistic_next(emb, act_emb, z_goal)
        d = ((z_all - z_goal.unsqueeze(0)) ** 2).mean(dim=(-1, -2))  # (J,b,1)
    for i in range(z_all.shape[1]):
        j = int(d[:, i, 0].argmin())
        assert torch.equal(got[i], z_all[j, i]), f"sample {i} did not take head {j}"


def test_chosen_head_is_never_beaten_by_another_head():
    """The invariant, stated as an inequality so it holds under ties too."""
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, z_goal = _rollout_inputs(cfg, model)
    with torch.no_grad():
        emb, act_emb, z_all = _heads_and_goal(model, obs_0, act)
        got = model._optimistic_next(emb, act_emb, z_goal)
        d_got = ((got - z_goal) ** 2).mean(dim=(-1, -2))                  # (b,1)
        d_all = ((z_all - z_goal.unsqueeze(0)) ** 2).mean(dim=(-1, -2))   # (J,b,1)
    assert torch.all(d_got <= d_all.min(dim=0).values + 1e-6)


def test_head_selection_accepts_both_goal_shapes():
    """Callers hand over (b, p, d) or (b, 1, p, d) depending on the code path."""
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, z_goal = _rollout_inputs(cfg, model)
    with torch.no_grad():
        a = model.rollout(obs_0, act, z_goal=z_goal)[1]
        b = model.rollout(obs_0, act, z_goal=z_goal.squeeze(1))[1]
    assert torch.equal(a, b)


def test_different_goals_select_different_heads():
    """The selection must actually depend on the goal, not just on the state."""
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, _ = _rollout_inputs(cfg, model)
    with torch.no_grad():
        emb, act_emb, z_all = _heads_and_goal(model, obs_0, act)
        for j in range(z_all.shape[0]):
            # aim exactly at head j's own prediction; it must then be chosen
            got = model._optimistic_next(emb, act_emb, z_all[j, :, 0])
            assert torch.equal(got, z_all[j]), f"aiming at head {j} did not pick it"


def test_gradients_flow_through_the_selected_head():
    """planning/gd.py backprops through rollout, so gather must not sever it."""
    cfg, model = _model(LTV + ["n_heads=4"])
    obs_0, act, z_goal = _rollout_inputs(cfg, model)
    act = act.clone().requires_grad_(True)
    _, z = model.rollout(obs_0, act, z_goal=z_goal)
    z.sum().backward()
    assert act.grad is not None and torch.isfinite(act.grad).all()
    assert (act.grad != 0).any(), "no gradient reached the actions"


@pytest.mark.parametrize("n_heads", [1, 4])
def test_planner_call_sites_pass_a_goal(n_heads):
    """Guards the wiring: a call site that omits z_goal silently reverts to head 0."""
    import inspect

    import planning.cem as cem
    import planning.gd as gd
    import planning.evaluator as ev

    for mod in (cem, gd, ev):
        src = inspect.getsource(mod)
        assert "self.wm.rollout(" in src
        assert "z_goal=" in src, f"{mod.__name__} calls rollout without z_goal"
