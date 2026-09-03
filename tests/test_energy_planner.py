"""T7 - the energy/ranking head and its CEM leaf.

Four properties are pinned, and each one corresponds to a way this arm could silently be
a no-op or a lie:

  1. `_ScoredCEMPlanner.plan` is `CEMPlanner.plan`. The score hook could not be added to
     `planning/cem.py` (another agent owns it this wave), so the loop is a COPY -- and a
     copy that has drifted would make every energy number incomparable to the archive.
     From the same RNG state, the copy must return `mu` bit-for-bit.
  2. `EnergyCEMPlanner` REFUSES to plan without a trained head on disk, and the head it
     loads is provably the head that was saved (checksum). `path_int` was absent from
     every checkpoint, silently randomly initialised at plan time, and produced a
     conclusion that had to be retracted (diary 2026-09-03 sec 13.3). A fresh init must
     not be able to masquerade as a trained head.
  3. The energy leaf never calls `wm.rollout` -- that is the whole design (no compounding,
     no `_roll_pose`), so a rollout in the score path would mean the arm is not T7.
  4. Gradient reaches every head parameter after one backward, and the negatives are the
     distribution the planner actually proposes (`planning/cem.py:99-106`).

Run:  PYTHONPATH=<repo> python -m pytest tests/test_energy_planner.py -q
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from planning.cem import CEMPlanner  # noqa: E402
from planning.energy import (  # noqa: E402
    EnergyCEMPlanner,
    SeqEnergyHead,
    _ScoredCEMPlanner,
    flatten_z,
    load_energy_head,
    save_energy_head,
    state_dict_checksum,
)
from planning.objectives import create_objective_fn  # noqa: E402
from stub_wm import (  # noqa: E402
    StubPreprocessor,
    StubWandb,
    StubWorldModel,
    make_obs,
)

ACTION_DIM = 10
EMB = 8
HORIZON = 5


def _kw(**over):
    kw = dict(
        horizon=HORIZON,
        topk=3,
        num_samples=7,
        var_scale=1,
        opt_steps=3,
        eval_every=100,
    )
    kw.update(over)
    return kw


def build(cls, wm=None, **over):
    wm = wm if wm is not None else StubWorldModel(emb_dim=EMB, action_dim=ACTION_DIM)
    extra = {k: over.pop(k) for k in list(over) if k in ("energy_head", "energy_ckpt")}
    return cls(
        wm=wm,
        action_dim=ACTION_DIM,
        objective_fn=create_objective_fn(alpha=0, base=2, mode="last"),
        preprocessor=StubPreprocessor(),
        evaluator=None,
        wandb_run=StubWandb(),
        log_filename=None,
        name="mpc_cem",              # the planner config carries name=/target=
        target="planning.energy.EnergyCEMPlanner",
        env=None,
        **_kw(**over),
        **extra,
    )


def head(**over):
    kw = dict(emb_dim=EMB, horizon=HORIZON, act_dim=ACTION_DIM, hidden=16)
    kw.update(over)
    torch.manual_seed(0)
    return SeqEnergyHead(**kw)


# ------------------------------------------------------------------ 1. the copy is exact


def test_scored_cem_reproduces_cem_bit_for_bit():
    """The refactored loop must be the inline loop. Same wm, same seed, same mu."""
    obs_0, obs_g = make_obs(b=2), make_obs(b=2)
    wm = StubWorldModel(emb_dim=EMB, action_dim=ACTION_DIM)

    torch.manual_seed(1234)
    mu_ref, len_ref = build(CEMPlanner, wm=wm).plan(obs_0, obs_g)
    n_rollouts_ref = len(wm.rollout_batches)

    wm.rollout_batches.clear()
    torch.manual_seed(1234)
    mu_new, len_new = build(_ScoredCEMPlanner, wm=wm).plan(obs_0, obs_g)

    assert torch.equal(mu_ref, mu_new), (mu_ref - mu_new).abs().max()
    assert np.array_equal(len_ref, len_new)
    assert len(wm.rollout_batches) == n_rollouts_ref == 2 * 3   # n_evals * opt_steps


def test_scored_cem_contract():
    obs_0, obs_g = make_obs(b=2), make_obs(b=2)
    p = build(_ScoredCEMPlanner)
    p.horizon = 5                                    # plan.py sets this from goal_H
    p.logging_prefix = "plan_0"
    mu, lens = p.plan(obs_0, obs_g)
    assert isinstance(mu, torch.Tensor) and mu.shape == (2, HORIZON, ACTION_DIM)
    assert isinstance(lens, np.ndarray) and lens.shape == (2,)


# ------------------------------------------------------------------ 2. anti-path_int


def test_energy_planner_raises_without_head(monkeypatch):
    monkeypatch.delenv("ENERGY_CKPT", raising=False)
    with pytest.raises(ValueError, match="TRAINED head"):
        build(EnergyCEMPlanner)


def test_energy_planner_raises_on_missing_file(monkeypatch):
    monkeypatch.delenv("ENERGY_CKPT", raising=False)
    with pytest.raises(FileNotFoundError):
        build(EnergyCEMPlanner, energy_ckpt="/nonexistent/energy_head.pt")


def test_saved_head_is_the_loaded_head(tmp_path):
    """The checksum printed at save must be the checksum printed at load, and a fresh
    init must NOT be able to match it."""
    h = head()
    with torch.no_grad():                     # make it distinguishable from any init
        h.z_scale.fill_(3.25)
        for p in h.parameters():
            p.add_(0.01)
    out = tmp_path / "energy.pt"
    saved = save_energy_head(h, str(out), run="stub", step=7, mode="rank", val_top1=0.5)
    back = load_energy_head(str(out))
    assert back.checksum() == pytest.approx(saved, abs=1e-6)
    assert state_dict_checksum(back.state_dict()) == pytest.approx(saved, abs=1e-6)
    assert float(back.z_scale) == 3.25
    fresh = SeqEnergyHead(**h.arch)
    assert abs(fresh.checksum() - saved) > 1e-3, "a fresh init matched a trained checksum"
    # the file really carries the parameters, not just the arch
    payload = torch.load(str(out), map_location="cpu")
    assert set(payload["state_dict"]) == set(h.state_dict())
    assert payload["state_dict"]["mlp.0.weight"].shape == (16, 3 * EMB + 16)


def test_load_refuses_a_tampered_checksum(tmp_path):
    h = head()
    out = tmp_path / "energy.pt"
    save_energy_head(h, str(out), run="stub", step=1, mode="rank", val_top1=0.0)
    payload = torch.load(str(out), map_location="cpu")
    payload["state_dict"]["mlp.0.bias"] += 1.0        # a different head, same checksum key
    torch.save(payload, str(out))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_energy_head(str(out))


def test_load_refuses_a_foreign_state_dict(tmp_path):
    h = head()
    out = tmp_path / "energy.pt"
    save_energy_head(h, str(out), run="stub", step=1, mode="rank", val_top1=0.0)
    payload = torch.load(str(out), map_location="cpu")
    payload["arch"]["hidden"] = 32                     # arch and weights disagree
    torch.save(payload, str(out))
    with pytest.raises(RuntimeError):
        load_energy_head(str(out))


def test_env_var_is_a_fallback_path(tmp_path, monkeypatch):
    h = head()
    out = tmp_path / "energy.pt"
    save_energy_head(h, str(out), run="stub", step=1, mode="rank", val_top1=0.0)
    monkeypatch.setenv("ENERGY_CKPT", str(out))
    p = build(EnergyCEMPlanner)
    assert p.energy_head.checksum() == pytest.approx(h.checksum(), abs=1e-6)


def test_horizon_mismatch_is_refused_at_score_time():
    """A head trained for 3-step blocks must not silently score 5-step proposals."""
    p = build(EnergyCEMPlanner, energy_head=head(horizon=3))
    with pytest.raises(ValueError, match="horizon"):
        p.plan(make_obs(b=1), make_obs(b=1))


def test_horizon_mismatch_is_refused_at_load_time(tmp_path):
    f = tmp_path / "h3.pt"
    save_energy_head(head(horizon=3), str(f), run="s", step=1, mode="rank", val_top1=0.0)
    assert load_energy_head(str(f), expect_horizon=3).horizon == 3
    with pytest.raises(ValueError, match="horizon"):
        load_energy_head(str(f), expect_horizon=5)


# ------------------------------------------------------------------ 3. no rollout


def test_energy_leaf_never_rolls_out():
    wm = StubWorldModel(emb_dim=EMB, action_dim=ACTION_DIM)
    calls = {"n": 0}
    real = wm.rollout

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    wm.rollout = counting
    p = build(EnergyCEMPlanner, wm=wm, energy_head=head())
    mu, lens = p.plan(make_obs(b=2), make_obs(b=2))
    assert calls["n"] == 0 and wm.rollout_batches == []
    assert mu.shape == (2, HORIZON, ACTION_DIM)
    assert isinstance(lens, np.ndarray)


def test_energy_leaf_orders_candidates_by_the_head():
    """The elites CEM keeps must be the head's argsort, i.e. the head really is the leaf."""
    wm = StubWorldModel(emb_dim=EMB, action_dim=ACTION_DIM)
    h = head()
    p = build(EnergyCEMPlanner, wm=wm, energy_head=h, opt_steps=1, num_samples=16, topk=4)
    obs_0, obs_g = make_obs(b=1), make_obs(b=1)
    torch.manual_seed(7)
    mu, _ = p.plan(obs_0, obs_g)
    # replay the same proposal and check mu == mean of the head's top-4
    torch.manual_seed(7)
    action = torch.randn(16, HORIZON, ACTION_DIM) * torch.ones(HORIZON, ACTION_DIM)
    action[0] = torch.zeros(HORIZON, ACTION_DIM)
    z0 = flatten_z(wm.encode_obs_linked(obs_0)["visual"]).expand(16, EMB)
    zg = flatten_z(wm.encode_obs_linked(obs_g)["visual"]).expand(16, EMB)
    with torch.no_grad():
        E = h(z0, action, zg)
    want = action[torch.argsort(E)[:4]].mean(dim=0)
    assert torch.allclose(mu[0], want, atol=1e-6)


def test_flatten_z_accepts_the_m1_ensemble_shape():
    z = torch.randn(4, 1, 1, EMB)              # (b, t, M=1 or p=1, D)
    assert flatten_z(z).shape == (4, EMB)
    assert torch.equal(flatten_z(z), z[:, -1, 0])
    with pytest.raises(AssertionError):
        flatten_z(torch.randn(4, 1, 3, EMB))   # patch tokens need their own head


# ------------------------------------------------------------------ 4. the head itself


def test_gradient_reaches_every_head_parameter():
    h = head()
    B, M = 6, 7
    z0, zg = torch.randn(B, EMB), torch.randn(B, EMB)
    a = torch.randn(B, 1 + M, HORIZON, ACTION_DIM)
    z0f = z0[:, None].expand(B, 1 + M, EMB).reshape(-1, EMB)
    zgf = zg[:, None].expand(B, 1 + M, EMB).reshape(-1, EMB)
    E = h(z0f, a.reshape(-1, HORIZON, ACTION_DIM), zgf).view(B, 1 + M)
    loss = F.cross_entropy(-E, torch.zeros(B, dtype=torch.long))
    loss.backward()
    rep = h.grad_report()
    assert rep and all(g is not None and g > 0 for _, g in rep), rep
    assert len(rep) == len(list(h.parameters()))


def test_head_is_a_function_of_the_action_block():
    h = head()
    z0, zg = torch.randn(2, EMB), torch.randn(2, EMB)
    a = torch.randn(2, HORIZON, ACTION_DIM)
    with torch.no_grad():
        e1 = h(z0, a, zg)
        e2 = h(z0, a + 1.0, zg)
    assert (e1 - e2).abs().max() > 0, "the head ignores the action sequence"


def test_head_rejects_the_wrong_action_shape():
    h = head()
    with pytest.raises(AssertionError):
        h(torch.randn(2, EMB), torch.randn(2, HORIZON + 1, ACTION_DIM), torch.randn(2, EMB))


def test_negatives_match_the_cem_proposal():
    """planning/cem.py:99-106 draws `randn * sigma + mu` with var_scale=1: N(0,1) at opt
    step 0, and a shrinking ball around the elite mean later."""
    train_energy = pytest.importorskip("train_energy")
    torch.manual_seed(0)
    a_pos = torch.randn(512, HORIZON, ACTION_DIM)
    g = torch.Generator().manual_seed(0)
    neg = train_energy.sample_negatives(a_pos, 63, g)
    assert neg.shape == (512, 63, HORIZON, ACTION_DIM)
    far, near = neg[:, :32], neg[:, 32:]
    assert abs(float(far.mean())) < 0.02 and abs(float(far.std()) - 1.0) < 0.02
    # the near half is centred on the positive, the far half is not
    d_near = (near - a_pos[:, None]).pow(2).mean().sqrt()
    d_far = (far - a_pos[:, None]).pow(2).mean().sqrt()
    assert float(d_near) < float(d_far)
    assert 0.1 <= float((near - a_pos[:, None]).std()) <= 1.0
    cand = train_energy.candidates(a_pos, 63, torch.Generator().manual_seed(0))
    assert cand.shape == (512, 64, HORIZON, ACTION_DIM)
    assert torch.equal(cand[:, 0], a_pos), "the executed sequence must be index 0"


def test_window_geometry_is_the_planner_geometry():
    train_energy = pytest.importorskip("train_energy")
    assert (train_energy.NUM_HIST, train_energy.NUM_PRED, train_energy.FRAMESKIP) == (3, 5, 5)
    # anchor -> goal is exactly goal_H = 5 model steps, and the action rows in between
    assert train_energy.IG - train_energy.I0 == 5
    assert train_energy.RAWG - train_energy.RAW0 == 5 * 5
    assert train_energy.WIN_RAW == 40

    class _Traj:                                    # T - 40 + 1 windows per episode
        lens = [45, 39, 40, 100]

        def get_seq_length(self, i):
            return self.lens[i]

        def __len__(self):
            return len(self.lens)

    assert train_energy.count_windows(_Traj()) == 6 + 0 + 1 + 61


def test_last_bias_is_unidentified_under_the_rank_objective():
    """The cross-entropy is a softmax over the 64 candidates of ONE anchor, and the head's
    last bias adds the same constant to all of them, so dL/db is EXACTLY 0.

    This is pinned because it broke a real run: `train_energy.py`'s grad-liveness
    assertion fired on seed s6 (`mlp.4.bias grad_norm=0.0`) while the other seven seeds
    got float dust ~4e-8 and passed. The liveness check now exempts this one parameter
    under `mode="rank"` and nothing else, instead of being loosened into a coin flip. It
    is harmless for planning: CEM only ever argsorts, and a global constant does not
    change an ordering.
    """
    h = head()
    B, K = 4, 64
    z0, zg = torch.randn(B, EMB), torch.randn(B, EMB)
    a = torch.randn(B, K, HORIZON, ACTION_DIM)
    z0f = z0[:, None].expand(B, K, EMB).reshape(-1, EMB)
    zgf = zg[:, None].expand(B, K, EMB).reshape(-1, EMB)
    E = h(z0f, a.reshape(-1, HORIZON, ACTION_DIM), zgf).view(B, K)
    F.cross_entropy(-E, torch.zeros(B, dtype=torch.long)).backward()
    last = h.mlp[-1]
    assert float(last.bias.grad.abs().max()) < 1e-7
    assert float(last.weight.grad.norm()) > 0        # everything else IS identified
    # and the constant genuinely cannot change what CEM selects
    with torch.no_grad():
        before = torch.argsort(h(z0f, a.reshape(-1, HORIZON, ACTION_DIM), zgf))
        last.bias.add_(7.0)
        after = torch.argsort(h(z0f, a.reshape(-1, HORIZON, ACTION_DIM), zgf))
    assert torch.equal(before, after)


def test_distill_objective_does_identify_the_last_bias():
    """The control regresses a value, so its offset IS identified -- which is why the
    exemption above is mode-specific and not a blanket loosening."""
    h = head()
    B, K = 4, 8
    z0, zg = torch.randn(B, EMB), torch.randn(B, EMB)
    a = torch.randn(B, K, HORIZON, ACTION_DIM)
    z0f = z0[:, None].expand(B, K, EMB).reshape(-1, EMB)
    zgf = zg[:, None].expand(B, K, EMB).reshape(-1, EMB)
    E = h(z0f, a.reshape(-1, HORIZON, ACTION_DIM), zgf).view(B, K)
    F.mse_loss(E, torch.rand(B, K) + 3.0).backward()
    assert float(h.mlp[-1].bias.grad.abs().max()) > 0
