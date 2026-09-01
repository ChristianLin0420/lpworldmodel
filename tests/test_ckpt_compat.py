"""Checkpoint format compatibility between train.py (writer) and plan.py (reader).

train.py now stores state_dicts rather than pickled modules, so plan.py has to
rebuild every submodule from train_cfg before loading. If these two drift, the
symptom is a CEM eval that silently plans with a freshly initialised model, which
is far worse than a crash -- hence the weight-equality assertions below.
"""
import torch

import plan as plan_mod
from lpwm_build import build, load_cfg, seed_all, synthetic_batch

ARM_OVERRIDES = {
    "upstream": [],
    "kwta": ["kwta_k=8"],
    "gate": ["predictor=ltv", "gate_input=support", "gate_norm=softmax"],
    "union": ["predictor=ltv", "n_heads=4", "head_entropy_coef=0.1"],
}


def write_ckpt(path, model, cfg, epoch=1, batch_idx=0):
    """Same payload shape as train.py save_ckpt, minus the optimizers."""
    ckpt = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "rng": {"torch": torch.get_rng_state()},
        "encoder": model.encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "action_encoder": model.action_encoder.state_dict(),
        "proprio_encoder": model.proprio_encoder.state_dict(),
    }
    if model.link is not None:
        ckpt["link"] = model.link.state_dict()
    torch.save(ckpt, path)
    return ckpt


def _build(overrides, seed=0):
    cfg = load_cfg(overrides)
    seed_all(seed)
    model, _ = build(cfg)
    model.eval()  # VWorldModel.eval() returns None, so this cannot be chained
    return cfg, model


def _roundtrip(tmp_path, overrides, name):
    cfg, ref = _build(overrides)
    p = tmp_path / f"{name}.pth"
    write_ckpt(p, ref, cfg)
    got = plan_mod.load_model(p, cfg, cfg.num_action_repeat, torch.device("cpu"))
    got.eval()
    return cfg, ref, got


def test_state_dict_checkpoint_restores_identical_weights(tmp_path):
    _, ref, got = _roundtrip(tmp_path, ARM_OVERRIDES["upstream"], "upstream")
    a = dict(ref.encoder.state_dict())
    b = dict(got.encoder.state_dict())
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a), "encoder weights were not restored"
    pa, pb = ref.predictor.state_dict(), got.predictor.state_dict()
    assert all(torch.equal(pa[k], pb[k]) for k in pa), "predictor weights not restored"


def test_reloaded_model_reproduces_the_original_forward(tmp_path):
    """The end-to-end statement: identical inputs give identical predictions."""
    cfg, ref, got = _roundtrip(tmp_path, ARM_OVERRIDES["upstream"], "fwd")
    gen = torch.Generator().manual_seed(99)
    obs, act = synthetic_batch(cfg, 2, gen)
    with torch.no_grad():
        za = ref.encode_obs_linked(obs)["visual"]
        zb = got.encode_obs_linked(obs)["visual"]
    assert torch.equal(za, zb)


def test_accepts_a_str_path_as_well_as_a_path(tmp_path):
    cfg, ref = _build([])
    p = tmp_path / "str.pth"
    write_ckpt(p, ref, cfg)
    assert plan_mod.load_model(str(p), cfg, cfg.num_action_repeat, torch.device("cpu"))


def test_legacy_pickled_module_checkpoints_still_load(tmp_path):
    """Checkpoints written before the state_dict switch pickle whole modules."""
    cfg, ref = _build([])
    p = tmp_path / "legacy.pth"
    torch.save({
        "epoch": 7,
        "encoder": ref.encoder,
        "predictor": ref.predictor,
        "action_encoder": ref.action_encoder,
        "proprio_encoder": ref.proprio_encoder,
        "link": ref.link,
    }, p)
    got = plan_mod.load_model(p, cfg, cfg.num_action_repeat, torch.device("cpu"))
    a, b = ref.encoder.state_dict(), got.encoder.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_intervention_arms_round_trip(tmp_path):
    """Each intervention adds or reshapes parameters (k-WTA none, gate none,
    union head W_heads), so every arm's checkpoint must reload strictly."""
    for name, ov in ARM_OVERRIDES.items():
        cfg, ref, got = _roundtrip(tmp_path, ov, name)
        assert got.n_heads == ref.n_heads, name
        assert getattr(got.link, "kwta_k", None) == getattr(ref.link, "kwta_k", None), name
        a, b = ref.predictor.state_dict(), got.predictor.state_dict()
        assert set(a) == set(b), f"{name}: predictor keys differ"
        assert all(torch.equal(a[k], b[k]) for k in a), f"{name}: predictor weights differ"


def test_union_head_checkpoint_carries_every_head(tmp_path):
    """A J=4 checkpoint that reloaded only W would plan with 1 of 4 heads."""
    _, ref, got = _roundtrip(tmp_path, ARM_OVERRIDES["union"], "heads")
    assert len(got.predictor.W_heads) == len(ref.predictor.W_heads) == 3
    for i, (x, y) in enumerate(zip(ref.predictor.W_heads, got.predictor.W_heads)):
        assert torch.equal(x.weight, y.weight), f"head {i + 1} weights differ"


def test_j1_checkpoint_has_no_union_head_keys(tmp_path):
    """The J=1 state dict must stay interchangeable with an upstream checkpoint."""
    cfg, ref = _build(["predictor=ltv", "n_heads=1"])
    ckpt = write_ckpt(tmp_path / "j1.pth", ref, cfg)
    assert not any("W_heads" in k for k in ckpt["predictor"])


def test_probe_cell_has_one_patch_at_the_loss_site():
    """Plan item 1a: P == 1 was asserted only by a comment. The union head's min
    is over (sample, timestep) with patches meaned inside, so P is load-bearing."""
    cfg, model = _build([])
    gen = torch.Generator().manual_seed(3)
    obs, act = synthetic_batch(cfg, 2, gen)
    with torch.no_grad():
        z = model.encode_obs_linked(obs)["visual"]
    assert z.shape[2] == 1, f"expected P == 1 for cls features, got {z.shape[2]}"
    assert model.encoder.num_patches == 1
