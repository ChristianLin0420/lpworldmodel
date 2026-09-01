"""Every arm of the campaign, composed and stepped once on CPU.

Cheap insurance: a mistyped hydra key or a flag that silently fails to reach a
module would otherwise surface only after a run has queued, trained and evalled.
Each arm here is exactly the override list the launcher will use.
"""
import math

import pytest
import torch

from lpwm_build import build, load_cfg, seed_all, synthetic_batch

D = 384
K_002 = int(round(0.02 * D))          # k/D = 0.02 arm
K_MATCHED = int(round(0.50 * D))      # placeholder until the probe measures rho

ARMS = {
    # Step 2 -- k-WTA codes (3 arms)
    "step2/upstream":      [],
    "step2/kwta_matched":  [f"kwta_k={K_MATCHED}"],
    "step2/kwta_002":      [f"kwta_k={K_002}"],
    # Step 3 -- support gate, factorized into input x normalization (4 arms)
    "step3/mag_sigmoid":   ["predictor=ltv", "gate_input=magnitude", "gate_norm=sigmoid"],
    "step3/sup_sigmoid":   ["predictor=ltv", "gate_input=support", "gate_norm=sigmoid"],
    "step3/mag_softmax":   ["predictor=ltv", "gate_input=magnitude", "gate_norm=softmax"],
    "step3/sup_softmax":   ["predictor=ltv", "gate_input=support", "gate_norm=softmax"],
    # Step 4 -- union head (3 arms)
    "step4/J1":            ["predictor=ltv", "n_heads=1"],
    "step4/J4_ent0":       ["predictor=ltv", "n_heads=4", "head_entropy_coef=0.0"],
    "step4/J4_ent":        ["predictor=ltv", "n_heads=4", "head_entropy_coef=0.1"],
}


_CACHE = {}


def _step(overrides, seed=0):
    """One optimizer step for an arm. Cached: building a D=384 ViT per test would
    otherwise dominate the suite runtime for no extra coverage."""
    key = (tuple(overrides), seed)
    if key in _CACHE:
        return _CACHE[key]
    cfg = load_cfg(overrides)
    seed_all(seed)
    model, opts = build(cfg)
    gen = torch.Generator().manual_seed(1234)
    obs, act = synthetic_batch(cfg, 2, gen)
    model.train()
    for o in opts:
        o.zero_grad()
    _, _, _, loss, comps = model(obs, act)
    loss.backward()
    for o in opts:
        o.step()
    _CACHE[key] = (cfg, model, loss, comps)
    return _CACHE[key]


@pytest.mark.parametrize("arm", list(ARMS))
def test_arm_trains_one_step(arm):
    cfg, model, loss, comps = _step(ARMS[arm])
    assert torch.isfinite(loss), f"{arm} produced a non-finite loss"
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, f"{arm} produced no gradients at all"
    assert all(torch.isfinite(g).all() for g in grads), f"{arm} has non-finite grads"
    assert torch.isfinite(comps["reg_loss"]), f"{arm} has a non-finite RDMReg loss"


@pytest.mark.parametrize("arm", list(ARMS))
def test_arm_overrides_actually_reach_the_modules(arm):
    """Guards against a flag that composes but is dropped before instantiation --
    the failure mode that makes a variant arm silently identical to its control."""
    ov = ARMS[arm]
    cfg, model, _, _ = _step(ov)
    if any(o.startswith("kwta_k=") for o in ov):
        k = int(next(o for o in ov if o.startswith("kwta_k=")).split("=")[1])
        assert model.link.kwta_k == k
    else:
        assert model.link.kwta_k is None
    for key, attr in (("gate_input", "gate_input"), ("gate_norm", "gate_norm")):
        want = next((o.split("=")[1] for o in ov if o.startswith(f"{key}=")), None)
        if want is not None:
            assert getattr(model.predictor, attr) == want
    want_j = next((int(o.split("=")[1]) for o in ov if o.startswith("n_heads=")), 1)
    assert model.n_heads == want_j
    assert model.predictor.n_heads == want_j


def test_upstream_arm_is_untouched_by_every_default():
    """The controls must be plain upstream: no k-WTA, upstream gate, one head."""
    cfg, model, _, comps = _step([])
    assert cfg.kwta_k is None
    assert (cfg.gate_input, cfg.gate_norm) == ("magnitude", "sigmoid")
    assert cfg.n_heads == 1 and cfg.head_entropy_coef == 0.0
    assert model.n_heads == 1
    assert "head_switch_rate" not in comps


def _matched_mu(k):
    """train.py's rule: for the rectified Laplace target, P(GN_1 + mu > 0) =
    0.5*exp(mu/sigma), so mu = sigma*ln(2k/D) makes the target density k/D."""
    from models.infojepa_modules import gng_unit_sigma

    return gng_unit_sigma(1.0) * math.log(2.0 * k / D)


@pytest.mark.parametrize("k", [K_002, K_MATCHED])
def test_kwta_arm_is_at_most_k_sparse_and_mu_matched(k):
    """L0 is capped at k, not pinned to it: reprelu can leave fewer than k units
    positive, and top-k then selects zeros. Equality holds only when the link
    supplies at least k survivors, which is why the tight arm is the strict test."""
    mu = _matched_mu(k)
    cfg, model, _, comps = _step([f"kwta_k={k}", f"mu={mu}"])
    assert cfg.mu == pytest.approx(mu)
    assert comps["l0_frac"].item() <= k / D + 1e-6


def test_tight_kwta_arm_hits_the_k_over_d_density_exactly():
    """At k/D = 0.02 there are always more than k positives, so k-WTA binds and
    the arm really does train at the density the gate claims."""
    k = K_002
    _, _, _, comps = _step([f"kwta_k={k}", f"mu={_matched_mu(k)}"])
    assert comps["l0_frac"].item() == pytest.approx(k / D, abs=1e-6)


def test_matched_rho_arm_leaves_kwta_nearly_inactive():
    """That is what 'matched' means: at k = rho*D the link already produces about
    k survivors, so this arm isolates the density change from the top-k operator."""
    k = K_MATCHED
    _, _, _, comps = _step([f"kwta_k={k}", f"mu={_matched_mu(k)}"])
    assert comps["l0_frac"].item() > 0.8 * k / D


def test_softmax_gate_keeps_mean_gate_magnitude_near_one():
    """r*softmax, not bare softmax: a bare softmax over r=16 has mean 1/16 versus
    sigmoid's ~0.5, an 8x shrink of the gradient path that would read as
    'support gating is worse' after only 2 epochs."""
    _, model, _, _ = _step(ARMS["step3/sup_softmax"])
    z = torch.rand(2, 3, model.encoder.num_patches, D).abs()
    g = model.predictor.gates(z)
    assert g.mean().item() == pytest.approx(1.0, abs=0.15)


def test_union_head_arm_logs_its_preconditions():
    _, _, _, comps = _step(ARMS["step4/J4_ent"])
    assert {"head_switch_rate", "head_burst_rate", "head_usage_max"} <= set(comps)
    assert comps["head_usage_max"].item() <= 1.0


def test_paired_arms_share_a_seed_and_differ_only_in_the_flag():
    """Matched controls: same predictor, D, HP and seed, so the contrast is the flag."""
    base = load_cfg(ARMS["step4/J1"])
    var = load_cfg(ARMS["step4/J4_ent0"])
    # "predictor" also differs, but only because predictor.n_heads interpolates
    # ${oc.select:n_heads,1}; that propagation is the intended wiring.
    differing = {k for k in base if base[k] != var[k]}
    assert differing == {"n_heads", "predictor"}, differing
    assert base.predictor.n_heads == 1 and var.predictor.n_heads == 4
    assert {k for k in base.predictor if base.predictor[k] != var.predictor[k]} == {"n_heads"}
    assert base.training.seed == var.training.seed
    assert base.reg_weight == var.reg_weight
    assert base.training.mup_lr == var.training.mup_lr
