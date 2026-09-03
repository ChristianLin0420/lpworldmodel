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
    # Round 5 / T1 -- attach the decoder. aux_decoder, NOT has_decoder: see
    # conf/train_rdmreg.yaml (plan.py rebuilds a decoder for has_decoder runs and
    # raises on every checkpoint this repo writes).
    "t1/decode":           ["predictor=ltv", "aux_decoder=true", "model.train_decoder=true",
                            "decoder=transposed_conv", "decode_grad=true", "lamb_decode=0.1"],
    "t1/decode_detach":    ["predictor=ltv", "aux_decoder=true", "model.train_decoder=true",
                            "decoder=transposed_conv", "decode_grad=false", "lamb_decode=0.1"],
    # Round 5 / T2 -- patch tokens x a per-patch decoder (the missing 2x2 cell).
    "t2/patchdecode":      ["predictor=ltv", "encoder=vit_scratch_patch", "aux_decoder=true",
                            "model.train_decoder=true", "decoder=patch_head",
                            "decode_grad=true", "lamb_decode=0.1"],
    "t2/patchdecode_detach": ["predictor=ltv", "encoder=vit_scratch_patch", "aux_decoder=true",
                            "model.train_decoder=true", "decoder=patch_head",
                            "decode_grad=false", "lamb_decode=0.1"],
    # Round 5 / T3 -- weight each transition by the visual change proprio does NOT
    # explain. No states.pth anywhere: the agent's disc is masked out using obs["proprio"],
    # which is a model INPUT.
    "t3/contact":          ["predictor=ltv", "contact_gamma=1.0"],
    "t3/contact_shuf":     ["predictor=ltv", "contact_gamma=1.0", "contact_shuffle=true"],
    "t3/contact_g05":      ["predictor=ltv", "contact_gamma=0.5"],
    # Round 5 / T6 -- num_pred IS K. jump5 predicts z_t -> z_{t+5} in one call;
    # overshoot5 chains the 1-step map 5 times over the SAME window.
    "t6/jump5":            ["predictor=ltv", "num_pred=5"],
    "t6/overshoot5":       ["predictor=ltv", "num_pred=5", "overshoot=true"],
    # Round 5 / T4 -- hindsight goal-conditioned value by in-sample expectile TD, plus
    # V4's BC policy head, which rides the same arm so one checkpoint yields the CEM, the
    # V5 and the V4 number on bit-identical encoder/predictor weights.
    "round5/value_td":     ["predictor=ltv", "value_w=1.0"],
    "round5/value_mc":     ["predictor=ltv", "value_w=1.0", "value_mode=mc"],
    "round5/value_geom":   ["predictor=ltv", "value_w=1.0", "value_mode=geom"],
    "round5/policy":       ["predictor=ltv", "policy_w=1.0"],
    "round5/vp":           ["predictor=ltv", "value_w=1.0", "policy_w=1.0"],
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
    # T4/V4: the head must be built iff its weight is on, and must be ABSENT otherwise --
    # a head built at defaults would consume RNG and break bit-identity, and a head not
    # built when asked for is a silently-inert arm (the path_int defect).
    want_v = float(next((o.split("=")[1] for o in ov if o.startswith("value_w=")), 0.0))
    want_p = float(next((o.split("=")[1] for o in ov if o.startswith("policy_w=")), 0.0))
    assert (model.value_head is not None) == (want_v > 0)
    assert (model.value_target is not None) == (want_v > 0)
    assert (model.policy_head is not None) == (want_p > 0)
    want_mode = next((o.split("=")[1] for o in ov if o.startswith("value_mode=")), "td")
    assert model.value_mode == want_mode


def test_upstream_arm_is_untouched_by_every_default():
    """The controls must be plain upstream: no k-WTA, upstream gate, one head."""
    cfg, model, _, comps = _step([])
    assert cfg.kwta_k is None
    assert (cfg.gate_input, cfg.gate_norm) == ("magnitude", "sigmoid")
    assert cfg.n_heads == 1 and cfg.head_entropy_coef == 0.0
    assert model.n_heads == 1
    assert "head_switch_rate" not in comps
    # T4/V4 inertness, the leg tests/test_bit_identity.py cannot state on its own:
    # set(loss_components) is pinned there, so a new key at defaults fails that test --
    # this one says WHICH keys and that the modules do not exist at all.
    assert cfg.value_w == 0.0 and cfg.policy_w == 0.0
    assert model.value_head is None and model.value_target is None
    assert model.policy_head is None
    assert "value_loss" not in comps and "policy_loss" not in comps


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


# --- Round 5, T1: the decoder is attached and its gradient reaches the encoder ------

def test_t1_decoder_is_built_optimised_and_reaches_the_encoder():
    """NEW HEAD CHECKLIST, gradient leg. With decode_grad the pixel loss must move
    every encoder parameter; without it, none -- that difference IS the arm."""
    out = {}
    for name, dg in (("decode", "true"), ("decode_detach", "false")):
        ov = ARMS[f"t1/{name}"]
        _, model, _, comps = _step(ov)          # already stepped once
        assert model.decoder is not None, "decoder was not built"
        assert model.decode_grad is (dg == "true")
        assert "decoder_recon_loss" in comps and torch.is_tensor(comps["decoder_recon_loss"])
        # the decoder's own parameters must be stepped (an optimizer group exists)
        dgrads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
        assert dgrads and max(float(g.abs().max()) for g in dgrads) > 0
        # fresh graph: _step's backward already freed the cached one
        cfg2 = load_cfg(ov)
        seed_all(0)
        m2, _o = build(cfg2)
        g = torch.Generator().manual_seed(1234)
        obs, act = synthetic_batch(cfg2, 2, g)
        _, _, _, _, c2 = m2(obs, act)
        enc = [p for p in m2.encoder.parameters() if p.requires_grad]
        reach = torch.autograd.grad(
            c2["decoder_recon_loss"], enc, retain_graph=False, allow_unused=True
        )
        out[name] = (sum(x is not None for x in reach), len(enc))
    assert out["decode"] == (out["decode"][1], out["decode"][1]), out
    assert out["decode_detach"][0] == 0, out


def test_t1_arms_differ_from_their_control():
    """ARMS-MUST-DIFFER. Same seed, same data, same construction: only the detach."""
    a = _step(ARMS["t1/decode"])[3]["loss"].item()
    b = _step(ARMS["t1/decode_detach"])[3]["loss"].item()
    # step 0 losses coincide (the detach only changes the GRADIENT, not the value)
    assert a == pytest.approx(b, rel=1e-6)
    from lpwm_build import loss_trace
    ta = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["t1/decode"])
    tb = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["t1/decode_detach"])
    assert ta[0]["loss"] == pytest.approx(tb[0]["loss"], rel=1e-6)
    assert ta[-1]["z_loss"] != tb[-1]["z_loss"], (ta[-1], tb[-1])


def test_defaults_build_no_decoder():
    """Inertness: at defaults nothing is built, so no new loss component appears."""
    cfg, model, _, comps = _step([])
    assert cfg.aux_decoder is False and cfg.has_decoder is False
    assert cfg.decode_grad is False
    assert model.decoder is None
    assert "decoder_recon_loss" not in comps


# --- Round 5, T2: the per-patch head is per-PATCH -----------------------------------

def test_t2_head_is_positional_and_reaches_the_encoder():
    """What makes the head 'per-patch': token i owns exactly its own 14x14 block.
    If it were not positional, T2 would be PiWM-columns with extra parameters."""
    cfg = load_cfg(ARMS["t2/patchdecode"])
    seed_all(0)
    model, opts = build(cfg)
    assert model.encoder.num_patches == 256
    assert sum(p.numel() for p in model.decoder.parameters()) == 226_380
    g = torch.Generator().manual_seed(1234)
    obs, act = synthetic_batch(cfg, 2, g)
    _, _, vr, loss, comps = model(obs, act)
    assert tuple(vr.shape) == (2, cfg.num_hist + cfg.num_pred, 3, 224, 224)
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    reach = torch.autograd.grad(
        comps["decoder_recon_loss"], enc, retain_graph=True, allow_unused=True
    )
    assert sum(x is not None for x in reach) == len(enc)
    loss.backward()
    assert float(model.decoder.head.weight.grad.abs().sum()) > 0

    model.eval()
    z = torch.randn(1, 1, 256, cfg.embed_dim)
    with torch.no_grad():
        y0, _ = model.decoder(z)
        for tok, (r0, c0) in ((0, (0, 0)), (17, (14, 14)), (255, (210, 210))):
            zp = z.clone()
            zp[:, :, tok] += 5.0
            y1, _ = model.decoder(zp)
            d = (y1 - y0).abs()[0].sum(0)
            inside = float(d[r0:r0 + 14, c0:c0 + 14].sum())
            outside = float(d.sum()) - inside
            assert inside > 1.0, (tok, inside)
            assert outside < 1e-3 * inside, (tok, inside, outside)


# --- Round 5, T3: the contact weight is live, inert at default, and ESS-matched ------

def test_t3_is_inert_at_default_and_live_at_gamma_1():
    """No new loss_components key at defaults (tests/test_bit_identity.py asserts
    set(got) == set(want)), both new keys present and TENSORS when the flag is on."""
    _, _, _, base = _step([])
    assert "contact_ess" not in base and "contact_w_max" not in base
    _, model, _, comps = _step(ARMS["t3/contact"])
    assert model.contact_gamma == 1.0 and model.contact_geom is not None
    for k in ("contact_ess", "contact_w_max"):
        assert k in comps and torch.is_tensor(comps[k]), k
    ess = float(comps["contact_ess"])
    assert 0.0 < ess <= 1.0 + 1e-6, ess


def test_t3_requires_geometry_rather_than_silently_falling_back():
    """A weighting arm that quietly reverts to uniform is the path_int failure mode:
    it looks like a run and IS its own control."""
    from models.visual_world_model import VWorldModel

    with pytest.raises(ValueError):
        VWorldModel(
            image_size=224, num_hist=3, num_pred=1, encoder=torch.nn.Identity(),
            proprio_encoder=torch.nn.Identity(), action_encoder=torch.nn.Identity(),
            decoder=None, predictor=None, contact_gamma=1.0, contact_geom=None,
        )


def test_t3_weight_reaches_the_predictor_gradient():
    """T3 adds no parameters, so the checklist item is that the reweighted term still
    moves the parameters it is supposed to move -- and by a DIFFERENT amount."""
    cfg = load_cfg(ARMS["t3/contact"])
    seed_all(0)
    model, _ = build(cfg)
    g = torch.Generator().manual_seed(1234)
    obs, act = synthetic_batch(cfg, 4, g)
    _, _, _, loss, comps = model(obs, act)
    loss.backward()
    for name, mod in (("predictor", model.predictor), ("encoder", model.encoder)):
        gs = [p.grad for p in mod.parameters() if p.requires_grad]
        assert all(x is not None for x in gs), f"{name} has a None grad"
        assert max(float(x.norm()) for x in gs) > 0, f"{name} grads are all zero"


def test_t3_shuffle_control_holds_the_weight_distribution_exactly():
    """That is what makes it ESS-matched: same weights, same effective compute,
    alignment with contact destroyed."""
    cfg = load_cfg(ARMS["t3/contact"])
    seed_all(0)
    model, _ = build(cfg)
    g = torch.Generator().manual_seed(1234)
    obs, _act = synthetic_batch(cfg, 8, g)
    w = model._contact_weight(obs)
    model.contact_shuffle = True
    model._contact_gen = None
    ws = model._contact_weight(obs)
    assert w.shape == ws.shape == (8, cfg.num_hist)
    assert torch.allclose(w.reshape(-1).sort().values, ws.reshape(-1).sort().values)


def test_t3_arms_differ_from_their_controls():
    from lpwm_build import loss_trace

    base = loss_trace(n_steps=3, batch_size=4, overrides=["predictor=ltv"])
    for arm in ("t3/contact", "t3/contact_shuf", "t3/contact_g05"):
        t = loss_trace(n_steps=3, batch_size=4, overrides=ARMS[arm])
        assert t[-1]["z_loss"] != base[-1]["z_loss"], (arm, t[-1], base[-1])
    a = loss_trace(n_steps=3, batch_size=4, overrides=ARMS["t3/contact"])
    b = loss_trace(n_steps=3, batch_size=4, overrides=ARMS["t3/contact_shuf"])
    assert a[-1]["z_loss"] != b[-1]["z_loss"], (a[-1], b[-1])


# --- Round 5, T6: num_pred IS K ------------------------------------------------------

def test_t6_option_act_is_the_identity_at_k1():
    """Bit-identity leg: at K=1 _option_act must return the INPUT OBJECT, so the
    default path gains no op and no graph node."""
    cfg = load_cfg(["predictor=ltv"])
    seed_all(0)
    model, _ = build(cfg)
    x = torch.randn(2, 4, cfg.action_emb_dim)
    assert model._option_act(x) is x


def test_t6_window_grows_to_k_plus_num_hist_and_pairs_are_k_apart():
    cfg = load_cfg(ARMS["t6/jump5"])
    assert cfg.num_pred == 5
    seed_all(0)
    model, _ = build(cfg)
    obs, act = synthetic_batch(cfg, 2, torch.Generator().manual_seed(1))
    assert obs["visual"].shape[1] == cfg.num_hist + 5 == 8
    z_pred, _, _, loss, _ = model(obs, act)
    assert tuple(z_pred.shape[:2]) == (2, cfg.num_hist)
    ae = model.encode_act(act)
    oa = model._option_act(ae)
    assert oa.shape == ae.shape
    assert torch.allclose(oa[:, 0], ae[:, :5].mean(1))
    assert torch.allclose(oa[:, 2], ae[:, 2:7].mean(1))


def test_t6_one_predictor_call_covers_the_whole_planner_horizon():
    """goal_H = 5 with K = 5 and plan.py's single-frame obs_0 is ONE call: zero
    compounding. The returned length stays act_rows + 1 so that
    planning/evaluator._get_traj_last(i_z_obses, action_len + 1) -- action_len is
    FINITE (mpc.py:113) once an episode succeeds -- still indexes in range."""
    cfg = load_cfg(ARMS["t6/jump5"])
    seed_all(0)
    model, _ = build(cfg)
    model.eval()
    obs, _act = synthetic_batch(cfg, 2, torch.Generator().manual_seed(1))
    n = [0]
    h = model.predictor.register_forward_hook(lambda *a: n.__setitem__(0, n[0] + 1))
    obs0 = {k: v[:, :1] for k, v in obs.items()}
    z_obses, _ = model.rollout(obs_0=obs0, act=torch.rand(2, 5, 10))
    h.remove()
    assert n[0] == 1, n[0]
    assert z_obses["visual"].shape[1] == 6, z_obses["visual"].shape


def test_t6_overshoot_control_compounds_and_differs():
    """Same window, same horizon, K chained calls instead of one -- and a different
    loss, which is the ARMS-MUST-DIFFER check for this pair."""
    cfg = load_cfg(ARMS["t6/overshoot5"])
    seed_all(0)
    model, _ = build(cfg)
    obs, act = synthetic_batch(cfg, 2, torch.Generator().manual_seed(1))
    n = [0]
    h = model.predictor.register_forward_hook(lambda *a: n.__setitem__(0, n[0] + 1))
    z_pred, _, _, loss, _ = model(obs, act)
    h.remove()
    assert n[0] == 5, n[0]
    assert tuple(z_pred.shape[:2]) == (2, 5)
    loss.backward()
    gs = [p.grad for p in model.predictor.parameters() if p.requires_grad]
    assert all(x is not None for x in gs) and max(float(x.norm()) for x in gs) > 0

    from lpwm_build import loss_trace

    a = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["t6/jump5"])
    b = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["t6/overshoot5"])
    assert a[0]["z_loss"] != b[0]["z_loss"], (a[0], b[0])
    assert set(a[0]) == set(b[0])


def test_t6_overshoot_at_k1_is_rejected():
    """At K=1 overshoot IS the default one-step loss, so such an arm would silently
    be its own control."""
    with pytest.raises(Exception):
        cfg = load_cfg(["predictor=ltv", "overshoot=true"])
        seed_all(0)
        build(cfg)


# --- Round 5, T4 / V4: the NEW HEAD CHECKLIST, item by item --------------------------
#
# path_int had none of these: no optimizer group, no _keys_to_save entry, no checkpoint
# key, no liveness test -- and its "null" result had to be retracted (diary 2026-09-03
# s13.3). Each test below is one line of that checklist made executable.

def _fresh(overrides, batch=4, seed=0):
    """A model + optimizers on a fresh graph (the shared _step already backwarded)."""
    cfg = load_cfg(overrides)
    seed_all(seed)
    model, opts = build(cfg)
    obs, act = synthetic_batch(cfg, batch, torch.Generator().manual_seed(1))
    return cfg, model, opts, obs, act


def test_t4_heads_are_built_optimised_and_receive_gradient():
    """Checklist legs (1) optimizer group and (5) gradient reaches the parameters."""
    cfg, model, opts, obs, act = _fresh(ARMS["round5/vp"])
    _, _, _, loss, comps = model(obs, act)
    assert "value_loss" in comps and torch.is_tensor(comps["value_loss"])
    assert "policy_loss" in comps and torch.is_tensor(comps["policy_loss"])
    loss.backward()
    ids = {id(p) for o in opts for g in o.param_groups for p in g["params"]}
    for name, head in (("value_head", model.value_head), ("policy_head", model.policy_head)):
        ps = list(head.parameters())
        assert all(p.grad is not None for p in ps), f"{name} has a None grad"
        assert min(float(p.grad.norm()) for p in ps) > 0, f"{name} has a zero grad"
        assert all(id(p) in ids for p in ps), f"{name} is in NO optimizer -- path_int"
    # the target net is a Polyak copy, never optimised, never given a gradient
    assert all(not p.requires_grad for p in model.value_target.parameters())
    assert not any(id(p) in ids for p in model.value_target.parameters())
    before = model.value_head.net[0].weight.detach().clone()
    for o in opts:
        o.step()
    assert float((model.value_head.net[0].weight - before).abs().max()) > 0


def test_t4_head_init_is_deterministic_and_does_not_touch_the_global_stream():
    """The fork_rng leg. An unforked nn.Linear init would advance the global CPU stream
    that RDMReg draws its target from, so the arm would differ from its control by the
    head AND by every later random draw -- more than one factor."""
    from models.heads import MLPHead

    seed_all(0)
    a = torch.randn(3)
    cfg = load_cfg(ARMS["round5/vp"])
    seed_all(0)
    m, _ = build(cfg)
    seed_all(0)
    b = torch.randn(3)
    assert torch.equal(a, b)  # (sanity: seed_all resets it either way)
    # the init is reproducible from the documented seed alone, which is what makes
    # "was this head ever optimised?" decidable against a checkpoint
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260903)
        ref = MLPHead(3 * D, D, 1)
    assert torch.equal(ref.net[0].weight, m.value_head.net[0].weight)
    assert torch.equal(m.value_head.net[0].weight, m.value_target.net[0].weight)


def test_t4_encoder_and_predictor_are_bit_identical_to_the_control():
    """The heads read DETACHED latents and sample from a PRIVATE generator, so the
    encoder/predictor trajectory must be the control's exactly. That is what makes this
    run's own CEM eval the matched control for V4 and V5."""
    from lpwm_build import loss_trace

    base = loss_trace(n_steps=3, batch_size=4, overrides=["predictor=ltv"])
    vp = loss_trace(n_steps=3, batch_size=4, overrides=ARMS["round5/vp"])
    for i, (b, v) in enumerate(zip(base, vp)):
        assert v["z_loss"] == b["z_loss"], (i, v["z_loss"], b["z_loss"])
        assert v["reg_loss"] == b["reg_loss"], (i, v["reg_loss"], b["reg_loss"])
        assert v["l0_frac"] == b["l0_frac"], i
    # ...and the TOTAL loss must still differ, or the arm is its own control
    assert vp[-1]["loss"] != base[-1]["loss"]
    assert set(vp[-1]) - set(base[-1]) == {"value_loss", "policy_loss"}


def test_t4_arms_differ_from_their_controls():
    """ARMS MUST DIFFER: td vs mc vs geom are single-token VALUE_MODE flips and must
    produce three different value losses on identical data and identical seeds."""
    from lpwm_build import loss_trace

    v = {}
    for name in ("round5/value_td", "round5/value_mc", "round5/value_geom"):
        v[name] = loss_trace(n_steps=3, batch_size=4, overrides=ARMS[name])[-1]
    keys = list(v)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert v[keys[i]]["value_loss"] != v[keys[j]]["value_loss"], (keys[i], keys[j])
            assert v[keys[i]]["loss"] != v[keys[j]]["loss"], (keys[i], keys[j])


def test_t4_value_head_can_be_saved_and_restored_by_name():
    """Checklist leg (2). save_ckpt/load_ckpt walk state_dicts by name, and
    VWorldModel.load_head_state is what puts them back at PLAN time -- plan.py's
    load_model restores a fixed list of submodules and knows nothing about these heads."""
    cfg, model, opts, obs, act = _fresh(ARMS["round5/vp"])
    _, _, _, loss, _ = model(obs, act)
    loss.backward()
    for o in opts:
        o.step()
    payload = {
        "value_head": model.value_head.state_dict(),
        "value_target": model.value_target.state_dict(),
        "policy_head": model.policy_head.state_dict(),
    }
    seed_all(0)
    fresh, _ = build(load_cfg(ARMS["round5/vp"]))
    assert fresh.heads_restored_from is None
    assert not torch.equal(fresh.value_head.net[0].weight, model.value_head.net[0].weight)
    fresh.load_head_state(payload)
    assert fresh.heads_restored_from is not None
    assert torch.equal(fresh.value_head.net[0].weight, model.value_head.net[0].weight)
    assert torch.equal(fresh.policy_head.net[0].weight, model.policy_head.net[0].weight)


def test_t4_restore_refuses_a_checkpoint_without_the_head():
    """A checkpoint that lacks the key is exactly the path_int situation, and it must be
    an error rather than a silent fresh init."""
    _, model, _, _, _ = _fresh(ARMS["round5/vp"])
    with pytest.raises(KeyError):
        model.load_head_state({"encoder": {}})


def test_t4_policy_head_needs_the_raw_action_width():
    """A BC head sized by the 384-d action EMBEDDING would emit something that is not an
    action. Loud at construction, not at plan time."""
    from models.visual_world_model import VWorldModel

    enc = torch.nn.Identity()
    enc.emb_dim = 8               # the head's input width is 3 * base_encoder.emb_dim
    with pytest.raises(ValueError):
        VWorldModel(
            image_size=224, num_hist=3, num_pred=1, encoder=enc,
            proprio_encoder=torch.nn.Identity(), action_encoder=torch.nn.Identity(),
            decoder=None, predictor=None, policy_w=1.0, act_dim_raw=None,
        )


def test_t4_policy_head_emits_the_raw_action_width():
    cfg, model, _, _, _ = _fresh(ARMS["round5/vp"])
    assert model.policy_head.d_out == 2 * cfg.frameskip == 10
    assert model.value_head.d_out == 1
    assert model.value_head.d_in == 3 * cfg.embed_dim


def test_t4_value_loss_is_not_the_latent_distance_it_is_meant_to_replace():
    """`geom` regresses V onto -||z-g||; `td` must not land on the same function, or T4
    is a smooth reparametrisation of the leaf score CEM already uses."""
    cfg_td, m_td, _, obs, act = _fresh(ARMS["round5/value_td"])
    _, m_geom, _, _, _ = _fresh(ARMS["round5/value_geom"])
    assert m_td.value_mode == "td" and m_geom.value_mode == "geom"
    z = torch.rand(16, 4, 1, cfg_td.embed_dim)
    a = m_td._value_loss(z)
    b = m_geom._value_loss(z)
    assert torch.is_tensor(a) and torch.is_tensor(b) and float(a) != float(b)


def test_t4_target_network_tracks_but_lags_the_online_net():
    """Polyak at ema=0.005: after one update the target must have MOVED and must still
    differ from the online net. A target pinned at init is a constant regression."""
    _, model, _, obs, act = _fresh(ARMS["round5/value_td"])
    model.train()
    w0 = model.value_target.net[0].weight.detach().clone()
    # perturb the online net so the Polyak step has something to track
    with torch.no_grad():
        model.value_head.net[0].weight.add_(1.0)
    model._value_loss(torch.rand(8, 4, 1, model.value_head.d_in // 3))
    w1 = model.value_target.net[0].weight.detach()
    moved = float((w1 - w0).abs().max())
    assert moved > 0, "target network never updates"
    assert not torch.allclose(w1, model.value_head.net[0].weight)
    assert moved == pytest.approx(model.value_ema, rel=1e-4)


def test_t4_no_privileged_state_can_reach_either_head():
    """Images and proprioception only. Both heads take LATENTS, and the only tensors in
    scope are z (from obs['visual']) and act -- states.pth has no path into either."""
    import ast
    import inspect
    import textwrap

    from models.visual_world_model import VWorldModel

    for fn in (VWorldModel._value_loss, VWorldModel._policy_loss,
               VWorldModel._hindsight_pairs):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)):
            node.body = node.body[1:]          # drop the docstring, keep the CODE
        code = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        for banned in ("state", "pose", "block", "goal_state"):
            assert banned not in code.lower(), (fn.__name__, banned)


# --- Round 5, V4: the PolicyPlanner contract ----------------------------------------

def _policy_planner(**over):
    """PolicyPlanner on the shared planner stub (tests/stub_wm.py), no env and no GPU."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stub_wm import StubPreprocessor, StubWandb, StubWorldModel

    from models.heads import MLPHead
    from planning.policy import PolicyPlanner

    wm = StubWorldModel(action_dim=10)
    wm.policy_head = MLPHead(3 * wm.emb_dim, 16, 10)
    wm.heads_restored_from = "test://stub"      # stands in for a real restore
    kw = dict(horizon=5, wm=wm, action_dim=10,
              objective_fn=None, preprocessor=StubPreprocessor(), evaluator=None,
              wandb_run=StubWandb(), log_filename=None,
              name="mpc_policy", env=None, topk=30, num_samples=300)
    kw.update(over)
    return wm, PolicyPlanner(**kw)


def test_v4_policy_planner_satisfies_the_planner_contract():
    """plan.py:187-204 + planning/base_planner.py: **kwargs must swallow name=/env=/the
    CEM-only keys, plan() must return (Tensor (B,H,A), ndarray (B,)), and .horizon must
    be settable because MPCPlanner assigns it from goal_H (plan.py:198)."""
    import numpy as np
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stub_wm import make_obs

    from planning.base_planner import BasePlanner
    from planning.policy import PolicyPlanner

    assert issubclass(PolicyPlanner, BasePlanner)
    wm, p = _policy_planner()
    p.horizon = 5
    p.logging_prefix = "plan_0"
    obs_0, obs_g = make_obs(b=2), make_obs(b=2)
    mu, lens = p.plan(obs_0, obs_g)
    assert torch.is_tensor(mu) and tuple(mu.shape) == (2, 5, 10), mu.shape
    assert isinstance(lens, np.ndarray) and lens.shape == (2,) and np.all(np.isinf(lens))
    # H(H+1)/2 rollout steps, i.e. H-1 rollout CALLS -- not CEM's 300 x 30
    assert wm.rollout_batches == [2, 2, 2, 2], wm.rollout_batches


def test_v4_policy_planner_is_goal_conditioned():
    """d(action)/d(goal) must be nonzero: a head that ignores z_g has learned the
    marginal action prior, and V4 would then be measuring the prior, not planning."""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stub_wm import make_obs

    torch.manual_seed(0)
    _, p = _policy_planner()
    obs_0 = make_obs(b=8)
    a1, _ = p.plan(obs_0, make_obs(b=8))
    a2, _ = p.plan(obs_0, make_obs(b=8))
    assert float((a1 - a2).pow(2).mean().sqrt()) > 0
    # ...and the shuffled-goal control must be a DIFFERENT plan on the same goals
    _, ps = _policy_planner(shuffle_goal=True)
    ps.policy.load_state_dict(p.policy.state_dict())
    torch.manual_seed(3)
    obs_g = make_obs(b=8)
    assert not torch.allclose(p.plan(obs_0, obs_g)[0], ps.plan(obs_0, obs_g)[0])


def test_v4_policy_planner_refuses_a_fresh_or_missing_head():
    """The anti-path_int guard: planning on a head plan.py never restored would be
    measuring torch.manual_seed(20260903), and it must raise instead."""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stub_wm import StubPreprocessor, StubWandb, StubWorldModel

    from planning.policy import PolicyPlanner

    base = dict(horizon=5, action_dim=10, objective_fn=None,
                preprocessor=StubPreprocessor(), evaluator=None, wandb_run=StubWandb(),
                log_filename=None)
    with pytest.raises(ValueError, match="TRAINED policy_head"):
        PolicyPlanner(wm=StubWorldModel(action_dim=10), **base)
    wm, _ = _policy_planner()
    wm.heads_restored_from = None
    with pytest.raises(ValueError, match="FRESH INIT"):
        PolicyPlanner(wm=wm, **base)
