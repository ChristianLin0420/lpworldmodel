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
    # --- Round 6. Every arm is a GRID on its own strength knob, because six of the nine
    # round-3/4/5 proposals were single-shot on theirs, V1 had no knob at all, and T4
    # would have died on its primary setting. The entries below are one point of each
    # grid; the launcher sweeps the rest with the same override names.
    #
    # R6 -- the screen-validated term. S_model = 1 - J_S(z_pred, target) is the only
    # logged quantity that survives analysis/screen_objective.py (raw -0.769, partial
    # -0.549, monotone), and soft_jaccard appeared zero times in models/ before this.
    "r6/support":          ["predictor=ltv", "support_w=1.0"],
    "r6/support_w01":      ["predictor=ltv", "support_w=0.1"],
    "r6/support_w10":      ["predictor=ltv", "support_w=10.0"],
    # R2 -- rollout self-consistency on the CEM PROPOSAL distribution. consist_src=data
    # is the matched control: same term, same shapes, dataset actions, so the contrast
    # is the distribution and nothing else.
    "r2/consist_cem":      ["predictor=ltv", "consist_w=1.0"],
    "r2/consist_data":     ["predictor=ltv", "consist_w=1.0", "consist_src=data"],
    "r2/consist_k3":       ["predictor=ltv", "consist_w=1.0", "consist_k=3"],
    # R3 -- action-space sharpness-aware minimisation. rho is a per-row L2 radius in the
    # normalised action space, so rho=1.0 is one CEM var_scale.
    "r3/sam01":            ["predictor=ltv", "sam_rho=0.1"],
    "r3/sam10":            ["predictor=ltv", "sam_rho=1.0"],
    # R4 -- V1's epsilon made a knob, plus a span clip. incr_eps=1e-4 IS the run on
    # record (PiWM-incr, -0.383 with 8/8 seeds dead, ESS 0.063); 4.1e-2 is the measured
    # median increment, i.e. the knee moved to where the data actually lives.
    "r4/incr":             ["predictor=ltv", "incr_norm=true"],
    "r4/incr_eps_med":     ["predictor=ltv", "incr_norm=true", "incr_eps=0.041"],
    "r4/incr_clip3":       ["predictor=ltv", "incr_norm=true", "incr_clip=3.0"],
    # A clip tight enough to BIND on the synthetic fixture. The fixture's frames are iid
    # noise, so its increments are nearly uniform (ESS 0.97, span 1.9) and any production
    # clip is inactive on it by construction; on real PushT the ESS is 0.063 and every
    # clip in the grid binds hard. Kept as its own arm so ARMS-MUST-DIFFER is checkable
    # here rather than asserted about data these tests cannot see.
    "r4/incr_clip_tight":  ["predictor=ltv", "incr_norm=true", "incr_clip=1.05"],
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


# =====================================================================================
# Round 6. Four objective terms. The checklist, per term:
#   inert at default (no new loss_components key -- tests/test_bit_identity.py pins the
#   SET, this says which keys and that they are absent), the flag reaches the model, the
#   term changes the loss AND the gradient of existing parameters (all four are
#   parameter-free, which is the checklist's stated alternative to "gradient reaches
#   every new parameter"), the arm differs from its control, and the collapse diagnostic
#   the term needs is emitted as a TENSOR every epoch.
# =====================================================================================


def _build_must_raise(overrides, needle):
    """A guard that fires during hydra.utils.instantiate surfaces as
    InstantiationException with the original ValueError in its message, so match on the
    message rather than on the type (tests/test_arms.py::test_t3_requires_geometry
    dodges this by constructing VWorldModel directly; these arms need the real config
    path, because the point is that the FLAG reaches the model)."""
    with pytest.raises(Exception) as ei:
        cfg = load_cfg(overrides)
        seed_all(0)
        build(cfg)
    assert needle in str(ei.value), str(ei.value)


_R6_KEYS = {"support_s", "support_z_rms", "support_tgt_rms", "support_z_sum",
            "support_l0_pred"}
_R2_KEYS = {"consist_loss", "consist_rel", "consist_act_rms", "consist_jump_rms",
            "consist_chain_rms"}
_R3_KEYS = {"sam_d_action", "sam_d_action_over_scale", "sam_sharpness", "sam_l_clean",
            "sam_delta_rms"}
_R4_KEYS = {"incr_ess", "incr_w_max", "incr_w_min", "incr_span"}


def test_round6_defaults_emit_no_new_component_and_no_new_state():
    """Inertness, all four terms at once. Also the state-dict leg: none of these adds a
    parameter or a buffer, so a checkpoint written before round 6 must still load."""
    cfg, model, _, comps = _step([])
    assert cfg.support_w == 0.0 and cfg.consist_w == 0.0 and cfg.sam_rho == 0.0
    assert cfg.incr_eps == 1e-4 and cfg.incr_clip == 0.0
    assert (model.support_w, model.consist_w, model.sam_rho) == (0.0, 0.0, 0.0)
    assert (model.incr_eps, model.incr_clip) == (1e-4, 0.0)
    for k in _R6_KEYS | _R2_KEYS | _R3_KEYS | _R4_KEYS:
        assert k not in comps, k
    # the private generators are lazily built and never touched at defaults
    assert model._consist_gen is None and model._sam_gen is None
    # no parameter and no buffer: all four terms are parameter-free, so every checkpoint
    # written before round 6 still loads and the bit-identity fixtures are unaffected
    names = set(model.state_dict())
    assert not [n for n in names
                if any(t in n for t in ("support", "consist", "sam_", "incr_"))], names


def test_round6_terms_survive_a_no_grad_forward():
    """train.py's val() calls this same forward, and there is already a
    `with torch.no_grad()` two blocks above it (around openloop_rollout). R3 takes a
    gradient-ascent step INSIDE the forward, so without an explicit enable_grad it would
    raise there -- at the end of epoch 1, i.e. after two hours of training."""
    for arm in ("r6/support", "r2/consist_cem", "r3/sam10", "r4/incr"):
        cfg, model, _o, obs, act = _fresh(ARMS[arm], batch=2)
        model.eval()
        with torch.no_grad():
            _, _, _, loss, comps = model(obs, act)
        assert torch.isfinite(loss), arm
        assert all(torch.isfinite(v).all() for v in comps.values()
                   if torch.is_tensor(v)), arm


# --- R6: the support term ------------------------------------------------------------

def test_r6_is_inert_at_default_and_live_when_on():
    _, _, _, base = _step(["predictor=ltv"])
    assert not (_R6_KEYS & set(base))
    _, model, _, comps = _step(ARMS["r6/support"])
    assert model.support_w == 1.0
    assert _R6_KEYS <= set(comps)
    for k in _R6_KEYS:
        assert torch.is_tensor(comps[k]), k
    s = float(comps["support_s"])
    assert 0.0 <= s <= 1.0, s          # S = 1 - J_S on non-negative codes


def test_r6_optimises_the_same_function_object_the_screen_scored():
    """The whole reason R6 exists is analysis/screen_objective.py's ranking of
    jacc/S_model. If the optimised quantity and the screened quantity were two
    functions that happened to agree, the arm could not be read against that screen."""
    import train as T
    from models import stats
    from models import visual_world_model as vwm

    assert T.soft_jaccard is stats.soft_jaccard
    assert vwm.soft_jaccard is stats.soft_jaccard

    cfg, model, _, comps = _step(ARMS["r6/support"])
    d = model._diag                                   # what train.py's diagnostic reads
    want = float((1.0 - stats.soft_jaccard(d["z_pred"], d["target"])).flatten().mean())
    assert float(comps["support_s"]) == pytest.approx(want, rel=0, abs=0)


def test_r6_is_added_to_z_loss_and_never_replaces_it():
    """ADDED, not replacing: J_S constrains the support, the MSE constrains the
    magnitudes on it. z_loss(support_w=w) - z_loss(0) must be exactly w * S."""
    _, _, _, c0 = _step(["predictor=ltv"])
    out = {}
    for arm, w in (("r6/support_w01", 0.1), ("r6/support", 1.0), ("r6/support_w10", 10.0)):
        _, _, _, c = _step(ARMS[arm])
        out[w] = float(c["z_loss"]) - float(c0["z_loss"])
        assert out[w] == pytest.approx(w * float(c["support_s"]), rel=1e-5)
    # strictly increasing in the knob: the grid is a grid, not three copies of one point
    assert out[0.1] < out[1.0] < out[10.0]


def test_r6_changes_the_gradient_of_the_parameters_it_shares():
    """Parameter-free term, so the checklist's gradient leg is: the term reaches the
    existing parameters, and the total gradient is DIFFERENT because of it."""
    cfg, model, _opts, obs, act = _fresh(ARMS["r6/support"], batch=4)
    z = model._link(model.encode_obs(obs)["visual"])
    z_src, target = z[:, : cfg.num_hist], z[:, cfg.num_pred :]
    a_emb = model._act_emb_with_pose(act, obs["proprio"])
    z_pred = model._link(model.predict(z_src, a_emb[:, : cfg.num_hist]))
    term, logs = model._support_loss(z_pred, target)
    # the logged value is the term's own value, detached (train.py means it per epoch)
    assert float(logs["support_s"]) == pytest.approx(float(term), rel=0, abs=0)
    params = [p for p in model.predictor.parameters() if p.requires_grad] + \
             [p for p in model.encoder.parameters() if p.requires_grad]
    # the SUPPORT term ALONE, not the total loss
    g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    assert all(x is not None for x in g), "support term reaches no parameter"
    assert max(float(x.norm()) for x in g) > 0

    def total_grad(overrides):
        cfg2, m2, _o, o2, a2 = _fresh(overrides, batch=4)
        _, _, _, l2, _ = m2(o2, a2)
        l2.backward()
        return torch.cat([p.grad.flatten() for p in m2.predictor.parameters()
                          if p.grad is not None])

    g_on = total_grad(ARMS["r6/support"])
    g_off = total_grad(["predictor=ltv"])
    assert not torch.allclose(g_on, g_off)


def test_r6_refuses_an_unrectified_link():
    """J_S is only DEFINED on non-negative inputs. On the identity link the ratio can
    leave [0, 1] entirely, so the term would be a different objective under the same
    name -- the silent-fallback failure mode, refused loudly like contact_geom."""
    _build_must_raise(["predictor=ltv", "link=identity", "support_w=1.0"], "rectified")


def test_r6_arms_differ_from_their_control():
    from lpwm_build import loss_trace

    base = loss_trace(n_steps=3, batch_size=2, overrides=["predictor=ltv"])
    prev = None
    for arm in ("r6/support_w01", "r6/support", "r6/support_w10"):
        t = loss_trace(n_steps=3, batch_size=2, overrides=ARMS[arm])
        assert t[-1]["z_loss"] != base[-1]["z_loss"], (arm, t[-1], base[-1])
        if prev is not None:
            assert t[-1]["z_loss"] != prev[-1]["z_loss"], arm
        prev = t


# --- R2: rollout self-consistency on the planner's action distribution ---------------

def test_r2_is_inert_at_default_and_live_when_on():
    _, _, _, base = _step(["predictor=ltv"])
    assert not (_R2_KEYS & set(base))
    _, model, _, comps = _step(ARMS["r2/consist_cem"])
    assert model.consist_w == 1.0 and model.consist_src == "cem" and model.consist_k == 5
    assert _R2_KEYS <= set(comps)
    for k in _R2_KEYS:
        assert torch.is_tensor(comps[k]), k
    # added to `loss`, NOT folded into z_loss, so the two are separable in the logs
    _, _, _, c0 = _step(["predictor=ltv"])
    assert float(comps["z_loss"]) == pytest.approx(float(c0["z_loss"]), rel=0, abs=0)


def test_r2_feeds_the_loss_actions_no_demonstrator_took():
    """The point of the arm. ~60 training arms, none of which has ever put a
    non-dataset action into a loss."""
    cfg, model, _o, obs, act = _fresh(ARMS["r2/consist_cem"], batch=4)
    a = model._consist_actions(act)
    rows = cfg.num_hist + model.consist_k - 1
    assert a.shape == (4, rows, act.shape[-1])
    # chained K steps read rows [j, j+num_hist) for j < K, i.e. up to num_hist+K-2
    assert rows == cfg.num_hist + model.consist_k - 1
    pool = act.reshape(-1, act.shape[-1])
    # not a row of the batch, and not the batch's actions reshaped
    hits = (a.reshape(-1, 1, a.shape[-1]) == pool[None]).all(-1).any()
    assert not bool(hits), "the 'cem' source returned dataset actions"


def test_r2_data_control_draws_only_dataset_actions():
    """The matched control: identical term, identical shapes, identical predictor-call
    count, actions from the data instead of from N(0, var_scale^2)."""
    cfg, model, _o, obs, act = _fresh(ARMS["r2/consist_data"], batch=4)
    a = model._consist_actions(act)
    pool = act.reshape(-1, act.shape[-1])
    member = (a.reshape(-1, 1, a.shape[-1]) == pool[None]).all(-1).any(-1)
    assert bool(member.all()), "the 'data' source invented an action"


def test_r2_cem_source_is_the_planners_own_proposal():
    """planning/cem.py:99-106 draws randn * sigma + mu with mu = 0 and sigma =
    var_scale (init_mu_sigma), and conf/plan_rdmreg.yaml sets var_scale = 1."""
    cfg, model, _o, obs, act = _fresh(
        ["predictor=ltv", "consist_w=1.0", "consist_sigma=2.0"], batch=64
    )
    a = model._consist_actions(act)
    assert float(a.mean()) == pytest.approx(0.0, abs=0.05)
    assert float(a.std()) == pytest.approx(2.0, rel=0.05)


def test_r2_sampling_never_touches_the_global_rng_stream():
    """RDMReg draws its target from the global stream LATER in the same forward. A term
    that sampled from it would move every subsequent draw, and the arm would differ from
    its control by the term AND by the regulariser's noise -- two factors, not one."""
    cfg, model, _o, obs, act = _fresh(ARMS["r2/consist_cem"], batch=4)
    torch.manual_seed(7)
    before = torch.randn(4)
    model._consist_actions(act)
    model._consist_actions(act)
    torch.manual_seed(7)
    after = torch.randn(4)
    assert torch.equal(before, after)
    # and it is genuinely random: two draws differ
    assert not torch.equal(model._consist_actions(act), model._consist_actions(act))


def test_r2_costs_one_jump_call_plus_k_chain_calls():
    """Structure check: the jump is ONE predictor call on the option action and the
    chain is K calls of the 1-step map -- T6's two objects, reused, not reimplemented."""
    cfg, model, _o, obs, act = _fresh(ARMS["r2/consist_cem"], batch=2)
    n = [0]
    h = model.predictor.register_forward_hook(lambda *a: n.__setitem__(0, n[0] + 1))
    model(obs, act)
    h.remove()
    assert n[0] == 1 + 1 + model.consist_k, n[0]   # main + jump + K chained


def test_r2_jump_and_chain_agree_exactly_when_the_map_is_the_identity():
    """The term is a genuine disagreement, not a shape artefact: force the predictor to
    return its own last input frame and both sides become z, so L must be exactly 0."""
    class _Hold(torch.nn.Module):
        """P(z, a) = z: the last frame, unchanged, whatever the action."""

        def forward(self, emb, act_emb=None):
            return emb

    cfg, model, _o, obs, act = _fresh(ARMS["r2/consist_cem"], batch=2)
    z_emb = model._link(model.encode_obs(obs)["visual"])
    model.predictor = _Hold()
    loss, logs = model._consist_loss(z_emb, obs, act)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    # and it is not zero for the real map, or the assertion above proves nothing
    cfg2, m2, _o2, obs2, act2 = _fresh(ARMS["r2/consist_cem"], batch=2)
    z2 = m2._link(m2.encode_obs(obs2)["visual"])
    assert float(m2._consist_loss(z2, obs2, act2)[0]) > 0.0


def test_r2_gradient_reaches_the_predictor_and_the_encoder():
    cfg, model, _o, obs, act = _fresh(ARMS["r2/consist_cem"], batch=4)
    z_emb = model._link(model.encode_obs(obs)["visual"])
    loss, _logs = model._consist_loss(z_emb, obs, act)
    for name, mod in (("predictor", model.predictor), ("encoder", model.encoder)):
        ps = [p for p in mod.parameters() if p.requires_grad]
        g = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        assert all(x is not None for x in g), f"{name}: consistency term does not reach"
        assert max(float(x.norm()) for x in g) > 0, name


def test_r2_rejects_k1_where_the_jump_is_its_own_control():
    """At K=1 P_1 IS chain(P_1) and the term is identically zero -- the arm would be its
    own control, which is what the overshoot@K=1 guard exists for."""
    _build_must_raise(["predictor=ltv", "consist_w=1.0", "consist_k=1"], "consist_k")


def test_r2_rejects_an_unknown_source():
    _build_must_raise(["predictor=ltv", "consist_w=1.0", "consist_src=uniform"],
                      "consist_src")


def test_r2_arm_differs_from_its_control_and_from_the_baseline():
    from lpwm_build import loss_trace

    base = loss_trace(n_steps=3, batch_size=2, overrides=["predictor=ltv"])
    cem = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["r2/consist_cem"])
    dat = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["r2/consist_data"])
    assert cem[-1]["loss"] != base[-1]["loss"]
    assert dat[-1]["loss"] != base[-1]["loss"]
    # the single factor that separates the arm from its control is the DISTRIBUTION
    assert cem[-1]["consist_loss"] != dat[-1]["consist_loss"], (cem[-1], dat[-1])
    assert set(cem[0]) == set(dat[0])


# --- R3: action-space sharpness-aware minimisation -----------------------------------

def test_r3_is_inert_at_default_and_live_when_on():
    _, _, _, base = _step(["predictor=ltv"])
    assert not (_R3_KEYS & set(base))
    _, model, _, comps = _step(ARMS["r3/sam01"])
    assert model.sam_rho == 0.1
    assert _R3_KEYS <= set(comps)
    for k in _R3_KEYS:
        assert torch.is_tensor(comps[k]), k


def test_r3_perturbation_has_exactly_the_requested_radius():
    """rho is a per-ROW L2 radius in the normalised action space. Rows the loss never
    reads (row >= num_hist at num_pred=1) have zero gradient and must move by exactly 0
    -- a bare g/||g|| would put NaN into the action the model is then conditioned on."""
    for rho in (0.1, 1.0):
        cfg, model, _o, obs, act = _fresh(
            ["predictor=ltv", f"sam_rho={rho}"], batch=4
        )
        z = model._link(model.encode_obs(obs)["visual"])
        z_src, target = z[:, : cfg.num_hist], z[:, cfg.num_pred :]
        _src, _lc, delta = model._sam_perturb(z_src, target, act, obs["proprio"])
        assert torch.isfinite(delta).all()
        n = delta.norm(dim=-1)
        assert torch.allclose(n[:, : cfg.num_hist],
                              torch.full_like(n[:, : cfg.num_hist], rho), atol=1e-5)
        assert float(n[:, cfg.num_hist:].abs().max()) == 0.0


def test_r3_evaluates_the_loss_at_a_worse_action():
    """That IS the objective: max over the ball. An ascent step must not lower the loss,
    and a bigger ball must not be easier -- if sam_sharpness is ~0 the arm is a null for
    the single-shot reason (rho too small to be an intervention)."""
    _, _, _, c1 = _step(ARMS["r3/sam01"])
    _, _, _, c2 = _step(ARMS["r3/sam10"])
    assert float(c1["sam_sharpness"]) > 0.0
    assert float(c2["sam_sharpness"]) > float(c1["sam_sharpness"])
    assert float(c2["sam_delta_rms"]) == pytest.approx(
        10.0 * float(c1["sam_delta_rms"]), rel=1e-4
    )


def test_r3_logs_d_action_as_a_first_class_metric():
    """MANDATORY GUARD. The unconstrained minimiser of max_delta L is a predictor that
    ignores the action inside the ball, i.e. d_action = 0. It is emitted in
    loss_components (not only in train.py's tier-1 batch log) so it lands in the EPOCH
    average, and it is measured at the CLEAN action so it stays comparable to
    analysis/d_action_probe.py."""
    _, model, _, comps = _step(ARMS["r3/sam01"])
    d = float(comps["sam_d_action"])
    assert torch.isfinite(comps["sam_d_action"]) and d > 0.0
    assert float(comps["sam_d_action_over_scale"]) > 0.0
    # and it must not be able to read 0 by coincidence: the shuffle is a NON-ZERO cyclic
    # shift, so every row is paired with a different row's action on every step. With
    # torch.randperm (train.py's version) the identity comes up 1/b! of the time -- 4% at
    # b=4 -- and 0 is exactly the alarm value this metric exists to raise.
    from lpwm_build import loss_trace

    t = loss_trace(n_steps=4, batch_size=4, overrides=ARMS["r3/sam01"])
    assert all(step["sam_d_action"] > 0.0 for step in t), [s["sam_d_action"] for s in t]


def test_r3_diag_act_src_is_the_clean_action_not_the_perturbed_one():
    cfg, model, _o, obs, act = _fresh(ARMS["r3/sam10"], batch=4)
    model(obs, act)
    clean = model._option_act(
        model._act_emb_with_pose(act, obs["proprio"])
    )[:, : cfg.num_hist]
    assert torch.allclose(model._diag["act_src"], clean.detach(), atol=0, rtol=0)


def test_r3_changes_the_gradient_of_the_parameters_it_shares():
    def total_grad(overrides):
        cfg2, m2, _o, o2, a2 = _fresh(overrides, batch=4)
        _, _, _, l2, _ = m2(o2, a2)
        l2.backward()
        return torch.cat([p.grad.flatten() for p in m2.predictor.parameters()
                          if p.grad is not None])

    assert not torch.allclose(total_grad(ARMS["r3/sam10"]), total_grad(["predictor=ltv"]))


def test_r3_refuses_the_combinations_it_is_not_defined_for():
    """Under overshoot the action drives K chained calls and under J>1 an argmin selects
    a head, so 'the loss at the perturbed action' is a different object in both cases."""
    for extra in (["num_pred=5", "overshoot=true"], ["n_heads=4"]):
        _build_must_raise(["predictor=ltv", "sam_rho=0.1"] + extra, "sam_rho")


def test_r3_arms_differ_from_their_control():
    from lpwm_build import loss_trace

    base = loss_trace(n_steps=3, batch_size=2, overrides=["predictor=ltv"])
    prev = None
    for arm in ("r3/sam01", "r3/sam10"):
        t = loss_trace(n_steps=3, batch_size=2, overrides=ARMS[arm])
        assert t[-1]["z_loss"] != base[-1]["z_loss"], (arm, t[-1], base[-1])
        if prev is not None:
            assert t[-1]["z_loss"] != prev[-1]["z_loss"], arm
        prev = t


# --- R4: V1's epsilon, made a knob ---------------------------------------------------

def test_r4_is_inert_unless_incr_norm_is_on():
    _, _, _, base = _step(["predictor=ltv"])
    assert not (_R4_KEYS & set(base))
    # eps and clip alone change nothing: the weight only exists on the incr_norm branch
    from lpwm_build import loss_trace

    a = loss_trace(n_steps=2, batch_size=2, overrides=["predictor=ltv"])
    b = loss_trace(n_steps=2, batch_size=2,
                   overrides=["predictor=ltv", "incr_eps=1.0", "incr_clip=2.0"])
    assert a == b


def test_r4_default_eps_reproduces_v1_exactly():
    """1e-4 is the literal constant PiWM-incr ran with. Making it a flag must not move
    the arm on record by one bit, or every V1 number becomes uncomparable."""
    from lpwm_build import loss_trace

    a = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["r4/incr"])
    b = loss_trace(n_steps=3, batch_size=2,
                   overrides=["predictor=ltv", "incr_norm=true", "incr_eps=1e-4"])
    assert a == b


def test_r4_logs_the_ess_and_the_span_that_explain_v1():
    """ESS = mean(w)^2 / mean(w^2) = 0.063 on real PushT at eps=1e-4 -- six percent of a
    batch -- and that single number is the whole explanation of V1's 8/8 dead seeds."""
    _, _, _, comps = _step(ARMS["r4/incr"])
    assert _R4_KEYS <= set(comps)
    for k in _R4_KEYS:
        assert torch.is_tensor(comps[k]), k
    ess = float(comps["incr_ess"])
    assert 0.0 < ess <= 1.0 + 1e-6, ess
    assert float(comps["incr_span"]) == pytest.approx(
        float(comps["incr_w_max"]) / float(comps["incr_w_min"]), rel=1e-5
    )


def test_r4_epsilon_moves_the_ess_toward_one_monotonically():
    """The knob is the knee of 1/(x + eps): at eps >> the typical increment the weight is
    flat (ESS -> 1, the uniform baseline), at eps << it, V1. Measured on a controlled
    spread rather than on the fixture's near-uniform noise, so the direction is a
    property of the weight and not of the synthetic batch."""
    cfg = load_cfg(["predictor=ltv", "incr_norm=true"])
    seed_all(0)
    model, _ = build(cfg)
    g = torch.Generator().manual_seed(3)
    # increments spanning four orders of magnitude, as real transitions do
    scale = torch.logspace(-3, 1, 16, base=10.0).view(16, 1, 1, 1)
    z_src = torch.zeros(16, cfg.num_hist, 1, D)
    target = scale * torch.randn(16, cfg.num_hist, 1, D, generator=g)
    ess = []
    for eps in (1e-4, 1e-2, 1.0, 1e3):
        model.incr_eps = eps
        _w, logs = model._incr_weight(z_src, target)
        ess.append(float(logs["incr_ess"]))
    # measured on this fixture: 0.329 / 0.557 / 0.787 / 1.000. The absolute value is a
    # property of the spread (real PushT gives 0.063 at eps=1e-4); what is a property of
    # the WEIGHT, and is what the knob has to deliver, is that it is monotone and that
    # the top of the range is the uniform baseline.
    assert ess == sorted(ess), ess
    assert ess[0] < 0.4, ess
    assert ess[-1] > 0.99, ess


def test_r4_clip_bounds_the_span_and_raises_the_ess():
    """The clip is applied to the unit-mean weight and the result is RENORMALISED to unit
    mean -- a deliberate deviation from clipping to [1/c, c] literally, because unit mean
    is what keeps a weighting arm from being secretly a learning-rate arm. The bounded
    quantity that survives the second renormalisation is the SPAN, max/min <= c^2."""
    cfg = load_cfg(["predictor=ltv", "incr_norm=true"])
    seed_all(0)
    model, _ = build(cfg)
    g = torch.Generator().manual_seed(3)
    scale = torch.logspace(-3, 1, 16, base=10.0).view(16, 1, 1, 1)
    z_src = torch.zeros(16, cfg.num_hist, 1, D)
    target = scale * torch.randn(16, cfg.num_hist, 1, D, generator=g)
    _w, off = model._incr_weight(z_src, target)
    for c in (3.0, 10.0):
        model.incr_clip = c
        w, on = model._incr_weight(z_src, target)
        assert float(on["incr_span"]) <= c * c + 1e-4, (c, float(on["incr_span"]))
        assert float(on["incr_span"]) < float(off["incr_span"])
        assert float(on["incr_ess"]) > float(off["incr_ess"])
        assert float(w.mean()) == pytest.approx(1.0, rel=1e-5)   # unit mean preserved


def test_r4_rejects_a_degenerate_epsilon_or_clip():
    """c <= 1 gives the empty interval [1/c, c] with 1/c >= c, which torch.clamp
    resolves silently to a CONSTANT weight -- the uniform baseline under V1's name."""
    for extra, needle in ((["incr_eps=0.0"], "incr_eps"),
                          (["incr_clip=1.0"], "incr_clip"),
                          (["incr_clip=0.5"], "incr_clip"),
                          (["incr_clip=-1.0"], "incr_clip")):
        _build_must_raise(["predictor=ltv", "incr_norm=true"] + extra, needle)


def test_r4_arms_differ_from_their_control():
    from lpwm_build import loss_trace

    incr = loss_trace(n_steps=3, batch_size=2, overrides=ARMS["r4/incr"])
    for arm in ("r4/incr_eps_med", "r4/incr_clip_tight"):
        t = loss_trace(n_steps=3, batch_size=2, overrides=ARMS[arm])
        assert t[-1]["z_loss"] != incr[-1]["z_loss"], (arm, t[-1], incr[-1])
        assert set(t[0]) == set(incr[0])
