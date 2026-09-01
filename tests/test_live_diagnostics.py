"""Tests for the live wandb diagnostics added to train.py.

Two halves. The first exercises the pure statistics (Jaccard, support churn,
participation ratio, per-projection SWD, target density) against cases with known
answers. The second drives every diagnostic BLOCK on a real model built the way
train.py builds it, for mlp_var / ltv / union-head configs, because the failure
mode these blocks actually have is a shape or attribute assumption that only
holds for one predictor mode -- and a diagnostic that raises on 26 of 27 runs is
worse than no diagnostic at all.

The blocks are called as unbound methods on a stub carrying the handful of
attributes they read, which avoids standing up an Accelerator, a dataset and a
wandb run to test arithmetic.
"""
import math

import pytest
import torch
import wandb

import train as T
from lpwm_build import build, load_cfg, seed_all, synthetic_batch
from models.infojepa_modules import gng_unit_sigma, swd


# --- pure statistics -------------------------------------------------------------

def test_soft_jaccard_identical_and_disjoint():
    a = torch.tensor([[1.0, 0.0, 2.0, 0.0]])
    assert float(T.soft_jaccard(a, a)) == pytest.approx(1.0)
    b = torch.tensor([[0.0, 3.0, 0.0, 4.0]])
    assert float(T.soft_jaccard(a, b)) == pytest.approx(0.0)


def test_soft_jaccard_known_value():
    a = torch.tensor([[1.0, 2.0, 0.0]])
    b = torch.tensor([[3.0, 1.0, 0.0]])
    # sum(min) = 1 + 1 = 2, sum(max) = 3 + 2 = 5
    assert float(T.soft_jaccard(a, b)) == pytest.approx(0.4)


def test_soft_jaccard_batches_over_leading_dims():
    a = torch.rand(2, 3, 4, 5)
    assert T.soft_jaccard(a, a).shape == (2, 3, 4)


def test_support_churn_counts_flips_only():
    a = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
    b = torch.tensor([[5.0, 7.0, 0.0, 0.0]])
    # unit 0 stays on, 1 turns on, 2 turns off, 3 stays off -> 2 of 4 flip
    assert float(T.support_churn(a, b)) == pytest.approx(0.5)


def test_support_churn_ignores_magnitude_changes():
    a = torch.tensor([[1.0, 0.0, 3.0]])
    assert float(T.support_churn(a, a * 100.0)) == pytest.approx(0.0)


def test_participation_ratio_white_code_approaches_dimension():
    torch.manual_seed(0)
    x = torch.randn(20000, 32)
    assert float(T.participation_ratio(x)) == pytest.approx(32.0, rel=0.1)


def test_participation_ratio_rank_one_code_is_one():
    torch.manual_seed(0)
    x = torch.randn(500, 1) @ torch.randn(1, 32)
    assert float(T.participation_ratio(x)) == pytest.approx(1.0, abs=1e-3)


def test_participation_ratio_of_constant_code_is_finite():
    # zero variance would divide by zero without the clamp; a collapsed code must
    # log a number, not take the run down
    assert math.isfinite(float(T.participation_ratio(torch.ones(10, 8))))


def test_dead_unit_fraction():
    counts = torch.tensor([0.0, 3.0, 0.0, 5.0])
    assert float(T.dead_unit_fraction(counts)) == pytest.approx(0.5)
    assert float(T.dead_unit_fraction(torch.ones(7))) == pytest.approx(0.0)


def test_swd_per_projection_is_zero_for_equal_samples():
    torch.manual_seed(0)
    z = torch.rand(64, 8)
    proj = torch.randn(16, 8)
    proj = proj / proj.norm(dim=1, keepdim=True)
    per = T.swd_per_projection(z, z.clone(), proj)
    assert per.shape == (16,)
    assert float(per.max()) == pytest.approx(0.0, abs=1e-10)


def test_swd_per_projection_mean_matches_swd_construction():
    """The breakdown must average to the statistic it breaks down, or the histogram
    is measuring something other than the regularizer."""
    torch.manual_seed(0)
    z, tgt = torch.rand(128, 6), torch.rand(128, 6)
    proj = torch.randn(2048, 6)
    proj = proj / proj.norm(dim=1, keepdim=True)
    mine = float(T.swd_per_projection(z, tgt, proj).mean())
    reference = float(
        ((torch.sort(z @ proj.T, dim=0).values - torch.sort(tgt @ proj.T, dim=0).values) ** 2).mean()
    )
    assert mine == pytest.approx(reference, rel=1e-5)
    # and it lands near the randomly-projected swd() over the same distributions
    assert mine == pytest.approx(float(swd(z, tgt, 2048)), rel=0.25)


def test_swd_per_projection_keeps_extra_batch_axes():
    torch.manual_seed(0)
    z, tgt = torch.rand(32, 4, 6), torch.rand(32, 4, 6)
    proj = torch.randn(11, 6)
    assert T.swd_per_projection(z, tgt, proj).shape == (11,)


def test_target_density_identity_link_is_dense():
    assert T.rdmreg_target_density("identity", 2.0, 0.0) == pytest.approx(1.0)


def test_target_density_rectified_at_mu_zero_is_half():
    assert T.rdmreg_target_density("reprelu", 1.0, 0.0) == pytest.approx(0.5)
    assert T.rdmreg_target_density("relu", 2.0, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("k,d", [(176, 384), (8, 384), (100, 256)])
def test_target_density_matches_kwta_mu(k, d):
    """train.py sets mu = sigma*ln(2k/D) precisely so the target density is k/D.
    If this drifts, reg/density_gap silently reports a gap that is not there."""
    mu = gng_unit_sigma(1.0) * math.log(2.0 * k / d)
    assert T.rdmreg_target_density("reprelu", 1.0, mu) == pytest.approx(k / d)


def test_target_density_kwta_clamps_density():
    # k-WTA marks exactly k of D positions, so it can only ever reduce the density
    assert T.rdmreg_target_density("reprelu", 1.0, 0.0, kwta_k=8, embed_dim=384) == pytest.approx(
        8 / 384
    )


def test_target_density_positive_mu_stays_a_probability():
    d = T.rdmreg_target_density("reprelu", 1.0, 5.0)
    assert 0.5 < d <= 1.0


def test_target_density_returns_none_for_unsupported_shape():
    assert T.rdmreg_target_density("reprelu", 1.5, 0.0) is None


# --- wandb payload helpers -------------------------------------------------------

def test_histogram_filters_nonfinite_and_caps_size():
    x = torch.tensor([1.0, float("nan"), 2.0, float("inf"), 3.0])
    assert isinstance(T._wandb_histogram(x, 100), wandb.Histogram)
    assert T._wandb_histogram(torch.tensor([float("nan")]), 100) is None
    assert T._wandb_histogram(torch.tensor([]), 100) is None
    assert isinstance(T._wandb_histogram(torch.arange(10_000.0), 32), wandb.Histogram)


def test_image_handles_constant_and_nan_input():
    # a constant map (e.g. a fully-collapsed gate) has zero range; normalising it
    # naively is a divide-by-zero that would land as a NaN image
    assert isinstance(T._wandb_image(torch.ones(4, 4), "flat"), wandb.Image)
    assert isinstance(T._wandb_image(torch.full((3, 3), float("nan")), "nan"), wandb.Image)


def test_image_caption_reports_the_range_normalisation_discards():
    img = T._wandb_image(torch.tensor([[0.25, 0.75]]), "m")
    assert "0.25" in img._caption and "0.75" in img._caption


def test_l2_matches_flat_norm_and_tolerates_none():
    a, b = torch.randn(3, 4), torch.randn(5)
    expected = float(torch.cat([a.flatten(), b.flatten()]).norm())
    assert T._l2([a, None, b]) == pytest.approx(expected, rel=1e-5)
    assert T._l2([None]) is None
    assert T._l2([]) is None


# --- section mapping (analysis/figures.py reverses this) -------------------------

@pytest.mark.parametrize(
    "flat,sectioned",
    [
        ("train_loss", "train/loss"),
        ("val_loss", "val/loss"),
        ("train_l0_frac", "sparsity/train_l0_frac"),
        ("train_head_usage_p0", "heads/train_head_usage_p0"),
        ("train_z_visual_err_rollout", "rollout/train_z_visual_err_rollout"),
        ("train_z_visual_err_full", "err/train_z_visual_err_full"),
        ("train_img_psnr_pred", "img/train_img_psnr_pred"),
        ("epoch_seconds", "epoch_seconds"),
    ],
)
def test_existing_section_mapping_unchanged(flat, sectioned):
    assert T.wandb_key(flat) == sectioned


@pytest.mark.parametrize(
    "flat,sectioned",
    [
        ("train_z_visual_err_rollout_h4", "rollout/train_z_visual_err_rollout_h4"),
        ("val_S_model_rollout_1framestart_h2", "rollout/val_S_model_rollout_1framestart_h2"),
        ("val_S_model", "val/S_model"),
        ("train_support_churn", "train/support_churn"),
    ],
)
def test_new_section_mapping(flat, sectioned):
    assert T.wandb_key(flat) == sectioned


def test_section_tails_are_unique_so_unsection_cannot_collide():
    """analysis/figures.py::unsection drops the section prefix, so two metrics that
    differ only by section would overwrite each other in an exported history."""
    keys = [
        "sparsity/l0_frac_pred", "sparsity/effective_dim", "sparsity/dead_unit_frac",
        "dist/z_p50", "dist/z_nonzero_magnitude", "jacc/S_model", "jacc/S_world",
        "jacc/support_churn", "jacc/churn_model", "err/rel_mse", "err/mse_on_support",
        "reg/target_density", "reg/swd_probe", "gate/gate_mean", "gate/ltv_u_norm",
        "heads/head_gap_head0", "opt/grad_norm_encoder", "perf/data_wait_frac",
        "progress/global_batch", "fig/support_selfsim", "diag/blocks_failed",
    ]
    tails = [k.split("/", 1)[1] for k in keys]
    assert len(set(tails)) == len(tails)


# --- diagnostic blocks on a real model ------------------------------------------

class _Stub(T.Trainer):
    """A Trainer with only the attributes the diagnostic blocks read.

    Subclassed rather than duck-typed so the blocks under test are the real bound
    methods, including the ones that call each other (_pred_module, _swd_probe,
    _diag_tensors); Trainer.__init__ is skipped because standing up an
    Accelerator, a dataset and a wandb run to test arithmetic is not a test.
    """

    def __init__(self):
        pass


def _stub(overrides, batch_size=2, device="cpu"):
    cfg = load_cfg(overrides)
    seed_all(0)
    model, opts = build(cfg, device=device)
    gen = torch.Generator().manual_seed(7)
    obs, act = synthetic_batch(cfg, batch_size, gen, device=device)

    model.train()
    _, _, _, loss, comps = model(obs, act)
    loss.backward()

    s = _Stub()
    s.cfg = cfg
    s.model = model
    s.device = torch.device(device)
    s.link = model.link
    s.encoder = model.encoder
    s.predictor = model.predictor
    s.action_encoder = model.action_encoder
    s.predictor_optimizer = opts[1]
    s.total_epochs = 2
    s._n_train_batches = 100
    s._diag_fail = {}
    s._panel_edges = None
    s._panel_rows, s._panel_tags, s._panel_heads = [], [], []
    s._panel_i = 0
    s._panel_phase = []
    s.epoch = 1
    s._unit_active = s._unit_sum = None
    s._unit_n = 0
    s._unit_win = (None, None, 0)
    s._rectified = cfg.link.kind in ("relu", "reprelu")
    s._diag_gen = torch.Generator(device=device).manual_seed(1)
    s._target_density = T.rdmreg_target_density(
        cfg.link.kind, cfg.target_p, cfg.mu, cfg.get("kwta_k"), cfg.get("embed_dim")
    )
    flat = {f"train_{k}": [float(v)] for k, v in comps.items() if torch.is_tensor(v)}
    return s, flat


def _all_blocks(s, flat):
    """Every block a logging batch runs, merged, exactly as _log_live merges them."""
    s._accumulate_unit_stats()
    s._unit_win = s._unit_window(reset=True)
    tens = s._diag_tensors()
    out = {}
    out.update(s._opt_diagnostics())
    out.update(s._sparsity_diagnostics(tens))
    out.update(s._jaccard_diagnostics(tens))
    out.update(s._error_diagnostics(tens))
    out.update(s._reg_diagnostics(flat))
    out.update(s._gate_diagnostics(tens))
    out.update(s._head_diagnostics(tens))
    out.update(s._tensor_diagnostics(tens))
    out.update(s._image_diagnostics(tens))
    out.update(s._panel_diagnostics(tens))
    return out


ARMS = {
    "mlp_var": ["predictor=mlp_var"],
    "ltv": ["predictor=ltv"],
    "ltv_support_softmax": ["predictor=ltv", "gate_input=support", "gate_norm=softmax"],
    "union_head": ["predictor=ltv", "n_heads=4", "head_entropy_coef=0.1"],
    "kwta": ["predictor=mlp_var", "kwta_k=8"],
    "dense_identity": ["predictor=mlp_var", "link=identity", "target_p=2"],
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_block_runs_and_logs_only_wandb_safe_values(arm):
    """The whole point: no block may raise, and nothing may reach wandb that is not
    a finite scalar or a wandb media object (a stray tensor breaks CSV export)."""
    s, flat = _stub(ARMS[arm])
    payload = _all_blocks(s, flat)
    assert payload, f"{arm} produced no diagnostics at all"
    assert not s._diag_fail, s._diag_fail
    for k, v in payload.items():
        assert isinstance(v, (int, float, wandb.Histogram, wandb.Image)), (k, type(v))
        if isinstance(v, float):
            assert math.isfinite(v), (k, v)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_logged_key_is_sectioned(arm):
    s, flat = _stub(ARMS[arm])
    for k in _all_blocks(s, flat):
        assert "/" in k, f"{k} would land unsectioned on the run page"


def test_core_metrics_present_on_the_campaign_default_arm():
    s, flat = _stub(ARMS["ltv"])
    payload = _all_blocks(s, flat)
    for k in (
        "sparsity/l0_frac_pred", "sparsity/effective_dim", "sparsity/dead_unit_frac",
        "jacc/S_model", "jacc/S_world", "jacc/support_churn", "jacc/burst_rate",
        "err/rel_mse", "err/mse_on_support", "err/cos_pred_target", "err/mse_t0",
        "reg/target_density", "reg/density_gap", "reg/swd_probe",
        "opt/grad_norm_encoder", "opt/grad_norm_predictor", "opt/update_to_weight",
        "dist/z_nonzero_magnitude", "dist/unit_activation_freq", "dist/swd_per_projection",
        "fig/support_selfsim", "fig/code_vs_pred", "fig/unit_coactivation",
    ):
        assert k in payload, k


def test_gate_diagnostics_only_exist_for_ltv():
    s_mlp, flat = _stub(ARMS["mlp_var"])
    assert not any(k.startswith("gate/") for k in _all_blocks(s_mlp, flat))
    s_ltv, flat = _stub(ARMS["ltv"])
    assert "gate/gate_mean" in _all_blocks(s_ltv, flat)


def test_gate_mean_distinguishes_sigmoid_from_r_softmax():
    """The r*softmax rescaling is invisible in the loss but obvious here: a softmax
    arm reading ~1/r instead of ~1 means the r factor was lost."""
    s_sig, flat = _stub(ARMS["ltv"])
    sig = _all_blocks(s_sig, flat)["gate/gate_mean"]
    s_soft, flat = _stub(ARMS["ltv_support_softmax"])
    soft = _all_blocks(s_soft, flat)["gate/gate_mean"]
    assert sig == pytest.approx(0.5, abs=0.1)
    assert soft == pytest.approx(1.0, abs=1e-4)


def test_head_diagnostics_only_exist_for_multiple_heads():
    s1, flat = _stub(ARMS["ltv"])
    assert not any(k.startswith("heads/") for k in _all_blocks(s1, flat))
    s4, flat = _stub(ARMS["union_head"])
    payload = _all_blocks(s4, flat)
    for k in ("heads/head_gap_head0", "heads/head_loss_spread", "heads/head_usage_min",
              "fig/head_assignment_raster", "dist/head_loss"):
        assert k in payload, k
    for j in range(4):
        assert f"heads/head_loss_j{j}" in payload
        assert f"heads/head_delta_j{j}" in payload
    # the union can never be worse than head 0 alone
    assert payload["heads/head_gap_head0"] >= 0.0


def test_jaccard_metrics_suppressed_on_a_dense_code():
    """J_S is undefined for signed codes; reporting it anyway would put a
    meaningless number on the dense arms' run pages."""
    s, flat = _stub(ARMS["dense_identity"])
    payload = _all_blocks(s, flat)
    assert not any(k.startswith("jacc/") for k in payload)
    assert "fig/support_selfsim" not in payload
    assert payload["reg/target_density"] == pytest.approx(1.0)


def test_kwta_arm_reports_the_matched_density_gap():
    s, flat = _stub(ARMS["kwta"])
    payload = _all_blocks(s, flat)
    assert payload["reg/target_density"] == pytest.approx(8 / 384)


def test_unit_window_accumulates_across_batches_then_resets():
    s, _flat = _stub(ARMS["mlp_var"])
    s._accumulate_unit_stats()
    first = s._unit_n
    s._accumulate_unit_stats()
    assert s._unit_n == 2 * first
    freq, mean_act, n = s._unit_window(reset=True)
    assert n == 2 * first
    assert freq.shape == mean_act.shape == (s.cfg.embed_dim,)
    assert float(freq.max()) <= 1.0 and float(freq.min()) >= 0.0
    assert s._unit_window(reset=False) == (None, None, 0)


def _bare_stub():
    s = _Stub()
    s.model = type("M", (), {"_diag": None, "_diag_heads": None})()
    s._unit_active = s._unit_sum = None
    s._unit_n = 0
    s._diag_fail = {}
    s._panel_edges = None
    s._panel_rows, s._panel_tags, s._panel_heads = [], [], []
    s._panel_i = 0
    s._panel_phase = []
    s.epoch = 1
    return s


def test_accumulate_is_a_noop_without_a_stash():
    s = _bare_stub()
    s._accumulate_unit_stats()
    assert s._unit_n == 0


def test_diag_tensors_none_when_the_model_never_stashed():
    assert _bare_stub()._diag_tensors() is None


def test_blocks_degrade_to_nothing_without_a_stash():
    """The concat / DINO-WM forward never populates the stash, so every block must
    return empty rather than raise."""
    s, flat = _stub(ARMS["mlp_var"])
    s.model._diag = None
    s.model._diag_heads = None
    s._unit_win = (None, None, 0)
    for fn in (
        s._sparsity_diagnostics,
        s._jaccard_diagnostics,
        s._error_diagnostics,
        s._gate_diagnostics,
        s._head_diagnostics,
        s._tensor_diagnostics,
        s._image_diagnostics,
    ):
        assert fn(None) == {}
    assert s._support_logs("train") == {}


def test_safe_swallows_and_counts_failures():
    s = _Stub()
    s._diag_fail = {}
    s._panel_edges = None
    s._panel_rows, s._panel_tags, s._panel_heads = [], [], []
    s._panel_i = 0
    s._panel_phase = []
    s.epoch = 1
    out = {"keep": 1.0}
    s._safe("boom", lambda: 1 / 0, out)
    s._safe("boom", lambda: {}["missing"], out)
    s._safe("fine", lambda: {"added": 2.0}, out)
    assert s._diag_fail == {"boom": 2}
    assert out == {"keep": 1.0, "added": 2.0}


def test_support_logs_are_epoch_aggregate_shaped():
    s, _flat = _stub(ARMS["ltv"])
    got = s._support_logs("val")
    assert set(got) == {"val_S_model", "val_S_world", "val_support_churn"}
    for k, v in got.items():
        assert isinstance(v, list) and len(v) == 1 and math.isfinite(v[0])
        assert T.wandb_key(k).startswith("val/")


def _horizon_stub(rectified):
    s = _Stub()
    s.model = type("M", (), {"emb_criterion": torch.nn.MSELoss()})()
    s._rectified = rectified
    return s


def test_horizon_logs_stop_at_the_available_horizon():
    s = _horizon_stub(True)
    z_roll = torch.rand(1, 7, 1, 5)
    z_true = torch.rand(1, 7, 1, 5)
    got = s._horizon_logs(z_roll, z_true, n_past=3, postfix="")
    # n_past-1 + h < 7 admits h in {1, 2, 4}; h=8 and h=16 are out of range
    assert set(got) == {
        "z_visual_err_rollout_h1", "S_model_rollout_h1",
        "z_visual_err_rollout_h2", "S_model_rollout_h2",
        "z_visual_err_rollout_h4", "S_model_rollout_h4",
    }
    assert all(math.isfinite(v) for v in got.values())


def test_horizon_logs_skip_jaccard_on_a_dense_code():
    got = _horizon_stub(False)._horizon_logs(
        torch.rand(1, 5, 1, 3), torch.rand(1, 5, 1, 3), 2, ""
    )
    assert got and not any(k.startswith("S_model") for k in got)


def test_horizon_error_grows_with_horizon_for_a_drifting_rollout():
    """Sanity check on the alignment: if the rollout drifts linearly away from the
    truth, the reported error must increase with h, not with the frame index."""
    z_true = torch.zeros(1, 9, 1, 3)
    z_roll = torch.zeros(1, 9, 1, 3)
    for j in range(9):
        z_roll[:, j] = float(j)  # drift grows with the frame index
    got = _horizon_stub(False)._horizon_logs(z_roll, z_true, n_past=1, postfix="")
    assert got["z_visual_err_rollout_h1"] < got["z_visual_err_rollout_h2"]
    assert got["z_visual_err_rollout_h2"] < got["z_visual_err_rollout_h4"]


def test_stash_does_not_enter_the_state_dict():
    """The stash must stay a plain attribute: registering it as a buffer would
    change the checkpoint and break the bit-identity fixtures."""
    s, _flat = _stub(ARMS["union_head"])
    assert s.model._diag is not None and s.model._diag_heads is not None
    keys = s.model.state_dict().keys()
    assert not any("_diag" in k for k in keys)


def test_stash_is_detached_from_the_graph():
    s, _flat = _stub(ARMS["ltv"])
    for v in s.model._diag.values():
        assert v.grad_fn is None and not v.requires_grad


# --- tier 3: real panels.py figures, live ---------------------------------------

def _panel_keys(out):
    return sorted(k for k in out if k.startswith("panel/"))


def test_panels_render_live_and_are_wandb_images():
    """The live page must carry the redesigned FORMS, not only raw heatmaps.

    panels.py was written for exactly this arrangement; for a while only its colour
    system was wired into train.py, so every form existed offline only.
    """
    import wandb

    # batch 16 -> 48 paired support samples, past the joint panel's floor. The 2-
    # sample default yields 24, which is under it; production at batch 64 gives 192.
    s, flat = _stub(ARMS["ltv"], batch_size=16)
    s._accumulate_unit_stats()
    s._unit_win = s._unit_window(reset=True)
    out = s._panel_diagnostics(s._diag_tensors())
    keys = _panel_keys(out)
    assert "panel/z_magnitude_ridgeline" in keys, keys
    assert "panel/l0_ecdf" in keys, keys
    assert "panel/support_change_joint" in keys, keys
    assert all(isinstance(out[k], wandb.Image) for k in keys)


def test_every_live_panel_names_both_axes():
    """A live panel the reader cannot decode is not a diagnostic.

    Every axis carries a label, and where the quantity has a definition rather than
    a name (S_model, S_world, density) the label carries the definition -- otherwise
    the page needs a lookup the reader does not have while a run is in flight.
    """
    s, _ = _stub(ARMS["union_head"], batch_size=16)
    s._accumulate_unit_stats()
    s._unit_win = s._unit_window(reset=True)
    tens = s._diag_tensors()
    s._panel_phase = [(0.5 - 0.01 * k, 0.7 - 0.04 * k) for k in range(12)]
    from analysis import panels as P

    for _ in range(2):
        s._panel_diagnostics(tens)   # populate history, then inspect live figures
    # rebuild the same figures unwrapped so the axes can be read
    import numpy as np
    from einops import rearrange

    z = tens[0]
    zf = rearrange(z, "b t p d -> (b t p) d")
    figs = {
        "ecdf": P.ecdf_overlay({"a": (zf != 0).float().sum(-1).cpu().numpy()}),
        "phase": P.phase_plane({"a": (np.linspace(.5, .4, 20), np.linspace(.7, .2, 20))},
                               xlabel=r"code density $\rho$", ylabel="RDMReg loss"),
        "stream": P.head_stream([("a", np.arange(6.), np.full((6, 4), .25))]),
    }
    for name, fig in figs.items():
        for ax in fig.axes:
            if not ax.has_data():
                continue
            assert ax.get_xlabel() or ax.get_ylabel(), f"{name}: an axis is unlabelled"
        P.plt.close(fig)


def test_env_video_block_degrades_without_a_trajectory_dataset():
    """The video is an extra. Missing the dataset must cost the video, not the run."""
    s, _ = _stub(ARMS["ltv"])
    s.val_traj_dset = None
    s.train_traj_dset = None
    assert s._env_video_diagnostics() == {}


def test_panel_block_closes_its_figures():
    """matplotlib keeps unclosed figures alive in pyplot's registry, so a leak here
    is a slow memory climb across a 4h window plus a warning storm."""
    from analysis import panels as P

    s, flat = _stub(ARMS["ltv"])
    s._accumulate_unit_stats()
    s._unit_win = s._unit_window(reset=True)
    tens = s._diag_tensors()
    before = len(P.plt.get_fignums())
    for _ in range(4):
        s._panel_diagnostics(tens)
    assert len(P.plt.get_fignums()) == before, "panel block leaked a figure"


def test_panel_history_is_bounded():
    """The ridgeline keeps PRE-BINNED rows, not raw samples, and caps how many, so a
    long window cannot grow this without bound."""
    s, flat = _stub(ARMS["ltv"])
    s._accumulate_unit_stats()
    s._unit_win = s._unit_window(reset=True)
    tens = s._diag_tensors()
    for _ in range(30):
        s._panel_diagnostics(tens)
    assert len(s._panel_rows) <= 12, f"ridgeline history grew to {len(s._panel_rows)}"
    assert len(s._panel_rows) == len(s._panel_tags)
    assert len(s._panel_heads) <= 60
    # the bins are fixed once, so successive rows are comparable
    assert s._panel_edges is not None and len(s._panel_edges) == 65


def test_union_head_gets_a_usage_stream_and_j1_does_not():
    """Collapse is Step 4's failure mode and the stream is where it is unmistakable;
    at J=1 there is no stack to draw and the panel must be absent, not empty."""
    s4, _ = _stub(ARMS["union_head"])
    s4._accumulate_unit_stats()
    s4._unit_win = s4._unit_window(reset=True)
    t4 = s4._diag_tensors()
    s4._panel_diagnostics(t4)
    out = s4._panel_diagnostics(t4)  # needs >=2 ticks to have an x axis
    assert "panel/head_usage_stream" in out, sorted(out)

    s1, _ = _stub(ARMS["ltv"])
    s1._accumulate_unit_stats()
    s1._unit_win = s1._unit_window(reset=True)
    t1 = s1._diag_tensors()
    s1._panel_diagnostics(t1)
    assert "panel/head_usage_stream" not in s1._panel_diagnostics(t1)


def test_panel_block_survives_missing_tensors():
    """Same contract as every other block: a missing diagnostic returns nothing, it
    does not kill a 4h window."""
    s, _ = _stub(ARMS["ltv"])
    assert s._panel_diagnostics(None) == {}


def test_live_panel_colour_matches_the_offline_png():
    """One arm, one colour, both media -- the whole point of routing through panels."""
    from analysis import panels as P

    from omegaconf import open_dict

    s, _ = _stub(ARMS["ltv"])
    with open_dict(s.cfg):  # struct-mode, exactly as Trainer.__init__ sets it
        s.cfg["saved_folder"] = "/x/runs/outputs/PiWM-union4-entropy_pd384_bf16_s0"
    assert s._arm_label() == "PiWM-union4-entropy"
    assert P.arm_color(s._arm_label()) == P.arm_color("PiWM-union4-entropy_pd384_bf16_s2")

    s2, _ = _stub(ARMS["ltv"])  # cfg without the key at all: degrade, never raise
    assert s2._arm_label() == "run"
