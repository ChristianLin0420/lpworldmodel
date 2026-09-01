"""Figure suite: the helpers get value tests, the panels get a smoke render.

A broken figure that only shows up after a 4-hour training run is expensive, so
every panel is rendered here against synthetic inputs with the real schema.
"""
import json

import numpy as np
import pytest

from analysis import figures as F


# --- helpers --------------------------------------------------------------------

def test_roc_curve_endpoints_and_monotonicity():
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    fpr, tpr = F.roc_curve(s, y)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


def test_roc_curve_area_matches_auroc():
    """The gate is quoted as AUROC, so the curve must integrate to the same number."""
    from analysis.predictive_jaccard import auroc
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=500), (rng.normal(size=500) > 0).astype(int)
    fpr, tpr = F.roc_curve(s, y)
    assert np.trapz(tpr, fpr) == pytest.approx(auroc(s, y), abs=2e-3)


def test_roc_curve_is_perfect_for_separable_scores():
    fpr, tpr = F.roc_curve([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0])
    assert np.trapz(tpr, fpr) == pytest.approx(1.0)


def test_peri_event_windows_are_centred_on_the_event():
    x = np.arange(20.0)
    onset = np.zeros(20)
    onset[10] = 1.0
    rows = F.peri_event(x, onset, window=3)
    assert len(rows) == 1
    assert list(rows[0]) == [7, 8, 9, 10, 11, 12, 13]
    assert rows[0][3] == 10, "column `window` must be the onset frame"


def test_peri_event_drops_events_too_close_to_the_edges():
    onset = np.zeros(20)
    onset[[1, 10, 18]] = 1.0
    assert len(F.peri_event(np.arange(20.0), onset, window=3)) == 1


def test_peri_event_returns_nothing_without_events():
    assert F.peri_event(np.arange(20.0), np.zeros(20), window=3) == []


def test_mean_ci_brackets_the_mean_and_shrinks_with_n():
    rng = np.random.default_rng(2)
    small = rng.normal(size=(8, 5))
    large = rng.normal(size=(400, 5))
    for rows in (small, large):
        m, lo, hi = F.mean_ci(rows)
        assert np.allclose(m, rows.mean(0))
        assert np.all(lo <= m) and np.all(m <= hi)
    assert (F.mean_ci(large)[2] - F.mean_ci(large)[1]).mean() < \
           (F.mean_ci(small)[2] - F.mean_ci(small)[1]).mean()


# --- statistics -----------------------------------------------------------------

def test_t_crit_matches_published_values():
    """n=3 seeds means df=2, where t is 4.303 and not 1.96; the gates live or die
    on that factor, so it is asserted rather than approximated."""
    assert F.t_crit(2) == pytest.approx(4.303, abs=1e-3)
    assert F.t_crit(10) == pytest.approx(2.228, abs=1e-3)
    assert F.t_crit(1000) == pytest.approx(1.96, abs=1e-2)
    assert F.t_crit(11) < F.t_crit(10), "t must shrink as df grows"
    assert F.t_crit(0) == np.inf


def test_paired_effect_differences_only_shared_seeds():
    arms = {"ctrl": {"0": 0.4, "1": 0.3, "2": 0.5},
            "var": {"0": 0.5, "1": 0.4, "3": 9.9}}
    eff = F.paired_effect(arms, "ctrl", "var")
    assert eff["seeds"] == ["0", "1"]
    assert eff["n"] == 2
    assert eff["mean"] == pytest.approx(0.1)
    assert eff["sd"] == pytest.approx(0.0, abs=1e-12)


def test_paired_effect_dz_is_mean_over_sd():
    arms = {"c": {0: 0.0, 1: 0.0, 2: 0.0}, "v": {0: 1.0, 1: 2.0, 2: 3.0}}
    eff = F.paired_effect(arms, "c", "v")
    assert eff["mean"] == pytest.approx(2.0)
    assert eff["sd"] == pytest.approx(1.0)
    assert eff["dz"] == pytest.approx(2.0)
    assert eff["lo"] < eff["mean"] < eff["hi"]


def test_paired_power_rises_with_effect_and_with_n():
    sd = 1.0
    assert F.paired_power(0.0, sd, 3) < 0.10, "power at a null effect is alpha-ish"
    assert F.paired_power(0.5, sd, 3) < F.paired_power(3.0, sd, 3)
    assert F.paired_power(1.0, sd, 3) < F.paired_power(1.0, sd, 20)
    assert np.isnan(F.paired_power(1.0, 0.0, 3))


def test_mde_shrinks_with_more_seeds():
    a, b = F.mde(0.05, 3), F.mde(0.05, 10)
    assert np.isfinite(a) and np.isfinite(b) and b < a
    assert F.paired_power(a, 0.05, 3) >= 0.8


def test_sign_flip_min_p_makes_n3_hopeless():
    """2/2**n at n=3 is 0.25, so a paired permutation test cannot reach 0.05."""
    assert F.sign_flip_min_p(3) == pytest.approx(0.25)
    assert F.sign_flip_min_p(6) == pytest.approx(0.03125)
    assert F.sign_flip_min_p(1) == pytest.approx(1.0)


def test_variance_decomposition_separates_arm_from_seed_noise():
    tight = {"a": {0: 0.0, 1: 0.0}, "b": {0: 1.0, 1: 1.0}}
    b, w, icc = F.variance_decomposition(tight)
    assert w == pytest.approx(0.0) and b > 0 and icc == pytest.approx(1.0)
    noisy = {"a": {0: -5.0, 1: 5.0}, "b": {0: -5.0, 1: 5.0}}
    b2, w2, icc2 = F.variance_decomposition(noisy)
    assert b2 == pytest.approx(0.0) and w2 > 0 and icc2 == pytest.approx(0.0)
    assert np.isnan(F.variance_decomposition({"a": {0: 1.0}})[0])


def test_unit_activity_counts_frequency_and_active_magnitude():
    z = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    freq, mag = F.unit_activity(z)
    assert list(freq) == [0.5, 0.0]
    assert mag[0] == pytest.approx(2.0)
    assert mag[1] == pytest.approx(0.0)


def test_participation_ratio_spans_one_to_D():
    assert F.participation_ratio(np.ones(16)) == pytest.approx(16.0)
    assert F.participation_ratio(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)
    assert F.participation_ratio(np.zeros(4)) == 0.0


def test_dead_fraction_counts_never_active_units():
    z = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    assert F.dead_fraction(z) == pytest.approx(1 / 3)


def test_xcorr_lag_finds_a_planted_lead():
    """A statistic that rises 2 frames BEFORE the event must peak at lag -2."""
    rng = np.random.default_rng(3)
    T = 400
    event = np.zeros(T)
    event[rng.choice(np.arange(20, T - 20), 40, replace=False)] = 1.0
    sig = np.roll(event, -2) + 0.01 * rng.normal(size=T)
    lags, r = F.xcorr_lag(sig, event, max_lag=5)
    assert lags[int(np.nanargmax(r))] == -2


def test_hist_matrix_rows_are_normalised_distributions():
    entries = [{"step": 10, "edges": [0, 1, 2], "counts": [1, 3]},
               {"step": 0, "edges": [0, 1, 2], "counts": [4, 0]}]
    steps, edges, mat = F.hist_matrix(entries, n_bins=4)
    assert list(steps) == [0, 10], "rows must be ordered by step"
    assert mat.shape == (2, 4)
    assert np.allclose(mat.sum(axis=1), 1.0)
    assert F.hist_matrix([]) is None


# --- key sectioning and history loading -----------------------------------------

def test_unsection_reverses_the_wandb_key_mapping():
    """train.py sections wandb keys; the figure code addresses the flat names."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from train import wandb_key

    for flat in ("train_loss", "val_loss", "train_head_usage_p0", "train_l0_frac",
                 "val_z_loss", "train_diag_cov_loss", "train_z_visual_err_rollout",
                 "train_img_visual_reconstructed", "epoch", "epoch_seconds"):
        assert F.unsection(wandb_key(flat)) == flat, flat


def test_unsection_survives_sections_it_has_never_seen():
    """The training side keeps adding sections; an unknown one must be dropped, not
    rejected, or the whole suite breaks the next time one appears."""
    assert F.unsection("brand_new_section/train_gate_mean") == "train_gate_mean"
    assert F.unsection("brand_new_section/val_gate_mean") == "val_gate_mean"
    assert F.unsection("brand_new_section/gate_mean") == "gate_mean"
    assert F.unsection("progress/epoch_frac") == "epoch_frac"
    assert F.unsection("perf/batches_per_sec") == "batches_per_sec"
    assert F.unsection("a/b/train_loss") == "train_loss", "nested sections too"
    assert F.unsection("no_section_at_all") == "no_section_at_all"


def test_flat_aliases_peels_one_section_at_a_time():
    assert F.flat_aliases("train/loss") == ["train/loss", "train_loss"]
    assert F.flat_aliases("heads/train_head_usage_p0") == [
        "heads/train_head_usage_p0", "train_head_usage_p0"]
    assert F.flat_aliases("a/b/c") == ["a/b/c", "b/c", "c"]
    assert F.flat_aliases("plain") == ["plain"]


def test_load_csv_turns_blanks_into_nan(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("epoch,train_loss\n1,0.5\n2,\n3,0.1\n")
    h = F.load_csv(p)
    assert list(h["epoch"]) == [1, 2, 3]
    assert np.isnan(h["train_loss"][1])


def test_load_csv_registers_flat_aliases_for_sectioned_columns(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("_step,train/loss,sparsity/train_l0_frac,opt/grad_norm\n"
                 "0,0.5,0.4,0.01\n50,0.4,0.3,0.02\n")
    h = F.load_csv(p)
    for k in ("train/loss", "train_loss", "train_l0_frac", "grad_norm"):
        assert k in h, k
    assert list(h["train_loss"]) == [0.5, 0.4]


def test_load_csv_handles_a_header_only_export(tmp_path):
    """A run that has not logged yet exports a header and nothing else; that used
    to raise IndexError and take the whole figure run down."""
    p = tmp_path / "h.csv"
    p.write_text("_step,train/loss\n")
    h = F.load_csv(p)
    assert h["train_loss"].size == 0


def test_load_csv_sorts_rows_by_step(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("_step,train/loss\n100,0.1\n0,0.9\n50,0.5\n")
    h = F.load_csv(p)
    assert list(h["_step"]) == [0, 50, 100]
    assert list(h["train_loss"]) == [0.9, 0.5, 0.1]


def test_load_csv_prefers_the_alias_with_more_data(tmp_path):
    """Two sectioned columns can collapse onto one flat name; the emptier one is
    almost always a stale duplicate and must not win."""
    p = tmp_path / "h.csv"
    p.write_text("a/train_l0_frac,b/train_l0_frac\n,0.3\n,0.4\n")
    h = F.load_csv(p)
    assert list(h["train_l0_frac"]) == [0.3, 0.4]


def test_x_axis_prefers_epoch_frac_over_a_pinned_epoch():
    """Live runs sit inside epoch 1 for hours, so an epoch axis would collapse the
    whole run onto one vertical line."""
    h = {"epoch": np.ones(5), "epoch_frac": np.linspace(0, 0.2, 5),
         "_step": np.arange(5.0) * 50}
    x, label, key = F.x_axis(h)
    assert key == "epoch_frac" and "epoch" in label
    assert np.allclose(x, h["epoch_frac"])


def test_x_axis_falls_back_through_step_then_index():
    h = {"epoch": np.ones(4), "_step": np.arange(4.0)}
    assert F.x_axis(h)[2] == "_step"
    assert F.x_axis({"train_loss": np.arange(4.0)})[2] == "index"
    assert F.x_axis({"epoch": np.arange(1.0, 5.0)})[2] == "epoch"


def test_marks_for_only_returns_markers_in_the_plotted_unit():
    run = {"resumes": {"epoch_frac": [0.5], "step": [15000.0]}}
    assert F.marks_for(run, "epoch_frac") == [0.5]
    assert F.marks_for(run, "_step") == [15000.0]
    assert F.marks_for(run, "epoch") == [], "must not plot a step on an epoch axis"
    assert F.marks_for(run, "index") == []
    assert F.marks_for({"resumes": [4, 8]}, "epoch") == [4.0, 8.0], "legacy list"


def test_run_arm_and_seed_match_the_campaign_naming():
    assert F.run_arm("PiWM-union4-entropy_pd384_bf16_s0") == "PiWM-union4-entropy"
    assert F.run_arm("PiWM-sparse-2pct_pd384_bf16_s2") == "PiWM-sparse-2pct"
    assert F.run_seed("LpWM-ltv_pd384_bf16_s1") == 1
    assert F.run_seed("no_seed_here") is None


def test_default_baseline_prefers_the_upstream_control():
    assert F.default_baseline(["PiWM-gate-sup-softmax", "LpWM-ltv"]) == "LpWM-ltv"
    assert F.default_baseline(["PiWM-sparse-2pct", "LpWM-base"]) == "LpWM-base"
    assert F.default_baseline(["b", "a"]) == "b", "no control: keep given order"


# --- io -------------------------------------------------------------------------

def test_load_step1_round_trips_the_analysis_output(tmp_path):
    t = F._synth(tmp_path)
    summary, eps = F.load_step1(t / "analysis_step1.json")
    assert len(eps) == summary["n_episodes"] == 6
    for e in eps:
        assert set(e) >= {"S_world", "S_model", "onset", "block_disp", "z", "j_star"}
        assert len(e["S_world"]) == len(e["onset"])


def test_load_runs_skips_dirs_without_a_history_export(tmp_path):
    t = F._synth(tmp_path)
    (t / "runs" / "no_export").mkdir()
    runs = F.load_runs(str(t / "runs" / "*"))
    assert {r["name"] for r in runs} == {n for n, *_ in F.SYNTH_RUNS}
    assert all(r["hist"] for r in runs)


def test_load_runs_attaches_arm_seed_and_exporter_metadata(tmp_path):
    t = F._synth(tmp_path)
    runs = {r["name"]: r for r in F.load_runs(str(t / "runs" / "*"))}
    r = runs["PiWM-union4-entropy_pd384_bf16_s1"]
    assert r["arm"] == "PiWM-union4-entropy" and r["seed"] == 1
    assert r["meta"]["config"]["n_heads"] == 4
    assert r["resumes"]["epoch_frac"]
    assert r["hists"]["dist/z_l0_per_sample"]


def test_load_runs_tolerates_a_header_only_export(tmp_path, capsys):
    t = F._synth(tmp_path)
    d = t / "runs" / "just_started"
    d.mkdir()
    (d / "wandb_history.csv").write_text("_step,train/loss\n")
    names = {r["name"] for r in F.load_runs(str(t / "runs" / "*"))}
    assert "just_started" not in names
    assert "no rows yet" in capsys.readouterr().out


def test_by_arm_groups_seeds_and_puts_controls_first(tmp_path):
    t = F._synth(tmp_path)
    groups = F.by_arm(F.load_runs(str(t / "runs" / "*")))
    assert list(groups)[0] in ("LpWM-base", "LpWM-ltv")
    assert [r["seed"] for r in groups["LpWM-base"]] == [0, 1, 2]


def test_align_series_puts_runs_on_a_shared_grid_and_nans_outside(tmp_path):
    t = F._synth(tmp_path)
    runs = F.load_runs(str(t / "runs" / "*"))
    al = F.align_series(runs, "train_loss")
    assert al["mat"].shape[0] == len(runs)
    assert al["mat"].shape[1] == 140
    assert np.isfinite(al["mat"]).any()
    assert F.align_series(runs, "a_metric_nobody_logs") is None


# --- panels ---------------------------------------------------------------------

STEP1_PANELS = [
    "01_peri_event", "02_roc_overlay", "06_jaccard_decomposition",
    "07_support_selfsim", "08_head_raster", "12_burst_vs_error",
    "26_code_geometry", "27_onset_lead_lag", "28_head_onset_alignment",
    "29_per_head_dynamics",
]
CAMPAIGN_PANELS = [
    "00_campaign_overview", "03_paired_dumbbell", "05_gate_scorecard",
    "09_scale_perturbation", "11_success_vs_k", "30_ladder", "31_effect_sizes",
    "32_gate_values", "33_gate_heatmap", "34_power_curve",
]
RUN_PANELS = [
    "04_head_usage", "10_engagement", "13_training_curves", "14_training_health",
    "15_metric_coverage", "16_sparsity_trajectories", "17_rdmreg_vs_l0",
    "18_loss_decomposition", "19_gradient_health", "20_throughput",
    "21_preemption_timeline", "22_seed_variance", "23_head_specialisation",
    "24_head_switch_burst", "25_l0_distribution",
]
ALL_PANELS = STEP1_PANELS + CAMPAIGN_PANELS + RUN_PANELS + ["99_contact_sheet"]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("figs")
    t = F._synth(root / "in")
    out = root / "out"
    _, eps = F.load_step1(t / "analysis_step1.json")
    c = json.loads((t / "campaign.json").read_text())
    runs = F.load_runs(str(t / "runs" / "*"))
    made = [
        F.fig_campaign_overview(c["arms"], out),
        F.fig_peri_event(eps, out),
        F.fig_roc_overlay(eps, out, n_boot=20),
        F.fig_paired_dumbbell(c["arms"], out),
        F.fig_gate_scorecard(c["gates"], out),
        F.fig_jaccard_decomposition(eps, out),
        F.fig_support_selfsim(eps, out),
        F.fig_head_raster(eps, out),
        F.fig_scale_perturbation(c["scale"], out),
        F.fig_success_vs_k(c["k_sweep"], out),
        F.fig_burst_vs_error(eps, out),
        F.fig_code_geometry(eps, out),
        F.fig_onset_lead_lag(eps, out, n_null=20),
        F.fig_head_onset_alignment(eps, out),
        F.fig_per_head_dynamics(eps, out, n_boot=40),
        F.fig_ladder(c["ladder"], out),
        F.fig_effect_sizes(c["arms"], out),
        F.fig_gate_values(c["gate_values"], out),
        F.fig_gate_heatmap(c["gate_heatmap"], out),
        F.fig_power_curve(c["arms"], out),
        F.fig_metric_coverage(runs, out),
        F.fig_head_usage(runs, out),
        F.fig_engagement(runs, out),
        F.fig_training_curves(runs, out),
        F.fig_training_health(runs, out),
        F.fig_sparsity_trajectories(runs, out),
        F.fig_rdmreg_vs_l0(runs, out),
        F.fig_loss_decomposition(runs, out),
        F.fig_gradient_health(runs, out),
        F.fig_throughput(runs, out),
        F.fig_preemption_timeline(runs, out),
        F.fig_seed_variance(runs, out),
        F.fig_head_specialisation(runs, out),
        F.fig_head_switch_burst(runs, out),
        F.fig_l0_distribution(runs, out),
    ]
    F.fig_contact_sheet(made, out)
    return out


@pytest.mark.parametrize("name", ALL_PANELS)
def test_panel_renders_a_nonempty_png(rendered, name):
    p = rendered / f"{name}.png"
    assert p.exists(), f"{name} was skipped despite its input being present"
    assert p.stat().st_size > 5_000, f"{name} rendered a near-empty image"


def test_load_bearing_five_are_all_present(rendered):
    """The plan names these five as load-bearing; the rest are reach items."""
    for name in ("01_peri_event", "02_roc_overlay", "03_paired_dumbbell",
                 "04_head_usage", "05_gate_scorecard"):
        assert (rendered / f"{name}.png").exists()


def test_no_stray_panels_were_written(rendered):
    assert sorted(p.stem for p in rendered.glob("*.png")) == sorted(ALL_PANELS)


def test_arms_group_into_the_three_gates():
    """Each gate is (title, named control, variant prefix), control first.

    "LpWM-ltv" is the shared control of the gating and union gates, so it heads BOTH
    of those panels and must not leak into the sparse one. The control is named
    rather than prefix-derived: baselines are "LpWM-*" and variants "PiWM-*", so a
    prefix rule would leave every contrast without its reference.
    """
    arms = {"LpWM-base": {}, "PiWM-sparse-2pct": {}, "LpWM-ltv": {},
            "PiWM-gate-sup-sigmoid": {}, "PiWM-union4-entropy": {}}
    g = F.group_arms(arms)
    assert set(g) == {"Sparse codes: k-WTA", "Support gating", "Union head"}
    assert g["Sparse codes: k-WTA"][0] == "LpWM-base"
    assert g["Support gating"][0] == "LpWM-ltv"
    assert g["Union head"][0] == "LpWM-ltv"
    assert "LpWM-ltv" not in g["Sparse codes: k-WTA"]
    assert "LpWM-base" not in g["Support gating"]


def test_success_vs_k_accepts_keys_that_do_not_survive_str_round_tripping(tmp_path):
    """'0.10' floats to 0.1 and str()s back to '0.1', which used to KeyError."""
    assert F.fig_success_vs_k({"0.10": {"success": 0.3, "rdmreg": 0.2},
                               "0.50": {"success": 0.4, "rdmreg": 0.1}},
                              tmp_path) is not None


# --- graceful degradation -------------------------------------------------------

def test_panels_skip_rather_than_crash_on_missing_inputs(tmp_path):
    """Steps 1-2 report before Step 4 runs, so head figures must return None
    instead of raising when no run has head columns."""
    assert F.fig_head_usage([], tmp_path) is None
    assert F.fig_engagement([], tmp_path) is None
    assert F.fig_training_curves([], tmp_path) is None
    assert F.fig_head_raster([{"j_star": None, "onset": np.zeros(3)}], tmp_path) is None


@pytest.mark.parametrize("fn", [
    "fig_metric_coverage", "fig_sparsity_trajectories", "fig_rdmreg_vs_l0",
    "fig_loss_decomposition", "fig_gradient_health", "fig_throughput",
    "fig_preemption_timeline", "fig_seed_variance", "fig_head_specialisation",
    "fig_head_switch_burst", "fig_l0_distribution", "fig_training_health",
])
def test_run_panels_skip_on_an_empty_campaign(fn, tmp_path):
    assert getattr(F, fn)([], tmp_path) is None


@pytest.mark.parametrize("fn", [
    "fig_ladder", "fig_effect_sizes", "fig_gate_values", "fig_gate_heatmap",
    "fig_power_curve", "fig_scale_perturbation", "fig_success_vs_k",
    "fig_gate_scorecard", "fig_campaign_overview",
])
def test_campaign_panels_skip_on_empty_input(fn, tmp_path):
    assert getattr(F, fn)({}, tmp_path) is None


@pytest.mark.parametrize("fn", [
    "fig_code_geometry", "fig_onset_lead_lag", "fig_head_onset_alignment",
    "fig_per_head_dynamics", "fig_roc_overlay", "fig_burst_vs_error",
    "fig_support_selfsim", "fig_head_raster",
])
def test_step1_panels_skip_without_episodes(fn, tmp_path):
    assert getattr(F, fn)([], tmp_path) is None


def test_peri_event_panel_skips_when_no_onsets_are_labelled(tmp_path):
    eps = [{"S_world": np.zeros(20), "S_model": np.zeros(20), "onset": np.zeros(20)}]
    assert F.fig_peri_event(eps, tmp_path) is None


def test_a_skip_records_why_and_what_would_unblock_it(tmp_path):
    """The point of the rewrite: a panel must never vanish without saying why."""
    F._LAST_SKIP.clear()
    assert F.fig_l0_distribution([], tmp_path) is None
    reason, unblock = F.take_skip()
    assert "dist/z_l0_per_sample" in reason
    assert "export_wandb" in unblock


def test_contact_sheet_marks_missing_panels_instead_of_reflowing(tmp_path):
    """A skipped panel has to stay visible on the summary page."""
    F.fig_gate_scorecard([{"name": "g", "observed": 0.1, "lo": 0.0, "hi": 0.2,
                           "threshold": 0.0}], tmp_path)
    p = F.fig_contact_sheet([tmp_path / "05_gate_scorecard.png"], tmp_path)
    assert p is not None and p.stat().st_size > 5_000
    assert F.fig_contact_sheet([], tmp_path) is None


# --- driver ---------------------------------------------------------------------

def test_selftest_writes_every_panel(tmp_path, monkeypatch):
    """`--selftest` is the pre-run smoke check; it must cover every panel."""
    rc = F.main(["--selftest", "--selftest_dir", str(tmp_path / "in"),
                 "--out", str(tmp_path / "out"), "--strict"])
    got = sorted(p.stem for p in (tmp_path / "out").glob("*.png"))
    assert got == sorted(ALL_PANELS)
    assert rc == 0, "--selftest must satisfy --strict"


def test_strict_mode_fails_when_a_panel_is_skipped(tmp_path, capsys):
    """CI needs a way to insist a panel stays renderable."""
    rc = F.main(["--out", str(tmp_path / "out"), "--strict"])
    printed = capsys.readouterr().out
    assert rc == 1
    assert "SKIPPED" in printed
    assert "export_wandb.py" in printed, "the fix must name the exporter"
    assert F.main(["--out", str(tmp_path / "out2")]) == 0, "no --strict: exit 0"
