"""Step 1 statistics. These are the numbers the gate is read off, so they are
tested against closed-form values rather than golden outputs."""
import numpy as np
import pytest

from analysis.predictive_jaccard import (
    AGENT_RADIUS,
    auroc,
    contact_signals,
    episode_bootstrap,
    pearson,
    soft_jaccard,
    tee_surface_distance,
)


# --- J_S ------------------------------------------------------------------------

def test_jaccard_of_identical_vectors_is_one():
    a = np.array([[0.0, 1.0, 2.0, 0.0]])
    assert soft_jaccard(a, a) == pytest.approx(1.0, abs=1e-6)


def test_jaccard_of_disjoint_supports_is_zero():
    a = np.array([[1.0, 0.0, 3.0, 0.0]])
    b = np.array([[0.0, 2.0, 0.0, 4.0]])
    assert soft_jaccard(a, b) == pytest.approx(0.0, abs=1e-6)


def test_jaccard_matches_set_jaccard_on_binary_supports():
    a = np.array([[1.0, 1.0, 1.0, 0.0]])
    b = np.array([[1.0, 1.0, 0.0, 1.0]])
    assert soft_jaccard(a, b) == pytest.approx(2.0 / 4.0, abs=1e-6)


def test_continuous_jaccard_is_below_one_on_a_shared_support():
    """The reason the chance floor rho/(2-rho) is only an UPPER bound: matching
    supports with different magnitudes still score strictly less than 1."""
    a = np.array([[1.0, 1.0]])
    b = np.array([[1.0, 3.0]])
    assert soft_jaccard(a, b) == pytest.approx(2.0 / 4.0)
    assert soft_jaccard(a, b) < 1.0


def test_jaccard_is_symmetric_and_scale_invariant():
    rng = np.random.default_rng(0)
    a, b = np.abs(rng.normal(size=(16, 32))), np.abs(rng.normal(size=(16, 32)))
    assert np.allclose(soft_jaccard(a, b), soft_jaccard(b, a))
    assert np.allclose(soft_jaccard(a, b), soft_jaccard(3.0 * a, 3.0 * b), atol=1e-6)


# --- AUROC ----------------------------------------------------------------------

def test_auroc_perfect_separation_is_one():
    assert auroc(np.array([0.0, 0.1, 0.9, 1.0]), [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auroc_reversed_separation_is_zero():
    assert auroc(np.array([1.0, 0.9, 0.1, 0.0]), [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auroc_all_ties_is_one_half():
    """Tie correction matters: a constant statistic must score chance, not 1.0."""
    assert auroc(np.ones(6), [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)


def test_auroc_is_nan_without_both_classes():
    assert np.isnan(auroc(np.arange(5.0), [0, 0, 0, 0, 0]))
    assert np.isnan(auroc(np.arange(5.0), [1, 1, 1, 1, 1]))


def test_auroc_equals_probability_of_correct_ranking():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=200), rng.integers(0, 2, size=200).astype(bool)
    brute = np.mean([sp > sn for sp in s[y] for sn in s[~y]])
    assert auroc(s, y) == pytest.approx(brute, abs=1e-9)


# --- episode bootstrap ----------------------------------------------------------

def test_bootstrap_point_estimate_uses_all_frames():
    rng = np.random.default_rng(2)
    scores = [rng.normal(size=30) for _ in range(8)]
    labels = [rng.integers(0, 2, size=30).astype(float) for _ in range(8)]
    pt, lo, hi = episode_bootstrap(auroc, scores, labels, n_boot=200, seed=0)
    assert pt == pytest.approx(auroc(np.concatenate(scores), np.concatenate(labels)))
    assert lo <= pt <= hi


def test_episode_bootstrap_is_wider_than_frame_bootstrap():
    """The whole point of resampling episodes: frame-level CIs are fraudulently
    tight when the statistic is autocorrelated within an episode."""
    rng = np.random.default_rng(3)
    n_ep, n_fr = 10, 60
    # strong within-episode correlation: label and score share an episode offset
    scores, labels = [], []
    for _ in range(n_ep):
        off = rng.normal() * 3.0
        scores.append(rng.normal(size=n_fr) + off)
        labels.append((rng.normal(size=n_fr) + off > 0).astype(float))
    ep = episode_bootstrap(auroc, scores, labels, n_boot=400, seed=0)
    flat_s, flat_l = np.concatenate(scores), np.concatenate(labels)
    frame = episode_bootstrap(
        auroc, [np.array([v]) for v in flat_s], [np.array([v]) for v in flat_l],
        n_boot=400, seed=0,
    )
    assert (ep[2] - ep[1]) > (frame[2] - frame[1])


def test_bootstrap_is_deterministic_given_seed():
    rng = np.random.default_rng(4)
    s = [rng.normal(size=20) for _ in range(6)]
    l = [rng.integers(0, 2, size=20).astype(float) for _ in range(6)]
    assert episode_bootstrap(auroc, s, l, 200, seed=7) == episode_bootstrap(auroc, s, l, 200, seed=7)


# --- PushT geometry and onset labels -------------------------------------------

def _states(agent_xy, block_xy=(0.0, 0.0), ang=0.0):
    a = np.asarray(agent_xy, float).reshape(-1, 2)
    s = np.zeros((len(a), 5))
    s[:, 0:2] = a
    s[:, 2:4] = block_xy
    s[:, 4] = ang
    return s


def _below(gaps):
    """Agent centres directly under the bar's bottom face (y=0), `gap` clear of it.

    Approaching from below keeps the bar the nearest face at every distance; a
    point above the bar hits the stem's side face and the clearance saturates.
    """
    return _states([(0.0, -(AGENT_RADIUS + g)) for g in np.atleast_1d(gaps)])


def test_distance_is_negative_inside_the_tee():
    """(0, 15) is the centre of the horizontal bar, so clearance is very negative."""
    assert tee_surface_distance(_states([(0.0, 15.0)]))[0] < -AGENT_RADIUS


def test_distance_is_zero_at_one_agent_radius_from_the_surface():
    assert tee_surface_distance(_below(0.0))[0] == pytest.approx(0.0, abs=1e-6)


def test_distance_grows_linearly_moving_away():
    gaps = np.array([0.0, 5.0, 10.0, 20.0])
    assert np.allclose(tee_surface_distance(_below(gaps)), gaps, atol=1e-6)


def test_distance_uses_the_stem_not_just_the_bar():
    """(0, 100) is inside the vertical stem (cy=75, hy=45), which a bar-only or
    centroid-based distance would miss."""
    assert tee_surface_distance(_states([(0.0, 100.0)]))[0] < 0.0


def test_distance_takes_the_nearest_of_the_two_rects():
    """Above the bar and beside the stem, the stem's side face (x=15) is nearer
    than the bar's top face once you are far enough up."""
    d = tee_surface_distance(_states([(40.0, 65.0)]))[0]
    assert d == pytest.approx(40.0 - 15.0 - AGENT_RADIUS, abs=1e-6)


def test_distance_respects_block_rotation():
    """Rotating the block must carry its faces with it: the point that just
    touches the upright bar from below ends up inside the rotated block."""
    p = [(0.0, -(AGENT_RADIUS + 15.0))]
    upright = tee_surface_distance(_states(p, ang=0.0))[0]
    turned = tee_surface_distance(_states(p, ang=np.pi / 2))[0]
    assert upright == pytest.approx(15.0, abs=1e-6)
    assert turned < 0.0


def test_distance_is_translation_invariant():
    d0 = tee_surface_distance(_states([(0.0, 60.0)], block_xy=(0.0, 0.0)))[0]
    d1 = tee_surface_distance(_states([(200.0, 260.0)], block_xy=(200.0, 200.0)))[0]
    assert d0 == pytest.approx(d1, abs=1e-6)


def test_onset_fires_once_per_approach_not_every_touching_frame():
    """Onset must be an edge, not a level: otherwise it is just 'touching'."""
    sig = contact_signals(_below([10.0, 5.0, 1.0, 1.0, 1.0, 10.0, 1.0]), tau=2.0)
    assert list(sig["touching"]) == [0, 0, 1, 1, 1, 0, 1]
    assert list(sig["onset"]) == [0, 0, 1, 0, 0, 0, 1]
    assert list(sig["offset"]) == [0, 0, 0, 0, 0, 1, 0]


def test_onset_is_never_set_on_the_first_frame():
    """d_{t-1} does not exist at t=0, so the label is undefined there."""
    sig = contact_signals(_states([(0.0, 15.0), (0.0, 15.0)]), tau=2.0)
    assert sig["onset"][0] == 0.0


def test_onset_precedes_block_displacement():
    """The premise of the whole test: at the onset frame the block has not moved,
    so any statistic built from observed frame-to-frame change is blind to it."""
    s = _below([10.0, 5.0, 1.0, 0.5])
    s[3, 2:4] = (0.0, 3.0)  # block only moves the frame AFTER contact
    sig = contact_signals(s, tau=2.0)
    onset_t = int(np.argmax(sig["onset"]))
    assert sig["onset"][onset_t] == 1.0
    assert sig["block_disp"][onset_t] == pytest.approx(0.0)
    assert sig["block_disp"][onset_t + 1] > 0.0


def test_displacements_are_zero_at_frame_zero_and_are_norms():
    s = _states([(0.0, 0.0), (3.0, 4.0)])
    sig = contact_signals(s, tau=2.0)
    assert sig["agent_disp"][0] == 0.0
    assert sig["agent_disp"][1] == pytest.approx(5.0)


def test_larger_tau_cannot_lower_the_touch_rate():
    """Monotonicity, which is what makes the tau sweep interpretable."""
    rng = np.random.default_rng(5)
    s = _states(rng.uniform(-150, 150, size=(400, 2)))
    rates = [contact_signals(s, tau=t)["touching"].mean() for t in (0.0, 2.0, 5.0, 20.0)]
    assert rates == sorted(rates)


# --- pearson --------------------------------------------------------------------

def test_pearson_is_one_for_a_positive_linear_relation():
    x = np.arange(10.0)
    assert pearson(x, 2 * x + 1) == pytest.approx(1.0)


def test_pearson_is_nan_for_a_constant_input():
    assert np.isnan(pearson(np.ones(5), np.arange(5.0)))
