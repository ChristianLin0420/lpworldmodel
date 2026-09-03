"""M1 guards.

The probe's whole value is that its NULLS are trustworthy, so the tests are about the nulls
and about the two ways a probe silently lies:

  * an angular error computed without wrapping reports ~360 deg for a 0.08 deg miss, and a
    median over such values looks like a healthy null while hiding a perfect decoder;
  * a frame-level split leaks, because consecutive frames of one trajectory are
    near-duplicates.  Grouping must be on the episode, in the CV and in the bootstrap.

Plus: chance for a wrapped angle is 90 deg, and the ridge solver must actually be ridge --
the same estimator in its primal and its dual form, which is the only reason a 98688-feature
patch checkpoint is affordable at all.
"""
import os

import numpy as np
import pytest
import torch

from analysis.latent_probe import (ProbeData, _eig_solver, _predict, _weights, errors,
                                   group_kfold, run_probe, _median_se)

DATASET = os.path.join(os.environ.get("DATASET_DIR", ""), "pusht_noise")
has_data = os.path.isdir(DATASET)


# ---------------------------------------------------------------------------- ridge
@pytest.mark.parametrize("n,p", [(60, 12), (12, 60)])       # primal and dual regimes
def test_ridge_matches_closed_form(n, p):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, p, generator=g, dtype=torch.float64)
    Y = torch.randn(n, 3, generator=g, dtype=torch.float64)
    sol = _eig_solver(X, Y)
    for lam in (1e-2, 1.0, 1e3):
        want = torch.linalg.solve(X.T @ X + lam * torch.eye(p, dtype=torch.float64), X.T @ Y)
        assert torch.allclose(_weights(sol, lam), want, atol=1e-8), lam
        assert torch.allclose(_predict(sol, lam, X), X @ want, atol=1e-8), lam


def test_primal_and_dual_agree():
    """p > n and p < n must be the same estimator, not two."""
    g = torch.Generator().manual_seed(1)
    X = torch.randn(30, 80, generator=g, dtype=torch.float64)
    Y = torch.randn(30, 2, generator=g, dtype=torch.float64)
    dual = _eig_solver(X, Y)                       # p > n
    assert dual["dual"]
    lam = 2.0
    want = torch.linalg.solve(X.T @ X + lam * torch.eye(80, dtype=torch.float64), X.T @ Y)
    assert torch.allclose(_weights(dual, lam), want, atol=1e-8)


# ---------------------------------------------------------------------------- angles
def test_angular_error_wraps():
    """theta near +pi predicted near -pi is a small error, not ~360 deg."""
    st = np.zeros((3, 7))
    st[:, 4] = [np.pi - 1e-3, 0.0, -np.pi + 1e-3]
    pred = np.stack([st[:, 2], st[:, 3],
                     np.cos(st[:, 4] + 0.02), np.sin(st[:, 4] + 0.02)], 1)
    _, ang = errors(pred, st)
    assert np.allclose(ang, np.degrees(0.02), atol=1e-6), ang


def test_chance_is_ninety_degrees():
    """A probe on pure noise must land at the wrapped-angle chance level, ~90 deg."""
    rs = np.random.RandomState(0)
    n_ep, per = 40, 12
    st = np.zeros((n_ep * per, 7))
    st[:, 4] = np.repeat(rs.uniform(-np.pi, np.pi, n_ep), per)
    st[:, 2:4] = np.repeat(rs.uniform(100, 400, (n_ep, 2)), per, axis=0)
    cache = {"state": torch.as_tensor(st, dtype=torch.float32),
             "ep": np.repeat(np.arange(n_ep), per),
             "split": np.array(["train"] * (n_ep * per), dtype=object),
             "role": np.array(["fit"] * (n_ep * per // 2) + ["val"] * (n_ep * per // 2),
                              dtype=object)}
    pd_ = ProbeData(cache, "cpu")
    F = torch.randn(n_ep * per, 24, generator=torch.Generator().manual_seed(3))
    m, _ = run_probe(F, pd_, list(10.0 ** np.arange(-3, 10)))
    assert 55.0 < m["err_ang_deg"] < 125.0, m["err_ang_deg"]


# ---------------------------------------------------------------------------- grouping
def test_group_kfold_never_splits_an_episode():
    groups = np.repeat(np.arange(23), 7)
    seen = np.zeros(len(groups), bool)
    for tr, te in group_kfold(groups, k=5, seed=0):
        assert not (set(groups[tr]) & set(groups[te])), "episode in both train and test"
        seen |= te
    assert seen.all(), "some rows were never in a test fold"


def test_bootstrap_se_resamples_episodes():
    """A cluster bootstrap must be wider than a frame bootstrap on clustered data."""
    groups = np.repeat(np.arange(20), 30)
    vals = np.repeat(np.random.RandomState(0).uniform(0, 100, 20), 30)
    _, se_cluster = _median_se(vals, groups, n_boot=200)
    _, se_frame = _median_se(vals, np.arange(len(vals)), n_boot=200)
    assert se_cluster > 2 * se_frame, (se_cluster, se_frame)


# ---------------------------------------------------------------------------- cache
@pytest.mark.skipif(not has_data, reason="DATASET_DIR/pusht_noise not present")
def test_cache_is_the_tensor_the_model_sees(tmp_path):
    from analysis.probe_cache import build
    c = build(DATASET, {"val"}, train_eps=3, heldout_eps=0, stride=25, val_stride=25,
              val_eps=6)
    v, st = c["visual"], c["state"]
    assert v.dtype == torch.float16 and v.shape[1:] == (3, 224, 224)
    assert -1.05 <= v.min().item() and v.max().item() <= 1.05      # Normalize(0.5, 0.5)
    assert st.shape[1] == 7                                        # [ax ay bx by th vx vy]
    assert set(c["role"]) == {"fit", "val"}
    fit_eps = set(c["ep"][c["role"] == "fit"])
    val_eps = set(c["ep"][c["role"] == "val"])
    assert not (fit_eps & val_eps), "an episode is in both the fit and the eval set"
