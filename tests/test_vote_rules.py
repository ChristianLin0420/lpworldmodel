"""V1: the four rank rules are order statistics of ONE rank matrix.

`planning/objectives.create_vote_objective_fn` gained `cvar` (mean rank + lam * rank
std) and `max` (worst member's rank). The whole point of the V1 sweep is that borda /
median / cvar / max differ ONLY in how the same (B, M) rank matrix is reduced, so these
tests pin the two properties the experiment depends on:

  * `cvar` at the default `lam=0.0` is bit-identical to `borda` -- so the new key is
    provably inert unless set, and `PiWM-vote5-cvar LAM=0` is a free correctness control;
  * `max >= borda` elementwise -- the pessimism ladder is ordered by construction.

Plus the M=1 nan trap: `PiWMvoteM1` is a live arm and `std(unbiased=True)` of one sample
is nan, which would poison every candidate score. `unbiased=False` is load-bearing.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from planning.objectives import create_vote_objective_fn as vote  # noqa: E402

RULES = ("mean", "borda", "median", "cvar", "max")


def _cand(n_cand=64, m=5, d=8, seed=0):
    """(pred, tgt) shaped like the ensemble's (B, T, M, D_max) CEM batch."""
    g = torch.Generator().manual_seed(seed)
    pred = {"visual": torch.randn(n_cand, 1, m, d, generator=g)}
    tgt = {"visual": torch.randn(1, 1, m, d, generator=g).expand(n_cand, 1, m, d)}
    return pred, tgt


def test_cvar_lam0_is_bit_identical_to_borda():
    p, g = _cand()
    assert torch.equal(vote(5, "cvar", lam=0.0)(p, g), vote(5, "borda")(p, g))


def test_cvar_default_lam_is_zero():
    """The signature default must be inert: not passing `lam` == borda."""
    p, g = _cand()
    assert torch.equal(vote(5, "cvar")(p, g), vote(5, "borda")(p, g))


def test_max_dominates_borda_elementwise():
    p, g = _cand()
    assert bool((vote(5, "max")(p, g) >= vote(5, "borda")(p, g)).all())


def test_pessimism_ladder_is_monotone_in_lam():
    """borda = cvar(0) <= cvar(1) <= cvar(2); all bounded above by max."""
    p, g = _cand()
    b = vote(5, "borda")(p, g)
    c1 = vote(5, "cvar", lam=1.0)(p, g)
    c2 = vote(5, "cvar", lam=2.0)(p, g)
    mx = vote(5, "max")(p, g)
    assert bool((c1 >= b).all()) and bool((c2 >= c1).all())
    # max is the M-th order statistic; mean + 2*std need not reach it, but mean does not
    # exceed it, which is the direction the ladder claims.
    assert bool((mx >= b).all())


def test_existing_rules_unchanged():
    """mean / borda / median must be exactly what they were before V1."""
    p, g = _cand(seed=3)
    per = ((p["visual"][:, -1:] - g["visual"]) ** 2).mean(dim=(1, 3))
    ranks = per.argsort(dim=0).argsort(dim=0).to(per.dtype)
    assert torch.equal(vote(5, "mean")(p, g), per.mean(dim=1))
    assert torch.equal(vote(5, "borda")(p, g), ranks.mean(dim=1))
    assert torch.equal(vote(5, "median")(p, g), ranks.median(dim=1).values)
    assert torch.equal(vote(5, "max")(p, g), ranks.max(dim=1).values)


@pytest.mark.parametrize("rule", RULES)
def test_m1_all_rules_order_like_raw_mse_and_are_finite(rule):
    """PiWMvoteM1 is live: no rule may produce nan at M=1, and all five must induce
    the same candidate ordering as the plain MSE (the ensemble wrapper is inert)."""
    p, g = _cand(n_cand=32, m=1, seed=7)
    per = ((p["visual"][:, -1:] - g["visual"]) ** 2).mean(dim=(1, 3)).squeeze(-1)
    v = vote(1, rule, lam=1.0)(p, g)
    assert not bool(torch.isnan(v).any()), f"{rule} produced nan at M=1"
    assert torch.equal(v.argsort(), per.argsort())


def test_unknown_rule_rejected():
    with pytest.raises(AssertionError, match="not supported"):
        vote(5, "cvarr")


@pytest.mark.parametrize("rule", RULES)
def test_shapes_and_dtype(rule):
    p, g = _cand()
    v = vote(5, rule, lam=1.5)(p, g)
    assert v.shape == (64,) and v.dtype == torch.float32
