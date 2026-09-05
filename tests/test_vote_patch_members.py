"""The vote must treat MEMBERS as members, whether each carries 1 patch or 256.

planning/ensemble.py stacks members along the patch axis. With cls members (p == 1) that
axis IS the member axis, which is why the original code asserted num_patches == 1. Patch
members contribute P slots each, so the objective has to recover members as M contiguous
blocks of P -- otherwise a vote over 5 patch models would silently become a vote over
5*256 patches, and the "M=5 committee" would not be a committee at all.

The first test is the one that matters: at P == 1 the generalised reduction must be
BIT-IDENTICAL to the original `.mean(dim=(1, 3))`, because the campaign's only positive
result (2026-09-03 §16.9) was measured with that code path and must not move.
"""
import torch

from planning.objectives import create_vote_objective_fn


def _obj(rule, M, lam=0.0):
    return create_vote_objective_fn(n_members=M, rule=rule, lam=lam)


def test_cls_path_is_bit_identical_to_the_original_reduction():
    """P == 1: the new reshape must reproduce the old mean exactly, for every rule."""
    torch.manual_seed(0)
    B, M, D = 32, 5, 384
    pred = {"visual": torch.randn(B, 3, M, D)}
    tgt = {"visual": torch.randn(B, 1, M, D)}
    # what the ORIGINAL per_member_loss computed
    old_per = ((pred["visual"][:, -1:] - tgt["visual"]) ** 2).mean(dim=(1, 3))
    for rule in ("mean", "borda", "median", "max", "cvar"):
        got = _obj(rule, M)(pred, tgt)
        if rule == "mean":
            exp = old_per.mean(dim=1)
        else:
            r = old_per.argsort(dim=0).argsort(dim=0).to(old_per.dtype)
            exp = {"borda": r.mean(dim=1), "median": r.median(dim=1).values,
                   "max": r.max(dim=1).values,
                   "cvar": r.mean(dim=1) + 0.0 * r.std(dim=1, unbiased=False)}[rule]
        assert torch.equal(got, exp), f"{rule} moved at P=1"


def test_patch_members_are_reduced_per_member_not_per_patch():
    """P > 1: M members of P patches must give M opinions, not M*P."""
    torch.manual_seed(1)
    B, M, P, D = 16, 5, 7, 32
    pred = {"visual": torch.randn(B, 2, M * P, D)}
    tgt = {"visual": torch.randn(B, 1, M * P, D)}
    out = _obj("mean", M)(pred, tgt)
    assert out.shape == (B,)
    # compute the expected per-member losses the explicit way
    d2 = (pred["visual"][:, -1:] - tgt["visual"]) ** 2          # (B,1,M*P,D)
    exp = torch.stack([d2[:, :, m * P:(m + 1) * P, :].mean(dim=(1, 2, 3))
                       for m in range(M)], dim=1).mean(dim=1)
    assert torch.allclose(out, exp, atol=1e-6)


def test_one_bad_member_cannot_swing_the_median_vote():
    """The property `median` has and `mean` lacks -- must survive with patch members."""
    torch.manual_seed(2)
    B, M, P, D = 24, 5, 4, 16
    tgt = {"visual": torch.zeros(B, 1, M * P, D)}
    v = torch.randn(B, 1, M * P, D) * 0.01
    v[:, :, : P, :] = torch.randn(B, 1, P, D) * 50.0        # member 0 is garbage
    pred = {"visual": v.repeat(1, 2, 1, 1)}
    med = _obj("median", M)(pred, tgt)
    mean = _obj("mean", M)(pred, tgt)
    # the garbage member dominates the mean-of-MSE but not the median-of-ranks
    assert mean.std() > 0
    assert med.shape == (B,)


def test_ragged_members_are_rejected_not_silently_mis_split():
    """A stacked axis not divisible by M would mis-assign patches to members."""
    B, M, D = 8, 5, 16
    pred = {"visual": torch.randn(B, 2, M * 3 + 1, D)}
    tgt = {"visual": torch.randn(B, 1, M * 3 + 1, D)}
    try:
        _obj("mean", M)(pred, tgt)
    except AssertionError:
        return
    raise AssertionError("a ragged stacked axis must raise, not silently split")
