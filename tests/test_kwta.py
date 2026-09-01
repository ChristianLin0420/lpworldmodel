"""Step 2 -- k-WTA codes in the shared Link."""
import math

import pytest
import torch

from models.infojepa_modules import Link, gng_unit_sigma, kwta


def test_exactly_k_nonzero_per_sample():
    """L0 of every row is exactly k when at least k coordinates are positive."""
    torch.manual_seed(0)
    x = torch.rand(8, 384) + 0.1  # all positive, so top-k are all > 0
    for k in (1, 7, 64):
        z = kwta(x, k)
        assert torch.equal(
            (z != 0).sum(-1), torch.full((8,), k)
        ), f"k={k}: L0 per row is not exactly k"


def test_never_more_than_k_nonzero():
    """Even with heavy ties/zeros (post-rectification), never more than k survive.

    A `x >= kth_value` implementation fails this: with fewer than k positives the
    k-th largest is 0 and every zero coordinate passes the test.
    """
    x = torch.zeros(4, 384)
    x[:, :3] = torch.tensor([3.0, 2.0, 1.0])  # only 3 positives, k larger than that
    z = kwta(x, 7)
    assert (z != 0).sum(-1).max().item() <= 7
    assert torch.equal((z != 0).sum(-1), torch.full((4,), 3))


def test_keeps_the_largest_values():
    x = torch.tensor([[0.1, 5.0, 0.2, 4.0, 0.3]])
    z = kwta(x, 2)
    assert z.tolist() == [[0.0, 5.0, 0.0, 4.0, 0.0]]


def test_masked_units_receive_gradient():
    """Straight-through: a coordinate zeroed in the forward pass still gets grad,
    otherwise a losing unit could never win again."""
    x = torch.tensor([[0.1, 5.0, 0.2, 4.0, 0.3]], requires_grad=True)
    kwta(x, 2).sum().backward()
    masked = x.grad[0, [0, 2, 4]]
    assert (masked != 0).any(), "no masked unit received gradient"
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


def test_link_default_is_identical_to_upstream():
    """kwta_k=None must leave the link untouched (invariant 1)."""
    torch.manual_seed(0)
    u = torch.randn(4, 3, 1, 384)
    for kind in ("identity", "relu", "reprelu"):
        assert torch.equal(Link(kind).forward(u), Link(kind, kwta_k=None).forward(u))


def test_link_applies_kwta_and_stays_nonnegative():
    torch.manual_seed(0)
    u = torch.randn(4, 3, 1, 384)
    z = Link("reprelu", kwta_k=7).forward(u)
    assert (z >= 0).all(), "J_S needs a nonnegative code"
    assert (z != 0).sum(-1).max().item() <= 7


def test_link_has_no_parameters():
    """Link must stay stateless so checkpoints are unchanged by this flag."""
    assert list(Link("reprelu", kwta_k=7).state_dict().keys()) == []


@pytest.mark.parametrize("k,D", [(7, 384), (64, 384), (20, 1000)])
def test_mu_matching_gives_target_density_k_over_D(k, D):
    """mu = sigma*ln(2k/D) must make P(GN_1 + mu > 0) == k/D for the Laplace target."""
    sigma = gng_unit_sigma(1.0)
    mu = sigma * math.log(2.0 * k / D)
    torch.manual_seed(0)
    base = torch.distributions.Laplace(0.0, sigma).sample((400_000,)) + mu
    got, want = (base > 0).float().mean().item(), k / D
    assert got == pytest.approx(want, rel=0.05), f"density {got:.4f} != {want:.4f}"
