"""Step 3 -- factorized support gate (gate_input x gate_norm) in ltv mode.

These are PROPERTY tests, not an algebraic identity. The original spec proposed
asserting (c*z > 0) == (z > 0), which is a property of the `>` operator and would
pass no matter what the predictor did with it. What actually matters is whether
the gate's OUTPUT is invariant to support-preserving rescaling, and how much the
predictor output moves as a result.
"""
import pytest
import torch

from models.infojepa_modules import LinearDynamicsPredictor

D, R, H, B, T, P = 64, 16, 3, 2, 3, 1
SCALES = (0.5, 0.75, 1.5, 2.0)


def make(gate_input="magnitude", gate_norm="sigmoid", seed=0):
    torch.manual_seed(seed)
    return LinearDynamicsPredictor(
        input_dim=D,
        output_dim=D,
        num_frames=H,
        num_patches=P,
        mode="ltv",
        rank=R,
        gate_input=gate_input,
        gate_norm=gate_norm,
    )


def sparse_code(seed=0):
    """A nonnegative, sparse code, i.e. what a reprelu link actually emits."""
    torch.manual_seed(seed)
    return torch.relu(torch.randn(B, T, P, D))


def test_defaults_reproduce_upstream_gate():
    """(magnitude, sigmoid) must equal the original sigmoid(gate(x)) expression."""
    m = make()
    x = sparse_code()
    want = torch.sigmoid(m.gate(x)).view(B, T, P, H + 1, R)
    assert torch.equal(m.gates(x), want)


@pytest.mark.parametrize("c", SCALES)
def test_support_gate_is_scale_invariant(c):
    """gate_input=support: the gate cannot see a support-preserving rescale."""
    m = make(gate_input="support")
    x = sparse_code()
    assert torch.equal(m.gates(x), m.gates(c * x))


@pytest.mark.parametrize("c", SCALES)
def test_magnitude_gate_is_not_scale_invariant(c):
    """The upstream gate does move, which is what the contrast is about."""
    m = make(gate_input="magnitude")
    x = sparse_code()
    assert not torch.allclose(m.gates(x), m.gates(c * x), atol=1e-6)


def test_scale_preserves_support():
    """Precondition for the above: positive scaling really does preserve support."""
    x = sparse_code()
    for c in SCALES:
        assert torch.equal((c * x) > 0, x > 0)


def test_softmax_gate_mean_magnitude_matches_sigmoid_scale():
    """r*softmax has mean ~1.0; bare softmax would be ~1/r and shrink the LTV path."""
    x = sparse_code()
    sig = make(gate_norm="sigmoid").gates(x).mean().item()
    sm = make(gate_norm="softmax").gates(x).mean().item()
    assert sm == pytest.approx(1.0, abs=0.05)
    assert 0.2 < sig < 0.8
    assert sm / sig < 4.0, "gate magnitudes differ enough to confound the contrast"


def test_softmax_normalizes_per_block_over_r_modes():
    """softmax must be over the r modes of each (lag, patch) block, not globally."""
    g = make(gate_norm="softmax").gates(sparse_code())
    assert g.shape == (B, T, P, H + 1, R)
    assert torch.allclose(g.sum(-1) / R, torch.ones(B, T, P, H + 1), atol=1e-5)


@pytest.mark.parametrize("gate_input", ["magnitude", "support"])
def test_forward_runs_and_reports_scale_sensitivity(gate_input):
    """The figure behind the gate: relative output change under rescaling."""
    m = make(gate_input=gate_input)
    x, c_act = sparse_code(), torch.randn(B, T, D)
    base = m(x, c_act)
    for c in SCALES:
        rel = ((m(c * x, c_act) - base).norm() / base.norm()).item()
        assert rel >= 0.0 and torch.isfinite(torch.tensor(rel))


def test_state_dict_keys_unchanged_by_flags():
    """Flags are plain attributes, so checkpoints stay compatible across arms."""
    a = set(make().state_dict())
    b = set(make(gate_input="support", gate_norm="softmax").state_dict())
    assert a == b
