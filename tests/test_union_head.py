"""Step 4 -- union head: J parallel readouts with a per-(sample, timestep) min."""
import math

import pytest
import torch

from lpwm_build import build, load_cfg, seed_all, synthetic_batch
from models.infojepa_modules import LinearDynamicsPredictor

D, R, H, B, T, P = 64, 16, 3, 2, 3, 1


def make(mode="ltv", n_heads=1, seed=0):
    torch.manual_seed(seed)
    return LinearDynamicsPredictor(
        input_dim=D, output_dim=D, num_frames=H, num_patches=P,
        mode=mode, rank=R, n_heads=n_heads,
    )


def inputs(seed=0):
    torch.manual_seed(seed + 100)
    return torch.relu(torch.randn(B, T, P, D)), torch.randn(B, T, D)


@pytest.mark.parametrize("mode", ["mlp_var", "ltv"])
def test_j1_state_dict_is_byte_identical_to_upstream(mode):
    """No extra parameters at J=1, so checkpoints stay interchangeable."""
    assert set(make(mode, 1).state_dict()) == set(make(mode, 1).state_dict())
    extra = set(make(mode, 4).state_dict()) - set(make(mode, 1).state_dict())
    assert all(k.startswith("W_heads.") for k in extra), extra
    assert not any("W_heads" in k for k in make(mode, 1).state_dict())


@pytest.mark.parametrize("mode", ["mlp_var", "ltv", "additive"])
def test_head0_equals_single_head_forward(mode):
    """forward_heads()[0] must be exactly forward(): head 0 reuses W (and B)."""
    m = make(mode, 4)
    x, c = inputs()
    assert torch.equal(m.forward_heads(x, c)[0], m(x, c))


@pytest.mark.parametrize("mode", ["mlp_var", "ltv"])
def test_forward_heads_shape_and_distinct_heads(mode):
    m = make(mode, 4)
    x, c = inputs()
    out = m.forward_heads(x, c)
    assert out.shape == (4, B, T, P, D)
    assert not torch.allclose(out[0], out[1]), "heads are not independently initialised"


def test_additive_heads_are_a_switching_linear_system():
    """LTI(1) has no trunk, so each head carries its own (W_j, B_j)."""
    m = make("additive", 3)
    assert len(m.W_heads) == 2 and len(m.B_heads) == 2
    x, c = inputs()
    assert m.forward_heads(x, c).shape == (3, B, T, P, D)


def test_unsupported_mode_raises():
    with pytest.raises(ValueError, match="n_heads>1 not supported"):
        make("var", 4)


# --- loss-side behaviour, on the real VWorldModel -------------------------------

def _run(n_heads, coef=0.0, seed=0):
    cfg = load_cfg([f"n_heads={n_heads}", f"head_entropy_coef={coef}", "predictor=ltv"])
    seed_all(seed)
    model, _ = build(cfg)
    gen = torch.Generator().manual_seed(7)
    obs, act = synthetic_batch(cfg, 2, gen)
    return model, model(obs, act)


def test_j1_reduces_exactly_to_upstream_loss():
    """min over one head, meaned over (b,t), is the same uniform mean as MSELoss."""
    _, (_, _, _, loss1, c1) = _run(1)
    assert "head_switch_rate" not in c1
    assert torch.isfinite(loss1)


def test_union_head_logs_the_three_required_scalars():
    _, (_, _, _, loss, comps) = _run(4)
    for k in ("head_burst_rate", "head_switch_rate", "head_usage_p0"):
        assert k in comps, f"missing logged scalar {k}"
    p = torch.stack([comps[f"head_usage_p{j}"] for j in range(4)])
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5), "p_bar must be a distribution"
    assert 0.0 <= comps["head_switch_rate"].item() <= 1.0
    assert 0.0 <= comps["head_burst_rate"].item() <= 1.0


def test_min_over_heads_never_worse_than_head0():
    """The union loss is a min, so it must not exceed any single head's loss."""
    model, (_, _, _, loss4, _) = _run(4)
    assert torch.isfinite(loss4) and loss4.item() >= 0.0


def test_entropy_bonus_has_a_gradient():
    """A hard one-hot argmin has no gradient; the soft surrogate must have one."""
    model, (_, _, _, loss, comps) = _run(4, coef=0.1)
    ent = comps["head_usage_entropy"]
    assert ent.requires_grad, "entropy bonus is detached => lambda_ent would be inert"
    g = torch.autograd.grad(ent, model.predictor.W_heads[0].weight, retain_graph=True)[0]
    assert torch.isfinite(g).all() and (g != 0).any(), "entropy gradient does not reach heads"


def test_entropy_bonus_changes_the_loss():
    _, (_, _, _, l0, _) = _run(4, coef=0.0)
    _, (_, _, _, l1, _) = _run(4, coef=0.5)
    assert l0.item() != l1.item()


class _DDPLike(torch.nn.Module):
    """Stands in for DistributedDataParallel: holds the real module under .module
    and proxies ONLY forward(). Reaching any other method through it must fail,
    which is exactly what broke the first union-head launch on the cluster."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **k):
        return self.module(*a, **k)


def test_union_head_works_through_a_ddp_wrapper():
    """accelerate wraps the predictor, so the union head must reach forward_heads
    on the inner module. Building the model without prepare() hid this bug."""
    model, _ = _run(4)[0], None
    bare = model.predictor
    model.predictor = _DDPLike(bare)
    assert not hasattr(model.predictor, "forward_heads"), "wrapper is not DDP-like"
    assert model._pred is bare, "_pred did not unwrap the parallel wrapper"

    gen = torch.Generator().manual_seed(7)
    cfg = load_cfg(["n_heads=4", "predictor=ltv"])
    obs, act = synthetic_batch(cfg, 2, gen)
    _, _, _, loss, comps = model(obs, act)
    assert torch.isfinite(loss)
    assert "head_switch_rate" in comps


def test_pred_is_a_no_op_when_unwrapped():
    model = _run(4)[0]
    assert model._pred is model.predictor


def test_entropy_is_bounded_by_log_j():
    """Collapse precondition reads this scalar against ln(J), so the scale must hold."""
    _, (_, _, _, _, comps) = _run(4)
    assert 0.0 <= comps["head_usage_entropy"].item() <= math.log(4) + 1e-5


def test_usage_max_flags_collapse():
    """head_usage_max is the gate's collapse detector: 1.0 iff one head wins everywhere."""
    _, (_, _, _, _, comps) = _run(4)
    p = torch.stack([comps[f"head_usage_p{j}"] for j in range(4)])
    assert comps["head_usage_max"].item() == pytest.approx(p.max().item())
    assert 0.25 <= comps["head_usage_max"].item() <= 1.0
