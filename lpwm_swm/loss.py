"""LpWM training losses (lpwm_swm).

Ported from the LpWM stable-worldmodel fork (wm/loss.py). Pure PyTorch + mpmath.

  - RDMReg: the core regularizer. Sliced-Wasserstein distance
    matching the per-timestep latent marginals to a Rectified Generalized
    Gaussian target (link=ReLU, shape p, location mu -> non-negative, sparse).
  - TemporalJaccardLoss: soft support-stability between consecutive frames
    (an optional add-on; used by the OGBench-Cube runs, off for Piecewise).
  - rectified_gengaus_mean_var_unified: closed-form moments of ReLU(GN_p),
    used to RMS-normalize the target when target_dist_rms_norm=True.
"""

import math

import mpmath as mp
import torch


def rectified_gengaus_mean_var_unified(p, mu, sigma):
    """Mean / variance / second moment for Y = ReLU(X),
    X ~ GN_p(mu, sigma) with density proportional to
        exp(-|x-mu|^p / (p*sigma^p)).

    Returns: (EY, VarY, EY2).
    """
    p = mp.mpf(p)
    mu = mp.mpf(mu)
    sigma = mp.mpf(sigma)

    if sigma <= 0:
        raise ValueError('sigma must be > 0')
    if p <= 0:
        raise ValueError('p must be > 0')

    sgn = mp.sign(mu)  # -1, 0, +1
    s1 = mp.mpf(1) / p
    s2 = mp.mpf(2) / p
    s3 = mp.mpf(3) / p

    t = (abs(mu) ** p) / (p * (sigma**p))
    G1 = mp.gamma(s1)

    lower1 = mp.gammainc(s1, 0, t)  # γ(1/p, t)
    lower3 = mp.gammainc(s3, 0, t)  # γ(3/p, t)
    upper2 = mp.gammainc(s2, t, mp.inf)  # Γ(2/p, t)

    A = (G1 + sgn * lower1) / G1
    B = upper2 / G1
    C = (mp.gamma(s3) + sgn * lower3) / G1

    p1 = p ** (mp.mpf(1) / p)
    p2 = p ** (mp.mpf(2) / p)

    EY = mp.mpf('0.5') * (mu * A + p1 * sigma * B)
    EY2 = mp.mpf('0.5') * (
        mu**2 * A + 2 * mu * p1 * sigma * B + p2 * sigma**2 * C
    )
    VarY = EY2 - EY**2

    return float(EY), float(VarY), float(EY2)


class TemporalJaccardLoss(torch.nn.Module):
    """Soft temporal support-instability loss (non-negative activations).

    loss = mean_{b,t}(1 - J_soft(emb[b,t], emb[b,t+1])), with
        J_soft = sum_d min(a_d, b_d) / (sum_d max(a_d, b_d) + eps).
    Minimizing it (positive weight) keeps the encoder support stable between
    consecutive frames. Requires a ReLU/RepReLU link (non-negative features).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (B, T, D) non-negative encoder outputs -> scalar."""
        a = emb[:, :-1, :]  # (B, T-1, D)
        b = emb[:, 1:, :]  # (B, T-1, D)

        intersection = torch.minimum(a, b).sum(dim=-1)  # (B, T-1)
        union = torch.maximum(a, b).sum(dim=-1)  # (B, T-1)
        jaccard = intersection / (union + self.eps)  # (B, T-1) in [0, 1]

        return (1.0 - jaccard).mean()


class RDMReg(torch.nn.Module):
    """RDMReg: sliced-Wasserstein distribution-matching regularizer (mono-GPU).

    Matches the marginal of the (linked) features against a Rectified
    Generalized Gaussian target via W2 over many random 1-D projections.
    ``link_function_type='ReLU'`` + ``p=1`` gives a rectified product-Laplace
    (sparse) target; ``'Identity'`` + ``p=2`` recovers the dense isotropic
    Gaussian.
    """

    def __init__(
        self,
        p=1.0,
        mu=0.0,
        link_function_type='ReLU',
        loss_func_type='sliced_wasserstein',
        num_slices=512,
        matching_mode='b_t_d',
        target_dist_rms_norm=False,
    ):
        super().__init__()
        self.p = p
        self.mu = mu

        self.sigma = math.sqrt(math.gamma(1 / p) / math.gamma(3 / p)) / (
            p ** (1 / p)
        )
        self.link_function_type = link_function_type
        self.matching_mode = matching_mode
        self.target_dist_rms_norm = target_dist_rms_norm

        if self.link_function_type == 'ReLU':
            self.link_function = torch.nn.ReLU()

            if self.target_dist_rms_norm:
                _, _, self.target_dist_second_moment = (
                    rectified_gengaus_mean_var_unified(
                        p=self.p, mu=self.mu, sigma=self.sigma
                    )
                )

        elif self.link_function_type == 'Identity':
            self.link_function = torch.nn.Identity()
            assert not self.target_dist_rms_norm, (
                'target_dist_rms_norm is not supported for Identity link'
            )
        else:
            raise ValueError(
                f'Unsupported link function type: {self.link_function_type}'
            )

        print(f'Using link function: {self.link_function_type}')

        self.loss_func_type = loss_func_type
        self.num_slices = num_slices

    def sample_product_laplace(
        self, shape, device, loc=0.0, scale=1 / math.sqrt(2)
    ):
        """Sample a product Laplace directly on ``device``."""
        loc_t = torch.tensor(loc, device=device)
        scale_t = torch.tensor(scale, device=device)
        laplace_dist = torch.distributions.Laplace(loc=loc_t, scale=scale_t)
        return laplace_dist.sample(shape)

    def sample_rgg(self, shape, p=1.0, mu=0.0, device='cpu', eps=1e-12):
        """Sample from Rectified Generalized Gaussian: ReLU(mu + sigma * GN_p).
        p=1.0 is Rectified Product Laplace (sparsity prior).
        """
        assert p > 0.0, 'p must be > 0.0'

        if p == 1.0:
            target_samples = self.link_function(
                self.sample_product_laplace(
                    shape, device, loc=mu, scale=self.sigma
                )
            )
        elif p == 2.0:
            target_samples = self.link_function(
                mu + self.sigma * torch.randn(shape, device=device)
            )
        else:
            sign = torch.empty(shape, device=device).bernoulli_(0.5) * 2 - 1
            gamma_dist = torch.distributions.Gamma(
                concentration=1.0 / p, rate=1.0
            )
            g = gamma_dist.sample(shape).to(device)
            gn_samples = sign * (p * g).pow(1.0 / p)
            target_samples = self.link_function(mu + self.sigma * gn_samples)

        # Decouple sparsity from scale.
        if self.target_dist_rms_norm:
            target_samples = target_samples / math.sqrt(
                self.target_dist_second_moment + eps
            )
        return target_samples

    def rdmreg_loss(
        self,
        z,
        p=1.0,
        mu=0.0,
        num_projections=8192,
    ):
        D = z.shape[-1]
        device = z.device
        target_samples = self.sample_rgg(tuple(z.shape), p, mu, device)

        # random unit projections [num_projections, D]
        projections = torch.randn(num_projections, D, device=device)
        projections = projections / projections.norm(dim=1, keepdim=True)

        proj_z = torch.matmul(z, projections.T)
        proj_target = torch.matmul(target_samples, projections.T)

        proj_z_sorted, _ = torch.sort(proj_z, dim=0)
        proj_target_sorted, _ = torch.sort(proj_target, dim=0)

        return torch.mean((proj_z_sorted - proj_target_sorted) ** 2)

    def forward(self, x):
        """x: [B*T, D] or [B, T, D] (samples over dim 0)."""
        return self.rdmreg_loss(
            x,
            p=self.p,
            mu=self.mu,
            num_projections=self.num_slices,
        )
