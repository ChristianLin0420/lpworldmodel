"""Sparsity / collapse diagnostics logged during training (feats: (B, D))."""

import torch


def avg_per_dim_var(feats: torch.Tensor) -> torch.Tensor:
    """Average per-dimension variance. feats: (B, D)."""
    with torch.no_grad():
        assert feats.dim() == 2, (
            'Input tensor must be 2D (batch_size, feature_dim)'
        )
        return feats.var(dim=0).mean()


def off_diag_cov_sum(feats: torch.Tensor) -> torch.Tensor:
    """Sum of squared off-diagonal covariance entries. Near 0 => decorrelated."""
    with torch.no_grad():
        assert feats.dim() == 2, (
            'Input tensor must be 2D (batch_size, feature_dim)'
        )
        B = feats.size(0)
        if B <= 1:
            raise ValueError(
                'Batch size must be greater than 1 to compute covariance'
            )
        centered = feats - feats.mean(dim=0)
        cov = centered.T @ centered / (B - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum()


def l1_sparsity_metric(feats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean l1 sparsity metric: (1/D) * (||z||_1 / ||z||_2)^2. feats: (B, D)."""
    with torch.no_grad():
        assert feats.dim() == 2, (
            'Input tensor must be 2D (batch_size, feature_dim)'
        )
        D = feats.shape[1]
        l1 = torch.linalg.norm(feats, ord=1, dim=1)
        l2 = torch.linalg.norm(feats, ord=2, dim=1)
        return ((1.0 / D) * (l1 / (l2 + eps)) ** 2).mean()


def l0_sparsity_metric(feats: torch.Tensor) -> torch.Tensor:
    """Mean l0 sparsity metric: fraction of nonzero entries per sample."""
    with torch.no_grad():
        assert feats.dim() == 2, (
            'Input tensor must be 2D (batch_size, feature_dim)'
        )
        D = feats.shape[1]
        return (feats != 0).sum(dim=1).float().div(D).mean()


__all__ = [
    'avg_per_dim_var',
    'off_diag_cov_sum',
    'l1_sparsity_metric',
    'l0_sparsity_metric',
]
