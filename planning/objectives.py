import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss = loss_visual
        if "proprio" in z_obs_pred and "proprio" in z_obs_tgt:
            loss_proprio = metric(
                z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]
            ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
            loss = loss + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss = loss_visual
        if "proprio" in z_obs_pred and "proprio" in z_obs_tgt:
            loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(2, z_obs_pred["proprio"].ndim))
            )
            loss_proprio = (loss_proprio * coeffs).mean(dim=1)
            loss = loss + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError


def create_vote_objective_fn(n_members, rule="mean", alpha=0, base=2, mode="last"):
    """Consensus objective for `planning.ensemble.EnsembleWorldModel`.

    The ensemble stacks the M members' latents on the patch axis, so the input is
    (B, T, M, D_max) and every member's opinion about a CEM candidate is separable.
    `rule` picks how the M opinions are combined into the single per-candidate score
    CEM sorts on:

      "mean"   : mean_m MSE_m  -- equal-weight vote. Only sound when members share a
                 representation scale (same link + same RDMReg target_p); a dense
                 identity-link member and a sparse reprelu member do NOT.
      "borda"  : mean_m rank_m -- rank of the candidate within this CEM batch under
                 member m. Scale-free, so it is the rule to use for heterogeneous
                 columns (mixed link / mixed D).
      "median" : median_m rank_m -- majority consensus. With M=3 a single outlier
                 column (a dead seed) cannot move the winner, which is the property
                 `mean` lacks.

    NB rank rules return ranks, not MSEs: CEM only ever does `argsort(loss)[:topk]`
    (planning/cem.py) so this is exact, but the value logged as `plan_*/loss` is then a
    rank, and GDPlanner (which differentiates the objective) must not be used with them.
    `alpha`/`base`/`mode` exist only so this factory is drop-in for the plan configs;
    the adaln path has no proprio latent and the objective is computed on the last
    predicted frame.
    """
    assert rule in ("mean", "borda", "median"), f"vote rule {rule} not supported"
    M = int(n_members)

    def per_member_loss(z_obs_pred, z_obs_tgt):
        pv = z_obs_pred["visual"][:, -1:]          # (B, 1, M, D)
        tv = z_obs_tgt["visual"]                   # (B, 1, M, D)
        assert pv.shape[-2] == M, f"expected {M} stacked members, got {pv.shape[-2]}"
        return ((pv - tv) ** 2).mean(dim=(1, 3))   # (B, M)

    def objective_fn_vote(z_obs_pred, z_obs_tgt):
        per = per_member_loss(z_obs_pred, z_obs_tgt)      # (B, M)
        if rule == "mean":
            return per.mean(dim=1)
        # within-batch rank per member: argsort of argsort along the candidate axis
        ranks = per.argsort(dim=0).argsort(dim=0).to(per.dtype)
        return ranks.mean(dim=1) if rule == "borda" else ranks.median(dim=1).values

    return objective_fn_vote
