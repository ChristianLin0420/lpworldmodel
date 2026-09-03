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


def create_vote_objective_fn(
    n_members, rule="mean", alpha=0, base=2, mode="last", lam=0.0
):
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
      "cvar"   : mean_m rank_m + lam * std_m rank_m -- mean rank penalised by member
                 DISAGREEMENT. `lam=0` (the default) is exactly "borda", so the key is
                 inert unless set; raising lam walks continuously from the pure
                 variance-reduction rule toward pessimism.
      "max"    : max_m rank_m -- minimax. The candidate is scored by its WORST member,
                 the far end of the same pessimism axis.

    `borda`/`median`/`cvar`/`max` are all order statistics of the SAME rank matrix, so a
    sweep over them holds members, episodes, budget and rollouts fixed and isolates the
    combination rule. That is the point: median-of-ranks (the only rule ever run, and the
    campaign's one positive) is simultaneously variance-reduced AND robust/pessimistic,
    so `borda` (variance reduction only) vs `cvar`/`max` (pessimism) identifies which.

    `lam` is only read by `cvar`. The std is UNBIASED=False deliberately: `PiWMvoteM1` is
    a live M=1 arm and the unbiased std of a single sample is nan, which would poison
    every candidate score; the biased std is 0 there, so cvar degenerates to borda.

    NB rank rules return ranks, not MSEs: CEM only ever does `argsort(loss)[:topk]`
    (planning/cem.py) so this is exact, but the value logged as `plan_*/loss` is then a
    rank, and GDPlanner (which differentiates the objective) must not be used with them.
    `alpha`/`base`/`mode` exist only so this factory is drop-in for the plan configs;
    the adaln path has no proprio latent and the objective is computed on the last
    predicted frame.
    """
    assert rule in (
        "mean",
        "borda",
        "median",
        "cvar",
        "max",
    ), f"vote rule {rule} not supported"
    lam = float(lam)
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
        ranks = per.argsort(dim=0).argsort(dim=0).to(per.dtype)   # (B, M)
        if rule == "borda":
            return ranks.mean(dim=1)
        if rule == "median":
            return ranks.median(dim=1).values
        if rule == "max":                      # minimax: score = worst member's rank
            return ranks.max(dim=1).values
        # cvar: mean rank + lam * disagreement.  lam == 0 reproduces borda exactly.
        return ranks.mean(dim=1) + lam * ranks.std(dim=1, unbiased=False)

    return objective_fn_vote
