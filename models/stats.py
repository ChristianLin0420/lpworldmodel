"""Statistics shared by the TRAINING LOSS and the LOGGED DIAGNOSTIC.

One definition, two consumers. `train.py` imports `soft_jaccard` from here for the
live `jacc/*` panel, and `models/visual_world_model.py` imports the same function
for R6's SUPPORT term. That is not tidiness: R6 exists because
`analysis/screen_objective.py` ranked `jacc/S_model` (raw -0.769, partial -0.549,
monotone over healthy predictors) as the one logged quantity that survives the
screen. If the optimised quantity and the screened quantity were two functions
that happened to agree today, the arm could not be read against the screen that
motivated it. `analysis/predictive_jaccard.py` holds the numpy mirror used for the
offline evaluation and is tested against this one
(tests/test_predictive_jaccard.py, tests/test_live_diagnostics.py).

Torch only, no project imports: `models/` may not import `train.py` (train.py
imports models), so a leaf module is the only place both sides can reach.
"""
import torch


def soft_jaccard(a, b, eps=1e-8):
    """J_S(a,b) = sum(min(a,b)) / sum(max(a,b)) over the last axis. Needs a,b >= 0.

    The project's core statistic: S = 1 - J_S. Mirrors
    analysis/predictive_jaccard.py's numpy version so the live curve and the
    offline evaluation measure the same thing.
    """
    num = torch.minimum(a, b).sum(-1)
    den = torch.maximum(a, b).sum(-1).clamp_min(eps)
    return num / den
