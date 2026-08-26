"""The LpWM world model (lpwm_swm).

An action-conditioned JEPA: a (frozen or finetuned) ViT encoder produces a CLS
code per frame, a projector applies the sparse RepReLU link, and an AdaLN-zero
transformer predictor rolls the code forward under actions. Rebuilt on the
public stable-worldmodel `Dynamics` contract (encode / predict / rollout); the
planning cost lives in `stable_worldmodel.planning.ShootingCostEvaluator` +
`GoalMSE` (see lpwm_swm/eval.py), not on the model.

Ported from the LpWM fork (wm/lpwm/lpwm.py); the only changes are: `history_size`
is a constructor arg (not a rollout positional), and the old model-owned
`get_cost`/`criterion` are dropped in favor of the new pluggable objective.
"""

import torch
from einops import rearrange
from torch import nn


class LpWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        history_size: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.history_size = history_size

    def encode(self, info, **kwargs):
        """Encode observations (and actions if present) into embeddings.

        info: dict with 'pixels' (B, T, C, H, W) and optionally 'action'.
        Writes info['emb'] = (B, T, D) (CLS token -> projector/link) and, when
        'action' is present, info['act_emb']. Extra kwargs are accepted and
        ignored for compatibility with the planning goal-encode dispatch.
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next-state embeddings.
        emb: (B, T, D); act_emb: (B, T, A_emb) -> (B, T, D).
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds


    def rollout(self, info, action_sequence):
        """Roll the world model forward given an initial info dict and actions.

        pixels: (B, S, T_ctx, C, H, W); action_sequence: (B, S, T, action_dim).
        S = # action-plan samples, T = horizon. Writes info['predicted_emb']
        (B, S, T_ctx + n_steps + 1, D). History length is self.history_size.
        """
        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        # flatten batch and sample dims for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )  # (BS, T, A_emb)

        HS = self.history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of (BS, D)
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + n_steps + 1, D)

        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info


__all__ = ['LpWM']
