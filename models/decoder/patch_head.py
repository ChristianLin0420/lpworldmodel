"""T2 -- a per-patch reconstruction head.

Round 5, T2. `PiWM-columns` (256 patch tokens instead of 1 cls token) was a null:
paired CEM delta +0.072, 95% CI [-0.064, +0.207], n = 12. It added spatial CAPACITY
with nothing asking for spatial CONTENT. The predictor cannot supply that demand --
`LinearDynamicsPredictor` has no positional embedding and no per-patch parameters, so
one operator is broadcast identically over all 256 tokens and the loss it induces is
permutation-invariant in the patch axis.

`PatchHead` is the first parameter in this model that could distinguish token i from
token j: token i is decoded to the 14x14 pixel block at grid position i, and nothing
else. The weights are shared across positions (one nn.Linear), but the *assignment* of
outputs to image regions is positional, so a token that carries the wrong region's
content is penalised. That is what "per-patch" buys.

Deliberately tiny: 226,380 params against `TransposedConvDecoder`'s 31.3M, so a T2 win
cannot be attributed to decoder capacity.

`TransposedConvDecoder` is not usable at p=256: `horizontal_forward` would render
b*t*p = 65,536 separate 224x224 images per batch, and its `dist.mean.squeeze(2)` is a
no-op at p != 1, so the following `rearrange("b t c h w -> ...")` fails on a 6-D tensor.
"""
import torch
import torch.nn as nn
from einops import rearrange


class PatchHead(nn.Module):
    """z (b, t, p, d) with p == grid**2  ->  (img (b*t, c, H, W), diff scalar).

    The (img, diff) tuple is the `decode_obs` contract
    (models/visual_world_model.py:468-469); `diff` is the VQ commitment term that only
    VQVAE produces, so it is a zero of the right dtype/device here.
    """

    def __init__(self, emb_dim=384, patch_size=14, grid=16, out_chans=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.p, self.g, self.c = int(patch_size), int(grid), int(out_chans)
        self.head = nn.Linear(emb_dim, self.c * self.p * self.p)

    def forward(self, z):
        assert z.shape[2] == self.g * self.g, (
            f"PatchHead expects p == grid**2 = {self.g * self.g}, got {z.shape[2]}. "
            "This head is only meaningful with encoder=vit_scratch_patch."
        )
        x = rearrange(
            self.head(z),
            "b t (gh gw) (c ph pw) -> (b t) c (gh ph) (gw pw)",
            gh=self.g, gw=self.g, c=self.c, ph=self.p, pw=self.p,
        )
        return x, torch.zeros((), device=z.device, dtype=z.dtype)
