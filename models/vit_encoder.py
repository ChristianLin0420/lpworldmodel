"""Self-contained ViT encoder trained from scratch (LeWM-style).

Conforms to the InfoJEPA encoder contract used by `train.py` / `VWorldModel`:
  - attributes: `.name`, `.emb_dim`, `.latent_ndim`, `.patch_size`
  - forward(x: (N, 3, H, W)) -> (N, num_patches, emb_dim)
    where num_patches == 1 when `feature == "cls"`.

Built from the shared blocks in `models.infojepa_modules` (no timm / HF dependency,
since the cluster image ships transformers==4.21.1 and no timm). A projector MLP
(LayerNorm head) is folded into the encoder so its output is what gets
predicted and regularized.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange

from models.infojepa_modules import MLP, Transformer


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=14,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        feature="cls",  # "cls" or "patch"
        proj_hidden=2048,
        proj_dim=None,  # projector output dim; defaults to `dim`
        name="vit_scratch",
    ):
        super().__init__()
        assert feature in {"cls", "patch"}, f"feature {feature} not supported"
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.name = name
        self.patch_size = patch_size
        self.feature = feature
        self.dim = dim

        grid = image_size // patch_size
        self.grid_size = grid
        num_patches = grid * grid

        proj_dim = proj_dim or dim
        self.emb_dim = proj_dim
        self.latent_ndim = 1 if feature == "cls" else 2
        self.num_patches = 1 if feature == "cls" else num_patches

        # patchify
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=False,
        )

        self.projector = MLP(
            input_dim=dim,
            hidden_dim=proj_hidden,
            output_dim=proj_dim,
            norm_fn=nn.LayerNorm,
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate the [patch] part of the pos-embedding to an (h, w) grid."""
        npatch = x.shape[1] - 1
        n = self.pos_embedding.shape[1] - 1
        if npatch == n:
            return self.pos_embedding
        cls_pos = self.pos_embedding[:, :1]
        patch_pos = self.pos_embedding[:, 1:]
        dim = x.shape[-1]
        patch_pos = patch_pos.reshape(
            1, int(math.sqrt(n)), int(math.sqrt(n)), dim
        ).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos, size=(h, w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        """
        x: (N, 3, H, W)
        returns: (N, num_patches, emb_dim)  (num_patches == 1 for CLS)
        """
        b = x.shape[0]
        x = self.patch_embed(x)  # (N, dim, h, w)
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, "n d h w -> n (h w) d")

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (N, 1 + h*w, dim)
        x = x + self.interpolate_pos_encoding(x, h, w)
        x = self.dropout(x)

        x = self.transformer(x)  # (N, 1 + h*w, dim)

        if self.feature == "cls":
            tokens = x[:, :1]  # (N, 1, dim)
        else:
            tokens = x[:, 1:]  # (N, h*w, dim)

        # projector applied per token
        p = tokens.shape[1]
        tokens = self.projector(rearrange(tokens, "n p d -> (n p) d"))
        tokens = rearrange(tokens, "(n p) d -> n p d", p=p)
        return tokens
