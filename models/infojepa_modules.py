"""Shared building blocks for InfoJEPA.

Ported from the LeWM codebase (lucas-maes/le-wm, `module.py`). LeWM is the
baseline that the InfoJEPA / RDMReg method extends, so these blocks live under
the project name rather than the baseline's:
  - SIGReg            : anti-collapse regularizer (isotropic-Gaussian goodness-of-fit)
  - FeedForward / Attention / Block / ConditionalBlock / Transformer : transformer stack
  - ARPredictor      : autoregressive predictor with AdaLN-zero action conditioning
  - Embedder         : action encoder (Conv1d smoothing + MLP)
  - MLP              : projector / pred_proj head
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Pushes the marginal distribution of the embeddings toward a standard
    isotropic Gaussian via an Epps-Pulley characteristic-function goodness-of-fit
    test on random 1D projections (Cramer-Wold). Parameterless (buffers only).
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        returns: scalar statistic averaged over projections and time
        """
        # sample random unit projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class FeedForward(nn.Module):
    """FeedForward network used in Transformers."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with optional causal masking."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, causal=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, attn_mask=None):
        """
        x : (B, L, D)
        attn_mask : optional (L, L) bool/float mask (True/0 = attend). When given,
            it overrides `is_causal` (e.g. the block-causal mask used for patches).
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=self.causal
            )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=False):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout, causal=causal
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        causal=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, causal=causal)
            )

    def forward(self, x, c=None, attn_mask=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            if isinstance(block, Block):
                x = block(x, attn_mask=attn_mask)
            else:
                x = block(x, c, attn_mask=attn_mask)
        x = self.norm(x)

        x = self.output_proj(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (N, D)
        """
        return self.net(x)


class Embedder(nn.Module):
    """Action/proprio encoder: Conv1d (k=1) smoothing + MLP.

    Follows the InfoJEPA encoder-construction convention (`in_chans`, `emb_dim`),
    so it is a drop-in for the Hydra `action_encoder` / `proprio_encoder` groups.
    Exposes `.emb_dim` (consumed by train.py).
    """

    def __init__(self, in_chans=10, emb_dim=192, smoothed_dim=None, mlp_scale=4):
        super().__init__()
        smoothed_dim = smoothed_dim or in_chans
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(in_chans, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, T, emb_dim)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction.

    Causal transformer with AdaLN-zero action conditioning. The LeWM `pred_proj`
    head is folded in as `output_proj` so it is owned (and optimized) by the
    predictor module.

    Works for any number of patch tokens per frame:
      - num_patches == 1 (CLS): one token per frame; plain temporal causality.
      - num_patches  > 1 (patch features): `num_patches` tokens per frame with a
        factorized (temporal + spatial) pos-embed and a block-causal mask
        (full attention within a frame and across past frames, causal over time).

    forward(x, c):
        x: (B, T, P, input_dim)   embeddings (P == num_patches)
        c: (B, T, act_dim==input_dim)   per-frame action embeddings (conditioning)
        returns: (B, T, P, output_dim)
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        num_patches=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pred_proj_hidden=2048,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        self.num_patches = num_patches
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, 1, input_dim) * 0.02)
        self.spatial_pos = (
            nn.Parameter(torch.randn(1, 1, num_patches, input_dim) * 0.02)
            if num_patches > 1
            else None
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            causal=True,
        )
        self.output_proj = MLP(
            input_dim=output_dim,
            hidden_dim=pred_proj_hidden,
            output_dim=output_dim,
            norm_fn=nn.LayerNorm,
        )

    @staticmethod
    def _block_causal_mask(T, P, device):
        """(T*P, T*P) bool mask, True = attend. Token (t,p) attends to (t',p')
        iff t' <= t (full attention within current + past frames). Frame-major
        flattening: token_index // P == frame_index."""
        frame_idx = torch.arange(T * P, device=device) // P
        return frame_idx[None, :] <= frame_idx[:, None]

    def forward(self, x, c):
        """
        x: (B, T, P, input_dim)
        c: (B, T, act_dim)
        """
        B, T, P, _ = x.shape
        x = x + self.temporal_pos[:, :T]
        if self.spatial_pos is not None:
            x = x + self.spatial_pos
        c = c.unsqueeze(2).expand(B, T, P, c.size(-1))  # broadcast action over patches

        x = rearrange(x, "b t p d -> b (t p) d")  # frame-major
        c = rearrange(c, "b t p a -> b (t p) a")
        x = self.dropout(x)

        attn_mask = self._block_causal_mask(T, P, x.device) if P > 1 else None
        x = self.transformer(x, c, attn_mask=attn_mask)

        x = self.output_proj(rearrange(x, "b l d -> (b l) d"))
        x = rearrange(x, "(b t p) d -> b t p d", b=B, t=T, p=P)
        return x


class LinearDynamicsPredictor(nn.Module):
    """(Action-conditioned) LINEAR dynamics predictor -- the linear-core rungs
    (LTI(1)/LTI(k)/MLP∘LTI/MLP∘LTV) of the predictor-complexity ladder. Used via the
    VWorldModel "adaln" path -- a misnomer that gates the from-scratch JEPA path (action
    passed separately, shared link on enc+pred outputs), NOT AdaLN conditioning: this
    predictor consumes the action linearly (P(a)z or Wz+Ba), not via AdaLN. Same forward(x, c)
    contract as ARPredictor (x=(B,T,P,D) codes, c=(B,T,D) action embeddings, returns
    (B,T,P,D)), and the map is LINEAR in the code. `action_linear`/`additive` use only the
    current frame z_t (no temporal attention); `var` is history-aware (last `num_frames`
    frames, a causal linear map) — the fair linear analog of the transformer's num_hist
    window, needed where one frame is not a Markov-sufficient state (e.g. pusht: velocity).
    output[:, t] predicts frame t+1. For adaln the model returns the PRE-link u; VWorldModel
    then applies the (fixed, non-learned) link, so the code evolves in the linked space.

    mode="action_linear" (excluded rung, config linear_pa): P(a) = diag(d(a)) + U(a) V(a)^T -- a
        per-action diagonal rescale plus a rank-r mixing, with d/U/V from small MLPs of the action.
    mode="additive"    (LTI(1), config linear_wb): z_{t+1} = W z_t + B a_t -- fully linear,
        single-frame, action additive.
    mode="var"         (LTI(k), config linear_var): z_{t+1} = sum_{k=0}^{H-1} A_k z_{t-k} + B a_t,
        H=num_frames (=num_hist). A first-order linear system on the STACKED history [z_t..z_{t-H+1}]
        (state-augmentation) -- can capture 2nd-order dynamics (velocity via z_t,z_{t-1}).
        CAUSAL: output[:,t] uses only frames <= t; missing lags (t<k, incl. rollout
        cold-start where T<H) are dropped (= zero, exact since A_k*0=0), mirroring the
        transformer's min(num_hist, L) growing context. H=1 reduces exactly to `additive`.
    mode="mlp_var"  (MLP∘LTI(k), config mlp_var): z_{t+1} = W ReLU(sum_{k} A_k z_{t-k} + B a_t) -- a
        1-hidden-layer MLP on the LTI(k) features. The nonlinear CORE is shared by dense & sparse
        (only the output link differs), so the dense/sparse comparison is complexity-matched
        (unlike `var`, where sparse gets a "free" RepReLU nonlinearity dense lacks). Sits between
        LTI(k) (linear) and Shallow-AdaLN (attention) on the complexity ladder.
    mode="ltv" (MLP∘LTV(k), config ltv): z_{t+1} = W ReLU(sum_k A_k(z_t) z_{t-k} + B(z_t) a_t)
        with A_k(z_t)/B(z_t) = fixed base + STATE-GATED low-rank correction (the input
        never emits a matrix, only a small gate g(z_t)=sigmoid(gate·z_t)∈R^r selects among fixed
        low-rank directions). rank r = # state-selected modes; U-factors 0-inited so it starts == mlp_var.
        Shared nonlinear core across dense/sparse (only the link differs). Between MLP∘LTI and Shallow-AdaLN.

    Initialised near the identity map (d(a)=1, low-rank=0 / W=I / lag0=I & older-lags=0, B=0)
    so the untrained predictor is z_{t+1}=z_t (mirrors AdaLN-zero); the invariance loss shapes it.
    EXCEPTION: `mlp_var` uses NON-identity muP fan_in-scaled init (the inner ReLU precludes a clean
    identity map on a signed code; init variance ~ 1/fan_in per the muP recipe).
    """

    def __init__(
        self,
        *,
        input_dim,
        output_dim=None,
        num_frames=1,
        num_patches=1,
        hidden_dim=None,
        mode="action_linear",
        rank=16,
        act_hidden=256,
        gate_input="magnitude",
        gate_norm="sigmoid",
        n_heads=1,
        lag_mask_p=0.0,
        lag_dilation=None,
        lie_sim=False,
        act_gain=None,
        **kwargs,
    ):
        super().__init__()
        D = input_dim
        out = output_dim or input_dim
        assert out == D, f"linear predictor assumes output_dim==input_dim (code space), got {out} vs {D}"
        assert mode in {"action_linear", "additive", "var", "mlp_var", "ltv", "lie", "ssm"}, \
            f"mode {mode} not supported"
        assert gate_input in {"magnitude", "support", "both"}, f"gate_input {gate_input} not supported"
        assert gate_norm in {"sigmoid", "softmax"}, f"gate_norm {gate_norm} not supported"
        self.mode = mode
        # ROUND 8 / T4. Masking and dilation over the LAG axis. Both default to the
        # upstream behaviour exactly: p=0 draws no mask, and dilation None means lag k
        # reads z_{t-k}, which is what every arm before round 8 did.
        self.lag_mask_p = float(lag_mask_p)
        # "1,2,5" -> lag slot k reads z_{t-d_k}. Same parameter count, wider temporal span:
        # the block is static in 48.1% of single steps, so tight uniform lags may spend two
        # thirds of the history on frames carrying no block motion.
        # Accept either a list ([1,2,5], hydra list syntax) or a string ("1,2,5", quoted on
        # the command line). Hydra treats a bare comma as a sweep, so the shell must quote it;
        # taking both forms means a config file and a CLI override behave the same.
        if not lag_dilation:
            self.lag_dilation = None
        elif isinstance(lag_dilation, str):
            self.lag_dilation = [int(v) for v in lag_dilation.split(",") if v.strip() != ""]
        else:
            self.lag_dilation = [int(v) for v in lag_dilation]
        self.rank = rank
        # Step 3, ltv only. Factorized deliberately: the original single flag changed
        # the gate's INPUT and its NORMALIZATION at the same time, so a result could
        # not be attributed to either. (magnitude, sigmoid) == upstream.
        self.gate_input = gate_input
        self.gate_norm = gate_norm
        # P3 falsifier: None => the action term is untouched and the whole path is
        # bit-identical to upstream. A float makes ||B a|| a fixed fraction beta of
        # ||sum_k A_k z||, i.e. it turns d_state/d_action into a hyperparameter.
        self.act_gain = None if act_gain is None else float(act_gain)
        if mode == "additive":  # LTI(1): z' = W z + B a
            self.W = nn.Linear(D, D, bias=True)
            self.B = nn.Linear(D, D, bias=False)   # action embedding has dim D (adaln)
            nn.init.eye_(self.W.weight)
            nn.init.zeros_(self.W.bias)
            nn.init.zeros_(self.B.weight)           # AdaLN-zero style: no action effect at init
        elif mode == "var":
            self.n_lags = num_frames
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)    # action embedding has dim D (adaln)
            nn.init.eye_(self.lags[0].weight); nn.init.zeros_(self.lags[0].bias)  # lag0 (z_t) = identity
            for k in range(1, num_frames):
                nn.init.zeros_(self.lags[k].weight)  # older frames ignored at init -> z' = z_t
            nn.init.zeros_(self.B.weight)            # no action effect at init
        elif mode == "ssm":
            # ROUND 8 / T1. The IIR generalisation of `var`. Every other mode here is FIR:
            # z_{t+1} = sum_{k=0}^{H-1} A_k z_{t-k}, a fixed H-tap window (H = num_frames = 3
            # in all 400 sampled configs across seven rounds). This carries a STATE instead:
            #     s_t = A s_{t-1} + Bz z_t + Ba a_t ,   core_t = C s_t
            # so the past enters with unbounded, learned decay rather than a hard cutoff.
            # `var`'s own docstring calls itself "state-augmentation"; this is that, without
            # the truncation.
            #
            # Init follows `additive`'s conventions exactly so the arm starts near-identity
            # and action-free: A near-identity (0.9 * I, a contraction so the state cannot
            # blow up before it has learned anything), Bz identity, Ba ZERO (AdaLN-zero: no
            # action effect at init), C identity.
            self.n_lags = num_frames        # unused by ssm; kept so shared code paths hold
            self.A = nn.Linear(D, D, bias=False)
            self.Bz = nn.Linear(D, D, bias=True)
            self.Ba = nn.Linear(D, D, bias=False)
            self.C = nn.Linear(D, D, bias=True)
            self.W = nn.Linear(D, D, bias=True)   # same ReLU->W readout as mlp_var / ltv
            nn.init.eye_(self.A.weight); self.A.weight.data.mul_(0.9)
            nn.init.eye_(self.Bz.weight); nn.init.zeros_(self.Bz.bias)
            nn.init.zeros_(self.Ba.weight)
            nn.init.eye_(self.C.weight); nn.init.zeros_(self.C.bias)
            nn.init.normal_(self.W.weight, mean=0.0, std=D ** -0.5)
            nn.init.zeros_(self.W.bias)
        elif mode == "mlp_var":
            self.n_lags = num_frames
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)     # action embedding has dim D (adaln)
            self.W = nn.Linear(D, D, bias=True)      # nonlinear readout: z' = W ReLU(var_core)
            for lin in (*self.lags, self.B, self.W):
                nn.init.normal_(lin.weight, mean=0.0, std=D ** -0.5)  # muP hidden init N(0, 1/fan_in)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        elif mode == "ltv":  # DATA-DEPENDENT VAR + nonlinear readout:
            r = rank
            self.n_lags = num_frames
            self.rank = r
            self.lags = nn.ModuleList([nn.Linear(D, D, bias=(k == 0)) for k in range(num_frames)])
            self.B = nn.Linear(D, D, bias=False)
            self.W = nn.Linear(D, D, bias=True)                                   # nonlinear readout
            self.Vlag = nn.ModuleList([nn.Linear(D, r, bias=False) for _ in range(num_frames)])  # down-proj
            self.Ulag = nn.ModuleList([nn.Linear(r, D, bias=False) for _ in range(num_frames)])  # up-proj (0-init)
            self.VB = nn.Linear(D, r, bias=False)
            self.UB = nn.Linear(r, D, bias=False)                                 # up-proj (0-init)
            # gate_input="both" feeds [z_t ; 1[z_t>0]], so the gate's fan-in is 2D.
            gate_in = 2 * D if gate_input == "both" else D
            self.gate = nn.Linear(gate_in, (num_frames + 1) * r)                  # per-lag + B gates, from z_t
            # NB the gate is initialised OUTSIDE the loop below. That loop hardcodes
            # std = D**-0.5, which is muP's fan_in rule only while fan_in == D; with
            # "both" the gate would start sqrt(2) too large and the arm would measure
            # an init change on top of the gate change.
            for m in (*self.lags, self.B, self.W, *self.Vlag, self.VB):
                nn.init.normal_(m.weight, mean=0.0, std=D ** -0.5)                # muP fan_in init
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.gate.weight, mean=0.0, std=gate_in ** -0.5)      # muP on the REAL fan_in
            nn.init.zeros_(self.gate.bias)
            for m in (*self.Ulag, self.UB):
                nn.init.zeros_(m.weight)                                          # LoRA-style: LTV off at init
        elif mode == "lie":
            # P1: the action acts on the code as a GROUP ELEMENT rather than as a bias.
            #     v_t     = sum_k s_k * R(phi_k) z_{t-k}      history combined linearly
            #     u_{t+1} = R(theta(a_t)) v_t                 the action IS the dynamics
            # R(.) is block-diagonal in the FIXED coordinate pairs (2j, 2j+1): orthogonal
            # by construction, composition exact (angles add), and O(D) per patch instead
            # of O(D^2). In an SDR the coordinates ARE the features, so a phase advance
            # per feature pair is grid-cell path integration -- TBT's reference frame in
            # the latent, with no ground-truth pose. NB no dense matrix_exp: that is O(D^3)
            # per forward and conf/predictor/ltv.yaml was built to avoid exactly that.
            assert D % 2 == 0, f"lie needs an even code width, got D={D}"
            self.n_lags = num_frames
            self.half = D // 2
            self.lie_sim = bool(lie_sim)
            # 1-D on purpose: mup_param_groups gives dim<2 params wd=0 (models/mup.py:124),
            # and weight-decaying an angle or a lag scale toward 0 is meaningless.
            self.phi = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.half)) for _ in range(num_frames)])
            self.lag_scale = nn.ParameterList(
                [nn.Parameter(torch.full((self.half,), 1.0 if k == 0 else 0.0))
                 for k in range(num_frames)])
            # an nn.Linear, NOT a bare Parameter: mup_fan_in_map only walks _LINEARISH
            # modules, so a raw Parameter would silently train at base_lr and mup_init_
            # would never touch it.
            self.W_theta = nn.Linear(D, self.half)
            nn.init.zeros_(self.W_theta.weight)
            nn.init.zeros_(self.W_theta.bias)          # theta = 0 => R = I at init
            # P1b: enlarge the GROUP to similitudes rather than bolting an MLP back on.
            # Composition stays exact -- angles add, scales multiply.
            self.log_scale = nn.Parameter(torch.zeros(self.half)) if self.lie_sim else None
        else:
            self.to_d = nn.Sequential(
                nn.Linear(D, act_hidden), nn.SiLU(), nn.Linear(act_hidden, D)
            )
            self.to_uv = nn.Sequential(
                nn.Linear(D, act_hidden), nn.SiLU(), nn.Linear(act_hidden, 2 * D * rank)
            )
            nn.init.zeros_(self.to_d[-1].weight); nn.init.ones_(self.to_d[-1].bias)
            nn.init.zeros_(self.to_uv[-1].weight); nn.init.zeros_(self.to_uv[-1].bias)

        # Step 4, union head. Head 0 keeps the existing `W` (and `B`) attribute names and
        # the extras live in a separate ModuleList, so at n_heads=1 nothing is added and
        # the state dict stays byte-identical to upstream.
        self.n_heads = n_heads
        if n_heads > 1:
            if mode in {"mlp_var", "ltv", "ssm"}:
                self.W_heads = nn.ModuleList(
                    [nn.Linear(D, D, bias=True) for _ in range(n_heads - 1)]
                )
                for lin in self.W_heads:
                    nn.init.normal_(lin.weight, mean=0.0, std=D ** -0.5)
                    nn.init.zeros_(lin.bias)
            elif mode == "additive":
                # LTI(1) is z' = W z + B a and has NO trunk -- W *is* the predictor. So
                # J heads here is J copies of (W_j, B_j), i.e. a switching linear system:
                # a different and strictly stronger model than "J readouts on a shared
                # trunk". Step 4 results on this rung must be read with that in mind.
                self.W_heads = nn.ModuleList(
                    [nn.Linear(D, D, bias=True) for _ in range(n_heads - 1)]
                )
                self.B_heads = nn.ModuleList(
                    [nn.Linear(D, D, bias=False) for _ in range(n_heads - 1)]
                )
                for lin in self.W_heads:
                    nn.init.eye_(lin.weight)
                    nn.init.zeros_(lin.bias)
                for lin in self.B_heads:
                    nn.init.zeros_(lin.weight)
            else:
                raise ValueError(f"n_heads>1 not supported for mode={mode}")

    @staticmethod
    def _rotate(x, ang, scale=None):
        """Block-diagonal 2x2 rotation of x (..., D) by ang (..., D/2).

        Exactly orthogonal, so R(a)R(b) = R(a+b) and R^T R = I hold to machine precision;
        `scale` (..., D/2) turns it into a similitude, where scales multiply.
        """
        xe, xo = x[..., 0::2], x[..., 1::2]
        cos, sin = torch.cos(ang), torch.sin(ang)
        ye = cos * xe - sin * xo
        yo = sin * xe + cos * xo
        if scale is not None:
            ye, yo = ye * scale, yo * scale
        return torch.stack((ye, yo), dim=-1).flatten(-2)

    def _action_term(self, state_part, Ba):
        """P3: optionally rescale the action branch to beta * ||state branch||.

        act_gain unset returns Ba untouched, so the default path stays bit-identical.
        """
        if self.act_gain is None:
            return Ba
        s = state_part.pow(2).mean(dim=-1, keepdim=True).sqrt()
        a = Ba.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return self.act_gain * Ba * (s / a)

    def _lag_offset(self, k):
        """Which past frame lag slot k reads. Identity (k) unless T4 dilation is set."""
        if self.lag_dilation is None or k >= len(self.lag_dilation):
            return k
        return self.lag_dilation[k]

    def _lag_keep(self, k, B, device, dtype):
        """(B,1,1,1) per-sample keep mask for lag k, or None when T4 masking is off.

        Independent Bernoulli over lags k>=1, so the model cannot rely on any one past
        frame being present. Lag 0 (z_t) is never masked -- dropping the current frame is
        not a temporal-redundancy test, it is an ablation of the input.

        Structurally free: _trunk already drops unavailable lags at cold start
        (`if k >= T: break`) and A_k * 0 = 0 exactly, so a zeroed lag is a state the model
        already sees. Eval is untouched (self.training guard), so nothing about planning
        or the rollout changes.
        """
        if not self.training or self.lag_mask_p <= 0 or k == 0:
            return None
        keep = (torch.rand(B, device=device) >= self.lag_mask_p).to(dtype)
        return keep.view(B, 1, 1, 1)

    def _trunk(self, x, c):
        """Pre-readout core for mlp_var / ltv / ssm: the value fed to ReLU then W."""
        B, T, P, D = x.shape
        if self.mode == "ssm":
            # ROUND 8 / T1. Sequential scan over T. T = num_hist = 3 at train time, so the
            # Python loop is cheap; it is the same length the FIR modes unroll anyway.
            # Causal by construction, and the cold start matches `var`'s convention: the
            # state begins at zero and A * 0 = 0 exactly, so no lag is silently invented.
            s_t = x.new_zeros(B, P, D)
            outs = []
            for t in range(T):
                s_t = self.A(s_t) + self.Bz(x[:, t]) + self.Ba(c[:, t]).unsqueeze(1)
                outs.append(self.C(s_t))
            return torch.stack(outs, dim=1)

        if self.mode == "mlp_var":
            u = self.lags[0](x)
            for k in range(1, self.n_lags):
                dk = self._lag_offset(k)
                if dk >= T:
                    break
                xk = self.lags[k](x[:, : T - dk])
                keep = self._lag_keep(k, B, x.device, x.dtype)
                if keep is not None:
                    xk = xk * keep
                u = u + torch.cat([x.new_zeros(B, dk, P, D), xk], dim=1)
            return u + self._action_term(u, self.B(c).unsqueeze(2))

        g = self.gates(x)                            # ltv: gates g(z_t) from this frame
        core = self.lags[0](x) + self.Ulag[0](g[..., 0, :] * self.Vlag[0](x))
        for k in range(1, self.n_lags):
            dk = self._lag_offset(k)                 # T4: which past frame slot k reads
            if dk >= T:
                break                                # z_{t-dk} unavailable (cold-start)
            xk = x[:, : T - dk]                      # z_{t-dk} feeds output positions [dk:]
            base_k = self.lags[k](xk)                # A_k z_{t-dk}
            # the gate is indexed by the SLOT k (there are n_lags+1 slots) but sliced by the
            # TIME offset dk, so dilation moves which frame is read without renumbering gates.
            corr_k = self.Ulag[k](g[:, dk:, :, k, :] * self.Vlag[k](xk))
            contrib = base_k + corr_k
            keep = self._lag_keep(k, B, x.device, x.dtype)
            if keep is not None:
                contrib = contrib * keep
            core = core + torch.cat([x.new_zeros(B, dk, P, D), contrib], dim=1)
        corr_B = self.UB(g[..., self.n_lags, :] * self.VB(c).unsqueeze(2))
        return core + self._action_term(core, self.B(c).unsqueeze(2) + corr_B)

    def forward_heads(self, x, c):
        """(J,B,T,P,D): every head's PRE-link output. Index 0 equals forward(x, c)."""
        if self.mode in {"mlp_var", "ltv", "ssm"}:
            core = torch.relu(self._trunk(x, c))
            outs = [self.W(core)] + [Wj(core) for Wj in self.W_heads]
        elif self.mode == "additive":
            Ba = self.B(c).unsqueeze(2)
            outs = [self.W(x) + Ba] + [
                Wj(x) + Bj(c).unsqueeze(2)
                for Wj, Bj in zip(self.W_heads, self.B_heads)
            ]
        else:
            raise ValueError(f"n_heads>1 not supported for mode={self.mode}")
        return torch.stack(outs, dim=0)

    def gates(self, x):
        """LTV mode selectors g(z_t), shaped (B,T,P,n_lags+1,r).

        gate_input="support" feeds the binary support s_t = 1[z_t > 0] instead of z_t,
        making the gate exactly invariant to any support-preserving rescaling of the
        code. Note RDMReg already pins z's scale, so the defensible claim for this is
        inductive bias, not identifiability.
        """
        B, T, P, _ = x.shape
        if self.gate_input == "both":
            # support AND magnitude. The support alone is a deterministic function of
            # x, so by the data-processing inequality a support-only gate is bounded
            # ABOVE by a magnitude gate -- it can only ever win as an inductive bias.
            # Concatenating makes the gate's input a strict SUPERSET of either, which
            # removes the information penalty and isolates whether the support carries
            # inductive value once it is no longer paid for.
            src = torch.cat([x, (x > 0).to(x.dtype)], dim=-1)
        else:
            src = (x > 0).to(x.dtype) if self.gate_input == "support" else x
        logits = self.gate(src).view(B, T, P, self.n_lags + 1, self.rank)
        if self.gate_norm == "softmax":
            # r * softmax, not bare softmax: softmax over r modes has mean 1/r
            # (0.0625 at r=16) against sigmoid's ~0.5. That ~8x shrink of the
            # U_k(g * V_k z) gradient path is, at EPOCHS=2, indistinguishable from
            # "support gating is worse". The r factor sets mean gate magnitude to 1.
            return self.rank * torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def forward(self, x, c):
        """x: (B,T,P,D) codes; c: (B,T,D) action embeddings -> (B,T,P,D)."""
        B, T, P, D = x.shape
        if self.mode == "var":
            out = self.lags[0](x)                       # A z_t                       (B,T,P,D)
            for k in range(1, self.n_lags):
                if k >= T:
                    break                               # no z_{t-k} available (T<num_hist, e.g. cold-start)
                xk = self.lags[k](x[:, : T - k])        # lag-k on frames [0 .. T-1-k]
                out = out + torch.cat([x.new_zeros(B, k, P, D), xk], dim=1)  # place at output positions [k:]
            return out + self.B(c).unsqueeze(2)         # + B a_t   (B,T,1,D) broadcast over patches
        if self.mode in {"mlp_var", "ltv", "ssm"}:
            # z' = W ReLU(trunk); trunk is shared with forward_heads so the single-head
            # and multi-head paths cannot drift apart.
            # PRE-link; VWorldModel applies identity/reprelu.
            return self.W(torch.relu(self._trunk(x, c)))
        if self.mode == "lie":
            v = self._rotate(x, self.phi[0]) * self.lag_scale[0].repeat_interleave(2)
            for k in range(1, self.n_lags):
                if k >= T:
                    break                            # z_{t-k} unavailable (cold-start)
                xk = x[:, : T - k]
                vk = (self._rotate(xk, self.phi[k])
                      * self.lag_scale[k].repeat_interleave(2))
                v = v + torch.cat([x.new_zeros(B, k, P, D), vk], dim=1)
            theta = self.W_theta(c).unsqueeze(2)     # (B,T,1,D/2), broadcast over patches
            sc = torch.exp(-self.log_scale) if self.log_scale is not None else None
            return self._rotate(v, theta, scale=sc)  # PRE-link; VWorldModel applies h
        if self.mode == "additive":
            Wz = self.W(x)                          # (B,T,P,D)
            Ba = self.B(c).unsqueeze(2)             # (B,T,1,D) broadcast over patches
            return Wz + Ba
        d = self.to_d(c)                            # (B,T,D)
        uv = self.to_uv(c).view(B, T, D, 2 * self.rank)
        U, V = uv[..., : self.rank], uv[..., self.rank :]   # (B,T,D,r) each
        diag_part = d.unsqueeze(2) * x              # (B,T,P,D)  diag(d) z
        Vtz = torch.einsum("btdr,btpd->btpr", V, x)         # V^T z  (B,T,P,r)
        lowrank = torch.einsum("btdr,btpr->btpd", U, Vtz)   # U (V^T z)  (B,T,P,D)
        return diag_part + lowrank





def reprelu(x):
    """Straight-through "reparameterized ReLU": ReLU on the forward pass, GELU gradient on
    the backward pass. Forward value == ReLU(x) exactly (the detached ReLU/GELU terms cancel
    in value), so hard zeros are preserved; the backward gradient is GELU'(x), which is
    nonzero for x < 0 -- so a thresholded/zeroed coordinate (soft) or group (group) keeps
    receiving gradient instead of a dead zero. Drop-in for a max(., 0) whose zero region
    would otherwise stop learning."""
    return torch.relu(x).detach() + F.gelu(x) - F.gelu(x).detach()


def kwta(x, k):
    """k-Winners-Take-All along the last axis: keep the k largest entries, zero the rest.

    Straight-through backward (forward value is the hard top-k, gradient flows to every
    coordinate) for the same reason reprelu uses a GELU backward: a coordinate that lost
    this round still needs gradient, or it can never win again.

    Selection is by scatter on topk indices rather than an `x >= kth_value` comparison.
    After rectification many entries are exactly 0, and a threshold test would admit all
    of them at once whenever fewer than k coordinates are positive; scatter always marks
    exactly k positions.
    """
    D = x.shape[-1]
    if k is None or k >= D:
        return x
    idx = x.topk(k, dim=-1).indices
    mask = torch.zeros_like(x).scatter_(-1, idx, 1.0)
    hard = x * mask
    return hard.detach() + x - x.detach()


def swd(z, target, num_projections=8192):
    """Sliced-Wasserstein-2 between z and target, shape (N, *batch, D).
    Dim 0 (N) is the SAMPLE axis (sorted to form the empirical 1-D CDF); the last dim
    (D) is the feature/support axis (random projections live in R^D, shared); any
    MIDDLE dims are extra BATCH axes (batched-matmul style) -> an independent SWD per
    slice, averaged. So features across the batch axes are NOT pooled as i.i.d. samples."""
    D = z.shape[-1]
    proj = torch.randn(num_projections, D, device=z.device)
    proj = proj / proj.norm(dim=1, keepdim=True)
    pz = torch.sort(z @ proj.T, dim=0).values  # (N, *batch, num_proj), sorted over N
    pt = torch.sort(target @ proj.T, dim=0).values
    return ((pz - pt) ** 2).mean()


def gng_unit_sigma(p):
    """Auto sigma so the zero-mean Generalized Gaussian GN_p has UNIT variance, for the
    Gamma-based sampler below: sigma = sqrt(Gamma(1/p)/Gamma(3/p)) / p^(1/p)
    (p=1 -> Laplace scale 1/sqrt2 -> var 1; p=2 -> sigma 1 -> standard normal)."""
    return math.sqrt(math.gamma(1.0 / p) / math.gamma(3.0 / p)) / (p ** (1.0 / p))


def sample_generalized_gaussian(shape, p=2.0, device="cpu"):
    """Unified base sampler: zero-mean unit-variance Generalized Gaussian GN_p.
    p=2 -> Gaussian, p=1 -> Laplace, general p via the Gamma trick. The link function
    h(.) is applied to this base (outside) to form the target -> identity/relu
    targets all come from one interface (see RDMReg)."""
    assert p > 0.0, "p must be > 0"
    sigma = gng_unit_sigma(p)
    if p == 1.0:
        return torch.distributions.Laplace(
            torch.tensor(0.0, device=device), torch.tensor(sigma, device=device)
        ).sample(shape)
    if p == 2.0:
        return sigma * torch.randn(shape, device=device)
    sign = torch.empty(shape, device=device).bernoulli_(0.5) * 2 - 1
    g = torch.distributions.Gamma(1.0 / p, 1.0).sample(shape).to(device)
    gn = sign * (p * g).pow(1.0 / p)
    return sigma * gn


class Link(nn.Module):
    """Architectural link function h(.) defining the representation space the predictor
    and planner operate in. Owned by VWorldModel and applied to encoder & predictor
    outputs (one shared instance -> tied threshold).

      identity       -> u                              (dense; pairs with gaussian target)
      relu           -> relu(u)                         (rectified; pairs with rectified-GG)
      reprelu        -> relu(u) fwd, gelu-grad bwd         (rectified, straight-through;
                        zeroed coords keep receiving gradient -- pairs with rectified-GG)

    Optional k-WTA (kwta_k, default None = upstream): keeps only the k largest
    coordinates. It lives HERE, in the one shared instance, so z and z_hat are both
    exactly k-sparse and the tied-threshold invariant between encoder and predictor
    outputs is preserved. RDMReg builds its target as link(GN_p + mu) with this same
    instance, so the target is k-sparse too and its density matches by construction
    -- mu matching (train.py) then aligns the surviving MAGNITUDES rather than the
    density. Stateless: no parameters, so state_dicts stay byte-identical.
    """

    def __init__(self, kind="identity", kwta_k=None):
        super().__init__()
        assert kind in ("identity", "relu", "reprelu"), f"link {kind} not supported"
        self.kind = kind
        self.kwta_k = kwta_k

    def forward(self, u):
        if self.kind == "identity":
            z = u
        elif self.kind == "relu":
            z = F.relu(u)
        else:
            z = reprelu(u)
        if self.kwta_k is not None:
            z = kwta(z, self.kwta_k)
        return z


AGG_PATTERNS = {
    "btp": "b t p d -> (b t p) d",
    "b": "b t p d -> b (t p) d",
    "bp": "b t p d -> (b p) t d",
    "bt": "b t p d -> (b t) p d",
}


class RDMReg(nn.Module):
    """Sliced-Wasserstein distribution-matching regularizer (param-free).

    Unified target interface: always sample a zero-mean, unit-variance Generalized
    Gaussian base GN_p (p=2 Gaussian, p=1 Laplace, auto sigma) and apply the SAME link
    h(.) that produced the features. So the target is self-consistent with the
    representation space:
      identity link       -> target = GN_p                 (dense)
      relu / reprelu link -> target = relu(GN_p)           (rectified GG)
    The target is detached (a reference).

    `agg` chooses how (b,t,p,d) features map onto SWD's (sample, *batch, feature) axes
    (see AGG_PATTERNS). NB: the SWD scale is O(1), so `reg_weight` should be large
    (~100), unlike SIGReg's N-scaled statistic.
    """

    def __init__(self, target_p=2.0, num_projections=8192, agg="btp", mu=0.0):
        super().__init__()
        assert agg in AGG_PATTERNS, f"agg {agg} not in {list(AGG_PATTERNS)}"
        self.target_p = target_p          # GN_p shape: 2 -> Gaussian, 1 -> Laplace
        self.num_projections = num_projections
        self.agg = agg
        self.mu = mu                      # target-mean shift: with the relu link, mu<0 pushes

    def reg_loss(self, z, link=None):
        """z: linked encoder features (b, t, p, d). Reshaped per `agg` into
        (sample, *batch, feature). `link` is the model's link h(.); target is h(GN_p)."""
        z = rearrange(z, AGG_PATTERNS[self.agg])
        base = sample_generalized_gaussian(z.shape, self.target_p, z.device)
        if self.mu != 0.0:
            base = base + self.mu
        target = (link(base) if link is not None else base).detach()
        return swd(z, target, self.num_projections)


