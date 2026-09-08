"""ROUND 9. Temporal and spatial state operators for the ENCODER (not the predictor).

Round 8 put an SSM in the predictor (`LinearDynamicsPredictor(mode="ssm")`) and it was the
one mechanism that leaned positive. Round 9 asks whether the operator belongs in the
representation instead. Three modules live here:

    TemporalState   P1/P2/P4 -- a state across FRAMES:  s_t = A s_{t-1} + Bz x_t
    ScanMixer       P3/P4    -- a state across TOKENS, replacing attention
    ScanBlock       P3/P4    -- a Block whose `attn` is a ScanMixer

WHY THE STATE IS A RESIDUAL WITH C = 0, AND NOT `out = C s`.
The diary's first draft asked for `A = 0, Bz = C = I`, arguing that reproduces today's
encoder. It does -- but only at init, and only for that one triple. The residual form
`out_t = x_t + C s_t` with `C = 0` is strictly stronger and gives two separable claims:

    C = 0  ->  out == x BITWISE at step 0 (adding an exact zero is exact in IEEE), so
               tests/test_bit_identity.py passes with the arm switched ON.
    A = 0  ->  out_t is a PER-FRAME function of x_t for EVERY t, at every point in
               training -- not just at init. That is what makes the frozen-A control a
               control after 8 hours of gradient steps rather than only at step 0.

WHY THIS MATTERS MORE THAN THE DIARY THOUGHT (measured 2026-09-07).
`PiWM-blockcausal` was filed as "the goal is an object the model never produced during
training". That is FALSE, and the checkpoint says so: under a block-causal mask a frame-0
token attends only to frame-0 tokens at every layer, so `forward_temporal(x, T)[:, 0]`
equals `forward(x_0)` to 7e-6 -- float noise, not a difference. The gap is at t >= 1
(0.30-0.45 relative on random inputs), i.e. in the PREDICTION TARGETS: the predictor was
trained to emit history-conditioned codes while CEM scores it against a history-free goal.

So a well-defined single-frame limit is necessary but NOT sufficient -- block_causal had
one. What decides a stateful encoder is how far its t >= 1 codes drift from their own
single-frame codes, which is why `TemporalState.solo()` exists and why the model emits
`enc_state_gap` every step. An arm whose gap is large is a blockcausal repeat no matter
what its success rate says; an arm whose gap is ~0 is its own control.
"""
import math

import torch
from einops import rearrange
from torch import nn

from models.infojepa_modules import Block, FeedForward


class TemporalState(nn.Module):
    """P1/P2. A linear state across the FRAME axis:  s_t = A s_{t-1} + Bz x_t.

    Output is the RESIDUAL `x_t + C s_t` -- see the module docstring for why.

    The recurrence, the zero cold start and the naming are the predictor's `ssm` mode
    (models/infojepa_modules.py:695-708) minus the action term; that code documents the
    cold start as "A * 0 = 0 exactly, so no lag is silently invented", which holds here
    for the same reason.

    nn.Linear rather than bare Parameters so models/mup.py's `_LINEARISH` isinstance walk
    picks them up and assigns the hidden rule base_lr * base_width / fan_in. At D = 384
    that is exactly base_lr. The names A/Bz/C avoid every substring in mup.py's
    `_INPUT_WEIGHTS` (:55-64) and `_EMB_KEYS` (:34), both of which match on `in`, not `==`.
    """

    def __init__(self, dim, freeze_a=False, c_zero=True):
        super().__init__()
        self.A = nn.Linear(dim, dim, bias=False)
        self.Bz = nn.Linear(dim, dim, bias=True)
        self.C = nn.Linear(dim, dim, bias=False)
        self.freeze_a = bool(freeze_a)
        # C = 0 exists ONLY to make the state inert when it is ADDED to the code. In "aux"
        # mode the state never enters the code, so a zero readout is not inertness -- it is
        # an identically-zero signal that leaves the auxiliary objective comparing zeros to
        # zeros with no gradient. There, C starts at the identity.
        self.c_zero = bool(c_zero)
        self.reset_state_init_()

    @torch.no_grad()
    def reset_state_init_(self):
        """The deliberate init, RE-APPLIED by models/mup.py after mup_init_ clobbers it.

        `mup_init_` (mup.py:146-154) re-initialises every nn.Linear to N(0, 1/sqrt(fan_in))
        unconditionally, so calling this from __init__ alone is not enough: the init that
        ships is the one that survives construction, and those are different moments.

        A = 0.9 I follows the predictor's ssm mode (mup.py:203) -- a contraction, so the
        state cannot blow up before it has learned anything. The frozen control uses 0.0,
        which is what makes it exactly per-frame.
        """
        nn.init.eye_(self.A.weight)
        self.A.weight.mul_(0.0 if self.freeze_a else 0.9)
        nn.init.eye_(self.Bz.weight)
        nn.init.zeros_(self.Bz.bias)
        if self.c_zero:
            nn.init.zeros_(self.C.weight)      # "add" mode: state contributes NOTHING at step 0
        else:
            nn.init.eye_(self.C.weight)        # "aux" mode: readout is live from step 0
        if self.freeze_a:
            self.A.weight.requires_grad_(False)

    def forward(self, x, T):
        """x: (b*T, P, D), frames consecutive within each b. -> (out, s), both (b*T, P, D).

        The layout is the one `visual_world_model.encode_obs` produces via
        "b t ... -> (b t) ...", so this composes with the existing dispatch unchanged.
        """
        n = x.shape[0]
        assert n % T == 0, f"batch {n} not divisible by T={T}"
        x = rearrange(x, "(b t) p d -> b t p d", t=T)
        s = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        states = []
        for t in range(T):                     # T = 4 at train (num_hist + num_pred), 1 at goal
            s = self.A(s) + self.Bz(x[:, t])
            states.append(s)
        s = torch.stack(states, dim=1)
        out = x + self.C(s)
        return (rearrange(out, "b t p d -> (b t) p d"),
                rearrange(s, "b t p d -> (b t) p d"))

    def solo(self, x):
        """The T = 1 limit, evaluated per frame: x + C(Bz x).

        This is EXACTLY what the planner encodes -- plan.py:230-232 gives both obs_0 and
        obs_g a time axis of one -- so it is the reference the model's own t >= 1 codes are
        measured against (`enc_state_gap`). Identical expression to forward() at T = 1,
        because A(0) = 0 exactly.
        """
        return x + self.C(self.Bz(x))


class ScanMixer(nn.Module):
    """P3. Attention replaced by two linear sweeps over the TOKEN axis.

        u_l = Bz(LN(x))_l ,   h_l = a * h_{l-1} + u_l  (forward)
                              g_l = a * g_{l+1} + u_l  (backward)
        out = C(h + g)

    Bidirectional because patch tokens have no causal order; a one-way scan would impose a
    raster-order prior the image does not have.

    A IS DIAGONAL, and that is a cost decision, not a modelling shortcut. A dense
    transition forces a 257-step Python loop per block x 12 blocks -- about 3,000
    sequential autograd steps per forward. A diagonal transition makes the recurrence
    associative, so it is computed CHUNKWISE: within a chunk by a materialised
    (chunk, chunk, D) kernel, across chunks by a carry. At chunk = 32 that is
    ceil(257/32) = 9 sequential steps instead of 257.

    The kernel is built by CUMPROD, not by pow and not by cumsum-with-division. Every
    entry is a^k for k >= 0, so |K| <= 1 whenever |a| <= 1; the `a^{-n}` form that a
    naive cumulative-sum formulation needs overflows exactly where this underflows
    harmlessly to zero.

    `a = tanh(scan_a)` so |a| < 1 always and `scan_a = 0` gives a = 0 EXACTLY -- which is
    the frozen control, and which reduces the block to a per-token MLP.

    Drop-in for Attention (models/infojepa_modules.py:99-118): same forward signature, its
    own LayerNorm, dropout gated on self.training. Measured parameter count at D = 384,
    heads = 3, dim_head = 64: 296,832 against attention's 296,064 -- the 768 difference is
    exactly the two decay vectors, so this is NOT the "cannot be parameter-matched" case
    the diary claimed. (inner_dim = 3 * 64 = 192, so to_qkv 221,184 + to_out 74,112 =
    295,296, plus the LayerNorm's 768.)
    """

    CHUNK = 32

    def __init__(self, dim, dropout=0.0, freeze_a=False, a_init=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scan_in = nn.Linear(dim, dim, bias=False)
        self.scan_out = nn.Linear(dim, dim, bias=True)
        # (2, D): one decay per direction. A bare Parameter on purpose -- mup_init_ only
        # touches _LINEARISH modules and `_EMB_KEYS` names, so this survives construction
        # untouched and needs no re-apply. Same reason `lie`'s phi is a Parameter while its
        # W_theta is a Linear (infojepa_modules.py:592-596).
        # ROUND 9 FIX. a_init is the decay the sweep STARTS from, and 0 is the wrong choice.
        # a = tanh(scan_a), so scan_a = atanh(a_init).
        #
        # The first version initialised scan_a = 0, i.e. a = 0, i.e. h_l = u_l: NO TOKEN
        # MIXING AT ALL. The treatment therefore began as twelve blocks of per-patch MLP and
        # had to climb out of a degenerate encoder, and its "frozen" control STAYED there --
        # measured d_action 0.0000-0.0009 against 0.276-0.393 for anything that mixes, so the
        # control could not encode anything and the contrast could not isolate the scan.
        # Starting at 0.5 gives an exponential window of ~1/(1-a) = 2 tokens at step 0, and a
        # frozen control that still mixes.
        _a = float(a_init)
        assert -0.999 < _a < 0.999, f"a_init {_a} outside (-1, 1)"
        self.scan_a = nn.Parameter(torch.full((2, dim), math.atanh(_a)))
        self.dropout = dropout
        self.freeze_a = bool(freeze_a)
        if self.freeze_a:
            self.scan_a.requires_grad_(False)

    @staticmethod
    def _sweep(u, a, chunk):
        """h_l = a * h_{l-1} + u_l, in ceil(L/chunk) sequential steps. u: (N, L, D)."""
        n, ll, d = u.shape
        carry = u.new_zeros(n, d)
        out = []
        for c0 in range(0, ll, chunk):
            uc = u[:, c0:c0 + chunk]
            lc = uc.shape[1]
            idx = torch.arange(lc, device=u.device)
            diff = idx[:, None] - idx[None, :]                     # (lc, lc)
            # pw[k] = a^k, by cumprod so the sign is right for negative a and there is
            # never a division. pw[diff] indexes (lc, D) with (lc, lc) -> (lc, lc, D).
            ones = u.new_ones(1, d)
            pw = torch.cumprod(torch.cat([ones, a.expand(lc - 1, d)], dim=0), dim=0) \
                if lc > 1 else ones
            # A MULTIPLICATIVE mask, not torch.where. where()'s backward goes through
            # masked_scatter_, which under mixed precision compared a BFloat16 self against
            # a Float source and raised -- in the BACKWARD pass, so every scan arm trained
            # for one step and then died. A multiply has no such path and no dtype opinion.
            mask = (diff >= 0).to(pw.dtype)
            k = pw[diff.clamp_min(0)] * mask[:, :, None]
            h = torch.einsum("mnd,bnd->bmd", k, uc) + (pw * a)[None] * carry[:, None]
            carry = h[:, -1]
            out.append(h)
        return torch.cat(out, dim=1)

    def forward(self, x, attn_mask=None):
        # The encoder passes a mask only under block_causal, which is mutually exclusive
        # with the scan (both are answers to the same question). Assert rather than
        # silently ignore -- a dropped mask is a wrong model, not a slow one.
        assert attn_mask is None, "ScanMixer does not take an attention mask"
        h = self.norm(x)
        u = self.scan_in(h)
        # THE RECURRENCE RUNS IN fp32, with autocast off, whatever the surrounding
        # precision. Two independent reasons, and the second one cost a whole wave:
        #   * a scan multiplies decays together, so a low-precision error compounds along
        #     the sequence rather than staying local;
        #   * mixing the fp32 `scan_a` with bf16 activations inside the kernel produced a
        #     dtype error in BACKWARD that no fp32 CPU test could see. Every scan arm
        #     trained one step on GPU and then died with
        #     "masked_scatter_: expected self and source to have same dtypes".
        _dt = u.dtype
        with torch.autocast(device_type=u.device.type, enabled=False):
            uf = u.float()
            a = torch.tanh(self.scan_a.float())
            fwd = self._sweep(uf, a[0], self.CHUNK)
            bwd = self._sweep(uf.flip(1), a[1], self.CHUNK).flip(1)
            mixed = (fwd + bwd).to(_dt)
        out = self.scan_out(mixed)
        if self.training and self.dropout > 0:
            out = torch.nn.functional.dropout(out, p=self.dropout)
        return out


class ScanBlock(Block):
    """P3. A Block whose token mixer is a ScanMixer.

    Subclasses Block so `Transformer.forward`'s `isinstance(block, Block)` dispatch
    (infojepa_modules.py:220) calls it with the two-argument signature; a non-subclass is
    called `block(x, c, attn_mask=...)` and dies on arity mid-wave.

    nn.Module.__init__ is called DIRECTLY rather than super().__init__(): Block's ctor
    would allocate an Attention we immediately discard, which both wastes the parameters
    and -- because it draws from the global RNG stream -- would shift every subsequent
    random draw, so the arm would differ from its control by more than one factor. That is
    the same constraint visual_world_model.py:304-314 documents for the value head.
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=False,
                 freeze_a=False, a_init=0.5):
        nn.Module.__init__(self)
        self.attn = ScanMixer(dim, dropout=dropout, freeze_a=freeze_a, a_init=a_init)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
