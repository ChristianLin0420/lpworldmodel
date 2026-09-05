"""Maximal Update Parametrization (muP; Yang et al., Tensor Programs V, arXiv:2203.03466) —
LR + init, for width (code-dim `D` / `proj_dim`) transfer.

Summary:
- **LR rule (the part that matters for transfer):** `used_lr(param) = base_lr * base_width / fan_in`
  for matrix-like weights (`fan_in = prod(weight.shape[1:])` for nn.Linear/nn.Conv), and
  `used_lr = base_lr` for biases/LayerNorm/embeddings (fan_in := base_width,
  the muP "Adam LR = 1" row). `base_width` is a FIXED reference (= embed_dim = 384), NOT the swept D.
  Applied at every width: a fan_in==base_width matrix and all vector-like params get `base_lr`;
  wider-fan_in matrices get proportionally less; and the predictor's code-reading matrices (fan_in=D)
  transfer-scale to `base_lr*base_width/D`. So `base_lr` stays an ordinary LR (no bias inflation).
- **Weight classes (muP is by DIMENSION, not function):** a dim is "infinite" if it is a network
  WIDTH knob (embed_dim/attn_inner/d_head/mlp_dim/proj_hidden/code_dim — even ones we hold fixed this
  sweep), "finite" only if problem-imposed (input channels, patch pixels, raw action/proprio dim,
  seq/context). hidden = infinite→infinite; input = finite→infinite; output = infinite→FINITE.
  **This adaln JEPA has NO output weights** — the loss lives in the D-dim code (mean-reduced → O(1)),
  so every terminal readout (encoder code head, predictor prediction head) is infinite→infinite =
  HIDDEN. An output weight would appear only if we add a finite-dim head (decoder→pixels, reward/state/
  probe head); that head, and only it, would get zero-init.
- **Init:** hidden Linear/Conv weights ~ N(0, 1/fan_in) (== muP hidden init; PyTorch's default is
  already ∝1/fan_in, so init is ~transfer-neutral — LR is the real fix). Embeddings ~ N(0, 0.02),
  LayerNorm 1, biases 0. Deliberate structural inits are RE-APPLIED afterward (AdaLN-zero, LTI(1) W=I/B=0,
  linear_pa zero-last) so muP does not clobber them. No output-weight zero-init pass (there are no output
  weights); the predictor's prediction head is hidden-init, and AdaLN-zero / LTI(1) W=I already give the
  benign near-identity-at-init that a readout zero-init would (a §3.2-style denoiser, for free).

Scope: MLP-head predictors (ARPredictor / LinearDynamicsPredictor).
"""
import math
import torch
import torch.nn as nn

_LINEARISH = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_EMB_KEYS = ("cls_token", "pos_embedding", "temporal_pos", "spatial_pos")

# muP weight CLASSES. The module docstring above already states the rule: a dim is
# "finite" when it is problem-imposed ("input channels, patch pixels, raw action/proprio
# dim, seq/context"), and a finite->infinite map is an INPUT weight, whose muP Adam LR is
# Theta(1) -- i.e. base_lr, NOT base_lr*base_width/fan_in. That carve-out was documented
# and never implemented: the loop below applies the HIDDEN rule to every Linear/Conv.
#
# Measured consequence at base_lr=5e-4, base_width=384 (from the schema each run prints):
#   proprio_encoder.patch_embed.weight   fan_in=4   -> 4.800e-02   (96x base_lr)
#   action_encoder.patch_embed.weight    fan_in=10  -> 1.920e-02   (38x)
#   predictor.{Ulag.*, UB}.weight        fan_in=16  -> 1.200e-02   (24x)
# while action_encoder.embed.2.weight (fan_in=1536) gets 1.250e-04 -- the two layers of
# one 2-layer MLP end up 154x apart.
#
# This also matters for any experiment that CHANGES an input fan_in: concatenating a 4-D
# pose onto the 10-D action moves action_encoder.patch_embed from 1.920e-02 to 1.371e-02,
# so "adding pose" would silently also be "changing a learning rate". With the carve-out
# on, an input weight's LR is base_lr regardless of fan_in, and that confound disappears.
#
# Off by default: bit-identical to upstream unless input_lr_fix=True.
_INPUT_WEIGHTS = (
    "patch_embed.weight",   # encoder patch pixels (588); action (10); proprio (4)
    "embed.0.weight",       # Embedder's first Linear: raw action/proprio dim
    "Ulag.",                # LTV low-rank up-projection: fan_in = rank r (16), fixed in D
    "UB.",                  # ditto, action branch
)


def _is_input_weight(name):
    return any(pat in name for pat in _INPUT_WEIGHTS)


def _fan_in(weight):
    """prod of all dims except the output dim (dim 0)."""
    return int(weight[0].numel())


def mup_fan_in_map(model):
    """param-name -> fan_in. Linear/Conv `.weight` get a shape-derived fan_in.
    Every other param is absent here and treated as fan_in=1 by the caller."""
    fmap = {}
    for mname, m in model.named_modules():
        pfx = f"{mname}." if mname else ""
        if isinstance(m, _LINEARISH):
            fmap[f"{pfx}weight"] = _fan_in(m.weight)
    return fmap


def mup_param_groups(model, base_lr, base_width, weight_decay=0.0, tag="", verbose=True,
                     input_lr_fix=False):
    """Per-parameter AdamW param-groups implementing the muP Adam LR rule with a fixed global
    reference `base_width` (muP "tilde_d" formulation), applied at EVERY width including the base:

        matrix-like (Linear/Conv weights):                  used_lr = base_lr * base_width / fan_in
        vector/scalar-like (biases, LayerNorm, embeddings): used_lr = base_lr   (fan_in := base_width)

    So `used_lr` is a clean function of the *current* fan_in: a fan_in==base_width matrix and all
    vector-like params get `base_lr`; wider-fan_in matrices get proportionally less (even at the base
    model — a fan_in=2048 layer gets base_lr*base_width/2048); and matrices whose fan_in scales with
    width (the predictor's code-reading matrices, fan_in==D) transfer-scale to base_lr*base_width/D.
    `base_lr` therefore stays an ordinary LR and biases don't get an inflated LR (hidden-at-base ==
    bias == base_lr).

    **`base_width` MUST be a fixed constant (the tuned base = embed_dim = 384), NOT the swept code
    dim D.** No collision issue at D==2048: used_lr depends only on the current fan_in, so c_proj
    (fan_in=2048 at all widths) always gets base_lr*base_width/2048 while a code matrix gets
    base_lr*base_width/D -- they coincide at D=2048, which is correct (both are fan_in=2048 there).
    weight_decay only on dim>=2. Prints the LR schema (name | fan_in | used_lr | wd) when verbose."""
    fmap = mup_fan_in_map(model)
    groups = []
    if verbose:
        print(f"\n### muP LR schema [{tag}]  base_lr={base_lr}  base_width={base_width} ###")
        print(f"### used_lr = base_lr * base_width / fan_in ###")
        print(f"{'param':<46} {'fan_in':>8} {'base_w':>8} {'used_lr':>12} {'wd':>6}")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in fmap:  # matrix-like
            fan_in = fmap[name]
            if input_lr_fix and _is_input_weight(name):
                used_lr = base_lr                       # muP INPUT weight: Adam LR = Theta(1)
                cls = "input"
            else:
                used_lr = base_lr * base_width / fan_in  # muP HIDDEN weight
                cls = "hidden"
        else:
            fan_in = base_width
            used_lr = base_lr
            cls = "vector"
        wd = weight_decay if p.dim() >= 2 else 0.0
        groups.append({"params": [p], "name": name, "lr": used_lr, "weight_decay": wd})
        if verbose:
            offbase = " (fan_in!=base_width)" if name in fmap and fan_in != base_width else ""
            print(f"{name:<46} {fan_in:>8} {base_width:>8} {used_lr:>12.3e} {wd:>6} {cls}{offbase}")
    if verbose:
        n = sum(pp.numel() for g in groups for pp in g["params"])
        print(f"### [{tag}] {len(groups)} param-groups, {n:,} params ###")
    return groups


@torch.no_grad()
def mup_init_(model, emb_std=0.02, tag="", verbose=True):
    """In-place faithful muP init. Call on freshly-built (scratch) modules only — never on resume.
    Order: generic pass (hidden 1/sqrt(fan_in), norms, biases) -> embeddings -> RE-APPLY the
    deliberate structural inits (AdaLN-zero, LTI(1) W=I/B=0, linear_pa zero-last).
    Records + prints the init schema (name | scheme | value) when verbose."""
    from models.infojepa_modules import ConditionalBlock, LinearDynamicsPredictor

    schema = {}

    # 1) generic muP init
    for mname, m in model.named_modules():
        pfx = f"{mname}." if mname else ""
        if isinstance(m, _LINEARISH):
            std = 1.0 / math.sqrt(_fan_in(m.weight))
            nn.init.normal_(m.weight, mean=0.0, std=std)
            schema[f"{pfx}weight"] = f"hidden  N(0, 1/sqrt(fan_in)) std={std:.4g}"
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
                schema[f"{pfx}bias"] = "bias    zeros"
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
                schema[f"{pfx}weight"] = "norm    ones"
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                schema[f"{pfx}bias"] = "norm    zeros"

    for name, p in model.named_parameters():
        if any(k in name for k in _EMB_KEYS):
            nn.init.normal_(p, mean=0.0, std=emb_std)
            schema[name] = f"embed   N(0, {emb_std})"

    for mname, m in model.named_modules():
        pfx = f"{mname}." if mname else ""
        if isinstance(m, ConditionalBlock):
            nn.init.zeros_(m.adaLN_modulation[-1].weight)
            nn.init.zeros_(m.adaLN_modulation[-1].bias)
            schema[f"{pfx}adaLN_modulation.{len(m.adaLN_modulation)-1}.weight"] = "deliberate  AdaLN-zero (0)"
            schema[f"{pfx}adaLN_modulation.{len(m.adaLN_modulation)-1}.bias"] = "deliberate  AdaLN-zero (0)"
        if isinstance(m, LinearDynamicsPredictor):
            if m.mode == "additive":  # LTI(1): z' = W z + B a, W=I, B=0
                nn.init.eye_(m.W.weight); nn.init.zeros_(m.W.bias); nn.init.zeros_(m.B.weight)
                schema[f"{pfx}W.weight"] = "deliberate  identity (LTI(1) W=I)"
                schema[f"{pfx}W.bias"] = "deliberate  zeros"
                schema[f"{pfx}B.weight"] = "deliberate  zeros (LTI(1) B=0)"
            elif m.mode == "var":
                nn.init.eye_(m.lags[0].weight); nn.init.zeros_(m.lags[0].bias)
                schema[f"{pfx}lags.0.weight"] = "deliberate  identity (VAR lag0=I)"
                schema[f"{pfx}lags.0.bias"] = "deliberate  zeros"
                for i in range(1, m.n_lags):
                    nn.init.zeros_(m.lags[i].weight)
                    schema[f"{pfx}lags.{i}.weight"] = "deliberate  zeros (VAR older lag=0)"
                nn.init.zeros_(m.B.weight)
                schema[f"{pfx}B.weight"] = "deliberate  zeros (VAR B=0)"
            elif m.mode == "ssm":
                # ROUND 8 / T1. The generic muP pass above gives every Linear
                # N(0, 1/sqrt(fan_in)); this restores the deliberate init, exactly as
                # additive/var/lie do above. Without it the state starts random and the
                # arm is not a near-identity dynamics at step 0, which is the convention
                # every other predictor in this file follows.
                #   A  = 0.9 I  -- a CONTRACTION, so the recurrent state cannot blow up
                #                  before it has learned anything (this is the one init
                #                  choice with no precedent in the FIR modes, because they
                #                  have no state to diverge).
                #   Bz = I      -- the current frame passes straight through
                #   Ba = 0      -- AdaLN-zero: no action effect at init, as additive/var
                #   C  = I      -- readout starts as the state itself
                nn.init.eye_(m.A.weight); m.A.weight.data.mul_(0.9)
                nn.init.eye_(m.Bz.weight); nn.init.zeros_(m.Bz.bias)
                nn.init.zeros_(m.Ba.weight)
                nn.init.eye_(m.C.weight); nn.init.zeros_(m.C.bias)
                schema[f"{pfx}A.weight"] = "deliberate  0.9*I (SSM contraction)"
                schema[f"{pfx}Bz.weight"] = "deliberate  identity (SSM input=I)"
                schema[f"{pfx}Bz.bias"] = "deliberate  zeros"
                schema[f"{pfx}Ba.weight"] = "deliberate  zeros (SSM action=0 at init)"
                schema[f"{pfx}C.weight"] = "deliberate  identity (SSM readout=I)"
                schema[f"{pfx}C.bias"] = "deliberate  zeros"
            elif m.mode == "mlp_var":
                pass
            elif m.mode == "lie":
                # the generic pass above would have given W_theta N(0, 1/sqrt(D)); the
                # deliberate init is ZERO, so that R(theta) = I and the predictor is the
                # identity at step 0 (same convention as var/additive).
                nn.init.zeros_(m.W_theta.weight); nn.init.zeros_(m.W_theta.bias)
                schema[f"{pfx}W_theta.weight"] = "deliberate  zeros (lie: R(theta)=I at init)"
                schema[f"{pfx}W_theta.bias"] = "deliberate  zeros"
            elif m.mode == "ltv":
                for i, u in enumerate([*m.Ulag, m.UB]):
                    nn.init.zeros_(u.weight)
                    schema[f"{pfx}{'Ulag.' + str(i) if i < len(m.Ulag) else 'UB'}.weight"] = \
                        "deliberate  zeros (LTV up-proj U=0; correction off at init)"
            else:
                nn.init.zeros_(m.to_d[-1].weight); nn.init.ones_(m.to_d[-1].bias)
                nn.init.zeros_(m.to_uv[-1].weight); nn.init.zeros_(m.to_uv[-1].bias)
                schema[f"{pfx}to_d.{len(m.to_d)-1}.weight"] = "deliberate  zeros (action_linear diag)"
                schema[f"{pfx}to_uv.{len(m.to_uv)-1}.weight"] = "deliberate  zeros (action_linear lowrank)"

    if verbose:
        fmap = mup_fan_in_map(model)
        print(f"\n### muP init schema [{tag}]  (fan_in=1 => vector/scalar-like) ###")
        print(f"{'param':<52} {'fan_in':>8}  scheme")
        for name, _ in model.named_parameters():
            print(f"{name:<52} {fmap.get(name, 1):>8}  {schema.get(name, 'UNTOUCHED (constructor init)')}")
        print(f"### [{tag}] init done ###")
    return model
