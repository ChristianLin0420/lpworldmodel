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


def mup_param_groups(model, base_lr, base_width, weight_decay=0.0, tag="", verbose=True):
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
        if name in fmap:  # matrix-like: lr = base_lr * base_width / fan_in
            fan_in = fmap[name]
            used_lr = base_lr * base_width / fan_in
        else:
            fan_in = base_width
            used_lr = base_lr
        wd = weight_decay if p.dim() >= 2 else 0.0
        groups.append({"params": [p], "name": name, "lr": used_lr, "weight_decay": wd})
        if verbose:
            offbase = " (fan_in!=base_width)" if name in fmap and fan_in != base_width else ""
            print(f"{name:<46} {fan_in:>8} {base_width:>8} {used_lr:>12.3e} {wd:>6}{offbase}")
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
            elif m.mode == "mlp_var":
                pass
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
