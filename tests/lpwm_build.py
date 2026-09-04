"""Build the LpWM model the way train.py does, without a dataset or a GPU.

Shared by the bit-identity fixture and the Pi-WM intervention tests. Mirrors
train.py's init_models/init_optimizers for the adaln (RDMReg) path; if that
construction changes, this must change with it.
"""
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The probe cell: PushT x mlp_var x D=384, sparse (reprelu, target_p=1), from the
# reproduce_pusht.sh HP table (sparse:mlp_var:384 -> reg_weight=0.1, mup_lr=5e-4).
PROBE_OVERRIDES = [
    "env=pusht",
    "predictor=mlp_var",
    "link=reprelu",
    "regularizer=rdmreg",
    "target_p=1",
    "mu=0",
    "agg=b",
    "reg_weight=0.1",
    "num_hist=3",
    "frameskip=5",
    "mup=true",
    "training.mup_lr=5e-4",
    "training.seed=0",
]


def load_cfg(overrides=None):
    os.environ.setdefault("DATASET_DIR", "/nonexistent")  # never read here
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        return compose(
            config_name="train_rdmreg",
            overrides=list(PROBE_OVERRIDES) + list(overrides or []),
        )


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    import random

    random.seed(s)


def _contact_geom(cfg):
    """T3 geometry, mirroring train.py._contact_geom without loading a dataset."""
    from datasets.pusht_dset import PROPRIO_MEAN, PROPRIO_STD

    scale = float(cfg.img_size) / 512.0
    return (
        float(PROPRIO_MEAN[0]), float(PROPRIO_MEAN[1]),
        float(PROPRIO_STD[0]), float(PROPRIO_STD[1]),
        scale,
        15.0 * float(cfg.get("contact_pad", 1.6)) * scale,
    )


def build(cfg, device="cpu"):
    """Return (model, optimizers) with muP init applied, as in a fresh train.py run."""
    encoder = hydra.utils.instantiate(cfg.encoder)
    proprio_encoder = hydra.utils.instantiate(
        cfg.proprio_encoder, in_chans=4, emb_dim=(cfg.action_emb_dim if bool(cfg.get('use_pose', False)) else cfg.proprio_emb_dim)
    )
    action_encoder = hydra.utils.instantiate(
        cfg.action_encoder,
        in_chans=2 * cfg.frameskip,  # TrajSlicerDataset concats frameskip actions
        emb_dim=cfg.action_emb_dim,
    )
    predictor = hydra.utils.instantiate(
        cfg.predictor,
        num_frames=cfg.num_hist,
        num_patches=encoder.num_patches,
        input_dim=encoder.emb_dim,
        hidden_dim=encoder.emb_dim,
        output_dim=encoder.emb_dim,
    )
    link = hydra.utils.instantiate(cfg.link)
    regularizer = hydra.utils.instantiate(cfg.regularizer)

    # T1/T2 mirror of train.py init_models: the reconstruction head is built when
    # aux_decoder (or the legacy has_decoder) is set. train.py:743.
    decoder = None
    if bool(cfg.get("aux_decoder", False)) or bool(cfg.get("has_decoder", False)):
        decoder = hydra.utils.instantiate(cfg.decoder, emb_dim=encoder.emb_dim)

    if cfg.get("mup", False):
        from models.mup import mup_init_

        mup_init_(encoder, tag="encoder")
        mup_init_(predictor, tag="predictor")
        if decoder is not None and cfg.model.train_decoder:
            mup_init_(decoder, tag="decoder")

    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
        action_conditioning=cfg.get("action_conditioning", "concat"),
        regularizer=regularizer,
        reg_weight=cfg.reg_weight,
        detach_target=cfg.get("detach_target", True),
        link=link,
        lamb_var=cfg.get("lamb_var", 0.0),
        lamb_cov=cfg.get("lamb_cov", 0.0),
        var_space=cfg.get("var_space", "u"),
        var_gamma=cfg.get("var_gamma", 1.0),
        # V1-V3. This builder mirrors train.py's construction; its docstring says so, and
        # omitting these made every variant silently identical to the baseline in loss_trace.
        incr_norm=bool(cfg.get("incr_norm", False)),
        act_info=float(cfg.get("act_info", 0.0)),
        act_info_k=int(cfg.get("act_info_k", 4)),
        act_info_neg=str(cfg.get("act_info_neg", "perm")),
        ctrb_w=float(cfg.get("ctrb_w", 0.0)),
        ctrb_h=int(cfg.get("ctrb_h", 5)),
        path_int=bool(cfg.get("path_int", False)),
        path_int_w=float(cfg.get("path_int_w", 1.0)),
        path_int_dims=(4, 2 * int(cfg.get("frameskip", 5))),
        use_pose=bool(cfg.get("use_pose", False)),
        n_heads=cfg.get("n_heads", 1),
        head_entropy_coef=cfg.get("head_entropy_coef", 0.0),
        burst_tau=cfg.get("burst_tau", 0.5),
        # T1/T2
        decode_grad=bool(cfg.get("decode_grad", False)),
        lamb_decode=float(cfg.get("lamb_decode", 1.0)),
        # T3: mirrors train.py _contact_geom() with the dataset's literal constants
        # (datasets/pusht_dset.py PROPRIO_MEAN / PROPRIO_STD), so no dataset is needed.
        contact_gamma=float(cfg.get("contact_gamma", 0.0)),
        contact_shuffle=bool(cfg.get("contact_shuffle", False)),
        contact_eps=float(cfg.get("contact_eps", 1e-8)),
        contact_geom=_contact_geom(cfg),
        # T6: num_pred already flows through cfg.model (conf/train_rdmreg.yaml), so K
        # needs no mirror -- overshoot does.
        overshoot=bool(cfg.get("overshoot", False)),
        # T4/V4: mirrors train.py's model kwargs. act_dim_raw is the RAW action width
        # (2*frameskip), which is what TrajSlicerDataset.action_dim is on pushT; passing
        # the 384-d embedding dim here would build a head whose "actions" are not actions.
        value_w=float(cfg.get("value_w", 0.0)),
        value_mode=str(cfg.get("value_mode", "td")),
        value_tau=float(cfg.get("value_tau", 0.7)),
        value_gamma=float(cfg.get("value_gamma", 0.98)),
        value_hidden=cfg.get("value_hidden", None),
        value_ema=float(cfg.get("value_ema", 0.005)),
        value_p_future=float(cfg.get("value_p_future", 0.5)),
        policy_w=float(cfg.get("policy_w", 0.0)),
        act_dim_raw=2 * int(cfg.get("frameskip", 5)),
        # ROUND 6. This builder mirrors train.py's construction (its docstring says so);
        # omitting V1-V3's kwargs here once made every variant silently identical to the
        # baseline in loss_trace, which is the one thing this file exists to prevent.
        support_w=float(cfg.get("support_w", 0.0)),
        consist_w=float(cfg.get("consist_w", 0.0)),
        consist_src=str(cfg.get("consist_src", "cem")),
        consist_k=int(cfg.get("consist_k", 5)),
        consist_sigma=float(cfg.get("consist_sigma", 1.0)),
        sam_rho=float(cfg.get("sam_rho", 0.0)),
        incr_eps=float(cfg.get("incr_eps", 1e-4)),
        incr_clip=float(cfg.get("incr_clip", 0.0)),
    ).to(device)

    from models.mup import mup_param_groups

    lr = cfg.training.mup_lr
    bw = cfg.training.get("mup_base_width", cfg.get("embed_dim", 384))
    wd = cfg.training.get("mup_weight_decay", 0.01)
    ifix = bool(cfg.get("mup_input_lr_fix", False))
    opts = [
        torch.optim.AdamW(
            mup_param_groups(encoder, lr, bw, weight_decay=wd, tag="encoder", input_lr_fix=ifix),
            lr=lr,
            eps=1e-15,
        ),
        torch.optim.AdamW(
            mup_param_groups(predictor, lr, bw, weight_decay=wd, tag="predictor", input_lr_fix=ifix),
            lr=lr,
            eps=1e-15,
        ),
        torch.optim.AdamW(
            mup_param_groups(action_encoder, lr, bw, weight_decay=wd, tag="action_encoder", input_lr_fix=ifix)
            + mup_param_groups(proprio_encoder, lr, bw, weight_decay=wd, tag="proprio_encoder", input_lr_fix=ifix),
            lr=lr,
            eps=1e-15,
        ),
    ]
    if decoder is not None and cfg.model.train_decoder:
        # mirrors train.py init_optimizers:924 -- without this the decoder would be
        # built and never stepped, which is exactly the path_int failure mode.
        opts.append(
            torch.optim.AdamW(
                mup_param_groups(decoder, lr, bw, weight_decay=wd, tag="decoder", input_lr_fix=ifix),
                lr=lr,
                eps=1e-15,
            )
        )
    # T4/V4: mirrors train.py init_optimizers. Plain AdamW, NOT mup_param_groups -- the
    # heads' fan-in is 3D, so muP would make their lr a function of the swept width and
    # confound T4 with the wave6/wave7 factorial. Present here so tests/test_arms.py can
    # assert the heads are IN an optimizer, which is the item path_int was missing.
    head_lr = float(cfg.training.get("value_lr", 3e-4))
    for head in (getattr(model, "value_head", None), getattr(model, "policy_head", None)):
        if head is not None:
            opts.append(torch.optim.AdamW(head.parameters(), lr=head_lr))
    return model, opts


def synthetic_batch(cfg, batch_size, gen, device="cpu"):
    """Deterministic stand-in for a real batch.

    Synthetic rather than real data on purpose: the invariant under test is the
    model computation, and this keeps the fixture runnable with no dataset, no
    video decoding, and no GPU.
    """
    T = cfg.num_hist + cfg.num_pred
    return (
        {
            "visual": torch.rand(
                batch_size, T, 3, cfg.img_size, cfg.img_size, generator=gen
            ).to(device),
            "proprio": torch.rand(batch_size, T, 4, generator=gen).to(device),
        },
        torch.rand(batch_size, T, 2 * cfg.frameskip, generator=gen).to(device),
    )


def loss_trace(n_steps=5, batch_size=2, precision="fp32", seed=0, overrides=None):
    """Run n_steps of training on CPU deterministically; return the loss trace."""
    torch.use_deterministic_algorithms(True)
    cfg = load_cfg(overrides)
    seed_all(seed)
    model, opts = build(cfg, device="cpu")

    gen = torch.Generator().manual_seed(1234)
    batches = [synthetic_batch(cfg, batch_size, gen) for _ in range(n_steps)]

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    trace = []
    model.train()
    for obs, act in batches:
        for o in opts:
            o.zero_grad()
        if precision == "fp32":
            _, _, _, loss, comps = model(obs, act)
        else:
            with torch.autocast("cpu", dtype=dtype):
                _, _, _, loss, comps = model(obs, act)
        loss.backward()
        for o in opts:
            o.step()
        trace.append(
            {k: float(v) for k, v in sorted(comps.items()) if torch.is_tensor(v)}
        )
    return trace
