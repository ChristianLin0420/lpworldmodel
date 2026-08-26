"""Train an LpWM world model on a stable-worldmodel dataset (Hydra app).

Mirrors the LpWM fork's scripts/train/lpwm.py, re-pointed at the local
lpwm_swm.* modules and the public stable-worldmodel / stable-pretraining APIs.
The loss is assembled in `lpjepa_forward` (le-wm style):

    loss = pred_loss + rdmreg.weight * rdmreg_loss
           [+ temporal_jaccard.weight * temporal_jaccard_loss]   (optional)

Run:
    python train.py data=piecewise trainer.max_epochs=10
    python train.py data=ogb_cube loss.temporal_jaccard.enabled=true \
        loss.temporal_jaccard.weight=0.1
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torchmetrics
from einops import rearrange
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from torch import nn

from loss import RDMReg, TemporalJaccardLoss
from metrics import (
    avg_per_dim_var,
    l0_sparsity_metric,
    l1_sparsity_metric,
    off_diag_cov_sum,
)
from module import Probe_MLP


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Save an object checkpoint after each epoch via swm.save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            config_key='model',
            filename=f'weights_epoch_{epoch}.pt',
        )


def lpjepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute LpWM losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.rdmreg.weight

    # sequence-boundary NaNs -> 0
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)
    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # 1. prediction MSE
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()

    # 2. RDMReg
    loss_func_type = self.rdmreg.loss_func_type
    matching_mode = self.rdmreg.matching_mode
    if loss_func_type == 'sliced_wasserstein':
        if matching_mode == 'b_t_d':
            output['rdmreg_loss'] = self.rdmreg(emb)
        elif matching_mode == 'bt_d':
            output['rdmreg_loss'] = self.rdmreg(
                rearrange(emb, 'b t d -> (b t) d')
            )
        else:
            raise ValueError(f'Invalid matching_mode: {matching_mode}')
    elif loss_func_type == 'epps_pulley':
        output['rdmreg_loss'] = self.rdmreg(emb.transpose(0, 1))
    else:
        raise ValueError(f'Invalid loss_func_type: {loss_func_type}')

    output['loss'] = output['pred_loss'] + lambd * output['rdmreg_loss']

    if cfg.loss.temporal_jaccard.enabled:
        output['temporal_jaccard_loss'] = self.temporal_jaccard(emb)
        output['loss'] = (
            output['loss']
            + cfg.loss.temporal_jaccard.weight
            * output['temporal_jaccard_loss']
        )

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    # sparsity / variance / covariance diagnostics
    with torch.no_grad():
        emb_flat = rearrange(emb, 'b t d -> (b t) d')
        pred_flat = rearrange(pred_emb, 'b t d -> (b t) d')
        eval_dict = {
            f'{stage}/encoder_mean': emb_flat.mean(),
            f'{stage}/predictor_mean': pred_flat.mean(),
            f'{stage}/encoder_l1_sparsity': l1_sparsity_metric(emb_flat),
            f'{stage}/encoder_l0_sparsity': l0_sparsity_metric(emb_flat),
            f'{stage}/predictor_l1_sparsity': l1_sparsity_metric(pred_flat),
            f'{stage}/predictor_l0_sparsity': l0_sparsity_metric(pred_flat),
            f'{stage}/encoder_avg_per_dim_var': avg_per_dim_var(emb_flat),
            f'{stage}/encoder_off_diag_cov_sum': off_diag_cov_sum(emb_flat),
            f'{stage}/predictor_avg_per_dim_var': avg_per_dim_var(pred_flat),
            f'{stage}/predictor_off_diag_cov_sum': off_diag_cov_sum(
                pred_flat
            ),
        }
    self.log_dict(eval_dict, on_step=True, sync_dist=True)

    return output


@hydra.main(version_base=None, config_path='./config', config_name='lpwm')
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

        effective_act_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        cfg.model.action_encoder.input_dim = effective_act_dim

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=rnd_gen
    )
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)

    extra_losses = {}
    if cfg.loss.temporal_jaccard.enabled:
        assert cfg.loss.rdmreg.kwargs.link_function_type == 'ReLU', (
            'temporal_jaccard requires rdmreg link_function_type="ReLU"'
        )
        extra_losses['temporal_jaccard'] = TemporalJaccardLoss(
            **cfg.loss.temporal_jaccard.kwargs
        )

    world_model = spt.Module(
        model=world_model,
        rdmreg=RDMReg(**cfg.loss.rdmreg.kwargs),
        **extra_losses,
        forward=partial(lpjepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    callbacks = [
        SaveCkptCallback(
            run_name=cfg.output_model_name, cfg=cfg, epoch_interval=1
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith('pixels') or col == 'action':
            continue
        callbacks.append(
            spt.callbacks.OnlineProbe(
                world_model,
                target=col,
                input='emb',
                name=f'{col}_probe',
                probe=Probe_MLP(
                    input_dim=cfg.embed_dim,
                    hidden_dim=512,
                    output_dim=dataset.get_dim(col),
                ),
                loss=nn.MSELoss(),
                metrics={'mse': torchmetrics.MeanSquaredError()},
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
