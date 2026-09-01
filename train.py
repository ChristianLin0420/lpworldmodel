import os
import re
import sys
import json
import functools
import math
import time
import hydra
import torch
import wandb
import random
import signal
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from models.infojepa_modules import AGG_PATTERNS, gng_unit_sigma
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Sections for the wandb run page. Without these every scalar lands in one flat
# alphabetical list, which makes a 9-arm campaign unreadable; the tuple order is
# the match order, so put the specific patterns first.
_WANDB_SECTIONS = (
    ("head_", "heads"),
    ("l0_frac", "sparsity"),
    ("diag_", "diag"),
    ("_err_rollout", "rollout"),
    ("_rollout", "rollout"),
    ("_err_", "err"),
    ("img_", "img"),
)


def wandb_key(k):
    """Map a flat epoch_log key to a sectioned wandb key, keeping train/val visible.

    'train_loss' -> 'train/loss', 'train_head_usage_p0' -> 'heads/train_head_usage_p0'.
    analysis/figures.py reverses this when it reads a history export.
    """
    for phase in ("train_", "val_"):
        if k.startswith(phase):
            rest = k[len(phase):]
            for pat, sec in _WANDB_SECTIONS:
                if pat in rest:
                    return f"{sec}/{k}"
            return f"{phase[:-1]}/{rest}"
    return k


def wandb_identity(cfg, run_dir_name):
    """Concise, groupable wandb identity for one campaign arm.

    The run dir name carries arm + D + precision + seed, which is unreadable in a
    run list and impossible to group. Split it up: the arm becomes the group so the
    three seeds collapse to one row, the knobs become tags you can filter on, and
    the name is just arm + seed.
    """
    arm = re.sub(r"_pd\d+|_(bf16|fp16|no)(?=_|$)|_s\d+$", "", run_dir_name)
    seed = int(cfg.training.seed)
    pred = cfg.predictor.get("mode", None) or cfg.predictor._target_.rsplit(".", 1)[-1]
    kwta = cfg.get("kwta_k", None)
    j = int(cfg.get("n_heads", 1))

    tags = [
        f"pred:{pred}",
        f"D:{cfg.get('embed_dim', '?')}",
        f"prec:{cfg.get('precision', 'no')}",
        f"link:{cfg.link.get('kind', '?')}",
        f"seed:{seed}",
    ]
    # only tag an intervention when it is actually on, so filtering by tag in the
    # wandb UI selects exactly the variant arms and never the matched controls
    tags.append(f"kwta:{kwta}" if kwta else "kwta:off")
    tags.append(f"gate:{cfg.get('gate_input', 'magnitude')}+{cfg.get('gate_norm', 'sigmoid')}")
    tags.append(f"J:{j}" + (f"+ent{cfg.get('head_entropy_coef', 0.0)}" if j > 1 else ""))
    if kwta is None and j == 1 and cfg.get("gate_input", "magnitude") == "magnitude" \
            and cfg.get("gate_norm", "sigmoid") == "sigmoid":
        tags.append("upstream-control")

    gate = {"s2": "step2-kwta", "s3": "step3-gate", "s4": "step4-union"}.get(
        arm.split("_")[0], "other"
    )
    if arm.startswith("s34"):
        gate = "step3+4-control"

    return {
        "name": f"{arm}/s{seed}",
        "group": arm,
        "job_type": gate,
        "tags": tags,
        "notes": (
            f"{pred} D={cfg.get('embed_dim')} {cfg.link.get('kind','?')} "
            f"reg_weight={cfg.get('reg_weight')} mup_lr={cfg.training.get('mup_lr')} "
            f"precision={cfg.get('precision','no')} | kwta_k={kwta} "
            f"gate=({cfg.get('gate_input','magnitude')},{cfg.get('gate_norm','sigmoid')}) "
            f"n_heads={j} head_entropy_coef={cfg.get('head_entropy_coef',0.0)}"
        ),
    }


# --- diagnostic statistics (pure functions, tested in tests/test_live_diagnostics.py) ---

def soft_jaccard(a, b, eps=1e-8):
    """J_S(a,b) = sum(min(a,b)) / sum(max(a,b)) over the last axis. Needs a,b >= 0.

    The project's core statistic: S = 1 - J_S. Mirrors
    analysis/predictive_jaccard.py's numpy version so the live curve and the
    offline evaluation measure the same thing.
    """
    num = torch.minimum(a, b).sum(-1)
    den = torch.maximum(a, b).sum(-1).clamp_min(eps)
    return num / den


def support_churn(a, b):
    """Fraction of units whose BINARY support flips between a and b, over the last axis.

    The hard-support companion to 1 - J_S: J_S also moves when the surviving
    magnitudes change, so a large S with near-zero churn means "magnitudes wrong",
    not "support reorganised".
    """
    return ((a != 0) != (b != 0)).to(torch.float32).mean(-1)


def participation_ratio(x):
    """Effective dimension of a code matrix x (N, D), as the participation ratio
    tr(C)^2 / ||C||_F^2 of its covariance C.

    Equal to the participation ratio of C's eigenvalue spectrum, but with no
    eigendecomposition: 1 when the code is rank-1 (collapsed), D when it is white.
    This is the quantity that distinguishes "sparse and using many units" from
    "sparse because it collapsed onto a few".
    """
    x = x.to(torch.float32)
    x = x - x.mean(0, keepdim=True)
    c = (x.T @ x) / max(x.shape[0] - 1, 1)
    tr = torch.diagonal(c).sum()
    return tr * tr / (c * c).sum().clamp_min(1e-20)


def dead_unit_fraction(counts):
    """Fraction of units with zero activation count over an accumulation window.

    Takes counts rather than a batch on purpose: at rho ~ 0.5 no unit is dead
    within a single 64-sample batch, so a per-batch answer is always 0 and the
    metric would be worthless.
    """
    return (counts == 0).to(torch.float32).mean()


def swd_per_projection(z, target, proj):
    """Per-projection squared 1-D W2 contributions to models.infojepa_modules.swd.

    Same construction as swd() -- random unit directions, sort along the SAMPLE
    axis 0, mean squared quantile gap -- but returned per projection (shape
    (num_projections,)) instead of averaged, so a large reg_loss can be attributed
    to a few directions rather than the whole distribution.
    """
    pz = torch.sort(z @ proj.T, dim=0).values
    pt = torch.sort(target @ proj.T, dim=0).values
    diff = (pz - pt) ** 2
    return diff.mean(dim=tuple(range(diff.dim() - 1)))


def rdmreg_target_density(link_kind, target_p, mu, kwta_k=None, embed_dim=None):
    """Expected nonzero fraction of RDMReg's target h(GN_p + mu).

    Plotted against the code's measured l0_frac: a large persistent gap is exactly
    the objective conflict that mu-matching exists to remove (k-WTA pins L0 at k/D
    while an unmatched SWD target pulls toward 0.5). Returns None for a GN_p whose
    CDF is not elementary, so the caller can skip the metric instead of guessing.
    """
    if link_kind not in ("relu", "reprelu"):
        density = 1.0  # identity link: GN_p is continuous, nothing is exactly zero
    else:
        sigma = gng_unit_sigma(float(target_p))
        x = float(mu) / sigma
        if float(target_p) == 1.0:  # Laplace survival function
            density = 0.5 * math.exp(x) if x <= 0 else 1.0 - 0.5 * math.exp(-x)
        elif float(target_p) == 2.0:  # standard normal survival function
            density = 0.5 * math.erfc(-x / math.sqrt(2.0))
        else:
            return None
    if kwta_k is not None and embed_dim:
        # k-WTA marks exactly k positions, so it can only ever reduce the density
        density = min(density, int(kwta_k) / int(embed_dim))
    return density


def _wandb_histogram(x, cap):
    """wandb.Histogram from a tensor, finite-filtered and size-capped.

    Non-finite values make wandb.Histogram raise, which would turn a diagnostic
    into a crashed run, and an uncapped payload would ship megabytes per log.
    """
    a = x.detach().flatten().to(torch.float32).cpu().numpy()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return wandb.Histogram(a[:cap] if a.size > cap else a)


def plt_close(fig):
    """Close a matplotlib figure without importing pyplot at module scope.

    train.py is dataloader-bound and imports matplotlib nowhere else; keeping the
    import inside the call means the training fast path never pays for it.
    """
    from analysis import panels as P

    P.plt.close(fig)


@functools.lru_cache(maxsize=4)
def _lut(name):
    """256x3 uint8 lookup table for a design-system colormap.

    Built ONCE per colormap and then applied by array indexing, which is why this
    can follow the campaign's colour system without breaking the no-figures rule
    below: a LUT index on a small 2-D array is free, whereas rendering a matplotlib
    figure per log interval is not.
    """
    from analysis import panels as P  # local: keeps import cost off the fast path

    return (P.plt.get_cmap(name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def _wandb_image(arr, caption, diverging=False, n_classes=None):
    """wandb.Image from a 2D tensor, mapped through the campaign's colormap.

    Still figure-free: this node is dataloader-bound, so rendering matplotlib
    FIGURES inside the train loop would compete directly with feeding the GPU. A
    colormap LUT is an array index and costs nothing, and it means a live panel and
    its offline counterpart in analysis/figures.py read the same way.

    `diverging` maps through RdBu_r on limits centred exactly at zero, because a
    signed quantity on a sequential ramp hides its sign and an auto-scaled diverging
    map puts white somewhere other than 0. Otherwise min-max, with the caption
    carrying the absolute range that normalisation discards.
    """
    from analysis import panels as P

    a = arr.detach().to(torch.float32).cpu().numpy()
    if n_classes:
        # a head index is a CLASS, not a magnitude: a continuous ramp on it invents
        # an ordering between "head 1" and "head 3" that the raster does not have
        pal = (np.asarray(P.head_colors(n_classes))[:, :3] * 255).astype(np.uint8)
        idx = np.clip(np.nan_to_num(a), 0, n_classes - 1).astype(np.int64)
        return wandb.Image(pal[idx], caption=f"{caption}  ({n_classes} classes)")
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if diverging:
        m = max(abs(lo), abs(hi), 1e-12)
        lo, hi = -m, m
    img = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)
    idx = (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255).astype(np.uint8)
    return wandb.Image(
        _lut(P.DIV if diverging else P.SEQ)[idx],
        caption=f"{caption}  range [{lo:.4g}, {hi:.4g}]",
    )


def _l2(tensors):
    """L2 norm over a list of tensors, with ONE host sync instead of one per tensor.

    A ViT has ~150 parameter tensors; a float() per tensor would stall the pipeline
    150 times every log interval on a node that is already the bottleneck.
    """
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    norms = torch.stack([t.detach().to(torch.float32).norm() for t in tensors])
    return float(torch.linalg.vector_norm(norms))


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        # Campaign precision. Upstream ran pure fp32 (no mixed_precision argument),
        # and "no" reproduces that exactly, so the default is unchanged. bf16 was
        # chosen for this campaign from the measured probe wall-clock: 30,965
        # batches/epoch at 2.19 batch/s is 3.9h per epoch in fp32. It MUST be set
        # identically on control and variant arms, since it perturbs numerics.
        precision = self.cfg.get("precision", "no")
        assert precision in ("no", "fp16", "bf16"), f"precision {precision} not supported"
        self.accelerator = Accelerator(log_with="wandb", mixed_precision=precision)
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        # preemption-safe resume state (4h wall limit): batch_idx counts batches
        # already completed in the current epoch, _pending_rng carries the RNG
        # stream across a restart so the resumed epoch continues the same draws.
        self.batch_idx = 0
        self._pending_rng = None
        self._preempted = False
        self._save_every_sec = 60.0 * self.cfg.training.get("save_every_x_min", 20)
        self._last_ckpt_time = time.time()
        for _sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(_sig, self._on_preempt)

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            _proj = self.cfg.get("wandb_project", None) or f"InfoJEPA_train_{self.cfg.env.name}"
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                _proj = f"{_proj}_debug"
            meta = wandb_identity(cfg, cfg_dict["saved_folder"].split("outputs/")[-1])
            self.wandb_run = wandb.init(
                project=_proj,
                config=wandb_dict,
                id=wandb_run_id,
                resume="allow",
                **meta,
            )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            log.info(f"wandb: {meta['name']}  group={meta['group']}  tags={meta['tags']}")
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")
        # used as the wandb x-axis denominator and for the hours/epoch estimate
        self._n_train_batches = len(self.dataloaders["train"])
        self._last_rate_t, self._last_rate_i = None, 0

        # --- live-diagnostics state ---------------------------------------------
        # Per-unit activity has to be accumulated across batches, not measured
        # within one: at rho ~ 0.5 every unit fires somewhere in a 256-sample batch,
        # so a per-batch dead-unit count is identically zero and says nothing.
        self._unit_active = None      # (D,) nonzero counts over the current window
        self._unit_sum = None         # (D,) activation sums over the current window
        self._unit_n = 0              # samples in the current window
        self._unit_win = (None, None, 0)
        # dataloader-wait accounting: the gap between one iteration's body ending
        # and the next one starting IS the time spent inside the loader.
        self._t_body_end = None
        self._data_wait = 0.0
        self._data_span_t0 = None
        self._t_proc_start = time.time()
        self._diag_fail = {}
        # tier-3 panel state: pre-binned histogram rows, not raw samples, so
        # showing the code's distribution EVOLVING costs a few KB rather than
        # retaining every activation seen over a 4h window
        self._panel_edges = None
        self._panel_rows, self._panel_tags, self._panel_heads = [], [], []
        self._panel_phase = []
        self._panel_i = 0
        # Diagnostics that need random draws use their OWN generator. Sampling an
        # SWD probe target from the global stream would advance the same RNG
        # RDMReg draws its target from, so the trained model would silently depend
        # on the logging interval.
        self._diag_gen = torch.Generator(device=self.device)
        self._diag_gen.manual_seed(int(self.cfg.training.seed) + 9973)
        _link_cfg = self.cfg.get("link", None)
        self._link_kind = (_link_cfg.get("kind", "identity") if _link_cfg else "identity")
        # J_S is only defined on non-negative codes, so every Jaccard metric is
        # gated on a rectified link rather than silently reporting garbage on the
        # dense (identity-link) arms.
        self._rectified = self._link_kind in ("relu", "reprelu")
        self._target_density = None

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.link = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"]
            if self.train_decoder and self.cfg.has_decoder
            else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]
        # init_optimizers() builds one AdamW over action+proprio params, so its Adam
        # moments must travel with the checkpoint too. Omitting it restarts those
        # moments at zero on every resume, which puts a visible step in the loss
        # curve at each 4h boundary and makes seeds preempted a different number of
        # times non-comparable -- the exact failure the epoch-count fix rules out.
        self._keys_to_save += (
            ["action_encoder_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )

        link_cfg = self.cfg.get("link", None)
        if link_cfg is not None and link_cfg.get("_target_", None) is not None:
            self._keys_to_save += ["link"]

        self.init_models()
        self.init_optimizers()
        # after init_models, which is where a k-WTA arm rewrites cfg.mu to the
        # density-matched value the target is actually built from
        self._target_density = rdmreg_target_density(
            self._link_kind,
            self.cfg.get("target_p", 2.0),
            self.cfg.get("mu", 0.0),
            self.cfg.get("kwta_k", None),
            self.cfg.get("embed_dim", None),
        )
        # must run after both: state_dicts need constructed targets
        self._maybe_resume()

        self.epoch_log = OrderedDict()

    def _on_preempt(self, signum, frame):
        """SIGUSR1 (sent 300s before the wall limit) / SIGTERM -> checkpoint and exit."""
        log.warning(f"Received signal {signum}; will checkpoint and exit at next batch.")
        self._preempted = True

    def _should_checkpoint(self):
        if self._preempted:
            return True
        if self._save_every_sec <= 0:
            return False
        return (time.time() - self._last_ckpt_time) >= self._save_every_sec

    def _rng_state(self):
        return {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def _load_rng_state(self, st):
        torch.set_rng_state(st["torch"].cpu() if torch.is_tensor(st["torch"]) else st["torch"])
        if st.get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(st["cuda"])
            except (RuntimeError, ValueError) as e:  # device-count change across restarts
                log.warning(f"Could not restore CUDA RNG state: {e}")
        np.random.set_state(st["numpy"])
        random.setstate(st["python"])

    def save_ckpt(self, epoch_end=False):
        """Write ``model_latest.pth`` (and ``model_<epoch>.pth`` at epoch end).

        Stores ``state_dict``s rather than pickled modules so a checkpoint stays
        loadable across edits to the model classes, and writes via a temp file +
        os.replace so a kill mid-write cannot corrupt the resume point.
        """
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {
                "epoch": self.epoch,
                "batch_idx": self.batch_idx,
                "rng": self._rng_state(),
            }
            for k in self._keys_to_save:
                if k == "epoch":
                    continue
                obj = self.__dict__.get(k)
                if obj is None:
                    continue
                ckpt[k] = self.accelerator.unwrap_model(obj).state_dict()
            tmp = "checkpoints/.model_latest.pth.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, "checkpoints/model_latest.pth")
            if epoch_end:
                torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info(
                f"Saved checkpoint (epoch {self.epoch}, batch {self.batch_idx}) to {os.getcwd()}"
            )
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        self._last_ckpt_time = time.time()
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        """Restore state_dicts into already-constructed modules and optimizers."""
        ckpt = torch.load(filename, map_location="cpu")
        self.epoch = ckpt.get("epoch", 0)
        self.batch_idx = ckpt.get("batch_idx", 0)
        self._pending_rng = ckpt.get("rng")
        missing = []
        for k in self._keys_to_save:
            if k == "epoch":
                continue
            obj = self.__dict__.get(k)
            if obj is None:
                continue
            if k not in ckpt:
                missing.append(k)
                continue
            self.accelerator.unwrap_model(obj).load_state_dict(ckpt[k])
        if missing:
            log.warning("Keys not found in ckpt: %s", missing)

    def _maybe_resume(self):
        """Load ``model_latest.pth`` after models and optimizers are constructed."""
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if not model_ckpt.exists():
            return
        self.load_ckpt(model_ckpt)
        log.info(
            f"Resuming from epoch {self.epoch}, batch {self.batch_idx}: {model_ckpt}"
        )
        if self.accelerator.is_main_process:
            self._append_resume_mark()

    def _append_resume_mark(self):
        """Record this 4h window handover, in epoch units, to resume_steps.json.

        Every run in this campaign is chopped every 4h, so the only way to tell a
        lossy resume from real training dynamics is to see where the boundaries
        were: a step in the loss AT a marker is an infra bug, the same step
        elsewhere is science. analysis/figures.py draws these as vertical lines.
        """
        p = Path(self.cfg.saved_folder) / "resume_steps.json"
        try:
            marks = json.loads(p.read_text()) if p.exists() else []
            if not isinstance(marks, list):
                marks = []
        except (ValueError, OSError) as e:
            log.warning(f"could not read {p}, starting a fresh marker list: {e}")
            marks = []
        # batch_idx == 0 means the epoch completed, so the boundary sits exactly at
        # that epoch on the x-axis; mid-epoch it is a fraction of the way through.
        pos = (
            float(self.epoch)
            if self.batch_idx == 0
            else self.epoch - 1 + self.batch_idx / max(self._n_train_batches, 1)
        )
        marks.append(round(pos, 5))
        try:
            p.write_text(json.dumps(marks))
        except OSError as e:
            log.warning(f"could not write {p}: {e}")

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        action_conditioning = self.cfg.get("action_conditioning", "concat")
        if self.cfg.has_predictor:
            if self.predictor is None:
                if action_conditioning == "adaln":
                    assert action_emb_dim == self.encoder.emb_dim, (
                        f"adaln requires action_emb_dim ({action_emb_dim}) == "
                        f"encoder.emb_dim ({self.encoder.emb_dim})"
                    )
                    self.predictor = hydra.utils.instantiate(
                        self.cfg.predictor,
                        num_frames=self.cfg.num_hist,
                        num_patches=self.encoder.num_patches,
                        input_dim=self.encoder.emb_dim,
                        hidden_dim=self.encoder.emb_dim,
                        output_dim=self.encoder.emb_dim,
                    )
                else:
                    self.predictor = hydra.utils.instantiate(
                        self.cfg.predictor,
                        num_patches=num_patches,
                        num_frames=self.cfg.num_hist,
                        dim=self.encoder.emb_dim
                        + (
                            proprio_emb_dim * self.cfg.num_proprio_repeat
                            + action_emb_dim * self.cfg.num_action_repeat
                        )
                        * (self.cfg.concat_dim),
                    )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )

        # Step 2 (k-WTA): match RDMReg's target density to k/D before the regularizer is
        # built, since rdmreg.yaml interpolates ${mu}. For the rectified Laplace target
        # (target_p=1, scale sigma) P(GN_1 + mu > 0) = 0.5*exp(mu/sigma), so setting
        # mu = sigma*ln(2k/D) makes the target density exactly k/D. Without it the W2
        # term would pull toward 0.5*D while k-WTA pins L0 at k, and any degradation
        # would be objective conflict rather than an effect of sparsity.
        kwta_k = self.cfg.get("kwta_k", None)
        if kwta_k is not None:
            from models.infojepa_modules import gng_unit_sigma

            sigma = gng_unit_sigma(float(self.cfg.target_p))
            matched_mu = sigma * math.log(2.0 * int(kwta_k) / int(self.cfg.embed_dim))
            with open_dict(self.cfg):
                self.cfg.mu = matched_mu
            log.info(
                f"k-WTA k={kwta_k}/D={self.cfg.embed_dim}: mu matched to "
                f"{matched_mu:.6f} (sigma={sigma:.6f}, target density k/D="
                f"{int(kwta_k) / int(self.cfg.embed_dim):.4f})"
            )

        self.regularizer = None
        reg_cfg = self.cfg.get("regularizer", None)
        if reg_cfg is not None and reg_cfg.get("_target_", None) is not None:
            self.regularizer = hydra.utils.instantiate(reg_cfg).to(self.device)
            log.info(f"Regularizer: {type(self.regularizer).__name__}")

        link_cfg = self.cfg.get("link", None)
        if self.link is None and link_cfg is not None and link_cfg.get("_target_", None) is not None:
            self.link = hydra.utils.instantiate(link_cfg).to(self.device)
            log.info(f"Link: {getattr(self.link, 'kind', None)}")

        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            action_conditioning=action_conditioning,
            regularizer=self.regularizer,
            reg_weight=self.cfg.get("reg_weight", 0.0),
            detach_target=self.cfg.get("detach_target", True),
            link=self.link,
            lamb_var=self.cfg.get("lamb_var", 0.0),
            lamb_cov=self.cfg.get("lamb_cov", 0.0),
            var_space=self.cfg.get("var_space", "u"),
            var_gamma=self.cfg.get("var_gamma", 1.0),
            n_heads=self.cfg.get("n_heads", 1),
            head_entropy_coef=self.cfg.get("head_entropy_coef", 0.0),
            burst_tau=self.cfg.get("burst_tau", 0.5),
        )

        if self.cfg.get("mup", False) and not model_ckpt.exists():
            from models.mup import mup_init_
            if self.train_encoder:
                mup_init_(self.encoder, tag="encoder")
            if self.predictor is not None:
                mup_init_(self.predictor, tag="predictor")
            if self.decoder is not None and self.train_decoder:
                mup_init_(self.decoder, tag="decoder")
            log.info("Applied muP init to trained scratch modules "
                     f"(encoder={self.train_encoder}, predictor={self.predictor is not None}, "
                     f"decoder={self.decoder is not None and self.train_decoder}).")

    def init_optimizers(self):
        mup = self.cfg.get("mup", False)
        if mup:
            from models.mup import mup_param_groups
            mup_lr = self.cfg.training.get("mup_lr", self.cfg.training.predictor_lr)
            mup_wd = self.cfg.training.get("mup_weight_decay", 0.01)
            mup_bw = self.cfg.training.get("mup_base_width", self.cfg.get("embed_dim", 384))

        if self.train_encoder:
            if mup:
                self.encoder_optimizer = torch.optim.AdamW(
                    mup_param_groups(self.encoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="encoder"),
                    lr=mup_lr, eps=1e-15,
                )
            else:
                self.encoder_optimizer = torch.optim.Adam(
                    self.encoder.parameters(),
                    lr=self.cfg.training.encoder_lr,
                )
            self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        else:
            self.encoder_optimizer = None
        if self.cfg.has_predictor:
            if mup:
                self.predictor_optimizer = torch.optim.AdamW(
                    mup_param_groups(self.predictor, mup_lr, mup_bw, weight_decay=mup_wd, tag="predictor"),
                    lr=mup_lr, eps=1e-15,
                )
            else:
                self.predictor_optimizer = torch.optim.AdamW(
                    self.predictor.parameters(),
                    lr=self.cfg.training.predictor_lr,
                )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            if mup:
                act_groups = mup_param_groups(self.action_encoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="action_encoder") \
                    + mup_param_groups(self.proprio_encoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="proprio_encoder")
                self.action_encoder_optimizer = torch.optim.AdamW(
                    act_groups, lr=mup_lr, eps=1e-15,
                )
            else:
                self.action_encoder_optimizer = torch.optim.AdamW(
                    itertools.chain(
                        self.action_encoder.parameters(), self.proprio_encoder.parameters()
                    ),
                    lr=self.cfg.training.action_encoder_lr,
                )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            if mup:
                self.decoder_optimizer = torch.optim.AdamW(
                    mup_param_groups(self.decoder, mup_lr, mup_bw, weight_decay=mup_wd, tag="decoder"),
                    lr=mup_lr, eps=1e-15,
                )
            else:
                self.decoder_optimizer = torch.optim.Adam(
                    self.decoder.parameters(), lr=self.cfg.training.decoder_lr
                )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        # batch_idx == 0 means epoch self.epoch finished (run() clears it only after
        # val), so start the next one; otherwise re-enter self.epoch and skip forward.
        # Getting this wrong silently drops the tail of the interrupted epoch.
        init_epoch = self.epoch + 1 if self.batch_idx == 0 else self.epoch
        # bound by total_epochs, NOT init_epoch + total_epochs: the latter runs
        # total_epochs MORE epochs after every resume, so a run preempted n times
        # would train for (n+1)*total_epochs and seeds would not be comparable.
        for epoch in range(init_epoch, self.total_epochs + 1):
            self.epoch = epoch
            _t0 = time.time()
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            # logged because training here is dataloader-bound: an arm co-located
            # with other jobs on one node runs several times slower, which looks
            # nothing like divergence in the loss curve but ruins the ETA
            self.logs_update({"epoch_seconds": [time.time() - _t0]})
            self.logs_flash(step=self.epoch)
            self.batch_idx = 0  # epoch fully complete, including val
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt(epoch_end=True)
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

        # sentinel for scripts/submit_until_done.sh: present only after the full
        # epoch budget is complete, so a preempted exit(0) is not mistaken for done.
        if self.accelerator.is_main_process:
            Path(self.cfg.saved_folder, "DONE").write_text(
                f"epochs={self.total_epochs}\n"
            )
            log.info("Training complete; wrote DONE sentinel.")

    def _checkpoint_and_exit(self):
        """Save and leave with status 0 so the resubmit wrapper can chain a new job."""
        self.save_ckpt()
        if getattr(self, "wandb_run", None) is not None:
            self.wandb_run.finish()  # flush; re-opened by id with resume="allow"
        log.warning(
            f"Exiting after preemption checkpoint at epoch {self.epoch}, batch {self.batch_idx}."
        )
        sys.exit(0)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        # Batches already done in this epoch before a preemption. Skipping forward is
        # exact because the loader is shuffle=False (order fixed by TrajSlicerDataset).
        skip_until = self.batch_idx
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            if i < skip_until:
                continue
            if self._pending_rng is not None:
                # restore after skipping, so the first trained batch continues the
                # exact RNG stream the pre-preemption run would have used
                self._load_rng_state(self._pending_rng)
                self._pending_rng = None
            self._mark_iter_start()
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs_linked(obs)  # linked, to match z_obs_out
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)
            self._log_live(i, loss_components, z_out)

            self.batch_idx = i + 1
            if self._should_checkpoint():
                if self._preempted:
                    self._checkpoint_and_exit()
                self.save_ckpt()

        # batch_idx is left at len(loader) so that a preemption during val() resumes
        # into this same epoch, skips every train batch, and redoes only val.
        # run() resets it to 0 once the epoch is fully complete.

    def _wandb_step(self, i):
        """Global batch index, so the x-axis is continuous across epochs AND across
        the 4h window handovers (wandb rejects a step that goes backwards)."""
        return (self.epoch - 1) * self._n_train_batches + i

    def _mark_iter_start(self):
        """Charge the gap since the previous iteration's body to dataloader wait.

        Training here is dataloader-bound, so "is the GPU starved?" is the single
        most actionable number on the run page, and it is not visible in
        batches_per_sec alone (a slow step and a starved step look identical).
        """
        now = time.time()
        if self._t_body_end is not None:
            self._data_wait += now - self._t_body_end
        if self._data_span_t0 is None:
            self._data_span_t0 = now

    def _log_live(self, i, comps, z_out):
        """Per-batch logging, so a run has curves within minutes instead of hours.

        Epoch-level logging alone left every panel empty for the whole first epoch
        (~2h at 30k batches), which is indistinguishable from a broken run. Three
        tiers, because the costs differ by orders of magnitude on a node whose CPUs
        are the bottleneck:

          log_every_x_batch   (~50)     scalars: GPU reductions, no rendering
          diag_every_x_batch  (~2000)   distributions: host copies + wandb.Histogram
          heavy_every_x_batch (~10000)  images: D x D maps, rasters, code heatmaps

        Every block is individually guarded, so one bad diagnostic degrades to a
        warning plus a diag/blocks_failed counter instead of killing a 4h window.
        """
        if getattr(self, "wandb_run", None) is None or not self.accelerator.is_main_process:
            return
        self._accumulate_unit_stats()  # every batch: the window is the point
        every = int(self.cfg.training.get("log_every_x_batch", 50))
        if every <= 0 or i % every:
            self._t_body_end = time.time()
            return
        step = self._wandb_step(i)
        now = time.time()
        rate = None
        if self._last_rate_t is not None and now > self._last_rate_t:
            rate = (i - self._last_rate_i) / (now - self._last_rate_t)
        self._last_rate_t, self._last_rate_i = now, i

        payload = {wandb_key(k): v[0] for k, v in comps.items()}
        payload["epoch"] = self.epoch
        payload["progress/epoch_frac"] = self.epoch - 1 + i / max(self._n_train_batches, 1)
        payload["progress/global_batch"] = step
        payload["progress/window_seconds"] = now - self._t_proc_start
        if rate:
            payload["perf/batches_per_sec"] = rate
            payload["perf/hours_per_epoch"] = self._n_train_batches / rate / 3600.0
            left = max(self.total_epochs * self._n_train_batches - step, 0)
            payload["perf/eta_hours"] = left / rate / 3600.0
        if self._data_span_t0 is not None:
            span = now - self._data_span_t0
            if span > 0:
                payload["perf/data_wait_frac"] = min(self._data_wait / span, 1.0)
        self._data_wait, self._data_span_t0 = 0.0, now
        gn = self._grad_norm()
        if gn is not None:
            payload["opt/grad_norm"] = gn
        if self.cfg.has_predictor:
            payload["opt/lr"] = self.predictor_optimizer.param_groups[0]["lr"]

        # snapshot + reset the per-unit window once, so every tier below sees the
        # same window rather than tier 1 seeing whatever tier 0 left behind
        self._unit_win = self._unit_window(reset=True)
        tens = self._diag_tensors()

        self._safe("perf_mem", self._mem_diagnostics, payload)
        self._safe("opt_norms", self._opt_diagnostics, payload)
        self._safe("sparsity", lambda: self._sparsity_diagnostics(tens), payload)
        self._safe("jaccard", lambda: self._jaccard_diagnostics(tens), payload)
        self._safe("err", lambda: self._error_diagnostics(tens), payload)
        self._safe("reg", lambda: self._reg_diagnostics(comps), payload)
        self._safe("gate", lambda: self._gate_diagnostics(tens), payload)
        self._safe("heads", lambda: self._head_diagnostics(tens), payload)

        diag_every = int(self.cfg.training.get("diag_every_x_batch", 2000))
        if diag_every > 0 and i % diag_every == 0:
            self._safe("dist", lambda: self._tensor_diagnostics(tens), payload)
        heavy_every = int(self.cfg.training.get("heavy_every_x_batch", 10000))
        if heavy_every > 0 and i % heavy_every == 0:
            self._safe("fig", lambda: self._image_diagnostics(tens), payload)
            self._safe("panel", lambda: self._panel_diagnostics(tens), payload)
        video_every = int(self.cfg.training.get("video_every_x_batch", 0))
        if video_every > 0 and i % video_every == 0:
            self._safe("video", self._env_video_diagnostics, payload)

        # (density, reg loss) history for the live phase plane. Scalars only, capped,
        # so carrying the trajectory costs a few hundred floats rather than tensors.
        rho = comps.get("train_l0_frac", [None])[0]
        rl = comps.get("train_reg_loss", [None])[0]
        if rho is not None and rl is not None:
            self._panel_phase.append((float(rho), float(rl)))
            if len(self._panel_phase) > 400:
                self._panel_phase.pop(0)

        payload["diag/blocks_failed"] = sum(self._diag_fail.values())
        try:
            self.wandb_run.log(payload, step=step)
        except Exception as e:  # a wandb hiccup must not end a 4h window
            log.warning(f"wandb.log failed at step {step}: {e}")
        # epoch aggregates for the same statistics, so train and val are comparable
        # on the per-epoch axis that analysis/figures.py reads
        agg = {}
        self._safe("epoch_agg", lambda: self._support_logs("train"), agg)
        if agg:
            self.logs_update(agg)
        self._t_body_end = time.time()

    def _safe(self, name, fn, out):
        """Run a diagnostic block; on failure warn and count it, never raise.

        Bare Exception is deliberate. The alternative is a 4h GPU window lost to a
        shape bug in a histogram, and 27 runs makes that a certainty rather than a
        risk. Failures are counted and published as diag/blocks_failed so a block
        that quietly stopped working is visible on the run page.
        """
        try:
            got = fn()
            if got:
                out.update(got)
        except Exception as e:
            self._diag_fail[name] = self._diag_fail.get(name, 0) + 1
            log.warning(
                "diagnostic block '%s' skipped (%d failures so far): %s",
                name, self._diag_fail[name], e,
            )

    def _grad_norm(self):
        if not self.cfg.has_predictor or self.predictor is None:
            return None
        return _l2([p.grad for p in self.predictor.parameters()])

    @torch.no_grad()
    def _accumulate_unit_stats(self):
        """Fold this batch into the per-unit activity window (GPU only, no sync)."""
        d = getattr(self.model, "_diag", None)
        if not d or d.get("z") is None:
            return
        zf = d["z"].detach().to(torch.float32).flatten(0, -2)  # (N, D)
        a, s = (zf != 0).sum(0), zf.sum(0)
        if self._unit_active is None or self._unit_active.shape != a.shape:
            self._unit_active, self._unit_sum, self._unit_n = a, s, zf.shape[0]
        else:
            self._unit_active = self._unit_active + a
            self._unit_sum = self._unit_sum + s
            self._unit_n += zf.shape[0]

    def _unit_window(self, reset=False):
        """(activation frequency, mean activation, n samples) over the window."""
        if self._unit_active is None or self._unit_n == 0:
            return None, None, 0
        n = self._unit_n
        freq = self._unit_active.to(torch.float32) / n
        mean_act = self._unit_sum / n
        if reset:
            self._unit_active, self._unit_sum, self._unit_n = None, None, 0
        return freq, mean_act, n

    @torch.no_grad()
    def _diag_tensors(self):
        """This batch's (z, u, z_pred, target) in float32, or None.

        Stashed by VWorldModel._forward_adaln: the encoder code and the loss's own
        target are not on the forward's return path, and re-encoding to get them
        would cost a second ViT pass per logged batch. Absent on the concat /
        DINO-WM path, so every caller must tolerate None.
        """
        d = getattr(self.model, "_diag", None)
        if not d or d.get("z") is None or d.get("z_pred") is None:
            return None
        return tuple(
            None if d.get(k) is None else d[k].detach().to(torch.float32)
            for k in ("z", "u", "z_pred", "target")
        )

    def _mem_diagnostics(self):
        if not torch.cuda.is_available():
            return {}
        out = {
            "perf/gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "perf/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "perf/gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        torch.cuda.reset_peak_memory_stats()  # peak per interval, not since boot
        return out

    def _opt_diagnostics(self):
        """Per-module gradient and weight norms, plus the update-to-weight ratio.

        One aggregate grad norm cannot distinguish "the encoder is not learning"
        from "the predictor is not learning", which is the first question when an
        arm flatlines. The ratio is the standard scale-free check that the step
        size is sane (~1e-3 is healthy; 1e-6 is a frozen module, 1e-1 diverging).
        """
        out = {}
        mods = {
            "encoder": self.encoder,
            "predictor": self.predictor,
            "action_encoder": self.action_encoder,
        }
        for name, m in mods.items():
            if m is None:
                continue
            params = list(m.parameters())
            g = _l2([p.grad for p in params])
            w = _l2(params)
            if g is not None:
                out[f"opt/grad_norm_{name}"] = g
            if w is not None:
                out[f"opt/weight_norm_{name}"] = w
        gp, wp = out.get("opt/grad_norm_predictor"), out.get("opt/weight_norm_predictor")
        if gp is not None and wp:
            lr = self.predictor_optimizer.param_groups[0]["lr"] if self.cfg.has_predictor else 0.0
            out["opt/update_to_weight"] = lr * gp / wp
        return out

    @torch.no_grad()
    def _sparsity_diagnostics(self, tens):
        """How sparse the code is, how many units are alive, and how wide it spans.

        l0_frac alone is ambiguous: the same density can come from every unit firing
        half the time or from half the units firing always and the rest being dead.
        Dead-unit count and effective dimension separate those two, and only the
        first is compatible with the Pi-WM claim.
        """
        if tens is None:
            return {}
        z, _u, z_pred, _t = tens
        out = {}
        zf = rearrange(z, "b t p d -> (b t p) d")
        pf = rearrange(z_pred, "b t p d -> (b t p) d")
        l0 = (zf != 0).to(torch.float32).sum(-1)
        out["sparsity/l0_frac_pred"] = float((pf != 0).to(torch.float32).mean())
        out["sparsity/l0_std_across_samples"] = float(l0.std())
        out["sparsity/effective_dim"] = float(participation_ratio(zf))
        out["sparsity/effective_dim_pred"] = float(participation_ratio(pf))

        freq, _mean_act, n_win = self._unit_win
        if freq is not None:
            out["sparsity/dead_unit_frac"] = float(dead_unit_fraction(freq))
            out["sparsity/dead_unit_count"] = float((freq == 0).sum())
            out["sparsity/unit_freq_max"] = float(freq.max())
            out["sparsity/unit_freq_min"] = float(freq.min())
            out["sparsity/unit_window_samples"] = float(n_win)

        nz = zf[zf != 0]
        if nz.numel():
            q = torch.quantile(nz, torch.tensor([0.5, 0.9, 0.99], device=nz.device))
            out["dist/z_p50"], out["dist/z_p90"], out["dist/z_p99"] = (float(v) for v in q)
            out["dist/z_max"] = float(nz.max())
        return out

    @torch.no_grad()
    def _jaccard_diagnostics(self, tens):
        """The project's core statistic, live.

        S_model = 1 - J_S(z_hat, z) is what Step 1 evaluates offline; having it on
        the training curve means a run that never develops predictive support
        structure is visible in minutes instead of after a full CEM eval. S_world
        = 1 - J_S(z_t, z_{t+1}) is the observed-change baseline it has to beat, and
        support_churn separates support reorganisation from magnitude error.
        """
        if tens is None or not self._rectified:
            return {}
        z, _u, z_pred, target = tens
        out = {}
        if target is not None:
            s_model = (1.0 - soft_jaccard(z_pred, target)).flatten()
            out["jacc/S_model"] = float(s_model.mean())
            out["jacc/S_model_p90"] = float(torch.quantile(s_model, 0.9))
            out["jacc/burst_rate"] = float(
                (s_model > float(self.cfg.get("burst_tau", 0.5))).to(torch.float32).mean()
            )
            out["jacc/churn_model"] = float(support_churn(z_pred, target).mean())
        if z.shape[1] > 1:
            out["jacc/S_world"] = float((1.0 - soft_jaccard(z[:, :-1], z[:, 1:])).mean())
            out["jacc/support_churn"] = float(support_churn(z[:, :-1], z[:, 1:]).mean())
        return out

    @torch.no_grad()
    def _error_diagnostics(self, tens):
        """Prediction error, decomposed the three ways that change the diagnosis.

        By timestep (does the predictor only work at the start of the window), by
        support (is it getting the active units wrong or hallucinating into the
        inactive ones), and relative to the target scale (so arms whose codes have
        different magnitudes are still comparable).
        """
        if tens is None or tens[3] is None:
            return {}
        z, _u, z_pred, target = tens
        out = {}
        diff2 = (z_pred - target) ** 2
        out["err/rel_mse"] = float(diff2.mean() / target.pow(2).mean().clamp_min(1e-12))
        sup = target != 0
        if bool(sup.any()):
            out["err/mse_on_support"] = float(diff2[sup].mean())
        if bool((~sup).any()):
            out["err/mse_off_support"] = float(diff2[~sup].mean())
        out["err/cos_pred_target"] = float(
            torch.nn.functional.cosine_similarity(
                rearrange(z_pred, "b t p d -> (b t p) d"),
                rearrange(target, "b t p d -> (b t p) d"),
                dim=-1,
            ).mean()
        )
        per_t = diff2.mean(dim=(0, 2, 3))
        for ti in range(int(per_t.numel())):
            out[f"err/mse_t{ti}"] = float(per_t[ti])
        return out

    def _reg_diagnostics(self, comps):
        """Pair the regulariser loss with the density it is pulling toward.

        reg_loss on its own cannot say whether the SWD term is doing useful work or
        fighting k-WTA over the density; the gap between the measured l0_frac and
        the analytic target density can. Both are logged at the same step, so the
        relationship is directly plottable.
        """
        out = {}
        if self._target_density is None:
            return out
        out["reg/target_density"] = self._target_density
        l0 = comps.get("train_l0_frac", comps.get("val_l0_frac", [None]))[0]
        if l0 is not None:
            out["reg/density_gap"] = float(l0) - self._target_density
        return out

    def _pred_module(self):
        """The predictor with any DDP wrapper stripped, or None.

        accelerate wraps the predictor, and a DDP wrapper only proxies forward() --
        reaching gates() or W_heads through it raises AttributeError.
        """
        if self.predictor is None:
            return None
        return getattr(self.predictor, "module", self.predictor)

    @torch.no_grad()
    def _gate_diagnostics(self, tens):
        """LTV support-gate health (mode='ltv' only).

        gate_mean is the r*softmax-vs-sigmoid sanity check: ~1.0 for r*softmax,
        ~0.5 for sigmoid. A softmax arm reading ~1/r instead means the r factor
        was lost, which at EPOCHS=2 is indistinguishable from "support gating is
        worse". ltv_u_norm is the engagement precondition: the U factors are
        zero-initialised, so a norm that never leaves ~0 means the LTV mechanism
        never turned on and the arm cannot falsify anything.
        """
        pred = self._pred_module()
        if tens is None or getattr(pred, "mode", None) != "ltv":
            return {}
        z = tens[0]
        g = pred.gates(z[:, : int(self.cfg.num_hist)])
        out = {
            "gate/gate_mean": float(g.mean()),
            "gate/gate_std": float(g.std()),
            "gate/gate_max": float(g.max()),
            "gate/gate_frac_gt_half": float((g > 0.5).to(torch.float32).mean()),
        }
        u = _l2([m.weight for m in (*pred.Ulag, pred.UB)])
        if u is not None:
            out["gate/ltv_u_norm"] = u
        base = _l2([m.weight for m in pred.lags])
        if base:
            out["gate/ltv_u_rel"] = (u or 0.0) / base
        return out

    @torch.no_grad()
    def _head_diagnostics(self, tens):
        """Union-head specialisation (n_heads > 1 only).

        head_gap_head0 is the one that matters: it is how much the min over heads
        actually buys over head 0 alone. If it sits at zero the extra heads are
        decoration and J=4 is numerically J=1, which is the rigged-gate failure
        the plan's p_bar precondition exists to catch.
        """
        h = getattr(self.model, "_diag_heads", None)
        if not h or h.get("per_head") is None:
            return {}
        per_head = h["per_head"].to(torch.float32)  # (J, B, T)
        j = per_head.shape[0]
        lmin = per_head.min(dim=0).values
        out = {
            "heads/head_loss_spread": float((per_head.max(dim=0).values - lmin).mean()),
            "heads/head_gap_head0": float((per_head[0] - lmin).mean()),
        }
        for k in range(j):
            out[f"heads/head_loss_j{k}"] = float(per_head[k].mean())
        j_star = h.get("j_star")
        if j_star is not None:
            usage = torch.nn.functional.one_hot(j_star, j).to(torch.float32).mean(dim=(0, 1))
            out["heads/head_usage_min"] = float(usage.min())
        z_all = h.get("z_all")
        if z_all is not None and tens is not None:
            # per-head specialisation: how far each head moves the code away from
            # its input. Heads that all move it the same distance are not
            # specialising, whatever the usage histogram says.
            z_src = tens[0][:, : z_all.shape[2]]
            d = (z_all.to(torch.float32) - z_src.unsqueeze(0)).flatten(1).pow(2).mean(1).sqrt()
            for k in range(int(d.numel())):
                out[f"heads/head_delta_j{k}"] = float(d[k])
        return out

    @torch.no_grad()
    def _tensor_diagnostics(self, tens):
        """Tier 2: distributions that a mean cannot show. Host copies, so rarer."""
        if tens is None:
            return {}
        z, _u, z_pred, target = tens
        cap = int(self.cfg.training.get("diag_hist_max", 20000))
        out = {}
        zf = rearrange(z, "b t p d -> (b t p) d")
        nz = zf[zf != 0]
        if nz.numel():
            out["dist/z_nonzero_magnitude"] = _wandb_histogram(nz, cap)
        out["dist/z_l0_per_sample"] = _wandb_histogram((zf != 0).to(torch.float32).sum(-1), cap)

        freq, mean_act, _n = self._unit_win
        if freq is not None:
            # which units are alive, as a shape rather than a scalar: a bimodal
            # frequency histogram is the k-WTA signature, a flat one is upstream
            out["dist/unit_activation_freq"] = _wandb_histogram(freq, cap)
            out["dist/unit_mean_activation"] = _wandb_histogram(mean_act, cap)

        if target is not None:
            diff2 = (z_pred - target) ** 2
            out["dist/err_per_feature"] = _wandb_histogram(diff2.mean(dim=(0, 1, 2)), cap)
            if diff2.shape[2] > 1:
                out["dist/err_per_patch"] = _wandb_histogram(diff2.mean(dim=(0, 1, 3)), cap)
            if self._rectified:
                out["dist/S_model_per_sample"] = _wandb_histogram(
                    1.0 - soft_jaccard(z_pred, target), cap
                )
        if self._rectified and z.shape[1] > 1:
            out["dist/support_churn_per_sample"] = _wandb_histogram(
                support_churn(z[:, :-1], z[:, 1:]), cap
            )

        pred = self._pred_module()
        if getattr(pred, "mode", None) == "ltv":
            out["dist/gate_values"] = _wandb_histogram(
                pred.gates(z[:, : int(self.cfg.num_hist)]), cap
            )
        h = getattr(self.model, "_diag_heads", None)
        if h and h.get("per_head") is not None:
            out["dist/head_loss"] = _wandb_histogram(h["per_head"], cap)

        out.update(self._swd_probe(z))

        # support self-similarity within the batch: block structure here is the
        # direct visual evidence for discrete modes
        if self._rectified:
            n = int(self.cfg.training.get("diag_selfsim_n", 96))
            s = zf[: min(n, zf.shape[0])]
            out["fig/support_selfsim"] = _wandb_image(
                soft_jaccard(s[:, None], s[None]), "J_S(z_a, z_b) within one batch"
            )
        return out

    @torch.no_grad()
    def _swd_probe(self, z):
        """Independent low-projection re-estimate of the RDMReg statistic.

        The per-projection breakdown answers a question the scalar cannot: is the
        distribution mismatch spread over all directions, or carried by a handful?
        Uses the private diagnostics generator, never the global stream.
        """
        n_proj = int(self.cfg.training.get("diag_swd_proj", 256))
        if n_proj <= 0:
            return {}
        agg = str(self.cfg.get("agg", "btp"))
        zr = rearrange(z, AGG_PATTERNS[agg])
        base = self._diag_base_sample(zr.shape)
        if base is None:
            return {}
        mu = float(self.cfg.get("mu", 0.0))
        if mu != 0.0:
            base = base + mu
        target = self.link(base) if self.link is not None else base
        proj = torch.randn(
            n_proj, zr.shape[-1], device=zr.device, generator=self._diag_gen
        )
        proj = proj / proj.norm(dim=1, keepdim=True)
        per_proj = swd_per_projection(zr, target, proj)
        cap = int(self.cfg.training.get("diag_hist_max", 20000))
        return {
            "dist/swd_per_projection": _wandb_histogram(per_proj, cap),
            "reg/swd_probe": float(per_proj.mean()),
            "reg/swd_proj_max": float(per_proj.max()),
        }

    def _diag_base_sample(self, shape):
        """GN_p base draw for the SWD probe, from the private generator.

        torch.distributions takes no generator, so the Laplace case is sampled by
        inverse CDF rather than through RDMReg's sampler; anything other than
        p in {1, 2} returns None and the probe is skipped instead of approximated.
        """
        p = float(self.cfg.get("target_p", 2.0))
        sigma = gng_unit_sigma(p)
        if p == 2.0:
            return sigma * torch.randn(shape, device=self.device, generator=self._diag_gen)
        if p == 1.0:
            u = torch.rand(shape, device=self.device, generator=self._diag_gen) - 0.5
            a = u.abs().clamp_max(0.5 - 1e-7)  # |u| == 0.5 would give log(0)
            return -sigma * torch.sign(u) * torch.log1p(-2.0 * a)
        return None

    @torch.no_grad()
    def _panel_diagnostics(self, tens):
        """Real analysis/panels.py figures, live. Tier 3 -- heaviest, rarest.

        This is the arrangement panels.py was written for ("train.py wraps the
        returned Figure in wandb.Image, live during training"), and until now only
        the colour system was wired up: the live page got raw heatmaps while every
        redesigned FORM existed offline only.

        The no-figures rule still holds where it matters. These render on the HEAVY
        interval alone -- once per ~10k batches, roughly twice per 4h window -- so
        the cost is two matplotlib renders per window against ~30k training steps,
        rather than the per-log-interval cost the rule was written to prevent.

        A ring buffer of PRE-BINNED histogram rows is kept rather than raw samples,
        which is exactly the `(edges, matrix)` input `panels.ridgeline` accepts, so
        showing how the code's distribution EVOLVES costs a few KB of state instead
        of retaining every activation seen.
        """
        if tens is None:
            return {}
        from analysis import panels as P

        z, _u, _z_pred, _t = tens
        zf = rearrange(z, "b t p d -> (b t p) d")
        out = {}

        nz = zf[zf != 0].to(torch.float32)
        if nz.numel() > 16:
            v = nz.detach().cpu().numpy()
            if self._panel_edges is None:
                hi = float(np.quantile(v, 0.995))
                self._panel_edges = np.linspace(0.0, hi if hi > 0 else 1.0, 65)
            h, _ = np.histogram(v, bins=self._panel_edges)
            self._panel_rows.append(h.astype(float))
            self._panel_tags.append(f"ep{getattr(self, 'epoch', 1)} #{self._panel_i}")
            if len(self._panel_rows) > 12:  # keep the newest, bounded memory
                self._panel_rows.pop(0)
                self._panel_tags.pop(0)
            out["panel/z_magnitude_ridgeline"] = self._panel_fig(
                P.ridgeline(
                    (self._panel_edges, np.asarray(self._panel_rows)),
                    labels=list(self._panel_tags),
                    title="Surviving activation magnitudes over training",
                    subtitle="a second mode growing near zero = the code is SPARSIFYING, "
                             "not merely shrinking",
                )
            )

        l0 = (zf != 0).to(torch.float32).sum(-1).detach().cpu().numpy()
        if l0.size > 8:
            k = self.cfg.get("kwta_k", None)
            out["panel/l0_ecdf"] = self._panel_fig(
                P.ecdf_overlay(
                    {self._arm_label(): l0},
                    colors={self._arm_label(): P.arm_color(self._arm_label())},
                    vlines=((float(k), f"k-WTA k={int(k)}"),) if k else (),
                    title="Per-sample sparsity, as an ECDF",
                )
            )

        h = getattr(self.model, "_diag_heads", None)
        if h is not None and h.get("j_star") is not None:
            j_star = h["j_star"]
            j = max(int(self.cfg.get("n_heads", 1)), 1)
            if j > 1:
                usage = (
                    torch.nn.functional.one_hot(j_star, j).to(torch.float32).mean(dim=0)
                )  # (T, J)
                u = usage.detach().cpu().numpy()
                self._panel_heads.append(u.mean(axis=0))
                if len(self._panel_heads) > 60:
                    self._panel_heads.pop(0)
                mat = np.asarray(self._panel_heads)  # (T_logged, J)
                if mat.shape[0] >= 2:
                    out["panel/head_usage_stream"] = self._panel_fig(
                        P.head_stream(
                            [(self._arm_label(), np.arange(mat.shape[0], dtype=float), mat)],
                            xlabel="heavy-diagnostic tick",
                            title="Union-head usage over training",
                        )
                    )
        # the project's core statistic as a JOINT density: S_model and S_world are
        # only meaningful against each other, and two separate curves cannot show
        # whether the mass sits above or below the identity line
        if self._rectified and tens[3] is not None and z.shape[1] > 1:
            sm = (1.0 - soft_jaccard(tens[2], tens[3])).flatten().detach().cpu().numpy()
            sw = (1.0 - soft_jaccard(z[:, :-1], z[:, 1:])).flatten().detach().cpu().numpy()
            n = min(sm.size, sw.size)
            if n > 32:
                out["panel/support_change_joint"] = self._panel_fig(
                    P.joint_hexbin(
                        sw[:n], sm[:n],
                        xlabel=r"$S_{world} = 1 - J_S(z_t, z_{t+1})$   (observed change)",
                        ylabel=r"$S_{model} = 1 - J_S(\hat{z}, z)$   (prediction error)",
                        title="Support change: predicted vs observed",
                    )
                )

        # density against regulariser loss, walked over training. A floor is where
        # the trajectory STOPS MOVING, which is a shape and not a scalar.
        if len(self._panel_phase) >= 4:
            ph = np.asarray(self._panel_phase, dtype=float)
            out["panel/reg_phase_plane"] = self._panel_fig(
                P.phase_plane(
                    {self._arm_label(): (ph[:, 0], ph[:, 1])},
                    xlabel=r"code density $\rho$  (l0 fraction)",
                    ylabel="RDMReg loss",
                    title="Where this run is walking, and whether it has stopped",
                    subtitle="hollow = first logged point, filled = latest; colour "
                             "lightens to darkens with training",
                )
            )

        self._panel_i += 1
        return out

    def _arm_label(self):
        """The canonical arm name, so a live panel is the same colour as its PNG.

        Defensive on purpose: this is a LABEL. It must never be the reason the panel
        block raises, because a raise inside _safe() discards every figure in the
        block, not just the one that could not be named.
        """
        from analysis import panels as P

        try:
            return P.canon_arm(str(self.cfg["saved_folder"]).split("outputs/")[-1])
        except Exception:
            return "run"

    def _panel_fig(self, fig):
        """wandb.Image from a Figure, then CLOSE it.

        matplotlib keeps every unclosed figure alive in pyplot's registry; two leaks
        per heavy interval is a slow memory climb across a 4h window and an eventual
        "More than 20 figures have been opened" warning storm.
        """
        img = wandb.Image(fig)
        plt_close(fig)
        return img

    @torch.no_grad()
    def _env_video_diagnostics(self):
        """A real PushT rollout, rendered, paired with the model's error on it.

        The rest of this file is latent-space only: with has_decoder=False there are
        no pixels anywhere in the training signal, so "is the run learning" and "does
        the simulator still look right" are answered by completely separate evidence.
        This block closes that gap -- it resets the env to a held-out trajectory's
        recorded start state, replays that trajectory's actions, and renders.

        The video is paired with `panel/latent_error_timeline` computed on the SAME
        frames, so a spike in predictive error can be looked at rather than inferred.
        Costed deliberately: a handful of episodes on the video interval, which is
        ~1% of a window on a node that is dataloader-bound, not GPU-bound.
        """
        # A compute node has no display, and pygame aborts on display init without
        # this. Set before gym/pygame import, never after: SDL reads it at init.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import gym

        import env as _env_registry  # noqa: F401  registers "pusht" with gym
        from analysis import panels as P

        n_ep = int(self.cfg.training.get("video_episodes", 4))
        max_f = int(self.cfg.training.get("video_max_frames", 50))
        dset = getattr(self, "val_traj_dset", None) or getattr(self, "train_traj_dset", None)
        if dset is None or not len(dset):
            return {}

        out, all_frames, curves = {}, [], {}
        rng = np.random.default_rng(int(self._panel_i))
        for ep in range(n_ep):
            obs, act, state, _ = dset[int(rng.integers(0, len(dset)))]
            if state is None or len(state) < 2:
                continue
            e = gym.make(self.cfg.env.name, *self.cfg.env.args, **self.cfg.env.kwargs)
            try:
                # gym.make returns a TimeLimit wrapper; setting reset_to_state on the
                # wrapper writes an attribute nobody reads and the episode silently
                # starts from a random state instead of the trajectory's own
                e.unwrapped.reset_to_state = np.asarray(state[0])
                e.reset()
                frames = []
                # actions in the dataset are NORMALISED; the env expects raw ones.
                # Replaying normalised actions renders a plausible-looking video of
                # the wrong trajectory, which is worse than no video at all.
                mu = getattr(dset, "action_mean", None)
                sd = getattr(dset, "action_std", None)
                n = min(max_f, len(act))
                for t in range(n):
                    a = act[t].detach().cpu() if torch.is_tensor(act[t]) else torch.as_tensor(act[t])
                    if mu is not None and sd is not None:
                        a = a * torch.as_tensor(sd).cpu() + torch.as_tensor(mu).cpu()
                    a = np.asarray(a, dtype=float).reshape(-1)
                    e.step(a[: e.action_space.shape[0]])
                    frames.append(np.asarray(e.render("rgb_array")))
                if frames:
                    all_frames.append(np.stack(frames))
            finally:
                e.close()

            # the model's error on the very frames just rendered
            if ep == 0:
                curves.update(self._latent_error_on(obs))

        if all_frames:
            # (N, T, H, W, C) -> a single tiled (T, C, H, W) clip; wandb wants CHW
            t = min(f.shape[0] for f in all_frames)
            grid = np.concatenate([f[:t] for f in all_frames], axis=2)  # side by side
            out["video/env_rollout"] = wandb.Video(
                np.transpose(grid, (0, 3, 1, 2)).astype(np.uint8),
                fps=10, format="mp4",
                caption=f"{len(all_frames)} held-out episodes, {t} frames, actions replayed",
            )
        if curves:
            out["panel/latent_error_timeline"] = self._panel_fig(
                P.latent_error_timeline(
                    np.arange(len(next(iter(curves.values()))), dtype=float), curves,
                    xlabel="env-eval frame",
                    title="Predictive error over the rendered episode",
                )
            )
        return out

    @torch.no_grad()
    def _latent_error_on(self, obs):
        """Per-frame observed support change for one rendered trajectory.

        Pairs the video with the model's own view of the same frames: S_world is how
        much the LATENT support moves between consecutive rendered frames, so a spot
        in the clip where the code churns can be looked at rather than inferred.

        `encode_obs_linked` is the linked (post-h) code -- the same quantity every
        other diagnostic uses. Encoding without the link would report a dense code
        and make S_world incomparable to the scalar on the training curve.
        """
        if not self._rectified:
            return {}
        try:
            n = int(self.cfg.training.get("video_max_frames", 50))
            v = obs["visual"][:n].unsqueeze(0).to(self.device)
            pr = obs["proprio"][:n].unsqueeze(0).to(self.device)
            z = self._model_module().encode_obs_linked({"visual": v, "proprio": pr})
            z = z["visual"] if isinstance(z, dict) else z
            if z.dim() == 4:                       # (b, t, p, d) -> (b, t, p*d)
                z = z.flatten(2)
            if z.shape[1] < 2:
                return {}
            sw = (1.0 - soft_jaccard(z[:, :-1], z[:, 1:])).flatten()
            return {r"$S_{world}$  (latent support change)": sw.detach().cpu().numpy()}
        except Exception as e:  # the video is the deliverable; the curve is a bonus
            log.debug(f"latent-error companion unavailable: {e}")
            return {}

    def _model_module(self):
        """The model with any DDP wrapper stripped -- a wrapper only proxies forward."""
        return getattr(self.model, "module", self.model)

    @torch.no_grad()
    def _image_diagnostics(self, tens):
        """Tier 3: 2-D maps. Rarest, because these are the only host-heavy items."""
        if tens is None:
            return {}
        z, _u, z_pred, _t = tens
        out = {}
        rows = int(self.cfg.training.get("diag_img_rows", 64))
        zf = rearrange(z, "b t p d -> (b t p) d")
        pf = rearrange(z_pred, "b t p d -> (b t p) d")

        # the code itself, encoder above prediction: mismatched columns are visible
        # here in a way no scalar summary reproduces
        n = min(rows, zf.shape[0], pf.shape[0])
        out["fig/code_vs_pred"] = _wandb_image(
            torch.cat([zf[:n], pf[:n]], dim=0),
            f"top {n} rows z (encoder), bottom {n} rows z_hat (predictor); cols = units",
            diverging=True,
        )
        # unit co-activation: block structure means units group into modes, which is
        # the Pi-WM thesis stated as a picture
        cap = int(self.cfg.training.get("diag_coact_units", 384))
        s = (zf[:, :cap] != 0).to(torch.float32)
        out["fig/unit_coactivation"] = _wandb_image(
            s.T @ s / max(s.shape[0], 1), "P(unit_i and unit_j co-active)"
        )

        pred = self._pred_module()
        if getattr(pred, "mode", None) == "ltv":
            g = pred.gates(z[:, : int(self.cfg.num_hist)])  # (B,T,P,n_lags+1,r)
            out["fig/gate_heatmap"] = _wandb_image(
                g[: min(rows, g.shape[0]), -1, 0, 0],
                "gate g(z_T) lag 0: rows = samples, cols = r modes",
            )
            out["fig/gate_lag_map"] = _wandb_image(
                g.mean(dim=(0, 2)).flatten(0, 1),
                "mean gate: rows = (timestep, lag), cols = r modes",
            )
        h = getattr(self.model, "_diag_heads", None)
        if h and h.get("j_star") is not None:
            out["fig/head_assignment_raster"] = _wandb_image(
                h["j_star"][: min(rows * 2, h["j_star"].shape[0])].to(torch.float32),
                "winning head j*: rows = samples, cols = timesteps",
                # cfg.get("n_heads"), matching how the predictor is built above --
                # there is no cfg.model.predictor node, and reading one raises inside
                # _safe(), which discards the WHOLE fig/ block, not just this image
                n_classes=max(int(self.cfg.get("n_heads", 1)), 1),
            )
        return out

    @torch.no_grad()
    def _support_logs(self, prefix):
        """Support statistics as epoch aggregates, for the per-epoch axis.

        The live jacc/* curves are per-batch snapshots; these are the means that
        make train and val directly comparable and that analysis/figures.py can
        read out of a history export.
        """
        if not self._rectified:
            return {}
        tens = self._diag_tensors()
        if tens is None:
            return {}
        z, _u, z_pred, target = tens
        out = {}
        if target is not None:
            out[f"{prefix}_S_model"] = [float((1.0 - soft_jaccard(z_pred, target)).mean())]
        if z.shape[1] > 1:
            out[f"{prefix}_S_world"] = [
                float((1.0 - soft_jaccard(z[:, :-1], z[:, 1:])).mean())
            ]
            out[f"{prefix}_support_churn"] = [
                float(support_churn(z[:, :-1], z[:, 1:]).mean())
            ]
        return out

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            if self._preempted:
                self._checkpoint_and_exit()
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs_linked(obs)  # linked, to match z_obs_out
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)
            val_support = {}
            self._safe("val_support", lambda: self._support_logs("val"), val_support)
            if val_support:
                self.logs_update(val_support)

    @torch.no_grad()
    def _horizon_logs(self, z_roll, z_true, n_past, postfix, horizons=(1, 2, 4, 8, 16)):
        """Rolled-out error and support mismatch as a function of horizon.

        z_roll[j] is a prediction for true frame j once j >= n_past, so the horizon
        from the last observed frame is j - (n_past - 1). Only the final frame is
        measured today, which cannot distinguish a model that degrades gracefully
        from one that is already wrong at h=1 and stays flat.
        """
        out = {}
        n = min(z_roll.shape[1], z_true.shape[1])
        for h in horizons:
            j = n_past - 1 + h
            if j >= n:
                break
            out[f"z_visual_err_rollout{postfix}_h{h}"] = float(
                self.model.emb_criterion(z_roll[:, j], z_true[:, j])
            )
            if self._rectified:
                out[f"S_model_rollout{postfix}_h{h}"] = float(
                    (1.0 - soft_jaccard(
                        z_roll[:, j].flatten(1), z_true[:, j].flatten(1)
                    )).mean()
                )
        return out

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs_linked(obs_g)  # linked, to match the rollout space
            actions = act.unsqueeze(0)

            # Ground truth for the whole (subsampled) episode, so error can be read
            # against HORIZON rather than only at the final frame. One extra encoder
            # pass per rollout episode, ~10 per epoch: invisible next to 30k batches,
            # and error-vs-horizon is the one curve that separates "a good one-step
            # predictor" from "a usable world model".
            z_true = None
            try:
                z_true = self.model.encode_obs_linked(
                    {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}
                )["visual"]
            except (RuntimeError, ValueError, KeyError) as e:
                log.warning(f"per-horizon rollout diagnostic skipped: {e}")

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions, z_goal=z_g["visual"])
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if z_true is not None:
                    hz = {}
                    self._safe(
                        "rollout_horizon",
                        lambda: self._horizon_logs(
                            z_obses["visual"], z_true, n_past, postfix
                        ),
                        hz,
                    )
                    for key, val in hz.items():
                        logs.setdefault(key, []).append(val)

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            # sectioned keys for a readable run page; "epoch" stays top-level so it
            # can be the x-axis for every panel
            to_log = {wandb_key(k): v for k, v in epoch_log.items()}
            to_log["epoch"] = step
            # log at the epoch's final global batch index so these epoch aggregates
            # share one monotonically increasing x-axis with the per-batch curves
            self.wandb_run.log(
                to_log, step=step * getattr(self, "_n_train_batches", 1)
            )
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
