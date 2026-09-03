"""Build the frozen-frame cache the latent probe (M1) fits on.

WHY A CACHE.  `analysis/latent_probe.py` runs one ViT forward over the SAME frames for
every one of the 327 archived checkpoints.  Decoding the mp4s and applying
`default_transform(224)` once, and storing the exact tensor `encode_obs_linked` consumes,
turns 327 x (decode + transform) into 327 x (encode).

WHAT IS IN IT.  `datasets/pusht_dset.py:get_frames` already applies the run's transform, so
`visual` here is bit-identical to what the model sees at train and plan time (normalised to
roughly [-1, 1] by `Normalize([0.5]*3, [0.5]*3)`).  `state` is the RAW 7-vector
`[agent_x, agent_y, block_x, block_y, block_theta, agent_vx, agent_vy]` -- the PRIVILEGED
block pose.  That is legal here and only here: this file feeds MEASUREMENT.  Nothing built
from `state` may ever reach a training loss (see docs/round5-specs.md, "Constraints").

SPLIT UNIT IS THE EPISODE, NEVER THE FRAME.  Consecutive frames of one trajectory are
near-duplicates; a frame-level split leaks the answer and a linear probe would score well on
a latent that memorised nothing.  Every row carries its episode index and its split, and the
probe groups on `(split, ep)`.

ROLES.  `fit` = probe training episodes; `heldout` = episodes from the same split withheld
from the probe; `val` = the model's own validation split, the only episodes that are ALSO
held out from the encoder's training (`n_rollout: null` in conf/env/pusht.yaml means every
`train/` episode was seen by every checkpoint).  The headline number is reported on `val`.

Usage
    python analysis/probe_cache.py --out runs/probe_cache.pt \
        --train-eps 300 --heldout-eps 150 --stride 4 --val-stride 2
    # tiny smoke cache (no train split at all: the val episodes are split fit/val)
    python analysis/probe_cache.py --out /tmp/pc.pt --train-eps 4 --stride 20 --splits val
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.img_transforms import default_transform          # noqa: E402
from datasets.pusht_dset import PushTDataset                   # noqa: E402

# role codes kept as plain strings; the probe only ever compares them
FIT, HELDOUT, VAL = "fit", "heldout", "val"


def _episode_rows(ds, ep, stride):
    """(visual, state) for every `stride`-th frame of episode `ep`."""
    frames = list(range(0, int(ds.get_seq_length(ep)), stride))
    obs, _, state, _ = ds.get_frames(ep, frames)
    return obs["visual"].half(), state.float(), len(frames)


def build(dataset_root, splits, train_eps, heldout_eps, stride, val_stride, val_eps,
          sample="random", sample_seed=0):
    """Return the cache dict.  See the module docstring for the role semantics."""
    vis, st, eps, spl, role = [], [], [], [], []

    def take(ds, split, ep_ids, s, r):
        for ep in ep_ids:
            v, y, n = _episode_rows(ds, ep, s)
            vis.append(v)
            st.append(y)
            eps.append(np.full(n, ep, dtype=np.int64))
            spl.append(np.full(n, split, dtype=object))
            role.append(np.full(n, r, dtype=object))
            print(f"  {split} ep {ep:5d}  {n:4d} frames -> {r}", flush=True)

    if "train" in splits:
        n_load = train_eps + heldout_eps
        # THE TRAIN SPLIT IS ORDERED IN BLOCKS AND MUST NOT BE SLICED FROM THE FRONT.
        # Measured: episodes 0-79 share ONE initial block pose exactly (between-episode
        # std 0.00 px), and there are only 10 distinct initial block poses in the first
        # 1000 episodes, against a whole-split std of 92 px / 1.83 rad.  Fitting a probe
        # on "the first 300 train episodes" (as the spec's command reads) therefore fits
        # it on ~4 block poses, and every number on the val split becomes an
        # extrapolation artefact rather than a measurement.  Sample uniformly instead.
        ds = PushTDataset(n_rollout=None if sample == "random" else n_load,
                          transform=default_transform(224),
                          data_path=f"{dataset_root}/train")
        if sample == "random":
            pick = np.random.RandomState(sample_seed).choice(len(ds), n_load, replace=False)
            pick.sort()
        else:
            pick = np.arange(n_load)
        take(ds, "train", pick[:train_eps], stride, FIT)
        take(ds, "train", pick[train_eps:], stride, HELDOUT)

    if "val" in splits:
        ds = PushTDataset(n_rollout=val_eps, transform=default_transform(224),
                          data_path=f"{dataset_root}/val")
        n = len(ds)
        if "train" in splits:
            take(ds, "val", range(n), val_stride, VAL)
        else:
            # smoke mode: no train split, so carve the fit set out of val itself.
            k = min(train_eps, max(1, n - 1))
            take(ds, "val", range(k), stride, FIT)
            take(ds, "val", range(k, n), stride, VAL)

    if not vis:
        raise ValueError(f"no episodes collected for splits={splits}")
    return {
        "visual": torch.cat(vis),                    # (N, 3, 224, 224) half
        "state": torch.cat(st),                      # (N, 7) float32, PRIVILEGED
        "ep": np.concatenate(eps),                   # (N,) episode index within its split
        "split": np.concatenate(spl),                # (N,) "train" | "val"
        "role": np.concatenate(role),                # (N,) fit | heldout | val
        "meta": dict(stride=stride, val_stride=val_stride, train_eps=train_eps,
                     heldout_eps=heldout_eps, splits=sorted(splits),
                     state_cols=["agent_x", "agent_y", "block_x", "block_y",
                                 "block_theta", "agent_vx", "agent_vy"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/probe_cache.pt")
    ap.add_argument("--dataset-root",
                    default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_noise"))
    ap.add_argument("--splits", default="train,val",
                    help="comma list of dataset splits to read (train,val)")
    ap.add_argument("--train-eps", type=int, default=300, help="episodes the probe FITS on")
    ap.add_argument("--heldout-eps", type=int, default=150,
                    help="further train episodes withheld from the probe")
    ap.add_argument("--stride", type=int, default=4, help="frame stride in the train split")
    ap.add_argument("--val-stride", type=int, default=2, help="frame stride in the val split")
    ap.add_argument("--val-eps", type=int, default=None, help="cap on val episodes (default all)")
    ap.add_argument("--sample", choices=["random", "first"], default="random",
                    help="how train episodes are chosen. `first` reproduces the spec's "
                         "literal 'first 300 episodes' and is degenerate -- see build().")
    ap.add_argument("--sample-seed", type=int, default=0)
    a = ap.parse_args()

    cache = build(a.dataset_root, {s for s in a.splits.split(",") if s},
                  a.train_eps, a.heldout_eps, a.stride, a.val_stride, a.val_eps,
                  a.sample, a.sample_seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(cache, a.out)
    v = cache["visual"]
    print(f"\nwrote {a.out}")
    print(f"  visual {tuple(v.shape)} {v.dtype}  range [{v.min().item():.3f}, {v.max().item():.3f}]")
    print(f"  state  {tuple(cache['state'].shape)}")
    for r in (FIT, HELDOUT, VAL):
        m = cache["role"] == r
        if m.sum():
            print(f"  {r:8s} {int(m.sum()):6d} frames  "
                  f"{len(set(zip(cache['split'][m], cache['ep'][m])))} episodes")


if __name__ == "__main__":
    main()
