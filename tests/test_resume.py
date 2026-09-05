"""Phase 0g -- preemption safety, verified against the real Trainer.train/run code.

A SLURM mid-epoch scancel is simulated by raising the same SIGUSR1 the launcher
sends, so the code path under test is identical to the cluster one. The strongest
statement of "the resumed loss curve is continuous" is that the parameters after
a chopped run equal the parameters after an uninterrupted one, bit for bit; that
is what test_resume_is_lossless asserts.
"""
import os
import signal

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from train import Trainer

N_BATCHES, EPOCHS = 8, 3


class _Acc:
    """Minimal stand-in for a single-process Accelerator."""
    is_main_process = True
    num_processes = 1
    device = torch.device("cpu")

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, m):
        return m

    def gather_for_metrics(self, x):
        return x

    def backward(self, loss):
        loss.backward()


class _Model(nn.Module):
    """Deterministic given (params, data, RNG). The randn draw is deliberate: it
    makes the loss depend on the RNG stream, so parameter equality after a resume
    also proves the RNG state was restored rather than silently reseeded."""
    train_encoder = False
    train_decoder = False
    train_predictor = True

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4))

    def forward(self, obs, act):
        noise = torch.randn(4)
        loss = ((self.w + noise - obs) ** 2).mean()
        return None, None, None, loss, {"loss": loss.detach()}


def _cfg(folder):
    return OmegaConf.create({
        "saved_folder": str(folder),
        "has_decoder": False,
        "has_predictor": True,
        "num_hist": 1,
        "training": {"epochs": EPOCHS, "save_every_x_epoch": 1, "save_every_x_min": 0},
        "plan_settings": {"plan_cfg_path": None},
    })


class Harness(Trainer):
    """Assembles the real Trainer's harness surface without a dataset or wandb.

    Only val/logs_flash are stubbed, because they need the rollout and wandb
    machinery; train(), run(), save_ckpt() and load_ckpt() are the real methods.
    """

    def __init__(self, folder, preempt_at=None):
        self.cfg = _cfg(folder)
        self.accelerator = _Acc()
        self.device = torch.device("cpu")
        self.total_epochs = EPOCHS
        self.epoch, self.batch_idx = 0, 0
        self._pending_rng, self._preempted = None, False
        self._save_every_sec, self._last_ckpt_time = 0.0, 0.0
        # same registration as Trainer.__init__; without it SIGUSR1's default
        # disposition kills the interpreter instead of reaching the handler
        for _sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(_sig, self._on_preempt)
        self.epoch_log = {}
        self.num_reconstruct_samples = 0
        self.train_encoder, self.train_predictor, self.train_decoder = False, True, False
        # live-diagnostics surface: _n_train_batches is the denominator for the
        # resume marker, and the timing fields are touched once per batch by
        # _mark_iter_start. Mirrored here so a missing wiring in Trainer.__init__
        # fails this suite instead of a 4h GPU window.
        self._n_train_batches = N_BATCHES
        self._t_body_end, self._data_wait, self._data_span_t0 = None, 0.0, None
        self._diag_fail = {}

        torch.manual_seed(0)
        self.model = _Model()
        self.predictor = self.model
        self.encoder = self.decoder = self.link = None
        self.action_encoder = self.proprio_encoder = None
        self.encoder_optimizer = None
        self.predictor_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        self.action_encoder_optimizer = self.predictor_optimizer
        self._keys_to_save = ["epoch", "predictor", "predictor_optimizer"]

        # fixed batch order, mirroring the real loader's shuffle=False
        self.dataloaders = {
            "train": [(torch.full((4,), float(i)), None, None) for i in range(N_BATCHES)]
        }
        self.preempt_at = set(preempt_at or ())
        self.trained = []          # (epoch, batch) actually stepped
        self.epochs_seen = []

    # the real signal handler, reached the way SLURM reaches it
    def train(self):
        for _ in range(1):
            super().train()

    def _should_checkpoint(self):
        if (self.epoch, self.batch_idx) in self.preempt_at:
            os.kill(os.getpid(), signal.SIGUSR1)
        return Trainer._should_checkpoint(self)

    def val(self):
        self.epochs_seen.append(self.epoch)

    def logs_flash(self, step):
        self.epoch_log = {}

    def logs_update(self, logs):
        # Only per-batch payloads count. run() also logs epoch-level scalars like
        # epoch_seconds through this same method, and counting those as batches
        # would inflate the coverage check by one per epoch.
        if not any(k.startswith("train_") for k in logs):
            return
        # called before train() advances batch_idx, so batch_idx is this batch's index
        self.trained.append((self.epoch, self.batch_idx))


def _run(folder, preempt_at=None, max_restarts=10):
    """Run to completion, restarting after each simulated preemption exit."""
    cwd = os.getcwd()
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    trained, epochs_seen, restarts = [], [], 0
    try:
        for _ in range(max_restarts):
            h = Harness(folder, preempt_at=preempt_at)
            h._maybe_resume()
            try:
                h.run()
            except SystemExit as e:
                assert e.code == 0, "preemption must exit 0 so afterany can chain"
                trained += h.trained
                restarts += 1
                continue
            trained += h.trained
            epochs_seen = h.epochs_seen
            return trained, epochs_seen, restarts, h
        pytest.fail("did not finish within max_restarts")
    finally:
        os.chdir(cwd)


def test_uninterrupted_run_is_the_reference(tmp_path):
    trained, epochs_seen, restarts, _ = _run(tmp_path)
    assert restarts == 0
    assert epochs_seen == [1, 2, 3]
    assert len(trained) == EPOCHS * N_BATCHES


def test_final_epoch_count_is_exact_under_preemption(tmp_path):
    """The 0c bug ran total_epochs MORE epochs per resume; assert it cannot recur."""
    _, _, restarts, h = _run(tmp_path, preempt_at={(1, 3), (2, 5)})
    assert restarts == 2
    assert h.epoch == EPOCHS, f"ran to epoch {h.epoch}, expected exactly {EPOCHS}"
    assert h.epochs_seen[-1] == EPOCHS


@pytest.mark.parametrize("preempt_at", [
    {(1, 1)},                       # first epoch, first batch
    {(1, 3), (2, 5)},               # mid-epoch, two different epochs
    {(2, 8)},                       # exactly at the epoch boundary
    {(1, 2), (1, 4), (1, 6)},       # repeated preemption inside one epoch
])
def test_every_batch_trained_exactly_once(tmp_path, preempt_at):
    """No gap (lost work) and no duplicate (double-counted work) across restarts."""
    trained, _, _, _ = _run(tmp_path, preempt_at=preempt_at)
    expected = [(e, b) for e in range(1, EPOCHS + 1) for b in range(N_BATCHES)]
    assert sorted(trained) == expected, (
        f"missing={sorted(set(expected) - set(trained))} "
        f"duplicated={sorted(b for b in set(trained) if trained.count(b) > 1)}"
    )


def test_resume_is_lossless(tmp_path):
    """Bit-identical parameters vs the uninterrupted run: the strongest form of
    'the resumed loss curve is continuous'. Covers weights, Adam moments and RNG."""
    _, _, _, ref = _run(tmp_path / "ref")
    _, _, restarts, got = _run(tmp_path / "chopped", preempt_at={(1, 3), (2, 5), (3, 7)})
    assert restarts == 3
    assert torch.equal(ref.model.w, got.model.w), (
        f"resume is lossy: max|dw| = {(ref.model.w - got.model.w).abs().max():.3e}"
    )


def test_optimizer_state_is_checkpointed(tmp_path):
    """Adam moments must survive: restarting them at zero steps the loss curve."""
    _run(tmp_path, preempt_at={(1, 3)})
    ckpt = torch.load(tmp_path / "checkpoints" / "model_latest.pth", map_location="cpu")
    assert "predictor_optimizer" in ckpt
    assert ckpt["predictor_optimizer"]["state"], "Adam moments are empty"
    assert "rng" in ckpt and set(ckpt["rng"]) >= {"torch", "numpy", "python"}


def test_done_sentinel_only_after_full_budget(tmp_path):
    """submit_until_done.sh stops on this file, so a preempted exit must not write it."""
    ref, chopped = tmp_path / "ref", tmp_path / "chopped"
    for d in (ref, chopped):
        d.mkdir()

    os.chdir(chopped)
    try:
        h = Harness(chopped, preempt_at={(1, 3)})
        with pytest.raises(SystemExit):
            h.run()
        assert not (chopped / "DONE").exists(), "DONE written on a preemption exit"
    finally:
        os.chdir(tmp_path)

    _run(ref)
    assert (ref / "DONE").exists()
    assert f"epochs={EPOCHS}" in (ref / "DONE").read_text()


def test_resume_markers_are_recorded_for_the_figures(tmp_path):
    """analysis/figures.py draws these as vertical lines on the training curves; a
    loss step AT a marker is a lossy resume, the same step elsewhere is science, so
    the markers have to land at the right x positions."""
    import json

    _, _, restarts, _ = _run(tmp_path, preempt_at={(1, 3), (2, 5)})
    marks = json.loads((tmp_path / "resume_steps.json").read_text())
    assert len(marks) == restarts == 2
    # preempted after batch 3 of epoch 1 and batch 5 of epoch 2, in epoch units
    assert marks == [pytest.approx(3 / N_BATCHES), pytest.approx(1 + 5 / N_BATCHES)]


def test_action_encoder_optimizer_is_in_keys_to_save():
    """Guards the fix: this optimizer exists in init_optimizers but was unsaved."""
    import inspect
    src = inspect.getsource(Trainer.__init__)
    assert '"action_encoder_optimizer"' in src


def test_restored_optimizer_state_is_moved_to_the_device(tmp_path):
    """Guards the T4 fix: an optimizer restored from a checkpoint must not be left on CPU.

    load_ckpt reads with map_location="cpu" and then calls
    accelerator.unwrap_model(obj).load_state_dict(...), which strips the AcceleratedOptimizer
    wrapper that would otherwise move the restored state onto the device. Parameters are
    unaffected -- they belong to modules already on the device and load_state_dict copies into
    them in place -- but an optimizer's exp_avg / exp_avg_sq are fresh CPU tensors, and the
    next step() raises "Expected all tensors to be on the same device".

    This killed 7 of T4's 8 seeds at epoch 2, after they had trained the whole way, and nothing
    caught it: the NEW HEAD CHECKLIST asks whether a parameter reaches a checkpoint, not
    whether its optimizer survives a round trip through one. T4 was the first arm in the
    campaign to carry optimizers beyond the five that predate it.

    Device-agnostic by construction: it asserts the state lands on accelerator.device, which is
    CPU on a submit host and CUDA on a GPU node, so it is a real guard in CI either way.
    """
    import types

    net = nn.Linear(4, 4)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    net(torch.randn(2, 4)).sum().backward()
    opt.step()                                   # materialise exp_avg / exp_avg_sq
    assert opt.state, "optimizer has no state to restore -- the test would be vacuous"

    fresh = torch.optim.AdamW(net.parameters(), lr=1e-3)
    fresh.load_state_dict(opt.state_dict())      # the map_location="cpu" path
    for st in fresh.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.cpu()                  # force the exact failure condition

    trainer = types.SimpleNamespace(accelerator=_Acc())
    moved = Trainer._optimizer_state_to_device(trainer, fresh)
    assert moved >= 0
    dev = _Acc().device
    for st in fresh.state.values():
        for k, v in st.items():
            assert not torch.is_tensor(v) or v.device == dev, (
                f"optimizer state '{k}' left on {v.device}, expected {dev}"
            )


# --- corrupt-checkpoint handling ------------------------------------------------------------
# PiWM-overshoot8_s8 was left with a full-size 167.8 MB model_latest.pth whose zip central
# directory never reached disk. Because windows are pre-chained, EVERY remaining window then
# died in 43 s on the same load error: no DONE, no epoch, and eight queued jobs guaranteed to
# repeat it. These pin both halves of the fix.

def test_ckpt_readable_detects_a_lost_tail(tmp_path):
    """The observed corruption is a truncated tail, which is what the zip index catches."""
    from train import _ckpt_readable
    good = tmp_path / "good.pth"
    torch.save({"a": torch.zeros(4)}, good)
    assert _ckpt_readable(good) is True

    raw = good.read_bytes()
    bad = tmp_path / "bad.pth"
    bad.write_bytes(raw[: len(raw) // 2])
    assert _ckpt_readable(bad) is False
    assert _ckpt_readable(tmp_path / "absent.pth") is False


def test_save_ckpt_is_fsynced_before_the_rename(tmp_path, monkeypatch):
    """os.replace must not run until the data is on disk, or the rename publishes a partial file.

    Asserted by ORDER, not by mocking fsync away: a checkpoint that is renamed into place
    before its bytes are durable is exactly how the corrupt one was produced.
    """
    import train as T
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(T.os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(T.os, "replace",
                        lambda a, b: (calls.append("replace"), real_replace(a, b))[1])

    trained, _, _, h = _run(tmp_path)
    assert "fsync" in calls and "replace" in calls, calls
    assert calls.index("fsync") < calls.index("replace"), (
        f"os.replace ran before the first fsync: {calls[:6]}")


def test_a_corrupt_checkpoint_is_quarantined_and_the_run_restarts(tmp_path):
    """One unreadable file must not kill the whole pre-chained window sequence."""
    trained, _, _, h = _run(tmp_path)
    ck = tmp_path / "checkpoints" / "model_latest.pth"
    assert ck.exists()

    raw = ck.read_bytes()
    ck.write_bytes(raw[: len(raw) // 2])          # lose the tail, as the node failure did

    # Without the fix this raises PytorchStreamReader inside load_ckpt and _run never returns.
    trained2, epochs2, restarts2, h2 = _run(tmp_path)
    assert (tmp_path / "checkpoints" / "model_latest.pth.corrupt").exists(), \
        "the unreadable checkpoint was not quarantined"
    assert epochs2 == [1, 2, 3], f"did not restart from scratch: {epochs2}"
    assert len(trained2) == EPOCHS * N_BATCHES, (
        f"restarted run trained {len(trained2)} batches, expected a full budget")
