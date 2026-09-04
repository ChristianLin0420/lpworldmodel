"""A resume past the final epoch's last batch must not kill the run before it writes DONE.

PiWM-vp-mc_pd384_bf16_s10 resumed at "epoch 2, batch 30965", trained zero batches, and
logs_flash raised KeyError('train_loss') -- before save_ckpt could write the DONE sentinel.
Because the resume point is stored in the checkpoint, every resubmission failed identically:
8 consecutive jobs across two waves, 20 hours with no checkpoint written, and round 5 could
not close because that arm was stuck at n=7.
"""
import logging
from collections import OrderedDict

import pytest


class _Stub:
    """The two attributes logs_flash touches, plus a no-op accelerator."""

    def __init__(self, epoch_log):
        self.epoch = 2
        self.epoch_log = epoch_log
        self.accelerator = type("A", (), {"is_main_process": False})()

    logs_flash = None  # bound below


def _bind():
    from train import Trainer
    _Stub.logs_flash = Trainer.logs_flash


def test_empty_epoch_log_does_not_raise(caplog):
    """The regression: zero batches trained must log, not raise."""
    _bind()
    s = _Stub(OrderedDict())
    with caplog.at_level(logging.INFO):
        s.logs_flash(step=2)                    # must not raise KeyError
    assert "Epoch 2" in caplog.text
    assert "n/a" in caplog.text


def test_partial_epoch_log_does_not_raise():
    """train_loss present but val_loss missing (val preempted) is also survivable."""
    _bind()
    s = _Stub(OrderedDict(train_loss=(2, 1.0)))
    s.logs_flash(step=2)


def test_normal_epoch_still_reports_both_numbers(caplog):
    """The fix must not change the happy path's output."""
    _bind()
    s = _Stub(OrderedDict(train_loss=(2, 1.0), val_loss=(4, 1.0)))
    with caplog.at_level(logging.INFO):
        s.logs_flash(step=2)
    assert "Training loss: 0.5000" in caplog.text
    assert "Validation loss: 0.2500" in caplog.text
    assert "n/a" not in caplog.text
