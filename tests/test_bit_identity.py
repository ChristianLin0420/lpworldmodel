"""Invariant 1: at default flags, the Pi-WM interventions must not change the model.

The fixture is a 5-step deterministic CPU loss trace recorded at upstream HEAD,
BEFORE any intervention was implemented. Every intervention is added behind a
flag whose default is off, so re-running this with all defaults must reproduce
the recorded trace exactly. A mismatch means some default changed and the
"bit-identical to upstream at defaults" claim is broken.

Record (once, at HEAD):   python tests/test_bit_identity.py --record
Check:                    pytest tests/test_bit_identity.py
"""
import argparse
import json
from pathlib import Path

import pytest

from lpwm_build import loss_trace

FIXTURE_DIR = Path(__file__).parent / "fixtures"
N_STEPS = 5
BATCH_SIZE = 2


def fixture_path(precision):
    return FIXTURE_DIR / f"bit_identity_{precision}.json"


def record(precision):
    FIXTURE_DIR.mkdir(exist_ok=True)
    trace = loss_trace(n_steps=N_STEPS, batch_size=BATCH_SIZE, precision=precision)
    payload = {
        "precision": precision,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "trace": trace,
    }
    p = fixture_path(precision)
    p.write_text(json.dumps(payload, indent=2))
    print(f"wrote {p}")
    for i, step in enumerate(trace):
        print(f"  step {i}: loss={step['loss']:.10f}  z_loss={step['z_loss']:.10f} "
              f"reg={step.get('reg_loss', float('nan')):.10f} l0={step['l0_frac']:.6f}")
    return payload


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_defaults_match_upstream_fixture(precision):
    """fp32 is the upstream-provenance fixture: recorded at HEAD before any
    intervention existed, so it proves the flags-off path IS upstream. bf16 is a
    regression lock in the campaign's precision -- recorded after the fact, so it
    cannot prove provenance, only that defaults stop drifting from here on."""
    p = fixture_path(precision)
    if not p.exists():
        pytest.skip(f"no fixture at {p}; record it at HEAD first")
    want = json.loads(p.read_text())
    got = loss_trace(
        n_steps=want["n_steps"], batch_size=want["batch_size"], precision=precision
    )
    assert len(got) == len(want["trace"])
    for i, (g, w) in enumerate(zip(got, want["trace"])):
        assert set(g) == set(w), f"step {i}: loss components changed"
        for k in w:
            assert g[k] == pytest.approx(w[k], rel=0, abs=0), (
                f"step {i} component '{k}': {g[k]!r} != {w[k]!r} "
                "-- a default-off intervention is not actually off"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    a = ap.parse_args()
    if a.record:
        record(a.precision)
    else:
        print("pass --record to write the fixture")
