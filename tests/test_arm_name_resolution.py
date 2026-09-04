"""The feature tag in an arm key has silently broken three separate contrast reports.

collect() keys a patch-feature arm "PiWM-patchdecode_patch" but a cls arm by its bare name,
so `arms.get("PiWM-patchdecode", {})` returns {} and reads as "not evaluated yet" rather than
as the bug it is. These tests pin the resolver's contract.
"""
import pytest

from analysis.collect_evals import ArmNameError, arm_seeds, resolve_arm

ARMS = {
    "LpWM-ltv": {"3": 0.24},
    "PiWM-vp": {"3": 0.40},
    "PiWM-vp-mc": {"3": 0.31},
    "PiWM-vp-geom": {"3": 0.72},
    "PiWM-patchdecode_patch": {"3": 0.46},
    "PiWM-patchdecode-detach_patch": {"3": 0.33},
}


def test_exact_key_wins():
    assert resolve_arm(ARMS, "PiWM-vp") == "PiWM-vp"


def test_feature_tag_is_resolved():
    # the exact failure that reported n=0 with data on disk
    assert resolve_arm(ARMS, "PiWM-patchdecode") == "PiWM-patchdecode_patch"
    assert resolve_arm(ARMS, "PiWM-patchdecode-detach") == "PiWM-patchdecode-detach_patch"


def test_hyphen_sibling_is_not_swallowed():
    # "PiWM-vp" must not match "PiWM-vp-mc": the prefix test appends "_", not "".
    # Without this, the T4 baseline contrast would silently compare the wrong arm.
    assert resolve_arm(ARMS, "PiWM-vp") == "PiWM-vp"
    assert resolve_arm(ARMS, "PiWM-vp-mc") == "PiWM-vp-mc"


def test_unknown_name_raises_rather_than_returning_empty():
    # the whole point: a bad name must not read as "no data yet"
    with pytest.raises(ArmNameError):
        resolve_arm(ARMS, "PiWM-does-not-exist")


def test_ambiguous_name_raises():
    ambiguous = {"PiWM-x_patch": {}, "PiWM-x_cls": {}}
    with pytest.raises(ArmNameError):
        resolve_arm(ambiguous, "PiWM-x")


def test_arm_seeds_requires_explicit_opt_in_to_empty():
    assert arm_seeds(ARMS, "PiWM-patchdecode") == {"3": 0.46}
    with pytest.raises(ArmNameError):
        arm_seeds(ARMS, "nope")
    assert arm_seeds(ARMS, "nope", default={}) == {}


def test_resolves_against_the_live_campaign():
    """Every tagged arm in the real archive must be reachable by its bare name."""
    from analysis.collect_evals import collect

    arms = collect()[0]
    tagged = [k for k in arms if k.endswith("_patch")]
    if not tagged:
        pytest.skip("no feature-tagged arms in the archive")
    for key in tagged:
        bare = key[: -len("_patch")]
        assert resolve_arm(arms, bare) == key
