"""M2 -- block-only / terminal / per-episode-trace metric audit.

Covers env/pusht/pusht_wrapper.py (eval_state, _block_iou) and
planning/evaluator.py (terminal_* logs, per-episode npz traces).

The two invariants that make every archived number still reproducible:
  * `success` and `state_dist` keep their exact historical value AND dtype;
  * nothing that existed before changed shape or meaning -- the new keys are purely
    additive.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from env.pusht.pusht_wrapper import PushTWrapper  # noqa: E402
from env.pusht.pusht_env import pymunk_to_shapely  # noqa: E402
from planning.evaluator import PlanEvaluator  # noqa: E402
from tests.stub_wm import StubWorldModel  # noqa: E402


def _legacy_eval_state(goal_state, cur_state):
    """The pre-M2 body of PushTWrapper.eval_state, verbatim."""
    pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
    angle_diff = np.abs(goal_state[4] - cur_state[4])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    success = pos_diff < 20 and angle_diff < np.pi / 9
    state_dist = np.linalg.norm(goal_state - cur_state)
    return {"success": success, "state_dist": state_dist}


@pytest.fixture(scope="module")
def env():
    return PushTWrapper()


def _rand_states(n, rs, dtype=np.float32):
    s = np.stack(
        [
            rs.uniform(50, 450, n),
            rs.uniform(50, 450, n),
            rs.uniform(100, 400, n),
            rs.uniform(100, 400, n),
            rs.uniform(0, 2 * np.pi, n),
            np.zeros(n),
            np.zeros(n),
        ],
        axis=1,
    )
    return s.astype(dtype)


def test_success_and_state_dist_bit_identical(env):
    """The control: no archived number may move, not even in the last bit."""
    rs = np.random.RandomState(0)
    g = _rand_states(200, rs)
    c = _rand_states(200, rs)
    # plus 50 near-hits so the success branch is actually exercised
    c[:50] = g[:50] + rs.normal(0, 8, (50, 7)).astype(np.float32)
    n_succ = 0
    for i in range(len(g)):
        want = _legacy_eval_state(g[i], c[i])
        got = env.eval_state(g[i], c[i])
        assert bool(got["success"]) == bool(want["success"])
        assert got["state_dist"].dtype == want["state_dist"].dtype
        assert got["state_dist"] == want["state_dist"]  # exact, not allclose
        n_succ += bool(got["success"])
    assert n_succ > 0, "degenerate test: no successes were generated"


def test_new_keys_present_and_typed(env):
    rs = np.random.RandomState(1)
    g, c = _rand_states(1, rs)[0], _rand_states(1, rs)[0]
    got = env.eval_state(g, c)
    assert set(got) == {
        "success",
        "state_dist",
        "success_block",
        "block_pos_diff",
        "agent_pos_diff",
        "angle_diff",
        "block_iou",
    }
    assert np.isfinite(got["block_iou"]) and 0.0 <= got["block_iou"] <= 1.0


def test_success_implies_success_block(env):
    """block_pos_diff <= pos_diff, so the historical metric is the STRICTER one."""
    rs = np.random.RandomState(2)
    g = _rand_states(500, rs)
    c = g + rs.normal(0, 12, (500, 7)).astype(np.float32)
    n = 0
    for i in range(len(g)):
        r = env.eval_state(g[i], c[i])
        assert r["block_pos_diff"] <= np.linalg.norm(g[i][:4] - c[i][:4]) + 1e-5
        if r["success"]:
            assert r["success_block"]
            n += 1
    assert n > 0


def test_agent_parking_case(env):
    """The M2 spec's own check: block on target, end effector 300px away."""
    g = np.array([100.0, 100.0, 250.0, 250.0, 0.5, 0, 0], dtype=np.float32)
    c = g.copy()
    c[:2] += 300
    r = env.eval_state(g, c)
    assert r["success"] is False or bool(r["success"]) is False
    assert bool(r["success_block"]) is True
    assert r["block_iou"] == pytest.approx(1.0, abs=1e-9)
    assert r["block_pos_diff"] == pytest.approx(0.0)
    assert r["agent_pos_diff"] == pytest.approx(300 * np.sqrt(2), rel=1e-5)


def test_block_iou_matches_real_block_geometry(env):
    """The synthetic body used by _block_iou must reproduce the geometry pymunk
    actually simulates, otherwise the IoU scores a different T."""
    st = np.array([200.0, 300.0, 260.0, 240.0, 1.1, 0, 0])
    _, s = env.prepare(0, st)
    real = pymunk_to_shapely(env.block, env.block.shapes)
    synth = env._block_geom(s[2:5])
    inter = real.intersection(synth).area
    iou = inter / (real.area + synth.area - inter)
    assert iou > 0.9999
    # and a disjoint pose scores 0
    far = np.array([s[2] + 400, s[3] + 400, s[4]])
    assert env._block_iou(s[2:5], far) == pytest.approx(0.0)


def test_block_iou_never_raises_on_a_fresh_env():
    """eval_state is also called offline on envs that were never reset."""
    fresh = PushTWrapper()
    g = np.array([100.0, 100.0, 250.0, 250.0, 0.5, 0, 0])
    assert fresh.eval_state(g, g.copy())["block_iou"] == pytest.approx(1.0, abs=1e-9)


def test_quasi_static_zero_action_freezes_the_block(env):
    """mpc.py `_apply_success_mask` emits raw-zero RELATIVE actions after a latch
    (`actions[mask]=0` then re-normalise, so denormalise gives raw 0 = hold).

    space.damping == 0 zeroes every dynamic body's velocity each step, so the BLOCK
    is frozen exactly.  The agent is a KINEMATIC body, so it is not damped by the
    space -- it coasts for one control step under the k_v term before it settles.
    This bounds how far a latched state can be from the terminal state, which is
    what makes latched ~= terminal for the block-only metric.
    """
    st = np.array([250.0, 250.0, 260.0, 250.0, 0.3, 0, 0])
    env.prepare(0, st)
    push = np.tile(np.array([[0.15, 0.0]]), (10, 1))  # drive the agent into the block
    _, _, _, info = env.step_multiple(push)
    moved = info["state"][-1]
    assert np.linalg.norm(moved[2:4] - st[2:4]) > 5, "the block must actually move"

    zeros = np.zeros((40, 2))
    _, _, _, info2 = env.step_multiple(zeros)
    traj = info2["state"]
    rested = traj[-1]
    # block: bit-frozen
    assert np.linalg.norm(moved[2:4] - rested[2:4]) < 1e-3
    assert abs(moved[4] - rested[4]) < 1e-4
    # agent: coasts once, then stops dead
    assert np.linalg.norm(moved[:2] - rested[:2]) < 5.0
    # ... and everything is fixed to <0.02px from the second zero step onward
    assert np.linalg.norm(traj[1][:4] - traj[-1][:4]) < 2e-2
    assert np.allclose(traj[2][:5], traj[-1][:5], atol=1e-4)


# --------------------------------------------------------------------------------
# evaluator: terminal metrics + per-episode traces, on the REAL env
# --------------------------------------------------------------------------------
class _Pre:
    def transform_obs(self, obs):
        out = {}
        for k, v in obs.items():
            t = torch.as_tensor(np.asarray(v, dtype=np.float32))
            if k == "visual":
                t = t.permute(0, 1, 4, 2, 3) / 255.0
            out[k] = t
        return out

    def transform_obs_visual(self, v):
        return torch.as_tensor(np.asarray(v, dtype=np.float32)).permute(0, 1, 4, 2, 3)

    def denormalize_actions(self, a):
        return a * 0.1

    def normalize_actions(self, a):
        return a


class _Wm(StubWorldModel):
    decoder = None


def _build(tmpdir, n=3, frameskip=5, horizon=4, trace=True):
    from env.serial_vector_env import SerialVectorEnv

    envs = SerialVectorEnv([PushTWrapper() for _ in range(n)])
    rs = np.random.RandomState(7)
    state_0 = _rand_states(n, rs, dtype=np.float64)
    state_g = state_0 + np.concatenate(
        [rs.normal(0, 30, (n, 4)), rs.normal(0, 0.2, (n, 1)), np.zeros((n, 2))], axis=1
    )
    seeds = list(range(1, n + 1))
    obs_0, s0 = envs.prepare(seeds, state_0)
    obs_g, sg = envs.prepare(seeds, state_g)
    obs_0 = {k: np.expand_dims(v, 1) for k, v in obs_0.items()}
    obs_g = {k: np.expand_dims(v, 1) for k, v in obs_g.items()}
    wm = _Wm(emb_dim=8, num_patches=1, action_dim=frameskip * 2)
    ev = PlanEvaluator(
        obs_0=obs_0,
        obs_g=obs_g,
        state_0=s0,
        state_g=sg,
        env=envs,
        wm=wm,
        frameskip=frameskip,
        seed=seeds,
        preprocessor=_Pre(),
        n_plot_samples=0,
        trace_file=(str(Path(tmpdir) / "tr") if trace else ""),
    )
    actions = torch.zeros(n, horizon, frameskip * 2)
    actions[:, :, 0::2] = 0.6  # push +x every sub-step
    return ev, actions, n, horizon


def test_terminal_keys_appear_and_trace_is_written(tmp_path):
    ev, actions, n, horizon = _build(tmp_path)
    action_len = np.full(n, np.inf)
    action_len[0] = 1.0  # episode 0 "latched" after one MPC block
    logs, successes, e_obses, e_states = ev.eval_actions(
        actions, action_len, filename="output_final", save_video=True
    )

    for k in [
        "success_rate",
        "mean_state_dist",
        "mean_success_block",
        "mean_block_pos_diff",
        "mean_agent_pos_diff",
        "mean_angle_diff",
        "mean_block_iou",
        "terminal_success_rate",
        "terminal_mean_success_block",
        "terminal_mean_block_pos_diff",
        "terminal_mean_agent_pos_diff",
        "terminal_mean_angle_diff",
        "terminal_mean_block_iou",
        "terminal_mean_state_dist",
    ]:
        assert k in logs, k
    import json

    json.dumps({k: (v.item() if isinstance(v, np.float32) else v) for k, v in logs.items()})

    f = tmp_path / "tr_output_final.npz"
    assert f.exists()
    d = np.load(f)
    for k in [
        "seed",
        "state_0",
        "state_g",
        "action_len",
        "e_state_final",
        "e_state_latched",
        "success",
        "success_block",
        "block_pos_diff",
        "agent_pos_diff",
        "angle_diff",
        "block_iou",
        "latched_success",
        "latched_state_dist",
        "latched_block_pos_diff",
        "d_pred",
        "d_real",
        "div_visual_emb",
        "visual_dist",
        "proprio_dist",
    ]:
        assert k in d, k
        assert d[k].shape[0] == n, (k, d[k].shape)

    # the whole point: episode 0's latched state is NOT its terminal state
    assert not np.allclose(d["e_state_latched"][0], d["e_state_final"][0])
    # ... while an un-truncated episode's are identical
    assert np.allclose(d["e_state_latched"][1], d["e_state_final"][1])
    # terminal metrics are recomputed from the terminal state, not copied
    assert np.allclose(
        d["block_pos_diff"],
        np.linalg.norm(d["state_g"][:, 2:4] - d["e_state_final"][:, 2:4], axis=1),
    )
    assert np.allclose(
        d["latched_block_pos_diff"],
        np.linalg.norm(d["state_g"][:, 2:4] - d["e_state_latched"][:, 2:4], axis=1),
    )


def test_trace_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LPWM_TRACE_FILE", "")
    ev, actions, n, _ = _build(tmp_path, trace=False)
    ev.trace_file = None
    ev.eval_actions(actions, None, filename="output_final", save_video=True)
    assert not list(tmp_path.glob("*.npz"))


def test_trace_prefix_from_environment(tmp_path, monkeypatch):
    """This wave does not own plan.py, so the prefix has to arrive by env var."""
    monkeypatch.setenv("LPWM_TRACE_FILE", str(tmp_path / "envtr"))
    ev, actions, n, _ = _build(tmp_path, trace=False)
    ev.trace_file = None  # clear the ctor value the builder passed
    ev2 = PlanEvaluator(
        obs_0=ev.obs_0,
        obs_g=ev.obs_g,
        state_0=ev.state_0,
        state_g=ev.state_g,
        env=ev.env,
        wm=ev.wm,
        frameskip=ev.frameskip,
        seed=ev.seed,
        preprocessor=ev.preprocessor,
        n_plot_samples=0,
    )
    assert ev2.trace_file == str(tmp_path / "envtr")
    ev2.eval_actions(actions, None, filename="plan0", save_video=True)
    assert (tmp_path / "envtr_plan0.npz").exists()


def test_inner_cem_evals_do_not_write_traces(tmp_path, monkeypatch):
    """cem.py calls eval_actions once per opt step; only the checkpoint evals (the
    ones that save video) may leave a trace, or a 50-episode run writes ~300 npz."""
    ev, actions, n, _ = _build(tmp_path)
    ev.eval_actions(actions, None, filename="plan_0_output_1", save_video=False)
    assert not list(tmp_path.glob("*.npz"))
    monkeypatch.setenv("LPWM_TRACE_ALL", "1")
    ev.eval_actions(actions, None, filename="plan_0_output_1", save_video=False)
    assert (tmp_path / "tr_plan_0_output_1.npz").exists()
