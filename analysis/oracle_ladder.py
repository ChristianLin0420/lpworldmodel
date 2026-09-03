"""M3 -- the {learned, oracle} objective x {learned, oracle} dynamics 2x2, on existing checkpoints.

WHAT IT SEPARATES.  `ltv` (rel_mse 0.0092, CEM 0.357) and `linvar` (0.0095, 0.080) are
indistinguishable in one-step error and differ 4.5x in planning, so whatever CEM is limited
by, it is not one-step error.  Spearman(latent distance CEM minimises, TRUE task distance) =
+0.398 is compatible with a bad leaf metric OR a bad rollout, and only the factorial
separates them.  No training: this runs the archived checkpoints through a planner whose
inner score can be swapped, one factor at a time.

  cell                     what the CEM inner loop uses
  (learned, learned)       wm.rollout -> objective_fn.  Bit-identical to planning/cem.py; the
                           CONTROL.  It must reproduce the archived final_eval/success_rate.
  (oracle,  learned)       wm.rollout -> M1's frozen ridge decodes the terminal latent to a
                           pose -> the analytic task cost.  Isolates the OBJECTIVE.
  (learned, oracle )       true simulator -> render the terminal frame -> encode -> objective_fn.
                           Isolates the DYNAMICS.
  (oracle,  oracle )       true simulator -> the analytic task cost.  The ceiling of this
                           action space, horizon and MPC budget.

FALSIFIER.  If all four cells land within one MDE, the optimiser is the bottleneck and every
model proposal in round 5 is premature.  If (oracle obj, learned dyn) >> (learned, learned)
while (learned obj, oracle dyn) ~= (learned, learned), the objective is the defect.

=======================================================================================
THE SPEC'S PRECONDITION IS FALSE, AND THAT CHANGES THE DESIGN.
docs/round5-specs.md M3 asserts `prepare(seed, state)` is an exact restart because
`space.damping = 0` makes the sim quasi-static, and builds oracle dynamics on restarting from
`evaluator.state_0[traj]` mid-MPC.  Measured, it is not:

  * `PushTEnv._set_state` (env/pusht/pusht_env.py:650) ends with `space.step(1/sim_hz)`, so a
    restart displaces the agent by |v| * 0.01 px -- 0.638 px on the spec's own example, whose
    expected output is 0.0.
  * damping = 0 does NOT zero velocities here: the agent's persist, and the block's linear and
    angular velocity are NOT IN THE 7-VECTOR AT ALL, so a restart silently zeroes them.
  * restoring every pymunk body field by hand (position, angle, velocity, angular_velocity for
    both bodies) still does not reproduce the trajectory -- max |state| error 251 px over 30
    contact-rich episodes -- because the Space caches contact arbiters and warm-start impulses.

  Measured over 120 mid-trajectory restarts with the agent placed on the block:
    worst |block (x, y, theta)| restart error = 138.8 px      (expected 0.0)
    worst |agent (x, y)|        restart error =  15.6 px

So oracle dynamics here is REPLAY FROM THE EPISODE'S INITIAL STATE with the full action
prefix the MPC has already committed (`MPCPlanner.planned_actions`), plus the candidate
suffix.  That is exact by construction: it is the same `prepare` + `step_multiple` call
`PlanEvaluator.eval_actions` makes, from the same state, with the same actions.  It costs
~5.5x more simulator steps than a restart would; at 0.25 ms/step (render disabled) that is
~43 CPU-hours for the full 50-episode / 30-opt-step / 300-sample / 10-iteration budget, which
is why `--workers` defaults high and the job asks for many CPUs.

`--restart-mode restart` keeps the spec's cheaper-but-wrong version available for measuring
exactly how much it costs.  It is not the default and no reported cell should use it.
=======================================================================================

PRIVILEGED STATE.  The simulator, `state_0`, `state_g` and the analytic cost are all
privileged.  That is legal: M3 is measurement and evaluation, never a training loss.

Usage
    # the four cells, one checkpoint each
    python analysis/oracle_ladder.py --run LpWM-ltv_pd384_bf16_s3 --seed 3 \
        --dynamics oracle --objective oracle --n-evals 50 --max-iter 10 --workers 96
    # precondition + control checks only, no planning
    python analysis/oracle_ladder.py --check
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from multiprocessing.context import Process
import multiprocessing

import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from planning.cem import CEMPlanner                            # noqa: E402
from utils import move_to_device                               # noqa: E402

# state layout, from datasets/pusht_dset.py / PushTEnv._get_obs
AX, AY, BX, BY, TH, VX, VY = range(7)
POS_TOL, ANG_TOL = 20.0, np.pi / 9      # env/pusht/pusht_wrapper.py:62-64, the success test

_EMPTY = np.zeros((1, 1, 3), dtype=np.uint8)


# =======================================================================================
# render-free simulator pool
# =======================================================================================
def _sim_worker(conn, env_kwargs):
    """One PushTWrapper, rendering shut off at the INSTANCE level.

    `PushTEnv.step` calls `self._render_frame("rgb_array")` unconditionally
    (env/pusht/pusht_env.py:513) and that render is 1.35 of the 1.79 ms a step costs.
    Shadowing the bound method with an instance attribute removes it without touching
    env/pusht/pusht_env.py, which this proposal does not own.  The original stays reachable
    as `real_render` for the ONE terminal frame the learned-objective cell needs.
    """
    from env.pusht.pusht_wrapper import PushTWrapper
    env = PushTWrapper(**env_kwargs)
    real_render = env._render_frame                 # bound class method, captured first
    env._render_frame = lambda mode: _EMPTY
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            seed, s0, prefix, cands, want_visual, shape = msg
            if shape is not None:
                env.shape = shape
            n = len(cands)
            S = np.empty((n, 7), dtype=np.float32)
            V = (np.empty((n, env.render_size, env.render_size, 3), dtype=np.uint8)
                 if want_visual else None)
            for i in range(n):
                env.prepare(seed, s0)
                for a in prefix:
                    env.step(a)
                for a in cands[i]:
                    env.step(a)
                S[i] = env._get_obs()
                if want_visual:
                    V[i] = real_render("rgb_array")
            conn.send((S, V))
    finally:
        conn.close()


class SimPool:
    """Fan a batch of candidate action sequences out over worker processes.

    Forked AFTER the model is on the GPU, exactly as `SubprocVectorEnv` already is in
    plan.py; the children never touch CUDA.
    """

    def __init__(self, n_workers, env_kwargs=None):
        env_kwargs = env_kwargs or {}
        self.n = int(n_workers)
        self.pipes, self.procs = [], []
        for _ in range(self.n):
            a, b = multiprocessing.Pipe()
            p = Process(target=_sim_worker, args=(b, env_kwargs), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)

    def terminal(self, seed, s0, prefix, cands, want_visual=False, shape=None):
        """(states (n, 7), visuals (n, H, W, 3) or None) after prefix ++ each candidate."""
        n = len(cands)
        bounds = np.linspace(0, n, self.n + 1).round().astype(int)
        live = []
        for w in range(self.n):
            lo, hi = bounds[w], bounds[w + 1]
            if hi <= lo:
                continue
            self.pipes[w].send((seed, s0, prefix, cands[lo:hi], want_visual, shape))
            live.append(w)
        Ss, Vs = [], []
        for w in live:
            S, V = self.pipes[w].recv()
            Ss.append(S)
            Vs.append(V)
        return np.concatenate(Ss), (np.concatenate(Vs) if want_visual else None)

    def close(self):
        for p in self.pipes:
            try:
                p.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


# =======================================================================================
# the frozen pose decoder for the oracle objective on LEARNED dynamics
# =======================================================================================
def fit_pose_decoder(model, cache_path, device, lam_grid="-3:10", batch=128):
    """Ridge from the LINKED latent to [ax, ay, bx, by, cos th, sin th].

    This is M1's probe with the agent columns added, because the success test the ladder is
    scored by (`env/pusht/pusht_wrapper.py:62`) measures `norm(goal[:4] - cur[:4])` -- it
    scores the AGENT's position as well as the block's.  A block-only decoder would make the
    oracle objective optimise something the reported metric does not reward, and the cell
    would be uninterpretable.

    Fitted here, in-process, against THIS checkpoint, so the decoder handed to CEM is
    provably the decoder of the encoder being planned with.
    """
    from analysis.latent_probe import (_eig_solver, _predict, _weights, group_kfold, encode)
    cache = torch.load(cache_path)
    st = cache["state"].numpy().astype(np.float64)
    Y = np.stack([st[:, AX], st[:, AY], st[:, BX], st[:, BY],
                  np.cos(st[:, TH]), np.sin(st[:, TH])], 1)
    role = np.asarray(cache["role"])
    groups = np.array([f"{s}:{e}" for s, e in zip(cache["split"], cache["ep"])])
    fit = role == "fit"
    _, Z = encode(model, cache, device, batch)

    Yt = torch.as_tensor(Y, device=device, dtype=torch.float64)
    tr = torch.as_tensor(fit, device=device)
    mu = Z[tr].mean(0, keepdim=True)
    sd = Z[tr].std(0, keepdim=True)
    sd = sd.masked_fill(sd < 1e-6, float("inf"))   # reprelu dead units -> 0, not 1e6
    Zc = (Z - mu) / sd
    ymu = Yt[tr].mean(0, keepdim=True)
    Yc = Yt - ymu

    lo, hi = (int(x) for x in lam_grid.split(":"))
    lams = list(10.0 ** np.arange(lo, hi))
    ysd = Yc[tr].std(0, keepdim=True).clamp_min(1e-12)
    cv = np.zeros(len(lams))
    for a_m, b_m in group_kfold(groups[fit], k=5, seed=0):
        a = torch.as_tensor(a_m, device=device)
        b = torch.as_tensor(b_m, device=device)
        sol = _eig_solver(Zc[tr][a], Yc[tr][a])
        for j, lam in enumerate(lams):
            cv[j] += float((((_predict(sol, lam, Zc[tr][b]) - Yc[tr][b]) / ysd) ** 2).mean())
    lam = float(lams[int(np.argmin(cv))])
    sol = _eig_solver(Zc[tr], Yc[tr])
    W = _weights(sol, lam)

    diag = {}
    for r in ("val", "heldout"):
        m = role == r
        if not m.any():
            continue
        mt = torch.as_tensor(m, device=device)
        p = ((Zc[mt].double() @ W) + ymu).cpu().numpy()
        ang = np.degrees(np.abs((np.arctan2(p[:, 5], p[:, 4]) - st[m, TH] + np.pi)
                                % (2 * np.pi) - np.pi))
        diag[f"{r}_err_ang_deg"] = float(np.median(ang))
        diag[f"{r}_err_block_px"] = float(np.median(
            np.linalg.norm(p[:, 2:4] - st[m][:, [BX, BY]], axis=1)))
        diag[f"{r}_err_agent_px"] = float(np.median(
            np.linalg.norm(p[:, 0:2] - st[m][:, [AX, AY]], axis=1)))
    del Z, Zc
    torch.cuda.empty_cache()
    return dict(W=W, mu=mu, sd=sd, ymu=ymu, lam=lam, diag=diag)


# =======================================================================================
# the planner
# =======================================================================================
class LadderCEMPlanner(CEMPlanner):
    """CEMPlanner with the inner score factored out, so one factor can be swapped at a time.

    `plan` below is `planning/cem.py:plan` with its inline `no_grad` rollout + `objective_fn`
    replaced by `self._score(...)`, and NOTHING else moved.  At the defaults
    (`dynamics='learned'`, `objective='learned'`) `_score` is that inline block verbatim, so
    the control cell is bit-identical -- `tests/test_oracle_ladder.py` asserts it.
    """

    def __init__(self, *a, dynamics="learned", objective="learned",
                 probe_cache=None, workers=32, restart_mode="replay", **kw):
        super().__init__(*a, **kw)
        assert dynamics in ("learned", "oracle")
        assert objective in ("learned", "oracle", "oracle_block")
        assert restart_mode in ("replay", "restart")
        self.dynamics = dynamics
        self.objective = objective
        self.probe_cache = probe_cache
        self.workers = int(workers)
        self.restart_mode = restart_mode
        self.mpc = None            # set by the driver: the exact committed action prefix
        self.init_state_0 = None   # set by the driver: the EPISODE's original init state
        self.env_shape = None
        self._pool = None
        self._dec = None
        self._sim_calls = 0
        self._sim_steps = 0

    # ---- lazily built so the (learned, learned) control constructs nothing at all -----
    @property
    def pool(self):
        if self._pool is None:
            self._pool = SimPool(self.workers)
            print(f"[ladder] simulator pool: {self.workers} workers", flush=True)
        return self._pool

    @property
    def decoder(self):
        if self._dec is None:
            t0 = time.time()
            self._dec = fit_pose_decoder(self.wm, self.probe_cache, self.device)
            print(f"[ladder] pose decoder fitted in {time.time()-t0:.0f}s  "
                  f"lam={self._dec['lam']:.0e}  {self._dec['diag']}", flush=True)
        return self._dec

    # ---- pieces -----------------------------------------------------------------------
    def _exec_actions(self, action):
        """CEM's normalised (n, T, f*d) -> the env's raw (n, T*f, d).

        The same two lines `PlanEvaluator.eval_actions` uses (planning/evaluator.py:110-114),
        so the simulated trajectory is the one the evaluator will replay.
        """
        a = rearrange(action.detach().cpu(), "b t (f d) -> b (t f) d", f=self.evaluator.frameskip)
        return self.preprocessor.denormalize_actions(a).numpy()

    def _prefix(self, traj):
        """The raw actions the MPC has already committed for `traj`, or an empty (0, d)."""
        pa = getattr(self.mpc, "planned_actions", None) if self.mpc is not None else None
        if not pa:
            return np.zeros((0, 2), dtype=np.float32)
        return self._exec_actions(torch.cat(pa, dim=1)[traj:traj + 1])[0]

    def _simulate(self, action, traj):
        """True terminal states (and frames) for every candidate.  See the module header."""
        cands = self._exec_actions(action)
        if self.restart_mode == "replay":
            # NB NOT evaluator.get_init_cond(): MPCPlanner.plan reassigns the evaluator's
            # init condition to the CURRENT real state at the end of every iteration
            # (planning/mpc.py:118-121), so by the time the sub-planner runs it no longer
            # holds the episode's original state.  The driver hands us that separately.
            s0 = np.asarray(self.init_state_0[traj], dtype=np.float64)
            prefix = self._prefix(traj)
        else:                                    # the spec's version; measured non-exact
            s0 = np.asarray(self.evaluator.state_0[traj], dtype=np.float64)
            prefix = np.zeros((0, 2), dtype=np.float32)
        self._sim_calls += 1
        self._sim_steps += len(cands) * (len(prefix) + cands.shape[1])
        return self.pool.terminal(int(self.evaluator.seed[traj]), s0, prefix, cands,
                                  want_visual=(self.objective == "learned"),
                                  shape=self.env_shape)

    def _oracle_cost(self, ax, ay, bx, by, cos, sin, traj):
        """The success test of env/pusht/pusht_wrapper.py:62, made continuous.

        `success` is `norm(goal[:4] - cur[:4]) < 20 and wrapped angle < pi/9`, so the cost is
        those two terms each divided by their own tolerance and added -- it is below 2 exactly
        where a "both tolerances met" region lives, and it is what the reported metric rewards.
        `objective='oracle_block'` drops the agent columns: that is the spec's literal formula
        and it optimises the BLOCK-ONLY task, which the archive's success rate does not score.
        """
        g = torch.as_tensor(np.asarray(self.evaluator.state_g[traj], dtype=np.float32),
                            device=ax.device, dtype=ax.dtype)
        d = torch.stack([ax - g[AX], ay - g[AY], bx - g[BX], by - g[BY]], dim=1)
        if self.objective == "oracle_block":
            d = d[:, 2:]
        dth = torch.atan2(sin, cos) - g[TH]
        dth = torch.abs(torch.atan2(torch.sin(dth), torch.cos(dth)))
        return d.norm(dim=1) / POS_TOL + dth / ANG_TOL

    def _encode_states(self, S, V):
        """Encode the TRUE terminal frames into the linked space the objective compares in."""
        obs = {"visual": V[:, None], "proprio": S[:, [AX, AY, VX, VY]][:, None]}
        t = move_to_device(self.preprocessor.transform_obs(obs), self.device)
        with torch.no_grad():
            return self.wm.encode_obs_linked(t)

    # ---- the swappable inner score -----------------------------------------------------
    def _score(self, action, cur_trans_obs_0, cur_z_obs_g, traj):
        """(num_samples,) cost CEM sorts on."""
        if self.dynamics == "learned":
            with torch.no_grad():
                i_z_obses, i_zs = self.wm.rollout(
                    obs_0=cur_trans_obs_0,
                    act=action,
                    z_goal=cur_z_obs_g["visual"],
                )
            if self.objective == "learned":
                return self.objective_fn(i_z_obses, cur_z_obs_g)
            p = self.decoder
            z = i_z_obses["visual"][:, -1].flatten(1)
            pred = (((z - p["mu"]) / p["sd"]).double() @ p["W"]) + p["ymu"]
            return self._oracle_cost(pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3],
                                     pred[:, 4], pred[:, 5], traj).float()

        S, V = self._simulate(action, traj)
        if self.objective == "learned":
            return self.objective_fn(self._encode_states(S, V), cur_z_obs_g)
        t = torch.as_tensor(S, device=self.device)
        return self._oracle_cost(t[:, AX], t[:, AY], t[:, BX], t[:, BY],
                                 torch.cos(t[:, TH]), torch.sin(t[:, TH]), traj)

    # ---- planning/cem.py:plan, with the inline score replaced by _score ----------------
    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs_linked(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]
                loss = self._score(action, cur_trans_obs_0, cur_z_obs_g, traj)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1})
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu, np.full(n_evals, np.inf)


# =======================================================================================
# driver
# =======================================================================================
def build_cfg(run, seed, n_evals, max_iter, dynamics, objective, probe_cache, workers,
              restart_mode, ckpt_base, goal_h=5, opt_steps=None, num_samples=None):
    """`conf/plan_lewm.yaml` with the sub_planner retargeted at LadderCEMPlanner.

    `MPCPlanner.__init__` (planning/mpc.py:45-54) instantiates `sub_planner` through hydra
    from this dict, so the extra keys arrive as ctor kwargs and nothing under planning/ or
    conf/ has to change.  The `hydra:` block is dropped before resolving because its
    `${now:...}` / `${replace_slash:...}` interpolations only exist under `@hydra.main`.
    """
    cfg = OmegaConf.load(os.path.join(REPO, "conf", "plan_lewm.yaml"))
    del cfg["hydra"]
    d = OmegaConf.to_container(cfg, resolve=True)
    d.update(ckpt_base_path=ckpt_base, model_name=run, model_epoch="latest",
             n_evals=int(n_evals), seed=int(seed), goal_H=int(goal_h),
             wandb_logging=False, ensemble_members=None)
    d["planner"]["max_iter"] = int(max_iter)
    sp = d["planner"]["sub_planner"]
    sp["target"] = "analysis.oracle_ladder.LadderCEMPlanner"
    sp.update(dynamics=dynamics, objective=objective, probe_cache=probe_cache,
              workers=int(workers), restart_mode=restart_mode)
    if opt_steps:
        sp["opt_steps"] = int(opt_steps)
    if num_samples:
        sp["num_samples"] = int(num_samples)
    return d


def run_cell(args):
    import plan as planmod

    tag = f"Oracle-{args.objective}o-{args.dynamics}d"
    if getattr(args, "tag_extra", ""):
        tag += f"-{args.tag_extra}"      # keeps two budgets of the same cell distinguishable
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # analysis/collect_evals.py:38 parses "<stamp>_<run>_gH<h>" and then the usual
    # "<arm>_pd<d>_<prec>_s<seed>", so the label has to keep that shape: drop the base run's
    # own _s<seed> instead of appending a second one.
    base = re.sub(r"_s\d+$", "", args.run)
    out = os.path.join(REPO, "plan_outputs",
                       f"{stamp}_{tag}-{base}_s{args.seed}_gH{args.goal_h}")
    os.makedirs(out, exist_ok=True)

    cfg = build_cfg(args.run, args.seed, args.n_evals, args.max_iter, args.dynamics,
                    args.objective, args.probe_cache, args.workers, args.restart_mode,
                    args.ckpt_base, args.goal_h, args.opt_steps, args.num_samples)
    cfg["saved_folder"] = out

    base = planmod.PlanWorkspace

    class _LadderWorkspace(base):
        """Hands the sub-planner the two things `MPCPlanner` does not pass it: the parent
        (for the committed action prefix -- see the module header on why a restart is not
        exact) and the block shape the vector env was configured with."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            sp = self.planner.sub_planner
            sp.mpc = self.planner
            sp.init_state_0 = np.array(self.state_0, dtype=np.float64)   # never mutated here
            shapes = set(getattr(self.dset, "shapes", ["T"]))
            assert len(shapes) == 1, f"mixed block shapes {shapes}; sim pool needs one"
            sp.env_shape = shapes.pop()
            print(f"[ladder] dynamics={sp.dynamics} objective={sp.objective} "
                  f"restart_mode={sp.restart_mode} shape={sp.env_shape} "
                  f"opt_steps={sp.opt_steps} num_samples={sp.num_samples} "
                  f"horizon={sp.horizon} n_evals={self.n_evals} "
                  f"max_iter={self.planner.max_iter}", flush=True)

    planmod.PlanWorkspace = _LadderWorkspace
    cwd = os.getcwd()
    os.chdir(out)
    t0 = time.time()
    try:
        logs = planmod.planning_main(cfg)
    finally:
        planmod.PlanWorkspace = base
        os.chdir(cwd)
    print(f"[ladder] {tag} {args.run} s{args.seed} -> "
          f"final_eval/success_rate={logs.get('final_eval/success_rate')} "
          f"({time.time()-t0:.0f}s)  dir={out}", flush=True)
    with open(os.path.join(out, "ladder.json"), "w") as f:
        json.dump({"cell": tag, "run": args.run, "seed": args.seed,
                   "dynamics": args.dynamics, "objective": args.objective,
                   "restart_mode": args.restart_mode,
                   **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                      for k, v in logs.items()}}, f, indent=1)
    return logs


# =======================================================================================
# preconditions
# =======================================================================================
def check(workers=4):
    """Everything that has to hold before any cell is believable.  Prints real numbers."""
    from env.pusht.pusht_wrapper import PushTWrapper as W
    ok = True
    e = W()

    # (0) the spec's own precondition, verbatim
    s0 = np.array([278., 412., 235., 306., .84, 0, 0], np.float32)
    a = np.random.RandomState(0).randn(20, 2) * .2
    _, S = e.rollout(7, s0, a)
    _, S2 = e.rollout(7, S[10], a[10:])
    d = float(np.abs(S[10:] - S2).max())
    print(f"[check] spec precondition  max|S[10:]-S2| = {d:.6g}   (spec says 0.0)")

    # (1) how wrong a mid-trajectory restart is when the block is actually in play
    rs = np.random.RandomState(0)
    wb = wa = 0.0
    for _ in range(20):
        bx, by, th = rs.randint(150, 350), rs.randint(150, 350), rs.randn()
        s = np.array([bx - 40., by - 40., bx, by, th % (2 * np.pi), 0, 0], np.float32)
        acts = rs.randn(20, 2) * 1.5
        _, R = e.rollout(7, s, acts)
        for k in (5, 10, 15):
            _, R2 = e.rollout(7, R[k], acts[k:])
            wb = max(wb, float(np.abs(R[k:, 2:5] - R2[:, 2:5]).max()))
            wa = max(wa, float(np.abs(R[k:, :2] - R2[:, :2]).max()))
    print(f"[check] restart error with block contact: block {wb:.4g} px  agent {wa:.4g} px")

    # (2) the pool's replay-from-init IS the evaluator's rollout, bit for bit
    pool = SimPool(workers)
    try:
        rs = np.random.RandomState(1)
        worst = 0.0
        for _ in range(6):
            bx, by, th = rs.randint(150, 350), rs.randint(150, 350), rs.randn()
            s = np.array([bx - 40., by - 40., bx, by, th % (2 * np.pi), 0, 0], np.float32)
            pre = rs.randn(10, 2) * 1.5
            cand = rs.randn(3, 25, 2) * 1.5
            full = np.concatenate([np.repeat(pre[None], 3, 0), cand], axis=1)
            ref = np.stack([e.rollout(7, s, full[i])[1][-1] for i in range(3)])
            got, _ = pool.terminal(7, s.astype(np.float64), pre, cand)
            worst = max(worst, float(np.abs(ref - got).max()))
        print(f"[check] pool replay vs env.rollout terminal state: max|diff| = {worst:.6g}"
              f"   {'OK' if worst == 0.0 else 'MISMATCH'}")
        ok &= worst == 0.0
    finally:
        pool.close()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="LpWM-ltv_pd384_bf16_s3")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dynamics", choices=["learned", "oracle"], default="learned")
    ap.add_argument("--objective", choices=["learned", "oracle", "oracle_block"],
                    default="learned")
    ap.add_argument("--restart-mode", choices=["replay", "restart"], default="replay")
    ap.add_argument("--n-evals", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--goal-h", type=int, default=5)
    ap.add_argument("--tag-extra", default="",
                    help="appended to the cell label, so a second BUDGET of the same cell "
                         "does not overwrite the first in analysis/collect_evals.py")
    ap.add_argument("--opt-steps", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--probe-cache", default=os.path.join(REPO, "runs", "probe_cache.pt"))
    ap.add_argument("--ckpt-base", default=os.environ.get("CKPT_BASE",
                                                          os.path.join(REPO, "runs")))
    ap.add_argument("--check", action="store_true", help="preconditions only, no planning")
    ap.add_argument("--all-cells", action="store_true",
                    help="run the whole 2x2 in one process (canary use; production gives "
                         "the two oracle-dynamics cells their own job and their own CPUs)")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    if not a.all_cells:
        run_cell(a)
        return
    table = {}
    for obj in ("learned", "oracle"):
        for dyn in ("learned", "oracle"):
            a.objective, a.dynamics = obj, dyn
            logs = run_cell(a)
            table[f"{obj}o/{dyn}d"] = logs.get("final_eval/success_rate")
    print("\n[ladder] 2x2  (rows = objective, cols = dynamics)")
    for obj in ("learned", "oracle"):
        print("   %-8s " % obj + "  ".join(
            f"{dyn}={table[f'{obj}o/{dyn}d']}" for dyn in ("learned", "oracle")))


if __name__ == "__main__":
    main()
