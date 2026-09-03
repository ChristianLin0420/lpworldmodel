# Round 5 — design specifications

One implementable spec per proposal. Each carries the claim, the derivation, the exact files and
insertion points, the config keys, the control arm, the falsifier that would kill it, and the
verification commands. **This is the document an implementing agent works from**: if something is
not written here, it is not decided.

Written 2026-09-03. Read `diary/2026-09-03.md` §12–§14 first — that is where the evidence behind
every claim below lives.

## What round 5 is reacting to

Four rounds, ~48 arms, ~265 evaluated runs, **one** positive result (plan-time consensus over
independently trained models, +0.228 — and it is not a representation result). The round-5
proposals exist because five things came out of the round-4 post-mortem, and every one of them
sits *underneath* the model questions the first four rounds were asking.

**1 · The success metric is not the task.** `env/pusht/pusht_wrapper.py:62` scores
`goal_state[:4] − cur_state[:4]`, and `state[:4]` is `[agent_x, agent_y, T_x, T_y]` — half the
scored positional degrees of freedom are the *agent's own* position, which the policy actuates
directly with no contact and no physics. `planning/mpc.py:110-112` then latches success
(`is_success |= successes`) across up to ten replanning checkpoints, making the headline a maximum
over ten noisy draws rather than a terminal outcome. **M2 rescores the archive; every number
predating it is historical.**

**2 · The objective is a mean over transitions; the task is a tail.** Over 915,565 transitions at
the model's own frameskip, the block's median displacement is **exactly 0.000 px**, 75 % of
transitions move it less than half a pixel, and the top 1 % carry 34.8 % of all its motion. A
1-step MSE spends nearly all of its gradient where nothing happens. `LpWM-ltv` and `LpWM-linvar`
differ by 3 % in `rel_mse` and by 4.5× in planning. **T3, T5 and T6 are three different ways of
not averaging over the no-ops.**

**3 · `d_action` — the quantity two rounds were designed around — was measured wrong.** The
baseline's value was never logged (the diagnostic postdates those runs) and the number every
design used, 1.9 × 10⁻⁴, came from one hand-measured seed in a docstring. Re-measured over all 327
checkpoints by `analysis/d_action_probe.py`, `LpWM-ltv` sits at **0.549** — a factor of ~2900. The
relation to planning is an **inverted U** with its optimum near 0.6, so the baseline was already
just below the peak and every arm built to raise it in fact lowered it (`actgain 3.0` → 0.81×).

> **Standing consequence for this round.** "Raise `d_action`" is not a direction and no proposal
> here may use it as a motivation. It *is* a validated selection statistic: among predictors with
> `rel_mse` < 0.05 it orders checkpoints by planning success at ρ = +0.70, for one forward pass
> against four GPU-hours per CEM eval. Use it to choose which checkpoints to evaluate, never as an
> objective. See diary §12b.

**4 · A component with no optimizer and no checkpoint key is not a component.** `path_int` had
neither, is absent from every checkpoint on disk, and its loss was built from data-only leaves, so
it back-propagated into nothing — and at plan time `_roll_pose` constructed a fresh *randomly
initialised* `nn.Linear`. V3 was published as "path integration is a null"; it never ran.

> **NEW HEAD CHECKLIST — mandatory for every module added in this round.** An optimizer group in
> `train.py init_optimizers`; a `_keys_to_save` entry; a lazy `.to(device)` in forward (modules
> built inside `VWorldModel` are never `accelerator.prepare()`d); a `tests/lpwm_build.py` mirror;
> a `tests/test_arms.py` entry; a gradient-reaches-parameters assertion; and the parameter must be
> shown present in a real saved checkpoint. Plus: no new `loss_components` key at defaults
> (`tests/test_bit_identity.py` asserts `set(g) == set(w)`), and all `loss_components` values must
> be **tensors** (`train.py:1096-1099` gathers them).

**5 · Nothing has ever required the latent to retain visual content.** `has_decoder: False`
throughout (`conf/train_rdmreg.yaml:126`), and the decoder branch that exists passes
`z_emb.detach()` (`visual_world_model.py:728`). The only pressure on the encoder is that its own
output be predictable from its own past, whose global optimum is a constant; RDMReg blocks
*dimensional* collapse, not *informational* collapse. Four separate InfoNCE-style arms collapsing
to `rel_mse` = 1.0 is the predicted behaviour of this configuration, not four accidents.
**T1 and T2 test it directly.**

## Constraints that bind every spec here

- **Images and proprioception only.** `states.pth` — the privileged block pose — may be used for
  measurement, diagnostics and evaluation. It may **not** appear in any training loss. M1 and M4
  use it; no T-proposal may. T3 is specifically designed to get a contact signal without it.
- **CEM is inherited, not required.** It is the planner the LeWM/LpWM line uses, so it stays as
  the common yardstick. A reactive policy that beats it (V4) is an acceptable and interesting
  result, not a failure.
- **No fixed gates and no fitted thresholds.** Report effects with intervals; an uninformative
  interval is not a null.
- **Launch only via `scripts/run_campaign.sh`** — it exports `PRECISION=bf16` and
  `WANDB_PROJECT=PiWM-pushT`, and `train_slurm.sbatch` refuses a name/precision mismatch.
- **Eval completion is the terminal marker `final_eval/success_rate` in `logs.json`** — never a
  submission log line and never the mere existence of the file. Both have misled this campaign.
- **Dependencies are the only ordering.** T4 → V5 and T4 → T5 are the only ones. Everything else
  runs in parallel.

## Index

| # | name | kind | depends on |
|---|---|---|---|
| M1 | latent probe: can `z` decode the block pose at all? | measurement | — |
| M2 | metric audit: block-only, terminal, + per-episode traces | measurement | — |
| M3 | oracle ladder: {learned, oracle} objective × dynamics | measurement | — |
| M4 | error analysis: 200 paired episodes, four pre-declared counts | measurement | M2 |
| V1 | pessimistic consensus (`cvar`, `max`) | planner | — |
| V2 | gradient planner as a first-class arm | planner | — |
| V3 | two-level contact planner (subgoal × short controller) | planner | — |
| V4 | reactive goal-conditioned policy, no search | planner | T4 |
| V5 | value at the CEM leaves | planner | T4 |
| T1 | attach the decoder | training | — |
| T2 | patch tokens × per-patch decoder | training | — |
| T3 | proprio-residual contact weighting | training | — |
| T4 | hindsight goal-conditioned value by expectile TD | training | — |
| T5 | value-equivalent model loss | training | T4 |
| T6 | jumpy K=5 option model | training | — |
| T7 | frozen-encoder energy / ranking head | training | — |

Figures for every proposal are in `diary/assets/2026-09-03/`, generated by
`analysis/round5_figs_{measure,planners,repr,obj}.py`.

---

## M1 - probe: ridge-decode the block pose from frozen z across all 327 checkpoints

**Claim.** A linear probe on the frozen latent measures whether z contains the block pose *at all*, independent of dynamics. If held-out median angular error exceeds 20 deg, no predictor can plan the task and the whole dynamics campaign is optimising a quantity the representation cannot express.

**Derivation.** Spearman(latent distance CEM minimises, TRUE task distance) = +0.398 (n=296). That is produced either by bad dynamics or by a latent that never encoded the block, and the 1-step contrast cannot separate them: ltv rel_mse 0.0092 / CEM 0.357 vs linvar 0.0095 / CEM 0.080 — indistinguishable prediction error, 4.5x planning. Decodability of the block pose from a *single frozen frame* needs no dynamics at all, so it is the missing axis.

**Change.** Two new files, no model parameters, no training config.

`analysis/probe_cache.py` (run once). Uses `datasets/pusht_dset.py:105 get_frames`, which already applies `default_transform(224)` — the exact tensor `encode_obs_linked` consumes.
```python
ds = PushTDataset(data_path=f"{DATASET_DIR}/pusht_noise/{split}", transform=default_transform(224))
for ep in eps:                                   # ep = trajectory index = the split unit
    fr = list(range(0, ds.get_seq_length(ep), stride))
    obs, _, st, _ = ds.get_frames(ep, fr)        # st (F,7): [ax,ay,bx,by,theta,vx,vy]
    X.append(obs["visual"].half()); Y.append(st); E.append(np.full(len(fr), ep))
torch.save({"visual":..., "state":..., "ep":..., "split":...}, "runs/probe_cache.pt")
```
Fit on 300 TRAIN episodes (stride 4, ~9k frames); report on the 21 VAL episodes (`val/seq_lengths.pkl` has exactly 21 trajectories — verified) plus 150 held-out TRAIN episodes.

`analysis/latent_probe.py`. Copy the sweep loop of `analysis/spectral.py:129-156` verbatim (`RUN_RE` at :42, `CAMPAIGN_ALIAS` at :43, `glob runs/outputs/*/`); replace `_predictor_state` with `plan.load_model` (plan.py:422) fed `OmegaConf.load(run/"hydra.yaml")`. All 327 runs are img_size 224 / frameskip 5 / num_hist 3 (verified), so one cache serves every checkpoint.
```python
Z = cat([model.encode_obs_linked({"visual": xb[:,None].to(dev)})["visual"][:,0].flatten(1)
         for xb in X.split(128)])                      # (N, p*d): 384 cls, 98304 patch
Xc = (Z-mu)/sd;  K = Xc@Xc.T                           # dual ridge: N << p*d for patch runs
a  = solve(K + lam*I, Ytr);  Yhat = Kte @ a
err_ang = median(|wrap(atan2(sin_hat,cos_hat) - theta)|) in deg   # NEVER R2 on raw theta
err_pos = median(||Yhat[:,:2] - state[:,2:4]||)                   # px
```
`lam` by GroupKFold(5) **grouped on episode**, grid `10.0**arange(-3,6)`. Also probe pre-link `encode_obs` to separate "the link destroys it" from "the encoder never had it". Ensemble cell: concat the 5 standardised Z of `LpWM-ltv` seeds 3-7 and 8-12 (the actual `PiWM-vote5-median` members, from `slurm_logs/eval_PiWM-vote5-median_pd384_bf16_s3_33643422.out`), 1920 features.

**Wiring.** No `run_campaign.sh`/`train.sh`/conf chain: this adds no model kwarg, so `tests/lpwm_build.py:84-120`, `tests/test_arms.py` and `tests/test_bit_identity.py` are untouched by construction. CLI only:
`python analysis/probe_cache.py --out runs/probe_cache.pt --train-eps 300 --stride 4`
`python analysis/latent_probe.py --cache runs/probe_cache.pt --out probe.json --campaign campaign_fixed.json`
Join key `(arm, seed)` from `RUN_RE`; `campaign_fixed.json` supplies 133 CEM cells over 18 arms.

**Control.** `C_agent`: ridge from `[ax,ay,vx,vy]` alone. Measured over 2,000 archive episodes, the agent carries 90.9% of the initial `pos_diff^2` and 34.0% of goals are already inside the block-only tolerance at t=0, so agent-only decoding is the null any "z knows the block" claim must beat. Also `C_shuf` (labels permuted across episodes), `C_rand` (a freshly built `ViTEncoder` from the same `cfg.encoder`, never loaded — capacity-matched), and `C_resid` (z residualised on `[ax,ay,vx,vy,1]`).

**Falsifier.** Median angular error > 20 deg on the 21 held-out val episodes for every one of the 327 checkpoints kills every dynamics proposal. Second falsifier: if probe error and CEM are uncorrelated over the 133 cells (|Spearman| < 0.3, p > 0.05), representation quality is not the binding constraint either and M3 becomes the only live explanation.

**Verification.**
```bash
python analysis/probe_cache.py --out /tmp/pc.pt --train-eps 4 --stride 20 --splits val
python - <<'P'   # cache is the tensor the model actually sees
import torch; c=torch.load('/tmp/pc.pt'); print(c['visual'].shape, c['visual'].dtype,
  c['visual'].min().item(), c['visual'].max().item(), c['state'].shape, set(c['ep'].tolist()))
P
python analysis/latent_probe.py --cache /tmp/pc.pt --arm LpWM-ltv --limit 2 --out /tmp/p.json
python -c "import json;r=json.load(open('/tmp/p.json'));print(r[0]['err_ang_deg'],r[0]['ctrl_rand_err_ang_deg'],r[0]['ctrl_shuf_err_ang_deg'])"
# LIVE iff ctrl_shuf ~ 90 deg (chance for a wrapped angle) and trained != rand by > 2 s.e.
```

## M2 - metric: block-only pos_diff, terminal (not latched) success, IoU secondary, per-episode traces

**Claim.** The reported success rate is 90.9% a statement about where the *agent* is, and it is latched. Rescore the archive and persist per-episode traces so no future eval collapses to a mean before it is inspected.

**Derivation.** `env/pusht/pusht_wrapper.py:62` takes `pos_diff = norm(goal_state[:4]-cur_state[:4])` over `[ax,ay,bx,by]`. Measured on 2,000 episodes drawn from 40 archive `plan_targets.pkl`: median agent displacement to goal 142.55 px vs median block displacement 18.65 px; the agent contributes **90.9%** of mean `pos_diff^2` at t=0; **34.0%** of goals already satisfy the block-only criterion at t=0. Latching (`planning/mpc.py:110-112`, inside `while not all(is_success)`) then freezes an episode at its best moment, while `_apply_success_mask` (mpc.py:60-70) emits raw-zero *relative* actions — the PD controller only zeroes the agent's velocity, it does not stop the block — so latched and terminal states genuinely differ.

**Change.** (1) `env/pusht/pusht_wrapper.py`, replace the body of `eval_state` (:57-70). `success` and `state_dist` keep their exact definitions so every archive number stays reproducible; new keys are added.
```python
d = goal_state[:4]-cur_state[:4]; pos=norm(d); ag=norm(d[:2]); blk=norm(d[2:4])
ang = min(|goal_state[4]-cur_state[4]|, 2*pi-|...|)
return {"success": pos<20 and ang<pi/9,            # UNCHANGED
        "success_block": blk<20 and ang<pi/9,      # the task
        "state_dist": norm(goal_state-cur_state),  # UNCHANGED
        "block_pos_diff": blk, "agent_pos_diff": ag, "angle_diff": ang,
        "block_iou": self._block_iou(cur_state[2:5], goal_state[2:5])}
```
`_block_iou` uses `pymunk_to_shapely` (`env/pusht/pusht_env.py:345`) on two `_get_goal_pose_body` bodies. It is a pure function of two poses, so it needs no rollout widening — and unlike the env's own `final_coverage` (`pusht_env.py:524`) it scores the *planned* goal, not the fixed constant `self.goal_pose = [256,256,pi/4]` at `pusht_env.py:716`. Every new key auto-appears as `mean_<key>` through the comprehension at `planning/evaluator.py:222-225`.

(2) `planning/evaluator.py`. Change `_compute_rollout_metrics` (:209, `return` at :249) to return `(logs, successes, eval_results)`; at the call site (:127-131) add, after it:
```python
term = self.env.eval_state(self.state_g, e_states[:, -1])     # TERMINAL: not truncated by action_len
logs.update({f"terminal_{'success_rate' if k=='success' else 'mean_'+k}": float(np.mean(np.asarray(v,float)))
             for k,v in term.items()})
if self.trace_file:                                            # ctor kwarg, default None
    np.savez(f"{self.trace_file}_{filename}.npz", seed=np.asarray(self.seed),
             state_0=self.state_0, state_g=self.state_g, action_len=action_len,
             e_state_final=e_states[:,-1], e_state_latched=e_final_state,
             d_pred=torch.norm((i_final_z_obs["visual"]-z_g), dim=(1,2,3)).cpu().numpy(),
             d_real=torch.norm((e_z_obs["visual"]-z_g), dim=(1,2,3)).cpu().numpy(),
             **{k: np.asarray(v) for k,v in term.items()},
             **{f"latched_{k}": np.asarray(v) for k,v in eval_results.items()})
```
`d_pred`/`d_real` are the per-episode form of `mean_div_visual_emb`, which `evaluator.py:236` collapses with a whole-batch `torch.norm`.

(3) `analysis/rescore.py` — offline, no GPU, no env. From `plan_outputs/*/plan_targets.pkl` (present in all 320 dirs) recover per-episode required block displacement / angle change; from the filenames `output_final_real_{i}_{success|failure}.png` and `plan{k}_real_{i}_{tag}.png` recover per-episode latched outcome and latch time for episodes 0-9 (verified present). Report per arm: archive success vs success restricted to the 66.0% of episodes that require block motion.

**Wiring.** New conf key `trace_file: null` (inert) at the top level of `conf/plan_lewm.yaml`, `conf/plan_rdmreg.yaml`, `conf/plan_pusht.yaml`; read at `plan.py:168-178` into the `PlanEvaluator(...)` ctor as `trace_file=cfg_dict.get("trace_file")`. Enabled per run by `scripts/plan.sh` gaining `${TRACE:+trace_file=$TRACE}`, i.e. `TRACE=traces RUN_NAME=... sbatch scripts/plan_slurm.sbatch`. No `run_campaign.sh` ARM entry and no model kwarg, so `tests/lpwm_build.py:84-120` needs no mirror.

**Control.** The untouched `success`/`state_dist` keys. Re-running any archived (run, seed) must reproduce its recorded `final_eval/success_rate` exactly, which isolates every ranking change to the metric rather than to the re-run.

**Falsifier.** If the arm ranking under `terminal_success_block` on block-motion-required episodes is rank-identical to the archive ranking (Spearman >= 0.95 over the 18 arms of `campaign_fixed.json`), the metric defect is inert and the campaign's conclusions stand as written.

**Verification.**
```bash
python -c "
from env.pusht.pusht_wrapper import PushTWrapper as W; import numpy as np
e=W(); g=np.array([100.,100.,250.,250.,0.5,0,0],dtype=np.float32); c=g.copy(); c[:2]+=300
r=e.eval_state(g,c); print(r)"           # success False, success_block True, block_iou 1.0
python analysis/rescore.py --plan-outputs plan_outputs --out /tmp/rescore.json | tail -20
grep -n "trace_file" conf/plan_lewm.yaml plan.py planning/evaluator.py
TRACE=/tmp/tr NEVALS=4 scripts/plan.sh plan_lewm.yaml LpWM-ltv_pd384_bf16_s3 latest 4 1
python -c "import numpy as np;d=np.load('/tmp/tr_output_final.npz');print({k:d[k].shape for k in d})"
# LIVE iff the npz exists with 4-row arrays and terminal_success_rate appears in logs.json
```

## M3 - ladder: {learned, oracle} objective x {learned, oracle} dynamics on existing checkpoints

**Claim.** The 2x2 attributes the CEM failure to the objective (what the latent distance means) or to the dynamics (where the rollout goes), on checkpoints that already exist. No training.

**Derivation.** `ltv` (rel_mse 0.0092, CEM 0.357) and `linvar` (0.0095, 0.080) are indistinguishable in 1-step error and differ 4.5x in planning, so whatever CEM is limited by, it is not 1-step error. Spearman(latent distance CEM minimises, TRUE task distance) = +0.398 is then compatible with either a bad leaf metric or a bad rollout, and only the factorial separates them. (This spec previously cited a "double dissociation" between predictor nonlinearity and `d_action`. That claim is withdrawn — see diary §12b: no round-4 arm ever raised `d_action`, so the two factors remain confounded. The rel_mse/CEM mismatch above stands on its own and is all M3 needs.)

**Change.** (1) Refactor `planning/cem.py:107-114` — the `no_grad` rollout plus `loss = self.objective_fn(...)` — verbatim into `def _score(self, action, cur_trans_obs_0, cur_z_obs_g, traj) -> Tensor(num_samples,)`, called from :114. Nothing else in `plan` moves, so the learned/learned cell stays bit-identical.

(2) New `planning/oracle.py`:
```python
class OracleCEMPlanner(CEMPlanner):
    def __init__(self, *a, dynamics="learned", objective="learned", probe_file=None,
                 env=None, **kw):                      # env swallowed by CEMPlanner **kwargs today
    def _score(self, action, obs0, zg, traj):
        s0, sd = self.evaluator.state_0[traj], self.evaluator.seed[traj]   # privileged: EVAL ONLY
        g = self.evaluator.state_g[traj]
        if self.dynamics == "oracle":
            raw = self.preprocessor.denormalize_actions(
                rearrange(action.cpu(), "n t (f d) -> n (t f) d", f=self.evaluator.frameskip)).numpy()
            S = self._sim_terminal(sd, s0, raw)        # (n,7) via env.rollout_state_only
            if self.objective == "oracle":
                return norm(S[:,2:4]-g[2:4])/20 + wrap(S[:,4]-g[4])/(pi/9)
            return self.objective_fn(self.wm.encode_obs_linked(render(S)), zg)
        z, _ = super_rollout(...)                       # learned dynamics
        if self.objective == "oracle":                  # M1's frozen ridge decodes z -> block pose
            p = ((z[:,-1].flatten(1)-self.mu)/self.sd) @ self.Wp   # (n,4): x,y,cos,sin
            return norm(p[:,:2]-g[2:4])/20 + wrap(atan2(p[:,3],p[:,2])-g[4])/(pi/9)
        return self.objective_fn(z, zg)
```
(3) Render-free simulation. `env/pusht/pusht_env.py:513` becomes `visual = self._render_frame("rgb_array") if self.render_obs else _EMPTY`, with `self.render_obs = True` set beside `self.action_scale` at :390. **Measured: 1.521 ms/step -> 0.243 ms/step, 6.3x.** New `PushTWrapper.rollout_state_only(seed, init_state, actions)` after `pusht_wrapper.py:118`; new `"rollout_state_only"` worker branch after `env/venv.py:288`; `SubprocEnvWorker` method after `env/venv.py:444`; `SubprocVectorEnv` method after `env/venv.py:866`; `SerialVectorEnv` method after `env/serial_vector_env.py:94`. `rollout` itself is untouched, so nothing on the archive path can change.

(4) `planning/mpc.py:45-54`, add `env=self.env,` to the `sub_planner` instantiate. `CEMPlanner.__init__` (cem.py:25) and `GDPlanner.__init__` (gd.py:25) both take `**kwargs`, so this is inert for them.

**Wiring.** `conf/plan_oracle.yaml` = `conf/plan_lewm.yaml` plus three top-level keys with inert defaults — `dynamics: learned`, `objective_mode: learned`, `probe_file: null` — and `planner.sub_planner.target: planning.oracle.OracleCEMPlanner` with `dynamics: ${oc.select:dynamics,learned}`, `objective: ${oc.select:objective_mode,learned}`, `probe_file: ${oc.select:probe_file,null}`. New `scripts/plan_oracle.sh` runs `python plan.py --config-name plan_oracle.yaml ckpt_base_path=... model_name=$RUN dynamics=$DYN objective_mode=$OBJ probe_file=$PROBE hydra.run.dir=plan_outputs/${STAMP}_Oracle-${OBJ}o-${DYN}d_pd384_bf16_s${SEED}_gH5`. `run_campaign.sh` gains `ladder_arms()` looping the 4 cells over `LpWM-ltv_pd384_bf16_s{3,4,5}`; the label shape keeps `analysis/collect_evals.py` parsing it. Budget at `opt_steps: 30`, `num_samples: 300`, `n_evals: 50`, `max_iter: 10`: oracle dynamics is 225k sim steps/episode/iter, ~1.1 s spread over the 50 venv workers; raise `plan_slurm.sbatch:7` to `07:55:00` for the learned-objective/oracle-dynamics cell, which additionally renders and encodes 4.5M terminal frames.

**Control.** The (learned, learned) cell run *through* `OracleCEMPlanner`. It must reproduce the archived `final_eval/success_rate` for the same run and seed, and a unit test must assert `OracleCEMPlanner._score == CEMPlanner._score` output bit-for-bit at the defaults.

**Falsifier.** If (oracle, oracle) is at or below the archive's ~0.4, the defect is the action space, horizon or MPC budget, and every representation and dynamics proposal in the campaign is moot. If (oracle objective, learned dynamics) >> (learned, learned) while (learned objective, oracle dynamics) ~= (learned, learned), the objective is the defect and dynamics work should stop; the reverse pattern kills the objective work.

**Verification.**
```bash
# 0. PRECONDITION: prepare(seed,state) is an exact restart (space.damping=0 => quasi-static).
python -c "
import numpy as np; from env.pusht.pusht_wrapper import PushTWrapper as W
e=W(); s0=np.array([278.,412.,235.,306.,.84,0,0],np.float32); a=np.random.RandomState(0).randn(20,2)*.2
_,S=e.rollout(7,s0,a); _,S2=e.rollout(7,S[10],a[10:]); print(np.abs(S[10:]-S2).max())"   # must be 0.0
pytest tests/test_oracle_planner.py -q          # _score bit-identity at defaults
python -c "import hydra;from omegaconf import OmegaConf;c=OmegaConf.load('conf/plan_oracle.yaml');print(c.dynamics,c.objective_mode,c.planner.sub_planner.target)"
NEVALS=4 MAXITER=1 DYN=oracle OBJ=oracle scripts/plan_oracle.sh LpWM-ltv_pd384_bf16_s3 3 2>&1 | grep -E "Success rate|dynamics="
# LIVE iff the log prints dynamics=oracle objective=oracle AND the 4-episode success differs from the learned/learned run
```

## M4 - error analysis: 2 arms, 200 paired episodes, four counts

**Claim.** Every CEM failure is one of four things: the goal never needed the block moved, the wrong success term failed, the agent never touched the block, or the model predicted an improvement that did not happen. Count them.

**Derivation.** Success rate alone cannot discriminate: 34.0% of archive goals are already inside the block-only tolerance at t=0 (measured, 2,000 episodes), block motion is extremely heavy-tailed (top 1% of transitions carry 34.8% of all block motion, top 5% carry 77.9%), and 48% of transitions are fully static. A 0.35-vs-0.60 arm difference can be entirely a difference in how often the agent parks on the goal agent position. `planning/evaluator.py:209-249` is the single collapse point and `plan.py:343` throws `e_obses, e_states` away, so no existing artifact can answer this.

**Change.** `analysis/error_analysis.py`, consuming the M2 `.npz` traces. Per episode:
- **(a) needs block motion**: `norm(state_g[2:4]-state_0[2:4]) >= 20 or wrap(theta_g-theta_0) >= pi/9`. Archive prior: 66.0%.
- **(b) which term fails**: on `e_state_final`, the indicator triple `(agent_pos_diff>=20, block_pos_diff>=20, angle_diff>=pi/9)`; report the 2x2x2 contingency table, plus the count of `success and not success_block` (won by the agent alone) and `success_block and not success` (real task solved, scored a failure).
- **(c) zero-contact failures**: `extras["n_contacts"].sum(axis=1) == 0`. Requires `n_contacts` (`env/pusht/pusht_env.py:578`) out of the env: add `PushTWrapper.rollout_ex` after `pusht_wrapper.py:118` returning `(obses, states, {k: infos[k] for k in ("n_contacts","final_coverage","block_pose")})`; a `"rollout_ex"` worker branch after `env/venv.py:288` reusing the `obs_bufs` encode at :285-287; a `SubprocEnvWorker.rollout_ex` after :444 (`recv()` at :460-483 already handles a 3-tuple); `SubprocVectorEnv.rollout_ex` after :866; `SerialVectorEnv.rollout_ex` after :94. `planning/evaluator.py:119` opts in via `roll = getattr(self.env, "rollout_ex", None)`, falling back to `self.env.rollout` — so `rollout` is never modified.
- **(d) predicted vs realised**: from the trace, `pred_gain = d0 - d_pred`, `real_gain = d0 - d_real` (latent), `true_gain = T(s_0,s_g) - T(s_final,s_g)` with `T = norm(block)/20 + wrap(angle)/(pi/9)`. Headline: per-episode Spearman(`pred_gain`, `true_gain`) — the disaggregated form of the +0.398 aggregate — and the fraction of episodes where CEM predicted a gain and the true distance grew.

**Wiring.** Arms: `LpWM-ltv` (mean CEM 0.343 over 13 seeds) and `PiWM-vote5-median` (0.602 over 10 seeds) — the largest verified gap in `campaign_fixed.json`. 200 paired episodes as **four jobs of `NEVALS=50` at `SEED in {3,4,5,6}` per arm**, not one job of 200: `plan.py:140` builds `eval_seed = [seed*n_evals+n+1]`, so `n_evals=200` would move every episode off the archive's blocks *and* fork 200 pymunk workers at `plan.py:646`. `scripts/run_campaign.sh` gains `wave30_arms()` with `ORDER[wave30]="LpWM-ltv PiWM-vote5-median"`, submitting through `scripts/plan_slurm.sbatch` with `TRACE=traces` and `hydra.run.dir=error_analysis/<arm>_s<seed>` — outside `plan_outputs/`, so `analysis/collect_evals.py` (sorted, newest-wins) cannot overwrite the archived success rates. `PiWM-vote5-median` goes through `scripts/plan_vote.sh` with `MEMBERS` = `LpWM-ltv_pd384_bf16_s{3..7}` for SEED 3 (verified from `slurm_logs/eval_PiWM-vote5-median_pd384_bf16_s3_33643422.out`). No model kwarg, so `tests/lpwm_build.py:84-120` needs no mirror.

**Control.** Pairing. Both arms at the same `SEED` and `n_evals` draw the identical `eval_seed` block *and* the identical `sample_traj_segment_from_dset` draw (`plan.py:271-296`, seeded by `seed(cfg_dict["seed"])`), so every count is a within-episode difference. Report the (a)-restricted success rates as the primary contrast. Caveat to state in the output: the 200 episodes are segments of only **21 val trajectories** (`val/seq_lengths.pkl`), so all standard errors must be clustered on trajectory id, not on episode.

**Falsifier.** If the two arms' four counts are within Monte-Carlo error of each other at n=200 while their success rates differ by 0.26, the taxonomy does not explain the gap and the difference is planner variance, not a model property. If (a)-restricted success rates are equal across arms, the entire measured arm ordering is an artifact of agent-only goals.

**Verification.**
```bash
python -c "
import numpy as np; from env.pusht.pusht_wrapper import PushTWrapper as W
e=W(); s0=np.array([278.,412.,235.,306.,.84,0,0],np.float32)
o,s,x=e.rollout_ex(7,s0,np.random.RandomState(0).randn(25,2)*.2)
print(s.shape, {k:np.shape(v) for k,v in x.items()}, x['n_contacts'].sum())"
grep -n "rollout_ex" env/venv.py env/serial_vector_env.py planning/evaluator.py
TRACE=traces NEVALS=4 scripts/plan.sh plan_lewm.yaml LpWM-ltv_pd384_bf16_s3 latest 4 1
python analysis/error_analysis.py --traces 'traces_*.npz' --out /tmp/ea.json && cat /tmp/ea.json
# LIVE iff n_contacts is a (T,) integer array with nonzero entries AND ea.json reports all four counts on 4 episodes
DRYRUN=1 EVAL=1 SEEDS="3 4 5 6" scripts/run_campaign.sh wave30   # must list 8 jobs, all under error_analysis/
```

---

## V1 - PiWM-vote{M}-cvar / PiWM-vote{M}-max: pessimistic consensus, not variance reduction

**Claim.** The campaign's one positive is a *rank-pessimism* effect. Sharpening the combination rule from median-of-ranks toward the worst member's rank should increase success monotonically; the pure variance-reduction rule (mean-of-ranks) should not.

**Derivation.** Every vote arm ever run used `rule="median"` (all 38 dirs matching `plan_outputs/*vote*`). Paired against `LpWM-ltv` (`analysis/figures.py:344`): **M=3 → +0.165 (n=12)**, **M=5 → +0.228 (n=10)**. `PiWMvoteM1` (`rule="mean"`, M=1) scored 0.66 on block s5 against `LpWM-ltv` s5's 0.66 — **delta exactly 0.000**, so the ensemble wrapper itself is inert and the whole effect lives in the M>1 rule. Median-of-ranks is *simultaneously* variance-reduced and robust/pessimistic, and neither a mean-rank arm nor anything more pessimistic than the median exists, so the mechanism is unidentified. `borda`/`median`/`cvar`/`max` are order statistics of the **same** rank matrix `R ∈ {0..N-1}^{N×M}` built at `planning/objectives.py:109`, so the sweep holds members, episodes, budget and rollouts fixed.

**Change.** `planning/objectives.py`, in place, no new module.
- `:70` signature → `def create_vote_objective_fn(n_members, rule="mean", alpha=0, base=2, mode="last", lam=0.0):`. New key `lam`, **inert default `0.0`** — at `lam=0` `cvar` is bit-identical to `borda`.
- `:95` → `assert rule in ("mean", "borda", "median", "cvar", "max"), f"vote rule {rule} not supported"`.
- Replace `:108-110` with:
```python
        ranks = per.argsort(dim=0).argsort(dim=0).to(per.dtype)   # (B, M)
        if rule == "borda":
            return ranks.mean(dim=1)
        if rule == "median":
            return ranks.median(dim=1).values
        if rule == "max":                      # minimax: score = worst member's rank
            return ranks.max(dim=1).values
        # cvar: mean rank + lam * disagreement.  lam == 0 reproduces borda exactly.
        return ranks.mean(dim=1) + lam * ranks.std(dim=1, unbiased=False)
```
`unbiased=False` is load-bearing, not cosmetic: `PiWMvoteM1` is a live M=1 arm and the unbiased std of one sample is `nan`, which would poison every candidate score. Extend the `:78-93` docstring with the two rules.
- `scripts/plan_vote.sh:32` → append `${LAM:+ +objective.lam=${LAM}}`; unset ⇒ key absent ⇒ the 0.0 default.
- New `tests/test_vote_rules.py`: `torch.equal(f(5,"cvar",lam=0.0)(p,g), f(5,"borda")(p,g))`; `max >= borda` elementwise; at M=1 all five rules induce the same ordering as raw MSE.

No parameters are created, so the NEW HEAD CHECKLIST does not apply — no optimizer group, no `_keys_to_save`, no `tests/lpwm_build.py` / `tests/test_arms.py` mirror. The training graph is untouched.

**Wiring.** Plan-time only; `run_campaign.sh` / `train.sh` / `conf/train_rdmreg.yaml` are not involved. `MEMBERS,LABEL,RULE,LAM,SEED` → `scripts/vote_slurm.sbatch:23-31` (per-member `DONE` gate) → `:38` → `scripts/plan_vote.sh:28-33` → `python plan.py --config-name plan_lewm.yaml objective._target_=planning.objectives.create_vote_objective_fn +objective.n_members=5 +objective.rule=cvar +objective.lam=1.0` → merged onto `conf/plan_lewm.yaml:46-50` → `plan.py:149 hydra.utils.call` → `planning/cem.py:114`. `MEMBERS` for block *s* is `LpWM-ltv_pd384_bf16_s{s..s+4}` (the recorded vote5 config). `LABEL` must keep the `<arm>_pd384_bf16_s<block>` shape (`analysis/figures.py:224-231 _ARM_STRIP`) and `SEED` must equal `<block>`; the sbatch job name must be `eval_${LABEL}` or `analysis/collect_evals.py:67-89` cannot read the eval-seed scheme from `slurm_logs/`.

**Control.** `PiWM-vote5-borda` — same M, same members, same episodes, mean-of-ranks. It is the variance-reduction-only rule and has never been run. Free correctness control: `PiWM-vote5-cvar` at `LAM=0` must reproduce borda bit-identically.

**Falsifier.** Paired effects vs `LpWM-ltv` over ≥8 shared blocks. If `borda ≈ median ≈ cvar(1) ≈ max` within overlapping 95% CIs, or if `max` is the worst of the four, or if `cvar(1) < borda`, the mechanism is variance reduction / M-scaling and pessimism is dead.

**Verification (before launching).**
```bash
python -c "
import torch; from planning.objectives import create_vote_objective_fn as f
p={'visual':torch.randn(64,1,5,8)}; g={'visual':torch.randn(1,1,5,8).expand(64,1,5,8)}
print('cvar(0)==borda', torch.equal(f(5,'cvar',lam=0.0)(p,g), f(5,'borda')(p,g)))
print('max>=borda', bool((f(5,'max')(p,g)>=f(5,'borda')(p,g)).all()))
print('cvar1', f(5,'cvar',lam=1.0)(p,g)[:4])"
pytest tests/test_vote_rules.py -q
```
After the first job starts, prove the rule is live (this class of check is what the twice-repeated silent-baseline bug requires):
```bash
python -c "import yaml,glob; d=sorted(glob.glob('plan_outputs/*PiWM-vote5-cvar_pd384_bf16_s3_gH5'))[-1]; print(yaml.safe_load(open(d+'/.hydra/config.yaml'))['objective'])"
# must print rule: cvar AND lam: 1.0 -- not the default 'mean'/0.0
```

## V2 - PiWM-gd: gradient planner as a first-class arm

**Claim.** "The model is bad" and "CEM's sampling is bad" are separable, and `planning/gd.py` already separates them: the rollout is exactly differentiable w.r.t. actions, so an arm that optimises the *same* objective by gradient descent isolates the search from the model at zero training cost.

**Derivation.** `ltv` and `linvar` have indistinguishable 1-step error (rel_mse 0.0092 vs 0.0095) but a **4.5x** CEM gap (0.357 vs 0.080). A gap that is invisible in rollout error is what a *search* interacting with the objective's curvature produces. The competing explanation is that the objective is a bad proxy: Spearman(latent distance CEM minimises, TRUE task distance) = **+0.398** (n=296). GD discriminates them, because it drives the same proxy to a lower value than CEM can. Measured this session on the probe model (`tests/lpwm_build.build(load_cfg(["predictor=ltv"]))`): backprop through `rollout` to a `(2,5,10)` action tensor gives grad norm **2.76e-3** with **zero-fraction 0.0** — no dead entries, because `reprelu` uses a GELU surrogate on the backward pass (`conf/link/reprelu.yaml:2-3`). `models/visual_world_model.py` has exactly three `no_grad` sites (243, 573, 698), none on the rollout path.

**Change.** No new planner code; `planning/gd.py:8-122` is used as written.
- New `conf/plan_gd.yaml` = `conf/plan_lewm.yaml` with `:52-64` replaced:
```yaml
planner:
  _target_: planning.mpc.MPCPlanner
  max_iter: 10
  n_taken_actions: 5        # overwritten to goal_H at plan.py:202
  sub_planner:
    target: planning.gd.GDPlanner   # mpc.py:44 copies `target` -> `_target_`
    horizon: 5              # overwritten to goal_H at plan.py:201
    action_noise: 0.003
    sample_type: randn
    lr: 1
    opt_steps: 300          # NOT conf/planner/mpc_gd.yaml's 1000
    eval_every: 100         # NOT 10
  name: mpc_gd
```
`conf/planner/mpc_gd.yaml:10-11` (`opt_steps: 1000, eval_every: 10`) would fire `gd.py:112-115` 100 times per `plan()` — 100 real-env 50-episode pygame rollouts, ×10 MPC iters — which times out the 3h55 allocation (`vote_slurm.sbatch:7`). 300/100 gives 3.
- Guard, inserted after `planning/gd.py:42`:
```python
        assert getattr(objective_fn, "__name__", "") != "objective_fn_vote", \
            "rank-based vote objectives are piecewise constant (argsort); GD gets zero gradient"
```
This enforces the incompatibility already documented at `planning/objectives.py:88-90`. `conf/plan_gd.yaml` keeps `:46-50` unchanged (`create_objective_fn`, `alpha: 0`, inert on adaln).
- New `scripts/plan_arm.sh`, modelled line-for-line on `scripts/plan_vote.sh:28-33`:
```bash
python plan.py --config-name "${PLAN_CONFIG:-plan_gd.yaml}" \
    ckpt_base_path="${CKPT_BASE}" model_name="${LABEL}" model_epoch="${EPOCH:-latest}" \
    "ensemble_members=[${CKPT}]" \
    n_evals="${NEVALS}" planner.max_iter="${MAXITER}" seed="${SEED}" ${GOAL_H:+goal_H=$GOAL_H}
```
The `ensemble_members=[<one run>]` is **not** an ensemble; it is the only way to decouple the output label from the checkpoint dir. `plan.py` builds the output dir from `model_name` (`conf/plan_lewm.yaml:12`) and `collect_evals.py:153-159` parses the arm from that dir with newest-wins, so evaluating `LpWM-ltv_pd384_bf16_s5` under a different planner without relabelling would **silently overwrite the CEM baseline number for that seed**. The M=1 path is verified inert (`PiWMvoteM1` = 0.66 = `LpWM-ltv` s5).
- Cost note for the reviewer: GD does `opt_steps × n_evals` = 15,000 ViT encodes of `obs_0`; CEM does `opt_steps × n_evals × num_samples` = 450,000 (`cem.py:88-92` re-encodes the same `obs_0` 300×). GD is ~30x cheaper per MPC iter.

**Wiring.** `CKPT,LABEL,SEED,PLAN_CONFIG` → new `scripts/arm_slurm.sbatch` (copy of `vote_slurm.sbatch` with the `DONE` gate at `:26-31` applied to the single `CKPT`, job name `eval_${LABEL}`) → `scripts/plan_arm.sh` → `plan.py:187-197 hydra.utils.instantiate` → `MPCPlanner` → `mpc.py:44-54` → `GDPlanner`. Nothing is trained: no `run_campaign.sh` ARMS entry, no `train.sh` line, no `conf/train_rdmreg.yaml` key.

**Control.** `LpWM-ltv` — flat CEM, same checkpoints, same episode blocks s3..s15, already recorded. Second control `PiWM-gd-noise0` (`sub_planner.action_noise=0`) isolates the gradient from the Gaussian jitter injected at `gd.py:104-106`.

**Falsifier.** Paired effect vs `LpWM-ltv` over ≥8 blocks with a 95% CI containing 0 ⇒ CEM sampling is not the bottleneck and the 4.5x ltv/linvar CEM gap is a model property. The informative sub-case: if GD reaches a **lower** final objective (`plan_*/loss`, `gd.py:109-110` vs `cem.py:121-122`) at **equal or worse** success, V2 dies as a planner and simultaneously confirms that the +0.398 proxy — not the search — is the round-6 target.

**Verification (before launching).**
```bash
grep -n "no_grad" models/visual_world_model.py            # 243, 573, 698 only
python - <<'EOF'
import sys, torch; sys.path.insert(0,"tests")
from lpwm_build import build, load_cfg
m,_ = build(load_cfg(["predictor=ltv"]))
obs={"visual":torch.randn(2,1,3,224,224),"proprio":torch.randn(2,1,4)}
a=torch.randn(2,5,10,requires_grad=True)
z,_=m.rollout(obs_0=obs,act=a,z_goal=None); z["visual"][:,-1].pow(2).mean().backward()
print("grad_norm",float(a.grad.norm()),"zero_frac",float((a.grad==0).float().mean()))  # >0, 0.0
EOF
python -c "import yaml,glob; d=sorted(glob.glob('plan_outputs/*PiWM-gd_pd384_bf16_s*_gH5'))[-1]; c=yaml.safe_load(open(d+'/.hydra/config.yaml')); print(c['planner']['sub_planner']); print(c['objective']['_target_'])"
# must show planning.gd.GDPlanner + create_objective_fn, and opt_steps 300 / eval_every 100
python analysis/collect_evals.py --plan-outputs plan_outputs --out /tmp/c.json && \
  python -c "import json; a=json.load(open('/tmp/c.json'))['arms']; print(sorted(a['PiWM-gd']), sorted(a['LpWM-ltv']))"
```

## V3 - PiWM-2lvl: two-level contact planner (subgoal x short controller)

**Claim.** Flat CEM burns its 300 samples on sequences that never touch the block. Factorising the proposal into "one held approach command" × "a short push", and scoring a subgoal by the best push it admits, concentrates the *same* 300 rollouts on the branch where the objective is not flat.

**Derivation.** Block median motion per transition is **exactly 0.000 px**; **48%** of transitions are fully static; the block moves >0.5 px in only **25.3%**; the top 1% of transitions carry **34.8%** of all block motion and the top 5% carry **77.9%**. Under that action distribution ~3/4 of a sampled 5-step sequence's terminal latent is determined by the agent alone, so `||z_H - z_g||²` is near-constant across most of a 300-sample Gaussian cloud, the top-30 elites are close to a uniform subsample, and the refit at `cem.py:118-119` contracts toward noise. The fix is a better proposal, not more samples.

**Change.** NEW FILE `planning/two_level.py`, `class TwoLevelPlanner(BasePlanner)` (contract: `planning/base_planner.py:6-43`). With `H=self.horizon`, `f=self.evaluator.frameskip` (5), `A=self.action_dim` (10), `d=A//f` (2), `S=n_subgoals`, `C=n_ctrl`; per opt step, per episode, mirroring `cem.py:86-119`:
```python
k     = max(1, min(self.k_reach, H - 1))          # MUST be computed here, see traps
p     = torch.randn(S, d, device=dev) * sig_p + mu_p;  p[0] = mu_p
macro = p.repeat(1, f).view(S, 1, A).expand(S, k, A)          # tile, NOT repeat_interleave
u     = torch.randn(S, C, H - k, A, device=dev) * sig_u + mu_u;  u[:, 0] = mu_u
act   = torch.cat([macro[:, None].expand(S, C, k, A), u], dim=2).reshape(S * C, H, A)
with torch.no_grad():
    z, _ = self.wm.rollout(obs_0=rep(trans_obs_0, S * C), act=act, z_goal=rep_g["visual"])
loss  = self.objective_fn(z, rep_g).view(S, C)                # objective UNCHANGED
v_sub = loss.min(dim=1).values                                # L1: subgoal worth = best push it admits
e_sub = v_sub.argsort()[: self.topk_sub]
mu_p, sig_p = p[e_sub].mean(0), p[e_sub].std(0)
ue    = u[e_sub].reshape(-1, H - k, A)[loss[e_sub].reshape(-1).argsort()[: self.topk_ctrl]]
mu_u, sig_u = ue.mean(0), ue.std(0)
```
Returns `torch.cat([macro_from(mu_p), mu_u], dim=1)` of shape `(B,H,A)` and `np.full(n_evals, np.inf)`. The subgoal latent is free: `z_s = z["visual"][:, k]`; log elite-`z_s` dispersion as the diagnostic that level 1 resolves anything.

Traps, all verified: (1) `p.repeat(1, f)` tiles `[px,py,px,py,...]`, which is what `evaluator.py:116` (`"b t (f d) -> b (t f) d"`, f outer) unpacks; `repeat_interleave` yields `[px]*5+[py]*5`, a different and wrong command. (2) PushT actions are **relative** displacement targets (`pusht_env.py:370 relative=True`, not overridden in `conf/env/pusht.yaml`) tracked by a PD controller (`pusht_env.py:486-497`), so a held `p` is a straight-line approach at constant commanded speed — literally "approach". (3) `k` must be resolved inside `plan()`, because `plan.py:201` overwrites `sub_planner.horizon` *after* construction. (4) `__init__` must take `**kwargs` (`mpc.py:45-54` passes `env`/`name`) and `.logging_prefix` must be settable (`mpc.py:86`). (5) `topk_sub >= 2`, else `.std(0)` is `nan` — assert it.

NEW FILE `conf/plan_two_level.yaml` = `conf/plan_lewm.yaml` with `:52-64` replaced:
```yaml
planner:
  _target_: planning.mpc.MPCPlanner
  max_iter: 10
  n_taken_actions: 5
  sub_planner:
    target: planning.two_level.TwoLevelPlanner
    horizon: 5
    k_reach: 3        # approach steps; push gets H-k = 2
    n_subgoals: 15    # S
    n_ctrl: 20        # C;  S*C = 300 = flat CEM num_samples
    topk_sub: 5
    topk_ctrl: 30     # = flat CEM topk
    var_scale: 1
    opt_steps: 30
    eval_every: 1
  name: mpc_two_level
```
The inert default is *not loading this file*: `conf/plan_lewm.yaml` is untouched, so every existing arm stays bit-identical. No parameters are created ⇒ NEW HEAD CHECKLIST does not apply.

**Wiring.** `CKPT,LABEL,SEED` + `PLAN_CONFIG=plan_two_level.yaml` → `scripts/arm_slurm.sbatch` (V2) → `scripts/plan_arm.sh` → `python plan.py --config-name plan_two_level.yaml ensemble_members=[<ckpt>] model_name=PiWM-2lvl_pd384_bf16_s<b> seed=<b>` → `plan.py:187` → `MPCPlanner` → `mpc.py:44` (`target`→`_target_`) → `TwoLevelPlanner`. Same label discipline as V2 (`model_name` decoupled via the verified-inert M=1 ensemble path, job name `eval_${LABEL}`).

**Control.** `LpWM-ltv`, already recorded on blocks s3..s15, and **exactly budget-matched by construction**: `S*C = 300 =` `conf/plan_lewm.yaml:60`, `opt_steps 30 =` `:62`, same `H`, same objective, same checkpoints, same episodes (`seed=block` ⇒ `plan.py:139`). The only difference is the proposal factorisation plus the min-over-push elite rule. Second control if it wins: `PiWM-2lvl-free` (approach segment free per step, hold removed, everything else identical) isolates the hold from the nested selection.

**Falsifier.** Paired effect vs `LpWM-ltv` over ≥8 blocks with a 95% CI containing 0 ⇒ the structure buys nothing at matched budget and the 75%-static derivation is not the mechanism behind the CEM failure. Also killed if `PiWM-2lvl-free` matches `PiWM-2lvl` *and* both match flat CEM.

**Verification (before launching).**
```bash
pytest tests/test_two_level.py -q     # NEW: builds the probe wm via tests/lpwm_build, evaluator=None,
# asserts plan() returns (B,5,10) float tensor + (B,) np.ndarray; asserts the first k steps repeat
# the same (px,py) across all f sub-step slots; asserts one wm.rollout call of batch 300 per opt step.
python -c "import yaml; c=yaml.safe_load(open('conf/plan_two_level.yaml')); s=c['planner']['sub_planner']; print(s); assert s['n_subgoals']*s['n_ctrl']==300 and s['opt_steps']==30"
python -c "import yaml,glob; d=sorted(glob.glob('plan_outputs/*PiWM-2lvl_pd384_bf16_s*_gH5'))[-1]; print(yaml.safe_load(open(d+'/.hydra/config.yaml'))['planner']['sub_planner'])"
grep -h "\[2lvl\]" slurm_logs/eval_PiWM-2lvl_*.out | head -1
# TwoLevelPlanner.plan must print: [2lvl] H=5 k=3 S=15 C=20 rollouts/opt_step=300 (flat CEM: 300)
```

---

## T4 - PiWM-vp: hindsight goal-conditioned value V(z,g) by in-sample expectile TD

**Claim.** A head trained only on latents and step counts learns a time-to-goal scalar that orders states by *reachability* rather than by latent displacement, and it is trainable on this dataset without any privileged state.

**Derivation.** The quantity CEM currently minimises is ranked against the true task distance at Spearman **+0.398** (n=296) — it is barely better than a coin flip about which of two candidate end-states is closer to the goal. Latent MSE cannot do better in principle here: block median motion is **exactly 0.000 px**, **48%** of transitions are fully static and the top 1% carry **34.8%** of all block motion, so ‖z−g‖² is dominated by directions along which nothing task-relevant happened. `-E[steps]` is invariant to any monotone reparametrisation of the latent and is defined by the data's own temporal order, so it cannot be gamed by the scale of the code.

**Change.** New file `models/heads.py`:

```python
class MLPHead(nn.Module):           # fixed param names net.0/net.2/net.4 so plan.py
    def __init__(self, d_in, d_hid, d_out):   # can rebuild it from shapes alone
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(),
                                 nn.Linear(d_hid, d_hid), nn.GELU(),
                                 nn.Linear(d_hid, d_out))
    def forward(self, z, g): return self.net(torch.cat([z, g, z - g], -1))
```

`models/visual_world_model.py`: add ctor kwargs after line 47 — `value_w=0.0, value_mode="td", value_tau=0.7, value_gamma=0.98, value_hidden=None, value_ema=0.005, value_p_future=0.5, policy_w=0.0, act_dim_raw=None`. Build the heads **after line 118** (`self.emb_dim = ...`), because `base_encoder.emb_dim` is D, and build them inside `with torch.random.fork_rng(devices=[]): torch.manual_seed(20260903)` — an unforked `nn.Linear` init shifts the global stream RDMReg draws its target from, which would make the arm differ from its control by more than one factor. `self.value_target = copy.deepcopy(self.value_head).requires_grad_(False)`.

New method beside `_path_int_loss` (insert after line 332):

```python
def _hindsight_pairs(self, z):          # z: (b,T,p,d) LINKED
    f = z.mean(dim=2).detach()          # (b,T,D); p==1 (CLS) -> exact, patch -> mean-pool
    for i in range(T - 1):              # anchors 0..T-2 (T=4: num_hist 3 + num_pred 1)
        k    = randint(1, T - i, (b,))                       # future offset
        fut  = f[arange(b), i + k]
        rnd  = f[randperm(b), randint(0, T, (b,))]           # cross-trajectory goal
        useF = rand(b) < self.value_p_future
        g    = where(useF[:,None], fut, rnd);  done = (useF & (k == 1)).float()
    return zt, zn, g, done, k, useF     # concatenated over i
```

```python
def _value_loss(self, z):
    if self.value_head is None: return None
    if next(self.value_head.parameters()).device != z.device:   # heads built inside
        self.value_head.to(z.device); self.value_target.to(z.device)   # VWorldModel are
    zt, zn, g, done, k, useF = self._hindsight_pairs(z)         # never prepare()d
    v = self.value_head(zt, g).squeeze(-1)
    if self.value_mode == "mc":         # control (a): regression, no bootstrap
        return (((v + k)**2) * useF).sum() / useF.sum().clamp_min(1)
    if self.value_mode == "geom":       # control (b): geometric latent distance
        s   = (zn - zt).norm(-1).mean().detach().clamp_min(1e-6)   # 1 step, in latent units
        tgt = -((g - zt).norm(-1) / s).clamp(max=1/(1-self.value_gamma))
        return (v - tgt.detach()).pow(2).mean()
    with torch.no_grad():               # r = -1, terminal when g IS z_{t+1}
        y = (-1. + self.value_gamma * (1. - done) *
             self.value_target(zn, g).squeeze(-1)).clamp(min=-1/(1-self.value_gamma))
    u = y - v
    if self.training:                   # Polyak, before this batch's step (1-step lag, fine)
        for p, q in zip(self.value_target.parameters(), self.value_head.parameters()):
            p.mul_(1 - self.value_ema).add_(self.value_ema * q)
    return ((self.value_tau - (u < 0).float()).abs() * u.pow(2)).mean()   # expectile, tau=0.7
```

`tau=0.7` is the in-sample max: positive residuals (a transition that reached the goal faster than V expected) are weighted 0.7 vs 0.3, so V approaches the best *dataset* action from z without ever querying an OOD action. `z` is **detached** into the head, so encoder/predictor gradients are bit-identical to the control at the same seed — the CEM control for V5/V4 is literally this run's own eval.

Loss assembly, `visual_world_model.py` **after line 687, before 688**:

```python
if self.value_w > 0:
    _vl = self._value_loss(z_emb)
    if _vl is not None:
        loss = loss + self.value_w * _vl; loss_components["value_loss"] = _vl
if self.policy_w > 0:                                  # V4, same insertion point
    _bl = self._policy_loss(z_emb, act)
    if _bl is not None:
        loss = loss + self.policy_w * _bl; loss_components["policy_loss"] = _bl
```

**NEW HEAD CHECKLIST.** (1) `train.py init_optimizers`, appended after line 921: alias `self.value_head = getattr(self._model_module(), "value_head", None)` (plus `value_target`, `policy_head`) — `save_ckpt` reads `self.__dict__`, so the alias is what makes the key savable — then `self.value_optimizer = self.accelerator.prepare(torch.optim.AdamW(self.value_head.parameters(), lr=self.cfg.training.get("value_lr", 3e-4)))`. Plain AdamW, **not** `mup_param_groups`: the head's fan-in is 3D, so muP-scaling it would make the head LR a function of the swept width and confound T4 with the wave6/wave7 factorial. (2) `_keys_to_save`, after line 503: `+= ["value_head","value_target","value_optimizer"]` when `cfg.value_w>0`; `+= ["policy_head","policy_optimizer"]` when `cfg.policy_w>0`. (3) lazy `.to(device)` — in `_value_loss` above, mirroring lines 330-331. (4) `zero_grad` at train.py:1076-1082 and `.step()` after 1092, both `if self.value_optimizer is not None`. (5) `tests/lpwm_build.py` build(), mirror every kwarg after line 115 with `act_dim_raw=2*int(cfg.get("frameskip",5))`, and append the two head optimizers to the `opts` list at 128-141. (6) `tests/test_arms.py` ARMS (18-31): `round5/value_td`, `round5/value_mc`, `round5/value_geom`, `round5/policy`, plus an assertion in `test_arm_overrides_actually_reach_the_modules` that `model.value_head is None` and `"value_loss" not in comps` at defaults.

Diagnostics: `_value_diagnostics` beside `_causal_diagnostics` (after train.py:1493), registered `self._safe("value", lambda: self._value_diagnostics(), payload)` after line 1253. Keys (unique tails, floats, finite): `value/v_mean`, `value/v_std`, `value/td_abs`, `value/target_gap`, and the liveness metric `value/rho_k` = Spearman(V(z_t, z_{t+k}), −k) over k∈{1,2,3} on `self._diag["z"]`. Mirror in `tests/test_live_diagnostics.py::_all_blocks` (line 281-297) and add `"value": ["predictor=ltv","value_w=1.0"]` to its ARMS (300-307).

Config keys, appended to `conf/train_rdmreg.yaml` after line 179, all inert: `value_w: 0.0`, `value_mode: td`, `value_tau: 0.7`, `value_gamma: 0.98`, `value_hidden: null`, `value_ema: 0.005`, `value_p_future: 0.5`, `policy_w: 0.0`, and `training.value_lr: 3e-4`.

**Wiring.** `scripts/run_campaign.sh`: `wave23_arms()` beside `wave22_arms` (line 348), plus `wave23) wave23_arms; gate=wave23 ;;` in the case block (451-466) and in the error string at 466.

```bash
wave23_arms() {
    ORDER[wave23]="PiWM-vp PiWM-vp-mc PiWM-vp-geom"
    ARMS[PiWM-vp]="ltv 1.0 5e-4 VALUE_W=1.0 POLICY_W=1.0"
    ARMS[PiWM-vp-mc]="ltv 1.0 5e-4 VALUE_W=1.0 POLICY_W=1.0 VALUE_MODE=mc"
    ARMS[PiWM-vp-geom]="ltv 1.0 5e-4 VALUE_W=1.0 POLICY_W=1.0 VALUE_MODE=geom"
}
```

`scripts/train.sh`, after line 62: `[ -n "${VALUE_W:-}" ] && { add "value_w=${VALUE_W}"; TAG="${TAG}_v${VALUE_W}"; }`, likewise `VALUE_MODE` → `value_mode=` (`TAG=_${VALUE_MODE}`) and `POLICY_W` → `policy_w=` (`TAG=_bc`). Chain: ARMS entry → train.sh `add` → `conf/train_rdmreg.yaml` inert default → `train.py` model kwargs, inserted after line 820 next to `path_int_dims`, with `act_dim_raw=int(self.datasets["train"].action_dim)` (=10 at frameskip 5) → `VWorldModel`.

**Control.** `PiWM-vp-mc` (MC hindsight regression, no bootstrap, no target net) isolates TD from "any learned scalar that correlates with progress"; `PiWM-vp-geom` (same head, same optimiser, regressed to ‖z−g‖ in units of one mean latent step) isolates temporal structure from a smooth MLP reparametrisation of the distance CEM already uses. Both are single-token `VALUE_MODE` flips on identical code paths.

**Falsifier.** On held-out val windows at end of training: Spearman(V(z_t, z_{t+k}), −k) < 0.6, **or** Spearman(−V(z_t, z_g), true task distance from `states.pth`) ≤ **+0.398** — the value is then no better ordered than the latent MSE it is meant to replace, and V5 has no mechanism. (`states.pth` here is measurement only, never a training leaf.)

**Verification.**
```bash
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
$PY -m pytest tests/test_bit_identity.py tests/test_arms.py tests/test_live_diagnostics.py -q
$PY - <<'EOF'                     # flag reaches the model, head gets grad, head is OPTIMISED
import sys, torch; sys.path.insert(0, 'tests')
from lpwm_build import build, load_cfg, seed_all, synthetic_batch
cfg = load_cfg(["predictor=ltv","value_w=1.0","policy_w=1.0"]); seed_all(0)
m, opts = build(cfg); obs, act = synthetic_batch(cfg, 4, torch.Generator().manual_seed(1))
_,_,_,loss,c = m(obs, act); loss.backward()
print("keys:", "value_loss" in c, "policy_loss" in c)
print("grad:", all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.value_head.parameters()))
ids = {id(p) for o in opts for g in o.param_groups for p in g["params"]}
print("in optimizer:", all(id(p) in ids for p in m.value_head.parameters()))   # the path_int defect
EOF
# after ~200 steps of a real run, the key must EXIST on disk:
$PY -c "import torch;k=torch.load('runs/outputs/PiWM-vp_pd384_bf16_s3/checkpoints/model_latest.pth',map_location='cpu');print([x for x in k if 'value' in x or 'policy' in x])"
```
The last command must print `['value_head','value_target','value_optimizer','policy_head','policy_optimizer']`. Anything shorter is the path_int defect repeating; do not launch the wave.

## V5 - PiWM-vsearch: value at the CEM leaves instead of terminal latent MSE

**Claim.** Replacing `‖z_H − z_g‖²` with `−V(z_H, z_g)` changes what CEM ranks from latent displacement to predicted steps-to-goal. If it works, the M=5 plan-time ensemble gain and the V gain are non-additive, because both were proxies for the same missing quantity.

**Derivation.** The leaf score, not the 1-step model, is what CEM is limited by: `ltv` (rel_mse 0.0092, CEM 0.357) and `linvar` (rel_mse 0.0095, CEM 0.080) are indistinguishable in 1-step error and differ 4.5x at planning. And the leaf metric itself is ranked against the true task distance at only **+0.398**. CEM does nothing but `argsort(loss)[:topk]` (cem.py:115), so the leaf metric *is* the planner.

**Change.** `planning/objectives.py`, appended after line 112:

```python
def create_value_objective_fn(wm=None, lam=0.0, **_):
    """CEM score = -V(z_H, g) + lam * MSE(z_H, g). Lower is better; V <= 0 so -V >= 0."""
    heads = list(getattr(wm, "value_heads", [])) or [getattr(wm, "value_head", None)]
    dims  = list(getattr(wm, "dims", [])) or [None]
    assert all(h is not None for h in heads), (
        "value objective needs a TRAINED value head on every member -- refusing to plan "
        "with a randomly initialised one (see the path_int defect)")
    def objective_fn_value(z_obs_pred, z_obs_tgt):
        pv, tv = z_obs_pred["visual"][:, -1], z_obs_tgt["visual"][:, 0]   # (B, M, D_max)
        out = 0.
        for m, h in enumerate(heads):
            d = dims[m] or pv.shape[-1]
            z, g = pv[:, m, :d], tv[:, m, :d]
            s = -h(z, g).squeeze(-1)
            if lam > 0: s = s + lam * (z - g).pow(2).mean(-1)
            out = out + s / len(heads)
        return out                                                        # (B,)
    return objective_fn_value
```

Note the `mean` vote is *sound* here where it is not for MSE: objectives.py:78-84 documents that equal-weight MSE voting requires members to share a representation scale, but V is in units of **steps** for every member regardless of link, target_p or D, so heterogeneous columns are directly commensurable.

The factory needs the model. `plan.py:149-151` becomes `objective_fn = hydra.utils.call(cfg_dict["objective"], wm=self.wm)`; add `wm=None, **_` to the signatures of `create_objective_fn` (objectives.py:6) and `create_vote_objective_fn` (objectives.py:70) so the extra kwarg — and the yaml's inert `alpha`/`base`/`mode` — are swallowed by every factory.

The head must be *live at plan time*, which is exactly where path_int failed. `plan.py` `ALL_MODEL_KEYS` (lines 28-34): add `"value_head"`, `"policy_head"`. In `load_model`, between the instantiate block ending at line 544 and `_load_pose_dyn(...)` at line 545:

```python
for _name in ("value_head", "policy_head"):
    _sd = payload.get(_name)
    if _sd is None: continue
    from models.heads import MLPHead
    d_hid, d_in = _sd["net.0.weight"].shape;  d_out = _sd["net.4.weight"].shape[0]
    _h = MLPHead(d_in, d_hid, d_out); _h.load_state_dict(_sd)   # shapes come from the ckpt,
    setattr(model, _name, _h.to(device))                        # not from train_cfg
    print(f"{_name} restored: {d_in}->{d_hid}->{d_out}")
```

`planning/ensemble.py`, after line 51: `self.value_heads = [getattr(m, "value_head", None) for m in self.members]` and `self.policy_head = getattr(self.members[0], "policy_head", None)`. `self.dims` already exists at line 49 and is what un-pads member m's slice.

Config keys: `objective._target_` (default `planning.objectives.create_objective_fn`, unchanged) and `objective.lam` (added on the CLI with `+`, inert default 0.0 = pure value leaf; `lam=1.0` is the blended arm).

**Wiring.** Eval-only — no training arm, no `run_campaign.sh` entry. New `scripts/plan_value.sh`, a copy of `scripts/plan_vote.sh` with the objective overrides replaced:

```bash
python plan.py --config-name plan_lewm.yaml \
    ckpt_base_path="${CKPT_BASE}" model_name="${LABEL}" model_epoch="${EPOCH:-latest}" \
    ${MEMBERS:+ensemble_members=[${MEMBERS}]} \
    objective._target_=planning.objectives.create_value_objective_fn \
    +objective.lam="${LAM:-0.0}" \
    n_evals="${NEVALS}" planner.max_iter="${MAXITER}" ${SEED:+seed=$SEED}
```

`LABEL` keeps the `<arm>_pd<D>_<precision>_s<block>` shape so `analysis/collect_evals.py` files it and `paired_effect` pairs it on the same episode block; `SEED` must equal `<block>` (plan.py:141). The 2x2 non-additivity factorial, all eval-only on checkpoints that already exist:

| leaf \ columns | M=1 | M=5 |
|---|---|---|
| MSE | `LpWM-ltv` (recorded) | `PiWM-vote5-borda` (recorded) |
| V | `PiWM-vsearch_pd384_bf16_s3` | `PiWM-vsearch-vote5_pd384_bf16_s3` (`MEMBERS=` 5 PiWM-vp runs) |

**Control.** `PiWM-vsearch-geom`: identical objective code, but the checkpoint is `PiWM-vp-geom`, whose head was regressed to a geometric latent distance. It isolates "V is a learned time-to-goal" from "the leaf is a smooth MLP of (z, g, z−g)". Second control: the *same* checkpoint evaluated with the stock terminal-MSE objective — since T4's heads train on detached latents, that run's encoder and predictor are bit-identical, so the contrast is the leaf metric and nothing else.

**Falsifier.** V-at-leaf ≤ terminal-MSE on the same checkpoint, paired over the 50 episodes of one block. Separately, Sutton's prediction dies if the two gains simply add: if `V(M=5) − V(M=1) ≈ MSE(M=5) − MSE(M=1)`, the ensemble was measuring something V does not contain, and the claim "the ensemble was never the finding" is refuted.

**Verification.**
```bash
$PY -m pytest tests/test_ckpt_compat.py -q     # after adding a value/policy roundtrip arm:
                                               # weight equality after load_model, per the file's own docstring
NEVALS=2 MAXITER=1 LABEL=smoke_pd384_bf16_s3 SEED=3 scripts/plan_value.sh 2>&1 | tee /tmp/v5.log
grep -E "value_head restored|Success rate" /tmp/v5.log     # must print the restore line FIRST
grep -A4 '^objective:' plan_outputs/*smoke*/. hydra/config.yaml   # _target_ must be create_value_objective_fn
$PY - <<'EOF'   # is the head trained or random? V(g,g) must be ~0 -- zero steps to yourself
import torch, plan as P; from pathlib import Path; from omegaconf import OmegaConf
run='runs/outputs/PiWM-vp_pd384_bf16_s3'
cfg=OmegaConf.load(f'{run}/hydra.yaml'); m=P.load_model(Path(run)/'checkpoints/model_latest.pth',cfg,cfg.num_action_repeat,'cuda')
g=torch.randn(8,384,device='cuda').relu(); print(float(m.value_head(g,g).mean()))   # trained: > -0.5
EOF
```

## V4 - PiWM-policy: reactive goal-conditioned policy, no search, same evaluator

**Claim.** A one-shot MLP `a_t = pi(z_t, z_g)` trained by behaviour cloning on the same latent, evaluated through the same `PlanEvaluator`, matches CEM. If it does, the 300x30x10 = 90,000 model rollouts CEM spends per episode buy nothing, and every ranking in this campaign was ranking representations, not planners.

**Derivation.** CEM's advantage over a reactive policy can only come from the leaf metric being informative about the goal, and it is ranked against the true task distance at **+0.398**. A search whose objective is that weakly ordered is, at the margin, a draw from the action prior with extra steps — which a BC policy fits directly and far more cheaply.

> This spec previously also argued that the *candidates* are indistinguishable, citing `d_action` ≈ 0.0002 for the baseline. **That is withdrawn** (diary §12b): the baseline's `d_action/‖z‖` is 0.549, and among healthy predictors `d_action` orders CEM success at partial ρ = +0.70. Action discrimination is therefore *not* the missing ingredient, and V4's case rests entirely on the leaf metric. If V4 matches CEM, that is evidence about the objective, not about action-sensitivity.

**Change.** Training head: `policy_head = MLPHead(3D, H, act_dim_raw)` with `act_dim_raw = 10` (2 x frameskip), built in the same forked-RNG block as T4's value head. New method beside `_value_loss`:

```python
def _policy_loss(self, z, act):
    """V4: goal-conditioned BC in latent space. act is the NORMALISED action the loader
    already provides (conf/train_rdmreg.yaml normalize_action: True), so this needs
    nothing the CEM objective does not already have."""
    if self.policy_head is None: return None
    if next(self.policy_head.parameters()).device != z.device: self.policy_head.to(z.device)
    f = z.mean(dim=2).detach()                      # (b,T,D) LINKED, stop-grad
    out, n = 0., 0
    for i in range(f.shape[1] - 1):                 # anchors 0..T-2
        k = torch.randint(1, f.shape[1] - i, (f.shape[0],), device=z.device)
        g = f[torch.arange(f.shape[0], device=z.device), i + k]      # hindsight goal
        out = out + (self.policy_head(f[:, i], g) - act[:, i].detach()).pow(2).mean(); n += 1
    return out / n
```

New planner `planning/policy.py`, satisfying the contract in `planning/base_planner.py` (ctor swallows `name`/`env`/CEM-only keys in `**kwargs`; settable `.horizon` and `.logging_prefix`; `plan(obs_0, obs_g, actions=None) -> (Tensor (B,T,action_dim), np.ndarray (B,))`):

```python
class PolicyPlanner(BasePlanner):
    def __init__(self, horizon, wm, action_dim, objective_fn, preprocessor, evaluator,
                 wandb_run, logging_prefix="policy_0", log_filename="logs.json",
                 shuffle_goal=False, **kwargs):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run,
                         log_filename)
        self.horizon, self.logging_prefix = horizon, logging_prefix
        self.shuffle_goal = shuffle_goal
        self.policy = getattr(wm, "policy_head", None)
        assert self.policy is not None, "PolicyPlanner needs a TRAINED policy_head"

    @torch.no_grad()
    def plan(self, obs_0, obs_g, actions=None):      # `actions` ignored: no search to warm-start
        t0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        tg = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        zg_full = self.wm.encode_obs_linked(tg)["visual"]          # (B,1,p,D)
        if self.shuffle_goal: zg_full = zg_full[torch.randperm(zg_full.shape[0])]
        z_g = zg_full[:, 0].mean(dim=1)                            # (B,D), p==1 -> exact
        z_t = self.wm.encode_obs_linked(t0)["visual"][:, -1].mean(dim=1)
        acts = []
        for _ in range(self.horizon):
            acts.append(self.policy(z_t, z_g))                     # (B, action_dim)
            z_obses, _ = self.wm.rollout(obs_0=t0, act=torch.stack(acts, 1), z_goal=zg_full)
            z_t = z_obses["visual"][:, -1].mean(dim=1)             # reuse the VERIFIED
        mu = torch.stack(acts, dim=1)                              # rollout path; do not
        return mu, np.full(mu.shape[0], np.inf)                    # reimplement the window
```

Advancing the latent by re-calling `wm.rollout` with the growing action prefix costs H(H+1)/2 = 15 predictor calls per episode against CEM's 90,000, and reuses `_rollout_adaln` (visual_world_model.py:871-881) exactly, so no window/pose-rolling logic is duplicated.

New `conf/planner/mpc_policy.yaml` (so `build_plan_cfg_dicts` at plan.py:99-101 and `plan_settings.planner: ['mpc_policy']` both resolve it):

```yaml
_target_: planning.mpc.MPCPlanner
max_iter: 10
n_taken_actions: 5
sub_planner: {target: planning.policy.PolicyPlanner, horizon: 5, shuffle_goal: false}
name: mpc_policy
```

and `conf/plan_policy.yaml`, a copy of `conf/plan_lewm.yaml` with the `planner:` node (lines 55-66) replaced by that block. The MPC wrapper is mandatory, not optional: it gives the policy exactly the same 10 rounds of real env feedback, the same latched-success accounting (mpc.py:110-112) and the same `n_taken_actions=5` the CEM baseline gets, so the only difference is *how the 5 actions were chosen*.

**Wiring.** Training rides the T4 arm (`PiWM-vp` carries `POLICY_W=1.0`), so the same checkpoint yields the CEM number, the V5 number and the V4 number on bit-identical encoder/predictor weights. Chain: `ARMS[PiWM-vp]="... POLICY_W=1.0"` → `scripts/train.sh` `[ -n "${POLICY_W:-}" ] && { add "policy_w=${POLICY_W}"; TAG="${TAG}_bc"; }` → `conf/train_rdmreg.yaml policy_w: 0.0` → train.py model kwargs (after line 820) with `act_dim_raw=int(self.datasets["train"].action_dim)` → `VWorldModel`. Eval: new `scripts/plan_policy.sh`, a copy of `scripts/plan.sh` with `--config-name plan_policy.yaml` and `LABEL`-style naming (`PiWM-policy_pd384_bf16_s3`) so `collect_evals.py` pairs it against `LpWM-ltv` on the same episode block.

**Control.** `PiWM-policy-shufgoal` — `planner.sub_planner.shuffle_goal=true`, which permutes the goal latents across episodes and changes nothing else. PushT demos have a strong action prior, so a policy that ignores z_g entirely can still score; if shuffled-goal success equals real-goal success, the policy is replaying the prior and the arm says nothing about planning. Run the same shuffle against CEM for the symmetric statement.

**Falsifier.** Paired over 50 episodes x 3 seed blocks: policy success < 0.5 x CEM success on the same checkpoint kills the claim — search is buying something. The claim survives only if |policy − CEM| < 0.05 paired *and* `shufgoal` is clearly worse than the real goal (otherwise neither method is goal-conditioned and both numbers are prior-matching).

**Verification.**
```bash
$PY -m pytest tests/test_policy_planner.py -q     # NEW: contract test, no env, no GPU --
# asserts issubclass(PolicyPlanner, BasePlanner); ctor accepts name=/env=/topk= via **kwargs;
# plan() returns (Tensor (B,5,10), ndarray (B,)) on a stub wm; .horizon is settable.
NEVALS=2 MAXITER=1 scripts/plan_policy.sh plan_policy.yaml PiWM-vp_pd384_bf16_s3 latest 2>&1 | tee /tmp/v4.log
grep -E "policy_head restored|MPC iter|Success rate" /tmp/v4.log   # restore line must appear FIRST
$PY - <<'EOF'   # is the policy goal-conditioned at all? d(action)/d(goal) must be nonzero
import torch, plan as P; from pathlib import Path; from omegaconf import OmegaConf
run='runs/outputs/PiWM-vp_pd384_bf16_s3'; cfg=OmegaConf.load(f'{run}/hydra.yaml')
m=P.load_model(Path(run)/'checkpoints/model_latest.pth',cfg,cfg.num_action_repeat,'cuda')
z=torch.randn(64,384,device='cuda').relu(); g=torch.randn(64,384,device='cuda').relu()
a1,a2=m.policy_head(z,g), m.policy_head(z,g[torch.randperm(64)])
print(float((a1-a2).pow(2).mean().sqrt()), float(a1.pow(2).mean().sqrt()))   # ratio must be O(1)
EOF
```
A ratio near zero means the head learned the marginal action prior only; fix that before spending a wave on the comparison.

---

## T1 - PiWM-decode: attach the decoder and let reconstruction gradient reach the encoder

**Claim.** Nothing in this system has ever forced the latent to retain visual content: `conf/train_rdmreg.yaml:126` sets `has_decoder: False`, so `train.py:463` leaves `self.decoder = None`, and even on the branch that would run it, `visual_world_model.py:728` writes `z_dec = z_emb.detach()`. Attaching a decoder and dropping that detach makes pixel reconstruction an actual constraint on `z` for the first time.

**Derivation.** One-step latent error is measurably not the bottleneck: `ltv` rel_mse 0.0092 / CEM 0.357 vs `linvar` rel_mse 0.0095 / CEM 0.080 — indistinguishable prediction error, 4.5x CEM. And Spearman(latent distance CEM minimises, TRUE task distance) = **+0.398** (n=296). A latent can be perfectly self-predictive and still not metrise the task. Reconstruction is the only objective in this codebase whose optimum requires `z` to carry the block. Verified live on CPU: with the current `.detach()`, **0 of 144** encoder parameters receive gradient from `decoder_recon_loss`; without it, **144/144** do.

**Change.**
1. `models/visual_world_model.py:51` — add ctor kwargs `decode_grad=False,` and `lamb_decode` already exists at line 34. Assign after line 77: `self.decode_grad = bool(decode_grad)`.
2. `models/visual_world_model.py:728` — replace
   `z_dec = z_emb.detach()` with
   `z_dec = z_emb if self.decode_grad else z_emb.detach()`.
   Lines 726-733 are otherwise unchanged; `loss = loss + self.lamb_decode * decoder_loss` at 732 already exists.
3. `train.py:823` — after `burst_tau=...,` add
   `decode_grad=bool(self.cfg.get("decode_grad", False)), lamb_decode=float(self.cfg.get("lamb_decode", 1.0)),`.
4. `conf/train_rdmreg.yaml` — after line 184 (`block_causal: false`) add:
   `decode_grad: false` (inert), `lamb_decode: 0.1` (inert while `has_decoder: False`).
5. **`decoder: vqvae` (line 7) MUST be overridden to `transposed_conv`.** Measured: `VQVAE` at `num_patches=1` rearranges to a 1x1 grid and upsamples 16x, returning `(b*t, 3, 16, 16)` — the MSE against `(b,t,3,224,224)` cannot run. `TransposedConvDecoder` (`models/decoder/transposed_conv.py:103`, terminal `nn.Upsample(size=(224,224))`) returns `(b*t,3,224,224)`, verified, 31.3M params.
6. `tests/lpwm_build.py:90` — replace `decoder=None` with
   `decoder=(hydra.utils.instantiate(cfg.decoder, emb_dim=encoder.emb_dim) if cfg.has_decoder else None)`, mirror `decode_grad`/`lamb_decode` at line 119, and add `train_decoder` from `cfg.model`.
7. `tests/test_arms.py:31` — add `"t1/decode": ["predictor=ltv","has_decoder=true","model.train_decoder=true","decoder=transposed_conv","decode_grad=true"]` and `"t1/decode_detach"` with `decode_grad=false`.

No NEW-HEAD-CHECKLIST work is needed beyond the build mirror: the decoder optimizer already exists (`train.py:911-921`) and `_keys_to_save` already covers it (`train.py:484-488`) — both gated on `has_decoder and model.train_decoder`. Caveat to log: `models/mup.py:33 _LINEARISH` omits `nn.ConvTranspose2d`, so those weights keep their kaiming init and train at `base_lr` under the vector rule. That is identical in both arms, so it does not confound the contrast.

**Wiring.** `run_campaign.sh` new `wave23_arms()` (register in the `case` at `:465`):
`ARMS[PiWM-decode]="ltv 1.0 5e-4 HAS_DECODER=true DECODE_GRAD=true LAMB_DECODE=0.1"`,
`ARMS[PiWM-decode-detach]="ltv 1.0 5e-4 HAS_DECODER=true DECODE_GRAD=false LAMB_DECODE=0.1"`,
`ARMS[PiWM-decode-w1]="ltv 1.0 5e-4 HAS_DECODER=true DECODE_GRAD=true LAMB_DECODE=1.0"`.
`scripts/train.sh` — insert after line 68:
```
[ -n "${HAS_DECODER:-}" ]  && { add "has_decoder=${HAS_DECODER} model.train_decoder=${HAS_DECODER} decoder=${DECODER:-transposed_conv}"; TAG="${TAG}_dec"; }
[ -n "${DECODE_GRAD:-}" ]  && { add "decode_grad=${DECODE_GRAD}"; TAG="${TAG}_dg${DECODE_GRAD}"; }
[ -n "${LAMB_DECODE:-}" ]  && { add "lamb_decode=${LAMB_DECODE}"; TAG="${TAG}_ld${LAMB_DECODE}"; }
```
Chain A: `ARMS` extra -> `env ... ${extra}` (`run_campaign.sh:437-440`) -> `train.sh:69a` -> `conf/train_rdmreg.yaml:185+` -> `train.py:823a` -> `VWorldModel.__init__`.

`lamb_decode=0.1` is primary because the scales are not matched: healthy per-dim `var(z)` is 0.20-0.24 with rel_mse 0.0092, so `z_loss` is O(2e-3), while recon MSE in the `[-1,1]` pixel space of `datasets/img_transforms.py` is O(1e-1). `LAMB_DECODE=1.0` is the nuisance guard cell, exactly as `PiWM-sigreg-w0p5` guarded `reg_weight`.

**Control.** `PiWM-decode-detach` — **not** the existing `LpWM-ltv`. Building the decoder at `train.py:747` consumes RNG *before* `mup_init_(self.encoder)` at `:847`, so a `has_decoder=True` run is not seed-matched to a `has_decoder=False` run even with the detach kept. The control must also build the decoder and also add `decoder_recon_loss`; only `decode_grad` differs.

**Falsifier.** `PiWM-decode` reaches a lower `train_decoder_recon_loss` than `PiWM-decode-detach` (pixels improve) but the M1 block-pose probe — ridge from frozen `z` to `states.pth[..., 2:5]`, held-out-episode R² — does not exceed the detach arm's by more than its seed CI, and paired CEM Δ is null. That kills "pixel supervision buys task content".

**Verification (before launching).**
```
DATASET_DIR=/nonexistent $PY -c "
import sys,torch; sys.path[:0]=['.','tests']
from lpwm_build import load_cfg,build,synthetic_batch,seed_all
c=load_cfg(['predictor=ltv','has_decoder=true','model.train_decoder=true',
            'decoder=transposed_conv','decode_grad=true'])
seed_all(0); m,_=build(c); g=torch.Generator().manual_seed(1234)
o,a=synthetic_batch(c,2,g); _,_,vr,l,comp=m(o,a)
e=[p for p in m.encoder.parameters() if p.requires_grad]
gd=torch.autograd.grad(comp['decoder_recon_loss'],e,retain_graph=True,allow_unused=True)
print(vr.shape, float(comp['decoder_recon_loss']),
      'encoder params with decoder grad:', sum(x is not None for x in gd),'/',len(e))"
```
Must print `torch.Size([2, 4, 3, 224, 224])` and `144 / 144`; rerun with `decode_grad=false` and it must print `0 / 144`. Then `DRYRUN=1 scripts/run_campaign.sh wave23` (the printed env must contain `HAS_DECODER=true DECODE_GRAD=true`) and `pytest tests/test_bit_identity.py tests/test_arms.py -q` (defaults keep `has_decoder: False`, so the component-name set is unchanged).

---

## T2 - PiWM-patchdecode: patch tokens x a per-patch decoder, the missing 2x2 cell

**Claim.** The patch-token arm was a null because it added spatial capacity with no objective demanding spatial content. A per-patch reconstruction head is the objective that makes each of the 256 tokens responsible for its own 14x14 pixels, closing the (tokens) x (decoder) 2x2.

**Derivation.** `PiWM-columns` (256 tokens, `conf/encoder/vit_scratch_patch.yaml`) vs `LpWM-ltv`: paired Δ **+0.072, 95% CI [−0.064, +0.207], p = 0.269, n = 12** — a null. Capacity was added; nothing asked for it. The predictor cannot supply the demand: `LinearDynamicsPredictor` has **no positional embedding and no per-patch parameters**, so its operator is broadcast identically over all 256 tokens and the loss it induces is permutation-invariant in `p`. A per-patch head is therefore the only genuinely new object here — it is the first parameter in this model that distinguishes token `i` from token `j`. T1 supplies the other factor, so the four cells are (cls, no-dec) = `LpWM-ltv`, (patch, no-dec) = `PiWM-columns`, (cls, dec) = `PiWM-decode`, (patch, dec) = this arm.

**Change.** New file `models/decoder/patch_head.py`:
```python
class PatchHead(nn.Module):                       # emb_dim=384, patch_size=14, grid=16
    def __init__(self, emb_dim=384, patch_size=14, grid=16, out_chans=3):
        super().__init__(); self.p, self.g, self.c = patch_size, grid, out_chans
        self.head = nn.Linear(emb_dim, out_chans * patch_size * patch_size)
    def forward(self, z):                          # z: (b, t, p, d) with p == g*g
        assert z.shape[2] == self.g * self.g, (z.shape[2], self.g)
        x = rearrange(self.head(z), "b t (gh gw) (c ph pw) -> (b t) c (gh ph) (gw pw)",
                      gh=self.g, gw=self.g, c=self.c, ph=self.p, pw=self.p)
        return x, torch.zeros((), device=z.device, dtype=z.dtype)
```
Verified: `(2,4,256,384) -> (8,3,224,224)`, **226,380 params**. The `(img, diff)` tuple matches the `decode_obs` contract at `visual_world_model.py:468`, and `(b t) c h w` matches the `rearrange` at `:469`. New `conf/decoder/patch_head.yaml`:
```yaml
_target_: models.decoder.patch_head.PatchHead
patch_size: 14
grid: 16          # img_size / patch_size
out_chans: 3
```
(`emb_dim` is injected by `train.py:749`.) `TransposedConvDecoder` is **not** usable here: `horizontal_forward` would render `b*t*p` = 65,536 separate 224x224 images per batch, and its `dist.mean.squeeze(2)` at `transposed_conv.py:112` is a no-op at `p=256`, so the following `rearrange("b t c h w -> ...")` fails on a 6-D tensor. `VQVAE` at `p=256` returns `(b*t,3,256,256)` — verified, wrong size. Everything else is T1's change, reused unmodified: `decode_grad`, `lamb_decode`, the ctor kwarg, `train.py:823a`, `lpwm_build.py:90`.

`PatchHead` is a single `nn.Linear`, so `models/mup.py` handles it exactly (`_LINEARISH` at `:33`): `fan_in=384 == mup_base_width` -> `used_lr = mup_lr`, `mup_init_` gives `N(0, 1/sqrt(384))`, `mup_param_groups` at `train.py:914` builds its optimizer. NEW HEAD CHECKLIST is satisfied by the existing decoder plumbing (optimizer `:911-921`, `_keys_to_save` `:484-488`, `accelerator.prepare` `:754-756`); `lpwm_build.py:90` and `tests/test_arms.py` still need the mirror. Note the head is **138x smaller than T1's decoder** (0.23M vs 31.3M), so a T2 win cannot be attributed to decoder capacity.

**Wiring.** `run_campaign.sh` `wave24_arms()` (register in the `case` at `:465`), **launched with `FEATURE=patch`** so `train.sh:29-30` selects `vit_scratch_patch` and `FEAT_TAG="_patch"` (`run_campaign.sh:46`) keeps the run name distinct:
```
ORDER[wave24]="PiWM-patchdecode PiWM-patchdecode-detach"
ARMS[PiWM-patchdecode]="ltv 1.0 5e-4 HAS_DECODER=true DECODER=patch_head DECODE_GRAD=true LAMB_DECODE=0.1"
ARMS[PiWM-patchdecode-detach]="ltv 1.0 5e-4 HAS_DECODER=true DECODER=patch_head DECODE_GRAD=false LAMB_DECODE=0.1"
```
Launch: `FEATURE=patch SEEDS="0 1 2 3 4 5 6 7" scripts/run_campaign.sh wave24`. Chain: `ARMS` extra -> `run_campaign.sh:437-440` -> `train.sh:69a` (`DECODER` is read by the `HAS_DECODER` line, default `transposed_conv`) -> `decoder=patch_head` group override -> `train.py:747-750`.

**Control.** Two, and both already exist or are in-wave. `PiWM-patchdecode-detach` isolates the *gradient* at fixed tokens and fixed RNG stream (same construction, so seed-matched). `PiWM-columns` (n=12, already evaluated) isolates *tokens without any decoder*. Crossing with `PiWM-decode` (T1) gives the interaction term, which is the actual claim: decoder value should be **larger** at 256 tokens than at 1.

**Falsifier.** Per-patch recon MSE falls well below `PiWM-decode`'s (256 tokens x 384 dims is 256x the code budget for the same image, so it must), yet the paired CEM Δ against `PiWM-columns` is null AND the T1-vs-T2 interaction is within its CI. That says spatial content in the code is not what CEM lacks, and the whole (tokens x decoder) axis is closed.

**Verification (before launching).**
```
DATASET_DIR=/nonexistent $PY -c "
import sys,torch; sys.path[:0]=['.','tests']
from lpwm_build import load_cfg,build,synthetic_batch,seed_all
c=load_cfg(['predictor=ltv','encoder=vit_scratch_patch','has_decoder=true',
            'model.train_decoder=true','decoder=patch_head','decode_grad=true'])
seed_all(0); m,_=build(c); g=torch.Generator().manual_seed(1234)
o,a=synthetic_batch(c,2,g); _,_,vr,l,comp=m(o,a); l.backward()
print('num_patches',m.encoder.num_patches,'recon',vr.shape,
      'head grad', float(m.decoder.head.weight.grad.abs().sum()))"
```
Must print `num_patches 256`, `torch.Size([2, 4, 3, 224, 224])`, non-zero head grad. Locality check (this is what makes the head "per-patch"): perturb `z[:, :, 0]` only and assert the output changes **only** inside `[0:14, 0:14]`. Then `FEATURE=patch DRYRUN=1 scripts/run_campaign.sh wave24` must show `"${FEATURE}"` = `patch` in the `submit_until_done.sh` argv and `_patch` in the run name, and `pytest tests/test_arms.py tests/test_bit_identity.py -q`.

---

## T3 - PiWM-contact: weight transitions by the visual change proprio does not explain

**Claim.** Weight each transition's prediction loss by the pixel change **outside the agent's own disc**. Proprio is a model input, so the agent's occupancy is exactly known and can be masked out; what remains is, to first order, the block. This buys contact-targeting with zero privileged state.

**Derivation.** A mean objective spends capacity by frequency, and the frequency is pathological: block median motion is **exactly 0.000 px**, **48%** of transitions are fully static, and the top 5% carry **77.9%** of all block motion. `||dz||` cannot be the weight: on a latent that ignores the block it upweights AGENT motion, which is circular. Measured this session on 60 episodes / 1,494 transitions at frameskip 5 (agent disc = radius 15 sim units x 1.6 pad, scaled by 224/512):

| weight | static mean | block>2px mean | separation | ESS/N | weight mass on block>0.5px |
|---|---|---|---|---|---|
| raw `mean((Δpix)^2)` | 1.29e-3 | 3.54e-3 | **2.7x** | 0.689 | 0.700 |
| **agent-masked residual** | 1.51e-6 | 2.41e-3 | **1596x** | **0.359** | **0.975** |
| oracle `‖Δblock‖` (illegal) | — | — | — | 0.284 | 0.999 |

Spearman(masked residual, true block motion) = **+0.889** vs +0.799 unmasked; correlation with agent motion is −0.30 (masking actively *removes* agent signal). Uniform weighting puts 47.5% of its mass on block-moving transitions; this puts 97.5% — within 2.4 points of the oracle. The static fraction reproduces the known 48% (measured 48.9%), which cross-validates the estimator. **The ~20x compute shrink in the proposal is wrong: measured ESS/N = 0.359, i.e. 2.8x**, because the mp4 noise floor (q10 = 3.3e-9) is a natural regulariser. The oracle's own ESS is 3.5x, so 2.8x is near the floor for *any* contact-targeted weighting.

**Change.** New method in `models/visual_world_model.py`, inserted after `_path_int_loss` ends at line 332:
```python
def _contact_weight(self, obs):
    """Per-transition weight = pixel change outside the agent's own disc, ^gamma,
    renormalised to unit mean (so this is not secretly an LR change)."""
    if self.contact_gamma == 0.0 or obs.get("proprio") is None:
        return None
    v, pr = obs["visual"], obs["proprio"]
    with torch.no_grad():
        H = v.shape[-1]; mu, sd, s, R = self._contact_geom  # tensors/floats on device
        px = (pr[..., :2] * sd + mu) * s                    # (b, T, 2) pixels
        yy, xx = self._contact_grid                         # cached (H,H) each
        d0 = (xx - px[:, :-1, 0, None, None])**2 + (yy - px[:, :-1, 1, None, None])**2
        d1 = (xx - px[:, 1:,  0, None, None])**2 + (yy - px[:, 1:,  1, None, None])**2
        keep = ((d0 > R * R) & (d1 > R * R)).to(v.dtype)     # (b, T-1, H, H)
        sq = (v[:, 1:] - v[:, :-1]).pow(2).mean(2)           # (b, T-1, H, H)
        r = (sq * keep).sum((-1, -2)) / keep.sum((-1, -2)).clamp_min(1.0)
        w = (r[:, :self.num_hist] + self.contact_eps).pow(self.contact_gamma)
        if self.contact_shuffle:                             # ESS-matched control
            w = w.reshape(-1)[torch.randperm(w.numel(), device=w.device)].view_as(w)
        w = w / w.mean().clamp_min(1e-12)
    return w                                                 # (b, num_hist)
```
Use it in the `else` branch, replacing line 651 (`z_loss = self.emb_criterion(...)`) with a third case parallel to `incr_norm` at 638-649:
```python
elif (_w := self._contact_weight(obs)) is not None:
    z_loss = ((z_pred - target).pow(2) * _w[..., None, None]).mean()
    loss_components["contact_ess"] = _w.mean().pow(2) / _w.pow(2).mean()
    loss_components["contact_w_max"] = _w.max()
else:
    z_loss = self.emb_criterion(z_pred, target)
```
Verified shapes on a synthetic `(4,4,3,224,224)` batch: `keep (4,3,224,224)`, `w (4,3)`, broadcast against `(b,num_hist,p,d)` correct, `ESS` finite. No new parameters, so the NEW HEAD CHECKLIST does not apply — nothing to optimize, checkpoint, or move to device beyond the cached grid (`register_buffer("_contact_grid", ..., persistent=False)`, same non-persistence rationale as `pose_dyn` at `:108`). Both new `loss_components` are tensors, as `train.py:1096-1099` requires, and are emitted **only** when `contact_gamma > 0`, so `tests/test_bit_identity.py`'s component-name set is unchanged at defaults.

Ctor kwargs after line 50: `contact_gamma=0.0, contact_shuffle=False, contact_eps=1e-8, contact_geom=None` (geometry = `(mean_x, mean_y, std_x, std_y, px_per_unit, agent_radius_px)`, following the `path_int_dims` precedent at `:47`). `conf/train_rdmreg.yaml` after line 184: `contact_gamma: 0.0` (inert), `contact_shuffle: false`, `contact_eps: 1e-8`, `contact_pad: 1.6`. `train.py:823a`:
```python
contact_gamma=float(self.cfg.get("contact_gamma", 0.0)),
contact_shuffle=bool(self.cfg.get("contact_shuffle", False)),
contact_eps=float(self.cfg.get("contact_eps", 1e-8)),
contact_geom=(*self.train_traj_dset.proprio_mean[:2].tolist(),
              *self.train_traj_dset.proprio_std[:2].tolist(),
              self.cfg.img_size / 512.0,
              15.0 * float(self.cfg.get("contact_pad", 1.6)) * self.cfg.img_size / 512.0),
```
`self.train_traj_dset` is the `PushTDataset` (`train.py:400`), which owns `proprio_mean`/`proprio_std` (`datasets/pusht_dset.py:83-84`); window size 512 is `pusht_env.py:381`, agent radius 15 is `pusht_env.py:709`. Mirror all four kwargs in `tests/lpwm_build.py` at line 119 (with a literal `contact_geom` from `PROPRIO_MEAN/STD`) and add `"t3/contact"` + `"t3/contact_shuf"` to `tests/test_arms.py:31`.

**Wiring.** `train.sh` after line 68:
```
[ -n "${CONTACT_GAMMA:-}" ] && { add "contact_gamma=${CONTACT_GAMMA}"; TAG="${TAG}_cg${CONTACT_GAMMA}"; }
[ -n "${CONTACT_SHUF:-}" ]  && { add "contact_shuffle=${CONTACT_SHUF}"; TAG="${TAG}_cshuf"; }
```
`run_campaign.sh` `wave25_arms()` (register in the `case` at `:465`):
```
ORDER[wave25]="PiWM-contact PiWM-contact-shuf PiWM-contact-g05"
ARMS[PiWM-contact]="ltv 1.0 5e-4 CONTACT_GAMMA=1.0"
ARMS[PiWM-contact-shuf]="ltv 1.0 5e-4 CONTACT_GAMMA=1.0 CONTACT_SHUF=true"
ARMS[PiWM-contact-g05]="ltv 1.0 5e-4 CONTACT_GAMMA=0.5"
```

**Control.** `PiWM-contact-shuf` — the same weights, permuted within the batch. It holds the weight *distribution*, and therefore ESS and effective compute, **exactly** fixed (0.359 in both arms) while destroying the alignment with contact. `LpWM-ltv` (n=16, uniform) is the second, unmatched control that prices ESS loss on its own; `PiWM-contact-g05` (ESS/N 0.451) is the dose-response cell that separates "targeting helps" from "any reweighting helps".

**Falsifier.** `PiWM-contact` and `PiWM-contact-shuf` have overlapping paired-CEM CIs against `LpWM-ltv`. That means the weighting is acting only through its variance — an implicit LR/ESS change — and contact targeting is worth nothing. A second, sharper kill: `PiWM-contact` beats `LpWM-ltv` but `PiWM-contact-g05` does not sit between them, i.e. no dose-response.

**Verification (before launching).**
```
# 1. estimator is live and tracks the block (measurement only; states.pth NOT in the loss)
DATASET_DIR=$DATASET_DIR $PY analysis/contact_weight_probe.py --episodes 60
#    must report: spearman(w, block motion) > 0.85 ; mass on block>0.5px > 0.95 ; ESS/N ~ 0.36
# 2. the flag reaches the loss and is inert at default
DATASET_DIR=/nonexistent $PY -c "
import sys,torch; sys.path[:0]=['.','tests']
from lpwm_build import load_cfg,build,synthetic_batch,seed_all
for ov in ([], ['contact_gamma=1.0']):
  c=load_cfg(['predictor=ltv']+ov); seed_all(0); m,_=build(c)
  g=torch.Generator().manual_seed(1234); o,a=synthetic_batch(c,4,g)
  _,_,_,l,comp=m(o,a)
  print(ov, sorted(k for k,v in comp.items() if torch.is_tensor(v)))"
#    default must NOT contain contact_ess ; gamma=1.0 must contain it and contact_w_max
pytest tests/test_bit_identity.py tests/test_arms.py -q
DRYRUN=1 scripts/run_campaign.sh wave25   # env must show CONTACT_GAMMA=1.0 / CONTACT_SHUF=true
```

---

## T5 - PiWM-veq: value-equivalent model loss, with V frozen inside the same run

**Claim.** Replacing the uniform latent metric with the metric induced by a frozen goal-conditioned value head, `M(z) = E_g[∇V ∇Vᵀ]`, reweights the 1-step loss by task relevance without anyone choosing per-sample weights. Added on top of `z_loss`, not in place of it.

**Derivation.** The current objective is `E‖z_pred − z'‖²` with an identity metric, so 48.1% of transitions — the ones where the block displacement is exactly 0.000 px (re-measured this session on `states.pth[..., 2:4]`, reproducing the recorded 48% fully-static figure) — receive the same gradient weight as the 1% of transitions carrying 34.8% of all block motion. That the identity metric is the wrong one is already measured directly: Spearman(latent distance CEM minimises, TRUE task distance) = **+0.398**, and `ltv` (rel_mse 0.0092, CEM 0.357) vs `linvar` (rel_mse 0.0095, CEM 0.080) are indistinguishable under it while differing 4.5× at CEM. A first-order expansion of the T5 term with `e = z_pred − z'` gives `eᵀ M(z) e`, i.e. exactly the same squared error re-metricised, rank ≤ `value_goals` ≪ 384. Directions V is flat in — the no-ops — cost nothing.

**Change.** T5 consumes T4's head. T4 must expose `models/value_head.py: ValueHead(emb_dim, hidden, n_goals)` with `forward(z, g) -> (b, t, n_goal)` and be built in `VWorldModel.__init__` next to the `path_int` block (`models/visual_world_model.py:92-108`), stored as `self.value_head`.

V's own probe loss (T4, hindsight discounted reachability — no privileged state, no reward):
```
positives: g = z_{t+k}, k∈{1..num_hist+num_pred-1}, y = value_gamma**k
negatives: g = z from another batch row,            y = 0
L_V = (V(sg[z_t], sg[g]) - y)^2        # sg on BOTH inputs: V is a probe, never a shaper
```
T5's term, with V's parameters detached (torch 2.3.0 confirmed in the env):
```python
from torch.func import functional_call
def _veq_loss(self, z_pred, target, z_emb):
    g = self._sample_goals(z_emb)                       # (b, n_goal, p, d), detached
    p = {k: v.detach() for k, v in self.value_head.named_parameters()}
    b = {k: v.detach() for k, v in self.value_head.named_buffers()}
    v_p = functional_call(self.value_head, {**p, **b}, (z_pred, g))
    v_t = functional_call(self.value_head, {**p, **b}, (target.detach(), g))
    return (v_p - v_t).pow(2).mean()
```
`functional_call` with detached params gives input gradients and **no** parameter gradients — verified in-env (`x.grad` present, `w.grad is None`). `torch.no_grad()` would kill the `z_pred` gradient and is wrong here.

Insertion in `models/visual_world_model.py`, **after line 687** (`loss_components["ctrb_loss"] = _cl`), **before 688**:
```python
if self.value_head is not None:
    _vl = self._value_probe_loss(z_emb)          # trains V only
    loss = loss + self.value_w * _vl; loss_components["value_loss"] = _vl
    if self.veq_w > 0:
        _ve = self._veq_loss(z_pred, target, z_emb)
        loss = loss + self.veq_w * _ve; loss_components["veq_loss"] = _ve
```
Both values are tensors (`train.py:1096-1099` requires it). `_value_probe_loss` and `_veq_loss` each start with the lazy device move used by `_path_int_loss` (`visual_world_model.py:330-331`).

Config keys, all inert, added to `conf/train_rdmreg.yaml` after line 179: `value_head: false`, `value_w: 1.0`, `value_gamma: 0.9`, `value_goals: 4`, `value_hidden: 512`, `veq_w: 0.0`, `veq_random: false`. Constructor kwargs appended to `VWorldModel.__init__` (`:9-51`), passed at `train.py:789-824`, **mirrored at `tests/lpwm_build.py:84-120`** (omitting them made every V1-V3 arm silently the baseline, twice).

New-head checklist: optimizer group in `train.py init_optimizers` (a fourth AdamW after the action/proprio group at `:894-909`, built only when `value_head is not None` **and** `veq_random is False`); `self._keys_to_save += ["value_head", "value_head_optimizer"]` guarded the same way, inserted after `train.py:499`; `zero_grad`/`step` at `train.py:1076-1082` and `:1086-1092`; entry in `tests/test_arms.py ARMS` (`:18-31`).

**Wiring.** `scripts/run_campaign.sh` `wave23_arms()` + a `wave23)` line in the dispatch `case` (`:447-467`) →
`ARMS[PiWM-veq]="ltv 1.0 5e-4 VALUE_HEAD=true VEQ_W=1.0"` → `submit_arm` `${extra}` (`:437-440`) → `scripts/train.sh`, new lines after `:68`:
`[ -n "${VALUE_HEAD:-}" ] && { add "value_head=${VALUE_HEAD}"; TAG="${TAG}_val"; }`, `[ -n "${VEQ_W:-}" ] && { add "veq_w=${VEQ_W}"; TAG="${TAG}_veq${VEQ_W}"; }`, `[ -n "${VEQ_RANDOM:-}" ] && { add "veq_random=${VEQ_RANDOM}"; TAG="${TAG}_rnd"; }` → `${EXTRA}` at `train.sh:102` → `conf/train_rdmreg.yaml` → `train.py:789-824` → `VWorldModel`. Nothing changes at plan time: T5 is a training objective, and the head is not on the rollout path.

**Control.** Three, all in `wave23`. `PiWM-value` (`VALUE_HEAD=true VEQ_W=0`) = T4 alone: V trains and is logged but never touches the encoder — separates "the head exists" from "the metric acts". `PiWM-veq-rand` (`VEQ_RANDOM=true`, head never given an optimizer and `value_w` forced to 0) = the frozen **random** head: same architecture, same rank, same conditioning, zero task content. **T3** (saliency-weighted `z_loss`) is the ablation for the outer claim. Matched baseline is the existing `LpWM-ltv` (n=16) at the same seeds.

**Falsifier.** Any one of: (a) paired CEM(`PiWM-veq`) − CEM(`LpWM-ltv`) ≤ 0 over ≥ 3 shared seed blocks; (b) `PiWM-veq` ≈ `PiWM-veq-rand` — the gain is a random low-rank metric, not value; (c) `PiWM-veq` ≈ **T3** — value added nothing beyond saliency; (d) `value_loss` fails to fall below its `y≡mean` floor, in which case V never learned reachability and (a) is uninterpretable.

**Verification.**
```bash
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
$PY -m pytest tests/test_bit_identity.py tests/test_arms.py -q     # defaults still upstream
$PY - <<'EOF'
import sys; sys.path.insert(0,'tests'); import torch
from lpwm_build import build, load_cfg, seed_all, synthetic_batch
cfg = load_cfg(["predictor=ltv","value_head=true","veq_w=1.0","value_w=0.0"])
seed_all(0); m,_ = build(cfg)
obs,act = synthetic_batch(cfg,2,torch.Generator().manual_seed(1))
_,_,_,loss,c = m(obs,act)
assert "veq_loss" in c and float(c["veq_loss"]) > 0, "T5 term is dead"
loss.backward()
assert all(p.grad is None for p in m.value_head.parameters()), "V is NOT frozen"
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.predictor.parameters())
print("veq live, V frozen, predictor receives it")
EOF
DRYRUN=1 SEEDS=3 scripts/run_campaign.sh wave23 | grep -E 'VEQ_W|VALUE_HEAD'
grep -E '^(value_head|veq_w|veq_random):' runs/outputs/PiWM-veq_pd384_bf16_s3/hydra.yaml
```
Then after ~200 steps confirm `veq_loss` and `value_loss` are both present and non-constant on the wandb run page (`train.py:1096-1099` gathers them by name).

---

## T6 - PiWM-jump5: a K=5 option model, trained directly on 5-apart pairs

**Claim.** Train `z_t → z_{t+5}` as one step in option time. The planner's whole horizon (`goal_H=5`) is then a single predictor call: zero compounding, and the no-ops are absorbed inside the option instead of being 48% of the training signal.

**Derivation.** Measured this session on `pusht_noise/train/states.pth`: over one action row (5 env steps) the block's displacement is exactly 0.000 px in **48.1%** of transitions and its median is 0.084 px. Over a 5-row option (25 env steps) the exactly-zero fraction falls to **12.7%** and the median net displacement rises to **24.6 px** — a 293× increase in the median signal the predictor is asked to model, from the same data. Compounding is the other half: CEM at `goal_H=5` currently chains five 1-step predictions, and the 1-step error is already known not to be what matters (`ltv` 0.0092 → CEM 0.357 vs `linvar` 0.0095 → CEM 0.080). Multi-time option models compose exactly like one-step models, so nothing downstream needs to change.

**Change.** `num_pred` **is** K. `conf/train_rdmreg.yaml:124` currently reads `num_pred: 1 # only supports 1`; that comment is the pin. What actually breaks at `num_pred=5`, checked line by line:

* `datasets/pusht_dset.py:162` `num_frames = num_hist + num_pred` → 8 frames, spanning 40 env steps. Window count falls from **1,981,721 → 1,608,021** train windows (−18.9%; val 2,115 → 1,695). No trajectory is dropped (min length 49 ≥ 40).
* `models/visual_world_model.py:628` `z_tgt = z_emb[:, self.num_pred:]` → indices [5,6,7] against `z_src` [0,1,2]. Shapes already correct for any K; pairs are (0→5),(1→6),(2→7). **No change.**
* `models/visual_world_model.py:627` `act_src = act_emb[:, :self.num_hist]` is the real bug: row *i* is the action for frame *i*→*i*+1 only, so a K=5 arm would be conditioned on 1/5 of the actions and would be a null for a trivial reason. Replace with the **option action**:

```python
def _option_act(self, act_emb):
    """Mean of the K action embeddings the option spans. K == 1 returns the input
    object unchanged, so the default path is bit-identical."""
    k = self.num_pred
    if k <= 1:
        return act_emb
    x = torch.cat([act_emb, act_emb[:, -1:].expand(-1, k - 1, -1)], dim=1)
    return torch.stack([x[:, i:i + k].mean(1) for i in range(act_emb.shape[1])], dim=1)
```
Averaging **embeddings**, not raw actions, and averaging rather than summing: concatenating the K rows would move `action_encoder.patch_embed`'s fan_in from 10 to 50 and therefore its muP learning rate by 5× (`models/mup.py:57`, `used_lr = base_lr * base_width / fan_in`) — the exact confound documented for `use_pose` at `visual_world_model.py:343-346`; summing would multiply the AdaLN conditioning norm by K. At K=1 both are identities. The training slice never touches the pad (max row read is `num_hist-1+K-1 = 6 ≤ 7`).

* Rollout, `models/visual_world_model.py:871-881`: one predictor call must advance K rows. Give `_predict_next_adaln` (`:835`) an explicit `rows` argument (default `list(range(L))`, which reproduces `act_emb_all[:, L-h:L]` exactly) and index `opt_emb[:, rows[-h:]]`. `_rollout_adaln` keeps `rows` alongside `emb`, starting `r = n-1` and looping `while r + K <= act.shape[1]: ...; r += K`. At `goal_H=5`, K=5, `obs_0` carrying one frame (`plan.py` contract), this is **one** call with `h=1`, and `z_obses["visual"]` has time length 2. `evaluator._get_trajdict_last` (`planning/evaluator.py:62-73`) takes index −1 when `action_len` is `inf`, and `objectives.objective_fn_last` slices `[:, -1:]`, so both are unaffected.
* `train.py:1038-1039`/`:1105`/`:1120-1123` are all inside `if self.cfg.has_decoder and plot:` (`:1100`, `:2098`) and `has_decoder: False`. Dead.
* Plan time is free: `train.py:389` writes `hydra.yaml` with `resolve=True`, and every saved run already carries both `num_pred:` and `model.num_pred:` (verified across `runs/outputs/*/hydra.yaml`), so `plan.py:517` rebuilds with the right K.

Control mechanism, new key `overshoot: false` (inert): at the **same** `num_pred=5` window — identical data, identical window count, identical batch composition — unroll the 1-step map five times from frame 2 with per-row actions `act_emb[:, 2:7]` and supervise every intermediate frame 3..7, `L = mean_j ‖z_j − z_emb[:, 2+j]‖²`. Same effective horizon, compounding retained.

`tests/lpwm_build.py` needs nothing new — `num_pred` already flows through `cfg.model` (`conf/train_rdmreg.yaml:132`) and `synthetic_batch` already sizes `T = cfg.num_hist + cfg.num_pred` (`:156`) — but `overshoot` must be mirrored at `:84-120`, and both arms added to `tests/test_arms.py ARMS` (`:18-31`).

**Wiring.** `scripts/run_campaign.sh` `wave24_arms()` + a `wave24)` dispatch line (`:447-467`):
`ARMS[PiWM-jump5]="ltv 1.0 5e-4 NUM_PRED=5"`, `ARMS[PiWM-overshoot5]="ltv 1.0 5e-4 NUM_PRED=5 OVERSHOOT=true"` → `scripts/train.sh` after `:68`: `[ -n "${NUM_PRED:-}" ] && { add "num_pred=${NUM_PRED}"; TAG="${TAG}_K${NUM_PRED}"; }` and `[ -n "${OVERSHOOT:-}" ] && { add "overshoot=${OVERSHOOT}"; TAG="${TAG}_os"; }` → `conf/train_rdmreg.yaml:124` (comment deleted) and a new `overshoot: false` after line 179 → dataset (`train.py:393-398`) **and** `train.py:789-824` model kwargs. Plan side unchanged: `conf/plan_lewm.yaml` planner/objective stay `MPCPlanner`+`CEMPlanner`+`create_objective_fn`; `plan.py:201-202` still forces `horizon = n_taken_actions = goal_H = 5`, which at K=5 is exactly one option.

**Control.** `PiWM-overshoot5` — same 8-frame windows, same 25-env-step horizon, compounding kept. Jumpy > overshooting means compounding through no-ops was the problem; jumpy ≈ overshooting means it was the horizon, not the chaining. Matched baseline `LpWM-ltv` (n=16).

**Falsifier.** Paired CEM(`PiWM-jump5`) − CEM(`LpWM-ltv`) ≤ 0 over ≥ 3 shared seed blocks kills it. `PiWM-jump5` ≈ `PiWM-overshoot5` kills the *compounding* attribution specifically. Note `rel_mse` is **not** comparable across these arms (different targets); the only admissible error comparison is 5-step-rolled `LpWM-ltv` vs 1-option `PiWM-jump5` against the same frame `t+25`.

**Verification.**
```bash
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
$PY -m pytest tests/test_bit_identity.py tests/test_optimistic_rollout.py -q   # K=1 unchanged
$PY - <<'EOF'
import sys; sys.path.insert(0,'tests'); import torch
from lpwm_build import build, load_cfg, seed_all, synthetic_batch
cfg = load_cfg(["predictor=ltv","num_pred=5"]); seed_all(0); m,_ = build(cfg); m.eval()
obs,act = synthetic_batch(cfg,2,torch.Generator().manual_seed(1))
assert obs["visual"].shape[1] == 8, obs["visual"].shape
zp,_,_,loss,c = m(obs,act); assert zp.shape[:2] == (2,3), zp.shape
n = [0]
h = m.predictor.register_forward_hook(lambda *a: n.__setitem__(0, n[0]+1))
obs0 = {k: v[:, :1] for k,v in obs.items()}
zo,_ = m.rollout(obs_0=obs0, act=torch.rand(2,5,10))
h.remove(); assert n[0] == 1 and zo["visual"].shape[1] == 2, (n[0], zo["visual"].shape)
print("K=5: 8-frame window, ONE predictor call over goal_H=5, 2 latent frames")
EOF
DATASET_DIR=$(grep -oP '(?<=DATASET_DIR=).*' .env) $PY -c "
import hydra,pickle,numpy as np
L=np.asarray(pickle.load(open('$DATASET_DIR/pusht_noise/train/seq_lengths.pkl','rb')))
print('K=1 windows',np.maximum(L-20+1,0).sum(),'K=5 windows',np.maximum(L-40+1,0).sum())"
# expect: 1981721 / 1608021
DRYRUN=1 SEEDS=3 scripts/run_campaign.sh wave24 | grep NUM_PRED
grep -E '^num_pred:|^  num_pred:|^overshoot:' runs/outputs/PiWM-jump5_pd384_bf16_s3/hydra.yaml
```

---

## T7 - PiWM-energy: a frozen-encoder ranking head over action sequences, and a drop-in CEM objective

**Claim.** With the encoder frozen, train `E_θ(z_0, a_{1:H}, z_g)` by cross-entropy to rank the executed demo sequence above CEM-distributed samples. Because `z` is a constant function of the data, the representation cannot move, so the InfoNCE collapse that killed both `act_info` arms is structurally unavailable — not merely discouraged.

**Derivation.** InfoNCE over actions collapsed the code under both negative distributions we tried: batch-perm drove ρ 0.448 → 0.052, and kNN-from-`p(a|z)` reached `rel_mse 1.0000`, ρ 0.13, `d_action 0.000`, temporally frozen. In both, the encoder was the free parameter and zeroing the code was the cheapest way to satisfy the contrastive term. Freezing it removes that degree of freedom entirely. The target quantity is the one CEM actually consumes: the planner only ever computes `argsort(loss)[:topk]` (`planning/cem.py:115`), so it needs a correct **ranking** of candidate sequences, not a calibrated latent distance — and the current latent distance ranks poorly (Spearman vs TRUE task distance **+0.398**). Note the earlier version of this argument added that `d_action` is a non-predictor of CEM; **that is withdrawn** (diary §12b). It does predict CEM, non-monotonically, and the frozen predictor already sits near the optimum — so T7's case is about the *ranking* the leaf metric induces, not about action-sensitivity, and the energy head must be judged on rank agreement alone.

**Change.** Four new files plus one hook; **nothing is added to `VWorldModel`**, so its optimizer/`_keys_to_save`/bit-identity surface is untouched.

1. `models/energy_head.py`:
```python
class SeqEnergyHead(nn.Module):        # emb_dim=384, H=5, act_dim=10, hidden=512
    def forward(self, z0, ag, zg):     # (B,D),(B,H,act_dim),(B,D) -> (B,)
        a = self.act_mlp(ag.flatten(1))                     # fresh trunk, fan_in H*act_dim=50
        return self.mlp(torch.cat([z0, zg, z0 - zg, a], -1)).squeeze(-1)
```
A **fresh** action trunk on purpose: the checkpoint's `action_encoder` is the module suspected of carrying no action information, so reusing it would bottleneck the head with the defect under test.

2. `train_energy.py` (repo root, mirroring `analysis/d_action_probe.py:56-65,93` for loading): read `runs/outputs/<run>/hydra.yaml`, `load_model(...)` from `plan.py:422`, `model.eval()` + `requires_grad_(False)` on everything, then `hydra.utils.call(cfg.env.dataset, num_hist=3, num_pred=5, frameskip=5)` → 8-frame windows, **1,608,021** train / 1,695 val (measured). Per window: `z0 = encode_obs_linked(obs)["visual"][:, 2]`, `zg = ...[:, 7]`, positive `a⁺ = act[:, 2:7]` (frame 2 → frame 7 is exactly 5 rows). Negatives, matched to the CEM proposal at `planning/cem.py:99-106` (`randn * sigma + mu`, `var_scale=1`):
```
M = 63 negatives per anchor:  half  a⁻ ~ N(0, 1)                       # opt step 0
                              half  a⁻ = a⁺ + s*eps, s ~ U(0.1, 1.0)   # elite-concentrated steps
L = -log softmax(-E over the 64 sequences)[0]      # cross-entropy over sequences
```
Saves `{"state_dict", "arch", "run", "step", "val_top1"}` to one `.pt`.

3. `planning/cem.py`: extract `:107-114` verbatim into a hook, leaving the default path numerically identical:
```python
def _score(self, obs_0, action, z_g):
    with torch.no_grad():
        z_obses, _ = self.wm.rollout(obs_0=obs_0, act=action, z_goal=z_g["visual"])
    return self.objective_fn(z_obses, z_g)
```
`planning/energy_cem.py: EnergyCEMPlanner(CEMPlanner)` overrides it with `self.energy_head(z0, action, zg)` — **no rollout at all**, so no compounding and no `_roll_pose`.

4. `planning/mpc.py`: add `energy_head=None` to `__init__` (`:15-30`) and forward it in the `sub_planner` instantiate (`:45-54`). `plan.py`, inserted before `:187`: load `energy_ckpt`, `.eval()`, `requires_grad_(False)`, pass `energy_head=self.energy_head` into the planner instantiate at `:187-197`; and **raise** if `EnergyCEMPlanner` is selected while `energy_ckpt` is null. That last clause is the direct fix for the `path_int` failure mode — a head that is absent from disk and silently randomly initialised at plan time (`visual_world_model.py:383-389`) must be impossible here.

Config: `conf/plan_lewm.yaml` gains `energy_ckpt: null` (inert, alongside `ensemble_members` at `:36`); the planner is selected by `planner.sub_planner.target=planning.energy_cem.EnergyCEMPlanner`.

**Wiring.** `scripts/train_energy.sh` + `scripts/energy_slurm.sbatch` (copy `scripts/plan_slurm.sbatch:31-60`, keeping the `DONE`-sentinel refusal at `:52-55`), then eval exactly as `scripts/plan_vote.sh:28-33` does its override:
```bash
python plan.py --config-name plan_lewm.yaml \
  ckpt_base_path="${CKPT_BASE}" model_name="${LABEL}" model_epoch=latest \
  energy_ckpt="${EHEAD}" \
  planner.sub_planner.target=planning.energy_cem.EnergyCEMPlanner \
  n_evals=50 planner.max_iter=10 ${SEED:+seed=$SEED}
```
`LABEL` keeps the `<arm>_pd384_bf16_s<block>` shape so `analysis/collect_evals.py` files it and pairs it against `LpWM-ltv` on the same episode block. No training-side wiring: `run_campaign.sh` is not involved, since T7 trains no world model.

**Control.** `PiWM-energy-distill`: the **same** `SeqEnergyHead`, same triples, same negatives, trained by regression on `y = ‖z_H(a) − z_g‖²` taken from the current model's own rollout — i.e. the quantity CEM minimises today, distilled into the head. If it matches `PiWM-energy` at CEM, the gain was capacity and smoothing, not the ranking objective. Baseline `LpWM-ltv` (n=16) at the same seed blocks.

**Falsifier.** Any of: (a) paired CEM(`PiWM-energy`) − CEM(`LpWM-ltv`) ≤ 0 over ≥ 3 shared blocks; (b) `PiWM-energy` ≈ `PiWM-energy-distill`; (c) held-out top-1 ranking accuracy over 64 sequences ≈ 1/64 = 1.6%, in which case the head learned nothing and (a) is uninterpretable; (d) `val_top1` is high but CEM does not move, which localises the failure to the frozen `z` carrying no goal-relevant content — the T5/T6 question, not T7's.

**Verification.**
```bash
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
$PY -m pytest tests/test_energy_planner.py -q
#   asserts: (1) CEMPlanner._score reproduces the pre-refactor inline path bit-for-bit
#            (2) EnergyCEMPlanner raises when energy_head is None
#            (3) EnergyCEMPlanner never calls wm.rollout (forward hook count == 0)
$PY train_energy.py --run LpWM-ltv_pd384_bf16_s3 --steps 2000 --out assets/energy_s3.pt
$PY -c "
import torch; p=torch.load('assets/energy_s3.pt',map_location='cpu')
print(p['run'], p['step'], 'val_top1', p['val_top1'])
assert p['val_top1'] > 0.10, 'head did not learn to rank (chance = 0.0156)'"
# the anti-path_int check: the head the PLANNER uses is the head on disk
EHEAD=assets/energy_s3.pt scripts/plan_energy.sh 2>&1 | grep 'energy head loaded from'
$PY -c "
import torch
f=torch.load('assets/energy_s3.pt',map_location='cpu')['state_dict']
print('param checksum', float(sum(v.double().sum() for v in f.values())))"
# must equal the checksum printed by plan.py at load; a fresh init cannot match it
$PY -c "import plan" && grep -n 'energy_ckpt' plan.py conf/plan_lewm.yaml
```
