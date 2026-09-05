# How to read this record

This file is the glossary and the reading guide. It exists because almost every wrong turn in
this campaign was a **metric** misread rather than a model failure, and because the entries
themselves are written for someone who already knows what the numbers mean.

Read this first, then the four entries in date order.

Every definition below is verified against the source it is computed in; the file and line are
given so a disagreement can be settled without re-deriving anything.

---

## 0. What the system is, in plain words

**A model that predicts its own compressed view of the next frame, and a planner that searches
action sequences against that prediction.**

An encoder `f` turns a PushT camera frame `o_t` into a short vector of numbers `z_t` — the
*latent*, or *code*. A predictor `g` takes that code and the action the robot is about to take,
and produces a guess at the *next* code, `ẑ_{t+1} = g(z_t, a_t)`. Training compares that guess
against the code the encoder actually produces for the next frame, `z_{t+1}`, and pushes them
together. Nothing ever reconstructs a pixel — the target is the model's own encoding of the
future, which is what makes this a **JEPA** (Joint-Embedding Predictive Architecture) rather
than an autoencoder or a video model.

At evaluation time the weights are frozen and a **planner** is bolted on. It proposes several
hundred candidate action sequences, rolls each one through the predictor, scores how close the
final predicted code lands to the goal's code, keeps the best ones, and re-samples around them.
The winning first action is executed on the real simulator, the whole search restarts from the
new frame, and after up to ten such rounds the episode is scored a success or a failure. That
number — the fraction of episodes that succeed — is the **only** outcome this campaign is
judged on. Everything else in this glossary is a diagnostic.

Two consequences run through the whole record:

* The training objective asks the model to be **accurate**. The planner needs the model to be
  **discriminative between actions**. Those are not the same requirement, and the gap between
  them is the subject of `2026-09-02` §8 and all of `2026-09-03`.
* The planner consumes an **ordering** of candidate action sequences, not an error magnitude.
  Two models with identical prediction error can induce opposite orderings
  (`2026-09-03` §14 rule 3).

---

## 1. The four entries

| file | covers | length |
|---|---|---|
| `2026-09-01.md` | rounds 1–2: the sparse / SDR era, and the eval bug that invalidated it | 7 figures |
| `2026-09-02.md` | the architecture round (arch-00..09), the one positive, and the rounds 1–2 conclusion | 15 figures |
| `2026-09-03.md` | rounds 4–5: the causal round, the audit, and the measurement round that overturned its own premise | 41 figures |
| `2026-09-04.md` | round 6: the duplication audit, the objective screen, and the round it produced | 7 figures |

**Two measurement instruments, never pooled.** `plan.py:134` used to build the eval episode list
as `[seed*n + 1 for n in range(n_evals)]`, which degenerates at seed 0 to `[1]*50` — fifty
copies of one episode. It also gave *different* episode sets to different seeds, so eval noise
was not common-mode and did not cancel in a paired difference. Fixed in `49a3e55` to disjoint
blocks. `analysis/collect_evals.py` classifies every eval as `buggy` (pre-fix) or `fixed`
(post-fix) from the `eval_seed:` line each planning job prints, and **refuses to pool them**.

> **`2026-09-01.md` is entirely on the pre-fix instrument.** Its arms were never re-evaluated,
> so its figures are rebuilt on `scheme="buggy"` and carry a banner saying so. Its numbers are
> internally consistent and are not comparable to any later entry's. Everything from
> `2026-09-02.md` onward is `scheme="fixed"`.

Four corrections in the record overturn things the record previously asserted. They are left in
place, with their reasoning, because a corrected mistake is more useful than a clean story:
`2026-09-03` §8's SUPERSEDED banner, §12b (`d_action` wrong by ~2900×), §12's P4 correction
(a null by construction → an *interventional* null, corrected in `2026-09-04` §5.2), §16.4 (the
round-5 panel's premise refuted by its own measurement M3), and `2026-09-04` §1 (three of five
proposals were duplicates) and §2 (`h8/h1` and `d_action` are collapse detectors).

---

## 2. The outcome: CEM success rate

**Key:** `final_eval/success_rate` (`analysis/collect_evals.py:41`).
**Definition:** `env/pusht/pusht_wrapper.py:111` —

```python
pos_diff   = np.linalg.norm(goal_state[:4] - cur_state[:4])
angle_diff = wrapped |goal_theta - cur_theta|
success    = pos_diff < 20 and angle_diff < np.pi / 9
```

**In plain words.** Fifty PushT episodes. In each, the planner gets up to ten replanning rounds
(`max_iter: 10`) and each round runs CEM: 300 sampled action sequences of horizon 5, keep the
top 30, 30 refinement steps (`conf/planner/cem.yaml`). The episode counts as a success if the
scene ever came within 20 px and 20° of the goal. The reported number is the fraction of the
fifty that did.

| | value |
|---|---|
| floor | **0.000** — 23 % of all evaluated runs on the repaired instrument score exactly zero |
| baseline `LpWM-ltv` | **0.357** (n = 13) |
| best single model `LpWM-ltv-d2048` | 0.587 (n = 6) |
| best arm `PiWM-vote5-borda` (5-model plan-time vote) | **0.608** (n = 8), paired **+0.215 [+0.089, +0.341]** vs baseline |
| best observed single seed | 0.780 |
| oracle-dynamics ceiling | ≈ **0.913** (`2026-09-03` §16.1's ladder) |

**Good** is anything that clears its own matched control by more than the paired interval.
**Bad**, in this campaign, has a very specific shape: **0.000 on every seed**. A collapsed model
does not degrade gracefully; it plans at the floor. That is why so many diagnostics below look
predictive and are not (§8).

**Three traps.**

1. **Half the positional resolution is the end-effector.** `state` is
   `[agent_x, agent_y, T_x, T_y, T_theta, ...]`, so `[:4]` scores the agent's own position as
   well as the block's — a quantity the policy controls directly, with no contact and no
   physics. `2026-09-03` §16.5 measured it: **44 % of every success on record is on an episode
   where the block never had to move**, and re-ranking the 32 arms with ≥ 4 seeds on the
   block-only metric moves them a mean of 4.34 places. The published comparisons nonetheless
   survive at Spearman(old, new) = **+0.95**, because the reordering is concentrated in the
   floor of collapsed arms the old metric could not separate anyway.
2. **Success is latched.** `planning/mpc.py:110` does `self.is_success |= successes` inside the
   replanning loop, so the number is "did the episode pass through the success ball at *any* of
   up to ten checkpoints", a maximum over ten noisy draws. Measured on 1,604 live episodes,
   latched and terminal agree on **0 flips** (`2026-09-03` §16.5) — cosmetic here, but it means
   the metric rises with `max_iter` independently of the model.
3. **The block-only alternative has a 0.313 do-nothing floor**, because 31 % of episodes start
   with the block already inside tolerance. If you adopt `success_block`, restrict it to
   must-push episodes.

---

## 3. Training-time metrics

All of these are written to each run's `wandb-summary.json` at the end of training and are read
back by the figure scripts from `runs/outputs/<run>/wandb/latest-run/files/wandb-summary.json`.
Ranges below are over the **591 completed non-canary runs** on disk unless stated otherwise.

### 3.1 `err/rel_mse` — prediction error, scale-free

`train.py:1776`: `mean((ẑ − z)²) / mean(z²)`.

**Plain words.** The predictor's squared error divided by the energy of the thing it is
predicting. Dividing by the target's own scale is what makes arms with different code
magnitudes comparable at all.

| | |
|---|---|
| **healthy** | `LpWM-ltv` median over 16 seeds: **0.00919** |
| campaign quartiles | p25 0.0099 · median 0.0231 · p75 0.136 |
| **1.0 means predicting the mean — the model is dead** | |
| worse than predicting zero | k-WTA @ 2 %: **1.93** |
| observed maximum | 7.01 |

**The pre-registered death condition is `rel_mse ≥ 0.5`** (`2026-09-04` §4), and
`rel_mse > 0.025` at end of training flagged **23/23** catastrophic runs with zero false alarms
(`2026-09-02` §10 item 6). It is the cheapest screen in the project.

**The trap.** `rel_mse` is normalised by target *energy*, not against a trivial predictor, so it
cannot see a target that stopped moving. `PiWM-blockcausal` posted `rel_mse` = 0.0022, five
times better than any arm at full training, while the **identity predictor** ("copy the previous
latent") scored 0.000010 on the same data — the learned predictor was 220× *worse* than copying
(`2026-09-02` §2.9). A near-constant code makes `rel_mse` look superb.

### 3.2 `sparsity/val_l0_frac` (written ρ) — code density

`models/visual_world_model.py:1594`: `(z != 0).float().mean()`.

**Plain words.** The fraction of the code's entries that are non-zero. Because the link function
is a ReLU, "non-zero" means "this unit fired".

| | |
|---|---|
| **0.0** | dead code — nothing fires, ever |
| **healthy** | `LpWM-ltv` median **0.448**; campaign p25–p75 **0.372–0.603** |
| **1.0** | every unit always on; the rectifier has stopped doing anything |
| observed | 0.000 (`PiWM-union4`, `PiWM-drop95`) to 0.99999 (`PiWM-sigreg-w0p5`) |

Both ends are failure. `PiWM-sigreg-w0p5` sits at ρ = 0.9999 *and* `rel_mse` = 0.9995 — fully
dense and not predicting.

**The trap, and it is a big one.** ρ is ambiguous on its own: the same density can come from
every unit firing half the time, or from half the units firing always and the rest being dead
(`train.py:1575`). `sparsity/effective_dim` and `sparsity/dead_unit_frac` are what separate
those. And ρ ≈ 0.5 comes from the **ReLU link, not from the sparse prior** — `2026-09-02` §7
shows RDMReg structurally cannot distinguish a p=1 target from a p=2 one at D=384.

### 3.3 `sparsity/effective_dim` — how many directions the code actually uses

`train.py:141`: the participation ratio `tr(C)² / ‖C‖_F²` of the code's covariance. 1 when the
code is rank-1 (collapsed), D when it is white. `effective_dim_pred` is the same thing on the
*prediction* rather than the encoding.

| | |
|---|---|
| **0.0** | dead |
| collapsed-but-alive | 3.1 (`LeWM-ltv-p2`), 2.4–4.9 (`PiWM-sparse-matched`) |
| **healthy** | `LpWM-ltv` median **24.1**; campaign p25–p75 **9.8–24.5** |
| widest observed | 43.0 |

**Benchmark it against the task, not against D.** PushT's own participation ratio is **4.31**,
so an effective dimension of 24 is ~5.6× over-complete — comfortably healthy. Measured against
D = 384 the same number reads as a 94 % collapse, and that reading is wrong. This is correction
1 in `2026-09-01` §5. Quadrupling D buys +15 % effective dim (D=384 → 20.9, D=768 → 23.9,
D=1536 → 24.1 at matched epoch): the metric tracks the task, not the width.

**Second trap.** It is still rising at 100 % of training (18.2 → 28.7 on one run), so
mid-training cross-arm comparisons of it are not trustworthy.

### 3.4 `causal/d_action`, and the ratio `d_action/|z|`

`train.py:1635-1682`. Permute the actions within the batch, keep the states, and measure how far
the *prediction* moves: `d_action = RMS(g(z, a) − g(z, a_perm))`. `d_state` is the mirror image
(permute the states, keep the actions). `d_action_over_scale` divides by `RMS(g(z,a))`.

**Plain words.** "If I had pressed a different button, how different would the model's guess
be?" That is exactly the quantity CEM consumes, because CEM can only distinguish candidate
sequences to the extent that different actions produce different predictions.

| | |
|---|---|
| **0.000** | the predictor is action-blind. `PiWM-incr` measures exactly 0 on 7 of its 8 seeds |
| **healthy** | `LpWM-ltv`, re-probed over 16 seeds: `d_action` **0.354**, `d_action/‖z‖` **0.549** |
| campaign (327 checkpoints re-probed) | 0.000 · median **0.428** · 6.76 |
| **the optimum is near 0.6** | `LpWM-ltv-d2048`, the best single model, sits at 0.647 |
| too much | `PiWM-actinfo` at 1.25 plans at 0.09 |

**This metric carries the campaign's largest single error, and the record keeps it visible.** A
docstring quoted `d_action = 1.29e-04` for the baseline against `|z| = 0.68` — one hand-measured
seed. Nine arms across rounds 3 and 4 were designed to raise a quantity believed to be ≈ 0.
Re-probing every checkpoint in the archive (`analysis/d_action_probe.py`; agrees with the
training-time log at Spearman +0.992 over the 145 runs that have both) gives 0.549. **The quoted
number was wrong by a factor of ~2900** (`2026-09-03` §12b). The baseline was never
action-inert; it sat in the top quartile of the campaign, above every arm built to raise it.

**The trap is the shape, not the value.** The relation to planning is an **inverted U**
(rank-quadratic −323): too little action influence and the planner cannot tell candidates apart;
too much and the latent is dominated by the action rather than the world. Binned over healthy
predictors, mean CEM goes 0.014 → 0.134 → 0.328 → **0.465** → 0.376. A partial correlation
cannot see that, which is why `analysis/screen_objective.py` reports binned quartile means as a
mandatory fourth item. **"Raise `d_action`" is not a direction.**

**Second trap.** `causal/d_action` is computed inside the training loop, so it exists only for
runs trained after the diagnostic was added — which excludes `LpWM-ltv`, `-d2048`, `-vfloor`,
`-mupfix` and the entire gate family, i.e. every high-CEM arm. Any cross-arm claim must use the
checkpoint probe, not the log. A metric that exists only for some runs is not a measurement of
the population (`2026-09-03` §14 rule 4b).

### 3.5 `jacc/S_model` — support dissimilarity, the one screened target

`train.py:1608-1626`, on `models/stats.py:20`:
`S_model = 1 − soft_Jaccard(ẑ, z)` where `J_S(a,b) = Σ min(a,b) / Σ max(a,b)`.

**Plain words.** `rel_mse` asks *how far off* the predicted magnitudes are. `S` asks *which
units fire* — whether the prediction lights up the same set of code units as the truth. On a
code at ρ ≈ 0.45 those are genuinely different questions. `2026-09-04` §2 has the worked pair:
two predictions with **identical squared error (0.152)**, one with the right support and
overshooting magnitudes, one with exact magnitudes and four spurious units firing. Their `S`
differs, 0.280 against 0.357.

| | |
|---|---|
| **healthy** | `LpWM-ltv` median **0.082**; campaign p25–p75 **0.081–0.263** |
| **1.0 = total disagreement, and a dead code scores exactly 1.0** | `PiWM-union4`, `PiWM-drop95`, `PiWM-sparse-2pct` all sit at 1.0 |
| observed minimum | 0.031 |

**This is the only quantity in the project that has passed the screen.** Raw Spearman with CEM
−0.769; **partial −0.549** with ρ and `rel_mse` removed (n = 296, permutation p ≈ 0); and
**monotone** over healthy predictors, 0.407 → 0.365 → 0.249 → 0.058. `soft_jaccard` appears zero
times in `models/` as an objective — it has only ever been a diagnostic. R6 is the arm that
optimises it.

**The trap, stated before the arm ran.** The Jaccard is scale-invariant in a way MSE is not, so
a model could lower `S` by inflating the code's magnitude uniformly. R6 logs the code norm every
epoch so that is visible rather than inferred.

Companions: `jacc/S_world = 1 − J_S(z_t, z_{t+1})` is the observed-change baseline `S_model` has
to beat (median 0.259), and `jacc/support_churn` (`train.py:131`) is the hard-binary version —
the fraction of units whose on/off state flips. A large `S` with near-zero churn means
"magnitudes wrong", not "support reorganised".

### 3.6 `h1` and `h8/h1` — rollout error growth

Built from `rollout/val_z_visual_err_rollout_h1` and `..._h8`: the prediction error after one
chained step and after eight.

**Plain words.** The planner does not take one step; it rolls the predictor five steps into the
future and scores the end. `h8/h1` asks how much the error compounds along that rollout.

| | |
|---|---|
| **≈ 1.0** | the predictor is emitting a constant — error does not grow because nothing moves. `PiWM-actinfo-cond` measures **1.02** |
| **healthy** | arm medians ~12 (`LpWM-ltv-d2048`, the best single model) to ~25 (`LpWM-linvar`); `LpWM-ltv` s15 measures **19.4** |
| campaign span | arm medians 0.88 to 74; per-run 0.05 to 184 |

**`h8/h1` is a collapse detector and was killed as a design target before a line was written**
(`2026-09-04` §2), for zero GPU:

```
Spearman(h8/h1, CEM)                = +0.558   p = 8e-28
partial, removing rho and rel_mse   = -0.017   p = 0.77
healthy only (rel_mse < 0.05)       = +0.002
```

The raw number looks like a strong signal. It is entirely explained by dead models planning at
zero. `h1` on its own behaves the same way: raw −0.552, partial −0.090.

**A second trap it exposes.** A *low* `h8/h1` is not good news — the audit found that an
action-information objective produced the contraction it was proposed to prevent
(`2026-09-04` §1). The untreated baseline is not contracting.

### 3.7 `value/rho_k` — is the value head alive

`train.py`, `_value_diagnostics`: `Spearman(V(z_t, z_{t+k}), −k)` over k ∈ {1,2,3} from a common
anchor, so the correlation is about the *offset* and not about which frame the anchor was.

**Plain words.** "Does the value head know that a state three steps away is further than a state
one step away?" A head that was built but never optimised sits at ρ ≈ 0 here — which is the
liveness check `path_int` never had, and its absence cost a retracted conclusion
(`2026-09-03` §13.3).

| | |
|---|---|
| **gate** | T4's pre-declared liveness threshold is **> 0.6** |
| observed (24 runs) | 0.221 · median 0.541 · 0.833 |
| `PiWM-vp` (TD, the spec's *primary* setting) | **0.443 — fails its own gate** |
| `PiWM-vp-mc` | **0.715 — passes** |

That is why `2026-09-04` §3 insists on grids: T4 would have died on its primary setting, and
T5/V5/V4 must build on `vp-mc`.

### 3.8 The rest, briefly

* `sparsity/dead_unit_frac` / `dead_unit_count` — units that never fire over an accumulation
  window. Deliberately accumulated rather than per-batch, because at ρ ≈ 0.5 no unit is dead
  inside a single 64-sample batch and a per-batch answer would always be 0 (`train.py:157`).
* `causal/state_over_action` = `d_state / d_action`. 37× for the baseline as originally
  measured; read it alongside §3.4's correction.
* `err/cos_pred_target`, `err/mse_on_support`, `err/mse_off_support` — the same error split by
  direction and by whether the true unit was on.
* `reg/density_gap` — measured ρ minus the regulariser's analytic target density. A large
  persistent gap means the regulariser is fighting k-WTA over the density rather than doing
  useful work.
* `std_t(z)` and `mean|Δz|/|z|` — temporal variation. **A floor condition, not a quality axis**:
  Spearman(`std_t(z)`, CEM) = +0.05, p = 0.90 over 9 arms, and `union4-vfloor` has the highest
  temporal variation of any arm while planning at 0.000 (`2026-09-02` §2.9).

---

## 4. Checkpoint-only metrics

`analysis/spectral.py` reads the predictor's weights straight out of each checkpoint — no
dataset, no model construction, no GPU — and linearises the one-step map. It ran for the first
time on 2026-09-04 over 412 checkpoints, 399 of them non-canary.

### 4.1 Spectral radius ρ(A_aug)

The largest eigenvalue magnitude of the companion-form state matrix for
`s_{k+1} = A_aug s_k + B_aug a_k`.

**Plain words.** Does the learned latent dynamics amplify or shrink whatever you feed it?
Below 1 it contracts; above 1 it expands, and an eight-step rollout multiplies the error.

| | |
|---|---|
| **median** | **17.7** |
| range | [1.05, 145.4] |
| **fraction with ρ ≥ 1** | **1.000 — every single learned predictor is expansive** |

**This corrected a stated reason.** Round 4 dropped a `(r−1)²` spectral penalty on the grounds
that *"PushT is dissipative — r < 1 is the model being right"*. **Not one checkpoint has
r < 1.** The premise was never measured and it is false. The penalty should still not be built,
but now for a measured reason: **Spearman(ρ, CEM) = +0.028** over 309 evaluated runs. It does
not predict planning.

### 4.2 Controllability rank and `log10 cond(W_c)`

`C_H = [B_aug, A_aug B_aug, …]`, `W_c = C_H C_Hᵀ`. `rank/dim` is the fraction of latent
directions any action sequence can reach at all; `cond(W_c)` is how unevenly it reaches them.

| | |
|---|---|
| **median rank/dim** | **0.32** |
| **34 %** of checkpoints | below **0.05** |
| baseline `LpWM-ltv` | 255 / 1152 |
| `LpWM-ltv-d2048`, the best single model | **18 / 6144** |

**The trap, and it cost a correction.** `2026-09-03` §12 called P4 (`PiWM-ctrb`) "a null by
construction — the Gramian was already close to isotropic", reading its *converged*
`ctrb_loss` of 0.033 as a pre-treatment value. It is post-treatment. The penalty drove
`log10 cond(W_c)` from **10.18 to 0.60** — nine and a half orders of magnitude, to full rank
1152/1152 — and planning still did not move (+0.015 [−0.07, +0.10]). **P4 is therefore an
interventional null**, a much stronger result than the diary first recorded: it *rules out*
reachability-conditioning as a direction rather than merely failing to test it
(`2026-09-04` §5.2).

Note also `Spearman(log10 cond, CEM) = +0.473` raw. That looks like a strong signal and, given
every other example in this glossary, is almost certainly another collapse detector. Nothing
proposes to optimise it, so it has not been through the screen — but it must be if anyone ever
does.

---

## 5. Optimisation health: ESS and weight span

**ESS** (effective sample size) `= mean(w)² / mean(w²)`, over the per-sample loss weights `w`
(`conf/train_rdmreg.yaml:222`). **Span** is `max(w)/min(w)`.

**Plain words.** If an objective weights some training samples more than others, ESS says how
many samples the batch is *effectively* worth. Uniform weighting gives 1.0. An ESS of 0.06 means
a batch of 64 is doing the work of four.

| | ESS/N | span |
|---|---|---|
| uniform | **1.000** | 1× |
| `T3 contact` (γ=1.0) | 0.38 | — |
| **V1 `incr_norm` as shipped (ε = 1e-4)** | **0.063** | **3884×** |

V1's `w = 1/(‖Δz‖² + ε)` was designed to stop large autonomous motions dominating the loss.
`ε = 1e-4` sits **400× below** the median increment of 4.1e-2, so the floor never engages: the
weights span 3884×, the single heaviest sample takes 6.1 % of the batch loss, the top 1 % take
29 %, and the term handed the loss to the near-static frames instead. The arm scored
**−0.383 [−0.497, −0.268]** with 8/8 seeds dead.

**The trap is that ESS is seed-dependent and both measurements are kept.** Those figures are
from `LpWM-ltv` **s3** (median increment 0.0415). Calibrating on **s0** gives a median of 0.0771
and ESS **0.220** at the same ε. The median increment differs 1.9× between two seeds of the same
arm, so "the ESS at the shipped ε" is a property of the checkpoint, not of the objective.
Neither number was allowed to overwrite the other; `conf/train_rdmreg.yaml` records both. It
does not change the reading — 0.063 and 0.220 are a 16× and a 4.5× loss of effective batch.

---

## 6. Vocabulary

**JEPA** — Joint-Embedding Predictive Architecture. Predict the *encoding* of the next frame
rather than the frame. No pixels are ever reconstructed. The risk this buys is that the encoder
can cheat by making its own output trivially predictable, which is what every anti-collapse term
below exists to block. In this repo the target is not detached and there is no EMA teacher
(`detach_target: False`), so the two standard collapse guards of the BYOL-style literature are
both absent by configuration.

**The link function `h(·)`** (`models/infojepa_modules.py:790`) — the last operation of the
encoder, which defines the space the predictor and planner work in. Three settings:
`identity` (dense), `relu`, and `reprelu` — ReLU forward, GELU gradient backward, so a zeroed
coordinate keeps receiving gradient instead of dying (`:719`). **The link is why the code is
sparse.** 173 campaign runs used `(reprelu, p=1)` and 35 used `(identity, p=2)`, with *zero* runs
in either off-diagonal cell, so for most of the campaign "sparse vs dense" and "rectified vs not
rectified" were not separable (`2026-09-02` §7).

**RDMReg** (`models/infojepa_modules.py:835`) — the anti-collapse regulariser. It samples a
reference distribution (a generalised Gaussian, `p=2` Gaussian or `p=1` Laplace), pushes it
through the *same* link, and penalises the sliced-Wasserstein distance between the code's
distribution and that reference. "Sliced" means it compares them along many random 1-D
projections rather than in D dimensions.
**Its known weakness:** a dead code costs it only **0.51**, which is payable — that is exactly
how `PiWM-union4` reached ρ = 0.0000.

**SIGReg** (`:26`) — LeJEPA's alternative: an Epps–Pulley test that the embedding is an
**isotropic Gaussian**, with a provable collapse exclusion. Implemented in this repo all along
and never used until round 2. It charges a dead code **25.7** against RDMReg's 0.51, which is
why `reg_weight` cannot be carried between them.

**k-WTA** (`:729`) — k-Winners-Take-All: keep the k largest entries of the code, zero the rest,
with a straight-through backward so losers still get gradient. **Refuted here**, and the
mechanism matters: `Link.forward` rectifies *then* applies k-WTA, and `kwta` marks exactly k
positions **whose values may be zero**, so the realised density is `min(k, #positive)/D`. The
arms never ran at their configured operating point.

**SDR / union coding** — Sparse Distributed Representations, from the Numenta line. In a true
SDR the code is binary and 0.05–2 % sparse over n = 2048–10000 units, and the *union* of several
SDRs (a bitwise OR) is itself a usable sparse code. Both properties were transplanted here onto
a **non-binary, ~50 %-dense substrate**, where they do not hold: at ρ = 0.55 the OR of 4
readouts is 96 % ON and carries almost nothing.

**CEM** — the Cross-Entropy Method, the planner. Sample 300 action sequences from a Gaussian,
roll each through the frozen world model, keep the 30 best by the objective, refit the Gaussian
to those, repeat 30 times. Wrapped in receding-horizon MPC: execute, re-observe, replan, up to
ten times per episode.

**muP** (`models/mup.py`) — Maximal Update Parametrization: a rule for scaling learning rate and
initialisation with layer width so a hyperparameter tuned at one width transfers to another.
The rule is `used_lr = base_lr × base_width / fan_in` for matrix-like weights. **The
input-weight half of the rule was documented at `models/mup.py:12-16` and never implemented at
`:81-83`**, so `proprio_encoder.patch_embed` (fan_in = 4) trained at **96×** `base_lr`,
`action_encoder.patch_embed` at 38×, and `predictor.Ulag` at 24×. Verified real; has never shown
a planning benefit (Δ = −0.136, p = 0.100).

**LTV / `mlp_var` predictors** — the two predictor cores. `ltv` (linear time-varying) is the
substrate for the Step 3/4 work; `mlp_var` is `LpWM-base`. On the pre-fix instrument `mlp_var`
was both better (0.340 vs 0.300) and far more stable (sd 0.072 vs 0.267).

**Consensus voting / plan-time ensemble** — M independently trained models, each rolling the
same candidate action sequences, combined by a rank rule (Borda, median, CVaR) into one
ordering. **No retraining, no new loss term, no new module.** The campaign's only positive.

---

## 7. How to read a contrast

**Always paired, never unpaired.** `analysis/figures.py:344` differences the arm against its
control *seed by seed*, because the same seed carries the same data order and the same
initialisation. Seeds present in only one arm are **dropped, not mean-imputed**. The reason is
in the variance decomposition over four D=384 arms × 11 shared seeds
(`2026-09-02` §2.4): **arm 3.8 % / seed 51.5 % / arm×seed 44.7 %**. Seed instability, not
representation design, owns the variance. An unpaired comparison is measuring the seed.

**Choose the control by its variance, not only by its configuration match.** The first campaign
looked uniformly underpowered until a second baseline finished: `LpWM-ltv` had a dead seed
(0.000, sd 0.267) where `LpWM-base` had sd 0.072. Every contrast anchored on the unstable
control failed to resolve; the same contrasts anchored on the stable one resolved at n = 3.

**And by its configuration match on *every* factor.** Two rows in `2026-09-03` §1 change verdict
otherwise: V3 carries the muP fix and must pair against `-mupfix` (against plain `LpWM-ltv` a
null reads as a −0.170 negative, because the muP fix alone is worth −0.12); and
`LpWM-ltv-ident-p1` carries `reg_weight = 0.1` where `LpWM-ltv` carries 1.0, so pairing them
changes the link *and* the regulariser weight at once — against the wrong control it reads
−0.323, against the right one **−0.006**.

**Intervals are t-based.** A normal approximation badly understates the width at n = 3.

### Why n = 8 is the floor

At the paired sd this campaign actually has (0.133 for the Step 3 contrast), the minimum
detectable effect at 80 % power is:

| n | MDE | resolves a −0.153 effect? |
|---|---|---|
| 3 | 0.437 | no |
| 8 | 0.154 | no |
| 16 | 0.099 | **yes** |

So n = 8 does not guarantee resolution — it is the floor below which a contrast should not be
*read at all*, which is the rule `2026-09-03` §16.8 adopted.

**More seeds is the fix, not more eval episodes.** Training variance dominates eval variance
(sd_train 0.116 against binomial sd_eval 0.065 at `n_evals = 50`), so raising `n_evals`
50 → 200 would move the total sd only 0.133 → 0.121.

### The worked example: T1, and why an interval that excludes zero at n = 3 is not a result

`decode` vs `detach`, the same contrast as its seeds landed
(`2026-09-03` §16.7):

| n | paired difference |
|---|---|
| 2 | −0.240 |
| **3** | **−0.233 [−0.39, −0.08]** ← excluded zero |
| 6 | −0.123 [−0.27, +0.03] |
| **8** | **−0.060 [−0.21, +0.09]** ← final, a null |

It halved twice on the way to a null. **An interval that excludes zero at n = 3 is not evidence
of an effect; it is evidence that three numbers happened to agree.**

**And an uninformative interval is not a null either.** "No significant difference at n = 3" and
"no difference" are different claims, and only one of them is ever earned at n = 3.

---

### What the success rate actually is, and what the instrument can resolve (2026-09-05)

*(The user's instruction was to judge a method from the success rates and a deep analysis of
them, not from an arm-or-gate verdict. Doing that produced four findings that change how every
earlier number should be read — and two retractions of my own first attempt at it.)*

#### 1. `SR = 0` is a METRIC floor, not a model floor

`env/pusht/pusht_wrapper.py:eval_state` defines success as
`pos_diff = ‖goal[:4] − cur[:4]‖ < 20 px` — a **joint** norm over the agent position *and* the
block position. So the historical metric requires the gripper to be **parked** near its goal as
well as the block placed. The file says so itself: *"success ⇒ success_block … the historical
metric is the STRICTER one."*

Re-scoring the archived traces on the **task alone** (`block_pos_diff < 20 px`, `|angle| < π/9`,
dropping the end-effector term):

| arm | official SR | block-only SR | "parking tax" |
|---|---|---|---|
| `PiWM-consist-w0p3` | **0.005** | **0.255** | 0.250 |
| `PiWM-support-w0p3` | **0.003** | **0.260** | 0.258 |
| `PiWM-incr-eps0p041-clip10` | **0.000** | **0.217** | 0.217 |
| `PiWM-contact-shuf` | 0.048 | 0.287 | 0.240 |
| `LpWM-ltv` (baseline) | 0.357 | 0.580 | 0.223 |
| `PiWM-columns` | 0.418 | 0.630 | 0.212 |
| `PiWM-vote5-borda` | 0.608 | 0.665 | **0.057** |

**Arms that score ~0.00 place the block correctly 22–26 % of the time.** They are not dead
models; they are models that solve the task and fail to park the gripper. Note also that the
baseline pays a 0.223 tax and the consensus arm pays **0.057** — which is a lead worth chasing,
though it cannot be settled here because only 2 baseline runs have traces on disk.

The ordering is largely preserved (Spearman(official, block-only) ≈ 0.96 across arms), so past
*rankings* stand. What does not stand is any statement of the form "this arm cannot plan at all".

> ### ⚠ RETRACTED the same day — the "metric floor" was an artefact of dropping the term that
> ### detects divergence
>
> Block-only re-scoring drops `agent_pos_diff` from the success test. That term is the only one
> catching a **diverging simulation**, and the floor arms are diverging:
>
> | arm | median agent error | > 1000 px | block-only SR | **block-only AND agent < 100 px** |
> |---|---|---|---|---|
> | `LpWM-ltv` | 17.1 px | 11 % | 0.580 | **0.560** |
> | `PiWM-vote5-borda` | 13.3 px | 6 % | 0.665 | **0.645** |
> | `PiWM-patchdecode` | 16.7 px | 7 % | 0.615 | **0.603** |
> | `PiWM-consist-w0p3` | **2689.6 px** | **88 %** | 0.255 | **0.025** |
> | `PiWM-support-w0p3` | **1019.3 px** | 51 % | 0.260 | **0.003** |
> | `PiWM-contact-shuf` | **1290.9 px** | 63 % | 0.287 | **0.060** |
>
> Requiring the gripper merely to be *somewhere sane* — within 100 px, five times the success
> radius — collapses the floor arms from 0.22–0.29 back to **0.003–0.06**, while barely touching
> the healthy arms. A block that lands in tolerance while the end effector is 2700 px away is a
> diverged rollout, not a solved task.
>
> **So the floor arms genuinely fail, the original verdicts were right, and the parking tax is
> ≈ 0.02 for healthy arms rather than 0.22.** I verified the re-scored numbers but not what they
> meant, which is the same error in a new place.
>
> **What survives, and it is worth keeping:** the campaign has no divergence detector, and
> `median agent_pos_diff` is a good one — 13–28 px for every healthy arm against 1000–2700 px for
> every failing one, with no overlap. It is free, it comes from traces already on disk, and
> unlike `rel_mse` it cannot be fooled by a constant-output model.

**Amended 2026-09-05.** The "no overlap" in the line above holds for the eight arms tabulated here,
not for the archive. Over all 49 arms with traces the ranges overlap (worst healthy arm 1138 px,
best failing arm 122 px). At a 100 px threshold the detector is **94.5 % accurate (273/289 runs)**
with 14 healthy runs above the line — good enough to flag on, not to gate on. See
[2026-09-05 §7.2](2026-09-05.md).



#### 2. The instrument is coarser than the effects it was applied to

| contrast type | n | median sd of the paired difference | **MDE at n = 8, 80 % power** |
|---|---|---|---|
| **retrained arms** | 33 | **0.147** | **0.150** |
| same checkpoints, planner/eval varied | 13 | 0.084 | 0.086 |

Against that: the between-arm sd of the whole healthy-arm population is **0.099**
(noise-corrected). **The minimum effect a retrained-arm contrast at n = 8 can detect is larger
than the entire spread of the population it is measuring.**

So **every "null" verdict issued for a retrained arm is uninformative below ≈ ±0.13.** Only the
large negatives (−0.26 to −0.39) and the consensus positive were ever visible to this design.
The n = 8 floor was necessary and is not sufficient.

#### 3. The dominant axis of variation is the training seed, and pairing does not remove it

The `PiWM-soloM` matrix gives a fully crossed 5 × 5 **model × episode-block** design — identical
configs differing only in the training seed:

| family | SS(model) | SS(block) | SS(residual) | residual sd | binomial sd |
|---|---|---|---|---|---|
| A | **92 %** | 1 % | 7 % | 0.048 | 0.065 |
| B | **63 %** | 18 % | 19 % | 0.057 | 0.067 |

Model means in family A: **0.180, 0.268, 0.380, 0.412, 0.600** — a **3.3× range from one config
and five seeds**. Training-seed sd is **0.144** against a binomial sd of 0.043, i.e. ~2.4–3×
the episode noise. And the seed effect is an **arm × seed interaction**, not common-mode: mean
pairwise correlation of demeaned seed profiles across arms is only 0.150, so **pairing buys about
a 7 % variance reduction, not cancellation**.

#### 4. The pre-registered kill gate has a blind spot

`rel_mse ≥ 0.5` cannot see total representational collapse, because a constant-output model has
`rel_mse ≈ 0`. **`sparsity/effective_dim == 0` is the only perfect predictor of SR ≈ 0 found** —
24 runs across 12 arms, every evaluated one at SR ≤ 0.04. Check both.

#### Retracted from my own first attempt at this reanalysis

* **A death/quality decomposition of Δ mean.** Not identified: the interaction `Δp·Δq` is
  **48–84 %** of the effect, and every contrast flips label under the mirror ordering, under
  Shapley, and under the log form. "Death-dominated" turned out to be a re-encoding of `P_t`
  (Spearman −0.817).
* **Reporting `Δ P(SR > 0)` with McNemar as though it separated training from planning.** Three
  problems. The antimode of the distribution is at **SR ≈ 0.15, not 0**, so that cut sits inside
  the left mode. Five of the six arms it flagged show **no within-arm heterogeneity at all**
  (bootstrap LRT p = 0.08–0.63) — they are uniformly at the floor, not a mixture of live and dead
  seeds. And a training-side definition of "dead" (`rel_mse ≥ 0.5`) gives **ΔP = 0.00** for four
  of them. `consist-w0p3`'s eight seeds sit inside the baseline's range on `rel_mse`,
  `effective_dim` *and* `val/loss` while scoring 0.005 — that is a planning result, which is what
  the original verdict said.
* **Defending the conditional mean with Spearman(P, SR|work) = +0.814.** No power: a simulated
  world with **no death mechanism at all** gives +0.781 [+0.729, +0.823].

## 8. The standing trap## 8. The standing trap: nearly every metric here is a collapse detector

This is the single most useful thing to carry into the entries.

A quantity that separates dead models from live ones will correlate with CEM at r = 0.5–0.8,
because **dead models plan at exactly zero** and there are a lot of them (23 % of all evaluated
runs). That correlation says nothing about whether moving the quantity among *healthy* models
does anything. `analysis/screen_objective.py` is the four-item test that separates the two:

1. the raw Spearman — **the number that fools you**;
2. the rank-partial with ρ and `rel_mse` removed, against a permutation null — **the number that
   matters**;
3. the same restricted to predictors that actually predict (`rel_mse < 0.05`) — a *cut*,
   reported as a robustness check and never as the headline;
4. binned quartile means — because a partial correlation cannot see a non-monotone relation,
   which is exactly what `d_action` turned out to be.

`rel_mse` is included in every screen as a **self-null control**: it is one of the covariates, so
it must screen to zero or the estimator is broken. It screens to +0.003.

| candidate | raw | partial | monotone | verdict |
|---|---|---|---|---|
| **`jacc/S_model`** | −0.769 | **−0.549** | **yes** | the one endorsed target |
| `sparsity/effective_dim` | +0.487 | +0.338 | no | interior optimum |
| `sparsity/effective_dim_pred` | +0.609 | +0.331 | no | interior optimum |
| `jacc/S_world` | −0.035 | +0.315 | no | interior optimum |
| `jacc/churn_model` | −0.375 | −0.056 | — | collapse detector |
| `h8/h1` | +0.558 | **−0.017** | — | collapse detector |
| `h1` | −0.552 | −0.090 | — | collapse detector |
| `err/rel_mse` *(self-null control)* | −0.684 | **+0.003** | — | screens to zero, as it must |

Non-monotone is not a mild caveat: it means **"raise it" is not a direction**. That is the
`d_action` shape, and mistaking it for a direction is what nine arms across rounds 3 and 4 were
spent on.

The pattern is consistent enough to name: **nearly every metric that is a property of the model
in isolation turns out to be a collapse detector.** `rel_mse`, ρ, `effective_dim`, `d_action`,
`h8/h1`, `h1`, support churn — all of them separate dead from live and then go flat.
`S_model` is the first exception found, and it is an exception about *structure* (which units
fire) rather than *magnitude*.

Two corollaries, both learned the expensive way:

* **Measure the baseline's value of a quantity before building arms to change it**
  (`2026-09-03` §14 rule 4). One unverified number, propagated.
* **No fitted thresholds, ever** (rule 5). A boundary fitted to the same data it is tested on
  will give you a Fisher p of 4.7 × 10⁻⁸ and mean nothing. Rank-residualise and report the
  partial with its permutation null.

---

## 9. Figures

Every figure in every entry imports `analysis/style.py`, the palette derived from
`figures/motivation_teaser.svg`: **green** the system, **purple** the contrasting condition,
**amber** the intervention under test, **crimson** failure or retraction, **slate** neutral.
Hues are assigned by **identity, never by rank**, so regenerating a figure after more evals land
does not repaint the arms. Sans-serif throughout; titles state the finding, not the variable
name.

Regenerate:

```
python analysis/collect_evals.py --out campaign.json
PYTHONPATH=. python analysis/early_figs.py                      # 2026-09-01, 2026-09-02 PNGs
python analysis/arch_figs.py       --out diary/assets/2026-09-02 # arch-00..09 SVGs
python analysis/causal_figs.py     --out diary/assets/2026-09-03
python analysis/round5_data_figs.py --out diary/assets/2026-09-03
python analysis/round6_figs.py     --out diary/assets/2026-09-04
python analysis/spectral.py --out assets/spectral.json --campaign campaign.json
```
