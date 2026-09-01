# What every metric and figure means

Two audiences, one vocabulary. **Metrics** (`train.py` → wandb) tell you whether a
run is healthy *right now*. **Figures** (`analysis/figures.py` → PNG,
`analysis/report.py` → HTML) tell you whether the *campaign* answered its question.
They compute the same quantities on purpose: `soft_jaccard` in `train.py:120` mirrors
the numpy version in `analysis/predictive_jaccard.py`, so the live curve and the
offline gate cannot disagree.

## The one thing to understand first

The campaign tests a chain of claims about a **sparse** latent code `z`:

1. the code's *support* (which units are non-zero) carries predictive structure
2. making it sparser (k-WTA) does not destroy that
3. gating the predictor on *support* beats gating on *magnitude*
4. a union of J readouts beats one

Almost every metric exists to separate **"the support changed"** from **"the
magnitudes changed"**. That distinction is the project. A statistic that moves for
both is nearly useless here, which is why so many come in pairs.

---

# Part 1 — Metrics

## `jacc/` — the core statistic

`S = 1 - J_S(a, b)` where `J_S = Σ min(a,b) / Σ max(a,b)` (`train.py:120`). Zero means
identical, one means disjoint.

| key | meaning | how to read it |
|---|---|---|
| `jacc/S_model` | `1 - J_S(ẑ, z_target)` — how wrong the prediction is | the headline. Falling = the predictor is learning the code |
| `jacc/S_world` | `1 - J_S(z_t, z_{t+1})` — how much the world changed on its own | the **baseline S_model must beat**. If `S_model ≈ S_world`, predicting nothing does as well |
| `jacc/S_model_p90` | 90th percentile over samples | if p90 ≫ mean, error is bursty, not uniform |
| `jacc/burst_rate` | fraction with `S_model > burst_tau` (0.5) | the burst frequency. Should track contact events, not drift |
| `jacc/churn_model` | fraction of units whose **binary** support flips | the disambiguator |
| `jacc/support_churn` | same, for the world | ditto, baseline |

**The key read:** `S_model` high **with** `churn_model` high → the support genuinely
reorganised. `S_model` high **with** `churn_model` ≈ 0 → the support is right and only
the magnitudes are wrong. Those are different failures and want different fixes.

## `sparsity/` — is it sparse, and sparse in the right way?

The trap this namespace exists to defuse: **`l0_frac` alone is ambiguous.** Density
0.5 can mean every unit fires half the time, or half the units always fire and the
rest are dead. Only the first is compatible with the claim.

| key | meaning | how to read it |
|---|---|---|
| `sparsity/l0_frac_pred` | density of the *predicted* code | should track the encoder's. Divergence = predictor not honouring the link |
| `sparsity/effective_dim` | participation ratio `tr(C)²/‖C‖_F²` of the code covariance | **Compare it to the TASK, not to D.** PushT's own state has participation ratio **4.31** (7 raw dims, 99% of variance in 6), so an `effective_dim` of 15-18 is ~4x over-complete and healthy. The failure threshold is falling *below* the task dimension — which is where the collapsed union arms sit (3.7-5.5) |
| `sparsity/effective_dim_pred` | same for `ẑ` | a predictor can collapse while the encoder doesn't |
| `sparsity/dead_unit_frac` | units with zero activations **over a window** | windowed on purpose — at ρ≈0.5 no unit is dead within one 64-sample batch, so a per-batch answer is always 0 |
| `sparsity/dead_unit_count` | same, absolute | |
| `sparsity/unit_freq_max/min` | most / least used unit | max→1 and min→0 is the degenerate code |
| `sparsity/l0_std_across_samples` | spread of per-sample L0 | a correct *mean* with wide spread = dense on some samples, empty on others. Not "k active units" |
| `sparsity/unit_window_samples` | window size behind the above | bookkeeping; tells you the dead-unit estimate's n |

**Healthy:** density on target, `effective_dim` a decent fraction of D, `dead_unit_frac`
low, `l0_std` small. **Degenerate:** on-target density, `effective_dim` near 1.

## `reg/` — the RDMReg regulariser

RDMReg pulls the code's distribution toward a target via sliced Wasserstein. The
namespace exists because **`reg_loss` alone cannot tell useful work from a fight**.

| key | meaning | how to read it |
|---|---|---|
| `reg/target_density` | analytic density the regulariser is pulling toward | fixed per config |
| `reg/density_gap` | `measured l0_frac − target_density` | **the conflict detector.** k-WTA pins density at k/D; if that disagrees with the target, the W2 term can never reach zero. A persistent non-zero gap means any degradation is *objective conflict*, not sparsity |
| `reg/swd_probe` | independent re-estimate of the SWD statistic on 256 fresh projections | a sanity check on the training term. Should track `train_reg_loss` |
| `reg/swd_proj_max` | worst single projection | **`swd_proj_max ≫ swd_probe` = the mismatch lives in a few directions**, not spread over all of them. That is a structured, fixable failure |

## `gate/` — the LTV support gate (Step 3, `mode='ltv'` only)

| key | meaning | how to read it |
|---|---|---|
| `gate/gate_mean` | mean gate magnitude | **the `r·softmax` check.** ≈0.5 for sigmoid, ≈1.0 for `r·softmax`. Reading ≈1/r (0.0625 at r=16) means the `r` factor was lost — an 8× shrink of the gated gradient path that at 2 epochs is indistinguishable from "support gating is worse" |
| `gate/gate_std`, `gate/gate_max` | spread and peak | sharpening looks like high max with low mean |
| `gate/gate_frac_gt_half` | fraction of gates open | mode-selection sharpness |
| `gate/ltv_u_norm` | ‖U‖ of the LTV correction factors | **the engagement precondition.** U is zero-initialised, so a norm that never leaves ≈0 means the mechanism never turned on and the arm *cannot falsify anything* |
| `gate/ltv_u_rel` | `‖U‖ / ‖base lags‖` | scale-free version. Read this, not the raw norm |

## `heads/` — the union head (Step 4, `n_heads > 1` only)

| key | meaning | how to read it |
|---|---|---|
| `heads/head_gap_head0` | `mean(loss[head 0] − min over heads)` | **the one that matters.** How much the min over J heads buys over head 0 alone. **Sitting at zero means J=4 is numerically J=1** — the extra heads are decoration. This is the rigged-gate failure the plan's precondition exists to catch |
| `heads/head_loss_spread` | `max − min` over heads | heads that all cost the same are not specialising |
| `heads/head_usage_min` | least-used head's share of `j*` wins | near 0 = a dead head |
| `heads/head_loss_j{k}` | per-head loss | which head is carrying |
| `heads/head_delta_j{k}` | RMS distance each head moves the code from its input | **usage can look healthy while all heads do the same thing.** This catches that: equal deltas = no specialisation, whatever the histogram says |

## `err/` — prediction error, decomposed three ways

One MSE cannot diagnose. Each split changes the conclusion.

| key | meaning | how to read it |
|---|---|---|
| `err/rel_mse` | MSE ÷ target power | **scale-free** — the only one comparable across arms whose codes differ in magnitude |
| `err/mse_on_support` | error where target ≠ 0 | getting the *active* units wrong |
| `err/mse_off_support` | error where target = 0 | **hallucinating into inactive units.** A sparse model's characteristic failure |
| `err/cos_pred_target` | cosine similarity | direction vs magnitude: high cosine with high MSE = right shape, wrong scale |
| `err/mse_t{i}` | error at timestep i | rising with i = the predictor only works near the start of the window |

## `opt/` — is anything actually training?

| key | meaning | how to read it |
|---|---|---|
| `opt/grad_norm_{encoder,predictor,action_encoder}` | per-module grad norm | one aggregate norm cannot tell "encoder not learning" from "predictor not learning" — the first question when an arm flatlines |
| `opt/weight_norm_predictor` | predictor weight norm | |
| `opt/update_to_weight` | `lr · ‖g‖ / ‖w‖` | **the scale-free step-size check. ~1e-3 healthy, 1e-6 frozen, 1e-1 diverging** |
| `opt/lr` | current LR | confirms the schedule; muP gives different LRs per fan-in |

## `perf/` and `progress/` — infrastructure

| key | meaning | how to read it |
|---|---|---|
| `perf/batches_per_sec` | throughput | dataloader-bound here, so **dips = a neighbour landed on the node**, not the science |
| `perf/data_wait_frac` | fraction of time waiting on data | high = starved. Confirms the above |
| `perf/eta_hours`, `perf/hours_per_epoch` | projections | **compare against the 3h55m window.** Over it → the epoch never completes in one window |
| `perf/gpu_mem_{alloc,reserved,peak}_gb` | memory, peak reset per interval | |
| `progress/epoch_frac`, `global_batch`, `window_seconds` | where the run is | `window_seconds` resets each preemption — the resume marker |

## `dist/` — histograms a mean cannot show

Logged rarely (host copies are expensive). Each exists because its scalar summary is
ambiguous.

`z_nonzero_magnitude`, `z_l0_per_sample` (**bimodal = two populations, not one sparse
code**), `unit_activation_freq`, `unit_mean_activation` (separates "rarely on but
large" from "always on but small"), `err_per_feature`, `err_per_patch`,
`S_model_per_sample`, `support_churn_per_sample`, `gate_values`, `head_loss`,
`swd_per_projection`, `z_p50/p90/p99/max`.

## `fig/` — live raw maps (tier 3, heavy interval)

`support_selfsim`, `code_vs_pred`, `unit_coactivation`, `gate_heatmap`, `gate_lag_map`,
`head_assignment_raster`. Raw tensors pushed through a colormap **LUT** — no matplotlib
figure, so they cost nothing on a dataloader-bound node. Same colour system as the
offline PNGs: sequential for magnitudes, **diverging centred at zero** for signed `z`,
**categorical head colours** for `j*` (a head index is a class, not a magnitude).

## `panel/` — live rendered figures (tier 3, heavy interval)

Real `analysis/panels.py` figures, rendered live and logged as `wandb.Image`. These
are the same forms as the offline suite, so a live page and its PNG read identically.

| key | what it shows |
|---|---|
| `panel/z_magnitude_ridgeline` | the distribution of surviving \|z_i\| **evolving over training**, one ridge per heavy tick. **A second mode growing near zero = the code is sparsifying, not merely shrinking** — the distinction a mean cannot make |
| `panel/l0_ecdf` | per-sample L0 as an ECDF. No bins, so a shift, a spread change, and a **hard cap (k-WTA pinning L0 at exactly k)** are all directly legible |
| `panel/head_usage_stream` | union-head usage stacked over training (J>1 only). **One colour swallowing the band = collapse**; the dashed rule is the 0.9 precondition |

Rendered on the **heavy interval** (every 2,500 batches, ~14 min), so the cost is a
handful of matplotlib renders against ~30k training steps. The ridgeline keeps **pre-binned
histogram rows**, not raw samples, so showing the evolution costs a few KB of state.

`diag/blocks_failed` — count of diagnostic blocks that raised. **Non-zero means you
are reading an incomplete picture.** Check it before trusting an absence.

## `video/` — live simulation rendering

| key | what it shows |
|---|---|
| `video/env_rollout` | a **real PushT rollout**, rendered. Held-out trajectories are replayed in the simulator from their own recorded start state, tiled side by side. Every 5,000 batches (~20 min), 4 episodes x 50 frames |

Training is otherwise entirely latent — `has_decoder: False`, so no pixels appear
anywhere in the loss. This is the only place the simulator itself is visible, and it
is paired with `panel/latent_error_timeline` computed on the **same frames**, so a
spot where the code churns can be looked at rather than inferred.

Actions are **denormalised** before replay: the dataset stores normalised actions and
the env expects raw ones, so replaying them directly would render a plausible-looking
video of the wrong trajectory.

---

# Part 2 — Figures

Six classes; the *silhouette* tells you which kind of question a figure answers.

## A · Inference — "is the effect real?"

| figure | question | how to read it |
|---|---|---|
| `31_effect_sizes` | every contrast, with its seeds | **the flagship.** Top: each seed as a dot, controls as square/diamond. Bottom: paired bootstrap difference. **Hollow dot = inside the detection floor → underpowered, not null** |
| `05_gate_scorecard` | did each pre-registered gate pass? | interval vs the black threshold bar. Verdict is a glyph + word, never a fill |
| `34_power_curve` | *could* we have detected it? | power vs true effect at the observed seed sd. Read **before** any gate: an effect below the MDE was never detectable. At n=3 a sign-flip test cannot return p < 0.25 at all |
| `00_campaign_overview` | every arm of a gate side by side | thin lines are one seed carried across arms. **Consistent tilt = effect; crossing lines = seed spread swamps it** |
| `03_paired_dumbbell` | before → after per seed | control end neutral, variant end coloured |
| `22_seed_variance` | can we resolve arms at all? | seed-noise trajectory + the **ICC**. ICC near 0 = the campaign cannot separate the arms whatever the means say |
| `30_ladder` | the paper's Fig 1b shape | sparse vs dense across D |
| `11_success_vs_k` | does sparsity itself carry the effect? | success and the regulariser floor vs k/D on **two stacked axes**. Deliberately not a twin y-axis: two y-scales have an arbitrary alignment and would invent a crossing point. If the floor drops where success does not, mu-matching removed the conflict and sparsity is not what moved the result |

## B · Trajectory — "what happened over training?"

Faceted when >6 arms (colour cannot separate 9 series — see `panels.SERIES_LADDER`).

| figure | question | how to read it |
|---|---|---|
| `16_sparsity_trajectories` | does each arm hold its density? | **a k-WTA arm drifting off k/D is not running its own intervention** |
| `13_training_curves` | is loss descending, and are resumes lossy? | **a step at a resume marker = the resume lost state** |
| `18_loss_decomposition` | which term does each intervention move? | the trade is hidden in the total |
| `19_gradient_health` | exploding / vanishing / wrong schedule? | |
| `14_training_health` | loss and epoch cadence per arm | ops |
| `10_engagement` | **did the mechanism turn on at all?** | flat `ltv_u_norm` or entropy pinned at 0 = the arm is a no-op and cannot falsify anything. **Check this before reading any Step 3/4 gate** |
| `17_rdmreg_vs_l0` | is there an irreducible regulariser floor? | a **phase plane**: density against `reg_loss`, walked over training, one facet per arm with the rest ghosted. Hue is the arm, **lightness is time**. A floor is where the trace *stops moving* — an arm that stalls at high loss with density pinned is paying it, and its degradation is objective conflict rather than sparsity |

## C · Distribution — "what's the shape?"

| figure | question | how to read it |
|---|---|---|
| `32_gate_values` | did `r·softmax` restore scale? | ridgeline + rules at 0.5 and 1.0. **A mean can sit on target with no mass there** |
| `25_l0_distribution` | tight around target, or two populations? | bimodal = not a sparse code |
| `26_code_geometry` | using its width, or a few units? | rank-frequency (Zipf line = degenerate), participation ratio, frequency vs magnitude |
| `29_per_head_dynamics` | do heads own different regimes? | dot + CI per head. **Overlapping intervals = the heads partition nothing** |

## D · Structure — "what does the code look like?"

| figure | question | how to read it |
|---|---|---|
| `06_jaccard_decomposition` | support reorganised or magnitudes wrong? | **the central disambiguation.** Off-diagonal = support moved |
| `07_support_selfsim` | is the code discrete? | **block-diagonal structure = discrete states** |
| `08_head_raster` | do heads own regions of state space? | `j*` over time and painted on the (agent, block) plane |
| `12_burst_vs_error` | are bursts real or cosmetic? | bursts should coincide with error |
| `23_head_specialisation` | do the J readouts divide the work? | |
| `28_head_onset_alignment` | do heads align with contact? | `P(j*|lag)`. **A flat map is a real null result** — heads partition something, not contact |
| `33_gate_heatmap` | sharpening or just rescaling? | shared colour scale. **Sharpening = a few bright persistent columns; rescaling = same texture, different brightness** |
| `15_metric_coverage` | which panel can render for which run? | grey = never logged |

## E · Dynamics — event-aligned

| figure | question | how to read it |
|---|---|---|
| `01_peri_event` | **Step 1's money figure.** When does each statistic fire relative to contact? | `S_model` should **lead**, `S_world` should lag. That ordering is the claim |
| `27_onset_lead_lag` | does the lead beat chance? | vs a shuffled null |
| `02_roc_overlay` | the Step 1 gate, rendered | AUROC(`S_model`) > AUROC(`S_world`) |
| `04_head_usage` | Step 4's precondition | **one band filling the plot = heads collapsed**, and the rest of Step 4 is void |
| `24_head_switch_burst` | is switching event-driven or churn? | |
| `09_scale_perturbation` | the Step 3 property test | **a support gate is invariant to support-preserving rescale; a magnitude gate is not** |

## F · Ops

`20_throughput` — every logged window as a dot; a **left tail is co-tenancy**, a shifted
bar is the run itself. `21_preemption_timeline` — Gantt of wall clock actually spent
training, against the 3h55m boundaries.

`99_contact_sheet` — the nine load-bearing panels on one page, re-read from the written
PNGs so it can never disagree with them.

---

# Arm names

`LpWM-*` are the **baselines** (flags off); `PiWM-*` are the interventions.
Colour follows this: baselines are neutral grey, and the three PiWM families take
blue (sparse codes), magenta (support gating) and green (union head).

| arm | what it is |
|---|---|
| `LpWM-base` | baseline, MLP predictor, all flags off |
| `LpWM-ltv` | baseline, LTV predictor — shared control of the gating and union gates |
| `PiWM-sparse-matched` | k-WTA at density-matched *k* |
| `PiWM-sparse-2pct` | k-WTA at 2% density |
| `PiWM-gate-sup-sigmoid` | support gate, sigmoid normalisation |
| `PiWM-gate-mag-softmax` | magnitude gate, softmax — the ablation isolating normalisation |
| `PiWM-gate-sup-softmax` | support gate, softmax — **the Step 3 proposal** |
| `PiWM-union4` | J=4 union head, no entropy bonus |
| `PiWM-union4-entropy` | J=4 union head + entropy bonus — **the Step 4 proposal** |
| `LeWM-ltv` | the **dense** counterpart (identity link, Gaussian target) — a reference arm, drawn neutral |
| `PiWM-union4-kwta8` | union head **at k-WTA sparsity** — Step 2 x Step 4 composition |

---

# Reading order

1. **`10_engagement`** — did the mechanism turn on? If not, stop; nothing downstream means anything.
2. **`34_power_curve`** — what was detectable? Establishes whether a null is informative.
3. **`31_effect_sizes`** — what happened, with the seeds visible.
4. **`05_gate_scorecard`** — the verdicts.
5. Then the mechanism panels for whichever gate moved.

`diag/blocks_failed > 0` or a grey column in `15_metric_coverage` means a panel is
absent rather than negative. **The suite never fabricates a missing panel** — it prints
what would unblock it.
