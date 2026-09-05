# Measurement protocol for action-conditioned world models

**Status:** binding for all future rounds. Every rule here is the residue of a specific failure in
rounds 1–7; the failure is named beside each rule so the rule can be argued with rather than
obeyed. Numbers recomputed from the archive on 2026-09-05.

The short version: **one metric decides, four gates guard it, six popular metrics are disqualified
as targets, and nothing becomes a target without passing a four-stage screen that includes an
intervention.**

---

## 1. The primary metric — the only one that decides anything

**CEM planning success rate**, fixed evaluation scheme, 50 episodes per block, `goal_H = 5`,
`max_iter = 10`, 300 samples, 30 optimisation steps.

| requirement | value | the failure it exists for |
|---|---|---|
| **paired on SHARED seeds** | intersection of the two arms' seed sets | the baseline's registered mean is over seeds 3–15; almost every treated arm holds 3–10. On the shared eight the baseline is **0.3925**, not its registered **0.3569** — a free **+0.036** to every 8-seed arm. This alone moved 35 apparent winners to 16. |
| **minimum n** | 8 | one arm read −0.233 at n = 3 and settled at a null by n = 8 |
| **n for effects below +0.10** | **20** | `patchdecode` at +0.085 needs n = 20; at n = 8 it is unreadable |
| **per-seed values reported** | always | four "wins" in this campaign were one seed |
| **control named in the spec** | before launch | every uncontrolled claim here became a seed-set artefact or an unchanged re-draw |

### 1.1 The margin bar

From the `PiWM-solo` matrix — one **unchanged** checkpoint replayed across episode blocks, 40 cells,
10 checkpoints × 4 blocks:

| source of variance | share | sd |
|---|---|---|
| **training seed (checkpoint)** | **81.5 %** | 0.128 |
| episode block | 16.0 % | 0.030 |

A single unchanged model spans **0.04–0.24** across blocks; ten unchanged checkpoints span
**0.165–0.585**. So:

* an 8-seed arm needs roughly **+0.09** to be a 1-in-10 draw, **+0.10 – 0.115** for 1-in-20;
* **the unit of replication is the checkpoint, not the episode.** Adding evaluation episodes to a
  fixed checkpoint buys almost nothing; adding seeds buys everything.

### 1.2 Cluster structure must be declared

If an arm's *n* is not *n* independent models, say so and report at the level that is independent.
`PiWM-vote5-median`'s ten "seeds" are **two** committees; `vote3-median`'s twelve are **four**. At
the committee level the first arm's interval widens ~5× and the second's tightens — which reverses
which arm is better supported.

---

## 2. Mandatory health gates — measurement, never targets

Run on **every** arm before any contrast is read. Two of the three are not optional because the
pre-registered gate catches neither failure the campaign actually met.

| gate | threshold | notes |
|---|---|---|
| **collapse** | `sparsity/effective_dim == 0` | the only perfect predictor of SR ≈ 0. A constant-output model has `rel_mse ≈ 0` and scores *healthy* on the pre-registered gate. |
| **divergence** | median `agent_pos_diff > 100 px` | **94.5 % accurate (273/289 runs)** — a **flag, not a gate**. 14 healthy runs sit above the line, so gating on it discards ~a tenth of the healthy population. |
| fit | `rel_mse ≥ 0.5` | the pre-registered death condition. Kept, but it catches **neither** of the above: 139 flagged runs pass it. |

Worked example of why this is mandatory: `LpWM-ltv-p1` scored 0.02 with `rel_mse` 0.025–0.028 —
comfortably healthy on the pre-registered gate — while **4 of 4 seeds diverged** at 1400–1700 px.
Without the divergence flag that arm reads as "patch tokens are worse", which is a false conclusion
about the feature.

---

## 3. Disqualified as targets

Six metrics that correlate with success **across** arms and stop correlating **among healthy
arms**. Each is a *collapse detector*: it separates dead models from live ones, then goes flat or
inverts. None may be used as an optimisation target or a design justification.

| metric | pooled | among healthy / demeaned | how it was killed |
|---|---|---|---|
| `causal/d_action` | ρ = +0.624, n = 45 | **+0.316, p = 0.142** (n = 23) | **interventionally**: `jump2` reaches **1.8×** baseline action sensitivity and gains **nothing** (−0.035) |
| `h8/h1` rollout growth | ρ = +0.558 | **−0.001** demeaned within arm | closes any rollout-stability objective without a horizon sweep |
| probe decodability (position, angle) | ρ = −0.573 over 35 arms | **+0.45 / +0.51 — inverted** among the 16 arms that encode anything | the five best orientation decoders are five of the worst planners (−0.297 … −0.372) |
| `sparsity/*` (all measures) | weak | dead at every threshold | per-seed values are bimodal; arm means are mixtures of a draw |
| `effective_dim` | ρ = +0.560 | death detector with an interior optimum | `== 0` is diagnostic; "more" is not a direction |
| `log10 cond(W_c)` / controllability | ρ = +0.473 | 31 of 56 arms broken | never separated from collapse |

**The standing trap.** 31 of 56 trained arms in this archive are broken. Any metric computed over
all arms will correlate with success. That correlation is not evidence that the metric is a target.

---

## 4. The one diagnostic that survives

**One-step latent prediction accuracy** — `val/z_loss`, `err/mse_t1`, `err/rel_mse`. Demeaned
**within** arm, over 174 runs in the 19 non-degenerate arms: **ρ = −0.680**.

It is the only axis of any kind that survives arm-demeaning, and it is consistent with the one
mechanism the campaign found: `patchdecode`'s gain is specifically one-step (−20.6 % visual-latent
error at h1, −14 % on `mse_t1`) with `h8/h1` unchanged.

**Stated as a limitation, not a result:** that it *predicts* success within arm does not establish
that *optimising* it improves success. That is exactly the inference that failed for `d_action`,
and it has not yet been tested here. Any proposal resting on it must include the intervention.

---

## 5. The screen — required before any metric becomes a target

Four stages. A candidate that fails any stage is disqualified. Stages 1–3 cost no GPU.

1. **Pooled association.** Spearman against success across arms. *Necessary, worthless alone.*
2. **Healthy subset.** Restrict to arms with mean success ≥ 0.20. Does it survive?
   `d_action` dies here (p = 0.142).
3. **Arm-demeaning.** Remove the arm mean; correlate within arm across seeds. Does it survive?
   `h8/h1` dies here (−0.001); probe decodability **inverts** here.
4. **Intervention.** Build an arm that moves the metric and nothing else. Does success follow?
   *This is the stage that actually settles it, and the only one that costs GPU.*

Stage 4 is the contribution. Stages 1–3 are observational and every one of the six disqualified
metrics passed stage 1. `d_action` was only settled when an arm was built that raised it 1.8× and
planning did not move.

---

## 6. Reporting requirements

Every reported contrast carries, without exception:

- the **control**, named, and matched on seeds *and* on the axis being claimed;
- **n**, and the **seed sets** of both arms;
- **per-seed values**;
- the **health gates** for every run in both arms;
- the **cluster structure** if *n* is not *n* independent models;
- for any arm whose baseline is not the global one, **which** baseline — `patchdecode` is +0.085
  against its dimension-matched control `PiWM-columns` and +0.128 against `LpWM-ltv`, and the two
  are different claims.

### 6.1 Dimension matching

A representation contrast must hold **total latent** fixed: `tokens × proj_dim`. The campaign's
patch-vs-cls result compared 98,304 dims against 384 and could not separate feature from capacity.

**And matching dimensions is not sufficient.** Varying token count at fixed width means varying
`patch_size`, which is simultaneously the receptive field of
`nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)`. Coarse patches (224/112/56) break
the encoder outright while `patch_size = 14` is healthy — so a dimension-matched grid built that way
trades a capacity confound for a granularity one. State which confound remains; there may be no
setting that removes all of them, since `tokens × width` is the quantity being held fixed.

---

## 7. What this protocol costs

Round 6 ran 14 contrasts at n = 8 and could resolve almost none of them. Recomputed over the 39
healthy retrained arms with n ≥ 8 (median paired sd **0.169**), at 80 % power and α = .05:

| | n = 8 | n = 20 |
|---|---|---|
| MDE, two-sided normal approximation | **0.167** | **0.106** |
| MDE, the repo's paired-t simulator (re-estimates sd per experiment) | ≈ 0.183 | ≈ 0.116 |

against a healthy-arm **population sd of 0.111** (39 arms). **The instrument was coarser than the
entire spread of the population it sampled** — by 1.5× at n = 8.

*Quote the estimator with the number.* The normal approximation is optimistic by roughly 10 %; the
campaign's published "MDE 0.150" was the normal form of a smaller sd estimate and understated the
instrument twice over.

At n = 20 the MDE reaches the scale of the effects actually observed. **The honest cost of a
readable round is roughly 2.5× the seeds, or two-thirds fewer arms.** Rounds 1–7
consistently chose more arms; that is why 11 of the 16 arms that beat their baseline on means do
not survive the variance bar.
