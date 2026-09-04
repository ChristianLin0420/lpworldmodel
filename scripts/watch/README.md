# Campaign watchers

Long-running loops that keep a campaign moving without a human in the loop. Each one:

* **submits the work that is ready** — an eval for every training run that has just reached its
  `DONE` sentinel — using `RUN_NAME` as an **environment variable**. `plan_slurm.sbatch` requires
  it that way; passing it positionally makes every eval die instantly, which happened once in this
  campaign and produced a report of "6 evals running" when all six were already dead.
* **treats the terminal marker as the only proof of completion.** A run counts as evaluated when
  `final_eval/success_rate` appears in its `logs.json` — never when a submission line was printed,
  and never when the file merely exists. `logs.json` is written incrementally, which misled a
  progress report three separate times.
* **reports only when a number changes.** The first version of `wave22_monitor.sh` reprinted an
  identical nine-arm summary every 15 minutes once the arms had settled, which buries the one line
  that matters.
* **never reads a contrast below n = 8.** They print `n` beside every interval so a thin one is
  obvious. T1's `decode` contrast read −0.233 [−0.39, −0.08] at n = 3, excluding zero, and settled
  at −0.060 [−0.21, +0.09] at n = 8.

| script | what it does |
|---|---|
| `wave23_autopilot.sh` | Round 5: evaluates T1/T2/T3/T6 runs as they finish; reports each paired contrast against its own control. |
| `r6_watch.sh` | Announces the round-6 launch, and reports when every remaining round-5 contrast reaches n = 8. |
| `w25_watch.sh` | Reports each round-6 canary with a **verdict**, not just completion — every family is checked against the guard that would damn it (see below) — then announces the full 8-seed launch. |
| `ctrb_finish.sh` | Finished round 4's last arm. Kept as the smallest worked example. |

## Why the canary guards are per-family

A canary that only checks "did it finish" cannot catch the ways these objectives fail:

| family | proves the term is live | the guard |
|---|---|---|
| R6 support | `support_s` | `support_z_rms` — the Jaccard is scale-invariant where MSE is not, so a model can lower `S` by inflating the code uniformly |
| R2 consistency | `consist_loss` | `consist_rel` |
| R3 SAM | `sam_sharpness` | `sam_d_action_over_scale` — the objective's unconstrained minimiser is a predictor that ignores the action, i.e. `d_action = 0` |
| R4 ε | `incr_ess` | `incr_span` — ESS 0.063 at a 3884× span is the whole explanation of V1's −0.383 |
| R1 K sweep | — | `rel_mse` only; K changes no loss term |

Every family also fails on the pre-registered death condition `rel_mse >= 0.5` or on a NaN.

Match families by **longest prefix first**. `PiWM-jump5` is round 5's arm and `PiWM-jump2/3/8` are
round 6's; a plain `jump` prefix reports the wrong round's canaries as round-6 passes.

## Running one

These are driven by the agent's `Monitor` tool, which turns each stdout line into a notification,
so the filter must be selective:

```
bash scripts/watch/w25_watch.sh 2>&1 \
  | grep -E --line-buffered "^CANARY \|\||^WAVE25 FULL LAUNCH|Traceback|error:"
```

Filter to the lines you would act on — including the failure signatures. A watcher that greps only
for the success marker stays silent through a crashloop, and silence is indistinguishable from
"still running".
