#!/bin/bash
# Round 6 (wave25) watch, POST-LAUNCH phase.
#
# The launch-phase version of this file reported "WAVE25 FULL LAUNCH || N run dirs || M training
# jobs queued". M decrements every time a job finishes, so the line changed on every cycle and
# the monitor re-reported an event that had already happened -- the same noise failure that
# r6_watch.sh had. Anything that embeds a COUNTDOWN in a change-detected string will do this.
# Report state that is monotone or terminal, never a draining queue depth.
#
# What matters now:
#   1. DEATH CONDITION AT END OF TRAINING. rel_mse >= 0.5 was pre-registered as death, but it is
#      an END-OF-TRAINING threshold. The 1-epoch canaries are NOT a valid place to apply it
#      (3 of 4 incr-eps canaries exceeded it; judging them there would repeat the mid-training
#      read this campaign has four retractions for). So it is applied HERE, on finished runs.
#   2. Arms completing training, by proposal.
#   3. First CEM results per arm.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
ARMS="PiWM-support-w0p03 PiWM-support-w0p1 PiWM-support-w0p3 \
PiWM-consist-w0p03 PiWM-consist-w0p1 PiWM-consist-w0p3 PiWM-consist-w0p1-data \
PiWM-sam-r0p01 PiWM-sam-r0p03 PiWM-sam-r0p1 \
PiWM-incr-eps0p001 PiWM-incr-eps0p01 PiWM-incr-eps0p041 PiWM-incr-eps0p041-clip10 \
PiWM-jump2 PiWM-overshoot2 PiWM-jump3 PiWM-overshoot3 PiWM-jump8 PiWM-overshoot8"
LAST=""
DEAD_SEEN=""
# Report each guard AT MOST ONCE PER ARM. The verdict is an arm-level judgement across
# seeds (a single seed below the threshold is inside baseline variation), so per-seed
# reporting is both noisy and misleading -- R4 alone would emit ~32 identical lines.
ARM_DEATH=""
ARM_DACT=""
while true; do
  # --- 1. death condition on FINISHED runs only ---------------------------------
  for a in $ARMS; do
    for d in runs/outputs/${a}_pd*/; do
      [ -f "$d/DONE" ] || continue
      r=$(basename "$d"); case "$r" in CANARY-*) continue ;; esac
      case " $DEAD_SEEN " in *" $r "*) continue ;; esac
      # Two guards, applied to EVERY arm (the d_action one is not SAM-specific: it fired
      # first on R6's support arms). They are in DIFFERENT UNITS -- which has already produced one false
      # alarm. rel_mse is raw. The SAM guard is on causal/d_action_over_scale, NOT
      # causal/d_action: the threshold 0.2746 is half the LpWM-ltv PROBE median of 0.5491
      # (assets/d_action_probe.json, n=16), and that 0.5491 is an over_scale figure. The
      # baseline's raw d_action median is 0.3535. Comparing raw against the normalised
      # threshold understates every arm by ~1.55x and invents collapses. Baselines log no
      # causal/d_action at all (the diagnostic postdates them), so the probe is the only
      # valid source for the threshold.
      read -r v ov < <($PY - "$d" <<'EOF' 2>/dev/null
import json, glob, sys
f = glob.glob(sys.argv[1] + "/wandb/run-*/files/wandb-summary.json")
d = json.load(open(f[0])) if f else {}
rm = next((d[k] for k in d if "rel_mse" in k), None)
ov = d.get("causal/d_action_over_scale")
print(f"{rm if rm is not None else -1:.4f} {ov if ov is not None else -1:.4f}")
EOF
)
      [ -z "$v" ] && continue
      DEAD_SEEN="$DEAD_SEEN $r"
      arm="${r%%_pd*}"
      if awk -v v="$v" 'BEGIN{exit !(v>=0.5)}'; then
        case " $ARM_DEATH " in *" $arm "*) ;; *)
          ARM_DEATH="$ARM_DEATH $arm"
          echo "R6 DEATH-CONDITION $arm (first seed $r) rel_mse=$v >=0.5 at end of training" ;;
        esac
      fi
      # per-seed breach is INSIDE baseline variation (baseline probe range [0.222, 0.629]);
      # report it, but it is an arm-level judgement across seeds, not a per-seed kill switch.
      if awk -v v="$ov" 'BEGIN{exit !(v>=0 && v<0.2746)}'; then
        case " $ARM_DACT " in *" $arm "*) ;; *)
          ARM_DACT="$ARM_DACT $arm"
          echo "R6 DACTION-GUARD $arm (first seed $r) d_action_over_scale=$ov <0.2746; judge the ARM across seeds" ;;
        esac
      fi
    done
  done
  # --- 2/3. completion + first results, reported only on change -----------------
  $PY analysis/collect_evals.py --out /tmp/w25.json >/dev/null 2>&1
  CUR=$($PY - <<'EOF' 2>/dev/null
import json, glob, os
A = json.load(open("/tmp/w25.json"))["arms"]
GROUP = {"R6 support": ["support-w0p03","support-w0p1","support-w0p3"],
         "R1 K-sweep": ["jump2","overshoot2","jump3","overshoot3","jump8","overshoot8"],
         "R2 consist": ["consist-w0p03","consist-w0p1","consist-w0p3","consist-w0p1-data"],
         "R3 sam":     ["sam-r0p01","sam-r0p03","sam-r0p1"],
         "R4 incr-eps":["incr-eps0p001","incr-eps0p01","incr-eps0p041","incr-eps0p041-clip10"]}
out = []
for g, arms in GROUP.items():
    done = sum(1 for a in arms for d in glob.glob(f"runs/outputs/PiWM-{a}_pd*/")
               if os.path.exists(d + "/DONE") and "CANARY-" not in d)
    ev = sum(len(A.get(f"PiWM-{a}", {})) for a in arms)
    out.append(f"{g}: trained {done}/{len(arms)*8} evald {ev}")
print("R6 || " + " || ".join(out))
EOF
)
  [ -n "$CUR" ] && [ "$CUR" != "$LAST" ] && { echo "$CUR"; LAST="$CUR"; }
  sleep 900
done
