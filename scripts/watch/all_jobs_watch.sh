#!/bin/bash
# Fire once when EVERY outstanding round-6 and round-7 arm has finished training AND been
# evaluated, and nothing relevant is left in the queue.
cd /lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/lpworldmodel
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/envs/lpwm/bin/python
ARMS="PiWM-jump8 PiWM-overshoot8 LpWM-ltv-p1 PiWM-cols-p4 PiWM-cols-p16 \
LpWM-ltv-d1536 LpWM-ltv-d6144 LpWM-ltv-lr9e5 LpWM-ltv-d2048-hilr"
LAST=""
while true; do
  tot=0; dn=0; ev=0
  for a in $ARMS; do
    for d in runs/outputs/${a}_pd*/; do
      [ -d "$d" ] || continue; r=$(basename "$d"); case "$r" in CANARY-*) continue ;; esac
      tot=$((tot+1))
      [ -f "$d/DONE" ] && dn=$((dn+1)) || continue
      grep -lq "final_eval/success_rate" plan_outputs/*_${r}_gH5/logs.json 2>/dev/null && ev=$((ev+1))
    done
  done
  loo=$(ls -d plan_outputs/*_PiWM-loo11-*_gH5 2>/dev/null | wc -l)
  loodone=$(grep -l "final_eval/success_rate" plan_outputs/*_PiWM-loo11-*_gH5/logs.json 2>/dev/null | wc -l)
  q=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "^(PiWM-jump8|PiWM-overshoot8|LpWM-ltv-p1|PiWM-cols-p|LpWM-ltv-d1536|LpWM-ltv-d6144|LpWM-ltv-lr9e5|LpWM-ltv-d2048-hilr)_|^eval_(PiWM-jump8|PiWM-overshoot8|LpWM-ltv-p1|PiWM-cols-p|LpWM-ltv-d|LpWM-ltv-lr9e5|PiWM-loo11)")
  CUR="ALLJOBS || trained ${dn}/${tot} || evald ${ev}/${dn} || loo ${loodone}/${loo} || queue ${q}"
  if [ "$dn" -eq "$tot" ] && [ "$ev" -eq "$dn" ] && [ "$q" -eq 0 ] && [ "$tot" -gt 0 ]; then
    echo "ALLJOBS COMPLETE || ${tot} runs trained and evaluated || loo ${loodone}/${loo} || queue empty"
    exit 0
  fi
  [ "$CUR" != "$LAST" ] && { echo "$CUR"; LAST="$CUR"; }
  sleep 900
done
