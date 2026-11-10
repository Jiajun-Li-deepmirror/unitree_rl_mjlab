#!/bin/bash
# ws_eval_watch.sh R... (go2-depth 2026-11-03): jobs/ws_eval.sh once for each new wheelstep teacher average
# (logs/rsl_rl/<r>_depth_teacher/ws_v1_avg<K>/model_<K>.pt, a minute after it appears) of these robots, until killed.
cd /root/gpufree-data/go2-depth/depth_fe
while true; do
  for R in "$@"; do
    for A in logs/rsl_rl/${R,,}_depth_teacher/${WATCH_RUN:-ws_v}*_avg*/model_*.pt; do
      [ -f "$A" ] || continue
      K=$(basename $A .pt | cut -d_ -f2); NAME=${R,,}_$(basename $(dirname $A) | sed 's/_avg.*//')_$K
      grep -q " $NAME: " logs/checks/ws_eval.txt 2> /dev/null && continue
      [ $(( $(date +%s) - $(stat -c %Y $A) )) -ge 60 ] || continue
      bash jobs/ws_eval.sh $R $A $NAME
    done
  done
  sleep 60
done
