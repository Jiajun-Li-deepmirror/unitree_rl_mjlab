#!/bin/bash
# ws_box.sh R... (go2-depth 2026-11-03): the wheelstep rounds of these robots in turn on this GPU (jobs/ws_<r>.sh).
# Starts once the G1 perception teacher here has paused for us (g1-loc 13:10: pause_at.sh stops it at model_3000 and
# writes WS_WAIT_STATE = paused) and no training has run here for 3 minutes in a row. Log logs/jobs/ws_box.txt.
cd /root/gpufree-data/go2-depth/depth_fe
L=logs/jobs/ws_box.txt
echo "$(date '+%m-%d %H:%M:%S') ws_box: waiting for ${WS_WAIT_STATE:-no state file} = paused and no training ($*)" >> $L
n=0
until [ $n -ge 3 ]; do
  if { [ -z "${WS_WAIT_STATE:-}" ] || grep -q '^paused' "$WS_WAIT_STATE" 2> /dev/null; } \
     && [ "$(ps -eo args | grep -cE '^[^ ]*python[^ ]* scripts/train[.]py')" = 0 ]; then n=$((n + 1)); else n=0; fi
  sleep 60
done
for R in "$@"; do
  echo "$(date '+%m-%d %H:%M:%S') ws_box: start $R (free $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) MiB)" >> $L
  bash jobs/ws_${R,,}.sh
  echo "$(date '+%m-%d %H:%M:%S') ws_box: $R chain exit $?" >> $L
done
echo "$(date '+%m-%d %H:%M:%S') ws_box: done ($*)" >> $L
