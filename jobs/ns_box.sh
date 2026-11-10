#!/bin/bash
# ns_box.sh R... (go2-depth 2026-11-04): the noslide rounds of these robots in turn on this GPU (jobs/ns_<r>.sh),
# once no training has run here for 3 minutes. Log logs/jobs/ns_box.txt.
cd /root/gpufree-data/go2-depth/depth_fe
L=logs/jobs/ns_box.txt
n=0
until [ $n -ge 3 ]; do
  if [ "$(ps -eo args | grep -cE '^[^ ]*python[^ ]* scripts/train[.]py')" = 0 ]; then n=$((n + 1)); else n=0; fi
  sleep 60
done
for R in "$@"; do
  echo "$(date '+%m-%d %H:%M:%S') ns_box: start $R" >> $L
  bash jobs/ns_${R,,}.sh
  rc=$?
  echo "$(date '+%m-%d %H:%M:%S') ns_box: $R chain exit $rc" >> $L
done
echo "$(date '+%m-%d %H:%M:%S') ns_box: done ($*)" >> $L
