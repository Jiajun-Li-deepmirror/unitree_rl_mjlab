#!/bin/bash
# repro_queue.sh TAG R... (go2-depth 2026-11-10): jobs/full_chain.sh for each robot in turn on this GPU (the user's order
# go2w, b2, as2w, go2, as2, b2w over the two boxes: t2 Go2W As2W As2, t1 B2 Go2 B2W). A chain that fails after a real
# run is logged and the next robot starts (the reproducible run's outcome per robot is the point); one that fails within
# REPRO_MIN_S (1800 s) stops the queue: that is the pipeline broken, not the recipe (the first start, 11-10 16:56, ran
# through all six in two minutes on a log-layout mismatch).
set -u
TAG=$1; shift
cd /root/gpufree-data/go2-depth/depth_fe
Q=logs/jobs/repro_queue.txt
for R in "$@"; do
  t0=$(date +%s)
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: $R starts" >> $Q
  bash jobs/full_chain.sh $R $TAG
  rc=$? dt=$(( $(date +%s) - t0 ))
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: $R ended (exit $rc after ${dt} s)" >> $Q
  if [ $rc != 0 ] && [ $dt -lt ${REPRO_MIN_S:-1800} ]; then
    echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: stopped -- $R failed within ${dt} s (a pipeline fault)" >> $Q
    exit 1
  fi
done
