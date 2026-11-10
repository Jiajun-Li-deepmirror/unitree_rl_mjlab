#!/bin/bash
# repro_queue.sh TAG R... (go2-depth 2026-11-10): jobs/full_chain.sh for each robot in turn on this GPU (the user's order
# go2w, b2, as2w, go2, as2, b2w over the two boxes: t2 Go2W As2W As2, t1 B2 Go2 B2W). A failed chain is logged and the
# next robot starts (the reproducible run's outcome per robot is the point).
set -u
TAG=$1; shift
cd /root/gpufree-data/go2-depth/depth_fe
for R in "$@"; do
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: $R starts" >> logs/jobs/repro_queue.txt
  bash jobs/full_chain.sh $R $TAG
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: $R ended (exit $?)" >> logs/jobs/repro_queue.txt
done
