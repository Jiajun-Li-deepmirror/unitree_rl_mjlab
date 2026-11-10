#!/bin/bash
# repro_queue.sh TAG JOB... (go2-depth 2026-11-10): jobs/full_chain.sh for each JOB in turn on this GPU. JOB = R:stage
# (stage blind | teacher | s1 | s2 | all) or R (stage $REPRO_STAGE, default all), either with @TAG2 for a robot rerun under
# its own tag (2026-11-11, e.g. Go2W:teacher@rp4 after Go2W's blind round was rerun as rp4). The reproducible run (the user's order
# go2w, b2, as2w, go2, as2, b2w; t2 Go2W As2W As2, t1 B2 Go2 B2W) in four rounds -- blind, teacher, stage-1 student,
# stage-2 student -- each robot's round only after its previous one passed; a box done with a round goes on to the
# next round of its robots that are ready (the user: no GPU waiting for the other box). REPRO_PKG_<R> (e.g.
# REPRO_PKG_GO2W): a blind package copied from the other box. REPRO_AFTER_PID: wait for that process (the queue
# before) to end first.
# A job that is not ready (exit 3: its previous round did not pass) is skipped. One that fails after a real run is
# logged and the queue goes on; one that fails within REPRO_MIN_S (1800 s) stops the queue: the pipeline is broken,
# not the recipe (the first start, 11-10 16:56, ran through all six in two minutes on a log-layout mismatch).
set -u
TAG=$1; shift
cd /root/gpufree-data/go2-depth/depth_fe
Q=logs/jobs/repro_queue.txt
if [ -n "${REPRO_AFTER_PID:-}" ]; then
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG ($*): waiting for pid $REPRO_AFTER_PID" >> $Q
  while kill -0 $REPRO_AFTER_PID 2> /dev/null; do sleep 60; done
fi
for J in "$@"; do
  JT=$TAG
  [[ $J == *@* ]] && { JT=${J#*@}; J=${J%@*}; }
  R=${J%%:*} ST=${REPRO_STAGE:-all}
  [[ $J == *:* ]] && ST=${J#*:}
  t0=$(date +%s) pv=REPRO_PKG_${R^^}
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: $R $ST $JT starts" >> $Q
  REPRO_PKG=${!pv:-} bash jobs/full_chain.sh $R $JT $ST
  rc=$? dt=$(( $(date +%s) - t0 ))
  case $rc in
    0) w=passed ;; 3) w="skipped (its previous round has not passed)" ;; *) w="FAILED (exit $rc)" ;;
  esac
  echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: $R $ST $JT $w after $((dt / 60)) min" >> $Q
  if [ $rc != 0 ] && [ $rc != 3 ] && [ $dt -lt ${REPRO_MIN_S:-1800} ]; then
    echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: stopped -- $R $ST failed within ${dt} s (a pipeline fault)" >> $Q
    exit 1
  fi
done
echo "$(date '+%m-%d %H:%M:%S') repro queue $TAG: all done ($*)" >> $Q
