#!/bin/bash
# fe_queue.sh QUEUE (depth_fe, 2026-10-31): jobs/fe_chain.sh for each line of the file QUEUE in turn, one robot at a
# time on this GPU ("R [VAR=VALUE ...]", e.g. "B2 B2_N=8000"). A line is taken out before its chain starts, so lines
# can be added, removed or reordered while a chain runs (write QUEUE.new, then mv). Ends when QUEUE is empty; a robot
# whose chain fails is logged and the next one starts. Log: logs/jobs/queue.txt.
cd /root/gpufree-data/go2-depth/depth_fe || exit 1
Q=$1
while line=$(head -n 1 $Q 2> /dev/null) && [ -n "$line" ]; do
  tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q
  read -r R vars <<< "$line"
  echo "$(date '+%m-%d %H:%M:%S') queue: start $line" >> logs/jobs/queue.txt
  env $vars bash jobs/fe_chain.sh $R
  echo "$(date '+%m-%d %H:%M:%S') queue: $R chain exit $?" >> logs/jobs/queue.txt
done
echo "$(date '+%m-%d %H:%M:%S') queue: $Q empty, done" >> logs/jobs/queue.txt
