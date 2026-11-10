#!/bin/bash
# wait_logs.sh: exits when a chain log on t1 / t2 grows (logs/jobs/{blind,fe,repro}_*.txt), a fall diagnosis finishes or
# a tinynav verdict appears (MAXS seconds at most, default 7200); the monitoring loop waits on it.
R=/home/dm/.cache/go2-depth-work/ssh/r
D=/root/gpufree-data/go2-depth/depth_fe
q="cd $D; cat logs/jobs/blind_[a-z0-9]*.txt logs/jobs/fe_[a-z0-9]*.txt logs/jobs/repro_*.txt 2>/dev/null | wc -l; ls logs/accept/*.verdict 2>/dev/null | wc -l; tail -c 300 logs/jobs/repro_queue.txt 2>/dev/null | md5sum | cut -c1-6"
st() {
  a=$(timeout 40 $R 30714 "$q" < /dev/null 2>/dev/null | tr '\n' ' ')
  b=$(timeout 40 $R 31365 "$q" < /dev/null 2>/dev/null | tr '\n' ' ')
  echo "t1[$a] t2[$b]"
}
s0=$(st); echo "start $(date +%T) $s0"; t=0
while [ $t -lt ${MAXS:-7200} ]; do
  sleep 60; t=$((t + 60)); s=$(st)
  [ -n "$s" ] && [ "$s" != "$s0" ] && [[ "$s" != *"[]"* ]] && { echo "change $(date +%T) $s"; exit 0; }
done
echo "timeout $(date +%T) $s"
