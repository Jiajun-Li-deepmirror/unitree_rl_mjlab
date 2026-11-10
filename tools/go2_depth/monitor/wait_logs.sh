#!/bin/bash
# wait_logs.sh: exits when blind_as2w.txt / blind_go2.txt (t1) or fe_go2w.txt (t2) grows, a fall diagnosis finishes, or a tinynav verdict appears
R=/home/dm/.cache/go2-depth-work/ssh/r
D=/root/gpufree-data/go2-depth/depth_fe
st() {
  a=$(timeout 40 $R 30714 "cd $D; cat logs/jobs/blind_as2w.txt logs/jobs/blind_go2.txt logs/jobs/fe_as2w.txt logs/jobs/blind_as2.txt 2>/dev/null | wc -l; ls logs/accept/*.verdict | wc -l; tail -c 300 logs/jobs/blind_as2.txt | md5sum | cut -c1-6" < /dev/null 2>/dev/null | tr '\n' ' ')
  b=$(timeout 40 $R 31365 "cd $D; cat logs/jobs/fe_go2w.txt logs/jobs/blind_go2.txt 2>/dev/null | wc -l; ls logs/accept/*.verdict | wc -l; tail -c 300 logs/jobs/fe_go2w.txt | md5sum | cut -c1-6" < /dev/null 2>/dev/null | tr '\n' ' ')
  echo "t1[$a] t2[$b]"
}
s0=$(st); echo "start $(date +%T) $s0"; t=0
while [ $t -lt ${MAXS:-7200} ]; do
  sleep 60; t=$((t + 60)); s=$(st)
  [ -n "$s" ] && [ "$s" != "$s0" ] && [[ "$s" != *"[]"* ]] && { echo "change $(date +%T) $s"; exit 0; }
done
echo "timeout $(date +%T) $s"
