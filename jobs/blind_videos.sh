#!/bin/bash
# blind_videos.sh R PKG (go2-depth 2026-11-11, the user: "盲走通过后自动录视频", "像深度学生那样合成一个上下对比的视频"):
# the acceptance videos of a blind delivery package. jobs/record_blind.sh records the four clips (flat: the posture
# reset, march in place and the speed matrix; the map3 tour with its stair stops; each without and with the payload
# limit of jobs/accept_payload.sh), then tools/go2_depth/video/stack_compare.py stacks each pair, unloaded on top:
#   PKG/videos/<r>_flat_compare.mp4   PKG/videos/<r>_map3_tour_compare.mp4
# Logs "R BLIND VIDEOS PKG/videos ..." to BL_LOG (default logs/jobs/blind_<r>.txt); the workstation's monitor
# (tools/go2_depth/monitor/events.py) shows the line and the two videos are fetched and sent to the user.
# blind_chain.sh starts it in the background after the delivery, so the depth round does not wait; PKG/videos.pid
# while it runs. Videos are not acceptance: exit 1 with "BLIND VIDEOS FAILED <why>" does not stop any chain.
set -u
R=$1 PKG=$2
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
L=${BL_LOG:-logs/jobs/blind_$r.txt}
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
V=$PKG/videos
mkdir -p $V
echo $$ > $PKG/videos.pid
trap 'rm -f $PKG/videos.pid' EXIT
PAY=$(bash jobs/accept_payload.sh $R) || { log "$R BLIND VIDEOS FAILED no payload for $R"; exit 1; }
t0=$(date +%s)
bash jobs/record_blind.sh $R $PKG $V > $V/record.log 2>&1
T="$R 盲走 $(basename $PKG)"
out="" bad=""
for c in "flat:${r}_flat:平地" "tour:${r}_map3_tour:map3 楼梯"; do
  IFS=: read -r k base what <<< "$c"
  a=$V/$base.mp4 b=$V/${base}_payload.mp4 o=$V/${base}_compare.mp4
  if [ ! -s $a ] || [ ! -s $b ]; then bad="$bad $k: no clip ($(tail -n 1 $V/$k.log $V/${k}_pay.log 2> /dev/null | grep -v '^==' | tr '\n' ' ' | cut -c1-120))"; continue; fi
  j=$($PY tools/go2_depth/video/stack_compare.py $a $b $o --title "$T $what" --top-label "无负载" \
      --bottom-label "负载 ${PAY%%,*} kg（前 $(echo $PAY | cut -d, -f2) m、上 $(echo $PAY | cut -d, -f4) m）" 2> $V/stack_$k.log | tail -n 1)
  if [ -s $o ]; then out="$out $(basename $o) ($(echo "$j" | grep -oE '"seconds": [0-9.]+' | grep -oE '[0-9.]+$') s, $(du -m $o | cut -f1) MB)"
  else bad="$bad $k: stacking failed ($(tail -n 1 $V/stack_$k.log | cut -c1-120))"; fi
done
m=$(( ($(date +%s) - t0) / 60 ))
if [ -n "$bad" ]; then log "$R BLIND VIDEOS FAILED after $m min:$bad${out:+; made$out}"; exit 1; fi
log "$R BLIND VIDEOS $V:$out (after $m min)"
