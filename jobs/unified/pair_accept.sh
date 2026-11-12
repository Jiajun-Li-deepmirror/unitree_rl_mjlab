#!/bin/bash
# pair_accept.sh R DEPTH_CKPT BLIND_CKPT (unitree-rl-mjlab-62 2026-11-12; doc/redesign.md 2.2.1, the user: "加「相机失效、
# 切到盲走」测试"): the camera-failure test of the delivered pair -- the depth student and the blind student, each
# accepted on its own -- in mjlab: scripts/unified/camera_fail.py's flat and stairs parts, unloaded and with the
# walking payload (jobs/accept_payload.sh), ACCEPT_JOBS (2) at a time. Files logs/checks/u_<r>_pair_<tag>_<part>[_pay].txt
# (tag: the depth checkpoint's run and candidate); the verdict in logs/jobs/unified_<r>_pair.verdict and the log
# logs/jobs/unified_<r>_accept.txt. Exit 0 PASS, 1 FAIL (the depth student is not delivered; the blind one is not
# affected), 4 a tool fault (a part without its verdict line). PAIR_SMOKE=1: scale 0.05, unloaded only, judged as usual
# (jobs/unified/smoke_students.sh: the tool chain, not the students).
# tinynav (tier 3, doc/redesign.md 2.3: map3 blinded mid-climb / mid-descent) is added once tinynav-sim has its switch.
set -u
R=$1 DCK=$2 BCK=$3 r=${1,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1
L=logs/jobs/unified_${r}_accept.txt V=logs/jobs/unified_${r}_pair.verdict
SMOKE=${PAIR_SMOKE:-0} JOBS=${ACCEPT_JOBS:-2}
log() { echo "$(date '+%m-%d %H:%M:%S') $R pair: $([ $SMOKE = 1 ] && echo 'SMOKE ')$*" >> $L; }
PAY=$(bash jobs/accept_payload.sh $R) || exit 2
tag=$(basename $(dirname $DCK))_$(basename $DCK .pt)
P=logs/checks/u_${r}_pair_$tag
SCALE=1.0 conds="unloaded loaded"
[ $SMOKE = 1 ] && SCALE=0.05 conds="unloaded"
mkdir -p logs/checks
t0=$(date +%s)
log "camera-failure test: depth $DCK + blind $BCK ($conds)"
outs=()
for cond in $conds; do
  if [ $cond = unloaded ]; then pay=0,0,0,0 X=""; else pay=$PAY X=_pay; fi
  for part in flat stairs; do
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do wait -n; done
    out=${P}_$part$X.txt
    outs+=($out)
    ( export U_PAYLOAD=$pay
      for i in 1 2 3 4 5 6 7; do   # out of GPU memory (a training beside it): wait 5 min and retry, as accept.sh
        $PY scripts/unified/camera_fail.py $R $DCK $BCK --part $part --scale $SCALE > $out 2>&1 && break
        grep -qiE "out of memory|CUDA error 2" $out || break
        sleep 300
      done ) &
  done
done
wait
fails=() fault=""
for out in "${outs[@]}"; do
  v=$(grep -E "^camera_fail verdict:" $out | tail -n 1 | sed 's/^camera_fail verdict: //')
  name=$(basename $out .txt | sed "s/^u_${r}_pair_${tag}_//")
  if [ -z "$v" ]; then fault="$fault $name"; log "TOOL FAULT: $name: $(grep -E 'Error' $out | tail -n 1 | cut -c1-160)"
  elif [[ "$v" != PASS ]]; then fails+=("$name: ${v#FAIL: }"); fi
done
mins=$(( ($(date +%s) - t0) / 60 ))
if [ -n "$fault" ]; then echo "TOOL FAULT:$fault" > $V; log "($mins min) tool fault:$fault (not a result)"; exit 4; fi
if [ ${#fails[@]} = 0 ]; then echo "PASS $DCK $BCK" > $V; log "($mins min) PASS"; exit 0; fi
echo "FAIL: ${fails[*]}" > $V
log "($mins min) FAIL: ${fails[*]}"
exit 1
