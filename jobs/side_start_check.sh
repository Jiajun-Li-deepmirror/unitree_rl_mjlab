#!/bin/bash
# side_start_check.sh R TAG RUN K... (go2-depth 2026-11-10): the sideways start from a stand (term_diag, cmd vy 0.3, 16 robots)
# on checkpoints of a blind chain run, with that chain's pilot; lines in logs/jobs/blind_<r>.txt
set -u
R=$1 TAG=$2 RD=$3; shift 3
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R > /dev/null; export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null
r=$(echo $R | tr A-Z a-z); L=logs/jobs/blind_$r.txt
export BLIND_PILOT=$(grep "$R blind chain $TAG: pilot " $L | tail -1 | sed "s/.*chain $TAG: pilot //; s/; polish adds.*//") BLIND_ANCHOR_COEFF=0
for k in "$@"; do
  [ -f $RD/model_$k.pt ] || continue
  out=$(flock /root/gpufree-data/go2-depth/ws_check.lock $PY jobs/term_diag.py --task Unitree-$R-Blind-Flat-Scratch $RD/model_$k.pt       --cmd 0,0.3,0 --seconds 4 --ramp 2 --settle 1.5 --num-envs 16 2>&1 | grep '^cmd' | sed 's/^cmd 0,0.3,0: //')
  echo "$(date '+%m-%d %H:%M:%S') $R $TAG model_$k sideways start: $out" >> $L
done
