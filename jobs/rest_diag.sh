#!/bin/bash
# rest_diag.sh TAG CKPT (go2-depth 2026-11-10): Go2 standing at the zero command for 8 s (play config, 128 robots), with
# 0 kg, 12 kg at the front corner and the config's payload draw; term_diag counts per termination; chain TAG's pilot.
set -u
TAG=$1 CK=$2
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh Go2 > /dev/null; export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh Go2 > /dev/null
export BLIND_PILOT=$(grep "Go2 blind chain $TAG: pilot " logs/jobs/blind_go2.txt | tail -1 | sed "s/.*chain $TAG: pilot //; s/; polish adds.*//") BLIND_ANCHOR_COEFF=0
for pay in 0,0,0,0 12,0.15,0,0.20 cfg; do
  extra="--payload $pay"; [ $pay = cfg ] && extra="--train"
  r=$(flock /root/gpufree-data/go2-depth/ws_check.lock $PY jobs/term_diag.py --task Unitree-Go2-Blind-Flat-Scratch $CK --cmd 0,0,0 --seconds 8 --num-envs 128 $extra 2>&1 | grep '^cmd' | sed 's/^cmd 0,0,0: //')
  echo "$(date '+%m-%d %H:%M:%S') Go2 $TAG rest 8 s payload $pay: $r" >> logs/jobs/blind_go2.txt
done
