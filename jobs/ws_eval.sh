#!/bin/bash
# ws_eval.sh R CKPT NAME (go2-depth 2026-11-03): jobs/ws_stair_eval.py for a wheeled teacher checkpoint with its
# delivery's pilot + wheelstep (fe_robot.sh, the ws_<r>.sh package), 512 robots, 30 s; result line appended to
# logs/checks/ws_eval.txt as "NAME: ...".
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 CK=$2 NAME=$3
case $R in Go2W) export GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300;; As2W) export AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000;;
  B2W) export B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc;; esac
source jobs/fe_robot.sh $R || exit 2
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)${WS_EVAL_PILOT:-,wheelstep,noslide}
out=$($PY jobs/ws_stair_eval.py --task Unitree-$R-Depth-$PH $CK 2>&1 | tail -n 1)
echo "$(date '+%m-%d %H:%M') $NAME: $out" >> logs/checks/ws_eval.txt
