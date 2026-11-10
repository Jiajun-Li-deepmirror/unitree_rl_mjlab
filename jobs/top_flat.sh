#!/bin/bash
# top_flat.sh R CKPT NAME (go2-depth 2026-11-04): jobs/top_flat_eval.py for a wheeled teacher checkpoint (as ws_eval.sh,
# pilot + wheelstep, wheelstep4, stairspeed2, noslide), 512 robots, 30 s (env TF_ARGS: e.g. "--num-envs 1024 --steps 3000");
# both result lines to logs/checks/top_flat.txt.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 CK=$2 NAME=$3
case $R in Go2W) export GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300;; As2W) export AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000;;
  B2W) export B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc;; esac
source jobs/fe_robot.sh $R || exit 2
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt),wheelstep,wheelstep4,stairspeed2,noslide${TF_EXTRA:-}  # TF_EXTRA (11-05): e.g. ",longexit,endwall"
$PY jobs/top_flat_eval.py --task Unitree-$R-Depth-$PH $CK ${TF_ARGS:-} > logs/checks/top_flat_$NAME.log 2>&1
grep -a '^top_flat_eval\|^ws_stair_eval' logs/checks/top_flat_$NAME.log | sed "s/^/$(date '+%m-%d %H:%M') $NAME: /" >> logs/checks/top_flat.txt
tail -n 2 logs/checks/top_flat_$NAME.log | cut -c1-220
