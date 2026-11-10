#!/bin/bash
# top_flat_v2.sh R CKPT NAME [EXTRA] (go2-depth 2026-11-09): jobs/top_flat_eval.py for a v2 depth teacher with the v2 env
# (package pilot + EXTRA, default the v2 wheeled extras), 1024 robots; lines to logs/checks/top_flat.txt.
set -u
R=$1 CK=$2 NAME=$3
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
EXTRA=${4:-$V2_WHEELED_EXTRA$V2_ROBOT_EXTRA}
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null || exit 2
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 \
  BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
flock ${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} $PY jobs/top_flat_eval.py --task Unitree-$R-Depth-$PH $CK --num-envs 1024 --steps 3000 > logs/checks/top_flat_$NAME.log 2>&1
grep -a '^top_flat_eval\|^ws_stair_eval' logs/checks/top_flat_$NAME.log | sed "s/^/$(date '+%m-%d %H:%M') $NAME: /" >> logs/checks/top_flat.txt
tail -n 2 logs/checks/top_flat_$NAME.log | cut -c1-220
