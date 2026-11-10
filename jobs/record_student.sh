#!/bin/bash
# record_student.sh R CK OUT (go2-depth 2026-11-09, the user: videos of the final depth model): scripts/go2_depth/
# record_map3.py's map3 tour for a v2 stage-2 student, the env as manual_student_seeds.sh builds it.
set -u
R=$1 CK=$2 OUT=$3
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA; else V2_EXTRA=$V2_LEGGED_EXTRA; fi
EXTRA=$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE}
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null || exit 2
if [ -n "$V2_CONS" ]; then export BLIND_CONS_INIT=$V2_CONS; else unset BLIND_CONS_INIT; fi
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 \
  BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
mkdir -p $(dirname $OUT)
flock ${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} env PYTHONPATH=. $PY scripts/go2_depth/record_map3.py --checkpoint $CK --robot $r --out $OUT --width 1280 --height 720 ${REC_ARGS:-} > ${OUT%.mp4}.log 2>&1
tail -n 3 ${OUT%.mp4}.log
