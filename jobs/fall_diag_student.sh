#!/bin/bash
# fall_diag_student.sh R SEED OUT -- SCRIPT ARGS (go2-depth 2026-11-08): jobs/fall_diag.py on a flat check script for a
# v2 stage-2 student, the env as manual_student_seeds.sh builds it, through student_eval.py + seed_run.py SEED.
set -u
R=$1 SEED=$2 OUT=$3; shift 4
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA; else V2_EXTRA=$V2_LEGGED_EXTRA; fi
EXTRA=$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE}
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null || exit 2
if [ -n "$V2_CONS" ]; then export BLIND_CONS_INIT=$V2_CONS; else unset BLIND_CONS_INIT; fi
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 \
  BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
flock ${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} $PY jobs/student_eval.py jobs/seed_run.py $SEED jobs/fall_diag.py "$@" > $OUT 2>&1
