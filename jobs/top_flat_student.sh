#!/bin/bash
# top_flat_student.sh R CKPT NAME (go2-depth 2026-11-05): top_flat.sh for a depth student, through student_eval.py
# (--runner) on Unitree-<R>-Depth-Student-Stairs (clean depth); env TASK (whole task id, e.g. Unitree-Go2W-Depth-Student-Edge
# for the map3 training terrain), TASK_SUFFIX (e.g. -RealEdge), TF_EXTRA, TF_ARGS.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 CK=$2 NAME=$3
case $R in Go2W) export GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300;; As2W) export AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000;;
  B2W) export B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc;; esac
source jobs/fe_robot.sh $R || exit 2
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt),wheelstep,wheelstep4,stairspeed2,noslide${TF_EXTRA:-}
$PY jobs/student_eval.py jobs/top_flat_eval.py --task ${TASK:-Unitree-$R-Depth-Student-Stairs${TASK_SUFFIX:-}} --runner $CK ${TF_ARGS:-} > logs/checks/top_flat_$NAME.log 2>&1
grep -a '^top_flat_eval\|^ws_stair_eval' logs/checks/top_flat_$NAME.log | sed "s/^/$(date '+%m-%d %H:%M') $NAME: /" >> logs/checks/top_flat.txt
tail -n 2 logs/checks/top_flat_$NAME.log | cut -c1-300
