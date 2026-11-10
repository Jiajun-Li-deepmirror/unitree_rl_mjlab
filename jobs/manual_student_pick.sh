#!/bin/bash
# manual_student_pick.sh R TAG CK (go2-depth 2026-11-08): fe_chain.sh's stage-2 pick for a checkpoint picked by hand
# (luck-level misses: accept-luck-pass), the env as manual_student_seeds.sh builds it: logs/picks_v2_<TAG>/<r>_student
# and the student export logs/exports/<r>_fe_student_<run>_avg<K>.onnx.
set -u
R=$1 TAG=$2 CK=$3
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
L=logs/jobs/fe_$r.txt PK=logs/picks_v2_$TAG
K=$(basename $CK .pt | cut -d_ -f2) RUN=$(basename $(dirname $CK) | sed 's/_avg[0-9]*$//')
X=logs/exports/${r}_fe_student_${RUN}_avg$K.onnx
echo "$CK" > $PK/${r}_student
$PY scripts/go2_depth/export_student.py --task Unitree-$R-Depth-Student-Edge --checkpoint $CK --out $X > logs/jobs/fe_${r}_export.log 2>&1 \
  && echo "$(date '+%m-%d %H:%M:%S') $R STUDENT PICK $CK (by hand, luck-level misses); exported $X (md5 $(md5sum < $X | cut -c1-8))" >> $L \
  || echo "$(date '+%m-%d %H:%M:%S') $R STUDENT PICK $CK; export FAILED: $(tail -n 3 logs/jobs/fe_${r}_export.log)" >> $L
tail -n 1 $L
