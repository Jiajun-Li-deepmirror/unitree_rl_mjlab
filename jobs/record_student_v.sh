#!/bin/bash
# record_student_v.sh R CK OUTDIR TITLE (go2-depth 2026-11-09, the user: the depth student's acceptance videos, flat and
# stairs, with and without the payload): the blind delivery videos' scripts on a v2 stage-2 student, the env as
# manual_student_seeds.sh builds it (pilot = package + V2 extras - V2_DROP + V2_MORE, payload DR pinned):
#   flat      scripts/blind/record_flat.py --reset-demo on Unitree-<R>-Depth-Student-Flat (through jobs/student_eval.py)
#   flat_pay  the same with the payload acceptance's point mass
#   tour      scripts/go2_depth/record_map3.py (the map3 tour, as record_student.sh)
#   tour_pay  the same with the payload
# CLIPS picks clips (default all four). Logs beside the videos.
set -u
R=$1 CK=$2 OUT=$3 TITLE=$4
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA; else V2_EXTRA=$V2_LEGGED_EXTRA; fi
for k in ${V2_DROP:+${V2_DROP//,/ }}; do V2_EXTRA=$(echo "$V2_EXTRA" | tr ',' '\n' | grep -v -x "$k" | paste -sd,); done
EXTRA=$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE}
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null || exit 2
if [ -n "$V2_CONS" ]; then export BLIND_CONS_INIT=$V2_CONS; else unset BLIND_CONS_INIT; fi
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 \
  BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
case $R in Go2|Go2W) PAY=12,0.15,0,0.20 ;; As2) PAY=10,0.15,0,0.20 ;; As2W) PAY=13,0.15,0,0.20 ;; *) PAY=40,0.15,0,0.30 ;; esac
TX=""; case $R in B2|B2W) TX="--r1-top-y 5.6 --r1-turn-x -9.3" ;; *) TX="--r1-top-y 5.6" ;; esac
CLIPS=${CLIPS:-flat,flat_pay,tour,tour_pay}
want() { [[ ",$CLIPS," == *",$1,"* ]]; }
mkdir -p $OUT
echo "recording $CK (pilot md5 $(echo $BLIND_PILOT | md5sum | cut -c1-8), + $EXTRA)" | tee $OUT/recording.txt
L=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
want flat && flock $L $PY jobs/student_eval.py scripts/blind/record_flat.py --task Unitree-$R-Depth-Student-Flat $CK \
  --out $OUT/${r}_depth_flat.mp4 --title "$TITLE" --reset-demo > $OUT/flat.log 2>&1
want flat_pay && flock $L $PY jobs/student_eval.py scripts/blind/record_flat.py --task Unitree-$R-Depth-Student-Flat $CK \
  --out $OUT/${r}_depth_flat_payload.mp4 --title "$TITLE" --reset-demo --payload $PAY > $OUT/flat_pay.log 2>&1
want tour && flock $L env PYTHONPATH=. $PY scripts/go2_depth/record_map3.py --checkpoint $CK --robot $r \
  --out $OUT/${r}_depth_map3_tour.mp4 --width 1280 --height 720 $TX > $OUT/tour.log 2>&1
want tour_pay && flock $L env PYTHONPATH=. $PY scripts/go2_depth/record_map3.py --checkpoint $CK --robot $r \
  --out $OUT/${r}_depth_map3_tour_payload.mp4 --width 1280 --height 720 --payload $PAY $TX > $OUT/tour_pay.log 2>&1
ls -la $OUT
