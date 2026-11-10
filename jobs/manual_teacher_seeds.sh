#!/bin/bash
# manual_teacher_seeds.sh R CK TAGPREFIX "SEEDS" (go2-depth 2026-11-09): fe_check.sh teacher on more seeds for a v2 depth
# teacher average picked by hand (the As2W a2 / a3 teachers collapsed on map3 climbs after +1000; a3 avg12000 failed
# seed 0 on one loaded-stand fall only), the environment rebuilt as manual_student_seeds.sh does (pilot = package + V2
# extras + V2_MORE, anchor = blind_init_fe, payload without the ramp). No tinynav request (a scan teacher has no ONNX
# for it). Verdicts in logs/jobs/fe_<r>.txt.
set -u
R=$1 CK=$2 T=$3 SEEDS=$4
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA; else V2_EXTRA=$V2_LEGGED_EXTRA; fi
for k in ${V2_DROP:+${V2_DROP//,/ }}; do V2_EXTRA=$(echo "$V2_EXTRA" | tr ',' '\n' | grep -v -x "$k" | paste -sd,); done
EXTRA=$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE}
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0 FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
source jobs/fe_robot.sh $R > /dev/null || exit 2
if [ -n "$V2_CONS" ]; then export BLIND_CONS_INIT=$V2_CONS; else unset BLIND_CONS_INIT; fi
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 \
  BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
[ -f $BLIND_ANCHOR_CHECKPOINT ] || { echo "no $BLIND_ANCHOR_CHECKPOINT"; exit 2; }
L=logs/jobs/fe_$r.txt
echo "$(date '+%m-%d %H:%M:%S') manual teacher $T: $CK seeds $SEEDS, pilot + $EXTRA (md5 $(echo $BLIND_PILOT | md5sum | cut -c1-8))" >> $L
for s in $SEEDS; do
  bash jobs/fe_check.sh teacher $R $CK $s logs/checks/${T}_s$s
  echo "$(date '+%m-%d %H:%M:%S') manual teacher $T s$s: $(cat logs/checks/${T}_s${s}_verdict.txt)" >> $L
done
