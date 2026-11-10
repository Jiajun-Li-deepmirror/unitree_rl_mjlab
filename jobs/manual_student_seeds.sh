#!/bin/bash
# manual_student_seeds.sh R CK TAGPREFIX "SEEDS" (go2-depth 2026-11-08): the stage-2 student checks of a v2_chain run
# whose chain stopped at a luck-like seed-0 miss (accept-luck-pass), the environment rebuilt as v2_chain.sh + fe_chain.sh
# set it for the student checks (pilot = package + V2 extras, anchor = blind_init_fe, payload without the ramp); the
# tinynav request (logs/accept/<TAGPREFIX>.req) written first so the tinynav acceptance runs alongside, then
# fe_check.sh student on SEEDS. Verdicts in logs/jobs/fe_<r>.txt.
set -u
R=$1 CK=$2 T=$3 SEEDS=$4
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
if [ $V2_KIND = wheeled ]; then V2_EXTRA=$V2_WHEELED_EXTRA; else V2_EXTRA=$V2_LEGGED_EXTRA; fi
EXTRA=$V2_EXTRA$V2_ROBOT_EXTRA${V2_MORE:+,$V2_MORE}
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0 FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
source jobs/fe_robot.sh $R > /dev/null || exit 2
if [ -n "$V2_CONS" ]; then export BLIND_CONS_INIT=$V2_CONS; else unset BLIND_CONS_INIT; fi
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 \
  BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
[ -f $BLIND_ANCHOR_CHECKPOINT ] || { echo "no $BLIND_ANCHOR_CHECKPOINT"; exit 2; }
L=logs/jobs/fe_$r.txt
echo "$(date '+%m-%d %H:%M:%S') manual student $T: $CK seeds $SEEDS, pilot + $EXTRA (md5 $(echo $BLIND_PILOT | md5sum | cut -c1-8))" >> $L
mkdir -p logs/accept
echo "$R $CK $EXTRA" > logs/accept/$T.req
for s in $SEEDS; do
  bash jobs/fe_check.sh student $R $CK $s logs/checks/${T}_s$s
  echo "$(date '+%m-%d %H:%M:%S') manual student $T s$s: $(cat logs/checks/${T}_s${s}_verdict.txt)" >> $L
done
