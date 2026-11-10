#!/bin/bash
# manual_seeds_log.sh R TAG CK TAGPREFIX "SEEDS" (go2-depth 2026-11-08): manual_seeds.sh for a blind chain whose training
# has ended: the recipe environment rebuilt as blind_chain.sh's P phase (its "P pilot" line in logs/jobs/blind_<r>.txt for
# TAG, its "T PICK" checkpoint, the payload without the ramp), then fe_check.sh blind on SEEDS and the tinynav request
# (written first, so the tinynav acceptance runs alongside).
set -u
R=$1 TAG=$2 CK=$3 T=$4 SEEDS=$5
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
export $V2_PKG_VAR=$V2_PKG
export FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null || exit 2
unset BLIND_CONS_INIT; [ -n "$V2_CONS" ] && export BLIND_CONS_INIT=$V2_CONS
L=logs/jobs/blind_$r.txt
# the chain's lines for TAG: from its "blind chain TAG:" line on
seg=$(awk -v t="blind chain $TAG:" 'index($0, t) {f=1} f' $L)
TP=$(echo "$seg" | grep -o "blind T PICK[^/]*\(logs/[^ ]*\.pt\)" | head -n 1 | grep -o "logs/.*\.pt")
PP=$(echo "$seg" | grep -o "blind P pilot .*" | head -n 1 | sed "s/blind P pilot //")
[ -n "$TP" ] && [ -n "$PP" ] || { echo "no T pick / P pilot for $TAG in $L"; exit 2; }
export BLIND_PILOT=$PP BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$TP BLIND_PHASE_START_IT=$(basename $TP .pt | cut -d_ -f2)
echo "$(date '+%m-%d %H:%M:%S') manual $T: $CK seeds $SEEDS, T $TP, pilot md5 $(echo $PP | md5sum | cut -c1-8)" >> $L
mkdir -p logs/accept
echo "$R $CK BLIND $BLIND_PILOT" > logs/accept/$T.req
for s in $SEEDS; do
  bash jobs/fe_check.sh blind $R $CK $s logs/checks/${T}_s$s
  echo "$(date '+%m-%d %H:%M:%S') manual $T s$s: $(cat logs/checks/${T}_s${s}_verdict.txt)" >> $L
done
