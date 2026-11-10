#!/bin/bash
# fall_diag_blind.sh R TAG SEED OUT -- SCRIPT ARGS (go2-depth 2026-11-08): jobs/fall_diag.py on a blind check script under
# the recipe env of blind chain TAG (rebuilt from logs/jobs/blind_<r>.txt as manual_seeds_log.sh does), env seed SEED.
set -u
R=$1 TAG=$2 SEED=$3 OUT=$4; shift 5
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/v2_robot.sh $R || exit 2
export $V2_PKG_VAR=$V2_PKG PAYLOAD_RAMP=0
source jobs/fe_robot.sh $R > /dev/null || exit 2
L=logs/jobs/blind_$r.txt
seg=$(awk -v t="blind chain $TAG:" 'index($0, t) {f=1} f' $L)
TP=$(echo "$seg" | grep -o "blind T PICK[^/]*\(logs/[^ ]*\.pt\)" | head -n 1 | grep -o "logs/.*\.pt")
PP=$(echo "$seg" | grep -o "blind P pilot .*" | head -n 1 | sed "s/blind P pilot //")
[ -n "$TP" ] && [ -n "$PP" ] || { echo "no T pick / P pilot for $TAG in $L" > $OUT; exit 2; }
export BLIND_PILOT=$PP BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$TP
flock ${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} $PY jobs/seed_run.py $SEED jobs/fall_diag.py "$@" > $OUT 2>&1
