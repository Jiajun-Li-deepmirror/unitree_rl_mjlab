#!/bin/bash
# manual_seeds.sh R TRAIN_PID CK TAGPREFIX "SEEDS" (go2-depth 2026-11-07): the whole acceptance (fe_check.sh blind) of an
# average checkpoint CK on the given seeds, with the recipe environment read from a training process of the same
# polish (/proc/TRAIN_PID/environ), and the tinynav request after it (logs/accept/<TAGPREFIX>.req). For a checkpoint
# whose chain stopped at a luck-like seed-0 miss (one robot of 256; accept-luck-pass).
set -u
R=$1 TP=$2 CK=$3 T=$4 SEEDS=$5
cd /root/gpufree-data/go2-depth/depth_fe
for v in BLIND_PILOT BLIND_ANCHOR_COEFF BLIND_ANCHOR_CHECKPOINT BLIND_PHASE_START_IT BLIND_PAYLOAD BLIND_CONS_INIT BLIND_FOOT_R FE_LOCK; do
  val=$(tr '\0' '\n' < /proc/$TP/environ | grep "^$v=" | head -n 1 | cut -d= -f2-)
  [ -n "$val" ] && export $v="$val"
done
for s in $SEEDS; do
  bash jobs/fe_check.sh blind $R $CK $s logs/checks/${T}_s$s
  echo "$(date '+%m-%d %H:%M:%S') manual $T s$s: $(cat logs/checks/${T}_s${s}_verdict.txt)" >> logs/jobs/blind_${R,,}.txt
done
mkdir -p logs/accept
echo "$R $CK BLIND $BLIND_PILOT" > logs/accept/$T.req
