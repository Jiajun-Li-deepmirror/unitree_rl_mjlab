#!/bin/bash
# next_round.sh R PREV_TAG PREV_PID STOP_K NEW_TAG "P_EXTRA" [WIDEN_EXTRA] (go2-depth 2026-11-07): once the blind polish
# run *_bl2P_<PREV_TAG> has saved model_<STOP_K>, stop its training (its chain keeps checking what exists), average its
# last 8 checkpoints and start polish <NEW_TAG> from there with BL_P_EXTRA="P_EXTRA" (the whole extra key list). With
# WIDEN_EXTRA (keys adding critic inputs, e.g. spinhold) the average is widened first (jobs/widen_pkg.sh).
set -u
R=$1 PT=$2 TP=$3 K=$4 NT=$5 PX=$6 WX=${7:-}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
RD=$(ls -d logs/rsl_rl/${r}_blind/*_bl2P_$PT | sort | tail -n 1)
until [ -f $RD/model_$K.pt ]; do kill -0 $TP 2> /dev/null || break; sleep 20; done
sleep 20
kill -INT $TP 2> /dev/null; for _ in $(seq 60); do kill -0 $TP 2> /dev/null || break; sleep 2; done; kill $TP 2> /dev/null
K=$(ls $RD/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -n 1)
k0=$(( K / 100 * 100 )); CKS=""
for k in $(seq $((k0 - 700)) 100 $k0); do [ -f $RD/model_$k.pt ] && CKS="$CKS $RD/model_$k.pt"; done
I=logs/rsl_rl/${r}_blind/${NT}_init
mkdir -p $I
if [ -n "$WX" ]; then
  PK=logs/pkg_tmp/${r}_${PT}_$K
  mkdir -p $PK
  $PY jobs/avg_ckpt.py $PK/model_$K.pt $CKS
  grep -a "$R blind P pilot" logs/jobs/blind_$r.txt | tail -n 1 | sed 's/.*P pilot //' > $PK/blind_pilot.txt
  bash jobs/widen_pkg.sh $R $PK $I $WX
else
  $PY jobs/avg_ckpt.py $I/model_$K.pt $CKS
fi
case $R in Go2W) TPK=logs/rsl_rl/go2w_blind/bl2T_g3_avg2000/model_2000.pt ;; B2) TPK=logs/rsl_rl/b2_blind/bl2T_l2_avg2500/model_2500.pt ;; *) echo "next_round.sh: no T pick for $R"; exit 2 ;; esac
echo "$(date '+%m-%d %H:%M:%S') $R $PT training stopped at $K; $NT from $I/model_$K.pt (last 8: $CKS), keys $PX" >> logs/jobs/blind_$r.txt
BL_T_PICK=$TPK BL_T_RUN=$(dirname $TPK) BL_M_PICK=$I/model_$K.pt BL_M_RUN=$I BL_P_EXTRA=$PX bash jobs/blind_chain.sh $R $NT
