#!/bin/bash
# g18_start.sh G17_PID (go2-depth 2026-11-07): once Go2W g17 has saved model_14600, stop its training (its chain keeps
# checking 14100 / 14600), average its last 8 checkpoints and start Go2W g18 from that with spinflip added (the spin
# hold saw no reversals: spin_alt.py 0.24 / 0.20 / 0.17 m at g17 14100, 13600 0.25 / 0.21 / 0.16).
set -u
TP=$1
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
RD=$(ls -d logs/rsl_rl/go2w_blind/*_bl2P_g17 | sort | tail -n 1)
until [ -f $RD/model_14600.pt ]; do kill -0 $TP 2> /dev/null || break; sleep 30; done
sleep 20
kill -INT $TP 2> /dev/null; for _ in $(seq 60); do kill -0 $TP 2> /dev/null || break; sleep 2; done; kill $TP 2> /dev/null
K=$(ls $RD/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -n 1)
k0=$(( K / 100 * 100 )); CKS=""
for k in $(seq $((k0 - 700)) 100 $k0); do [ -f $RD/model_$k.pt ] && CKS="$CKS $RD/model_$k.pt"; done
I=logs/rsl_rl/go2w_blind/g18_init
mkdir -p $I
$PY jobs/avg_ckpt.py $I/model_$K.pt $CKS
echo "$(date '+%m-%d %H:%M:%S') Go2W g17 training stopped at $K; g18 from $I/model_$K.pt (g17 last 8: $CKS) + spinflip" >> logs/jobs/blind_go2w.txt
BL_T_PICK=logs/rsl_rl/go2w_blind/bl2T_g3_avg2000/model_2000.pt BL_T_RUN=logs/rsl_rl/go2w_blind/bl2T_g3_avg2000 \
BL_M_PICK=$I/model_$K.pt BL_M_RUN=$I \
BL_P_EXTRA=marchpose,marchclear,spinhold,stepclean,stairroll,tilt3,stairstab,spinflip bash jobs/blind_chain.sh Go2W g18
