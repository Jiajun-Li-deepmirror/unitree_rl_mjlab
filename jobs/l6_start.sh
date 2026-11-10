#!/bin/bash
# l6_start.sh (go2-depth 2026-11-06/07): when B2 l5's training (pid $1) ends, average its last 8 checkpoints, widen the
# average for spinhold's critic input (tilt3 / spinhold / spinflip on top of l5's keys) and start B2 l6 (polish,
# 2000 it, 4096 robots) from it. l5's chain keeps checking its 9500 / 10000 / 10500 averages.
set -u
TP=$1
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
while kill -0 $TP 2> /dev/null; do sleep 30; done
# from l5's validated avg9500 (11-07 00:42: 9500 passed seeds 0-2 except the alternating spins, which l6 is for;
# avg10000 also failed map3 without the operator and fell on the stairs, so not l5's last checkpoints)
K=9500
PK=logs/pkg_tmp/b2_l5_$K
mkdir -p $PK
cp logs/rsl_rl/b2_blind/bl2P_l5_avg9500/model_9500.pt $PK/model_$K.pt
CKS="logs/rsl_rl/b2_blind/bl2P_l5_avg9500/model_9500.pt"
L5P=$(grep -a 'B2 blind P pilot' logs/jobs/blind_b2.txt | tail -n 1 | sed 's/.*P pilot //')
echo "$L5P" > $PK/blind_pilot.txt
bash jobs/widen_pkg.sh B2 $PK logs/rsl_rl/b2_blind/l6_init tilt3,spinhold,spinflip
echo "$(date '+%m-%d %H:%M:%S') B2 l6 from $PK/model_$K.pt (= $CKS)" >> logs/jobs/blind_b2.txt
BL_T_PICK=logs/rsl_rl/b2_blind/bl2T_l2_avg2500/model_2500.pt BL_T_RUN=logs/rsl_rl/b2_blind/bl2T_l2_avg2500 \
BL_M_PICK=logs/rsl_rl/b2_blind/l6_init/model_$K.pt BL_M_RUN=logs/rsl_rl/b2_blind/l6_init \
BL_P_EXTRA=stairroll,support2,feetonce,edgecons,tilt3,spinhold,spinflip bash jobs/blind_chain.sh B2 l6
