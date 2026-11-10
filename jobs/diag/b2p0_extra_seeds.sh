#!/bin/bash
# B2 student v2s2_b2n1p0 avg500: seeds 1 and 2 by hand (11-06), s0 failed on one stairs fall (15-18 cm up, 1 of ~25).
# The environment is the running student training's (the same pilot / payload / package fe_chain exported).
cd /root/gpufree-data/go2-depth/depth_fe
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/$1/environ
A=logs/rsl_rl/b2_depth_student/v2s2_b2n1p0_avg500/model_500.pt
for SEED in 1 2; do
  P=logs/checks/fe_b2_student_v2s2_b2n1p0_avg500_s$SEED
  bash jobs/fe_check.sh student B2 $A $SEED $P
  echo "$(date '+%m-%d %T') B2 student avg500 s$SEED (by hand): $(cat ${P}_verdict.txt)" >> logs/jobs/fe_b2_extra.txt
done
