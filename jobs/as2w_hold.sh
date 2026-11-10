#!/bin/bash
# as2w_hold.sh (go2-depth 2026-11-02, my call): the As2-W student PPO fine-tune as as2w_rl.sh (t1, fe_edge_v6rl) but with
# the B2W's standstill leg hold ("leghold20": legs held 2 s into a zero-command stand; logs/pkg/as2w_mixed_8000_hold =
# as2w_mixed_8000 with leghold20 added to its pilot), training and checks alike: go2-home's swap test, the remaining
# As2-W faller (env 1588, s2) shifts a foot 4-5 s into the stand under the largest delay and falls; the teacher does
# not. On t2 (idle since the B2W chain), checks after training; the deployment would need the leg hold too.
cd /root/gpufree-data/go2-depth/depth_fe
[ -s logs/pkg/as2w_mixed_8000_hold/blind_pilot.txt ] && grep -q leghold20 logs/pkg/as2w_mixed_8000_hold/blind_pilot.txt || { echo "no hold pilot"; exit 1; }
echo "$(date '+%m-%d %H:%M:%S') As2W hold: starting the As2W chain with the leg hold and the student PPO fine-tune (run fe_edge_v6hold)" >> logs/jobs/fe_as2w.txt
exec env AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000_hold \
  FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/2026-11-02_14-03-46_fe_edge_v5stand/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_v6hold \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
