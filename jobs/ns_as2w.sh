#!/bin/bash
# ns_as2w.sh: As2W noslide round (ns_common.sh): teacher from ws_v3 model_13000 (wheelstep round: no pass, 4% falls on
# 12-15 cm up); student PPO + BC 1.0 from the delivered fe_edge_v7bc avg1000 with its round's options, 1500 it,
# checks after training.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/as2w_depth_teacher/2026-11-03_17-58-06_ws_v3/model_13000.pt \
  FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_v7bc_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
