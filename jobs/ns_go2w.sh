#!/bin/bash
# ns_go2w.sh: Go2W noslide round (ns_common.sh): teacher from ws_v3 model_19100 (the delivered depth_v5_ws's teacher
# run); student PPO + BC (decaying to 0.5) from the delivered fe_edge_ws1 avg1999, 1500 it.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/go2w_depth_teacher/2026-11-03_20-30-52_ws_v3/model_19100.pt \
  FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_ws1_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=0.5 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  bash jobs/fe_chain.sh Go2W
