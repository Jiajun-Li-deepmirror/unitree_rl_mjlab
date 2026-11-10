#!/bin/bash
# ws_as2w.sh: As2W wheelstep round (ws_common.sh): teacher fe_v1_avg11000 (no constraints in its pilot); student the
# delivered round's recipe (as2w_bc.sh, v7bc: Student-Edge-RL, BC held at 1.0, hard commands, DR edges, friction,
# long stands, checks after training) from the delivered fe_edge_v7bc_avg1000.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ws_common.sh
exec env AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_TEACHER_FROM=${WS_FROM_AS2W:-$PWD/logs/rsl_rl/as2w_depth_teacher/fe_v1_avg11000/model_11000.pt} \
  FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_v7bc_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS=--agent.algorithm.bc-coef-final=1.0 \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
