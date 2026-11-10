#!/bin/bash
# ws_go2w.sh: Go2W wheelstep round (ws_common.sh): teacher fe_v1_avg13800 (multipliers ~1x over 13100..13800, so none
# set); student from the delivered fe_edge_v1_avg1500, Student-Edge, 2000 it, lr 2e-4 -> 1e-4.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ws_common.sh
exec env GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_TEACHER_FROM=${WS_FROM_GO2W:-$PWD/logs/rsl_rl/go2w_depth_teacher/fe_v1_avg13800/model_13800.pt} \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_v1_avg1500/model_1500.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 bash jobs/fe_chain.sh Go2W
