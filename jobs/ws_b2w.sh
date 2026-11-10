#!/bin/bash
# ws_b2w.sh: B2W wheelstep round (ws_common.sh): teacher fe_v1_avg9800 of the sidecmd+b2wdesc package, its multipliers
# as the run left them (round 2 from ws_v1 model_11100: medians over 10900..11150, back_drift 1.68, descent 1.90; fe_v1 9100..9800 had 1.08 / 1.58), speed margin 0.07 as its chain; student
# from the delivered fe_edge_v4desc_avg2000, Student-Edge, 2000 it, lr 2e-4 -> 1e-4.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ws_common.sh
exec env B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc FE_CONS_INIT=${WS_CONS_B2W:-back_drift:1.68,descent:1.90} FE_MARGIN=0.07 \
  FE_TEACHER_FROM=${WS_FROM_B2W:-$PWD/logs/rsl_rl/b2w_depth_teacher/fe_v1_avg9800/model_9800.pt} \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_v4desc_avg2000/model_2000.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 bash jobs/fe_chain.sh B2W
