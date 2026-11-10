#!/bin/bash
# ns_go2w_d.sh (go2-depth 2026-11-04 ~10:10): Go2W noslide student by distillation. Both PPO+BC students from v5_ws
# (ns1, ns2) kept v5_ws's slides (tinynav 12%) and failed drift; As2W's PPO student took nothing of its new teacher
# while its distilled one (fe_edge_nsd) slid back in 6% (v4_fe 14%). Teacher = the ns pick (ns_v2 avg22600: slide back
# 2.8%, margin 0.051). Student-Edge (distillation) from v5_ws (fe_edge_ws1 avg1999), 2000 it, lr 2e-4 -> 1e-4.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75 FE_TEACHER_RUN=ns_v2 FE_STUDENT_RUN=fe_edge_nsd \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_ws1_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 \
  bash jobs/fe_chain.sh Go2W
