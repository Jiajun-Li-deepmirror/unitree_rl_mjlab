#!/bin/bash
# ns_as2w_d.sh (go2-depth 2026-11-04 ~09:40): As2W noslide student by distillation. The PPO+BC student from the
# delivered v7bc (fe_edge_ns1) took nothing of the new teacher in 1500 it (tinynav: landing steps 3.0 per climb as
# v4_fe, slid back 17 vs 14%), as B2W's PPO from its delivered student did; B2W's fix came through distillation first.
# Teacher = the ns pick (ns_v1 avg15000: slide back 1.3%, clean 86%, margin 0.057). Student-Edge (distillation) from
# v7bc avg1000 with the As2W student options (hard commands, DR edges 0.4, friction 0.3-1.6, long stands), 2000 it,
# lr 2e-4 -> 1e-4, checks after training. Picks logs/picks_ns.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= FE_STUDENT_RUN=fe_edge_nsd \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_v7bc_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
