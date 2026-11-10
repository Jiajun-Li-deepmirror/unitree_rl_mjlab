#!/bin/bash
# tf3_go2w.sh (go2-depth 2026-11-04 ~22:45): Go2W round 3 on t1. tf_v2 (ns_v2 + topflat30 + longexit) gave the first
# student that keeps its front wheels off the top landing (fe_edge_tf2 avg1499, tinynav: top 0.2 / climb, start 0,
# v5_ws 2.6), but it slid back in 8.2% of the step-ups (bar 5%; 18-36% when the apex stayed under the tread top) and
# climbed map3 at 0.764-0.775 (bar 0.80) under a teacher with margin 0.007 (topflat30 also paid the last swing of every
# climb and descent). Now: tf_v2's raw last (23800) with topoff15 in place of topflat30, liftm8 (riser lift paid up to
# 8 cm over the riser top), stairspeed3 (stairs deficit 1.5x more), multipliers as tf_v2 left them (side_drift 2,
# clock_side 2), checks +800 / +1200 / +1600, teacher margin 0.08 wanted; student PPO + BC 1.0 from fe_edge_tf2 avg1499.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_tf3 GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 FE_MARGIN=0.08 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,topoff,topoff15,longexit,liftm8,stairspeed3 FE_TEACHER_RUN=tf_v3 FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/go2w_depth_teacher/2026-11-04_19-30-38_tf_v2/model_23800.pt \
  FE_STUDENT_RUN=fe_edge_tf3 FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_tf2_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh Go2W
