#!/bin/bash
# tf_go2w.sh (go2-depth 2026-11-04 ~17:55): Go2W topflat round at -30 (topflat30; see tf_as2w.sh, tf2_as2w.sh), on t1. Teacher from ns_v2's last (22600; its
# check passed sideways drift at p90 0.10-0.12 m forward over ~3 m) + wheelahead + topflat, side_drift starting at its
# cap 3x: the two wheelahead teachers from it (nsa_v1 at 1x, nsa_v2 at 2x) crept forward 0.17-0.20 / 0.15-0.17 m
# walking sideways at 0.3 m/s (bar 5% of ~3.1 m) while training held side_drift's drift share at 0.07-0.08, under
# its 0.095 guard, so the multiplier never rose. Checks +800 / +1200 (an average needs 8 saves of the run). Student
# PPO + BC 1.0 (the As2W nsrl recipe) from the distilled nsd avg1000, waiting for 14 GB free.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
LAST=$(ls logs/rsl_rl/go2w_depth_teacher/2026-11-04_03-07-36_ns_v2/model_*.pt | sort -V | tail -n 1)
exec env FE_PICKS=logs/picks_tf GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:3.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,wheelahead,topflat,topflat30 FE_TEACHER_RUN=tf_v1 FE_TEACHER_KS="800 1200" \
  FE_TEACHER_FROM=$PWD/$LAST \
  FE_STUDENT_RUN=fe_edge_tf1 FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_nsd_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh Go2W
