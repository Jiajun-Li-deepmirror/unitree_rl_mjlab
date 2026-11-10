#!/bin/bash
# tf7_as2w.sh (go2-depth 2026-11-05 10:55, t1): As2W top-landing round 2. Every As2W policy lifted its front wheels on
# map3's top landing (tinynav 2.8-3.2 per climb: tf6d distilled, tf6p PPO+BC, deployed v5_ns; not the end wall: 4.2
# with it out of view), and the tf_v6 teacher does it as much as its students (pyramid stairs top landing 39 / 46%):
# that state (fronts on a landing, rears still climbing) is rare in training. Teacher tf_v7 = tf_v6 raw 16200 +
# landings (half the up-flights: a 0.8 m landing after every 2 steps), wumore (up-flights 0.30 of the tiles, flat
# 0.15), endwall; then PPO+BC student fe_edge_tf7 from tf6p avg1000 (start-landing fix, tinynav start 0.2) under it.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_ta7 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,topairup,longexit,endwall,landings,wumore FE_TEACHER_RUN=tf_v7 \
  FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/as2w_depth_teacher/2026-11-05_02-54-55_tf_v6/model_16200.pt \
  FE_STUDENT_RUN=fe_edge_tf7 FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_tf6p_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  bash jobs/fe_chain.sh As2W
