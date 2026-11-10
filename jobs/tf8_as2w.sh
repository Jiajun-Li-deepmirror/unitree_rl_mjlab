#!/bin/bash
# tf8_as2w.sh (go2-depth 2026-11-05 11:35, t1; tf7_as2w.sh without endwall): As2W top-landing round 2. Every As2W policy lifted its front wheels on
# map3's top landing (tinynav 2.8-3.2 per climb: tf6d distilled, tf6p PPO+BC, deployed v5_ns; not the end wall: 4.2
# with it out of view), and the tf_v6 teacher does it as much as its students (pyramid stairs top landing 39 / 46%):
# that state (fronts on a landing, rears still climbing) is rare in training. Teacher tf_v7 = tf_v6 raw 16200 +
# landings (half the up-flights: a 0.8 m landing after every 2 steps), wumore (up-flights 0.30 of the tiles, flat
# 0.15), endwall; then PPO+BC student fe_edge_tf7 from tf6p avg1000 (start-landing fix, tinynav start 0.2) under it.
# tf_v7 (with endwall) fell over ~0.5 per log interval from +150 to +500 (tf_v6 0.04-0.2) and its mean reward slid
# 66 -> 55-62: top_flat_eval of tf_v6 on landings + wumore 0.52 falls per 1000 robot-s, + endwall 2.47 (original
# terrain 0.20): robots driving into the end wall. The As2W top-landing steps are not the wall's (tinynav 4.2 / climb
# with it out of view), so no endwall here.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_ta8 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,topairup,longexit,landings,wumore FE_TEACHER_RUN=tf_v8 \
  FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/as2w_depth_teacher/2026-11-05_02-54-55_tf_v6/model_16200.pt \
  FE_STUDENT_RUN=fe_edge_tf8 FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_tf6p_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  bash jobs/fe_chain.sh As2W
