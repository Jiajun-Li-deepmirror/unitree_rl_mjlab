#!/bin/bash
# tf5_as2w.sh (go2-depth 2026-11-05 ~00:00): As2W topair round (tf4_as2w.sh with topair in place of topoff15: topoff wrecked tf_v4 within ~60 iterations, contact flickers 3-5 take-offs per zone wheel-second; topair charges the airborne time of wheels that took off in the zone). tf4 header:
# tf_v3 (topflat30 + longexit from nsa_v1) cut the
# teacher's top-flat lifts 131 -> 81 per 100 wheel-s at +400, but failed its checks at +800 / +1200: map3 descents
# 0.65-0.67 of the command (bar 0.80; topflat paid the last swing of every descent over the bottom landing), stops,
# clock (0, 0.6) 79%. Now topoff15 (mdp.wheel_takeoff_flat_ahead: -15 per take-off from a landing with flat ahead,
# next to stairs) + longexit, from ns_v1's raw 15000 (the teacher of the deployed as2w_depth_v5_ns; no wheelahead:
# every As2W fine-tune from nsa_v1 failed the clock), checks +800 / +1200 / +1600, pick last passing. Student PPO +
# BC 1.0 from nsrl avg1499 (as2w_depth_v5_ns) as tf3_as2w.sh.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_ta5 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,topair,longexit FE_TEACHER_RUN=tf_v5 FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/as2w_depth_teacher/2026-11-04_05-39-29_ns_v1/model_15000.pt \
  FE_STUDENT_RUN=fe_edge_tf5 FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_nsrl_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
