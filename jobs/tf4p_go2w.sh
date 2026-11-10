#!/bin/bash
# tf4p_go2w.sh (go2-depth 2026-11-05 06:40, t1): Go2W two-stage student. The distilled fe_edge_tf4d carried tf_v4's
# top landing (gate top 0.1) but climbs slow (avg1000 / 1500 / 1999 up 0.710 / 0.734 / 0.747, bar 0.80); the PPO+BC
# fe_edge_tf4 climbed 0.77-0.78. This PPO+BC (tf4's recipe) starts from tf4d avg1999 under tf_v4 avg24600 so RL can win
# the speed back, and adds noslide2x to the reward (slide gate 7.4% for both earlier students). Student-only.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_tg4p && cp logs/picks_tg4d/go2w_teacher logs/picks_tg4p/go2w_teacher
exec env FE_PICKS=logs/picks_tg4p GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,sidetgt75,topair,longexit,liftm8,stairspeed3 FE_TEACHER_RUN=tf_v4 \
  FE_STUDENT_RUN=fe_edge_tf4p FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_tf4d_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  bash jobs/fe_chain.sh Go2W
